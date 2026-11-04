"""Mode D — RTL co-simulation of the TITAN matrix engine (rtl/titan_mxu_tile.sv) with Verilator.

The emulator's MATMUL semantics (exact integer products, two's-complement ACC_W
accumulation) are checked bit-for-bit against the synthesizable RTL, and integer
MATMUL instructions of a program can be executed on the RTL model instead of the
NumPy golden model ("hybrid RTL backend").
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ..dtypes import DType, wrap_int32
from ..isa.spec import Flag, Instruction, Opcode

ROOT = Path(__file__).resolve().parent.parent.parent
RTL = ROOT / "rtl" / "titan_mxu_tile.sv"
TB = ROOT / "tb" / "tb_titan_mxu_tile.sv"
MAX_WORDS = (1 << 22) - 16


class RTLUnavailable(RuntimeError):
    pass


def verilator_version() -> str | None:
    exe = shutil.which("verilator")
    if not exe:
        return None
    return subprocess.run([exe, "--version"], capture_output=True, text=True).stdout.strip()


class MXUTileSim:
    """Builds and drives the Verilator model of one MXU tile."""

    def __init__(self, rows: int = 8, cols: int = 8, data_w: int = 8, acc_w: int = 32,
                 build_root: Path | None = None):
        if not shutil.which("verilator"):
            raise RTLUnavailable("Verilator is not installed")
        self.rows, self.cols, self.data_w, self.acc_w = rows, cols, data_w, acc_w
        self.build_root = Path(build_root or ROOT / "build" / "cosim")
        self.jobs_run = 0
        self.rtl_cycles = 0
        self.binary = self._build()

    def _build(self) -> Path:
        src = RTL.read_bytes() + TB.read_bytes()
        key = hashlib.sha256(src + f"{self.rows},{self.cols},{self.data_w},{self.acc_w}".encode()).hexdigest()[:12]
        d = self.build_root / f"mxu_{self.rows}x{self.cols}_d{self.data_w}_{key}"
        exe = d / "Vtb_titan_mxu_tile"
        if exe.exists():
            return exe
        d.mkdir(parents=True, exist_ok=True)
        cmd = ["verilator", "--binary", "--timing", "-Wall", f"-GROWS={self.rows}", f"-GCOLS={self.cols}",
               f"-GDATA_W={self.data_w}", f"-GACC_W={self.acc_w}", "--top-module", "tb_titan_mxu_tile",
               "--Mdir", str(d), str(RTL), str(TB)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not exe.exists():
            raise RTLUnavailable(f"Verilator build failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
        return exe

    @staticmethod
    def _w(v) -> str:
        return f"{int(v) & 0xFFFFFFFF:08x}"

    def run_jobs(self, jobs) -> list[np.ndarray]:
        """jobs: list of (a [rows x K], b [K x cols], init [rows x cols] or None) -> list of int32 tiles."""
        out: list[np.ndarray] = []
        batch, words = [], 1
        for job in jobs:
            need = 2 + self.rows * self.cols + job[0].shape[1] * (self.rows + self.cols)
            if batch and words + need > MAX_WORDS:
                out += self._run(batch)
                batch, words = [], 1
            batch.append(job)
            words += need
        if batch:
            out += self._run(batch)
        return out

    def _run(self, jobs) -> list[np.ndarray]:
        lo, hi = -(1 << (self.data_w - 1)), (1 << (self.data_w - 1)) - 1
        lines = [self._w(len(jobs))]
        for a, b, init in jobs:
            if a.shape[0] != self.rows or b.shape[1] != self.cols or a.shape[1] != b.shape[0]:
                raise ValueError("tile shape mismatch")
            if a.size and (a.min() < lo or a.max() > hi or b.min() < lo or b.max() > hi):
                raise ValueError(f"operand outside INT{self.data_w} range")
            k = a.shape[1]
            lines += [self._w(k), self._w(init is not None)]
            if init is not None:
                lines += [self._w(v) for v in np.asarray(init).ravel()]
            for kk in range(k):
                lines += [self._w(v) for v in a[:, kk]] + [self._w(v) for v in b[kk, :]]
            self.rtl_cycles += k + 1
        with tempfile.TemporaryDirectory() as td:
            vec, res = Path(td) / "vec.hex", Path(td) / "out.hex"
            vec.write_text("\n".join(lines) + "\n")
            r = subprocess.run([str(self.binary), f"+vec={vec}", f"+out={res}"], capture_output=True, text=True)
            if r.returncode != 0 or "PASS" not in r.stdout:
                raise RuntimeError(f"RTL simulation failed: {r.stdout[-1000:]} {r.stderr[-1000:]}")
            vals = np.array([int(x, 16) for x in res.read_text().split()], dtype=np.uint32).view(np.int32)
        self.jobs_run += len(jobs)
        n = self.rows * self.cols
        return [vals[i * n:(i + 1) * n].reshape(self.rows, self.cols) for i in range(len(jobs))]

    def matmul(self, A: np.ndarray, B: np.ndarray, C0: np.ndarray | None = None) -> np.ndarray:
        """C = A @ B (+ C0) computed by the RTL tile, decomposed into ROWS x COLS output tiles."""
        A, B = np.asarray(A, np.int64), np.asarray(B, np.int64)
        m, k = A.shape
        n = B.shape[1]
        R, C = self.rows, self.cols
        jobs, pos = [], []
        for i0 in range(0, m, R):
            for j0 in range(0, n, C):
                a = np.zeros((R, k), np.int64)
                b = np.zeros((k, C), np.int64)
                a[:min(R, m - i0)] = A[i0:i0 + R]
                b[:, :min(C, n - j0)] = B[:, j0:j0 + C]
                init = None
                if C0 is not None:
                    init = np.zeros((R, C), np.int64)
                    init[:min(R, m - i0), :min(C, n - j0)] = C0[i0:i0 + R, j0:j0 + C]
                jobs.append((a, b, init))
                pos.append((i0, j0))
        out = np.zeros((m, n), np.int32)
        for (i0, j0), t in zip(pos, self.run_jobs(jobs)):
            out[i0:i0 + R, j0:j0 + C] = t[:min(R, m - i0), :min(C, n - j0)]
        return out


def make_matmul_hook(sim: MXUTileSim, stats: dict | None = None):
    """Engine hook: execute INT8/INT4 MATMUL instructions on the RTL model. Others fall back (return False)."""
    stats = stats if stats is not None else {}

    def hook(core: int, ins: Instruction, read, write) -> bool:
        if ins.opcode != Opcode.MATMUL:
            return False
        c, a, b = ins.operands
        if a.dtype not in (DType.INT8, DType.INT4) or a.dtype.bits > sim.data_w:
            stats["fallback"] = stats.get("fallback", 0) + 1
            return False
        A, B = read(a).astype(np.int64), read(b).astype(np.int64)
        if ins.has(Flag.TRANS_A):
            A = A.T
        if ins.has(Flag.TRANS_B):
            B = B.T
        C0 = read(c).astype(np.int64) if ins.has(Flag.ACC) else None
        write(c, sim.matmul(A, B, C0))
        stats["rtl_matmuls"] = stats.get("rtl_matmuls", 0) + 1
        return True

    hook.stats = stats
    return hook


def correspondence_test(n_cases: int = 20, seed: int = 0, rows: int = 8, cols: int = 8, data_w: int = 8) -> dict:
    """Random MATMUL cases: RTL vs emulator golden model, bit-exact. Includes accumulate and overflow cases."""
    from ..emu.semantics import matmul_values
    sim = MXUTileSim(rows, cols, data_w)
    rng = np.random.default_rng(seed)
    lim = 1 << (data_w - 1)
    dt = DType.INT8 if data_w == 8 else DType.INT4
    mismatches = 0
    cases = []
    for i in range(n_cases):
        m, n, k = (int(x) for x in rng.integers(1, 3 * rows, 3))
        A = rng.integers(-lim, lim, (m, k))
        B = rng.integers(-lim, lim, (k, n))
        C0 = None
        if i % 3 == 1:
            C0 = rng.integers(-2**31, 2**31, (m, n))
        elif i % 3 == 2:
            C0 = np.full((m, n), 2**31 - 5)          # forces wrap-around
        rtl = sim.matmul(A, B, C0)
        ref = matmul_values(A.astype(np.int8), B.astype(np.int8), dt, False, False,
                            None if C0 is None else wrap_int32(C0))
        ok = bool(np.array_equal(rtl, ref))
        mismatches += not ok
        cases.append({"shape": (m, n, k), "accumulate": C0 is not None, "bit_exact": ok})
    return {"cases": n_cases, "mismatches": mismatches, "tile": f"{rows}x{cols}", "data_w": data_w,
            "rtl_jobs": sim.jobs_run, "rtl_cycles": sim.rtl_cycles, "simulator": verilator_version(),
            "details": cases, "method": "RTL co-simulation (Verilator) vs TITAN functional golden model"}
