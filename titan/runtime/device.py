"""Device runtime: loads executables, moves tensors between host and device, runs programs."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..compiler.lower import Executable
from ..config import load_config
from ..dtypes import DType
from ..emu.engine import CYCLE, FUNCTIONAL, Engine, Machine, RunResult
from ..emu.replay import ReplayProgram


@dataclass
class RunOutput:
    outputs: dict
    result: RunResult
    host_io_s: float = 0.0
    timings: dict = field(default_factory=dict)


class Device:
    """One virtual TITAN-X1 device. Host transfers are host-side work and are timed separately."""

    def __init__(self, cfg="dev"):
        self.cfg = load_config(cfg)
        self.machine = Machine(self.cfg)
        self._loaded: set[int] = set()
        self._replay: dict[int, ReplayProgram] = {}
        self._cycle: dict[int, RunResult] = {}
        self.fast_paths = True   # replay for functional runs; cached timing for cycle runs

    def load(self, exe: Executable, init: bool = True) -> None:
        """Write initial param/const values of ``exe`` into global memory."""
        if exe.cfg.config_hash() != self.cfg.config_hash():
            raise ValueError("executable was compiled for a different configuration")
        if init:
            for name, v in exe.init.items():
                self.write(exe, name, v)
        self._loaded.add(id(exe))

    def write(self, exe: Executable, name: str, value) -> None:
        ref = exe.ref(name)
        v = np.asarray(value)
        if v.ndim == 1:
            v = v[None, :]
        if v.ndim == 0:
            v = v.reshape(1, 1)
        if v.shape != ref.shape:
            raise ValueError(f"{name}: shape {v.shape} != {ref.shape}")
        if ref.dtype.is_float:
            v = v.astype(np.float32)
        else:
            v = v.astype(np.int64)
        self.machine.host_write(ref, v)

    def read(self, exe: Executable, name: str) -> np.ndarray:
        v = self.machine.host_read(exe.ref(name))
        return v.astype(np.float32) if exe.ref(name).dtype in (DType.FP16, DType.BF16) else v

    def run(self, exe: Executable, feeds: dict | None = None, mode: str = FUNCTIONAL, trace: bool = False,
            fetch: list[str] | None = None, matmul_hook=None) -> RunOutput:
        if id(exe) not in self._loaded:
            self.load(exe)
        t0 = time.perf_counter()
        feeds = feeds or {}
        missing = set(exe.inputs) - set(feeds)
        if missing:
            raise KeyError(f"missing inputs: {sorted(missing)}")
        for k, v in feeds.items():
            self.write(exe, k, v)
        t1 = time.perf_counter()
        res = self._execute(exe, mode, trace, matmul_hook)
        t2 = time.perf_counter()
        outs = {k: self.read(exe, k) for k in (exe.outputs if fetch is None else fetch)}
        t3 = time.perf_counter()
        return RunOutput(outs, res, (t1 - t0) + (t3 - t2), {"host_write_s": t1 - t0, "execute_s": t2 - t1,
                                                            "host_read_s": t3 - t2})

    def _execute(self, exe: Executable, mode: str, trace: bool, matmul_hook) -> RunResult:
        """Functional runs use the bit-exact replay executor. The cycle model's timing depends only on
        the instruction stream and addresses (never on data), so it is simulated once per executable
        and reused, while the data path runs through replay."""
        if not self.fast_paths or trace or matmul_hook is not None:
            return Engine(self.machine, mode, trace=trace, matmul_hook=matmul_hook).run(exe.program,
                                                                                         validate_program=False)
        key = id(exe)
        if mode == CYCLE and key not in self._cycle:
            res = Engine(self.machine, CYCLE).run(exe.program, validate_program=False)
            self._cycle[key] = res
            return res
        rp = self._replay.get(key)
        if rp is None:
            rp = self._replay[key] = ReplayProgram(self.machine, exe.program)
        res = rp.run()
        if mode == CYCLE:
            c = self._cycle[key]
            res.mode, res.cycles, res.counters = CYCLE, c.cycles, c.counters
            res.completions = list(c.completions)
        return res

    def params(self, exe: Executable) -> dict[str, np.ndarray]:
        return {p: self.read(exe, p) for p in exe.params}


MODES = (FUNCTIONAL, CYCLE)
