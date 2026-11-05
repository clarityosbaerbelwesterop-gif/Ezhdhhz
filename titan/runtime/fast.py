"""Fast executor: runs compiled programs at TITAN IR (kernel) granularity.

What is the same as the ISA-level emulator
  * the graph is compiled by the TITAN compiler: same autodiff, fusion, kernel formation,
    in-place updates and buffer sharing;
  * every kernel uses TITAN numerics: FP32 accumulation for float matrix products, exact INT32
    wrap-around for integer products, per-step rounding of fused element-wise chains to the
    step dtype (BF16/FP16 round-to-nearest-even), CONVERT saturation rules.
What differs
  * whole tensors per kernel instead of tiles and instructions, so the summation order of
    matrix products and reductions differs (results agree within FP32 tolerance; tests);
  * no ISA program, no cycle timing, no emulated memory limit (host RAM is the limit).
Label: "functional at TITAN IR level (kernel granularity, no timing)".

Programs run as compiled execution plans (one generated function per program, runtime/plan.py),
bit-identical to the kernel interpreter below, which remains in use for storage offload and
regenerated weights.
"""
from __future__ import annotations

import time

import numpy as np

from ..compiler.lower import Executable
from ..config import load_config
from ..dtypes import DType, convert, from_compute, round_to, wrap_int32
from ..emu.engine import RunResult
from ..emu.semantics import ACTIVATIONS, normalize
from ..isa.spec import Norm, ReduceOp
from ..compiler.graph import UNARY
from .device import RunOutput

MODE = "fast"
TOKEN_HEAD_KINDS = {"onehot", "embedding", "embedding_grad", "split_heads", "merge_heads", "repeat_heads",
                    "sum_head_groups"}


def token_head_kernel(K, xs):
    """Gather / scatter-add / one-hot / head layouts on whole tensors (NumPy)."""
    from ..compiler.reference import head_layout
    if K.kind == "onehot":
        ids = np.asarray(xs[0]).reshape(-1)
        return (ids[:, None] == np.arange(K.attrs["n"])[None, :]).astype(np.float32)
    if K.kind == "embedding":
        return np.asarray(xs[1])[np.asarray(xs[0]).reshape(-1).astype(np.int64)]
    if K.kind == "embedding_grad":
        ids = np.asarray(xs[0]).reshape(-1).astype(np.int64)
        dy = np.asarray(xs[1], dtype=np.float32)
        out = np.zeros(K.output.shape, dtype=np.float32)
        order = np.argsort(ids, kind="stable")
        uniq, starts = np.unique(ids[order], return_index=True)
        out[uniq] = np.add.reduceat(dy[order], starts, axis=0)
        return out
    return head_layout(K.kind, np.asarray(xs[0]), K.attrs)


def _store(x: np.ndarray, dt: DType) -> np.ndarray:
    """Logical value representation used by the fast executor (float32 for float types)."""
    if dt.is_int:
        return from_compute(np.asarray(x, dtype=np.int64), dt).astype(np.int32 if dt == DType.INT32 else np.int8)
    return round_to(dt, np.asarray(x, dtype=np.float32)) if dt != DType.FP32 else np.asarray(x, dtype=np.float32)


def _c(x, dt: DType):
    return x.astype(np.int64) if dt.is_int else x.astype(np.float32, copy=False)


class FastDevice:
    """Kernel-level executor. With ``offload_dir`` every persistent tensor (weights, optimizer state) and
    every temporary of at least ``offload_mb`` MiB lives in a memory-mapped file on that storage, so the
    model size is bounded by the storage, not by RAM (RAM holds the working set of one kernel plus the
    operating system's page cache)."""

    def __init__(self, cfg="dev", offload_dir=None, offload_mb: float = 64, compiled: bool = True):
        self.cfg = load_config(cfg)
        self.mem: dict[tuple, np.ndarray] = {}
        self._loaded: set[int] = set()
        self.offload = None
        if offload_dir:
            import os
            from pathlib import Path
            self.offload = Path(offload_dir)
            self.offload.mkdir(parents=True, exist_ok=True)
            self._tag = f"titan-{os.getpid()}-{id(self):x}"
        self.threshold = int(offload_mb * 2 ** 20)
        self._temp_files: dict[int, object] = {}
        self.offloaded_bytes = 0
        self.compiled = compiled           # run whole programs as one generated function (runtime/plan.py)
        self._plans: dict[int, tuple] = {}

    def _mmap(self, name: str, like: np.ndarray) -> np.ndarray:
        path = self.offload / f"{self._tag}-{name}.bin"
        mm = np.memmap(path, dtype=like.dtype, mode="w+", shape=like.shape)
        mm[...] = like
        return mm

    def _persist(self, key, v: np.ndarray) -> None:
        if self.offload is not None and v.nbytes >= self.threshold:
            cur = self.mem.get(key)
            if isinstance(cur, np.memmap) and cur.shape == v.shape and cur.dtype == v.dtype:
                cur[...] = v                     # in-place update of the file-backed tensor
                return
            self.mem[key] = self._mmap("p%d_%s" % (key[0], "x".join(map(str, key[1]))), v)
            self.offloaded_bytes += v.nbytes
        else:
            self.mem[key] = v

    def view(self, exe: Executable, name: str) -> np.ndarray:
        """Stored tensor without a copy (memory-mapped when offloaded); used for streaming checkpoints."""
        return self.mem[self._key(exe.buffers[name])]

    @staticmethod
    def _key(buf):
        return (buf.addr, buf.shape, int(buf.dtype))

    def load(self, exe: Executable, init: bool = True, regenerate=()) -> None:
        """``regenerate``: frozen lazily-initialized parameters that are recomputed from their seed on every
        use instead of being stored (memory independent of their size)."""
        regen = getattr(self, "regen", None)
        if regen is None:
            regen = self.regen = {}
        for name in regenerate:
            regen[self._key(exe.buffers[name])] = exe.lazy_init[name]
        if init:
            for name, v in exe.init.items():
                self.write(exe, name, v)
            for name, fn in getattr(exe, "lazy_init", {}).items():
                if name in regenerate:
                    continue
                self.write(exe, name, fn())      # one parameter at a time, straight into its storage
        self._loaded.add(id(exe))

    def write(self, exe: Executable, name: str, value) -> None:
        b = exe.buffers[name]
        v = np.asarray(value)
        if v.ndim < 2:
            v = v.reshape(1, -1) if v.ndim == 1 else v.reshape(1, 1)
        if v.shape != b.shape:
            raise ValueError(f"{name}: shape {v.shape} != {b.shape}")
        self._persist(self._key(b), _store(v, b.dtype))

    def read(self, exe: Executable, name: str) -> np.ndarray:
        k = self._key(exe.buffers[name])
        fn = getattr(self, "regen", {}).get(k)
        return fn() if fn is not None else self.mem[k].copy()

    def params(self, exe: Executable) -> dict:
        return {p: self.read(exe, p) for p in exe.params}

    # ------------------------------------------------------------------ kernels
    def _get(self, env, buf):
        if buf.persistent:
            k = self._key(buf)
            fn = getattr(self, "regen", {}).get(k)
            return fn() if fn is not None else self.mem[k]
        return env[id(buf)]

    def _put(self, env, buf, value):
        v = _store(np.broadcast_to(value, buf.shape), buf.dtype)
        if buf.persistent:
            self._persist(self._key(buf), v)
        elif self.offload is not None and v.nbytes >= self.threshold:
            env[id(buf)] = self._mmap(f"t{id(buf):x}", v)     # spill large temporaries to storage
            self._temp_files[id(buf)] = env[id(buf)].filename
        else:
            env[id(buf)] = v

    def _drop(self, env, bid) -> None:
        env.pop(bid, None)
        f = self._temp_files.pop(bid, None)
        if f:
            import os
            try:
                os.remove(f)
            except OSError:
                pass

    @staticmethod
    def _steps(steps, arg):
        vals = []
        for st in steps:
            xs = [arg(a, vals) for a in st.args]
            dt = st.dtype
            op = st.op
            with np.errstate(all="ignore"):
                if op in ("add", "sub", "mul", "div", "maximum"):
                    a, b = (_c(x, dt) for x in xs)
                    r = {"add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.divide,
                         "maximum": np.maximum}[op](a, b)
                elif op == "scale":
                    r = np.float32(st.attrs["alpha"]) * _c(xs[0], dt) + np.float32(st.attrs["beta"])
                elif op == "broadcast":
                    r = xs[0]
                elif op in UNARY:
                    r = ACTIVATIONS[UNARY[op]](xs[0].astype(np.float32)).astype(np.float32)
                elif op == "cast":
                    vals.append(_store(convert(xs[0], st.attrs["src_dtype"], dt, st.attrs["scale"]), dt))
                    continue
                else:
                    raise NotImplementedError(op)
            vals.append(_store(r, dt))
        return vals[-1]

    def _kernel(self, K, env):
        if K.kind == "gemm":
            a, b = self._get(env, K.inputs[0]), self._get(env, K.inputs[1])
            bt, ta, tb = K.attrs["batch"], K.attrs["ta"], K.attrs["tb"]
            dt = K.inputs[0].dtype
            A = a.reshape(bt, a.shape[0] // bt, a.shape[1])
            Bm = b.reshape(bt, b.shape[0] // bt, b.shape[1])
            A = A.swapaxes(1, 2) if ta else A
            Bm = Bm.swapaxes(1, 2) if tb else Bm
            if dt.is_int:
                acc = wrap_int32(np.matmul(A.astype(np.int64), Bm.astype(np.int64)))
            else:
                acc = np.matmul(A.astype(np.float32, copy=False), Bm.astype(np.float32, copy=False))
            acc = acc.reshape(-1, acc.shape[-1])
            if not K.steps:
                return acc

            def arg(x, vals):
                if x[0] == "acc":
                    return acc
                return self._get(env, K.inputs[x[1]]) if x[0] == "in" else vals[x[1]]
            return self._steps(K.steps, arg)
        if K.kind == "ewise":
            def arg(x, vals):
                return self._get(env, K.inputs[x[1]]) if x[0] == "in" else vals[x[1]]
            return self._steps(K.steps, arg)
        if K.kind in TOKEN_HEAD_KINDS:
            return token_head_kernel(K, [self._get(env, b) for b in K.inputs])
        x = self._get(env, K.inputs[0])
        if K.kind == "rownorm":
            fn = {"softmax": Norm.SOFTMAX, "layernorm": Norm.LAYERNORM, "rmsnorm": Norm.RMSNORM}[K.attrs["fn"]]
            with np.errstate(all="ignore"):
                return normalize(fn, x.astype(np.float32), K.attrs["eps"] or 1e-5)
        if K.kind == "reduce":
            ax = {0: 0, 1: 1, 2: None}[K.attrs["axis"]]
            xc = _c(x, K.inputs[0].dtype)
            rop = K.attrs["rop"]
            if rop == ReduceOp.SUM:
                r = xc.sum(axis=ax, keepdims=True, dtype=xc.dtype)
            elif rop == ReduceOp.MAX:
                r = xc.max(axis=ax, keepdims=True)
            else:
                r = xc.mean(axis=ax, keepdims=True, dtype=np.float32)
            return np.asarray(r).reshape(K.output.shape)
        if K.kind == "transpose":
            return x.T
        if K.kind == "copy":
            return x
        raise NotImplementedError(K.kind)

    def _frees(self, exe: Executable):
        cache = getattr(exe, "_fast_frees", None)
        if cache is None:
            last = {}
            for i, K in enumerate(exe.module.kernels):
                for b in K.inputs:
                    if not b.persistent:
                        last[id(b)] = i
            cache = [[] for _ in exe.module.kernels]
            for bid, i in last.items():
                cache[i].append(bid)
            exe._fast_frees = cache
        return cache

    def run(self, exe: Executable, feeds: dict | None = None, mode: str = MODE, trace: bool = False,
            fetch: list[str] | None = None, matmul_hook=None) -> RunOutput:
        if trace or matmul_hook is not None:
            raise ValueError("the fast executor has no instruction trace and no RTL offload")
        if id(exe) not in self._loaded:
            self.load(exe)
        t0 = time.perf_counter()
        for k, v in (feeds or {}).items():
            self.write(exe, k, v)
        missing = [n for n in exe.inputs if self._key(exe.buffers[n]) not in self.mem]
        if missing:
            raise KeyError(f"missing inputs: {missing}")
        t1 = time.perf_counter()
        if self.offload is None and not getattr(self, "regen", None) and self.compiled:
            plan = self._plans.get(id(exe))
            if plan is None:
                from .plan import numpy_plan
                plan = self._plans[id(exe)] = numpy_plan(self, exe)
            with np.errstate(all="ignore"):
                plan[0](self.mem)
            return self._finish(exe, fetch, t0, t1, plan[1])
        env: dict[int, np.ndarray] = {}
        ops = 0
        frees = self._frees(exe)
        for i, K in enumerate(exe.module.kernels):
            self._put(env, K.output, self._kernel(K, env))
            for bid in frees[i]:                      # free temporaries after their last use
                self._drop(env, bid)
            if K.kind == "gemm":
                a = K.inputs[0]
                kd = a.shape[0] // K.attrs["batch"] if K.attrs["ta"] else a.shape[1]
                ops += 2 * K.shape[0] * K.shape[1] * kd
        for bid in list(self._temp_files):
            self._drop(env, bid)
        return self._finish(exe, fetch, t0, t1, ops)

    def _finish(self, exe, fetch, t0, t1, ops) -> RunOutput:
        t2 = time.perf_counter()
        outs = {k: self.read(exe, k) for k in (exe.outputs if fetch is None else fetch)}
        t3 = time.perf_counter()
        res = RunResult(MODE, 0, {"kernels": len(exe.module.kernels)}, ops, ops // 2, 0, clock_ghz=self.cfg.clock_ghz,
                        host_wall_s=t2 - t1)
        return RunOutput(outs, res, (t1 - t0) + (t3 - t2), {"host_write_s": t1 - t0, "execute_s": t2 - t1,
                                                           "host_read_s": t3 - t2})

