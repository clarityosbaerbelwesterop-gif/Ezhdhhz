"""Accelerated executor: TITAN IR kernels on the strongest host device via PyTorch (CUDA GPU or CPU).

Use it on your own environment (workstation, server, cloud GPU) to train as fast and as large
as that hardware allows. It executes the same compiled TITAN kernels as the fast executor
(same autodiff, fusion, in-place updates, buffer sharing) with TITAN numerics:
FP32 accumulation, per-step rounding of fused element-wise chains to the step dtype
(BF16/FP16 round-to-nearest-even), CONVERT rules (round-half-even, saturation, NaN -> 0).
Integer kernels (INT8/INT4 inference) fall back to the NumPy fast executor for exactness.

Label: "functional at TITAN IR level on a host accelerator (PyTorch); not TITAN hardware".
"""
from __future__ import annotations

import math
import time

import numpy as np

from ..compiler.graph import UNARY
from ..compiler.lower import Executable
from ..config import load_config
from ..dtypes import DType
from ..emu.engine import RunResult
from ..isa.spec import Act, ReduceOp
from .device import RunOutput
from .fast import FastDevice

MODE = "accel"


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:  # pragma: no cover - import errors of broken installs
        return False


def default_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class AccelDevice:
    def __init__(self, cfg="dev", device: str | None = None):
        import torch
        self.torch = torch
        self.cfg = load_config(cfg)
        self.device = torch.device(device or default_device())
        self.mem: dict[tuple, object] = {}
        self._loaded: set[int] = set()
        self._np = FastDevice(cfg)          # exact integer kernels
        t = torch
        c = math.sqrt(2.0 / math.pi)

        def gelu(x):
            return 0.5 * x * (1.0 + t.tanh(c * (x + 0.044715 * x * x * x)))

        def gelu_grad(x):
            u = t.tanh(c * (x + 0.044715 * x * x * x))
            return 0.5 * (1.0 + u) + 0.5 * x * (1.0 - u * u) * c * (1.0 + 3 * 0.044715 * x * x)

        self.act = {Act.RELU: t.relu, Act.GELU: gelu, Act.SIGMOID: t.sigmoid, Act.TANH: t.tanh, Act.EXP: t.exp,
                    Act.LOG: t.log, Act.SQRT: t.sqrt, Act.RSQRT: t.rsqrt, Act.RECIP: t.reciprocal,
                    Act.STEP: lambda x: (x > 0).to(t.float32), Act.GELU_GRAD: gelu_grad,
                    Act.SQUARE: lambda x: x * x, Act.IDENTITY: lambda x: x}

    # ------------------------------------------------------------------ storage
    @staticmethod
    def _key(buf):
        return (buf.addr, buf.shape, int(buf.dtype))

    def _round(self, x, dt: DType):
        t = self.torch
        if dt == DType.BF16:
            return x.to(t.bfloat16).to(t.float32)
        if dt == DType.FP16:
            return x.to(t.float16).to(t.float32)
        if dt == DType.FP32:
            return x.to(t.float32)
        raise TypeError(dt)

    def _to_dev(self, v: np.ndarray, dt: DType):
        t = self.torch
        if dt.is_int:
            return np.asarray(v)            # integer tensors stay on the host (exact NumPy path)
        a = np.ascontiguousarray(v, dtype=np.float32)
        if not a.flags.writeable:
            a = a.copy()
        return self._round(t.as_tensor(a, device=self.device), dt)

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
                self.write(exe, name, fn())
        self._loaded.add(id(exe))

    def write(self, exe: Executable, name: str, value) -> None:
        b = exe.buffers[name]
        v = np.asarray(value)
        if v.ndim < 2:
            v = v.reshape(1, -1) if v.ndim == 1 else v.reshape(1, 1)
        if v.shape != b.shape:
            raise ValueError(f"{name}: shape {v.shape} != {b.shape}")
        if b.dtype.is_int:
            from .fast import _store
            self.mem[self._key(b)] = _store(v, b.dtype)
        else:
            self.mem[self._key(b)] = self._to_dev(v, b.dtype)

    def read(self, exe: Executable, name: str) -> np.ndarray:
        fn = getattr(self, "regen", {}).get(self._key(exe.buffers[name]))
        if fn is not None:
            return fn()
        x = self.mem[self._key(exe.buffers[name])]
        return x.copy() if isinstance(x, np.ndarray) else x.detach().to("cpu").numpy().copy()

    def params(self, exe: Executable) -> dict:
        return {p: self.read(exe, p) for p in exe.params}

    # ------------------------------------------------------------------ kernels
    def _get(self, env, buf):
        if buf.persistent:
            k = self._key(buf)
            fn = getattr(self, "regen", {}).get(k)
            return self._to_dev(fn(), buf.dtype) if fn is not None else self.mem[k]
        return env[id(buf)]

    def _put(self, env, buf, value):
        t = self.torch
        if isinstance(value, np.ndarray):
            if buf.dtype.is_int:
                v = value
            else:
                v = self._to_dev(np.broadcast_to(value, buf.shape), buf.dtype)
        else:
            v = self._round(value.expand(buf.shape) if tuple(value.shape) != buf.shape else value, buf.dtype)
            if not v.is_contiguous():
                v = v.contiguous()
        if buf.persistent:
            self.mem[self._key(buf)] = v
        else:
            env[id(buf)] = v
        return t

    def _host(self, x):
        return x if isinstance(x, np.ndarray) else x.detach().to("cpu").numpy()

    def _convert(self, x, src: DType, dst: DType, scale: float):
        t = self.torch
        if dst.is_int or src.is_int:
            from ..dtypes import convert
            return convert(self._host(x), src, dst, scale)
        return self._round(x * np.float32(scale), dst)

    def _steps(self, steps, arg):
        t = self.torch
        vals = []
        for st in steps:
            xs = [arg(a, vals) for a in st.args]
            dt, op = st.dtype, st.op
            if op == "cast":
                r = self._convert(xs[0], st.attrs["src_dtype"], dt, st.attrs["scale"])
                vals.append(r)
                continue
            xs = [self._to_dev(x, DType.FP32) if isinstance(x, np.ndarray) else x for x in xs]
            if op == "add":
                r = xs[0] + xs[1]
            elif op == "sub":
                r = xs[0] - xs[1]
            elif op == "mul":
                r = xs[0] * xs[1]
            elif op == "div":
                r = xs[0] / xs[1]
            elif op == "maximum":
                r = t.maximum(xs[0], xs[1])
            elif op == "scale":
                r = np.float32(st.attrs["alpha"]) * xs[0] + np.float32(st.attrs["beta"])
            elif op == "broadcast":
                r = xs[0]
            elif op in UNARY:
                r = self.act[UNARY[op]](xs[0])
            else:
                raise NotImplementedError(op)
            vals.append(self._round(r, dt) if dt.is_float else r)
        return vals[-1]

    def _kernel(self, exe, K, env):
        t = self.torch
        ins_int = any(b.dtype.is_int for b in K.inputs) or K.output.dtype.is_int
        if ins_int and K.kind == "gemm":
            env_np = {}
            for b in K.inputs:
                v = self._get(env, b)
                env_np[id(b)] = self._host(v)
            fd = self._np
            fd.mem = {}
            for b in K.inputs:
                if b.persistent:
                    fd.mem[fd._key(b)] = env_np[id(b)]
            return fd._kernel(K, env_np)
        if K.kind == "gemm":
            a, b = self._get(env, K.inputs[0]), self._get(env, K.inputs[1])
            bt, ta, tb = K.attrs["batch"], K.attrs["ta"], K.attrs["tb"]
            A = a.reshape(bt, a.shape[0] // bt, a.shape[1])
            B = b.reshape(bt, b.shape[0] // bt, b.shape[1])
            A = A.transpose(1, 2) if ta else A
            B = B.transpose(1, 2) if tb else B
            acc = t.matmul(A, B).reshape(-1, K.shape[1])
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
        x = self._get(env, K.inputs[0])
        if isinstance(x, np.ndarray):
            x = t.as_tensor(x.astype(np.float32), device=self.device)
        if K.kind == "rownorm":
            fn, eps = K.attrs["fn"], K.attrs["eps"] or 1e-5
            if fn == "softmax":
                e = t.exp(x - x.amax(dim=1, keepdim=True))
                return e / e.sum(dim=1, keepdim=True)
            if fn == "layernorm":
                xc = x - x.mean(dim=1, keepdim=True)
                return xc / t.sqrt((xc * xc).mean(dim=1, keepdim=True) + eps)
            return x / t.sqrt((x * x).mean(dim=1, keepdim=True) + eps)
        if K.kind == "reduce":
            ax, rop = K.attrs["axis"], K.attrs["rop"]
            if ax == 2:
                r = x.sum() if rop == ReduceOp.SUM else x.amax() if rop == ReduceOp.MAX else x.mean()
            else:
                r = (x.sum(dim=ax, keepdim=True) if rop == ReduceOp.SUM else
                     x.amax(dim=ax, keepdim=True) if rop == ReduceOp.MAX else x.mean(dim=ax, keepdim=True))
            return r.reshape(K.output.shape)
        if K.kind == "transpose":
            return x.t()
        if K.kind == "copy":
            return x.clone()
        raise NotImplementedError(K.kind)

    def run(self, exe: Executable, feeds: dict | None = None, mode: str = MODE, trace: bool = False,
            fetch: list[str] | None = None, matmul_hook=None) -> RunOutput:
        if trace or matmul_hook is not None:
            raise ValueError("the accelerated executor has no instruction trace and no RTL offload")
        if id(exe) not in self._loaded:
            self.load(exe)
        t0 = time.perf_counter()
        for k, v in (feeds or {}).items():
            self.write(exe, k, v)
        t1 = time.perf_counter()
        env: dict[int, object] = {}
        frees = self._np._frees(exe)
        ops = 0
        with self.torch.no_grad():
            for i, K in enumerate(exe.module.kernels):
                self._put(env, K.output, self._kernel(exe, K, env))
                for bid in frees[i]:
                    env.pop(bid, None)
                if K.kind == "gemm":
                    a = K.inputs[0]
                    kd = a.shape[0] // K.attrs["batch"] if K.attrs["ta"] else a.shape[1]
                    ops += 2 * K.shape[0] * K.shape[1] * kd
        if self.device.type == "cuda":
            self.torch.cuda.synchronize()
        t2 = time.perf_counter()
        outs = {k: self.read(exe, k) for k in (exe.outputs if fetch is None else fetch)}
        t3 = time.perf_counter()
        res = RunResult(MODE, 0, {"kernels": len(exe.module.kernels)}, ops, ops // 2, 0, clock_ghz=self.cfg.clock_ghz,
                        host_wall_s=t2 - t1)
        return RunOutput(outs, res, (t1 - t0) + (t3 - t2), {"host_write_s": t1 - t0, "execute_s": t2 - t1,
                                                           "host_read_s": t3 - t2})
