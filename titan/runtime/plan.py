"""Compiled execution plans: a whole TITAN program (IR kernels) becomes ONE generated Python function.

The interpreting executors spend most of a small model's step in per-kernel dispatch (dict lookups,
generic broadcasting, dtype checks, error-state switches). A plan resolves all of that once per
program: every temporary becomes a local variable, every persistent tensor a constant key, every
fused element-wise chain a straight-line expression. The arithmetic is unchanged:

  * FP32 kernels are emitted inline with exactly the operations of the interpreter
    (results are bit-identical; tests compare whole training runs);
  * kernels with other dtypes (BF16/FP16 rounding, INT8/INT32, CONVERT) and token/head kernels
    call the interpreter's own kernel function, so their numerics are shared by construction.

Storage offload and regenerated (LoRA) base weights keep the interpreter (their cost is I/O).
"""
from __future__ import annotations

import numpy as np

from ..compiler.graph import UNARY
from ..dtypes import DType
from ..emu.semantics import ACTIVATIONS, normalize
from ..isa.spec import Norm, ReduceOp

_BIN = {"add": "{0} + {1}", "sub": "{0} - {1}", "mul": "{0} * {1}", "div": "{0} / {1}",
        "maximum": "_np.maximum({0}, {1})"}
_NORM = {"softmax": Norm.SOFTMAX, "layernorm": Norm.LAYERNORM, "rmsnorm": Norm.RMSNORM}


def _fp32_kernel(K) -> bool:
    if K.output.dtype != DType.FP32 or any(b.dtype != DType.FP32 for b in K.inputs):
        return False
    return all(st.dtype == DType.FP32 and st.op != "cast" for st in K.steps)


def _fit(r, shape):
    r = np.asarray(r, dtype=np.float32)
    return r if r.shape == shape else np.broadcast_to(r, shape)


def gemm_ops(K) -> int:
    a = K.inputs[0]
    kd = a.shape[0] // K.attrs["batch"] if K.attrs["ta"] else a.shape[1]
    return 2 * K.shape[0] * K.shape[1] * kd


class _Builder:
    """Shared code generator; ``dialect`` supplies the array library expressions (NumPy or PyTorch)."""

    def __init__(self, dev, exe, g: dict, persistent_key):
        self.dev, self.exe, self.g = dev, exe, g
        self.names: dict[int, str] = {}
        self.keys: dict[tuple, str] = {}
        self.lines = ["def _plan(M):"]
        self.pkey = persistent_key

    def emit(self, line: str) -> None:
        self.lines.append("    " + line)

    def key(self, buf) -> str:
        k = self.pkey(buf)
        if k not in self.keys:
            self.keys[k] = f"_k{len(self.keys)}"
            self.g[self.keys[k]] = k
        return self.keys[k]

    def ref(self, buf) -> str:
        if buf.persistent:
            return f"M[{self.key(buf)}]"
        if id(buf) not in self.names:
            self.names[id(buf)] = f"t{len(self.names)}"
        return self.names[id(buf)]

    def steps(self, K, ins, binary, scale, acc=None) -> str:
        for j, st in enumerate(K.steps):
            a = [acc if x[0] == "acc" else ins[x[1]] if x[0] == "in" else f"s{x[1]}" for x in st.args]
            if st.op in binary:
                e = binary[st.op].format(*a)
            elif st.op == "scale":
                e = scale.format(a[0], float(st.attrs["alpha"]), float(st.attrs["beta"]))
            elif st.op == "broadcast":
                e = a[0]
            elif st.op in UNARY:
                e = f"_u_{st.op}({a[0]})"
            else:
                raise NotImplementedError(st.op)
            self.emit(f"s{j} = {e}")
        return f"s{len(K.steps) - 1}"

    def steps_inplace(self, K, ins, dialect: str, acc=None) -> str:
        """Same arithmetic as ``steps`` (bit-identical), but a value that is fresh (produced by an earlier
        step or the GEMM) and used for the last time is overwritten instead of allocating a new array."""
        refs = [[("acc",) if x[0] == "acc" else x for x in st.args] for st in K.steps]
        last = {}
        for j, args in enumerate(refs):
            for x in args:
                if x[0] != "in":
                    last[x] = j
        shape = {("acc",): tuple(K.shape)}
        owned = {("acc",)} if acc is not None else set()
        name = lambda x: acc if x[0] == "acc" else ins[x[1]] if x[0] == "in" else f"s{x[1]}"   # noqa: E731
        shp = lambda x: tuple(K.inputs[x[1]].shape) if x[0] == "in" else shape[x]             # noqa: E731
        np_ = dialect == "np"
        for j, (st, args) in enumerate(zip(K.steps, refs)):
            me = ("s", j)
            out_shape = tuple(np.broadcast_shapes(*[shp(x) for x in args]))
            reuse = [x for x in args if x in owned and last.get(x) == j and shp(x) == out_shape]
            a = [name(x) for x in args]
            op = st.op
            line = None
            if reuse and (op in _BIN or op == "scale" or op in _INPL_UNARY):
                o = name(reuse[0])
                if op in _BIN:
                    fn = {"add": "add", "sub": "subtract" if np_ else "sub", "mul": "multiply" if np_ else "mul",
                          "div": "divide" if np_ else "div", "maximum": "maximum"}[op]
                    line = f"s{j} = {'_np' if np_ else '_t'}.{fn}({a[0]}, {a[1]}, out={o})"
                elif op == "scale":
                    al, be = float(st.attrs["alpha"]), float(st.attrs["beta"])
                    line = (f"_np.multiply({o}, _f32({al!r}), out={o}); s{j} = _np.add({o}, _f32({be!r}), out={o})"
                            if np_ else f"s{j} = {o}.mul_(_f32({al!r})).add_(_f32({be!r}))")
                else:
                    line = (_INPL_UNARY[op][0] if np_ else _INPL_UNARY[op][1]).format(x=o, j=j)
                owned.discard(reuse[0])
            if line is None:
                if op in _BIN:
                    e = (_BIN if np_ else _T_BIN)[op].format(*a)
                elif op == "scale":
                    e = (_NP_SCALE if np_ else _T_SCALE).format(a[0], float(st.attrs["alpha"]), float(st.attrs["beta"]))
                elif op == "broadcast":
                    e = a[0]
                elif op in UNARY:
                    e = f"_u_{op}({a[0]})"
                else:
                    raise NotImplementedError(op)
                line = f"s{j} = {e}"
            self.emit(line)
            shape[me] = out_shape
            if op != "broadcast":
                owned.add(me)
            else:
                owned.discard(args[0])                 # aliased: never overwrite it afterwards
        return f"s{len(K.steps) - 1}"

    def free(self, bids) -> None:
        for bid in bids:
            if bid in self.names:
                self.emit(f"{self.names[bid]} = None")

    def finish(self, label: str):
        self.emit("return None")
        src = "\n".join(self.lines)
        exec(compile(src, f"<titan-plan {label}>", "exec"), self.g)
        fn = self.g["_plan"]
        fn.source = src
        return fn


def numpy_plan(dev, exe):
    """Return (fn, ops): ``fn(mem)`` executes every kernel of ``exe`` on the FastDevice storage ``mem``."""
    from .fast import _store
    g = {"_np": np, "_f32": np.float32, "_fit": _fit, "_norm": normalize, "_store": _store,
         "_bcast": np.broadcast_to, "_kern": dev._kernel}
    for op, act in UNARY.items():
        g[f"_u_{op}"] = ACTIVATIONS[act]
    B = _Builder(dev, exe, g, dev._key)
    frees = dev._frees(exe)
    ops = 0
    for i, K in enumerate(exe.module.kernels):
        out = K.output
        shape = tuple(out.shape)
        ins = [B.ref(b) for b in K.inputs]
        if K.kind == "gemm":
            ops += gemm_ops(K)
        if _fp32_kernel(K) and K.kind in ("gemm", "ewise", "rownorm", "reduce", "transpose", "copy"):
            if K.kind == "gemm":
                bt, ta, tb = K.attrs["batch"], K.attrs["ta"], K.attrs["tb"]
                a, b = K.inputs[0].shape, K.inputs[1].shape
                if bt == 1:
                    B.emit(f"acc = _np.matmul({ins[0]}{'.T' if ta else ''}, {ins[1]}{'.T' if tb else ''})")
                else:
                    A_ = f"{ins[0]}.reshape({bt}, {a[0] // bt}, {a[1]})" + (".swapaxes(1, 2)" if ta else "")
                    B_ = f"{ins[1]}.reshape({bt}, {b[0] // bt}, {b[1]})" + (".swapaxes(1, 2)" if tb else "")
                    B.emit(f"acc = _np.matmul({A_}, {B_}).reshape(-1, {K.shape[1]})")
                val = ((B.steps_inplace(K, ins, "np", "acc") if INPLACE else B.steps(K, ins, _BIN, _NP_SCALE, "acc"))
                       if K.steps else "acc")
            elif K.kind == "ewise":
                val = B.steps_inplace(K, ins, "np") if INPLACE else B.steps(K, ins, _BIN, _NP_SCALE)
            elif K.kind == "rownorm":
                g[f"_N{i}"] = _NORM[K.attrs["fn"]]
                val = f"_norm(_N{i}, {ins[0]}, {K.attrs['eps'] or 1e-5!r})"
            elif K.kind == "reduce":
                ax = {0: 0, 1: 1, 2: None}[K.attrs["axis"]]
                rop = K.attrs["rop"]
                fn = ("sum" if rop == ReduceOp.SUM else "max" if rop == ReduceOp.MAX else "mean")
                extra = ", dtype=_f32" if fn != "max" else ""
                val = f"_np.asarray({ins[0]}.{fn}(axis={ax}, keepdims=True{extra})).reshape({shape})"
            elif K.kind == "transpose":
                val = f"{ins[0]}.T"
            else:
                val = ins[0]
            B.emit(f"{B.ref(out)} = _fit({val}, {shape})")
        else:                                           # shared interpreter kernel (exact non-FP32 numerics)
            g[f"_K{i}"], g[f"_dt{i}"] = K, out.dtype
            env = ", ".join(f"{id(b)}: {ins[n]}" for n, b in enumerate(K.inputs) if not b.persistent)
            B.emit(f"{B.ref(out)} = _store(_bcast(_kern(_K{i}, {{{env}}}), {shape}), _dt{i})")
        B.free(frees[i])
    return B.finish("numpy"), ops


_NP_SCALE = "_f32({1!r}) * {0} + _f32({2!r})"
# in-place forms of unary activations, op for op identical to titan.emu.semantics / AccelDevice.act
_INPL_UNARY = {
    "exp": ("s{j} = _np.exp({x}, out={x})", "s{j} = {x}.exp_()"),
    "log": ("s{j} = _np.log({x}, out={x})", "s{j} = {x}.log_()"),
    "sqrt": ("s{j} = _np.sqrt({x}, out={x})", "s{j} = {x}.sqrt_()"),
    "tanh": ("s{j} = _np.tanh({x}, out={x})", "s{j} = {x}.tanh_()"),
    "square": ("s{j} = _np.multiply({x}, {x}, out={x})", "s{j} = {x}.mul_({x})"),
    "relu": ("s{j} = _np.maximum({x}, _f32(0), out={x})", "s{j} = {x}.relu_()"),
    "recip": ("s{j} = _np.divide(_f32(1), {x}, out={x})", "s{j} = {x}.reciprocal_()"),
    "rsqrt": ("_np.sqrt({x}, out={x}); s{j} = _np.divide(_f32(1), {x}, out={x})", "s{j} = {x}.rsqrt_()"),
    "sigmoid": ("_np.negative({x}, out={x}); _np.exp({x}, out={x}); _np.add(_f32(1), {x}, out={x}); "
                "s{j} = _np.divide(_f32(1), {x}, out={x})", "s{j} = {x}.sigmoid_()"),
}
INPLACE = __import__("os").environ.get("TITAN_PLAN_INPLACE", "1") != "0"
_T_BIN = {**_BIN, "maximum": "_t.maximum({0}, {1})"}
_T_SCALE = "_f32({1!r}) * {0} + _f32({2!r})"


def torch_plan(dev, exe):
    """PyTorch dialect of the same plan (CPU or CUDA tensors); non-FP32 kernels use AccelDevice._kernel."""
    import torch
    g = {"_t": torch, "_np": np, "_f32": np.float32, "_kern": dev._kernel, "_conv": dev._conv, "_exe": exe,
         "_fitt": _fit_torch}
    for op, act in UNARY.items():
        g[f"_u_{op}"] = dev.act[act]
    B = _Builder(dev, exe, g, dev._key)
    frees = dev._np._frees(exe)
    producer = {id(K.output): K.kind for K in exe.module.kernels}
    for K in exe.module.kernels:                      # negative ids only come from MoE routing tables
        if K.kind in ("embedding", "embedding_grad"):
            K.attrs["neg_ids"] = producer.get(id(K.inputs[0])) == "moe_slots"
    ops = 0
    for i, K in enumerate(exe.module.kernels):
        out = K.output
        shape = tuple(out.shape)
        ins = [B.ref(b) for b in K.inputs]
        if K.kind == "gemm":
            ops += gemm_ops(K)
        if _fp32_kernel(K) and K.kind in ("gemm", "ewise", "rownorm", "reduce", "transpose", "copy"):
            if K.kind == "gemm":
                bt, ta, tb = K.attrs["batch"], K.attrs["ta"], K.attrs["tb"]
                a, b = K.inputs[0].shape, K.inputs[1].shape
                if bt == 1:
                    B.emit(f"acc = _t.matmul({ins[0]}{'.t()' if ta else ''}, {ins[1]}{'.t()' if tb else ''})")
                else:
                    A_ = f"{ins[0]}.reshape({bt}, {a[0] // bt}, {a[1]})" + (".transpose(1, 2)" if ta else "")
                    B_ = f"{ins[1]}.reshape({bt}, {b[0] // bt}, {b[1]})" + (".transpose(1, 2)" if tb else "")
                    B.emit(f"acc = _t.matmul({A_}, {B_}).reshape(-1, {K.shape[1]})")
                val = ((B.steps_inplace(K, ins, "torch", "acc") if INPLACE else B.steps(K, ins, _T_BIN, _T_SCALE, "acc"))
                       if K.steps else "acc")
            elif K.kind == "ewise":
                val = B.steps_inplace(K, ins, "torch") if INPLACE else B.steps(K, ins, _T_BIN, _T_SCALE)
            elif K.kind == "rownorm":
                x, eps = ins[0], K.attrs["eps"] or 1e-5
                fn = K.attrs["fn"]
                if fn == "softmax":
                    B.emit(f"e = _t.exp({x} - {x}.amax(dim=1, keepdim=True))")
                    val = "e / e.sum(dim=1, keepdim=True)"
                elif fn == "layernorm":
                    B.emit(f"xc = {x} - {x}.mean(dim=1, keepdim=True)")
                    val = f"xc / _t.sqrt((xc * xc).mean(dim=1, keepdim=True) + {eps!r})"
                else:
                    val = f"{x} / _t.sqrt(({x} * {x}).mean(dim=1, keepdim=True) + {eps!r})"
            elif K.kind == "reduce":
                ax, rop = K.attrs["axis"], K.attrs["rop"]
                fn = "sum" if rop == ReduceOp.SUM else "amax" if rop == ReduceOp.MAX else "mean"
                val = (f"{ins[0]}.{fn}()" if ax == 2 else f"{ins[0]}.{fn}(dim={ax}, keepdim=True)") + \
                    f".reshape({shape})"
            elif K.kind == "transpose":
                val = f"{ins[0]}.t()"
            else:
                val = f"{ins[0]}.clone()"
            B.emit(f"{B.ref(out)} = _fitt({val}, {shape})")
        else:
            g[f"_K{i}"], g[f"_B{i}"] = K, out
            env = ", ".join(f"{id(b)}: {ins[n]}" for n, b in enumerate(K.inputs) if not b.persistent)
            B.emit(f"{B.ref(out)} = _conv(_B{i}, _kern(_exe, _K{i}, {{{env}}}))")
        B.free(frees[i])
    return B.finish("torch"), ops


def _fit_torch(r, shape):
    if tuple(r.shape) != shape:
        r = r.expand(shape)
    return r if r.is_contiguous() else r.contiguous()
