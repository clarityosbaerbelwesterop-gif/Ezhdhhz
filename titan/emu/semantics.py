"""Functional semantics of every data-processing TITAN instruction.

This is the architectural reference ("golden model"): the cycle model and the
RTL co-simulation must produce bit-identical results for integer operations
and results within documented tolerances for floating-point operations.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..dtypes import DType, convert, from_compute, round_to, to_compute, wrap_int32
from ..isa.spec import Act, Axis, ErrorCode, Flag, Instruction, ISAError, Norm, Opcode, ReduceOp, mm_dims

_SQRT_2_PI = np.float32(np.sqrt(2.0 / np.pi))


@dataclass
class OpStats:
    ops: int = 0           # arithmetic operations (1 MAC = 2 ops)
    mxu_macs: int = 0
    elems: int = 0         # elements processed by VPU/SFU
    bytes: int = 0         # bytes moved by DMA


def _gelu(x):
    x2 = x * x                       # x*x*x, not x**3: NumPy evaluates float powers element-wise via powf (~20x slower)
    return 0.5 * x * (1.0 + np.tanh(_SQRT_2_PI * (x + np.float32(0.044715) * (x2 * x))))


def _gelu_grad(x):
    x2 = x * x
    u = _SQRT_2_PI * (x + np.float32(0.044715) * (x2 * x))
    t = np.tanh(u)
    du = _SQRT_2_PI * (1.0 + np.float32(3 * 0.044715) * x2)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du


ACTIVATIONS = {
    Act.RELU: lambda x: np.maximum(x, np.float32(0)),
    Act.GELU: _gelu,
    Act.SIGMOID: lambda x: np.float32(1) / (np.float32(1) + np.exp(-x)),
    Act.TANH: np.tanh,
    Act.EXP: np.exp,
    Act.LOG: np.log,
    Act.SQRT: np.sqrt,
    Act.RSQRT: lambda x: np.float32(1) / np.sqrt(x),
    Act.RECIP: lambda x: np.float32(1) / x,
    Act.STEP: lambda x: (x > 0).astype(np.float32),
    Act.GELU_GRAD: _gelu_grad,
    Act.SQUARE: lambda x: x * x,
    Act.IDENTITY: lambda x: x,
}


def normalize(fn: Norm, x: np.ndarray, eps: float) -> np.ndarray:
    x = x.astype(np.float32)
    if fn == Norm.SOFTMAX:
        e = np.exp(x - x.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
    if fn == Norm.LAYERNORM:
        mu = x.mean(axis=1, keepdims=True)
        xc = x - mu
        var = (xc * xc).mean(axis=1, keepdims=True)
        return xc / np.sqrt(var + np.float32(eps))
    if fn == Norm.RMSNORM:
        return x / np.sqrt((x * x).mean(axis=1, keepdims=True) + np.float32(eps))
    raise ISAError(ErrorCode.BAD_FUNC, f"normalization {fn}")


def matmul_values(a: np.ndarray, b: np.ndarray, dt: DType, ta: bool, tb: bool,
                  c: np.ndarray | None = None) -> np.ndarray:
    """Matrix-engine arithmetic: ints -> exact products, INT32 wrap; floats -> FP32 accumulate."""
    A = a.T if ta else a
    B = b.T if tb else b
    if dt.is_int:
        r = A.astype(np.int64) @ B.astype(np.int64)
        if c is not None:
            r = r + c.astype(np.int64)
        return wrap_int32(r)
    r = A.astype(np.float32) @ B.astype(np.float32)
    if c is not None:
        r = r + c.astype(np.float32)
    return r.astype(np.float32)


def stats_of(ins: Instruction) -> OpStats:
    """Work done by one data instruction (depends on shapes only)."""
    op, o = ins.opcode, ins.operands
    st = OpStats()
    if op in (Opcode.LOAD, Opcode.STORE, Opcode.COPY):
        st.bytes = o[1].nbytes
    elif op == Opcode.MATMUL:
        m, n, k = mm_dims(ins)
        st.mxu_macs, st.ops = m * n * k, 2 * m * n * k
    elif op in (Opcode.VECTOR_ADD, Opcode.VECTOR_MULTIPLY, Opcode.VECTOR_SUB, Opcode.VECTOR_MAX,
                Opcode.VECTOR_DIV, Opcode.VECTOR_SCALE_ADD):
        st.elems = o[0].numel
        st.ops = o[0].numel * (3 if op == Opcode.VECTOR_SCALE_ADD else 1)
    elif op == Opcode.REDUCE:
        st.elems = st.ops = o[1].numel
    elif op in (Opcode.ACCUMULATE, Opcode.ACTIVATION, Opcode.CONVERT):
        st.elems = st.ops = o[0].numel
    elif op == Opcode.NORMALIZATION:
        st.elems, st.ops = o[0].numel, 5 * o[0].numel
    return st


def dma_copy(src_mem, dst_mem, ins: Instruction) -> OpStats:
    """LOAD / STORE / COPY: copy the stored bits (optionally transposed). DMA never converts."""
    dst, src = ins.operands
    d, s = dst_mem.raw_view(dst), src_mem.raw_view(src)
    if ins.has(Flag.TRANSPOSE):
        s = s.T
    if src_mem is dst_mem and dst.addr < src.end and src.addr < dst.end:
        s = s.copy()
    np.copyto(d, s)
    return OpStats(bytes=src.nbytes)


def execute(read, write, ins: Instruction) -> OpStats:
    """Execute one data instruction. ``read(ref)``/``write(ref, values)`` access the issuing core's view."""
    op, o = ins.opcode, ins.operands
    st = OpStats()
    if op in (Opcode.LOAD, Opcode.STORE, Opcode.COPY):
        dst, src = o
        v = read(src)
        if ins.has(Flag.TRANSPOSE):
            v = v.T
        write(dst, v)
        st.bytes = src.nbytes
        return st
    if op == Opcode.MATMUL:
        c, a, b = o
        m, n, k = mm_dims(ins)
        prev = read(c) if ins.has(Flag.ACC) else None
        write(c, matmul_values(read(a), read(b), a.dtype, ins.has(Flag.TRANS_A), ins.has(Flag.TRANS_B), prev))
        st.mxu_macs = m * n * k
        st.ops = 2 * m * n * k
        return st
    if op in (Opcode.VECTOR_ADD, Opcode.VECTOR_MULTIPLY, Opcode.VECTOR_SUB, Opcode.VECTOR_MAX,
              Opcode.VECTOR_DIV, Opcode.VECTOR_SCALE_ADD):
        dst = o[0]
        a = to_compute(read(o[1]), o[1].dtype)
        b = to_compute(read(o[2]), o[2].dtype) if len(o) == 3 else None
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if op == Opcode.VECTOR_ADD:
                r = a + b
            elif op == Opcode.VECTOR_MULTIPLY:
                r = a * b
            elif op == Opcode.VECTOR_SUB:
                r = a - b
            elif op == Opcode.VECTOR_MAX:
                r = np.maximum(a, b)
            elif op == Opcode.VECTOR_DIV:
                r = a / b
            else:
                alpha, beta = np.float32(ins.fimm(0)), np.float32(ins.fimm(1))
                r = alpha * a + (beta * b if b is not None else beta)
        r = np.broadcast_to(r, dst.shape)
        write(dst, from_compute(r, dst.dtype))
        st.elems = dst.numel
        st.ops = dst.numel * (3 if op == Opcode.VECTOR_SCALE_ADD else 1)
        return st
    if op == Opcode.REDUCE:
        dst, src = o
        x = to_compute(read(src), src.dtype)
        ax = {Axis.ROWS: 0, Axis.COLS: 1, Axis.ALL: None}[Axis(ins.imm[0])]
        fn = ReduceOp(ins.func)
        if fn == ReduceOp.SUM:
            r = x.sum(axis=ax, keepdims=True, dtype=x.dtype)
        elif fn == ReduceOp.MAX:
            r = x.max(axis=ax, keepdims=True)
        else:
            r = x.mean(axis=ax, keepdims=True, dtype=np.float32)
        write(dst, from_compute(np.asarray(r).reshape(dst.shape), dst.dtype))
        st.elems = st.ops = src.numel
        return st
    if op == Opcode.ACCUMULATE:
        dst, src = o
        r = to_compute(read(dst), dst.dtype) + to_compute(read(src), src.dtype)
        write(dst, from_compute(r, dst.dtype))
        st.elems = st.ops = dst.numel
        return st
    if op == Opcode.ACTIVATION:
        dst, src = o
        x = to_compute(read(src), src.dtype)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            r = ACTIVATIONS[Act(ins.func)](x).astype(np.float32)
        write(dst, from_compute(r, dst.dtype))
        st.elems = st.ops = dst.numel
        return st
    if op == Opcode.NORMALIZATION:
        dst, src = o
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            r = normalize(Norm(ins.func), to_compute(read(src), src.dtype), ins.fimm(0, 1e-5))
        write(dst, from_compute(r, dst.dtype))
        st.elems = dst.numel
        st.ops = 5 * dst.numel
        return st
    if op == Opcode.CONVERT:
        dst, src = o
        write(dst, convert(read(src), src.dtype, dst.dtype, ins.fimm(0, 1.0)))
        st.elems = st.ops = dst.numel
        return st
    if op == Opcode.NOP:
        return st
    raise ISAError(ErrorCode.ILLEGAL_OPCODE, f"{op.name} is not a data instruction")


