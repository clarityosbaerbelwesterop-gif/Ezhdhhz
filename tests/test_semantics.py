"""Per-instruction numerical tests of the functional golden model against NumPy references."""
import numpy as np
import pytest

from titan.dtypes import DType, round_bf16
from titan.emu import Machine, run
from titan.isa import Act, Norm, Opcode, Program, ReduceOp, Space
from titan.isa import build as B
from tests.conftest import Placer


def exec_local(cfg, tensors, instrs, mode="functional"):
    """tensors: list of (ref, value or None). Returns machine after running instrs on core 0."""
    m = Machine(cfg)
    for ref, v in tensors:
        if v is not None:
            m.local[0].write(ref, v)
    run(m, Program(instrs), mode)
    return m


@pytest.mark.parametrize("dt", [DType.INT4, DType.INT8, DType.FP16, DType.BF16, DType.FP32])
@pytest.mark.parametrize("ta,tb", [(False, False), (True, False), (False, True), (True, True)])
def test_matmul_all_precisions(tiny, rng, dt, ta, tb):
    m_, n_, k_ = 6, 10, 12
    P = Placer(Space.LOCAL)
    if dt.is_int:
        lo, hi = (-8, 8) if dt == DType.INT4 else (-128, 128)
        A = rng.integers(lo, hi, (m_, k_)).astype(np.int8)
        Bm = rng.integers(lo, hi, (k_, n_)).astype(np.int8)
        acc_dt = DType.INT32
    else:
        A = rng.standard_normal((m_, k_)).astype(np.float32)
        Bm = rng.standard_normal((k_, n_)).astype(np.float32)
        acc_dt = DType.FP32
        if dt == DType.BF16:
            A, Bm = round_bf16(A), round_bf16(Bm)
        if dt == DType.FP16:
            A, Bm = A.astype(np.float16), Bm.astype(np.float16)
    As = A.T.copy() if ta else A
    Bs = Bm.T.copy() if tb else Bm
    ra, rb, rc = P(*As.shape, dt), P(*Bs.shape, dt), P(m_, n_, acc_dt)
    C0 = (rng.integers(-1000, 1000, (m_, n_)) if dt.is_int else rng.standard_normal((m_, n_))).astype(
        np.int32 if dt.is_int else np.float32)
    mach = exec_local(tiny, [(ra, As), (rb, Bs), (rc, C0)], [B.matmul(0, rc, ra, rb, acc=True, ta=ta, tb=tb)])
    got = mach.local[0].read(rc)
    if dt.is_int:
        np.testing.assert_array_equal(got, A.astype(np.int64) @ Bm.astype(np.int64) + C0)
    else:
        ref = A.astype(np.float64) @ Bm.astype(np.float64) + C0
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)


def test_int8_matmul_accumulator_wraps_like_rtl(tiny):
    P = Placer(Space.LOCAL)
    ra, rb, rc = P(1, 2, DType.INT8), P(2, 1, DType.INT8), P(1, 1, DType.INT32)
    mach = exec_local(tiny, [(ra, [[127, 127]]), (rb, [[127], [127]]), (rc, [[2**31 - 100]])],
                      [B.matmul(0, rc, ra, rb, acc=True)])
    want = ((2**31 - 100 + 2 * 127 * 127 + 2**31) % 2**32) - 2**31
    assert mach.local[0].read(rc)[0, 0] == want < 0


@pytest.mark.parametrize("op,fn", [(Opcode.VECTOR_ADD, np.add), (Opcode.VECTOR_MULTIPLY, np.multiply),
                                   (Opcode.VECTOR_SUB, np.subtract), (Opcode.VECTOR_MAX, np.maximum),
                                   (Opcode.VECTOR_DIV, np.divide)])
@pytest.mark.parametrize("bc", [None, "row", "col", "scalar"])
def test_vector_binary_broadcast(tiny, rng, op, fn, bc):
    P = Placer(Space.LOCAL)
    a = rng.standard_normal((5, 7)).astype(np.float32)
    bshape = {None: (5, 7), "row": (1, 7), "col": (5, 1), "scalar": (1, 1)}[bc]
    b = rng.uniform(0.5, 2, bshape).astype(np.float32)
    ra, rb, rd = P(5, 7, DType.FP32), P(*bshape, DType.FP32), P(5, 7, DType.FP32)
    m = exec_local(tiny, [(ra, a), (rb, b)], [B.vec(op, 0, rd, ra, rb, bcast_b=bc)])
    np.testing.assert_allclose(m.local[0].read(rd), fn(a, b), rtol=1e-6)


def test_vector_int32_wraps_and_bf16_rounds(tiny):
    P = Placer(Space.LOCAL)
    ra, rb, rd = P(1, 2, DType.INT32), P(1, 2, DType.INT32), P(1, 2, DType.INT32)
    m = exec_local(tiny, [(ra, [[2**31 - 1, 5]]), (rb, [[1, -7]])], [B.vec(Opcode.VECTOR_ADD, 0, rd, ra, rb)])
    np.testing.assert_array_equal(m.local[0].read(rd), [[-2**31, -2]])
    fa, fb, fd = P(1, 1, DType.BF16), P(1, 1, DType.BF16), P(1, 1, DType.BF16)
    m = exec_local(tiny, [(fa, [[1.0]]), (fb, [[2**-8]])], [B.vec(Opcode.VECTOR_ADD, 0, fd, fa, fb)])
    assert m.local[0].read(fd)[0, 0] == 1.0      # 1 + 2^-8 rounds to even in BF16


def test_scale_add_forms(tiny, rng):
    P = Placer(Space.LOCAL)
    a, b = rng.standard_normal((3, 4)).astype(np.float32), rng.standard_normal((3, 4)).astype(np.float32)
    ra, rb, d1, d2 = P(3, 4, DType.FP32), P(3, 4, DType.FP32), P(3, 4, DType.FP32), P(3, 4, DType.FP32)
    m = exec_local(tiny, [(ra, a), (rb, b)], [B.scale_add(0, d1, ra, 0.5, -2.0, rb), B.scale_add(0, d2, ra, 3.0, 1.5)])
    np.testing.assert_allclose(m.local[0].read(d1), 0.5 * a - 2 * b, rtol=1e-6)
    np.testing.assert_allclose(m.local[0].read(d2), 3 * a + 1.5, rtol=1e-6)


@pytest.mark.parametrize("fn", list(ReduceOp))
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_reduce(tiny, rng, fn, axis):
    P = Placer(Space.LOCAL)
    x = rng.standard_normal((6, 9)).astype(np.float32)
    shape = {0: (1, 9), 1: (6, 1), 2: (1, 1)}[axis]
    rx, rd = P(6, 9, DType.FP32), P(*shape, DType.FP32)
    m = exec_local(tiny, [(rx, x)], [B.reduce(0, rd, rx, fn, axis)])
    npf = {ReduceOp.SUM: np.sum, ReduceOp.MAX: np.max, ReduceOp.MEAN: np.mean}[fn]
    ref = npf(x, axis={0: 0, 1: 1, 2: None}[axis], keepdims=True).reshape(shape)
    np.testing.assert_allclose(m.local[0].read(rd), ref, rtol=1e-5, atol=1e-6)


def test_accumulate(tiny, rng):
    P = Placer(Space.LOCAL)
    a, b = rng.standard_normal((4, 4)).astype(np.float32), rng.standard_normal((4, 4)).astype(np.float32)
    ra, rb = P(4, 4, DType.FP32), P(4, 4, DType.FP32)
    m = exec_local(tiny, [(ra, a), (rb, b)], [B.accumulate(0, ra, rb), B.accumulate(0, ra, rb)])
    np.testing.assert_allclose(m.local[0].read(ra), a + 2 * b, rtol=1e-6)


def _gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


ACT_REF = {
    Act.RELU: lambda x: np.maximum(x, 0), Act.GELU: _gelu, Act.SIGMOID: lambda x: 1 / (1 + np.exp(-x)),
    Act.TANH: np.tanh, Act.EXP: np.exp, Act.LOG: np.log, Act.SQRT: np.sqrt, Act.RSQRT: lambda x: 1 / np.sqrt(x),
    Act.RECIP: lambda x: 1 / x, Act.STEP: lambda x: (x > 0) * 1.0,
    Act.GELU_GRAD: lambda x: (_gelu(x + 1e-4) - _gelu(x - 1e-4)) / 2e-4,
    Act.SQUARE: np.square, Act.IDENTITY: lambda x: x,
}


@pytest.mark.parametrize("fn", list(Act))
def test_activation(tiny, rng, fn):
    P = Placer(Space.LOCAL)
    x = rng.standard_normal((4, 8)).astype(np.float32)
    if fn in (Act.LOG, Act.SQRT, Act.RSQRT, Act.RECIP):
        x = np.abs(x) + 0.1
    rx, rd = P(4, 8, DType.FP32), P(4, 8, DType.FP32)
    m = exec_local(tiny, [(rx, x)], [B.activation(0, rd, rx, fn)])
    np.testing.assert_allclose(m.local[0].read(rd), ACT_REF[fn](x.astype(np.float64)), rtol=2e-4, atol=1e-5)


@pytest.mark.parametrize("fn", list(Norm))
def test_normalization(tiny, rng, fn):
    P = Placer(Space.LOCAL)
    x = (rng.standard_normal((5, 16)) * 3 + 1).astype(np.float32)
    rx, rd = P(5, 16, DType.FP32), P(5, 16, DType.FP32)
    m = exec_local(tiny, [(rx, x)], [B.normalization(0, rd, rx, fn, 1e-5)])
    x64 = x.astype(np.float64)
    if fn == Norm.SOFTMAX:
        e = np.exp(x64 - x64.max(1, keepdims=True))
        ref = e / e.sum(1, keepdims=True)
    elif fn == Norm.LAYERNORM:
        ref = (x64 - x64.mean(1, keepdims=True)) / np.sqrt(x64.var(1, keepdims=True) + 1e-5)
    else:
        ref = x64 / np.sqrt((x64 ** 2).mean(1, keepdims=True) + 1e-5)
    np.testing.assert_allclose(m.local[0].read(rd), ref, rtol=1e-4, atol=1e-5)


def test_convert_quantize_dequantize(tiny, rng):
    P = Placer(Space.LOCAL)
    x = rng.standard_normal((4, 8)).astype(np.float32)
    rx, rq, ry, rh = P(4, 8, DType.FP32), P(4, 8, DType.INT8), P(4, 8, DType.FP32), P(4, 8, DType.BF16)
    s = 127 / np.abs(x).max()
    m = exec_local(tiny, [(rx, x)], [B.convert(0, rq, rx, s), B.convert(0, ry, rq, 1 / s), B.convert(0, rh, rx)])
    q = m.local[0].read(rq)
    np.testing.assert_array_equal(q, np.clip(np.rint(x * np.float32(s)), -128, 127))
    np.testing.assert_allclose(m.local[0].read(ry), x, atol=1 / s)
    np.testing.assert_array_equal(m.local[0].read(rh), round_bf16(x))


def test_copy_transpose_and_strided_views(tiny, rng):
    P = Placer(Space.LOCAL)
    x = rng.standard_normal((6, 8)).astype(np.float32)
    rx, rt = P(6, 8, DType.FP32), P(4, 3, DType.FP32)
    sub = rx.sub(1, 2, 3, 4)
    m = exec_local(tiny, [(rx, x)], [B.copy(0, rt, sub, transpose=True)])
    np.testing.assert_array_equal(m.local[0].read(rt), x[1:4, 2:6].T)
