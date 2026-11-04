"""End-to-end compiler tests: graph -> TITAN IR -> ISA -> virtual accelerator -> compare with NumPy reference."""
import numpy as np
import pytest

from titan.compiler import CompileOptions, Graph, compile_graph, run_reference
from titan.compiler.lower import CompileError
from titan.dtypes import DType, round_bf16
from titan.runtime import Device


def check(g, feeds, cfg, opts=None, modes=("functional",), rtol=1e-4, atol=1e-4, params=None):
    exe = compile_graph(g, cfg, opts)
    ref, assigned = run_reference(g, feeds, params)
    dev = Device(cfg)
    for mode in modes:
        dev = Device(cfg)
        dev.load(exe)
        out = dev.run(exe, feeds, mode)
        for k, v in ref.items():
            got = out.outputs[k]
            if np.issubdtype(got.dtype, np.integer):
                np.testing.assert_array_equal(got, v, err_msg=k)
            else:
                np.testing.assert_allclose(got, v, rtol=rtol, atol=atol, err_msg=f"{k} ({mode})")
        for k, v in assigned.items():
            np.testing.assert_allclose(dev.read(exe, k), v, rtol=rtol, atol=atol, err_msg=f"assign {k}")
    return exe, out


@pytest.mark.parametrize("m,k,n", [(1, 1, 1), (7, 13, 5), (64, 64, 64), (100, 37, 129), (3, 300, 2)])
@pytest.mark.parametrize("ta,tb", [(False, False), (True, True)])
def test_matmul_shapes(tiny, rng, m, k, n, ta, tb):
    g = Graph()
    a = g.input("a", (k, m) if ta else (m, k))
    b = g.input("b", (n, k) if tb else (k, n))
    g.output("c", g.matmul(a, b, ta, tb))
    feeds = {"a": rng.standard_normal(a.shape), "b": rng.standard_normal(b.shape)}
    check(g, feeds, tiny, modes=("functional", "cycle"))


@pytest.mark.parametrize("opts", [
    CompileOptions(), CompileOptions(gemm_macro=True), CompileOptions(resident_b=False),
    CompileOptions(double_buffer=False), CompileOptions(gemm_distribution="tiles"),
    CompileOptions(gemm_tiles=(8, 8, 8), resident_b=False), CompileOptions(cores=1),
])
def test_matmul_lowering_variants(dev, rng, opts):
    g = Graph()
    a, b = g.input("a", (96, 72)), g.input("b", (72, 40))
    g.output("c", g.matmul(a, b))
    exe, _ = check(g, {"a": rng.standard_normal((96, 72)), "b": rng.standard_normal((72, 40))}, dev, opts,
                   modes=("functional", "cycle"))
    if opts.gemm_macro:
        assert any(i.opcode.name == "GEMM" for i in exe.program.instructions)


@pytest.mark.parametrize("dt", [DType.INT8, DType.INT4, DType.BF16, DType.FP16])
def test_matmul_precisions(tiny, rng, dt):
    g = Graph()
    a, b = g.input("a", (20, 24), dt), g.input("b", (24, 18), dt)
    g.output("c", g.matmul(a, b))
    if dt.is_int:
        lim = 8 if dt == DType.INT4 else 128
        feeds = {"a": rng.integers(-lim, lim, (20, 24)), "b": rng.integers(-lim, lim, (24, 18))}
    else:
        feeds = {"a": rng.standard_normal((20, 24)), "b": rng.standard_normal((24, 18))}
    check(g, feeds, tiny, rtol=1e-3, atol=1e-3)


def test_quantized_linear_int8_pipeline(tiny, rng):
    """FP32 -> INT8 quantize -> INT8 GEMM (INT32 accumulate) -> dequantize, all on TITAN."""
    g = Graph()
    x = g.input("x", (16, 32))
    wq = g.param("wq", rng.integers(-127, 128, (32, 24)), DType.INT8)
    sx, sw = 127 / 3.0, 0.02
    xq = g.cast(x, DType.INT8, sx)
    acc = g.matmul(xq, wq)
    g.output("y", g.cast(acc, DType.FP32, 1 / sx * sw))
    check(g, {"x": rng.uniform(-3, 3, (16, 32))}, tiny, rtol=1e-5, atol=1e-5)


def test_bmm_batched(tiny, rng):
    g = Graph()
    q, k = g.input("q", (3 * 5, 8)), g.input("k", (3 * 6, 8))
    g.output("s", g.bmm(q, k, 3, tb=True))
    check(g, {"q": rng.standard_normal((15, 8)), "k": rng.standard_normal((18, 8))}, tiny,
          modes=("functional", "cycle"))


def _mlp_graph(rng, act="relu"):
    g = Graph("mlp")
    x = g.input("x", (40, 24))
    w1, b1 = g.param("w1", rng.standard_normal((24, 32)) * 0.2), g.param("b1", rng.standard_normal((1, 32)))
    w2 = g.param("w2", rng.standard_normal((32, 10)) * 0.2)
    h = getattr(g, act)(g.add(g.matmul(x, w1), b1))
    y = g.matmul(h, w2)
    g.output("y", g.scale(g.tanh(y), 2.0, 0.5))
    return g


@pytest.mark.parametrize("act", ["relu", "gelu", "sigmoid", "tanh"])
def test_fused_vs_unfused_equivalent(tiny, rng, act):
    g = _mlp_graph(rng, act)
    feeds = {"x": rng.standard_normal((40, 24))}
    fused, o1 = check(g, feeds, tiny, CompileOptions(fuse=True), modes=("cycle",))
    plain, o2 = check(g, feeds, tiny, CompileOptions(fuse=False), modes=("cycle",))
    assert fused.stats["kernels"] < plain.stats["kernels"]
    assert fused.stats["fused_nodes"] >= 4
    assert o1.result.dma_bytes < o2.result.dma_bytes
    np.testing.assert_allclose(o1.outputs["y"], o2.outputs["y"], rtol=1e-6, atol=1e-6)


def test_elementwise_broadcast_and_chunking(tiny, rng):
    g = Graph()
    x = g.input("x", (300, 700))
    r, c, s = g.input("r", (1, 700)), g.input("c", (300, 1)), g.input("s", (1, 1))
    y = g.div(g.mul(g.sub(g.add(x, r), c), s), g.add(g.exp(g.scale(c, 0.1)), r))
    g.output("y", g.maximum(y, g.relu(g.broadcast(r, (300, 700)))))
    feeds = {"x": rng.standard_normal((300, 700)), "r": rng.uniform(1, 2, (1, 700)),
             "c": rng.standard_normal((300, 1)), "s": rng.standard_normal((1, 1))}
    check(g, feeds, tiny, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("op", ["sum", "mean", "max"])
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("shape", [(5, 3), (130, 900)])
def test_reductions(tiny, rng, op, axis, shape):
    g = Graph()
    x = g.input("x", shape)
    g.output("r", getattr(g, op)(x, axis))
    check(g, {"x": rng.standard_normal(shape)}, tiny, rtol=1e-4, atol=1e-4, modes=("functional", "cycle"))


def test_reduce_int32(tiny, rng):
    g = Graph()
    x = g.input("x", (40, 50), DType.INT32)
    g.output("s", g.sum(x, 0))
    g.output("m", g.max(x, 2))
    check(g, {"x": rng.integers(-1000, 1000, (40, 50))}, tiny)


@pytest.mark.parametrize("fn", ["softmax", "layernorm", "rmsnorm"])
def test_row_normalizations(tiny, rng, fn):
    g = Graph()
    x = g.input("x", (77, 130))
    g.output("y", getattr(g, fn)(x))
    check(g, {"x": rng.standard_normal((77, 130)) * 3}, tiny, rtol=1e-4, atol=1e-5)


def test_transpose(tiny, rng):
    g = Graph()
    x = g.input("x", (70, 150))
    g.output("y", g.transpose(x))
    check(g, {"x": rng.standard_normal((70, 150))}, tiny)


def test_mixed_precision_bf16_matmul(tiny, rng):
    g = Graph()
    x = g.input("x", (32, 48))
    w = g.param("w", rng.standard_normal((48, 16)))
    g.output("y", g.matmul(g.cast(x, DType.BF16), g.cast(w, DType.BF16)))
    X = rng.standard_normal((32, 48))
    exe, out = check(g, {"x": X}, tiny, rtol=1e-4, atol=1e-4)
    fp32 = X @ exe.init["w"]
    assert np.abs(out.outputs["y"] - fp32).max() > 1e-4    # BF16 rounding is really applied
    np.testing.assert_allclose(out.outputs["y"], round_bf16(X.astype(np.float32)).astype(np.float64)
                               @ round_bf16(exe.init["w"].astype(np.float32)), rtol=1e-4, atol=1e-4)


def test_barrier_elimination_is_safe_and_effective(dev, rng):
    g = Graph()
    x = g.input("x", (256, 64))
    a = g.softmax(g.exp(g.scale(x, 0.1)))
    b = g.layernorm(g.add(a, x))
    g.output("y", g.mul(g.sum(b, 1), g.input("z", (256, 1))))
    feeds = {"x": rng.standard_normal((256, 64)), "z": rng.standard_normal((256, 1))}
    on, o1 = check(g, feeds, dev, CompileOptions(barrier_elim=True), modes=("cycle",))
    off, o2 = check(g, feeds, dev, CompileOptions(barrier_elim=False), modes=("cycle",))
    assert on.stats["barriers"] < off.stats["barriers"]
    assert on.stats["barriers_avoided"] > 0
    assert o1.result.cycles < o2.result.cycles


def test_memory_planner_reuses_temporaries(tiny, rng):
    g = Graph()
    x = g.input("x", (64, 64))
    h = x
    for _ in range(8):
        h = g.softmax(g.matmul(h, x))
    g.output("y", h)
    exe, _ = check(g, {"x": rng.standard_normal((64, 64)) * 0.1}, tiny, rtol=1e-4, atol=1e-6)
    temp_total = sum(b.nbytes for b in exe.module.buffers if b.kind == "temp")
    assert exe.stats["global_bytes_peak"] < 64 * 64 * 4 * 2 + temp_total // 2


def test_assign_inplace_and_copy(tiny, rng):
    g = Graph()
    p, v = g.param("p", rng.standard_normal((20, 30))), g.param("v", np.zeros((20, 30)))
    gr = g.input("g", (20, 30))
    v_new = g.add(g.scale(v, 0.9), gr)
    g.assign(v, v_new)
    g.assign(p, g.sub(p, g.scale(v_new, 0.1)))
    g.output("norm", g.sum(g.square(v_new), 2))
    exe, _ = check(g, {"g": rng.standard_normal((20, 30))}, tiny)
    assert exe.stats["inplace_updates"] + exe.stats["kernels_by_kind"].get("copy", 0) == 2


def test_share_buffers_between_executables(tiny, rng):
    g1 = Graph("train")
    w = g1.param("w", rng.standard_normal((8, 8)))
    x = g1.input("x", (4, 8))
    g1.assign(w, g1.scale(w, 2.0))
    g1.output("y", g1.matmul(x, w))
    g2 = Graph("eval")
    w2 = g2.param("w", np.zeros((8, 8)))
    x2 = g2.input("x", (2, 8))
    g2.output("y", g2.matmul(x2, w2))
    e1 = compile_graph(g1, tiny)
    e2 = compile_graph(g2, tiny, share_with=e1)
    assert e2.buffers["w"].addr == e1.buffers["w"].addr
    dev = Device(tiny)
    dev.load(e1)
    dev.load(e2, init=False)
    X = rng.standard_normal((4, 8))
    dev.run(e1, {"x": X})
    W = dev.read(e1, "w")
    np.testing.assert_allclose(W, 2 * w.value, rtol=1e-6)
    out = dev.run(e2, {"x": X[:2]})
    np.testing.assert_allclose(out.outputs["y"], X[:2] @ W, rtol=1e-5)
    np.testing.assert_allclose(dev.read(e1, "w"), W)


def test_compile_errors(tiny):
    g = Graph()
    x = g.input("x", (2, 200000))
    g.output("y", g.softmax(x))
    with pytest.raises(CompileError):
        compile_graph(g, tiny)


def test_generated_program_roundtrips_through_binary(tiny, rng):
    from titan.isa import decode_program, encode_program
    g = _mlp_graph(rng)
    exe = compile_graph(g, tiny)
    back = decode_program(encode_program(exe.program))
    assert back.instructions == exe.program.instructions
    assert back.meta["config_hash"] == tiny.config_hash()
