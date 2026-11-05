"""M6: neural-network forward passes on TITAN vs. NumPy float64 reference; autodiff correctness."""
import numpy as np
import pytest

from titan.compiler import Graph, compile_graph, evaluate, gradients, run_reference
from titan.dtypes import DType
from titan.nn import MLPClassifier, ParamStore, TinyTransformerLM, attention, transformer_block
from titan.runtime import Device


def run_titan(g, feeds, cfg, mode="functional"):
    exe = compile_graph(g, cfg)
    return Device(cfg).run(exe, feeds, mode).outputs, exe


def test_mlp_forward(dev, rng):
    g = Graph()
    ps = ParamStore(g, 1)
    x, y = MLPClassifier(10, 4, [32, 16], "gelu").build(g, ps, 50, DType.FP32)
    g.output("y", y)
    feeds = {"x": rng.standard_normal((50, 10))}
    out, _ = run_titan(g, feeds, dev, "cycle")
    ref, _ = run_reference(g, feeds)
    np.testing.assert_allclose(out["y"], ref["y"], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("causal", [True, False])
def test_attention_matches_reference_and_textbook(dev, rng, causal):
    b, t, d, h = 2, 8, 16, 2
    g = Graph()
    ps = ParamStore(g, 3)
    x = g.input("x", (b * t, d))
    g.output("y", attention(g, ps, x, b, t, h, "att", causal))
    X = rng.standard_normal((b * t, d))
    out, _ = run_titan(g, {"x": X}, dev)
    ref, _ = run_reference(g, {"x": X})
    np.testing.assert_allclose(out["y"], ref["y"], rtol=1e-5, atol=1e-5)
    # independent textbook implementation with fused Wq/Wk/Wv/Wo
    P = {k: t_.value for k, t_ in ps.params.items()}
    Wq = np.hstack([P[f"att.q{i}"] for i in range(h)])
    Wk = np.hstack([P[f"att.k{i}"] for i in range(h)])
    Wv = np.hstack([P[f"att.v{i}"] for i in range(h)])
    Wo = np.vstack([P[f"att.o{i}"] for i in range(h)])
    outs = []
    for s in range(b):
        xs = X[s * t:(s + 1) * t]
        heads = []
        for i in range(h):
            sl = slice(i * d // h, (i + 1) * d // h)
            sc = (xs @ Wq[:, sl]) @ (xs @ Wk[:, sl]).T / np.sqrt(d // h)
            if causal:
                sc = sc + np.triu(np.full((t, t), -1e9), 1)
            p = np.exp(sc - sc.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            heads.append(p @ (xs @ Wv[:, sl]))
        outs.append(np.hstack(heads) @ Wo + P["att.ob"])
    np.testing.assert_allclose(out["y"], np.vstack(outs), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("compute", [DType.FP32, DType.BF16])
def test_transformer_block(dev, rng, compute):
    g = Graph()
    ps = ParamStore(g, 5)
    x = g.input("x", (2 * 8, 16))
    g.output("y", transformer_block(g, ps, x, 2, 8, 2, 32, "blk", True, compute))
    X = rng.standard_normal((16, 16))
    out, exe = run_titan(g, {"x": X}, dev, "cycle")
    ref, _ = run_reference(g, {"x": X})
    tol = 1e-5 if compute == DType.FP32 else 2e-3
    np.testing.assert_allclose(out["y"], ref["y"], rtol=tol, atol=tol)
    assert exe.stats["fused_nodes"] > 0


def test_tiny_lm_forward(tiny, rng):
    g = Graph()
    ps = ParamStore(g, 0)
    m = TinyTransformerLM(vocab=12, seq=6, d_model=16, heads=2, d_ff=32, layers=2)
    x, logits = m.build(g, ps, 3, DType.FP32)
    g.output("logits", logits)
    X = rng.integers(0, 12, (18, 1))                     # token ids; embedding = row gather (ISA: one-hot GEMM)
    out, _ = run_titan(g, {"x": X}, tiny)
    ref, _ = run_reference(g, {"x": X})
    np.testing.assert_allclose(out["logits"], ref["logits"], rtol=1e-4, atol=1e-5)


# ------------------------------------------------------------------ autodiff

UNARY_CASES = ["relu", "gelu", "sigmoid", "tanh", "exp", "square", "softmax", "layernorm", "rmsnorm", "transpose"]


def _fd_check(build, shapes, rng, eps=1e-6, positive=False, tol=1e-5):
    """Compare autodiff gradients with central finite differences, both on the float64 reference."""
    g = Graph()
    ps = [g.param(f"p{i}", (np.abs(rng.standard_normal(s)) + 0.5) if positive else rng.standard_normal(s))
          for i, s in enumerate(shapes)]
    out = build(g, *ps)
    wvals = rng.standard_normal(out.shape)
    w = g.const(wvals)
    loss = g.sum(g.mul(out, w), 2)
    grads = gradients(g, loss, ps)
    for i, gi in enumerate(grads):
        g.output(f"g{i}", gi)
    g.output("loss", loss)
    env = evaluate(g, {})
    for p, gr in zip(ps, grads):
        num = np.zeros(p.shape)
        base = {q.name: q.value.copy() for q in ps}
        for idx in np.ndindex(p.shape):
            for sgn in (1, -1):
                vals = {k: v.copy() for k, v in base.items()}
                vals[p.name][idx] += sgn * eps
                e2 = evaluate(g, {}, vals)
                num[idx] += sgn * e2[loss.id][0, 0] / (2 * eps)
        np.testing.assert_allclose(env[gr.id], num, rtol=tol, atol=tol, err_msg=p.name)
    return g, grads


@pytest.mark.parametrize("op", UNARY_CASES)
def test_gradients_unary(rng, op):
    _fd_check(lambda g, a: getattr(g, op)(a), [(3, 5)], rng, tol=1e-4)


@pytest.mark.parametrize("op", ["log", "sqrt", "rsqrt", "recip"])
def test_gradients_positive_domain(rng, op):
    _fd_check(lambda g, a: getattr(g, op)(a), [(3, 4)], rng, positive=True, tol=1e-4)


@pytest.mark.parametrize("op", ["add", "sub", "mul", "div", "maximum"])
@pytest.mark.parametrize("bshape", [(4, 5), (1, 5), (4, 1), (1, 1)])
def test_gradients_binary_broadcast(rng, op, bshape):
    _fd_check(lambda g, a, b: getattr(g, op)(a, b), [(4, 5), bshape], rng, positive=(op == "div"), tol=1e-4)


@pytest.mark.parametrize("ta,tb", [(False, False), (True, False), (False, True), (True, True)])
def test_gradients_matmul(rng, ta, tb):
    _fd_check(lambda g, a, b: g.matmul(a, b, ta, tb), [(4, 3) if ta else (3, 4), (5, 4) if tb else (4, 5)], rng)


def test_gradients_bmm_reduce_scale_xent(rng):
    _fd_check(lambda g, a, b: g.bmm(a, b, 2, tb=True), [(6, 4), (8, 4)], rng)
    _fd_check(lambda g, a: g.scale(g.mean(a, 0), 3.0, 1.0), [(4, 3)], rng)
    _fd_check(lambda g, a: g.broadcast(g.sum(a, 1), (3, 6)), [(3, 6)], rng)
    _fd_check(lambda g, a: g.xent(a, g.const(np.eye(4)[[0, 2, 1]])), [(3, 4)], rng)


def test_backward_executes_on_titan(dev, rng):
    """Forward + backward of a 2-layer MLP compiled to TITAN ISA equals the reference gradients."""
    g = Graph()
    ps = ParamStore(g, 7)
    x, logits = MLPClassifier(6, 3, [16], "tanh").build(g, ps, 20, DType.FP32)
    y = g.input("y", (20, 3))
    loss = g.xent(logits, y)
    params = list(ps.params.values())
    for p, gr in zip(params, gradients(g, loss, params)):
        g.output("grad:" + p.name, gr)
    g.output("loss", loss)
    feeds = {"x": rng.standard_normal((20, 6)), "y": np.eye(3)[rng.integers(0, 3, 20)]}
    out, exe = run_titan(g, feeds, dev, "cycle")
    ref, _ = run_reference(g, feeds)
    for k in ref:
        np.testing.assert_allclose(out[k], ref[k], rtol=1e-4, atol=1e-6, err_msg=k)
    assert set(i.opcode.name for i in exe.program.instructions) >= {"MATMUL", "LOAD", "STORE", "REDUCE"}
