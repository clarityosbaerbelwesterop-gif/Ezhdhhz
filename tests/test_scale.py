"""Large-model support: meta graphs, planner accounting validated against compiled graphs, fast executor."""
import numpy as np
import pytest

from titan import load_config
from titan.cli import main
from titan.compiler import CompileOptions, Graph, compile_graph, gradients
from titan.dtypes import DType
from titan.nn import MLPClassifier, ParamStore, TinyTransformerLM, transformer_block
from titan.runtime import Device, FastDevice
from titan.scale import (DeviceSpec, TransformerSpec, compare_devices, evaluate_plan, host_training_bytes, plan, preset,
                         spec_for_params)
from titan.train import TrainConfig, Trainer


def _lm_graph(spec: TransformerSpec, batch: int, meta: bool, train: bool):
    g = Graph()
    ps = ParamStore(g, 0, meta=meta)
    m = TinyTransformerLM(spec.vocab, spec.seq, spec.d_model, spec.heads, spec.d_ff, spec.layers)
    _, logits = m.build(g, ps, batch, DType.FP32)
    g.output("logits", logits)
    if train:
        loss = g.xent(logits, g.onehot(g.input("y", (logits.rows, 1), DType.INT32), spec.vocab))
        gradients(g, loss, list(ps.params.values()))
    return g


@pytest.mark.parametrize("L,d,h,V,T", [(1, 16, 2, 11, 8), (2, 32, 4, 7, 12), (3, 64, 2, 50, 16)])
def test_spec_accounting_matches_compiled_graph(L, d, h, V, T):
    spec = TransformerSpec("t", L, d, h, 4 * d, V, T)
    B = 3
    fwd = _lm_graph(spec, B, meta=True, train=False)
    assert fwd.param_count() == spec.params()
    assert fwd.matmul_flops() == B * spec.fwd_flops_per_seq()          # embeddings are gathers, not GEMMs
    trn = _lm_graph(spec, B, meta=True, train=True)
    assert trn.matmul_flops() == 3 * B * spec.fwd_flops_per_seq()      # backward = 2x every matrix product


def test_100b_width_meta_graph_without_memory():
    full = preset("gpt-100b")
    assert 100e9 < full.params() < 103e9
    two = TransformerSpec("two", 2, full.d_model, full.heads, full.d_ff, full.vocab, 128)
    g = _lm_graph(two, 1, meta=True, train=False)
    assert g.param_count() == two.params()
    assert all(t.value is None for t in g.params())
    with pytest.raises(Exception, match="meta"):
        compile_graph(g, load_config("target"))
    ir = compile_graph(g, load_config("target"), CompileOptions(emit_isa=False))   # IR at 100B width
    gemms = [k for k in ir.module.kernels if k.kind == "gemm"]
    # per layer: q, k, v, o, ff1, ff2 + 2 head-batched attention products (fused across heads); + LM head
    assert len(gemms) == 2 * 8 + 1 and "tlm.b0.attn.q.w" not in ir.init
    assert sum(k.attrs["batch"] == full.heads for k in gemms) == 4


def test_spec_for_params():
    assert spec_for_params("100b").name == "gpt-100b"
    s = spec_for_params("30m", vocab=24, seq=64)
    assert s.layers % 2 == 0 and abs(s.params() - 30e6) / 30e6 < 0.15


def test_planner_memory_accounting_and_feasibility():
    spec = preset("gpt-100b")
    tgt = DeviceSpec.from_arch(load_config("target"))
    r = evaluate_plan(spec, tgt, 1, 8, 1, 1, 1, 1, False)
    m = r["memory_gb"]
    assert (m["weights"] + m["grads"] + m["optimizer"]) * 1e9 == pytest.approx(16 * spec.params())
    assert not r["fits"]
    p1, p8, p64 = plan(spec, tgt, 1), plan(spec, tgt, 8), plan(spec, tgt, 64)
    assert not p1["feasible"] and p1["min_devices_for_model_states"] == 3
    assert p8["feasible"] and p64["feasible"]
    assert p64["best"]["time_s"]["step"] < p8["best"]["time_s"]["step"]
    b = p64["best"]
    assert b["dp"] * b["tp"] * b["pp"] == 64 and b["memory_gb"]["total"] <= b["memory_gb"]["capacity_usable"]
    assert "not a measurement" in p64["label"]


def test_device_comparison_is_like_for_like():
    c = compare_devices(preset("gpt-13b"), DeviceSpec.from_arch(load_config("target")), DeviceSpec.b300(), 16, 256)
    assert c["speedup_a_over_b"] > 1 and "same analytical model" in c["label"]
    assert c["a"]["links"] == c["b"]["links"]


def test_host_memory_estimate_grows_with_size():
    assert host_training_bytes(spec_for_params("100m", 24, 64), 8) > host_training_bytes(spec_for_params("10m", 24, 64), 8)


def _graphs(rng):
    g1 = Graph("mlp-train")
    ps = ParamStore(g1, 0)
    _, lg = MLPClassifier(6, 3, [32, 16], "gelu").build(g1, ps, 24, DType.FP32)
    loss = g1.xent(lg, g1.input("y", lg.shape))
    params = list(ps.params.values())
    for p, gr in zip(params, gradients(g1, loss, params)):
        g1.assign(p, g1.sub(p, g1.scale(gr, 0.1)))
    g1.output("loss", loss)
    g2 = Graph("block-bf16")
    ps2 = ParamStore(g2, 1)
    g2.output("y", transformer_block(g2, ps2, g2.input("x", (16, 16)), 2, 8, 2, 32, "b", True, DType.BF16))
    g3 = Graph("int8")
    w = g3.param("w", rng.integers(-100, 100, (24, 12)), DType.INT8)
    g3.output("y", g3.cast(g3.matmul(g3.cast(g3.input("a", (20, 24)), DType.INT8, 20.0), w), DType.FP32, 0.01))
    g4 = Graph("reduce")
    z = g4.input("z", (37, 53))
    g4.output("r", g4.sum(g4.transpose(g4.softmax(z)), 2))
    g4.output("m", g4.max(z, 0))
    g4.output("s", g4.mean(g4.exp(z), 1))
    return [g1, g2, g3, g4]


@pytest.mark.parametrize("idx", range(4))
def test_fast_executor_matches_isa_level(dev, rng, idx):
    g = _graphs(rng)[idx]
    isa = compile_graph(g, dev)
    ir = compile_graph(g, dev, CompileOptions(emit_isa=False))
    assert len(ir.program) == 0
    feeds = {n: rng.standard_normal(isa.ref(n).shape) for n in isa.inputs}
    a, b = Device(dev), FastDevice(dev)
    for _ in range(3):
        oa, ob = a.run(isa, feeds), b.run(ir, feeds)
    tol = 2e-3 if idx == 1 else 1e-4
    for k in oa.outputs:
        np.testing.assert_allclose(ob.outputs[k], oa.outputs[k], rtol=tol, atol=tol, err_msg=k)
    for p in isa.params:
        np.testing.assert_allclose(b.read(ir, p), a.read(isa, p), rtol=1e-4, atol=1e-5)


def test_fast_training_tracks_isa_training(tmp_path):
    runs = {}
    for be in ("functional", "fast"):
        tr = Trainer(TrainConfig(model="transformer", profile="tiny", steps=25, backend=be, eval_every=1000,
                                 ckpt_every=0, out_dir=str(tmp_path / be)), log=None)
        tr.run()
        runs[be] = [h["loss"] for h in tr.history]
    np.testing.assert_allclose(runs["fast"], runs["functional"], rtol=2e-3)


def test_quickstart_params_trains_small_and_plans_100b(tmp_path, capsys):
    assert main(["quickstart", "--params", "1m", "--steps", "10", "--eval-every", "10", "--out", str(tmp_path / "q")]) == 0
    out = capsys.readouterr().out
    assert "training numerically" in out and "eval loss" in out
    assert main(["quickstart", "--params", "100b"]) == 0
    out = capsys.readouterr().out
    assert "--offload-dir" in out and "best plan" in out
    assert main(["plan", "--preset", "gpt-70b", "--devices", "16", "--compare-b300"]) == 0
    assert "TITAN / B300 step-time ratio" in capsys.readouterr().out
