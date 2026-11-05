"""Mixture of experts, SFT loss masking, KV-cached generation, test-time compute, RFT and model merging."""
import json
import random

import numpy as np
import pytest

from titan import load_config
from titan.cli import main
from titan.compiler import CompileOptions, Graph, compile_graph, evaluate, gradients, run_reference
from titan.dtypes import DType
from titan.infer import Generator, evaluate_problems, extract_answer, generator_for_run, sample, solve
from titan.merge import average, slerp, stack, task_arithmetic, ties, upcycle_moe, write_run
from titan.nn import ParamStore, TransformerConfig, TransformerLM
from titan.runtime import Device, FastDevice
from titan.scale import TransformerSpec
from titan.train import TrainConfig, Trainer

MOE = {"experts": 4, "top_k": 2, "shared_expert": True}


def _torch():
    from titan.runtime.accel import torch_available
    return torch_available()


def _moe_graph(dense, cf=1.25, seed=3, **kw):
    cfg = TransformerConfig(**{"vocab": 13, "seq": 6, "d_model": 16, "layers": 2, "heads": 2, "capacity_factor": cf,
                               **MOE, **kw})
    g = Graph()
    ps = ParamStore(g, seed)
    ps.dense_moe = dense
    _, logits = TransformerLM(cfg).build(g, ps, 2, DType.FP32)
    g.output("logits", logits)
    return g


def test_moe_sparse_equals_dense_and_all_executors_match(dev, rng):
    feeds = {"x": rng.integers(0, 13, (12, 1))}
    dense = run_reference(_moe_graph(True), feeds)[0]["logits"]
    nodrop = run_reference(_moe_graph(False, cf=100.0), feeds)[0]["logits"]
    np.testing.assert_array_equal(dense, nodrop)                       # same math without capacity drops
    dropped = run_reference(_moe_graph(False, cf=0.5), feeds)[0]["logits"]
    assert np.abs(dense - dropped).max() > 1e-3                        # capacity limits drop tokens
    g = _moe_graph(True)
    isa = Device(dev).run(compile_graph(g, dev), feeds, "functional").outputs["logits"]
    np.testing.assert_allclose(isa, dense, rtol=1e-4, atol=1e-5)
    g = _moe_graph(False)
    ref = run_reference(g, feeds)[0]["logits"]
    ir = compile_graph(g, dev, CompileOptions(emit_isa=False))
    np.testing.assert_allclose(FastDevice(dev).run(ir, feeds).outputs["logits"], ref, rtol=1e-4, atol=1e-5)
    if _torch():
        from titan.runtime.accel import AccelDevice
        np.testing.assert_allclose(AccelDevice(dev).run(ir, feeds).outputs["logits"], ref, rtol=1e-4, atol=1e-5)
    with pytest.raises(Exception, match="moe_slots"):
        compile_graph(g, dev)                                          # sparse routing has no ISA lowering


@pytest.mark.parametrize("dense", [True, False])
def test_moe_gradients_finite_differences(dense):
    g = _moe_graph(dense, cf=0.75, seed=5, layers=1, moe_aux=0.1)
    logits = g.outputs["logits"]
    w = g.const(np.random.default_rng(0).standard_normal(logits.shape))
    loss = g.sum(g.mul(logits, w), 2)
    for coef, aux in g.aux_losses:
        loss = g.add(loss, g.scale(aux, coef))
    ps = [t for t in g.params() if ".moe." in t.name]
    grads = gradients(g, loss, ps)
    feeds = {"x": np.arange(12).reshape(-1, 1) % 13}
    env = evaluate(g, feeds)
    for p, gr in zip(ps, grads):
        idx = tuple(np.unravel_index(np.argmax(np.abs(env[gr.id])), p.shape))
        vals = {}
        for sgn in (1, -1):
            v = p.value.astype(np.float64)                   # FP64 perturbation (FP32 would round it)
            v[idx] += sgn * 1e-6
            vals[sgn] = evaluate(g, feeds, {p.name: v})[loss.id][0, 0]
        num = (vals[1] - vals[-1]) / 2e-6
        np.testing.assert_allclose(env[gr.id][idx], num, rtol=1e-4, atol=1e-6, err_msg=p.name)


def test_moe_accounting_and_training(tmp_path):
    for kw in (MOE, {"experts": 8, "top_k": 1, "moe_every": 2, "activation": "swiglu", "bias": False}):
        cfg = TransformerConfig(vocab=29, seq=8, d_model=32, layers=4, heads=4, **kw)
        for dense in (True, False):
            g = Graph()
            ps = ParamStore(g, 0, meta=True)
            ps.dense_moe = dense
            TransformerLM(cfg).build(g, ps, 3, DType.FP32)
            spec = TransformerSpec.from_config(cfg)
            assert g.param_count() == spec.params() and g.matmul_flops() == spec.fwd_flops(3, dense)
        assert spec.active_params() < spec.params()
    losses = {}
    for backend in ["functional", "fast"] + (["accel"] if _torch() else []):
        tr = Trainer(TrainConfig(model="transformer", profile="tiny", backend=backend, steps=15, eval_every=15,
                                 ckpt_every=0, overrides=dict(MOE), out_dir=str(tmp_path / backend)), log=None)
        s = tr.run()
        tr.close()
        assert "aux_loss" in tr.train_exe.outputs and s["eval_loss"] < s["initial_loss"]
        losses[backend] = [h["loss"] for h in tr.history]
    # the ISA path uses dense dispatch (no capacity drops), so only the IR executors are compared exactly
    if "accel" in losses:
        np.testing.assert_allclose(losses["accel"], losses["fast"], rtol=2e-4)


# ------------------------------------------------------------------ SFT

def test_weighted_xent_gradient_ignores_masked_rows(rng):
    g = Graph()
    z = g.param("z", rng.standard_normal((5, 4)))
    y = g.onehot(g.const(np.array([[1], [-1], [3], [-1], [0]]), DType.INT32), 4)
    loss = g.xent(z, g.mul(y, g.const([[5 / 3]])), weighted=True)
    (gz,) = gradients(g, loss, [z])
    env = evaluate(g, {})
    assert np.abs(env[gz.id][[1, 3]]).max() == 0                       # prompt rows: no gradient
    num = np.zeros(z.shape)
    for idx in np.ndindex(z.shape):
        for s in (1, -1):
            v = z.value.copy()
            v[idx] += s * 1e-6
            num[idx] += s * evaluate(g, {}, {"z": v})[loss.id][0, 0] / 2e-6
    np.testing.assert_allclose(env[gz.id], num, rtol=1e-5, atol=1e-7)


def _arith(tmp_path, n=400):
    """Single-digit addition (100 facts, repeated): learnable in seconds, enough to test the machinery."""
    rnd = random.Random(0)
    facts = [(a, b) for a in range(10) for b in range(10)]
    rows = []
    while len(rows) < n:
        rnd.shuffle(facts)
        rows += [{"prompt": f"{a}+{b}=", "response": str(a + b)} for a, b in facts]
    rows = rows[:n]
    p = tmp_path / "add.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p, rows


@pytest.fixture(scope="module")
def sft_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sft")
    data, rows = _arith(tmp, 600)
    tr = Trainer(TrainConfig(data=str(data), backend="fast", steps=500, batch=32, eval_every=500, ckpt_every=500,
                             lr=3e-3, overrides={"seq": 16, "d_model": 48, "heads": 4, "layers": 2},
                             out_dir=str(tmp / "run")), log=None)
    s = tr.run()
    tr.close()
    return tr, s, rows


def test_sft_masks_prompts_and_learns(sft_run):
    tr, s, _ = sft_run
    ds = tr.ds
    assert ds.sft["trained_tokens"] < ds.sft["train_tokens"]
    f = tr.sampler.next()
    x, y = f["x"], f["y"]
    assert (y == -1).any() and (y >= 0).any() and x.shape == y.shape
    assert s["eval_loss"] < 0.5 * s["initial_loss"]
    tok = json.loads((tr.out / "tokenizer.json").read_text())
    assert tok["kind"] == "char" and "+" in tok["chars"]


# ------------------------------------------------------------------ generation

@pytest.mark.parametrize("kw", [{}, {"norm": "rmsnorm", "activation": "swiglu", "position": "rope", "heads": 4,
                                    "kv_heads": 2, "bias": False, "tie_embeddings": True},
                                {"position": "sinusoidal", "prenorm": False}, {**MOE, "capacity_factor": 100.0}])
def test_kv_cache_decode_equals_full_forward(kw):
    m = TransformerLM(TransformerConfig(vocab=17, seq=10, d_model=16, layers=2, **{"heads": 2, **kw}))
    g = Graph()
    ps = ParamStore(g, 5, lazy=True)
    _, lg = m.build(g, ps, 2, DType.FP32)
    g.output("logits", lg)
    exe = compile_graph(g, load_config("host"), CompileOptions(emit_isa=False))
    w = {n: f() for n, f in exe.lazy_init.items() if n in exe.params}
    fd = FastDevice("host")
    fd.load(exe)
    ids = np.random.default_rng(1).integers(0, 17, (2, 10))
    full = fd.run(exe, {"x": ids.reshape(-1, 1)}).outputs["logits"].reshape(2, 10, -1)
    for be in ["fast"] + (["accel"] if _torch() else []):
        for cache in (True, False):
            gen = Generator(m, w, batch=2, backend=be, use_cache=cache)
            inc = np.stack([gen.step(ids[:, t], t) for t in range(10)], 1)
            np.testing.assert_allclose(inc, full, rtol=1e-4, atol=2e-5)


def test_sampling_rules():
    rng = np.random.default_rng(0)
    z = np.array([[0.0, 3.0, 1.0, -2.0]] * 50)
    assert (sample(z, 0.0, 0, 1.0, rng) == 1).all()
    assert (sample(z, 1.0, 1, 1.0, rng) == 1).all()                        # top-k 1 = greedy
    assert (sample(z, 1.0, 0, 0.05, rng) == 1).all()                       # tiny nucleus = greedy
    s = sample(z, 1.0, 2, 1.0, rng)
    assert set(s) <= {1, 2} and len(set(s)) == 2
    assert extract_answer("7+8=15,4+3+1=8:85") == "85" and extract_answer("no digits") is None
    assert extract_answer("x=3; answer: 42.", r"answer:\s*(\d+)") == "42"


def test_generation_strategies_and_rft(sft_run, tmp_path):
    tr, _, rows = sft_run
    gen = generator_for_run(tr.out, batch=8)
    assert gen.cached
    probs = [{"prompt": r["prompt"], "answer": r["response"]} for r in rows[-40:]]
    greedy = evaluate_problems(gen, probs, "greedy", stop=("\n",), max_new=4)
    again = evaluate_problems(gen, probs, "greedy", stop=("\n",), max_new=4)
    assert greedy["correct"] == again["correct"] and greedy["accuracy"] > 0.5
    sc = evaluate_problems(gen, probs[:8], "self-consistency", 4, stop=("\n",), max_new=4)
    assert all(sum(r["votes"].values()) <= 4 for r in sc["results"])
    oracle = {r["prompt"]: r["answer"] for r in probs}
    bon = solve(gen, [p["prompt"] for p in probs[:8]], "best-of-n", 4, max_new=4, stop=("\n",),
                verifier=lambda p, c: float(c.strip() == oracle[p]))
    assert sum(r["answer"] == oracle[p["prompt"]] for r, p in zip(bon, probs[:8])) >= \
        sum(greedy["correct"][:8])                                        # an oracle verifier never loses
    beam = gen.beam_search(probs[0]["prompt"], width=4, max_new=4, stop=("\n",))
    assert beam and beam[0]["tokens"] >= 1
    rep = __import__("titan.infer.rft", fromlist=["x"]).rejection_finetune(
        tr.out, probs[:30], probs[30:], rounds=1, samples=2, steps=20, out=tmp_path / "rft", max_new=4, batch=16,
        train_batch=16, log=None)
    r = rep["rounds"][0]
    assert r["kept"] > 0 and r["heldout_greedy_after"] is not None
    assert (tmp_path / "rft" / "round1" / "checkpoints").exists()


# ------------------------------------------------------------------ merging

def _forward(cfg, weights, ids):
    g = Graph()
    ps = ParamStore(g, 0, lazy=True)
    m = TransformerLM(TransformerConfig.from_dict(cfg))
    m._no_drop = True
    _, lg = m.build(g, ps, ids.shape[0], DType.FP32)
    g.output("logits", lg)
    exe = compile_graph(g, load_config("host"), CompileOptions(emit_isa=False))
    for n in exe.params:
        exe.lazy_init[n] = (lambda v=weights[n]: v)
    fd = FastDevice("host")
    fd.load(exe)
    return fd.run(exe, {"x": ids.reshape(-1, 1)}).outputs["logits"]


def test_merge_methods(sft_run, tmp_path):
    from titan.merge import load_source
    tr, _, _ = sft_run
    w, cfg, tok = load_source(tr.out)
    rng = np.random.default_rng(0)
    w2 = {k: v + 0.01 * rng.standard_normal(v.shape).astype(np.float32) for k, v in w.items()}
    ids = np.random.default_rng(2).integers(0, cfg["vocab"], (2, cfg["seq"]))
    base = _forward(cfg, w, ids)
    for merged in (average([w, w]), slerp(w, w2, 0.0), task_arithmetic(w, [w2], 0.0), ties(w, [w], 1.0, 1.0)):
        np.testing.assert_allclose(_forward(cfg, merged, ids), base, rtol=1e-5, atol=1e-5)
    mid = average([w, w2])
    assert all(np.allclose(mid[k], (w[k] + w2[k]) / 2, atol=1e-6) for k in w)
    same, cfg_same = stack([w], [cfg], [(0, 0, cfg["layers"] - 1)])
    np.testing.assert_allclose(_forward(cfg_same, same, ids), base, rtol=1e-6, atol=1e-6)
    deep, cfg_deep = stack([w, w2], [cfg, cfg], [(0, 0, 1), (1, 0, 1)])
    assert cfg_deep["layers"] == 4 and "tlm.b3.ff1.w" in deep
    moe, cfg_moe = upcycle_moe(w, cfg, experts=4, top_k=2)
    np.testing.assert_allclose(_forward(cfg_moe, moe, ids), base, rtol=1e-4, atol=1e-4)   # function preserved
    out = write_run(tmp_path / "frank", deep, cfg_deep, tok, {"method": "stack"})
    gen = generator_for_run(out, batch=2)
    assert gen.generate(["1+2="], max_new=3, temperature=0)[0]["tokens"] >= 1
    tr2 = Trainer(TrainConfig(data=tr.tc.data, tokenizer=str(out / "tokenizer.json"), model_config=str(out / "model.json"),
                              init_from=str(out / "checkpoints" / "best.npz"), backend="fast", steps=3, eval_every=0,
                              ckpt_every=0, out_dir=str(tmp_path / "cont")), log=None)
    assert tr2.init_info["loaded"] == len(tr2.model_params) and tr2.model.config.layers == 4
    tr2.run()
    tr2.close()


def test_cli_generate_reason_merge(sft_run, tmp_path, capsys):
    tr, _, rows = sft_run
    assert main(["generate", str(tr.out), "--prompt", "12+30=", "--strategy", "self-consistency", "--n", "4",
                 "--max-new", "4"]) == 0
    out = capsys.readouterr().out
    assert "votes=" in out and "KV cache" in out
    probs = tmp_path / "p.jsonl"
    probs.write_text("\n".join(json.dumps({"prompt": r["prompt"], "answer": r["response"]}) for r in rows[:20]))
    assert main(["reason", str(tr.out), "--data", str(probs), "--strategy", "greedy", "beam", "--n", "2",
                 "--max-new", "4", "--out", str(tmp_path / "r.json")]) == 0
    assert len(json.loads((tmp_path / "r.json").read_text())) == 2
    assert main(["merge", "upcycle-moe", "--models", str(tr.out), "--experts", "2", "--top-k", "1",
                 "--out", str(tmp_path / "moe")]) == 0
    assert json.loads((tmp_path / "moe" / "model.json").read_text())["experts"] == 2
