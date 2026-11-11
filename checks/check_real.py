"""Rehearsal at the seed's real width with the real implants and the team's own code.
A 2-layer model (dim 1920, FFN 5120, GQA 30/10, vocab 8192) is initialised with the team's init_params, gets the
real seed tensors that exist (tok_emb, norm, attn_norm/ffn_norm gains of blocks 11 and 23), is upcycled with the
team's upcycle() to their MoE defaults (fe 640, 2 shared, top-6) and receives all four implants. The team's
forward() is compared before and after. Attention and FFN weights are random: this checks format and identity,
not model quality.  python -I check_real.py <teamcode> <repo> <work> <subset.safetensors> <implant>..."""
import json
import os
import subprocess
import sys

team, repo, work, subset = sys.argv[1:5]
implants = sys.argv[5:]
sys.path.insert(0, team)
sys.path.insert(0, repo)
import numpy as np                                                         # noqa: E402
from accel import moe as MOE, model as M                                   # noqa: E402
from accel.config import ModelConfig                                       # noqa: E402
from titan.donor import st                                                 # noqa: E402

f32 = np.float32
base = ModelConfig.from_dict({"vocab_size": 8192, "dim": 1920, "n_layers": 2, "n_heads": 30, "n_kv_heads": 10, "ffn_hidden": 5120,
                              "max_seq_len": 512, "rope_theta": 10000.0, "norm_eps": 1e-5})
dense = M.init_params(base, seed=3, exact=False)
rd = st.LocalReader(subset)
refs, _ = st.tensor_refs(rd, "subset")
real = lambda n: np.frombuffer(st.fetch(rd, refs[n]), "<f4").reshape(refs[n].shape).copy()
dense["tok_emb.weight"], dense["norm.weight"] = real("tok_emb.weight"), real("norm.weight")
for i, src in ((0, 11), (1, 23)):
    dense[f"blocks.{i}.attn_norm.weight"] = real(f"blocks.{src}.attn_norm.weight")
    dense[f"blocks.{i}.ffn_norm.weight"] = real(f"blocks.{src}.ffn_norm.weight")
rng = np.random.default_rng(11)
ids = rng.integers(0, base.vocab_size, (2, 24))
run = lambda *a: subprocess.run([sys.executable, *a], check=True, capture_output=True, text=True)


def fwd(p, b, c):
    lg, loads, _ = MOE.forward(p, b, c, ids, dispatch="dense")
    return np.asarray(lg), np.asarray(loads)


def write(d, p, b, c):
    os.makedirs(d, exist_ok=True)
    allp = {**p, **b}
    inv = st.write_safetensors(os.path.join(d, "model.safetensors"), {n: ("F32", a.shape, np.ascontiguousarray(a, "<f4").tobytes()) for n, a in allp.items()})
    json.dump({"step": 0, "moe": c.to_dict(), "model_sha256": inv["__file__"]["sha256"], "parent_sha256": None}, open(os.path.join(d, "state.json"), "w"))


def read(d):
    r = st.LocalReader(os.path.join(d, "model.safetensors"))
    rf, _ = st.tensor_refs(r, "m")
    w = {n: np.frombuffer(st.fetch(r, x), "<f4").reshape(x.shape).copy() for n, x in rf.items()}
    c = MOE.MoEConfig.from_dict(json.load(open(os.path.join(d, "state.json")))["moe"])
    want = dict(c.all_specs())
    assert set(w) == set(want) and all(tuple(w[n].shape) == tuple(s) for n, s in want.items())
    return c, {n: w[n] for n, _ in c.param_specs()}, {n: w[n] for n, _ in c.buffer_specs()}


out = {}
for case in ("zero_router_as_after_upcycle", "random_router"):
    cfg, params, buffers = MOE.upcycle(dense, base, n_routed=8, n_shared=2, top_k=6, expert_ffn=640, noise=0.05, seed=1)
    if case == "random_router":
        for n in list(params):
            if n.endswith("moe.router.weight"):
                params[n] = (rng.standard_normal(params[n].shape) * 0.05).astype(f32)
        for n in list(buffers):
            buffers[n] = (rng.standard_normal(buffers[n].shape) * 0.05).astype(f32)
    ck, gr = os.path.join(work, "real_ckpt"), os.path.join(work, "real_grown")
    write(ck, params, buffers, cfg)
    run(os.path.join(repo, "tools/donor/insert_implant.py"), "--checkpoint", ck, "--implant", *implants, "--layers", "0,1", "--out", gr)
    c0, p0, b0 = read(ck)
    c1, p1, b1 = read(gr)
    l0, load0 = fwd(p0, b0, c0)
    l1, load1 = fwd(p1, b1, c1)
    rep = json.load(open(os.path.join(gr, "implant_report.json")))
    out[case] = {"n_routed": [c0.n_routed, c1.n_routed], "params": [c0.num_params(), c1.num_params()],
                 "max_abs_logit_diff": float(np.abs(l0 - l1).max()), "logit_abs_max": float(np.abs(l0).max()), "logits_finite": bool(np.isfinite(l1).all()),
                 "old_slot_loads_equal": bool((load0 == load1[:, :c0.n_routed]).all()), "new_slot_load": int(load1[:, c0.n_routed:].sum()),
                 "rows_per_layer": {k: len(v) for k, v in rep["placement"].items()}, "filler_slots": rep["filler_slots"],
                 "parameters_added": rep["parameters_added"]}
    if case == "random_router":                                           # what the new slots do when switched on
        b_on = {n: np.concatenate([a[:c0.n_routed], np.full(c1.n_routed - c0.n_routed, 4.0, f32)]) for n, a in b1.items()}
        l_on, load_on = fwd(p1, b_on, c1)
        out[case]["switched_on"] = {"finite": bool(np.isfinite(l_on).all()), "max_abs_logit_diff": float(np.abs(l_on - l1).max()),
                                    "new_slot_load": int(load_on[:, c0.n_routed:].sum()), "slots_with_load": int((load_on[:, c0.n_routed:] > 0).sum())}
        out[case]["placement_layer_1"] = [f"L{p['donor_layer']} e{p['donor_expert']} {p['functions'][0]} ({p['repo'].split('/')[-1][:14]})" for p in rep["placement"]["1"]][:6]
print(json.dumps(out, indent=1))
