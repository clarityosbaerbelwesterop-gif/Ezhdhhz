"""Insert an implant into a checkpoint made by the team's own upcycle() and compare the team's own forward()
before and after. Run with: python -I check.py <teamcode dir> <repo dir> <work dir>"""
import json
import os
import subprocess
import sys

team, repo, work = sys.argv[1:4]
sys.path.insert(0, team)
sys.path.insert(0, repo)
import numpy as np                                                         # noqa: E402
from accel import moe as MOE, model as M                                   # noqa: E402
from accel.config import ModelConfig                                       # noqa: E402
from titan.donor import st                                                 # noqa: E402

f32 = np.float32
base = ModelConfig.from_dict({"vocab_size": 97, "dim": 48, "n_layers": 4, "n_heads": 6, "n_kv_heads": 2, "ffn_hidden": 64,
                              "max_seq_len": 32, "rope_theta": 10000.0, "norm_eps": 1e-5})
dense = M.init_params(base, seed=3, exact=False)
rng = np.random.default_rng(11)
for n in dense:                                                            # non-trivial norm gains
    if n.endswith("norm.weight"):
        dense[n] = rng.uniform(0.6, 1.4, dense[n].shape).astype(f32)
cfg, params, buffers = MOE.upcycle(dense, base, n_routed=10, n_shared=2, top_k=6, expert_ffn=8, noise=0.05, seed=1)
for n in list(params):                                                     # a trained-looking router: routing differs per token
    if n.endswith("moe.router.weight"):
        params[n] = (rng.standard_normal(params[n].shape) * 2.0).astype(f32)
for n in list(buffers):
    buffers[n] = (rng.standard_normal(buffers[n].shape) * 0.05).astype(f32)
ids = rng.integers(0, base.vocab_size, (3, 24))


def fwd(p, b, c, dispatch):
    lg, loads, _ = MOE.forward({k: np.asarray(v) for k, v in p.items()}, {k: np.asarray(v) for k, v in b.items()}, c, ids, dispatch=dispatch)
    return np.asarray(lg), np.asarray(loads)


def write(d, p, b, c):
    os.makedirs(d, exist_ok=True)
    allp = {**p, **b}
    inv = st.write_safetensors(os.path.join(d, "model.safetensors"), {n: ("F32", a.shape, np.ascontiguousarray(a, "<f4").tobytes()) for n, a in allp.items()})
    opt = {n: np.zeros_like(a) for n, a in p.items()}
    opt.update({"v." + n: np.zeros_like(a) for n, a in p.items()})
    st.write_safetensors(os.path.join(d, "optim.safetensors"), {n: ("F32", a.shape, a.tobytes()) for n, a in opt.items()})
    json.dump({"step": 0, "moe": c.to_dict(), "model_sha256": inv["__file__"]["sha256"], "parent_sha256": None}, open(os.path.join(d, "state.json"), "w"))


def read(d):
    rd = st.LocalReader(os.path.join(d, "model.safetensors"))
    refs, _ = st.tensor_refs(rd, "m")
    w = {n: np.frombuffer(st.fetch(rd, r), "<f4").reshape(r.shape).copy() for n, r in refs.items()}
    c = MOE.MoEConfig.from_dict(json.load(open(os.path.join(d, "state.json")))["moe"])
    want = dict(c.all_specs())
    assert set(w) == set(want), sorted(set(w) ^ set(want))[:5]             # the team's own tensor list for the new config
    for n, s in want.items():
        assert tuple(w[n].shape) == tuple(s), (n, w[n].shape, s)
    buf = {n: w[n] for n, _ in c.buffer_specs()}
    return c, {n: w[n] for n, _ in c.param_specs()}, buf


ck, out = os.path.join(work, "ckpt"), os.path.join(work, "grown")
write(ck, params, buffers, cfg)
# an implant of 7 experts in the receiving layout (random weights, large router rows)
d, fe, N = base.dim, cfg.expert_ffn, 7
slots, man = {}, {"provenance": {"repo": "synthetic", "revision": "-"}, "experts": []}
for j in range(N):
    for k, shp, sc in (("w1", (fe, d), 1.0), ("w3", (fe, d), 1.0), ("w2", (d, fe), 1.0), ("router", (d,), 5.0)):
        slots[f"slot.L{j * 5}.e{j}.{k}"] = (rng.standard_normal(shp) * sc).astype("<f4")
    man["experts"].append({"layer": j * 5, "expert": j})
st.write_safetensors(os.path.join(work, "slots.safetensors"), {n: ("F32", a.shape, a.tobytes()) for n, a in slots.items()})
json.dump(man, open(os.path.join(work, "man.json"), "w"))
run = lambda *a: subprocess.run([sys.executable, *a], check=True, capture_output=True, text=True)
run(os.path.join(repo, "tools/donor/build_implant.py"), "--projection", os.path.join(work, "slots.safetensors"), "--manifest", os.path.join(work, "man.json"),
    "--donor-layers", "48", "--prefix", "imp", "--out", os.path.join(work, "imp"))
run(os.path.join(repo, "tools/donor/insert_implant.py"), "--checkpoint", ck, "--implant", os.path.join(work, "imp", "imp_all.safetensors"),
    "--layers", "1,2,3", "--out", out)
c0, p0, b0 = read(ck)
c1, p1, b1 = read(out)
res = {"n_routed": [c0.n_routed, c1.n_routed], "params": [c0.num_params(), c1.num_params()], "active_params": [c0.active_params(), c1.active_params()]}
for disp in MOE.DISPATCH:
    try:
        l0, load0 = fwd(p0, b0, c0, disp)
        l1, load1 = fwd(p1, b1, c1, disp)
    except Exception as ex:                                                # e.g. a dispatch path this JAX build lacks
        res[disp] = f"not run: {type(ex).__name__}: {str(ex)[:80]}"
        continue
    res[disp] = {"max_abs_logit_diff": float(np.abs(l0 - l1).max()), "logit_abs_max": float(np.abs(l0).max()),
                 "old_slot_loads_equal": bool((load0 == load1[:, :c0.n_routed]).all()), "new_slot_load": int(load1[:, c0.n_routed:].sum()),
                 "distinct_experts_used": int((load0 > 0).sum())}
# switching the new slots on changes the output (they are live, not dead weight)
b_on = {n: np.concatenate([a[:c0.n_routed], np.full(c1.n_routed - c0.n_routed, 4.0, f32)]) for n, a in b1.items()}
l_on, load_on = fwd(p1, b_on, c1, "dense")
res["switched_on"] = {"max_abs_logit_diff": float(np.abs(l_on - fwd(p1, b1, c1, "dense")[0]).max()), "new_slot_load": int(load_on[:, c0.n_routed:].sum())}
print(json.dumps(res, indent=1))
