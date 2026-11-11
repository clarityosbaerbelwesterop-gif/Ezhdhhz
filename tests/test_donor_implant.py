"""Implant files in the receiving MoE layout and their insertion into a forge-layout checkpoint."""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from titan.donor import adapt, st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L, D, FE, E, S, K, V = 3, 12, 5, 6, 1, 3, 17


def tool(name, *args, ok=True):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "donor", name), *map(str, args)], capture_output=True, text=True)
    assert (r.returncode == 0) == ok, r.stdout + r.stderr
    return r


def specs(e):
    """The receiving format: one number of routed experts for every MoE layer."""
    out = [("tok_emb.weight", (V, D)), ("norm.weight", (D,))]
    for i in range(L):
        p = f"blocks.{i}"
        out += [(f"{p}.attn_norm.weight", (D,)), (f"{p}.attn.wq.weight", (D, D)), (f"{p}.ffn_norm.weight", (D,)),
                (f"{p}.moe.shared.w1", (S * FE, D)), (f"{p}.moe.shared.w3", (S * FE, D)), (f"{p}.moe.shared.w2", (D, S * FE)),
                (f"{p}.moe.experts.w1", (e, FE, D)), (f"{p}.moe.experts.w3", (e, FE, D)), (f"{p}.moe.experts.w2", (e, D, FE)),
                (f"{p}.moe.router.weight", (e, D)), (f"{p}.moe.router.bias", (e,))]
    return out


def read_all(path):
    rd = st.LocalReader(path)
    refs, _ = st.tensor_refs(rd, "x")
    return {n: np.frombuffer(st.fetch(rd, r), "<f4").reshape(r.shape) for n, r in refs.items()}


def sha_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


@pytest.fixture()
def world(tmp_path):
    rng = np.random.default_rng(5)
    ck = tmp_path / "ckpt"
    ck.mkdir()
    w = {}
    for name, shape in specs(E):
        a = rng.standard_normal(shape).astype("<f4")
        if name.endswith("norm.weight"):
            a = rng.uniform(0.5, 1.5, shape).astype("<f4")
        if name.endswith("router.bias"):
            a = (0.01 * rng.standard_normal(shape)).astype("<f4")
        w[name] = a
    inv = st.write_safetensors(ck / "model.safetensors", {n: ("F32", a.shape, a.tobytes()) for n, a in w.items()}, {"step": 7})
    opt = {}
    for n, a in w.items():
        if not n.endswith("router.bias"):
            opt[n] = rng.standard_normal(a.shape).astype("<f4")
            opt["v." + n] = rng.random(a.shape).astype("<f4")
    st.write_safetensors(ck / "optim.safetensors", {n: ("F32", a.shape, a.tobytes()) for n, a in opt.items()})
    state = {"step": 7, "model_sha256": inv["__file__"]["sha256"], "parent_sha256": None,
             "moe": {"n_routed": E, "n_shared": S, "top_k": K, "expert_ffn": FE, "route_scale": float(K), "moe_layers": [], "base": {"dim": D}}}
    (ck / "state.json").write_text(json.dumps(state))
    # five projected donor experts, two functions
    picks = [(40, 3, "fnA"), (10, 9, "fnA"), (25, 1, "fnB"), (47, 0, "fnB"), (30, 7, "fnA")]
    slots, man = {}, {"provenance": {"repo": "x/y", "revision": "r"}, "bridges": {}, "experts": []}
    for (dl, de, fn) in picks:
        t = {"w1": rng.standard_normal((FE, D)), "w3": rng.standard_normal((FE, D)), "w2": rng.standard_normal((D, FE)),
             "router": 30 * rng.standard_normal(D)}
        for k, a in t.items():
            slots[f"slot.L{dl}.e{de}.{k}"] = a.astype("<f4")
        man["experts"].append({"layer": dl, "expert": de, "selected_for": [{"function": fn, "rank": 1}]})
    st.write_safetensors(tmp_path / "slots.safetensors", {n: ("F32", a.shape, a.tobytes()) for n, a in slots.items()})
    (tmp_path / "manifest.json").write_text(json.dumps(man))
    tool("build_implant.py", "--projection", tmp_path / "slots.safetensors", "--manifest", tmp_path / "manifest.json", "--donor-layers", 48,
         "--group", "a=fnA", "--group", "b=fnB", "--prefix", "imp", "--out", tmp_path / "imp")
    return tmp_path, w, opt, slots, picks


def test_implant_file_has_the_receiving_layout_and_is_a_byte_copy(world):
    tmp, _, _, slots, picks = world
    ia, ib = read_all(tmp / "imp" / "imp_a.safetensors"), read_all(tmp / "imp" / "imp_b.safetensors")
    assert ia["implant.experts.w1"].shape == (3, FE, D) and ia["implant.experts.w2"].shape == (3, D, FE) and ia["implant.router.weight"].shape == (3, D)
    assert ib["implant.experts.w3"].shape == (2, FE, D)
    idx = json.load(open(tmp / "imp" / "imp_a.json"))
    assert [r["donor_layer"] for r in idx["rows"]] == [10, 30, 40] and idx["sha256"] == sha_file(tmp / "imp" / "imp_a.safetensors")
    for r in idx["rows"]:                                                   # rows are the source tensors, bit for bit
        key = f"slot.L{r['donor_layer']}.e{r['donor_expert']}."
        np.testing.assert_array_equal(ia["implant.experts.w1"][r["row"]], slots[key + "w1"])
        np.testing.assert_array_equal(ia["implant.experts.w2"][r["row"]], slots[key + "w2"])
        np.testing.assert_array_equal(ia["implant.router.weight"][r["row"]], slots[key + "router"])


def test_insertion_keeps_the_model_and_adds_switched_off_slots(world):
    tmp, w, opt, slots, picks = world
    out = tmp / "grown"
    tool("insert_implant.py", "--checkpoint", tmp / "ckpt", "--implant", tmp / "imp" / "imp_a.safetensors", tmp / "imp" / "imp_b.safetensors",
         "--layers", "1,2", "--out", out)
    g = read_all(out / "model.safetensors")
    rep = json.load(open(out / "implant_report.json"))
    state = json.load(open(out / "state.json"))
    add = 3                                                                 # 5 rows over 2 layers: 3 + 2, one filler in layer 2, 3 in layer 0
    assert rep["n_routed"] == {"before": E, "after": E + add} and state["moe"]["n_routed"] == E + add
    assert {n: a.shape for n, a in g.items()} == dict(specs(E + add))       # exactly the receiving format, one E everywhere
    assert state["model_sha256"] == sha_file(out / "model.safetensors") and state["parent_sha256"] == sha_file(tmp / "ckpt" / "model.safetensors")
    assert rep["filler_slots"] == {"0": 3, "1": 0, "2": 1}
    for n, a in w.items():
        if ".moe.experts." in n or ".moe.router." in n:
            np.testing.assert_array_equal(g[n][:E], a)                      # old slots bit for bit
        else:
            np.testing.assert_array_equal(g[n], a)                          # everything else bit for bit
    order = sorted(picks, key=lambda p: p[0])                               # donor-depth order: 10, 25, 30 -> layer 1; 40, 47 -> layer 2
    for layer, mine in ((1, order[:3]), (2, order[3:])):
        gain = w[f"blocks.{layer}.ffn_norm.weight"].astype(np.float64)
        p = f"blocks.{layer}.moe."
        for s, (dl, de, _) in enumerate(mine):
            key = f"slot.L{dl}.e{de}."
            np.testing.assert_allclose(g[p + "experts.w1"][E + s], slots[key + "w1"] / gain[None, :], rtol=1e-6)
            np.testing.assert_allclose(g[p + "experts.w3"][E + s], slots[key + "w3"] / gain[None, :], rtol=1e-6)
            np.testing.assert_array_equal(g[p + "experts.w2"][E + s], slots[key + "w2"])
            np.testing.assert_allclose(g[p + "router.weight"][E + s], slots[key + "router"] / gain, rtol=1e-6)
        for s in range(len(mine), add):                                     # filler: zero output, live input weights
            assert not g[p + "experts.w2"][E + s].any() and not g[p + "router.weight"][E + s].any() and g[p + "experts.w1"][E + s].any()
        assert (g[p + "router.bias"][E:] == -4.0).all()
    # the layer computes the same function: same experts per token, same output (up to BLAS rounding)
    x = np.random.default_rng(1).standard_normal((300, D)).astype(np.float32)
    for i in range(L):
        p = f"blocks.{i}.moe."
        ex = lambda t, e: {"w1": t[p + "experts.w1"][:e], "w3": t[p + "experts.w3"][:e], "w2": t[p + "experts.w2"][:e],
                           "router": t[p + "router.weight"][:e], "bias": t[p + "router.bias"][:e]}
        y0, i0 = adapt.seed_moe(x, ex(w, E), K, float(K))
        y1, i1 = adapt.seed_moe(x, ex(g, E + add), K, float(K))
        np.testing.assert_array_equal(i0, i1)
        np.testing.assert_allclose(y0, y1, rtol=2e-5, atol=2e-5)
    # optimizer: old moments kept, new rows zero
    go = read_all(out / "optim.safetensors")
    for n, a in opt.items():
        base = n[2:] if n.startswith("v.") else n
        if ".moe.experts." in base or base.endswith("moe.router.weight"):
            np.testing.assert_array_equal(go[n][:E], a)
            assert not go[n][E:].any() and go[n].shape[0] == E + add
        else:
            np.testing.assert_array_equal(go[n], a)


def test_insertion_refuses_what_would_change_the_model(world):
    tmp = world[0]
    args = ["--checkpoint", tmp / "ckpt", "--implant", tmp / "imp" / "imp_a.safetensors", "--layers", "1"]
    assert "off-bias" in tool("insert_implant.py", *args, "--off-bias", -0.5, "--out", tmp / "o1", ok=False).stderr
    assert tool("insert_implant.py", *args[:4], "--layers", "7", "--out", tmp / "o2", ok=False).returncode != 0
    assert tool("insert_implant.py", *args, "--out", tmp / "ckpt", ok=False).returncode != 0      # never in place
