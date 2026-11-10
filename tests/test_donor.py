"""Donor tooling: safetensors by range, FP8, vocabulary contracts, bridges, expert adaptation (exactness claims)."""
import json

import numpy as np
import pytest

from titan.donor import adapt, bridge, fp8, st
from titan.donor.vocab import SeedBPE, Tekken, align_boundaries, transplant_rows, vocab_contract


def test_safetensors_roundtrip_ranges_and_hashes(tmp_path, rng):
    a = rng.standard_normal((5, 7)).astype(np.float32)
    b16 = st.f32_to_bf16_bits(rng.standard_normal((3, 4)).astype(np.float32))
    inv = st.write_safetensors(tmp_path / "x.safetensors", {"a": ("F32", a.shape, a.tobytes()),
                                                            "b": ("BF16", (3, 4), b16.tobytes())}, {"note": "t"})
    rd = st.LocalReader(tmp_path / "x.safetensors")
    refs, info = st.tensor_refs(rd)
    assert info["metadata"] == {"note": "t"} and set(refs) == {"a", "b"}
    for n in refs:
        raw = st.fetch(rd, refs[n])
        assert st.sha256(raw) == inv[n]["sha256"]
    np.testing.assert_array_equal(st.decode(st.fetch(rd, refs["a"]), "F32", (5, 7)), a)
    v = st.decode(st.fetch(rd, refs["b"]), "BF16", (3, 4))
    np.testing.assert_array_equal(st.f32_to_bf16_bits(v), b16)              # widening is exact
    import hashlib
    assert hashlib.sha256((tmp_path / "x.safetensors").read_bytes()).hexdigest() == inv["__file__"]["sha256"]
    torch = pytest.importorskip("torch")
    x = rng.standard_normal(4096).astype(np.float32)
    np.testing.assert_array_equal(st.f32_to_bf16_bits(x), torch.from_numpy(x).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16))


def test_fp8_e4m3_table_and_block_scale():
    t = fp8.TABLES["F8_E4M3"]
    assert t[0x7E] == 448.0 and t[0x01] == 2.0 ** -9 and np.isnan(t[0x7F]) and t[0x38] == 1.0 and t[0xB8] == -1.0
    w = np.arange(12, dtype=np.float32).reshape(3, 4)
    s = np.array([[2.0, 0.5]], np.float32)
    np.testing.assert_array_equal(fp8.apply_block_scale(w, s, (4, 2)), w * np.array([2, 2, 0.5, 0.5], np.float32))


def _seed_tok(tmp_path):
    merges = [[104, 101, 256], [256, 108, 257], [32, 119, 258], [108, 111, 259]]         # he, hel, " w", lo
    p = tmp_path / "seed.json"
    p.write_text(json.dumps({"version": 2, "pattern": r" ?[^\W\d]+| ?\d+| ?[^\s\w]+|\s+", "specials": ["<bos>", "<eos>"],
                             "merges": merges}))
    return SeedBPE(p)


def test_vocab_contract_and_alignment(tmp_path):
    import base64
    seed = _seed_tok(tmp_path)
    assert seed.vocab_size == 262 and seed.decode_bytes(seed.encode("hello world")) == b"hello world"
    vocab = [{"rank": i, "token_bytes": base64.b64encode(bytes([i])).decode(), "token_str": ""} for i in range(256)]
    vocab += [{"rank": 256, "token_bytes": base64.b64encode(b"he").decode(), "token_str": "he"},
              {"rank": 257, "token_bytes": base64.b64encode(b"ll").decode(), "token_str": "ll"}]
    tp = tmp_path / "tekken.json"
    tp.write_text(json.dumps({"config": {"default_vocab_size": 261, "default_num_special_tokens": 3, "pattern": ""},
                              "vocab": vocab, "special_tokens": [{"rank": 0, "token_str": "<unk>"}, {"rank": 1, "token_str": "<s>"},
                                                                 {"rank": 2, "token_str": "</s>"}]}))
    donor = Tekken(tp)
    c = vocab_contract(seed, donor, {"<bos>": 1, "<eos>": 2})
    assert c["stats"]["unknown"] == 0 and c["kinds"][256] == "exact" and c["kinds"][257] == "composed"
    assert c["map"][256] == [donor.index[b"he"]] and c["map"][257] == [donor.index[b"he"], donor.index[b"l"]]
    assert c["map"][260] == [1] and c["kinds"][261] == "special"
    for i in range(seed.n_regular):                                         # every row spells the seed token exactly
        assert b"".join(donor.token_bytes(j) for j in c["map"][i]) == seed.token_bytes(i)
    table = np.arange(261 * 2, dtype=np.float32).reshape(261, 2)
    rows, unknown = transplant_rows(table, c["map"])
    np.testing.assert_array_equal(rows[256], table[donor.index[b"he"]])       # exact rows are copies
    np.testing.assert_allclose(rows[257], table[c["map"][257]].mean(0))
    assert unknown == []
    np.testing.assert_array_equal(align_boundaries(np.array([2, 5, 6, 11]), np.array([1, 2, 6, 9, 11])),
                                  [[0, 1], [2, 2], [3, 4]])


def test_ridge_bridge_recovers_a_linear_map_and_anchors(rng):
    x = rng.standard_normal((600, 12))
    W0 = rng.standard_normal((12, 9))
    y = x @ W0 + 0.5
    W, b, info = bridge.fit_ridge(x, y, ridge=1e-9, bias=True)
    np.testing.assert_allclose(W, W0, atol=1e-4)
    assert info["r2_fit"] > 0.999999 and bridge.evaluate(W, b, x, y) > 0.999999
    _, _, nb = bridge.fit_ridge(x, y, ridge=1e-9, bias=False)
    assert nb["r2_fit"] < info["r2_fit"]                                    # an offset needs the intercept
    E = rng.standard_normal((400, 12)).astype(np.float32)                   # seed table
    D = (E @ W0).astype(np.float32)                                         # donor rows in a linearly related space
    ok = np.ones(400, bool)
    Pin, _, i1 = bridge.input_anchor(E, D, ok, ridge=1e-9)
    assert i1["r2_holdout"] > 0.9999 and Pin.shape == (12, 9)
    Pout, i2 = bridge.output_anchor(E, D, ok, ridge=1e-9)                   # P_out @ E' ~ D'
    assert Pout.shape == (9, 12) and i2["r2_holdout"] > 0.9999
    agree = bridge.logit_effect_agreement(Pout, E, D, 32)
    assert agree["logit_cosine_min"] > 0.999


def test_expert_folding_is_exact_composition_and_padding_is_exact(rng):
    d_s, d_d, f = 10, 14, 6
    gate, up, down = rng.standard_normal((f, d_d)), rng.standard_normal((f, d_d)), rng.standard_normal((d_d, f))
    P_in, P_out = rng.standard_normal((d_s, d_d)), rng.standard_normal((d_d, d_s))
    w1, w3, w2 = adapt.fold_expert(gate, up, down, P_in, P_out)
    x = rng.standard_normal((50, d_s))
    want = adapt.swiglu(x @ P_in, gate, up, down) @ P_out                   # bridge -> donor expert -> bridge
    np.testing.assert_allclose(adapt.swiglu(x, w1, w3, w2), want, rtol=2e-4, atol=2e-4)
    p1, p3, p2, info = adapt.fit_slot(w1, w3, w2, 9)
    assert info["mode"].startswith("zero-pad") and p1.shape == (9, d_s) and p2.shape == (d_s, 9)
    np.testing.assert_array_equal(adapt.swiglu(x.astype(np.float32), p1, p3, p2),
                                  adapt.swiglu(x.astype(np.float32), w1, w3, w2))             # bit-identical
    s1, s3, s2, sinfo = adapt.fit_slot(w1, w3, w2, 4)
    assert sinfo["mode"].startswith("unit selection") and 0 < sinfo["kept_importance_share"] < 1 and s1.shape[0] == 4
    w2z = w2.copy()
    w2z[:, [1, 4]] = 0                                                      # two dead units: dropping them is exact
    d1, d3, d2, dinfo = adapt.fit_slot(w1, w3, w2z, 4)
    assert dinfo["kept_importance_share"] == pytest.approx(1.0)
    np.testing.assert_allclose(adapt.swiglu(x.astype(np.float32), d1, d3, d2), adapt.swiglu(x.astype(np.float32), w1, w3, w2z),
                               rtol=1e-5, atol=1e-5)


def test_inserted_slots_leave_the_layer_unchanged(rng):
    d, fe, E, k = 8, 5, 6, 3
    ex = {"w1": rng.standard_normal((E, fe, d)).astype(np.float32), "w3": rng.standard_normal((E, fe, d)).astype(np.float32),
          "w2": rng.standard_normal((E, d, fe)).astype(np.float32), "router": rng.standard_normal((E, d)).astype(np.float32),
          "bias": np.zeros(E, np.float32)}
    new = [{"w1": rng.standard_normal((fe, d)).astype(np.float32), "w3": rng.standard_normal((fe, d)).astype(np.float32),
            "w2": rng.standard_normal((d, fe)).astype(np.float32), "router": 50 * rng.standard_normal(d).astype(np.float32)}
           for _ in range(4)]
    x = rng.standard_normal((200, d)).astype(np.float32)
    y0, idx0 = adapt.seed_moe(x, ex, k, float(k))
    grown = adapt.insert_slots(ex, new)
    y1, idx1 = adapt.seed_moe(x, grown, k, float(k))
    np.testing.assert_array_equal(idx0, idx1)
    np.testing.assert_array_equal(y0, y1)                                    # identity at insertion, bit for bit
    assert idx1.max() < E
    with pytest.raises(ValueError):
        adapt.insert_slots(ex, new, off_bias=-0.5)
    r = adapt.project_router_row(rng.standard_normal(11), rng.standard_normal((d, 11)))
    assert r.shape == (d,)


def test_unit_selection_error_and_slot_bridges():
    """selection_error is the relative squared output error of the dropped units on given activations;
    slot_bridges reproduces the documented formulas; anchor bridges recover a planted linear relation."""
    from titan.donor import adapt
    rng = np.random.default_rng(3)
    f, dd, ds, n = 24, 10, 7, 200
    act = rng.standard_normal((n, f)) * np.linspace(2.0, 0.05, f)[None, :]
    w2 = rng.standard_normal((dd, f)).astype(np.float32)
    keep = np.arange(16)
    full = act @ w2.T.astype(np.float64)
    part = act[:, keep] @ w2[:, keep].T.astype(np.float64)
    want = ((full - part) ** 2).sum() / (full ** 2).sum()
    assert abs(adapt.selection_error(act, w2, keep) - want) < 1e-12
    assert adapt.selection_error(act, w2, np.arange(f)) == 0.0
    imp = adapt.unit_importance(w2[:, :1].T.repeat(f, 0), w2[:, :1].T.repeat(f, 0), w2, np.abs(act).mean(0))
    best = np.sort(np.argsort(-imp)[:16])
    worst = np.sort(np.argsort(imp)[:16])
    assert adapt.selection_error(act, w2, best) < adapt.selection_error(act, w2, worst)
    # formulas
    Q, P = rng.standard_normal((ds, dd)), rng.standard_normal((dd, ds))
    gs, wd, wp = rng.uniform(0.5, 2, ds), rng.uniform(0.5, 2, dd), rng.uniform(0.5, 2, dd)
    P_in, P_out = adapt.slot_bridges(Q, P, gs, wd, wp, donor_residual_rms=3.0, slot_gain=2.0, seed_residual_rms=0.5)
    assert np.allclose(P_in, np.diag(gs / 2.0) @ Q @ np.diag(wp / wd))
    assert np.allclose(P_out, np.diag(wd) @ P @ np.diag(1 / gs) * (0.5 / 3.0))
    # planted relation: donor head rows are a linear image of the seed rows -> P recovers it (small ridge)
    V = 400
    E = rng.standard_normal((V, ds))
    A = rng.standard_normal((ds, dd))
    H = E @ A
    Q2, P2, meta = adapt.anchor_pair(E, H, np.ones(V, bool), ridge=1e-8)
    assert np.allclose(P2.T, A, atol=1e-4) and meta["P_donor_to_seed_prehead"]["r2_holdout_tokens"] > 0.999
