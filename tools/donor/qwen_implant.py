#!/usr/bin/env python3
"""Qwen3-MoE experts -> seed MoE slot tensors, with the three kinds of operations kept apart.

  COPY        selected donor expert tensors by HTTP range, SHA-256 of the exact payload bytes
  PROJECTION  data-free anchor bridges from the token tables (seed tied embedding, donor head rows on the seed
              vocabulary, both final norm gains) folded into the expert:
                 Q   [d_s, d_d]  seed pre-head vector -> donor pre-head vector   (HB @ Q' ~ E_seed)
                 P   [d_d, d_s]  donor pre-head update -> seed pre-head update   (P @ E_seed' ~ HB')
                 P_in  = diag(g_seed / g_slot) @ Q @ diag(w_post(L) / w_final)
                 P_out = diag(w_final) @ P @ diag(1 / g_seed) * (rms_seed / rms_donor(L))
              w1' = gate @ P_in', w3' = up @ P_in', w2' = P_out' @ down ; router row r = P_in @ gate_row
  HEAL        none. Slots are meant to be inserted switched off (selection bias), see titan.donor.adapt.
rms_seed (RMS of the seed residual where the slot writes) is NOT known without the full seed model; it is a
plain scalar on w2' and is set to 1.0 here.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from titan.donor import adapt, bridge, st                                 # noqa: E402
from titan.donor.hf import HFRepo                                         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    ap.add_argument("--rev", default="b2cff646eb4bb1d68355c01b18ae02e7cf42d120")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--selection", required=True, help="JSON list of {layer, expert}")
    ap.add_argument("--seed-emb", required=True)
    ap.add_argument("--seed-norm", required=True)
    ap.add_argument("--donor-head-rows", required=True)
    ap.add_argument("--donor-norm", required=True)
    ap.add_argument("--donor-rms", required=True, help="JSON {layer: residual RMS after that layer}")
    ap.add_argument("--exact-rows", required=True, help=".npy bool [V_seed]: rows with an exact donor token")
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--slot-gain", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    repo = HFRepo(a.repo, a.rev, cache=a.cache)
    refs = repo.refs()
    sel = json.load(open(a.selection))
    Es = np.load(a.seed_emb).astype(np.float64)
    gs = np.load(a.seed_norm).astype(np.float64)
    HB = np.load(a.donor_head_rows).astype(np.float64)
    wd = np.load(a.donor_norm).astype(np.float64)
    rms_d = {int(k): float(v) for k, v in json.load(open(a.donor_rms)).items()}
    ok = np.load(a.exact_rows)
    # anchors over the whole seed vocabulary (the objective is over all tokens); hold-out R^2 reported for reference
    Q_T, _, qi = bridge.fit_ridge(HB, Es, a.ridge, bias=False)             # HB @ Q' ~ Es
    Q = Q_T.T.astype(np.float64)                                           # [d_s, d_d]
    W, _, pi = bridge.fit_ridge(Es, HB, a.ridge, bias=False)               # Es @ W ~ HB
    P = W.T.astype(np.float64)                                             # [d_d, d_s]
    _, _, qh = bridge.input_anchor(HB, Es, ok, a.ridge)
    _, ph = bridge.output_anchor(Es, HB, ok, a.ridge)
    meta = {"bridges": {"Q_seed_to_donor_prehead": {**qi, "r2_holdout_tokens": qh["r2_holdout"], "sha256": bridge.sha256_array(Q.astype(np.float32))},
                        "P_donor_to_seed_prehead": {**pi, "r2_holdout_tokens": ph["r2_holdout"], "sha256": bridge.sha256_array(P.astype(np.float32))}},
            "ridge": a.ridge, "slot_gain": a.slot_gain, "rms_seed": "MISSING (scalar on w2; 1.0 used)", "experts": []}
    copy_t, proj_t = {}, {}
    post = {}
    for s in sel:
        L, e = int(s["layer"]), int(s["expert"])
        pre = f"model.layers.{L}.mlp.experts.{e}."
        names = [pre + "gate_proj.weight", pre + "up_proj.weight", pre + "down_proj.weight"]
        blobs = repo.fetch_span(names + [f"model.layers.{L}.mlp.gate.weight", f"model.layers.{L}.post_attention_layernorm.weight"], threads=3)
        if L not in post:
            post[L] = (st.bf16_to_f32(blobs[f"model.layers.{L}.post_attention_layernorm.weight"]).astype(np.float64),
                       st.bf16_to_f32(blobs[f"model.layers.{L}.mlp.gate.weight"]).reshape(refs[f"model.layers.{L}.mlp.gate.weight"].shape))
        w_post, gate_rows = post[L]
        g, u, d = (st.bf16_to_f32(blobs[n]).reshape(refs[n].shape) for n in names)
        rec = {"layer": L, "expert": e, "copy": {}, "projection": {}}
        for n in names:
            r = refs[n]
            key = f"donor.L{L}.e{e}.{n.rsplit('.', 2)[1]}"
            copy_t[key] = ("BF16", r.shape, blobs[n])
            rec["copy"][key] = {"source": n, "shard": r.shard, "byte_range": [r.start, r.end], "dtype": r.dtype,
                                "shape": list(r.shape), "sha256": st.sha256(blobs[n])}
        P_in = (gs / a.slot_gain)[:, None] * Q * (w_post / wd)[None, :]
        P_out = wd[:, None] * P / gs[None, :] / rms_d[L]
        w1, w3, w2 = adapt.fold_expert(g, u, d, P_in, P_out)
        router = adapt.project_router_row(gate_rows[e], P_in)
        for nm, arr in (("w1", w1), ("w3", w3), ("w2", w2), ("router", router)):
            key = f"slot.L{L}.e{e}.{nm}"
            proj_t[key] = ("F32", arr.shape, np.ascontiguousarray(arr, "<f4").tobytes())
            rec["projection"][key] = {"dtype": "F32", "shape": list(arr.shape), "sha256": st.sha256(proj_t[key][2]),
                                      "rms": float(np.sqrt((arr.astype(np.float64) ** 2).mean()))}
        rec["donor_residual_rms"] = rms_d[L]
        meta["experts"].append(rec)
    inv_c = st.write_safetensors(f"{a.out}/donor_experts_COPY.bf16.safetensors", copy_t,
                                 {"repo": a.repo, "revision": repo.revision, "kind": "direct copy of hub byte ranges"})
    inv_p = st.write_safetensors(f"{a.out}/seed_slots_PROJECTION.f32.safetensors", proj_t,
                                 {"kind": "anchor-bridge projection (initialisation, not calibrated on activations)"})
    meta["files"] = {"copy": inv_c["__file__"], "projection": inv_p["__file__"]}
    meta["provenance"] = {"repo": a.repo, "revision": repo.revision}
    json.dump(meta, open(f"{a.out}/implant_manifest.json", "w"), indent=1)
    print(json.dumps({"experts": len(sel), "copy_file": inv_c["__file__"], "projection_file": inv_p["__file__"],
                      "Q": meta["bridges"]["Q_seed_to_donor_prehead"], "P": meta["bridges"]["P_donor_to_seed_prehead"]}, indent=1))


if __name__ == "__main__":
    main()
