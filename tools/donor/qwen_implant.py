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
from titan.donor import adapt, st                                 # noqa: E402
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
    ap.add_argument("--acts", default=None, help="directory of qwen_expert_acts.py: measured unit activations per expert")
    ap.add_argument("--slot-widths", default="768", help="comma list; the donor width is exact, smaller widths select units (lossy)")
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
    Q, P, bmeta = adapt.anchor_pair(Es, HB, ok, a.ridge)     # over the whole seed vocabulary; hold-out R^2 reported
    widths = [int(x) for x in a.slot_widths.split(",")]
    meta = {"bridges": bmeta, "ridge": a.ridge, "slot_gain": a.slot_gain, "rms_seed": "MISSING (scalar on w2; 1.0 used)",
            "slot_widths": widths, "experts": []}
    copy_t, proj_t = {}, {w: {} for w in widths}
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
        P_in, P_out = adapt.slot_bridges(Q, P, gs, wd, w_post, rms_d[L], a.slot_gain)
        w1, w3, w2 = adapt.fold_expert(g, u, d, P_in, P_out)
        router = adapt.project_router_row(gate_rows[e], P_in)
        ap_ = f"{a.acts}/act_L{L}_e{e}.npz" if a.acts else None
        acts = np.load(ap_) if ap_ and os.path.exists(ap_) else None
        imp = adapt.unit_importance(w1, w3, w2, acts["mean_abs"] if acts is not None else None)
        for fe in widths:
            f1, f3, f2, info = adapt.fit_slot(w1, w3, w2, fe, imp)
            keep = info.pop("kept_units", None)
            info["importance"] = ("measured mean |activation| on routed tokens x norm of the folded write vector" if acts is not None
                                  else "weight norms (data-free)")
            if acts is not None and keep is not None:
                an = acts["act"].astype(np.float32)
                info["rows_measured"] = int(an.shape[0])
                info["rel_sq_error_donor_space"] = adapt.selection_error(an, d, np.asarray(keep))
                info["rel_sq_error_seed_space"] = adapt.selection_error(an, w2, np.asarray(keep))
            rec["projection"][f"fe{fe}"] = {"fit": info, "tensors": {}}
            for nm, arr in (("w1", f1), ("w3", f3), ("w2", f2), ("router", router)):
                key = f"slot.L{L}.e{e}.{nm}"
                proj_t[fe][key] = ("F32", arr.shape, np.ascontiguousarray(arr, "<f4").tobytes())
                rec["projection"][f"fe{fe}"]["tensors"][key] = {"dtype": "F32", "shape": list(arr.shape), "sha256": st.sha256(proj_t[fe][key][2]),
                                                                "rms": float(np.sqrt((arr.astype(np.float64) ** 2).mean()))}
        rec["donor_residual_rms"] = rms_d[L]
        meta["experts"].append(rec)
    inv_c = st.write_safetensors(f"{a.out}/donor_experts_COPY.bf16.safetensors", copy_t,
                                 {"repo": a.repo, "revision": repo.revision, "kind": "direct copy of hub byte ranges"})
    meta["files"] = {"copy": inv_c["__file__"], "projection": {}}
    for fe in widths:
        name = "seed_slots_PROJECTION.f32.safetensors" if fe == g.shape[0] else f"seed_slots_PROJECTION_fe{fe}.f32.safetensors"
        inv_p = st.write_safetensors(f"{a.out}/{name}", proj_t[fe],
                                     {"kind": "anchor-bridge projection (initialisation, not calibrated on activations)", "slot_width": str(fe)})
        meta["files"]["projection"][f"fe{fe}"] = {"file": name, **inv_p["__file__"]}
    meta["provenance"] = {"repo": a.repo, "revision": repo.revision}
    json.dump(meta, open(f"{a.out}/implant_manifest.json", "w"), indent=1)
    print(json.dumps({"experts": len(sel), "files": meta["files"],
                      "kept_importance_share": {f"fe{fe}": [round(r["projection"][f"fe{fe}"]["fit"].get("kept_importance_share", 1.0), 4)
                                                             for r in meta["experts"]] for fe in widths},
                      "rel_sq_error_donor_space": {f"fe{fe}": [round(r["projection"][f"fe{fe}"]["fit"].get("rel_sq_error_donor_space", 0.0), 4)
                                                                for r in meta["experts"]] for fe in widths}}, indent=1))


if __name__ == "__main__":
    main()
