#!/usr/bin/env python3
"""Mistral Large 4 experts -> seed MoE slot tensors. Copy, projection and measurement are kept apart.

  COPY        the selected experts' FP8 payloads and block scales by HTTP range; SHA-256 of the exact bytes,
              shard and byte range in the manifest (the bytes are stored only with --store-copy: one expert is
              113 MB, and shard + range + hash regenerate it exactly)
  DECODE      W = fp8 * scale per 128x128 block (the checkpoint's own definition of the weight), float32
  PROJECTION  vocabulary transplant of the output matrix rows (copy / mean of pieces), data-free anchor
              bridges between the pre-head spaces, folded into the expert (titan.donor.adapt); the expert's
              6144 hidden units are reduced to the slot width by measured importance
              (mean |activation| on the tokens routed to the expert x norm of the unit's write vector)
  MEASURED    for every expert, on its real inputs from the forward (xn sample): relative squared error of
              the expert output when only the kept units remain, in donor space and after the output bridge;
              and the saved contribution norms are reproduced from the sample (checks the whole path)
  HEAL        none.
rms_seed and the gain of the receiving seed layer's ffn_norm are not known without the full seed model.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from titan.donor import adapt, st                                         # noqa: E402
from titan.donor import mistral_l4 as M                                   # noqa: E402
from titan.donor.hf import HFRepo                                         # noqa: E402
from titan.donor.vocab import transplant_rows                             # noqa: E402


def compact_rows(memmap_bf16, rows):
    """transplant_rows on a BF16 table without loading it: only the donor rows that are used are converted."""
    uniq = np.unique(np.array([i for r in rows for i in r], np.int64))
    pos = {int(u): k for k, u in enumerate(uniq)}
    table = st.bf16_to_f32(np.ascontiguousarray(memmap_bf16[uniq])).reshape(len(uniq), -1)
    return transplant_rows(table, [[pos[i] for i in r] for r in rows]), len(uniq)


def decode(refs, blobs, n):
    rf = refs[n]
    if rf.dtype == "F8_E4M3":
        return M.dequant(blobs[n], rf.shape, blobs[M.scale_name(n)])
    return torch.from_numpy(st.bf16_to_f32(blobs[n]).reshape(rf.shape).copy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="mistralai/Mistral-Large-4-1T-A52B")
    ap.add_argument("--rev", default=None, help="hub revision (default: the revision recorded by the run)")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--run", required=True, help="output directory of mistral_forward.py")
    ap.add_argument("--selection", required=True, help="output of mistral_select.py")
    ap.add_argument("--per-function", type=int, default=8)
    ap.add_argument("--contract", required=True, help="vocab_contract.json (seed token -> donor ids)")
    ap.add_argument("--seed-emb", required=True)
    ap.add_argument("--seed-norm", required=True)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--slot-width", type=int, default=640)
    ap.add_argument("--report-widths", default="320,640,768,1280,1920,3072")
    ap.add_argument("--slot-gain", type=float, default=1.0)
    ap.add_argument("--store-copy", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fin = f"{a.run}/final.json"
    rev = a.rev or (json.load(open(fin))["revision"] if os.path.exists(fin) else None)
    repo = HFRepo(a.repo, rev, cache=a.cache)
    refs = repo.refs()
    S = json.load(open(a.selection))
    picks = {}
    for fn, cs in S["selected"].items():
        for rank, c in enumerate(cs[:a.per_function]):
            picks.setdefault((c["layer"], c["expert"]), []).append({"function": fn, "rank": rank + 1, **{k: v for k, v in c.items() if k not in ("layer", "expert")}})
    contract = json.load(open(a.contract))
    Es = np.load(a.seed_emb).astype(np.float64)
    gs = np.load(a.seed_norm).astype(np.float64)
    ok = np.array([k == "exact" for k in contract["kinds"]]) if isinstance(contract["kinds"], list) else None
    head = np.memmap(os.path.join(a.cache, "output.weight.bf16"), dtype="<u2", mode="r").reshape(refs["output.weight"].shape)
    (HB, unk), n_used = compact_rows(head, contract["map"])
    r, b = repo.fetch("norm.weight")
    wd = st.bf16_to_f32(b).astype(np.float64)
    Q, P, bmeta = adapt.anchor_pair(Es, HB, ok, a.ridge)
    proj = {"transplant.head_rows": ("F32", HB.shape, np.ascontiguousarray(HB, "<f4").tobytes())}
    meta = {"provenance": {"repo": a.repo, "revision": repo.revision}, "bridges": bmeta, "ridge": a.ridge, "slot_gain": a.slot_gain,
            "slot_width": a.slot_width, "rms_seed": "MISSING (scalar on w2; 1.0 used)", "seed_ffn_norm_gain": "MISSING for the receiving layer (1.0 used)",
            "head_transplant": {"rows": int(HB.shape[0]), "donor_rows_used": n_used, "unknown": len(unk), "sha256": st.sha256(proj["transplant.head_rows"][2]),
                                "source": "output.weight", "rule": "exact: copy of the row; composed: mean of the piece rows"},
            "experts": []}
    rms_d = {x["layer"]: x["rms"] for x in json.load(open(f"{a.run}/layers.json"))}
    xs = np.load(f"{a.run}/samples.npz")["xn_idx"]
    widths = sorted({int(x) for x in a.report_widths.split(",")} | {a.slot_width})
    copy_t = {}
    for (L, e), why in sorted(picks.items()):
        en = M.expert_tensor_names(L, e)
        names = [en[k] for k in ("w1", "w2", "w3")]
        extra = [f"layers.{L}.ffn_norm.weight", f"layers.{L}.gate.weight"]
        scales = [M.scale_name(n) for n in names + extra if refs[n].dtype == "F8_E4M3"]
        blobs = repo.fetch_span(names + extra + scales, threads=24, piece=8 << 20)
        rec = {"layer": L, "expert": e, "selected_for": why, "copy": {}, "donor_residual_rms": rms_d[L]}
        for n in names + [M.scale_name(n) for n in names]:
            rf = refs[n]
            rec["copy"][n] = {"shard": rf.shard, "byte_range": [rf.start, rf.end], "dtype": rf.dtype, "shape": list(rf.shape), "sha256": st.sha256(blobs[n])}
            if a.store_copy:
                copy_t[n] = (rf.dtype, rf.shape, blobs[n])

        w1, w2, w3 = (decode(refs, blobs, n) for n in names)
        w_post = decode(refs, blobs, extra[0]).numpy().astype(np.float64)
        gate_row = decode(refs, blobs, extra[1]).numpy()[e]
        # ---- measurement on the expert's real inputs
        R = np.load(f"{a.run}/route_L{L:02d}.npz")
        topi, topw = R["topi"][xs].astype(np.int64), R["topw"][xs].astype(np.float32)
        rows, slot = np.where(topi == e)
        X = torch.from_numpy(np.load(f"{a.run}/xn_L{L:02d}.f16.npy", mmap_mode="r")[rows].astype(np.float32))
        with torch.no_grad():
            act = F.silu(F.linear(X, w1)) * F.linear(X, w3)
            y = F.linear(act, w2)
        saved = R["contrib"][xs][rows, slot].astype(np.float32)
        mine = (y.norm(dim=-1).numpy() * topw[rows, slot])
        rec["sample"] = {"tokens": int(len(rows)), "contribution_norm_rel_dev_median": float(np.median(np.abs(mine - saved) / np.maximum(saved, 1e-6))) if len(rows) else None}
        # ---- projection
        P_in, P_out = adapt.slot_bridges(Q, P, gs, wd, w_post, rms_d[L], a.slot_gain)
        f1, f3, f2 = adapt.fold_expert(w1.numpy(), w3.numpy(), w2.numpy(), P_in, P_out)
        imp = R["unit_act"][e].astype(np.float64) * np.linalg.norm(f2.astype(np.float64), axis=0)
        order = np.argsort(-imp, kind="stable")
        rec["unit_selection"] = {"importance": "measured mean |activation| on routed tokens x norm of the folded write vector", "by_width": {}}
        an = act.numpy()
        for fe in widths:
            keep = np.sort(order[:fe])
            d = {"kept_importance_share": float(imp[keep].sum() / max(imp.sum(), 1e-30))}
            if len(rows) >= 8:
                d["rel_sq_error_donor_space"] = adapt.selection_error(an, w2.numpy(), keep)
                d["rel_sq_error_seed_space"] = adapt.selection_error(an, f2, keep)
            rec["unit_selection"]["by_width"][str(fe)] = d
        keep = np.sort(order[:a.slot_width])
        router = adapt.project_router_row(gate_row, P_in)
        rec["projection"] = {}
        for nm, arr in (("w1", f1[keep]), ("w3", f3[keep]), ("w2", f2[:, keep]), ("router", router), ("kept_units", keep.astype("<i4"))):
            key = f"slot.L{L}.e{e}.{nm}"
            dt = "I32" if nm == "kept_units" else "F32"
            proj[key] = (dt, arr.shape, np.ascontiguousarray(arr, "<i4" if dt == "I32" else "<f4").tobytes())
            rec["projection"][key] = {"dtype": dt, "shape": list(arr.shape), "sha256": st.sha256(proj[key][2])}
        meta["experts"].append(rec)
        u = rec["unit_selection"]["by_width"][str(a.slot_width)]
        print(f"L{L:02d} e{e:03d} {[w['function'] for w in why]} sample {len(rows)} | dev {rec['sample']['contribution_norm_rel_dev_median']}"
              f" | fe{a.slot_width}: share {u['kept_importance_share']:.3f} err {u.get('rel_sq_error_donor_space', float('nan')):.3f}", flush=True)
        del w1, w2, w3, act, y, f1, f2, f3, blobs
    inv = st.write_safetensors(f"{a.out}/seed_slots_PROJECTION_fe{a.slot_width}.f32.safetensors", proj,
                               {"kind": "anchor-bridge projection + unit selection (initialisation, not calibrated on seed activations)",
                                "slot_width": str(a.slot_width), "repo": a.repo, "revision": repo.revision})
    meta["files"] = {"projection": {"file": f"seed_slots_PROJECTION_fe{a.slot_width}.f32.safetensors", **inv["__file__"]}}
    if a.store_copy:
        ic = st.write_safetensors(f"{a.out}/donor_experts_COPY.fp8.safetensors", copy_t,
                                  {"repo": a.repo, "revision": repo.revision, "kind": "direct copy of hub byte ranges"})
        meta["files"]["copy"] = {"file": "donor_experts_COPY.fp8.safetensors", **ic["__file__"]}
    else:
        meta["files"]["copy"] = "not stored: regenerate from shard + byte range, verify with sha256 (or run with --store-copy)"
    json.dump(meta, open(f"{a.out}/implant_manifest.json", "w"), indent=1)
    print(json.dumps({"experts": len(meta["experts"]), "bridges": {k: {x: v[x] for x in ("r2_fit", "r2_holdout_tokens")} for k, v in bmeta.items()},
                      "files": meta["files"]}, indent=1))


if __name__ == "__main__":
    main()
