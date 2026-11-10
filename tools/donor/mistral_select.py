#!/usr/bin/env python3
"""Expert and layer selection for Mistral Large 4 from the routing tables of tools/donor/mistral_forward.py.

Three functions are looked for, each through a contrast that is defined before looking at the data:

  reasoning          positions inside worked solution steps (GSM8K) against the problem statement of the same
                     items, encyclopedic prose and source code
  adaptive thinking  (a) the same tokens under reasoning_effort high and low (paired: only the header
                         differs, so any difference at a later position is caused by the setting)
                     (b) solution steps inside a [THINK] block against the same kind of text without one
  learning           positions that must produce the output of a rule shown only through examples in the
                     prompt (in-context learning) against the positions that read the inputs, problem text,
                     prose and code; "learning_gain" keeps those candidates whose mass is larger at late
                     examples (6+) than at early ones (1-2) in both halves, ranked by that growth

"mass" of expert e in a group = mean gate weight e receives from the group's tokens (each token spreads
weight 1 over its top-k experts); uniform = 1 / n_experts. An expert is a candidate when its mass in the target
group is at least `--ratio` times its mass in every baseline group, in both halves of the items separately
(split-half), with at least `--min-tokens` routed target tokens per half. Candidates are ranked by excess
mass. Nothing here says what an expert computes; it says where it is used.
"""
import argparse
import glob
import json
import os

import numpy as np

KINDS = ["wiki", "code", "gsm_raw", "chat", "icl"]


def mass(topi, topw, idx, n):
    if len(idx) == 0:
        return np.zeros(n)
    return np.bincount(topi[idx].ravel(), weights=topw[idx].ravel().astype(np.float64), minlength=n) / len(idx)


def count(topi, idx, n):
    return np.bincount(topi[idx].ravel(), minlength=n)


def tv(a, b):
    return float(0.5 * np.abs(a - b).sum())


def dense(topi, topw, idx, n):
    w = np.zeros((len(idx), n))
    np.put_along_axis(w, topi[idx].astype(np.int64), topw[idx].astype(np.float64), 1)
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ratio", type=float, default=2.0)
    ap.add_argument("--min-tokens", type=int, default=12)
    ap.add_argument("--top", type=int, default=16)
    ap.add_argument("--experts", type=int, default=128)
    a = ap.parse_args()
    T = np.load(f"{a.run}/tokens.npz")
    kind, seg, tseg, tshot, item, effort, think = (T[k] for k in ("kind", "seg", "tseg", "tshot", "item", "effort", "think"))
    half = item % 2
    PR = np.load(f"{a.run}/pairs.npz")
    S = np.load(f"{a.run}/samples.npz")
    E = a.experts
    once = (kind != 3) | (effort == 2)                       # chat items exist three times (efforts): count one copy
    G = {
        "reasoning": np.where((seg == 2) & once & ((kind == 2) | ((kind == 3) & (think == 1))))[0],
        "question": np.where((seg == 1) & once)[0],
        "wiki": np.where((kind == 0) & (seg == 6))[0],
        "code": np.where((kind == 1) & (seg == 6))[0],
        "think_steps": np.where((kind == 3) & (think == 1) & (seg == 2) & (effort == 2))[0],
        "plain_steps": np.where((kind == 2) & (seg == 2))[0],
        "icl_out": np.where(tseg == 5)[0],
        "icl_out_late": np.where((tseg == 5) & (tshot >= 6))[0],
        "icl_out_early": np.where((tseg == 5) & (tshot <= 2))[0],
        "icl_in": np.where((kind == 4) & (tseg == 4))[0],
    }
    CONTRASTS = {"reasoning": ("reasoning", ["question", "wiki", "code"]),
                 "think_mode": ("think_steps", ["plain_steps", "question", "wiki", "code"]),
                 "learning": ("icl_out", ["icl_in", "question", "wiki", "code"])}
    layers = sorted(int(os.path.basename(p)[7:9]) for p in glob.glob(f"{a.run}/route_L*.npz"))
    recs = json.load(open(f"{a.run}/layers.json"))
    rms = {r["layer"]: r["rms"] for r in recs}
    cand = {k: [] for k in list(CONTRASTS) + ["effort"]}
    prof = []
    lens_idx = S["lens_idx"]
    for l in layers:
        R = np.load(f"{a.run}/route_L{l:02d}.npz")
        topi, topw = R["topi"].astype(np.int64), R["topw"].astype(np.float32)
        tk = np.load(f"{a.run}/tok_L{l:02d}.npz") if os.path.exists(f"{a.run}/tok_L{l:02d}.npz") else None
        row = {"layer": l, "residual_rms": rms.get(l)}
        m = {g: mass(topi, topw, idx, E) for g, idx in G.items()}
        mh = {(g, h): mass(topi, topw, idx[half[idx] == h], E) for g, idx in G.items() for h in (0, 1)}
        for name, (tg, base) in CONTRASTS.items():
            row[f"tv_{name}"] = min(tv(m[tg], m[b]) for b in base)
            row[f"tv_{name}_noise"] = tv(mh[(tg, 0)], mh[(tg, 1)])
            for e in range(E):
                ok, ratios = True, []
                for h in (0, 1):
                    idx = G[tg][half[G[tg]] == h]
                    n_e = int((topi[idx] == e).any(1).sum())
                    b = max(mh[(bb, h)][e] for bb in base)
                    r = mh[(tg, h)][e] / max(b, 1e-6)
                    ratios.append(r)
                    ok &= (r >= a.ratio) and (n_e >= a.min_tokens)
                if ok:
                    c = {"layer": l, "expert": e, "mass_target": float(m[tg][e]),
                         "mass_baselines": {b: float(m[b][e]) for b in base},
                         "excess": float(m[tg][e] - max(m[b][e] for b in base)), "ratio_half": [float(x) for x in ratios],
                         "target_tokens_routed": int((topi[G[tg]] == e).any(1).sum()), "target_tokens": int(len(G[tg]))}
                    if name == "learning":
                        c["mass_late"], c["mass_early"] = float(m["icl_out_late"][e]), float(m["icl_out_early"][e])
                        c["late_above_early_in_both_halves"] = bool(all(mh[("icl_out_late", h)][e] > mh[("icl_out_early", h)][e] for h in (0, 1)))
                    cand[name].append(c)
        row["tv_icl_late_early"] = tv(m["icl_out_late"], m["icl_out_early"])
        row["tv_icl_late_noise"] = tv(mh[("icl_out_late", 0)], mh[("icl_out_late", 1)])
        # ---- paired effort contrast
        for key in ("high_low", "medium_low", "high_medium", "think_nothink"):
            ia, ib = PR[f"{key}_a"], PR[f"{key}_b"]
            if len(ia) == 0:
                continue
            wa, wb = dense(topi, topw, ia, E), dense(topi, topw, ib, E)
            row[f"route_tv_{key}"] = float(0.5 * np.abs(wa - wb).sum(1).mean())
            row[f"top1_changed_{key}"] = float((topi[ia, 0] != topi[ib, 0]).mean())
            if tk is not None and f"rel_{key}" in tk.files:
                row[f"resid_rel_{key}"] = float(tk[f"rel_{key}"].astype(np.float32).mean())
            if key == "high_low":
                grp = item[ia] * 2 + think[ia]
                gs = np.unique(grp)
                d = np.stack([(wa[grp == g] - wb[grp == g]).mean(0) for g in gs])          # [groups, E]
                mean, sd = d.mean(0), d.std(0, ddof=1) + 1e-12
                t = mean / (sd / np.sqrt(len(gs)))
                h0 = d[(gs // 2) % 2 == 0].mean(0)
                h1 = d[(gs // 2) % 2 == 1].mean(0)
                for e in range(E):
                    if abs(t[e]) >= 4.0 and h0[e] * h1[e] > 0 and abs(mean[e]) >= 0.25 / E:
                        cand["effort"].append({"layer": l, "expert": e, "delta_mass_high_minus_low": float(mean[e]), "t": float(t[e]),
                                               "sign_consistency": float((np.sign(d[:, e]) == np.sign(mean[e])).mean()),
                                               "groups": int(len(gs)), "mass_high": float(wa[:, e].mean()), "mass_low": float(wb[:, e].mean()),
                                               "excess": float(abs(mean[e]))})
        # ---- update sizes and logit lens by group
        if tk is not None:
            for g in ("reasoning", "question", "wiki", "code", "icl_out"):
                idx = G[g]
                row[f"mlp_rel_{g}"] = float((tk["mlp_norm"][idx].astype(np.float32) / tk["h_norm"][idx].astype(np.float32)).mean())
            row["attn_rel"] = float((tk["attn_norm"].astype(np.float32) / tk["h_norm"].astype(np.float32)).mean())
            if "lens_ce" in tk.files:
                lc = tk["lens_ce"]
                for g, sel in (("answer", tseg[lens_idx] == 3), ("icl_out_late", (tseg[lens_idx] == 5) & (tshot[lens_idx] >= 6)),
                               ("icl_out_early", (tseg[lens_idx] == 5) & (tshot[lens_idx] <= 2)), ("reasoning", tseg[lens_idx] == 2),
                               ("question", tseg[lens_idx] == 1), ("wiki", kind[lens_idx] == 0), ("code", kind[lens_idx] == 1)):
                    if sel.any():
                        row[f"lens_{g}"] = float(lc[sel].mean())
        prof.append(row)
    out = {"run": a.run, "layers_analysed": layers, "criteria": {"ratio": a.ratio, "min_tokens_per_half": a.min_tokens,
                                                                 "uniform_mass": 1.0 / E, "effort": "|t| >= 4 over item groups, same sign in both halves, |delta| >= 0.25/E"},
           "group_tokens": {g: int(len(i)) for g, i in G.items()}, "pair_tokens": {k[:-2]: int(len(PR[k])) for k in PR.files if k.endswith("_a")},
           "layer_profile": prof, "candidates": {}, "selected": {}}
    # learning candidates whose use grows with the number of examples (late > early in both halves), by that growth
    cand["learning_gain"] = [dict(c, excess=c["mass_late"] - c["mass_early"]) for c in cand["learning"] if c["late_above_early_in_both_halves"]]
    for name, cs in cand.items():
        cs.sort(key=lambda c: -c["excess"])
        out["candidates"][name] = {"n": len(cs), "per_layer": {str(l): sum(c["layer"] == l for c in cs) for l in layers}}
        out["selected"][name] = cs[:a.top]
    # ---- measured behaviour of the full forward (only when the run is complete)
    if os.path.exists(f"{a.run}/final_ce.npy"):
        ce = np.full(len(kind), np.nan)
        ce[T["tgt"] >= 0] = np.load(f"{a.run}/final_ce.npy")
        fin = {"icl_output_ce_by_shot": {str(k): float(np.nanmean(ce[(tseg == 5) & (tshot == k)])) for k in range(1, int(tshot.max()) + 1)},
               "ce_by_target": {g: float(np.nanmean(ce[sel])) for g, sel in (("wiki", kind == 0), ("code", kind == 1),
                                ("gsm_question", (kind == 2) & (tseg == 1)), ("gsm_steps", (kind == 2) & (tseg == 2)), ("gsm_answer", (kind == 2) & (tseg == 3)),
                                ("icl_input", tseg == 4), ("icl_output", tseg == 5))}}
        for key in ("high_low", "medium_low", "think_nothink"):
            ia, ib = PR[f"{key}_a"], PR[f"{key}_b"]
            for g, code in (("steps", 2), ("answer", 3)):
                sel = (tseg[ia] == code) & np.isfinite(ce[ia]) & np.isfinite(ce[ib])
                if sel.any():
                    fin[f"ce_{key}_{g}"] = {"a": float(ce[ia][sel].mean()), "b": float(ce[ib][sel].mean()), "n": int(sel.sum())}
        out["final_forward"] = fin
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({"layers": len(layers), "groups": out["group_tokens"], "pairs": out["pair_tokens"],
                      "candidates": {k: v["n"] for k, v in out["candidates"].items()}}, indent=1))
    for name, cs in out["selected"].items():
        print(name, [(c["layer"], c["expert"], round(c["excess"], 4)) for c in cs[:8]])


if __name__ == "__main__":
    main()
