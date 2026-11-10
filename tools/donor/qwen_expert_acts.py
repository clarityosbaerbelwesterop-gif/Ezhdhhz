#!/usr/bin/env python3
"""Hidden activations of selected Qwen3-MoE experts on their real inputs.

Restarts the streamed forward from a residual kept by qwen_forward.py (float16), runs the layers up to the
last selected one and stores, for every selected expert, act = silu(gate x) * (up x) on the native-regime
tokens routed to it (a sample of at most --max-tokens rows) together with the mean |act| per unit. The
restart is checked against the residuals the original run kept further up.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from titan.donor import qwen3moe as Q                                    # noqa: E402
from titan.donor.hf import HFRepo                                        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    ap.add_argument("--rev", default="b2cff646eb4bb1d68355c01b18ae02e7cf42d120")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--run", required=True, help="output directory of qwen_forward.py")
    ap.add_argument("--start-after", type=int, required=True, help="layer whose kept residual is the starting point")
    ap.add_argument("--selection", required=True)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    repo = HFRepo(a.repo, a.rev, cache=a.cache)
    cfg = Q.QwenMoeConfig.from_hf(json.loads(repo.read_bytes("config.json")))
    sel = {}
    for s in json.load(open(a.selection)):
        sel.setdefault(int(s["layer"]), []).append(int(s["expert"]))
    T = np.load(f"{a.run}/tokens.npz")
    NA = int(T["NA"])
    lens = [int(x) for x in T["lens"]]
    lens = lens[:len(lens) // 2]                             # native-regime sequences come first
    assert sum(lens) == NA
    fb = Q.FlatBatch(lens)
    h = torch.from_numpy(np.load(f"{a.run}/h_after_L{a.start_after:02d}.f16.npy", mmap_mode="r")[:NA].astype(np.float32))
    cos, sin = Q.rope_tables(max(fb.lengths), cfg.head_dim, cfg.rope_theta)
    rng = np.random.default_rng(0)
    report = {"start_after": a.start_after, "restart_check": {}, "experts": []}
    for l in range(a.start_after + 1, max(sel) + 1):
        names = Q.layer_tensor_names(l, cfg.experts)
        W = Q.unpack_layer(repo.fetch_span(list(names.values()), threads=a.threads), names, cfg)
        with torch.no_grad():
            h = fb.attend(h, W, cfg, cos, sin, batch=8)
            if l in sel:
                xn = Q.rms_norm(h, W["post_ln"], cfg.eps)
                topi, _ = Q.route(xn, W, cfg)
                for e in sel[l]:
                    tok = torch.where((topi == e).any(1))[0].numpy()
                    n_all = len(tok)
                    if len(tok) > a.max_tokens:
                        tok = np.sort(rng.choice(tok, a.max_tokens, replace=False))
                    x = xn[torch.from_numpy(tok)]
                    act = F.silu(F.linear(x, W["e_gate"][e])) * F.linear(x, W["e_up"][e])
                    np.savez(f"{a.out}/act_L{l}_e{e}.npz", act=act.numpy().astype(np.float16),
                             mean_abs=act.abs().mean(0).numpy().astype(np.float32))
                    report["experts"].append({"layer": l, "expert": e, "routed_native_tokens": n_all, "rows_kept": int(len(tok))})
            h, _ = Q.moe(h, W, cfg)
        kept = f"{a.run}/h_after_L{l:02d}.f16.npy"
        if os.path.exists(kept):
            ref = torch.from_numpy(np.load(kept, mmap_mode="r")[:NA].astype(np.float32))
            report["restart_check"][str(l)] = float((h - ref).norm() / ref.norm())
        print(f"L{l} done", report["restart_check"].get(str(l), ""), flush=True)
        del W
    json.dump(report, open(f"{a.out}/report.json", "w"), indent=1)
    print(json.dumps(report["restart_check"]), len(report["experts"]), "experts")


if __name__ == "__main__":
    main()
