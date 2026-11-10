#!/usr/bin/env python3
"""Output bridge donor -> seed, measured on real donor activations through the seed's REAL output path
(final RMSNorm gain + tied token embedding); the seed body is not needed.

  reference   donor final norm + vocabulary-transplanted donor head (seed vocabulary)
  anchor      data-free: P with P @ E_seed' ~ H_donor' (least squares over the vocabulary), then the seed's
              final norm and tied embedding                                   [PROJECTION]
  healed      the same linear map trained on donor activations to predict the next seed token through the
              frozen seed output path; fitted on one half of the calibration chunks, scored on the other
              [TRAINING/HEAL of the bridge only; seed tensors are read-only]
Positions: byte-aligned boundaries of the donor's native tokenization and the seed tokenization.
"""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", required=True, help="donor final residuals [N, d_d] (float16 .npy)")
    ap.add_argument("--pos", required=True, help=".npz with pos (rows of h), tgt (seed ids), nbytes, fit (bool)")
    ap.add_argument("--donor-norm", required=True, help=".npy donor final norm weight")
    ap.add_argument("--donor-eps", type=float, default=1e-6)
    ap.add_argument("--donor-head-rows", required=True, help=".npy [V_seed, d_d] vocabulary-transplanted head")
    ap.add_argument("--seed-emb", required=True)
    ap.add_argument("--seed-norm", required=True, help=".npy seed final norm weight")
    ap.add_argument("--seed-eps", type=float, default=9.999999747378752e-06)
    ap.add_argument("--softcap", type=float, default=0.0)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    z = np.load(a.pos)
    pos, tgt, nb, fit = z["pos"], torch.from_numpy(z["tgt"]), z["nbytes"].astype(np.float64), z["fit"]
    h = torch.from_numpy(np.load(a.h, mmap_mode="r")[pos].astype(np.float32))
    wd = torch.from_numpy(np.load(a.donor_norm).astype(np.float32))
    r = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + a.donor_eps) * wd          # donor pre-head vector
    HB = torch.from_numpy(np.load(a.donor_head_rows).astype(np.float32))
    Es = torch.from_numpy(np.load(a.seed_emb).astype(np.float32))
    gs = torch.from_numpy(np.load(a.seed_norm).astype(np.float32))
    ev = torch.from_numpy(~fit)

    def seed_head(x):                                          # the seed's output path: RMSNorm * gain, tied embedding
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + a.seed_eps) * gs) @ Es.T

    def score(logits_fn, mask):
        ce = []
        idx = torch.where(mask)[0]
        with torch.no_grad():
            for i in range(0, len(idx), 4096):
                j = idx[i:i + 4096]
                ce.append(F.cross_entropy(logits_fn(r[j]), tgt[j], reduction="none"))
        ce = torch.cat(ce).double().numpy()
        return {"ce_nats": float(ce.mean()), "bits_per_byte": float(ce.sum() / nb[mask.numpy()].sum() / np.log(2)), "n": int(len(ce))}

    cap = (lambda lg: torch.tanh(lg / a.softcap) * a.softcap) if a.softcap else (lambda lg: lg)
    res = {"positions": int(len(pos)), "fit_positions": int(fit.sum()), "eval_positions": int((~fit).sum()),
           "uniform_ce_nats": float(np.log(Es.shape[0]))}
    res["reference_transplanted_head"] = score(lambda x: cap(x @ HB.T), ev)
    # anchor: P @ Es' ~ HB'  (ridge over the vocabulary), expressed for the seed's pre-norm space: divide by the gain
    A = Es.double()
    lam = a.ridge * float(torch.trace(A.T @ A)) / A.shape[1]
    W = torch.linalg.solve(A.T @ A + lam * torch.eye(A.shape[1], dtype=torch.float64), A.T @ HB.double())   # Es @ W ~ HB
    P_anchor = (W.T / gs.double()[None, :]).float()                                                            # [d_d, d_s]
    res["anchor_r2_over_vocabulary"] = float(1 - ((A @ W - HB.double()) ** 2).sum() / ((HB.double() - HB.double().mean(0)) ** 2).sum())
    res["anchor_through_seed_output_path"] = score(lambda x: seed_head(x @ P_anchor), ev)
    res["anchor_without_seed_norm"] = score(lambda x: (x @ (P_anchor * gs[None, :])) @ Es.T, ev)
    # healed bridge: train P (and nothing else) on the fit half
    P = torch.nn.Parameter(P_anchor.clone())
    opt = torch.optim.Adam([P], lr=a.lr)
    fi = torch.where(torch.from_numpy(fit))[0]
    hist = []
    for step in range(a.steps):
        j = fi[torch.randint(0, len(fi), (a.batch,))]
        loss = F.cross_entropy(seed_head(r[j] @ P), tgt[j])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == a.steps - 1:
            hist.append((step, float(loss)))
    res["healed_through_seed_output_path"] = score(lambda x: seed_head(x @ P.detach()), ev)
    res["healed_fit_half"] = score(lambda x: seed_head(x @ P.detach()), torch.from_numpy(fit))
    res["heal"] = {"steps": a.steps, "batch": a.batch, "lr": a.lr, "trainable_parameters": int(P.numel()), "loss_history": hist,
                   "note": "bridge only; seed embedding and final norm frozen and read-only"}
    json.dump(res, open(a.out, "w"), indent=1)
    for k in ("reference_transplanted_head", "anchor_through_seed_output_path", "anchor_without_seed_norm",
              "healed_through_seed_output_path", "healed_fit_half"):
        print(f"{k:36s} CE {res[k]['ce_nats']:.4f} nats  {res[k]['bits_per_byte']:.4f} bits/byte  (n={res[k]['n']})")
    print("anchor R2 over the vocabulary", round(res["anchor_r2_over_vocabulary"], 4), "| uniform CE", round(res["uniform_ce_nats"], 3))


if __name__ == "__main__":
    main()
