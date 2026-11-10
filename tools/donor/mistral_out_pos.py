#!/usr/bin/env python3
"""Positions at which a Mistral residual can be scored in the seed vocabulary.

For the plain-text probes (prose, code, worked problems, in-context tasks) the text is spelled back from the
donor tokens, tokenised with the seed tokenizer, and every byte offset at which both tokenisations have a
token boundary gives one pair: the donor position that has read the text up to that offset, and the seed
token that starts there (the target). Output for tools/donor/output_bridge.py: pos, tgt, nbytes, fit
(fit = sequences with an even item number; the other half is held out).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from titan.donor.vocab import SeedBPE, Tekken                             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="output directory of mistral_forward.py (tokens.npz)")
    ap.add_argument("--tekken", required=True)
    ap.add_argument("--hf-tokenizer", default=None)
    ap.add_argument("--seed-tokenizer", required=True)
    ap.add_argument("--bos", type=int, default=1)
    ap.add_argument("--kinds", type=int, nargs="*", default=[0, 1, 2, 4], help="plain-text probe kinds")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    T = np.load(f"{a.run}/tokens.npz")
    tk, seed = Tekken(a.tekken, hf_tokenizer=a.hf_tokenizer), SeedBPE(a.seed_tokenizer)
    lens, ids, kind, item = T["lens"], T["ids"], T["kind"], T["item"]
    starts = np.concatenate([[0], np.cumsum(lens)])
    pos, tgt, nb, fit, knd = [], [], [], [], []
    stats = {"sequences": 0, "seed_tokens": 0, "bytes": 0}
    for s in range(len(lens)):
        a0, b0 = int(starts[s]), int(starts[s + 1])
        if int(kind[a0]) not in a.kinds:
            continue
        seq = ids[a0:b0]
        off = 1 if seq[0] == a.bos else 0
        pieces = [tk.token_bytes(int(t)) for t in seq[off:]]
        text = b"".join(pieces).decode("utf-8")
        d_ends = np.cumsum([len(p) for p in pieces])             # donor position a0+off+i has read d_ends[i] bytes
        sid = np.asarray(seed.encode(text), np.int64)
        s_ends = seed.end_offsets(sid)
        assert int(s_ends[-1]) == int(d_ends[-1]) == len(text.encode("utf-8"))
        donor_at = {int(b): a0 + off + i for i, b in enumerate(d_ends)}
        if off:
            donor_at[0] = a0                                     # the start token has read nothing
        s_start = np.concatenate([[0], s_ends[:-1]])
        for j in range(len(sid)):
            p = donor_at.get(int(s_start[j]))
            if p is not None:
                pos.append(p); tgt.append(int(sid[j])); nb.append(int(s_ends[j] - s_start[j]))
                fit.append(int(item[a0]) % 2 == 0); knd.append(int(kind[a0]))
        stats["sequences"] += 1
        stats["seed_tokens"] += len(sid)
        stats["bytes"] += int(s_ends[-1])
    np.savez(a.out, pos=np.array(pos), tgt=np.array(tgt), nbytes=np.array(nb), fit=np.array(fit), kind=np.array(knd))
    stats.update(aligned=len(pos), aligned_share=len(pos) / max(stats["seed_tokens"], 1), fit=int(np.sum(fit)), holdout=int(len(fit) - np.sum(fit)))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
