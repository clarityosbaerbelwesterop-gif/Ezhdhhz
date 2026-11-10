#!/usr/bin/env python3
"""Streamed Qwen3-MoE donor forward on calibration chunks, two input regimes in one pass:

  A  native     the donor reads the text with ITS tokenizer (in distribution)
  B  transplant the donor reads the SEED tokenization through vocabulary-transplanted embeddings and is
                scored through the vocabulary-transplanted head (what a vocab-transplant stitch feeds it)

Per layer: router statistics per expert (count, gate mass, output energy) for code/prose x two halves,
routing tables, residual scale, a logit-lens cross entropy on a fixed position sample; at the end the
full cross entropy and bits per byte of both regimes. Weights are read by HTTP range, one layer at a time.
Resumable (state after every layer).
"""
import argparse
import json
import os
import sys
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from titan.donor import qwen3moe as Q                                   # noqa: E402
from titan.donor import st                                              # noqa: E402
from titan.donor.hf import HFRepo                                       # noqa: E402
from titan.donor.vocab import HFByteLevelBPE, SeedBPE, transplant_rows, vocab_contract  # noqa: E402


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ce_rows(h, norm_w, head, bias, targets, eps, chunk=1024):
    out = []
    for i in range(0, h.shape[0], chunk):
        lg = F.linear(Q.rms_norm(h[i:i + chunk], norm_w, eps), head)
        if bias is not None:
            lg = lg + bias
        out.append(F.cross_entropy(lg.float(), targets[i:i + chunk], reduction="none"))
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    ap.add_argument("--rev", default="b2cff646eb4bb1d68355c01b18ae02e7cf42d120")
    ap.add_argument("--seed-tokenizer", required=True)
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--last-layer", type=int, default=-1)
    ap.add_argument("--keep", type=int, nargs="*", default=[12, 24, 28, 36, 42, 48])
    ap.add_argument("--lens-sample", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--attn-batch", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    repo = HFRepo(a.repo, a.rev, cache=a.cache)
    hc = json.loads(repo.read_bytes("config.json"))
    cfg = Q.QwenMoeConfig.from_hf(hc)
    last = cfg.layers - 1 if a.last_layer < 0 else a.last_layer
    seed = SeedBPE(a.seed_tokenizer)
    tok_path = os.path.join(a.cache, "tokenizer.json")
    if not os.path.exists(tok_path):
        open(tok_path, "wb").write(repo.read_bytes("tokenizer.json"))
    dt = HFByteLevelBPE(tok_path)
    chunks = json.load(open(a.chunks))
    K = len(chunks)
    contract = vocab_contract(seed, dt, special_map={s: hc["bos_token_id"] for s in seed.specials})
    json.dump({k: contract[k] for k in ("stats", "seed", "donor")}, open(f"{a.out}/vocab_contract_stats.json", "w"), indent=1)

    # ---- tokens of both regimes, byte offsets, groups
    idsA, endsA, idsB, endsB = [], [], [], []
    for c in chunks:
        i, e = dt.encode_with_offsets(c["text"])
        idsA.append(np.asarray(i, np.int64)); endsA.append(np.asarray(e, np.int64))
        s = np.asarray(seed.encode(c["text"]), np.int64)
        idsB.append(s); endsB.append(seed.end_offsets(s))
    lens = [len(x) for x in idsA] + [len(x) for x in idsB]
    fb = Q.FlatBatch(lens)
    NA = int(sum(lens[:K]))
    dom = np.array([0 if c["domain"] == "code" else 1 for c in chunks])
    half = np.array([c["half"] for c in chunks])
    g_seq = np.concatenate([dom * 2 + half, 4 + dom * 2 + half])
    groups = torch.from_numpy(g_seq)[fb.seq]
    flat_ids = torch.from_numpy(np.concatenate(idsA + idsB))
    # next-token targets inside each sequence (-1 for the last position)
    tgt = torch.full((fb.N,), -1, dtype=torch.long)
    not_last = fb.pos < torch.tensor(lens)[fb.seq] - 1
    tgt[not_last] = flat_ids[1:][not_last[:-1]] if fb.N > 1 else tgt[not_last]
    tgt_bytes = np.zeros(fb.N, np.int64)                              # bytes spelled by the target token
    for s_i, ends in enumerate(endsA + endsB):
        b = np.diff(ends)
        tgt_bytes[fb.starts[s_i]:fb.starts[s_i] + len(b)] = b
    np.savez(f"{a.out}/tokens.npz", lens=np.array(lens), flat_ids=flat_ids.numpy(), groups=groups.numpy(), NA=NA,
             tgt=tgt.numpy(), tgt_bytes=tgt_bytes, ends=np.concatenate(endsA + endsB))
    log(f"chunks {K}; tokens native {NA}, transplant {fb.N - NA}; layers 0..{last}")

    # ---- embeddings / head (cached to disk as bf16 payload)
    def cached_tensor(name):
        p = os.path.join(a.cache, name.replace("/", "__") + ".bf16")
        ref = repo.refs()[name]
        if os.path.exists(p) and os.path.getsize(p) == ref.nbytes:
            b = open(p, "rb").read()
        else:
            b = repo.fetch_span([name], threads=a.threads)[name]
            open(p, "wb").write(b)
        return ref, b

    prov = {}
    state_p = f"{a.out}/state.pt"
    ref_e, b = cached_tensor("model.embed_tokens.weight"); prov[ref_e.name] = st.sha256(b)
    table = st.bf16_to_f32(b).reshape(ref_e.shape); del b
    EB, unkE = transplant_rows(table, contract["map"])
    np.save(f"{a.out}/transplant_embed.npy", EB)
    if os.path.exists(state_p):
        stt = torch.load(state_p)
        h, start = stt["h"], stt["layer"] + 1
        log(f"RESUME after layer {stt['layer']}")
    else:
        h = torch.from_numpy(np.concatenate([table[np.concatenate(idsA)], EB[np.concatenate(idsB)]]).astype(np.float32))
        start = 0
    del table
    ref_h, b = cached_tensor("lm_head.weight"); prov[ref_h.name] = st.sha256(b)
    head = torch.from_numpy(st.bf16_to_f32(b).reshape(ref_h.shape).copy()); del b
    HB, unkH = transplant_rows(head.numpy(), contract["map"])
    np.save(f"{a.out}/transplant_head.npy", HB)
    HB = torch.from_numpy(HB)
    biasB = torch.zeros(HB.shape[0]); biasB[unkH] = -30.0
    ref_n, b = cached_tensor("model.norm.weight"); prov[ref_n.name] = st.sha256(b)
    norm_w = torch.from_numpy(st.bf16_to_f32(b).copy())
    json.dump({"tensor_sha256": prov, "unknown_embed_rows": unkE, "unknown_head_rows": unkH}, open(f"{a.out}/fetched_global.json", "w"), indent=1)

    # ---- fixed lens samples
    rng = np.random.default_rng(0)
    okA = np.where((tgt.numpy() >= 0) & (np.arange(fb.N) < NA))[0]
    okB = np.where((tgt.numpy() >= 0) & (np.arange(fb.N) >= NA))[0]
    sA = torch.from_numpy(np.sort(rng.choice(okA, min(a.lens_sample, len(okA)), replace=False)))
    sB = torch.from_numpy(np.sort(rng.choice(okB, min(a.lens_sample, len(okB)), replace=False)))

    def lens(hh):
        with torch.no_grad():
            ca = ce_rows(hh[sA], norm_w, head, None, tgt[sA], cfg.eps)
            cb = ce_rows(hh[sB], norm_w, HB, biasB, tgt[sB], cfg.eps)
        r = {}
        for nm, c, s in (("A", ca, sA), ("B", cb, sB)):
            d = (groups[s] % 4) // 2
            r[nm] = {"all": float(c.mean()), "code": float(c[d == 0].mean()), "prose": float(c[d == 1].mean())}
        return r

    cos, sin = Q.rope_tables(max(fb.lengths), cfg.head_dim, cfg.rope_theta)
    records = json.load(open(f"{a.out}/layers.json")) if os.path.exists(f"{a.out}/layers.json") and start else []
    if start == 0:
        records.append({"layer": -1, "lens": lens(h), "rms": float(h.pow(2).mean().sqrt())})

    # ---- layer loop with prefetch
    box = {}

    def fetch(l):
        try:
            names = Q.layer_tensor_names(l, cfg.experts)
            t0 = time.time()
            blobs = repo.fetch_span(list(names.values()), threads=a.threads)
            box[l] = (names, blobs, time.time() - t0)
        except Exception as e:                                         # surfaced by the main thread
            box[l] = e

    th = threading.Thread(target=fetch, args=(start,)); th.start()
    t_all = time.time()
    for l in range(start, last + 1):
        th.join()
        if isinstance(box[l], Exception):
            raise box[l]
        names, blobs, t_dl = box.pop(l)
        if l + 1 <= last:
            th = threading.Thread(target=fetch, args=(l + 1,)); th.start()
        t0 = time.time()
        sha = {n: st.sha256(blobs[n]) for n in (names["gate"], names["in_ln"], names["post_ln"])}
        W = Q.unpack_layer(blobs, names, cfg); del blobs
        with torch.no_grad():
            h = fb.attend(h, W, cfg, cos, sin, batch=a.attn_batch)
            h, info = Q.moe(h, W, cfg, groups=groups, n_groups=8)
        np.savez(f"{a.out}/route_L{l:02d}.npz", topi=info["topi"].numpy().astype(np.uint8),
                 topw=info["topw"].numpy().astype(np.float16), **{k: v.numpy() for k, v in info["stats"].items()},
                 out_energy=np.array([float(info["out_energy"][groups == g].sum()) for g in range(8)]),
                 in_energy=np.array([float(info["in_energy"][groups == g].sum()) for g in range(8)]),
                 n_tok=np.bincount(groups.numpy(), minlength=8))
        rec = {"layer": l, "lens": lens(h), "rms": float(h.pow(2).mean().sqrt()), "download_s": round(t_dl, 1),
               "compute_s": round(time.time() - t0, 1), "sha256": sha}
        records.append(rec)
        if (l + 1) in a.keep:
            np.save(f"{a.out}/h_after_L{l:02d}.f16.npy", h.numpy().astype(np.float16))
        del W, info
        torch.save({"h": h, "layer": l}, state_p + ".tmp"); os.replace(state_p + ".tmp", state_p)
        json.dump(records, open(f"{a.out}/layers.json", "w"))
        el = time.time() - t_all
        log(f"L{l:02d} dl {t_dl:.0f}s compute {rec['compute_s']:.0f}s | lens CE native {rec['lens']['A']['all']:.3f} "
            f"transplant {rec['lens']['B']['all']:.3f} | rms {rec['rms']:.2f} | eta {el / (l - start + 1) * (last - l) / 60:.0f} min")

    # ---- final losses, bits per byte
    with torch.no_grad():
        res = {}
        for nm, lo, hi, hd, bs in (("native", 0, NA, head, None), ("transplant", NA, fb.N, HB, biasB)):
            idx = torch.arange(lo, hi)[tgt[lo:hi] >= 0]
            ce = ce_rows(h[idx], norm_w, hd, bs, tgt[idx], cfg.eps)
            nb = torch.from_numpy(tgt_bytes)[idx]
            d = (groups[idx] % 4) // 2
            res[nm] = {"tokens": int(len(idx)), "bytes": int(nb.sum()), "mean_ce_nats": float(ce.mean()),
                       "bits_per_byte": float(ce.sum() / nb.sum() / np.log(2))}
            for k, name in ((0, "code"), (1, "prose")):
                m = d == k
                res[nm][name] = {"mean_ce_nats": float(ce[m].mean()), "bits_per_byte": float(ce[m].sum() / nb[m].sum() / np.log(2))}
    res.update(repo=a.repo, revision=repo.revision, last_layer=last, bytes_fetched=repo.bytes_fetched,
               wall_s=round(time.time() - t_all, 1))
    json.dump(res, open(f"{a.out}/final.json", "w"), indent=1)
    log("FINAL", json.dumps(res))


if __name__ == "__main__":
    main()
