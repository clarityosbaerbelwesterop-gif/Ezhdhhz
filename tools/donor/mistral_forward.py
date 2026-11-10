#!/usr/bin/env python3
"""Streamed Mistral Large 4 forward on a labelled probe set (tools/donor/mistral_probes.py).

Weights are read by HTTP range; routed experts are fetched one at a time (FP8 + block scales -> float32),
applied to the tokens routed to them and discarded, so memory stays at a few GB although a layer holds
14.8 GB of experts. Per layer the routing table of every token (expert ids, gate weights, norm of each
expert's contribution) is stored; contrasts between token groups are computed offline.
Resumable: state after every layer.
"""
import argparse
import json
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from titan.donor import mistral_l4 as M                                  # noqa: E402
from titan.donor import st                                               # noqa: E402
from titan.donor.hf import HFRepo                                        # noqa: E402

FP8 = "F8_E4M3"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def common_suffix(a, b) -> int:
    k = 0
    while k < min(len(a), len(b)) and a[-1 - k] == b[-1 - k]:
        k += 1
    return k


def paired_positions(seqs, starts) -> dict:
    """Chat probes come as the same conversation under three reasoning efforts, with and without a thinking
    block. Sequences that differ only in the header share a token suffix; the flat indices of that suffix
    are returned as pairs (a, b): the same token at the same distance from the end in two contexts."""
    by = {}
    for n, q in enumerate(seqs):
        if q.get("effort", 0):
            by[(q["item"], q["think"], q["effort"])] = n
    out = {"high_low": ([], []), "medium_low": ([], []), "high_medium": ([], []), "think_nothink": ([], [])}

    def add(key, na, nb):
        k = common_suffix(seqs[na]["ids"], seqs[nb]["ids"])
        ea, eb = starts[na] + len(seqs[na]["ids"]), starts[nb] + len(seqs[nb]["ids"])
        out[key][0].extend(range(ea - k, ea))
        out[key][1].extend(range(eb - k, eb))

    for (item, think, effort), n in sorted(by.items()):
        if effort == 3 and (item, think, 1) in by:
            add("high_low", n, by[(item, think, 1)])
        if effort == 2 and (item, think, 1) in by:
            add("medium_low", n, by[(item, think, 1)])
        if effort == 3 and (item, think, 2) in by:
            add("high_medium", n, by[(item, think, 2)])
        if think == 1 and (item, 0, effort) in by:
            add("think_nothink", n, by[(item, 0, effort)])
    return {k: (np.array(a, np.int64), np.array(b, np.int64)) for k, (a, b) in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="mistralai/Mistral-Large-4-1T-A52B")
    ap.add_argument("--rev", default="af26e7869af601441d5cdf75a1272469ddc3d059")
    ap.add_argument("--probes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--last-layer", type=int, default=-1)
    ap.add_argument("--max-seqs", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=0, help="truncate every sequence (pipeline tests)")
    ap.add_argument("--lens-sample", type=int, default=2048)
    ap.add_argument("--xn-sample", type=int, default=3072, help="expert inputs kept per layer (plus all ICL tokens)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--lookahead", type=int, default=6)
    ap.add_argument("--attn-batch", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    repo = HFRepo(a.repo, a.rev, cache=a.cache)
    refs = repo.refs()
    hc = json.loads(repo.read_bytes("config.json"))
    cfg = M.ML4Config.from_hf_text(hc["text_config"])
    last = cfg.layers - 1 if a.last_layer < 0 else a.last_layer
    P = json.load(open(a.probes))
    seqs = P["seqs"][:a.max_seqs] if a.max_seqs else P["seqs"]
    if a.max_tokens:
        for s in seqs:
            for k in ("ids", "seg", "shot"):
                s[k] = s[k][:a.max_tokens]
    lens = [len(s["ids"]) for s in seqs]
    fb = M.FlatBatch(lens)
    ids = torch.tensor([t for s in seqs for t in s["ids"]])
    tgt = torch.full((fb.N,), -1, dtype=torch.long)
    not_last = fb.pos < torch.tensor(lens)[fb.seq] - 1
    tgt[not_last] = ids[1:][not_last[:-1]]
    seg_all = np.array([g for q in seqs for g in q["seg"]])
    shot_all = np.array([g for q in seqs for g in q["shot"]])
    nl = not_last.numpy()
    tseg = np.full(fb.N, -1)
    tshot = np.full(fb.N, -1)
    tseg[nl] = seg_all[1:][nl[:-1]]                         # labels of the token being predicted
    tshot[nl] = shot_all[1:][nl[:-1]]
    pairs = paired_positions(seqs, fb.starts)
    np.savez(f"{a.out}/pairs.npz", **{f"{k}_{x}": v[i] for k, v in pairs.items() for i, x in enumerate("ab")})
    np.savez(f"{a.out}/tokens.npz", lens=np.array(lens), ids=ids.numpy(), tgt=tgt.numpy(), tseg=tseg, tshot=tshot,
             seq=fb.seq.numpy(), pos=fb.pos.numpy(),
             kind=np.array([s["kind"] for s in seqs])[fb.seq.numpy()],
             item=np.array([s["item"] for s in seqs])[fb.seq.numpy()],
             effort=np.array([s["effort"] for s in seqs])[fb.seq.numpy()],
             think=np.array([s["think"] for s in seqs])[fb.seq.numpy()],
             seg=seg_all, shot=shot_all)
    log(f"sequences {len(seqs)}, tokens {fb.N}, layers 0..{last}, {M.flops_per_token(cfg) * fb.N / 1e12:.1f} TFLOP per sparse layer")

    def bf16(name):
        r, b = repo.fetch(name)
        return torch.from_numpy(st.bf16_to_f32(b).reshape(r.shape).copy())

    def cached(name):
        p = os.path.join(a.cache, name + ".bf16")
        r = refs[name]
        if not (os.path.exists(p) and os.path.getsize(p) == r.nbytes):
            open(p, "wb").write(repo.fetch_span([name], threads=8)[name])
        return np.memmap(p, dtype="<u2", mode="r").reshape(r.shape)

    # rotary tables: YaRN frequencies from the checkpoint config, LeRoPE scaling from the checkpoint
    inv_freq, attn_scaling = M.rope_frequencies(hc["text_config"])
    cos, sin = M.rope_tables(max(lens), inv_freq, bf16("lerope.alpha"), attn_scaling)

    state_p = f"{a.out}/state.pt"
    if os.path.exists(state_p):
        stt = torch.load(state_p)
        h, shared, start = stt["h"], stt["shared"], stt["layer"] + 1
        records = json.load(open(f"{a.out}/layers.json"))
        log(f"RESUME after layer {stt['layer']}")
    else:
        emb = cached("tok_embeddings.weight")
        uniq, inv = np.unique(ids.numpy(), return_inverse=True)
        rows = st.bf16_to_f32(np.ascontiguousarray(emb[uniq])).reshape(len(uniq), cfg.hidden)
        h = torch.from_numpy(rows[inv].copy())
        del rows
        del emb
        shared, start, records = {}, 0, []
    norm_w = bf16("norm.weight")
    head = None                                             # output matrix stays BF16 on disk (memory map);
    if a.lens_sample:                                       # float32 blocks of rows are made on the fly
        head = cached("output.weight")
    rng = np.random.default_rng(0)
    ok = np.where(tgt.numpy() >= 0)[0]
    first = ok[np.isin(tseg[ok], (3, 5))]                   # every position predicting an answer / ICL output token
    rest = np.setdiff1d(ok, first)
    samp = torch.from_numpy(np.sort(np.concatenate([first, rng.choice(rest, max(0, min((a.lens_sample or 1) - len(first), len(rest))), replace=False)])))
    kind_all = np.array([q["kind"] for q in seqs])[fb.seq.numpy()]
    icl = np.where(kind_all == 4)[0]
    other = np.setdiff1d(np.arange(fb.N), icl)
    xs = torch.from_numpy(np.sort(np.concatenate([icl, np.random.default_rng(1).choice(other, min(a.xn_sample, len(other)), replace=False)])))
    np.savez(f"{a.out}/samples.npz", lens_idx=samp.numpy(), xn_idx=xs.numpy())

    def ce(hh, idx, chunk=2048, vblock=25600):
        out = []
        for i in range(0, len(idx), chunk):
            j = idx[i:i + chunk]
            x = M.rms_norm(hh[j], norm_w, cfg.eps)
            lg = torch.empty(len(j), cfg.vocab)
            for v0 in range(0, cfg.vocab, vblock):
                w = torch.from_numpy(st.bf16_to_f32(np.ascontiguousarray(head[v0:v0 + vblock])).reshape(-1, cfg.hidden))
                lg[:, v0:v0 + vblock] = x @ w.T
                del w
            out.append(F.cross_entropy(M.softcap_logits(lg, cfg.softcap), tgt[j], reduction="none"))
            del lg
        return torch.cat(out)

    def weight(blobs, name):
        r = refs[name]
        if r.dtype == FP8:
            return M.dequant(blobs[name], r.shape, blobs[M.scale_name(name)])
        return torch.from_numpy(st.bf16_to_f32(blobs[name]).reshape(r.shape).copy())

    def with_scales(names):
        return [n for n in names] + [M.scale_name(n) for n in names if refs[n].dtype == FP8]

    def fetch_expert(layer, e):
        names = list(M.expert_tensor_names(layer, e).values())
        return repo.fetch_span(with_scales(names), threads=3)

    t_all = time.time()
    pool = ThreadPoolExecutor(a.workers)
    for l in range(start, last + 1):
        t0 = time.time()
        b0 = repo.bytes_fetched
        names = M.layer_tensor_names(l, cfg)
        blobs = repo.fetch_span(with_scales(list(names.values())), threads=6)
        W = {k: weight(blobs, n) for k, n in names.items()}
        del blobs
        with torch.no_grad():
            h_in = h
            h, keys = fb.attend(h, W, cfg, l, cos, sin, shared.get(cfg.kv_source[l]), batch=a.attn_batch)
            if keys is not None:
                shared[l] = keys
            attn_norm = (h - h_in).norm(dim=-1).numpy().astype(np.float16)
            del h_in
            h_mid = h
            xn = M.rms_norm(h, W["post_ln"], cfg.eps)
            rec = {"layer": l, "type": cfg.mlp_types[l]}
            if cfg.mlp_types[l] == "dense":
                h = h + M.swiglu(xn, W["dense.w1"], W["dense.w3"], W["dense.w2"])
            else:
                topi, topw = M.route(xn, W["gate"], cfg)
                shared_out = M.swiglu(xn, W["shared.w1"], W["shared.w3"], W["shared.w2"])
                out = shared_out.clone()
                contrib = torch.zeros(fb.N, cfg.top_k)
                unit_act = np.zeros((cfg.experts, cfg.moe_inter), np.float16)      # mean |silu(w1 x) * (w3 x)| per unit
                unit_rms = np.zeros((cfg.experts, cfg.moe_inter), np.float16)      # root mean square of the same
                np.save(f"{a.out}/xn_L{l:02d}.f16.npy", xn[xs].numpy().astype(np.float16))
                hit = torch.unique(topi).tolist()
                q, it = deque(), iter(hit)
                for e in [x for _, x in zip(range(a.lookahead), it)]:
                    q.append((e, pool.submit(fetch_expert, l, e)))
                while q:
                    e, fut = q.popleft()
                    eb = fut.result()
                    nxt = next(it, None)
                    if nxt is not None:
                        q.append((nxt, pool.submit(fetch_expert, l, nxt)))
                    en = M.expert_tensor_names(l, e)
                    w1, w2, w3 = (M.dequant(eb[en[k]], refs[en[k]].shape, eb[M.scale_name(en[k])]) for k in ("w1", "w2", "w3"))
                    del eb
                    tok, slot = torch.where(topi == e)
                    act = F.silu(F.linear(xn[tok], w1)) * F.linear(xn[tok], w3)
                    unit_act[e] = act.abs().mean(0).numpy().astype(np.float16)
                    unit_rms[e] = act.pow(2).mean(0).sqrt().numpy().astype(np.float16)
                    y = F.linear(act, w2) * topw[tok, slot, None]
                    del act
                    out.index_add_(0, tok, y)
                    contrib[tok, slot] = y.norm(dim=-1)
                    del w1, w2, w3, y
                h = h + out
                np.savez(f"{a.out}/route_L{l:02d}.npz", topi=topi.numpy().astype(np.uint8), topw=topw.numpy().astype(np.float16),
                         contrib=contrib.numpy().astype(np.float16), unit_act=unit_act, unit_rms=unit_rms,
                         shared_norm=shared_out.norm(dim=-1).numpy().astype(np.float16),
                         out_norm=out.norm(dim=-1).numpy().astype(np.float16), in_norm=xn.norm(dim=-1).numpy().astype(np.float16))
                rec["experts_hit"] = len(hit)
            tok_rec = {"attn_norm": attn_norm, "mlp_norm": (h - h_mid).norm(dim=-1).numpy().astype(np.float16),
                       "h_norm": h.norm(dim=-1).numpy().astype(np.float16)}
            del h_mid
            for k, (ia, ib) in pairs.items():                # same token, two contexts: how far apart is the residual
                if len(ia):
                    ha, hb = h[torch.from_numpy(ia)], h[torch.from_numpy(ib)]
                    tok_rec[f"rel_{k}"] = ((ha - hb).norm(dim=-1) / hb.norm(dim=-1)).numpy().astype(np.float16)
                    tok_rec[f"cos_{k}"] = F.cosine_similarity(ha, hb, dim=-1).numpy().astype(np.float32)
                    rec[f"rel_{k}"] = float(tok_rec[f"rel_{k}"].astype(np.float32).mean())
            if head is not None:
                c = ce(h, samp)
                rec["lens_ce"] = float(c.mean())
                tok_rec["lens_ce"] = c.numpy().astype(np.float32)
            np.savez(f"{a.out}/tok_L{l:02d}.npz", **tok_rec)
        rec.update(rms=float(h.pow(2).mean().sqrt()), seconds=round(time.time() - t0, 1), bytes=repo.bytes_fetched - b0,
                   finite=bool(torch.isfinite(h).all()))
        records.append(rec)
        del W
        torch.save({"h": h, "shared": shared, "layer": l}, state_p + ".tmp")
        os.replace(state_p + ".tmp", state_p)
        json.dump(records, open(f"{a.out}/layers.json", "w"))
        el = time.time() - t_all
        log(f"L{l:02d} {rec['type']} {rec['seconds']:.0f}s {rec['bytes'] / 1e9:.2f} GB ({rec['bytes'] / 1e6 / max(rec['seconds'], 1e-9):.0f} MB/s)"
            f" experts {rec.get('experts_hit', '-')} | lens CE {rec.get('lens_ce', float('nan')):.3f} | rms {rec['rms']:.3f}"
            f" | hi/lo {rec.get('rel_high_low', float('nan')):.4f} | eta {el / (l - start + 1) * (last - l) / 3600:.2f} h")
    if head is not None:
        with torch.no_grad():
            idx = torch.from_numpy(ok)
            c = ce(h, idx)
        kind = np.load(f"{a.out}/tokens.npz")["kind"][ok]
        res = {"tokens": int(len(idx)), "mean_ce_nats": float(c.mean()),
               "by_kind": {P["meta"]["kinds"][k]: float(c[torch.from_numpy(kind == k)].mean()) for k in sorted(set(kind.tolist()))},
               "repo": a.repo, "revision": repo.revision, "last_layer": last, "bytes_fetched": repo.bytes_fetched,
               "wall_s": round(time.time() - t_all, 1)}
        np.save(f"{a.out}/final_ce.npy", c.numpy())
        json.dump(res, open(f"{a.out}/final.json", "w"), indent=1)
        log("FINAL", json.dumps(res))


if __name__ == "__main__":
    main()
