"""Model transplantation: join donor models with different tokenizers, widths and architectures.

    titan transplant --donors runA runB --layers 0:0-3 1:2-5 --calib text.txt --heal-steps 300 --out frank

1. Vocabulary transplant: every donor's token embedding and LM head are re-indexed to ONE target
   tokenizer. A target token whose bytes are also a single donor token is copied exactly; otherwise its
   row is the mean of the donor rows of the token's byte decomposition (donor tokenizer); tokens the
   donor cannot express get the donor's mean row. Coverage is reported.
2. Stitching: segments keep their donor's architecture exactly (StitchedLM); between segments a
   linear bridge maps the residual stream into the next donor's width.
3. Bridge fit: both donors run the SAME target-tokenized calibration text; the bridge is the ridge
   least-squares map from donor A's residual stream after its last transplanted block to donor B's
   residual stream before its first transplanted block (closed form; R^2 reported).
4. Healing: train the bridges only (everything else frozen), then optionally everything at a lower
   learning rate, on the calibration text.
5. Evaluation in bits per byte on held-out text, the tokenizer-independent measure, for every donor
   (with its own tokenizer), the stitched model before healing and after each healing phase.
"""
from __future__ import annotations

import json
import math
from dataclasses import fields
from pathlib import Path

import numpy as np

from .compiler import CompileOptions, Graph, compile_graph
from .config import load_config
from .dtypes import DType
from .nn.layers import ParamStore
from .nn.stitch import SEGMENT_KEYS, StitchConfig, StitchedLM
from .nn.transformer import TransformerConfig, TransformerLM
from .train.tokenizers import BPETokenizer, ByteTokenizer, CharTokenizer, load_tokenizer, tokenizer_from_description


# ------------------------------------------------------------------ tokens as bytes
def token_bytes(tok, i: int) -> bytes:
    if isinstance(tok, BPETokenizer):
        return tok.pieces[i]
    if isinstance(tok, ByteTokenizer):
        return bytes([i])
    if isinstance(tok, CharTokenizer):
        return tok.chars[i].encode("utf-8")
    return tok.decode([i]).encode("utf-8")


def encode_bytes(tok, b: bytes) -> list[int]:
    """Donor ids of a byte string (exact for byte-level tokenizers)."""
    if isinstance(tok, ByteTokenizer):
        return list(b)
    if isinstance(tok, BPETokenizer):
        s = list(b)
        while len(s) > 1:
            best = min(((tok.ranks.get(p, 1 << 30), i) for i, p in enumerate(zip(s, s[1:]))))
            if best[0] == 1 << 30:
                break
            r, i = best
            s = s[:i] + [256 + r] + s[i + 2:]
        return s
    text = b.decode("utf-8", errors="ignore")
    if isinstance(tok, CharTokenizer):
        return [tok.index[ch] for ch in text if ch in tok.index]
    try:
        return list(tok.encode(text))
    except Exception:
        return []


def vocab_map(src, dst) -> tuple[list[list[int]], dict]:
    """For every target token: the donor ids composing it (exact single-id matches first)."""
    exact_index = {token_bytes(src, i): i for i in range(src.vocab_size)}
    rows, stats = [], {"exact": 0, "composed": 0, "unknown": 0}
    for t in range(dst.vocab_size):
        b = token_bytes(dst, t)
        if b in exact_index:
            rows.append([exact_index[b]])
            stats["exact"] += 1
            continue
        ids = encode_bytes(src, b) if b else []
        rows.append(ids)
        stats["composed" if ids else "unknown"] += 1
    return rows, stats


def transplant_rows(table: np.ndarray, rows: list[list[int]], axis: int = 0) -> np.ndarray:
    """Re-index ``table`` (donor vocab along ``axis``) to the target vocabulary."""
    t = np.moveaxis(table, axis, 0)
    mean = t.mean(0)
    out = np.stack([t[r].mean(0) if r else mean for r in rows]).astype(np.float32)
    return np.moveaxis(out, 0, axis)


# ------------------------------------------------------------------ running graphs on given weights
def _run(build, weights: dict, feeds: dict, out: str, backend: str = "fast"):
    g = Graph("transplant")
    ps = ParamStore(g, 0, lazy=True)
    t = build(g, ps)
    g.output(out, t)
    exe = compile_graph(g, load_config("host"), CompileOptions(emit_isa=False))
    missing = [n for n in exe.params if n not in weights]
    if missing:
        raise ValueError(f"weights missing for {missing[:3]}")
    for n in exe.params:
        exe.lazy_init[n] = (lambda v=weights[n]: np.asarray(v, np.float32))
    if backend == "accel":
        from .runtime.accel import AccelDevice
        dev = AccelDevice("host")
    else:
        from .runtime import FastDevice
        dev = FastDevice("host")
    dev.load(exe)
    return dev, exe


def windows(ids: np.ndarray, seq: int, limit: int) -> np.ndarray:
    n = min(limit, (len(ids) - 1) // seq)
    if n < 1:
        raise ValueError(f"calibration text too short for sequence length {seq}")
    return np.stack([ids[i * seq:(i + 1) * seq + 1] for i in range(n)])


def bits_per_byte(model, weights, tok, text: str, batch: int = 8, max_windows: int = 64) -> float:
    """Tokenizer-independent loss: nats of every predicted token / bytes they cover / ln 2."""
    ids = np.asarray(tok.encode(text), dtype=np.int64)
    W = windows(ids, model.seq, max_windows)
    nb = {i: len(token_bytes(tok, i)) for i in set(W[:, 1:].ravel().tolist())}
    dev, exe = _run(lambda g, ps: model.build(g, ps, batch, DType.FP32)[1], weights,
                    {}, "logits")
    nats, nbytes = 0.0, 0
    for c in range(0, len(W), batch):
        chunk = W[c:c + batch]
        pad = np.concatenate([chunk, np.repeat(chunk[-1:], batch - len(chunk), 0)])
        z = dev.run(exe, {"x": pad[:, :-1].reshape(-1, 1)}, fetch=["logits"]).outputs["logits"].astype(np.float64)
        z = z.reshape(batch, model.seq, -1)[:len(chunk)]
        y = chunk[:, 1:]
        z = z - z.max(-1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
        nats -= np.take_along_axis(logp, y[..., None], -1).sum()
        nbytes += sum(nb[int(v)] for v in y.ravel())
    return float(nats / max(1, nbytes) / math.log(2))


def hidden(model: TransformerLM, weights, ids_windows: np.ndarray, upto: int, batch: int = 8) -> np.ndarray:
    dev, exe = _run(lambda g, ps: model.build_hidden(g, ps, batch, upto)[1], weights, {}, "h")
    hs = []
    for c in range(0, len(ids_windows), batch):
        chunk = ids_windows[c:c + batch, :model.seq]
        pad = np.concatenate([chunk, np.repeat(chunk[-1:], batch - len(chunk), 0)])
        h = dev.run(exe, {"x": pad.reshape(-1, 1)}, fetch=["h"]).outputs["h"]
        hs.append(h.reshape(batch, model.seq, -1)[:len(chunk)].reshape(-1, h.shape[-1]))
    return np.concatenate(hs).astype(np.float64)


def fit_bridge(ha: np.ndarray, hb: np.ndarray, ridge: float = 1e-3) -> tuple[np.ndarray, np.ndarray, float]:
    """Ridge least squares hb ~ ha @ W + b; returns (W [da, db], b [1, db], R^2)."""
    mu_a, mu_b = ha.mean(0), hb.mean(0)
    A, Bm = ha - mu_a, hb - mu_b
    lam = ridge * np.trace(A.T @ A) / A.shape[1]
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Bm)
    b = mu_b - mu_a @ W
    r2 = 1.0 - ((A @ W - Bm) ** 2).sum() / max((Bm ** 2).sum(), 1e-12)
    return W.astype(np.float32), b.reshape(1, -1).astype(np.float32), float(r2)


# ------------------------------------------------------------------ the transplant
def _donor(run):
    from .infer.generate import checkpoint_of
    run = Path(run)
    info = json.loads((run / "config.json").read_text())
    keys = {f.name for f in fields(TransformerConfig)}
    cfg = {k: v for k, v in info["model"].items() if k in keys}
    if info.get("model", {}).get("architecture") == "stitched" or not cfg:
        raise ValueError(f"{run}: donors must be transformer runs")
    with np.load(checkpoint_of(run)) as z:
        w = {k: z[k].astype(np.float32) for k in z.files if not k.endswith((".adam_m", ".adam_v", ".momentum"))}
    return cfg, w, tokenizer_from_description(info["tokenizer"])


def transplant(donors, layers, calib_text: str, target_tokenizer=0, out="runs/frank", heal_steps: int = 300,
               full_steps: int = 0, lr: float = 3e-3, batch: int = 8, calib_windows: int = 64, ridge: float = 1e-3,
               backend: str = "auto", seq: int | None = None, log=print) -> dict:
    """donors: run dirs; layers: [(donor, first, last)] in model order; target_tokenizer: donor index or spec."""
    log = log or (lambda *a, **k: None)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    D = [_donor(d) for d in donors]
    tok = D[target_tokenizer][2] if isinstance(target_tokenizer, int) else load_tokenizer(target_tokenizer)
    V = tok.vocab_size
    seq = seq or min(D[s][0]["seq"] for s, _, _ in layers)
    text = Path(calib_text).read_text(encoding="utf-8") if Path(str(calib_text)).is_file() else str(calib_text)
    cut = int(len(text) * 0.9)
    train_text, held_text = text[:cut], text[cut:]
    report = {"target_tokenizer": tok.describe(), "vocab": V, "segments": [], "bridges": []}
    # 1. vocabulary transplant of every donor
    adapted = []
    for k, (cfg, w, dtok) in enumerate(D):
        rows, stats = vocab_map(dtok, tok)
        name = cfg.get("name", "tlm")
        w2 = dict(w)
        w2[f"{name}.tok_emb"] = transplant_rows(w[f"{name}.tok_emb"], rows, 0)
        if f"{name}.head.w" in w:
            w2[f"{name}.head.w"] = transplant_rows(w[f"{name}.head.w"], rows, 1)
        if f"{name}.head.b" in w:
            w2[f"{name}.head.b"] = transplant_rows(w[f"{name}.head.b"], rows, 1)
        c2 = {**cfg, "vocab": V, "seq": min(seq, cfg["seq"]) if cfg.get("position") == "learned" else seq}
        if cfg.get("position") == "learned" and f"{name}.pos_emb" in w:
            w2[f"{name}.pos_emb"] = w[f"{name}.pos_emb"][:c2["seq"]]
        adapted.append((TransformerLM(TransformerConfig.from_dict(c2)), w2, name, [not r for r in rows]))
        report.setdefault("vocab_transplant", []).append({"donor": str(donors[k]), **stats})
        log(f"donor {k}: vocabulary transplant {stats}")
    # 2. stitched architecture and weights
    segs, W = [], {}
    sname = "frank"
    for s, (k, first, last) in enumerate(layers):
        cfg = D[k][0]
        segs.append({**{kk: v for kk, v in cfg.items() if kk in SEGMENT_KEYS}, "layers": last - first + 1,
                     "donor": str(donors[k]), "donor_layers": [first, last]})
        dname = adapted[k][2]
        for j, i in enumerate(range(first, last + 1)):
            pre = f"{dname}.b{i}."
            W.update({f"{sname}.s{s}.b{j}." + n[len(pre):]: v for n, v in adapted[k][1].items() if n.startswith(pre)})
        report["segments"].append({"donor": str(donors[k]), "layers": [first, last], "d_model": cfg["d_model"]})
    first_m, first_w, first_n, _ = adapted[layers[0][0]]
    last_m, last_w, last_n, unknown = adapted[layers[-1][0]]
    W[f"{sname}.tok_emb"] = first_w[f"{first_n}.tok_emb"]
    if first_m.config.position == "learned":
        W[f"{sname}.pos_emb"] = first_w[f"{first_n}.pos_emb"][:seq]
    for n in ("gamma", "beta"):
        if f"{last_n}.lnf.{n}" in last_w:
            W[f"{sname}.lnf.{n}"] = last_w[f"{last_n}.lnf.{n}"]
    W[f"{sname}.head.w"] = last_w[f"{last_n}.head.w"] if f"{last_n}.head.w" in last_w else last_w[f"{last_n}.tok_emb"].T.copy()
    hb = np.array(last_w.get(f"{last_n}.head.b", np.zeros((1, V), np.float32)), dtype=np.float32).reshape(1, V)
    hb[0, np.asarray(unknown, dtype=bool)] = -30.0      # tokens the head donor never had: ~zero probability, trainable
    W[f"{sname}.head.b"] = hb
    cfg = StitchConfig(vocab=V, seq=seq, segments=segs, head_bias=True, name=sname)
    model = StitchedLM(cfg)
    # 3. bridges by least squares on paired activations (same target-tokenized calibration windows)
    ids = np.asarray(tok.encode(train_text), dtype=np.int64)
    Wn = windows(ids, seq, calib_windows)
    for s in range(1, len(layers)):
        ka, _, la = layers[s - 1]
        kb, fb, _ = layers[s]
        ma, wa = adapted[ka][:2]
        mb, wb = adapted[kb][:2]
        ha = hidden(ma, wa, Wn, la + 1, batch)
        hb = hidden(mb, wb, Wn, fb, batch)
        Wb, bb, r2 = fit_bridge(ha, hb, ridge)
        W[f"{sname}.bridge{s}.w"], W[f"{sname}.bridge{s}.b"] = Wb, bb
        report["bridges"].append({"between": [s - 1, s], "shape": list(Wb.shape), "r2": round(r2, 4)})
        log(f"bridge {s}: {Wb.shape[0]} -> {Wb.shape[1]}, least-squares R^2 = {r2:.4f}")
    # 4. evaluation, healing
    bpb = {f"donor{k}": bits_per_byte(TransformerLM(TransformerConfig.from_dict(D[k][0])), D[k][1], D[k][2], held_text,
                                      batch) for k in sorted({k for k, _, _ in layers})}
    bpb["stitched"] = bits_per_byte(model, W, tok, held_text, batch)
    log(f"bits/byte on held-out text: {bpb}")
    from .merge import write_run
    run = write_run(out / "stitched", W, {}, tok.describe(), {"method": "transplant", "layers": layers},
                    model_dict=model.describe())
    (out / "train.txt").write_text(train_text, encoding="utf-8")
    phases = [("bridges", heal_steps, lr, [n for n in W if ".bridge" not in n])] if len(layers) > 1 else []
    phases += [("full", full_steps, lr / 3, [])] if full_steps else []
    cur = run
    for label, steps, plr, frozen in phases:
        if not steps:
            continue
        from .train import TrainConfig, Trainer
        tr = Trainer(TrainConfig(model="transformer", data=str(out / "train.txt"), tokenizer=str(run / "tokenizer.json"),
                                 model_config=str(run / "model.json"), init_from=str(cur / "checkpoints" / "best.npz"),
                                 freeze=frozen, steps=steps, lr=plr, batch=batch, eval_every=steps, ckpt_every=steps,
                                 backend=backend, out_dir=str(out / f"heal_{label}")), log=None)
        tr.run()
        tr.close()
        cur = tr.out
        with np.load(cur / "checkpoints" / "best.npz") as z:
            Wh = {k: z[k] for k in z.files if not k.endswith((".adam_m", ".adam_v"))}
        bpb[f"after_{label}"] = bits_per_byte(model, Wh, tok, held_text, batch)
        log(f"healing ({label}, {steps} steps): bits/byte {bpb[f'after_{label}']:.4f}")
    report.update(bits_per_byte=bpb, final_run=str(cur), parameters=int(sum(v.size for v in W.values())))
    (out / "transplant_report.json").write_text(json.dumps(report, indent=2, default=str))
    return report
