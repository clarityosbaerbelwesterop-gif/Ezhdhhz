"""Frankenstein models: merge, stack and upcycle trained checkpoints.

    average           weighted mean of N models (model soup)
    slerp             spherical interpolation of two models, tensor by tensor
    task-arithmetic   base + lambda * sum_i (model_i - base)
    ties              TIES-merging: trim each task vector to its top-density magnitudes, elect the sign
                      per entry, average the agreeing entries (Yadav et al., 2023)
    stack             layer stacking / depth up-scaling: blocks from several models (or one model twice)
                      concatenated in a given order, e.g. 0:0-5 1:2-7 -> a deeper model
    upcycle-moe       dense -> mixture of experts: every MLP becomes E identical experts plus a router
                      (sparse upcycling); with top-k renormalized gates the function is unchanged

The output is a run directory (config.json, tokenizer.json, model.json, checkpoints/best.npz with a
SHA-256 sidecar), so `titan generate/reason OUT` work directly and training continues with
`titan quickstart --model-config OUT/model.json --init-from OUT/checkpoints/best.npz`.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import fields
from pathlib import Path

import numpy as np

from .nn.transformer import TransformerConfig

OPT_STATE = (".adam_m", ".adam_v", ".momentum")


def load_source(src) -> tuple[dict, dict, dict | None]:
    """(weights, model config dict, tokenizer description) of a run directory or checkpoint .npz."""
    from .infer.generate import checkpoint_of
    p = Path(src)
    run = p if p.is_dir() else p.parent.parent
    ckpt = checkpoint_of(p) if p.is_dir() else p
    info = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else {}
    if info.get("train_config", {}).get("lora_rank"):
        raise ValueError(f"{src}: LoRA runs store adapters only; merge full fine-tunes")
    keys = {f.name for f in fields(TransformerConfig)}
    cfg = {k: v for k, v in info.get("model", {}).items() if k in keys}
    with np.load(ckpt) as z:
        w = {k: z[k].astype(np.float32) for k in z.files if not k.endswith(OPT_STATE)}
    return w, cfg, info.get("tokenizer")


def _same(ws):
    names = set(ws[0])
    for w in ws[1:]:
        if set(w) != names or any(w[k].shape != ws[0][k].shape for k in names):
            raise ValueError("models differ in parameter names or shapes; use 'stack' or train an adapter")
    return sorted(names)


def average(ws, weights=None):
    names = _same(ws)
    a = np.asarray(weights if weights else [1.0] * len(ws), dtype=np.float64)
    a = a / a.sum()
    return {k: sum(ai * w[k].astype(np.float64) for ai, w in zip(a, ws)).astype(np.float32) for k in names}


def slerp(wa, wb, t: float):
    out = {}
    for k in _same([wa, wb]):
        a, b = wa[k].astype(np.float64).ravel(), wb[k].astype(np.float64).ravel()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        cos = float(a @ b / (na * nb)) if na and nb else 1.0
        if abs(cos) > 0.9995:                                   # nearly parallel: linear interpolation
            r = (1 - t) * a + t * b
        else:
            th = np.arccos(np.clip(cos, -1, 1))
            r = (np.sin((1 - t) * th) * a + np.sin(t * th) * b) / np.sin(th)
        out[k] = r.reshape(wa[k].shape).astype(np.float32)
    return out


def task_arithmetic(base, ws, scale: float = 1.0):
    names = _same([base] + ws)
    return {k: (base[k] + scale * sum(w[k] - base[k] for w in ws)).astype(np.float32) for k in names}


def ties(base, ws, density: float = 0.2, scale: float = 1.0):
    names = _same([base] + ws)
    out = {}
    for k in names:
        taus = []
        for w in ws:
            tau = (w[k] - base[k]).astype(np.float64)
            if density < 1.0 and tau.size > 1:
                thr = np.quantile(np.abs(tau), 1 - density)
                tau = np.where(np.abs(tau) >= thr, tau, 0.0)
            taus.append(tau)
        T = np.stack(taus)
        sign = np.sign(T.sum(0))
        agree = (np.sign(T) == sign) & (T != 0)
        merged = np.where(agree, T, 0).sum(0) / np.maximum(agree.sum(0), 1)
        out[k] = (base[k] + scale * merged).astype(np.float32)
    return out


def stack(ws, cfgs, ranges):
    """ranges: [(source, first, last)] blocks in output order."""
    c0 = cfgs[ranges[0][0]]
    name = c0.get("name", "tlm")
    out, j = {}, 0
    for src, first, last in ranges:
        if cfgs[src].get("d_model") != c0.get("d_model") or cfgs[src].get("name", "tlm") != name:
            raise ValueError("stacked models must share d_model and parameter names")
        for i in range(first, last + 1):
            pre = f"{name}.b{i}."
            blk = {k: v for k, v in ws[src].items() if k.startswith(pre)}
            if not blk:
                raise ValueError(f"model {src} has no block {i}")
            out.update({f"{name}.b{j}." + k[len(pre):]: v for k, v in blk.items()})
            j += 1
    block = re.compile(rf"^{re.escape(name)}\.b\d+\.")
    out.update({k: v for k, v in ws[ranges[0][0]].items() if not block.match(k)})
    return out, {**c0, "layers": j}


def upcycle_moe(w, cfg, experts: int, top_k: int, noise: float = 0.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    name = cfg.get("name", "tlm")
    out = {k: v for k, v in w.items() if not re.match(rf"^{re.escape(name)}\.b\d+\.ff[123]\.", k)}
    E = experts
    for i in range(cfg["layers"]):
        pre, moe = f"{name}.b{i}.", f"{name}.b{i}.moe."
        d = w[pre + "ff1.w"].shape[0]
        for src, dst in (("ff1", "w1"), ("ff2", "w2"), ("ff3", "w3")):
            if pre + src + ".w" not in w:
                continue
            W = w[pre + src + ".w"]
            copies = [W] + [W + noise * W.std() * rng.standard_normal(W.shape).astype(np.float32) for _ in range(E - 1)]
            out[moe + dst] = np.concatenate(copies, 0).astype(np.float32)
            if pre + src + ".b" in w:
                out[moe + dst + "_b"] = np.repeat(w[pre + src + ".b"].reshape(1, -1), E, 0).astype(np.float32)
        out[moe + "router.w"] = (0.02 * rng.standard_normal((d, E))).astype(np.float32)
    return out, {**cfg, "experts": E, "top_k": top_k}


def write_run(out, weights: dict, cfg: dict, tokenizer: dict | None, recipe: dict) -> Path:
    out = Path(out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    ck = out / "checkpoints" / "best.npz"
    np.savez(ck, **weights)
    digest = hashlib.sha256(ck.read_bytes()).hexdigest()
    ck.with_suffix(".json").write_text(json.dumps({"sha256": digest, "params": sorted(weights), "merge": recipe}))
    TransformerConfig.from_dict(cfg)                          # validates the merged architecture
    (out / "model.json").write_text(json.dumps(cfg, indent=2))
    info = {"model": {"kind": "lm", **cfg}, "tokenizer": tokenizer, "merge": recipe,
            "parameters": int(sum(v.size for v in weights.values())),
            "train_config": {"seed": 0, "lora_rank": 0, "overrides": {"seq": cfg.get("seq")}}}
    (out / "config.json").write_text(json.dumps(info, indent=2, default=str))
    if tokenizer:
        (out / "tokenizer.json").write_text(json.dumps({"format": "titan-tokenizer-desc-v1", **tokenizer}))
    return out


def merge_cli(a) -> int:
    srcs = [load_source(m) for m in a.models]
    ws, cfgs, toks = [s[0] for s in srcs], [s[1] for s in srcs], [s[2] for s in srcs]
    tok = toks[0]
    if any(t != tok for t in toks[1:]):
        raise SystemExit("models were trained with different tokenizers; merge needs one vocabulary")
    cfg = dict(cfgs[0])
    recipe = {"method": a.method, "models": [str(m) for m in a.models]}
    if a.method == "average":
        w = average(ws, a.weights)
        recipe["weights"] = a.weights
    elif a.method == "slerp":
        if len(ws) != 2:
            raise SystemExit("slerp merges exactly two models")
        w = slerp(ws[0], ws[1], a.t)
        recipe["t"] = a.t
    elif a.method in ("task-arithmetic", "ties"):
        if not a.base:
            raise SystemExit(f"{a.method} needs --base")
        base = load_source(a.base)[0]
        w = task_arithmetic(base, ws, a.scale) if a.method == "task-arithmetic" else ties(base, ws, a.density, a.scale)
        recipe.update(base=a.base, scale=a.scale, density=a.density if a.method == "ties" else None)
    elif a.method == "stack":
        if not a.layers:
            raise SystemExit("stack needs --layers SOURCE:FIRST-LAST ...")
        ranges = []
        for r in a.layers:
            src, _, span = r.partition(":")
            first, _, last = span.partition("-")
            ranges.append((int(src), int(first), int(last or first)))
        w, cfg = stack(ws, cfgs, ranges)
        recipe["layers"] = a.layers
    else:
        w, cfg = upcycle_moe(ws[0], cfgs[0], a.experts, a.top_k, a.noise, a.seed)
        recipe.update(experts=a.experts, top_k=a.top_k, noise=a.noise)
    out = write_run(a.out, w, cfg, tok, recipe)
    n = sum(v.size for v in w.values())
    print(f"{a.method}: {len(ws)} model(s) -> {n:,} parameters, {cfg.get('layers')} layers"
          f"{', ' + str(cfg['experts']) + ' experts' if cfg.get('experts') else ''}; wrote {out} "
          f"(titan generate {out} ... | quickstart --model-config {out}/model.json --init-from {out}/checkpoints/best.npz)")
    return 0
