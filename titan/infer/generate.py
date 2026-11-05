"""Text generation on TITAN programs: a key/value-cached decode step, sampling and beam search.

The decode step (TransformerLM.build_decode) is compiled once for ``batch`` rows and ``tmax``
positions and runs on the fast/accel executors; each new token costs one step over the cache
(O(t) attention) instead of re-running the whole window (O(t^2)). Models without a decode graph
(plugins) fall back to full-window recomputation with the same API.
"""
from __future__ import annotations

import json
import time
from dataclasses import fields
from pathlib import Path

import numpy as np

from ..compiler import CompileOptions, Graph, compile_graph
from ..config import load_config
from ..dtypes import DType
from ..nn.layers import ParamStore
from ..nn.transformer import TransformerConfig, TransformerLM


def _device(backend: str, d_model: int, arch):
    from ..train.trainer import auto_backend
    if backend == "auto":
        backend = auto_backend(True, d_model)
    if backend == "accel":
        from ..runtime.accel import AccelDevice
        return AccelDevice(arch), backend
    from ..runtime import FastDevice
    return FastDevice(arch), "fast"


def checkpoint_of(run: Path, ckpt: str | None = None) -> Path:
    if ckpt:
        return Path(ckpt)
    best = run / "checkpoints" / "best.npz"
    if best.exists():
        return best
    steps = sorted((run / "checkpoints").glob("step_*.npz"))
    if not steps:
        raise FileNotFoundError(f"{run}: no checkpoint (train with --ckpt-every > 0)")
    return steps[-1]


def load_run(run, ckpt: str | None = None):
    """(model, weights, tokenizer, info) of a training run directory."""
    from ..nn.transformer import load_model_plugin
    from ..train.tokenizers import tokenizer_from_description
    run = Path(run)
    info = json.loads((run / "config.json").read_text())
    m = info["model"]
    tc = info["train_config"]
    if tc.get("model_file"):
        model = load_model_plugin(tc["model_file"], m.get("vocab") or info["dataset"]["classes"],
                                  tc["overrides"]["seq"], **{k: v for k, v in tc["overrides"].items() if k != "seq"})
    else:
        keys = {f.name for f in fields(TransformerConfig)}
        model = TransformerLM(TransformerConfig(**{k: v for k, v in m.items() if k in keys}))
    with np.load(checkpoint_of(run, ckpt)) as z:
        weights = {k: z[k] for k in z.files if not k.endswith((".adam_m", ".adam_v", ".momentum"))}
    tok = tokenizer_from_description(info["tokenizer"]) if info.get("tokenizer") else None
    return model, weights, tok, info


class Generator:
    def __init__(self, model, weights: dict, tokenizer=None, batch: int = 1, tmax: int | None = None,
                 backend: str = "auto", arch: str = "host", seed: int = 0, init_scheme: str = "xavier",
                 lora_rank: int = 0, lora_alpha: float = 16.0, use_cache: bool = True):
        self.model, self.tok, self.batch = model, tokenizer, batch
        c = getattr(model, "config", None)
        self.cached = use_cache and hasattr(model, "build_decode") and c is not None
        seq = model.seq
        self.tmax = tmax or seq
        if self.cached and c.position == "learned" and self.tmax > c.seq:
            raise ValueError(f"learned positions: tmax {self.tmax} exceeds the trained length {c.seq}")
        g = Graph("decode" if self.cached else "window")
        ps = ParamStore(g, seed, lazy=True)
        ps.scheme, ps.lora_rank, ps.lora_alpha, ps.fast_init = init_scheme, lora_rank, lora_alpha, bool(lora_rank)
        if self.cached:
            model.build_decode(g, ps, batch, self.tmax)
        else:
            _, logits = model.build(g, ps, batch, DType.FP32)
            g.output("logits", logits)
        cfg = load_config(arch)
        self.exe = compile_graph(g, cfg, CompileOptions(emit_isa=False))
        lora_base = {n for n in self.exe.params if lora_rank and ".lora_" not in n}
        missing = [n for n in self.exe.params if n not in weights and n not in lora_base]
        if missing:
            raise ValueError(f"checkpoint lacks {len(missing)} weights, e.g. {missing[:3]}")
        for n in self.exe.params:
            if n in weights:
                self.exe.lazy_init[n] = (lambda v=weights[n]: np.asarray(v, dtype=np.float32))
        self.dev, self.backend = _device(backend, getattr(c, "d_model", 0), cfg)
        self.dev.load(self.exe)
        self.steps = 0
        if self.cached:
            self.H, self.KV, self.dh = c.heads, c.kv_heads, c.head_dim
            self.cache = {f"{kv}c{i}": np.zeros((batch * self.KV * self.tmax, self.dh), np.float32)
                          for i in range(c.layers) for kv in "kv"}
        self.window = np.zeros((batch, seq), np.int64)

    # ------------------------------------------------------------------ one position for all rows
    def reset(self) -> None:
        if self.cached:
            for a in self.cache.values():
                a[...] = 0

    def step(self, tokens, pos: int) -> np.ndarray:
        """Feed ``tokens`` [batch] at position ``pos`` (all rows); return next-token logits [batch, vocab]."""
        self.steps += 1
        tokens = np.asarray(tokens, dtype=np.int64)
        if not self.cached:
            T = self.model.seq
            self.window[:, min(pos, T - 1)] = tokens
            if pos >= T:                                           # slide the window
                self.window[:, :-1] = self.window[:, 1:]
                self.window[:, -1] = tokens
            out = self.dev.run(self.exe, {"x": self.window.reshape(-1, 1).astype(np.int32)}, fetch=["logits"])
            return out.outputs["logits"].reshape(self.batch, T, -1)[:, min(pos, T - 1)]
        if pos >= self.tmax:
            raise ValueError(f"position {pos} exceeds the cache ({self.tmax}); raise tmax")
        c, B, H, KV, T = self.model.config, self.batch, self.H, self.KV, self.tmax
        f = {"tok": tokens.reshape(-1, 1).astype(np.int32)}
        if c.position == "learned":
            f["pos"] = np.full((B, 1), pos, np.int32)
        elif c.position == "sinusoidal":
            f["pvec"] = np.tile(self.model.sinusoid_table([pos]), (B, 1))
        elif c.position == "rope":
            cos, sin = self.model.rope_table([pos])
            f.update(rope_cq=np.tile(cos, (B * H, 1)), rope_sq=np.tile(sin, (B * H, 1)),
                     rope_ck=np.tile(cos, (B * KV, 1)), rope_sk=np.tile(sin, (B * KV, 1)))
        rows = np.arange(B * KV) * T + pos
        f["slot"] = rows.reshape(-1, 1).astype(np.int32)
        am = np.zeros((B * H, T), np.float32)
        am[:, pos + 1:] = -1e9
        f["amask"] = am
        for k, a in self.cache.items():
            a[rows] = 0
            f[k] = a
        fetch = ["logits"] + [f"{kv}{i}" for i in range(c.layers) for kv in "kv"]
        out = self.dev.run(self.exe, f, fetch=fetch).outputs
        for i in range(c.layers):
            self.cache[f"kc{i}"][rows] = out[f"k{i}"]
            self.cache[f"vc{i}"][rows] = out[f"v{i}"]
        return out["logits"]

    def reorder(self, parents) -> None:
        """Beam search: row b continues the cache of row parents[b]."""
        parents = np.asarray(parents)
        if self.cached:
            for k, a in self.cache.items():
                a[...] = a.reshape(self.batch, -1, self.dh)[parents].reshape(a.shape)
        self.window[...] = self.window[parents]

    # ------------------------------------------------------------------ decoding
    def _stopped(self, ids, stop) -> bool:
        if not stop or self.tok is None:
            return False
        text = self.tok.decode(ids)
        return any(s in text for s in stop)

    def _trim(self, ids, stop) -> str:
        text = self.tok.decode(ids) if self.tok is not None else " ".join(map(str, ids))
        for s in stop or ():
            if s in text:
                text = text[:text.index(s)]
        return text

    def generate(self, prompts, max_new: int = 64, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0,
                 stop=None, seed: int = 0) -> list[dict]:
        """Sample a continuation for each prompt (text or token ids); len(prompts) <= batch.
        Returns per prompt: text, ids, logprob (sum of log p under the model, temperature 1), tokens."""
        rng = np.random.default_rng(seed)
        P = [self.tok.encode(p) if isinstance(p, str) else list(p) for p in prompts]
        if not P or len(P) > self.batch or min(map(len, P)) == 0:
            raise ValueError(f"need 1..{self.batch} non-empty prompts")
        seqs = P + [P[0]] * (self.batch - len(P))
        lens = [len(p) for p in seqs]
        gen = [[] for _ in seqs]
        lp = np.zeros(len(seqs))
        done = [i >= len(P) for i in range(len(seqs))]
        total = max(lens) + max_new
        if self.cached:
            total = min(total, self.tmax)
        self.reset()
        t0, steps0 = time.perf_counter(), self.steps
        for t in range(total - 1):
            feed = [s[t] if t < len(s) else 0 for s in seqs]
            logits = self.step(feed, t)
            act = [b for b in range(len(seqs)) if not done[b] and t + 1 == len(seqs[b]) and t + 1 >= lens[b]]
            if not act:
                if all(done):
                    break
                continue
            z = logits[act].astype(np.float64)
            logp_full = z - z.max(1, keepdims=True)
            logp_full -= np.log(np.exp(logp_full).sum(1, keepdims=True))
            nxt = sample(z, temperature, top_k, top_p, rng)
            for j, b in enumerate(act):
                tok = int(nxt[j])
                seqs[b] = seqs[b] + [tok]
                gen[b].append(tok)
                lp[b] += logp_full[j, tok]
                if len(gen[b]) >= max_new or self._stopped(gen[b], stop):
                    done[b] = True
            if all(done):
                break
        dt = time.perf_counter() - t0
        return [{"text": self._trim(gen[b], stop), "ids": gen[b], "logprob": float(lp[b]), "tokens": len(gen[b]),
                 "decode_steps": self.steps - steps0, "seconds": dt} for b in range(len(P))]

    def beam_search(self, prompt, width: int = 4, max_new: int = 64, stop=None, length_alpha: float = 1.0) -> list[dict]:
        """Beam search (rows = beams); returns finished hypotheses best first."""
        if width > self.batch:
            raise ValueError(f"beam width {width} exceeds the generator batch {self.batch}")
        ids = self.tok.encode(prompt) if isinstance(prompt, str) else list(prompt)
        self.reset()
        for t in range(len(ids) - 1):
            self.step([ids[t]] * self.batch, t)
        beams = [([], 0.0)] * self.batch
        alive = 1                                    # at the start all beams are identical: expand row 0 only
        finished = []
        pos = len(ids) - 1
        last = [ids[-1]] * self.batch
        for _ in range(max_new):
            if self.cached and pos >= self.tmax:
                break
            logits = self.step(last, pos).astype(np.float64)
            logp = logits - logits.max(1, keepdims=True)
            logp -= np.log(np.exp(logp).sum(1, keepdims=True))
            cand = [(beams[b][1] + logp[b, v], b, v) for b in range(alive)
                    for v in np.argsort(-logp[b])[:width + 1]]
            cand.sort(key=lambda x: -x[0])
            new, parents = [], []
            for score, b, v in cand:
                seq = beams[b][0] + [int(v)]
                if self._stopped(seq, stop):
                    finished.append((seq, score))
                elif len(new) < width:
                    new.append((seq, score))
                    parents.append(b)
                if len(new) == width:
                    break
            if not new or len(finished) >= width:
                break
            alive = len(new)
            parents += [parents[0]] * (self.batch - alive)
            new += [new[0]] * (self.batch - alive)
            self.reorder(parents)
            beams = new
            last = [s[-1] for s, _ in beams]
            pos += 1
        finished += beams[:alive]
        finished.sort(key=lambda h: -h[1] / max(1, len(h[0])) ** length_alpha)
        return [{"text": self._trim(s, stop), "ids": s, "logprob": float(sc), "tokens": len(s)} for s, sc in finished[:width]]


def sample(z: np.ndarray, temperature: float, top_k: int, top_p: float, rng) -> np.ndarray:
    """Rows of logits -> token ids (greedy when temperature <= 0)."""
    if temperature <= 0:
        return z.argmax(1)
    z = z / temperature
    if top_k and top_k < z.shape[1]:
        kth = np.partition(z, -top_k, axis=1)[:, -top_k][:, None]
        z = np.where(z < kth, -np.inf, z)
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    if top_p < 1.0:
        order = np.argsort(-p, axis=1)
        sp = np.take_along_axis(p, order, 1)
        keep = np.cumsum(sp, 1) - sp < top_p                     # smallest set with mass >= top_p
        mask = np.zeros_like(p, dtype=bool)
        np.put_along_axis(mask, order, keep, 1)
        p = np.where(mask, p, 0)
        p /= p.sum(1, keepdims=True)
    u = rng.random((p.shape[0], 1))
    return np.minimum((np.cumsum(p, 1) < u).sum(1), p.shape[1] - 1)


def generator_for_run(run, batch: int = 1, tmax: int | None = None, backend: str = "auto", ckpt: str | None = None,
                      use_cache: bool = True):
    model, weights, tok, info = load_run(run, ckpt)
    tc = info["train_config"]
    return Generator(model, weights, tok, batch=batch, tmax=tmax, backend=backend, seed=tc.get("seed", 0),
                     init_scheme=info.get("quality", {}).get("init", "xavier"), lora_rank=tc.get("lora_rank", 0),
                     lora_alpha=tc.get("lora_alpha", 16.0), use_cache=use_cache)
