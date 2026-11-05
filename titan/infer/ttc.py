"""Test-time compute: spend more inference compute per problem to answer better.

    greedy             one most-likely continuation
    sample             one sampled continuation
    best-of-n          n samples, keep the best by a verifier (file.py:fn(prompt, completion) -> score)
                       or, without one, by the model's own mean token log-probability
    self-consistency   n samples, majority vote over the extracted final answers (ties: higher log-prob)
    beam               beam search of width n, length-normalized log-probability

Answers are extracted with a regular expression (default: the last number in the completion).
``evaluate_problems`` measures accuracy, generated tokens and wall time per strategy, so the
accuracy bought by extra compute is measured, not assumed.
"""
from __future__ import annotations

import importlib
import importlib.util
import re
import time
from collections import Counter
from pathlib import Path

STRATEGIES = ("greedy", "sample", "best-of-n", "self-consistency", "beam")
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def extract_answer(text: str, regex: str | None = None) -> str | None:
    if regex:
        m = re.findall(regex, text)
        if not m:
            return None
        last = m[-1]
        return (last[0] if isinstance(last, tuple) else last).strip()
    m = NUMBER.findall(text)
    return m[-1] if m else None


def load_verifier(spec: str | None):
    if not spec:
        return None
    mod_name, _, attr = spec.rpartition(":")
    if mod_name.endswith(".py"):
        sp = importlib.util.spec_from_file_location(Path(mod_name).stem, mod_name)
        mod = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(mod)
    else:
        mod = importlib.import_module(mod_name)
    return getattr(mod, attr)


def solve(gen, prompts: list[str], strategy: str = "greedy", n: int = 8, max_new: int = 64,
          temperature: float = 0.8, top_k: int = 0, top_p: float = 1.0, stop=("\n",), regex: str | None = None,
          verifier=None, seed: int = 0) -> list[dict]:
    """Answer every prompt with ``strategy``; batches prompts x samples into the generator's rows."""
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    out = []
    if strategy == "beam":
        for p in prompts:
            hyps = gen.beam_search(p, width=n, max_new=max_new, stop=stop)
            best = hyps[0]
            out.append({"completion": best["text"], "answer": extract_answer(best["text"], regex),
                        "candidates": len(hyps), "tokens": sum(h["tokens"] for h in hyps)})
        return out
    k = 1 if strategy in ("greedy", "sample") else n
    temp = 0.0 if strategy == "greedy" else temperature
    reqs = [(i, p) for i, p in enumerate(prompts) for _ in range(k)]
    samples: dict[int, list[dict]] = {i: [] for i in range(len(prompts))}
    for c in range(0, len(reqs), gen.batch):
        chunk = reqs[c:c + gen.batch]
        res = gen.generate([p for _, p in chunk], max_new=max_new, temperature=temp, top_k=top_k, top_p=top_p,
                           stop=stop, seed=seed + c)
        for (i, _), r in zip(chunk, res):
            r["answer"] = extract_answer(r["text"], regex)
            samples[i].append(r)
    for i, p in enumerate(prompts):
        cands = samples[i]
        toks = sum(r["tokens"] for r in cands)
        if strategy == "best-of-n":
            key = (lambda r: verifier(p, r["text"])) if verifier else (lambda r: r["logprob"] / max(1, r["tokens"]))
            best = max(cands, key=key)
            out.append({"completion": best["text"], "answer": best["answer"], "candidates": len(cands), "tokens": toks})
        elif strategy == "self-consistency":
            votes = Counter(r["answer"] for r in cands if r["answer"] is not None)
            if votes:
                top = max(votes.values())
                tied = {a for a, v in votes.items() if v == top}
                best = max((r for r in cands if r["answer"] in tied), key=lambda r: r["logprob"])
            else:
                best = cands[0]
            out.append({"completion": best["text"], "answer": best["answer"], "candidates": len(cands),
                        "votes": dict(votes), "tokens": toks})
        else:
            r = cands[0]
            out.append({"completion": r["text"], "answer": r["answer"], "candidates": 1, "tokens": toks})
    return out


def evaluate_problems(gen, problems: list[dict], strategy: str = "greedy", n: int = 8, **kw) -> dict:
    """problems: [{"prompt": ..., "answer": ...}] -> accuracy and cost of the strategy."""
    t0 = time.perf_counter()
    steps0 = gen.steps
    res = solve(gen, [p["prompt"] for p in problems], strategy, n, **kw)
    wall = time.perf_counter() - t0
    correct = [str(r["answer"]) == str(p["answer"]).strip() for r, p in zip(res, problems)]
    toks = sum(r["tokens"] for r in res)
    return {"strategy": strategy, "n": 1 if strategy in ("greedy", "sample") else n, "problems": len(problems),
            "accuracy": sum(correct) / max(1, len(problems)), "generated_tokens": toks,
            "decode_steps": gen.steps - steps0, "wall_s": round(wall, 3),
            "tokens_per_s": round(toks / wall, 1) if wall else None, "results": res, "correct": correct}
