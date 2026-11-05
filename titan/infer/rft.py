"""Reasoning post-training by rejection sampling (STaR / expert iteration / RFT).

Each round: sample ``samples`` solutions per training problem with the current model, keep the
ones whose extracted answer is correct (or that a verifier accepts), and fine-tune on them
(response-only loss). Problems need only a final answer; the reasoning traces are the model's own.
Accuracy on held-out problems is measured before and after every round.
"""
from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

from ..nn.transformer import TransformerConfig
from .generate import checkpoint_of, generator_for_run, load_run
from .ttc import evaluate_problems, extract_answer


def rejection_finetune(run, problems: list[dict], eval_problems: list[dict] | None = None, rounds: int = 1,
                       samples: int = 8, steps: int = 200, out="runs/rft", temperature: float = 0.8,
                       top_p: float = 1.0, max_new: int = 32, stop=("\n",), regex: str | None = None, verifier=None,
                       backend: str = "auto", batch: int = 64, train_batch: int | None = None, lr: float = 1e-3,
                       mix: list[dict] | None = None, seed: int = 0, log=print) -> dict:
    from ..train import TrainConfig, Trainer
    log = log or (lambda *a, **k: None)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cur = Path(run)
    report = {"rounds": []}
    for r in range(rounds):
        gen = generator_for_run(cur, batch=batch, backend=backend)
        before = evaluate_problems(gen, eval_problems, "greedy", max_new=max_new, stop=stop, regex=regex) \
            if eval_problems else None
        kept, seen, n_samples = [], set(), 0
        reqs = [p for p in problems for _ in range(samples)]
        for c in range(0, len(reqs), batch):
            chunk = reqs[c:c + batch]
            res = gen.generate([p["prompt"] for p in chunk], max_new=max_new, temperature=temperature, top_p=top_p,
                               stop=stop, seed=seed + 1000 * r + c)
            for p, x in zip(chunk, res):
                n_samples += 1
                ok = verifier(p["prompt"], x["text"]) > 0 if verifier else \
                    extract_answer(x["text"], regex) == str(p["answer"]).strip()
                key = (p["prompt"], x["text"])
                if ok and key not in seen:
                    seen.add(key)
                    kept.append({"prompt": p["prompt"], "response": x["text"]})
        solved = len({k["prompt"] for k in kept})
        log(f"round {r + 1}: {len(kept)} verified solutions from {n_samples} samples; "
            f"{solved}/{len(problems)} problems solved at least once")
        if not kept:
            break
        data = out / f"round{r + 1}.jsonl"
        data.write_text("\n".join(json.dumps(k) for k in kept + (mix or [])))
        model, _, _, info = load_run(cur)
        keys = {f.name for f in fields(TransformerConfig)}
        arch = {k: v for k, v in info["model"].items() if k in keys}
        tc = TrainConfig(model="transformer", data=str(data), tokenizer=str(cur / "tokenizer.json"),
                         init_from=str(checkpoint_of(cur)), overrides=arch, steps=steps, lr=lr, backend=backend,
                         batch=train_batch, eval_every=steps, ckpt_every=steps, seed=seed + r,
                         out_dir=str(out / f"round{r + 1}"))
        tr = Trainer(tc, log=None)
        s = tr.run()
        tr.close()
        cur = tr.out
        after = evaluate_problems(generator_for_run(cur, batch=batch, backend=backend), eval_problems, "greedy",
                                  max_new=max_new, stop=stop, regex=regex) if eval_problems else None
        rec = {"round": r + 1, "samples": n_samples, "kept": len(kept), "problems_solved": solved,
               "train_eval_loss": s.get("eval_loss"), "run": str(cur),
               "heldout_greedy_before": before["accuracy"] if before else None,
               "heldout_greedy_after": after["accuracy"] if after else None}
        report["rounds"].append(rec)
        log(f"round {r + 1}: held-out greedy accuracy {rec['heldout_greedy_before']} -> {rec['heldout_greedy_after']}")
    report["final_run"] = str(cur)
    (out / "rft_report.json").write_text(json.dumps(report, indent=2))
    return report
