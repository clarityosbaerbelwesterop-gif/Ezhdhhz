"""Reasoning demo: chain-of-thought addition, test-time compute and rejection-sampling fine-tuning.

    python examples/reasoning/arith.py --out runs/reasoning        # ~2-4 min on a 4-core CPU

1. SFT: a small transformer learns step-by-step addition from a few hundred worked solutions
   ("47+38=" -> "7+8=15,4+3+1=8:85"; loss on the solution only).
2. Test-time compute: greedy vs sampling vs self-consistency vs best-of-n vs beam on held-out problems.
3. RFT (STaR): the model solves 2000 answer-only problems itself, verified solutions are kept and
   trained on; held-out accuracy before/after is measured.
Every number in the written report is measured by this script.
"""
import argparse
import json
import random
import time
from pathlib import Path

from titan.infer import evaluate_problems, generator_for_run
from titan.infer.rft import rejection_finetune
from titan.infer.ttc import load_verifier
from titan.train import TrainConfig, Trainer


def solution(a: int, b: int) -> str:
    s1 = a % 10 + b % 10
    s0 = a // 10 + b // 10 + s1 // 10
    return f"{a % 10}+{b % 10}={s1},{a // 10}+{b // 10}+{s1 // 10}={s0}:{a + b}"


def problems(n, rng, exclude=()):
    out, seen = [], set(exclude)
    while len(out) < n:
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        if (a, b) not in seen:
            seen.add((a, b))
            out.append((a, b))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/reasoning")
    ap.add_argument("--sft", type=int, default=400)
    ap.add_argument("--sft-steps", type=int, default=600)
    ap.add_argument("--rft-problems", type=int, default=2000)
    ap.add_argument("--rft-steps", type=int, default=400)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--backend", default="auto")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    heldout = problems(300, rng)
    sft = problems(a.sft, rng, heldout)
    pool = problems(a.rft_problems, rng, heldout + sft)
    (out / "sft.jsonl").write_text("\n".join(json.dumps({"prompt": f"{x}+{y}=", "response": solution(x, y)})
                                             for x, y in sft))
    ev = [{"prompt": f"{x}+{y}=", "answer": str(x + y)} for x, y in heldout]
    train_probs = [{"prompt": f"{x}+{y}=", "answer": str(x + y)} for x, y in pool]
    t0 = time.perf_counter()
    tr = Trainer(TrainConfig(data=str(out / "sft.jsonl"), backend=a.backend, steps=a.sft_steps, batch=32,
                             eval_every=a.sft_steps, ckpt_every=a.sft_steps, lr=3e-3,
                             overrides={"seq": 32, "d_model": 64, "heads": 4, "layers": 3, "d_ff": 256},
                             out_dir=str(out / "sft")), log=None)
    s = tr.run()
    tr.close()
    report = {"sft": {"examples": a.sft, "steps": a.sft_steps, "parameters": s["parameters"],
                      "train_s": round(time.perf_counter() - t0, 1), "eval_loss": s.get("eval_loss")}, "ttc": []}
    print(f"SFT: {s['parameters']:,} params, {report['sft']['train_s']} s, eval loss {s.get('eval_loss'):.3f}")
    gen = generator_for_run(tr.out, batch=64, backend=a.backend)
    verifier = load_verifier(str(Path(__file__).with_name("verifier.py")) + ":check")
    for strat in ("greedy", "sample", "self-consistency", "best-of-n", "best-of-n+verifier", "beam"):
        if strat == "beam":
            g2 = generator_for_run(tr.out, batch=4, backend=a.backend)
            r = evaluate_problems(g2, ev[:100], "beam", 4, max_new=24, stop=("\n",))
        else:
            r = evaluate_problems(gen, ev, strat.split("+")[0], a.n, max_new=24, temperature=0.8, stop=("\n",),
                                  verifier=verifier if "verifier" in strat else None)
        row = {k: v for k, v in r.items() if k not in ("results", "correct")}
        row["strategy"] = strat
        report["ttc"].append(row)
        print(f"{strat:20s} n={row['n']:<2d} acc {100 * row['accuracy']:5.1f}%  tokens {row['generated_tokens']:6d}  "
              f"{row['wall_s']:6.2f} s  {row['tokens_per_s']:8,.0f} tok/s  ({row['problems']} problems)")
    # KV cache vs full-window recompute (same model, greedy, 64 problems)
    for cache in (True, False):
        g3 = generator_for_run(tr.out, batch=64, backend=a.backend, use_cache=cache)
        r = evaluate_problems(g3, ev[:64], "greedy", max_new=24, stop=("\n",))
        report.setdefault("decode", {})["kv_cache" if cache else "recompute"] = {
            "tokens_per_s": r["tokens_per_s"], "wall_s": r["wall_s"], "accuracy": r["accuracy"]}
    print("decode:", report["decode"])
    rep = rejection_finetune(tr.out, train_probs, ev, rounds=a.rounds, samples=4, steps=a.rft_steps,
                             out=out / "rft", max_new=24, backend=a.backend, lr=1e-3, train_batch=32)
    report["rft"] = rep
    final = generator_for_run(rep["final_run"], batch=64, backend=a.backend)
    report["after_rft"] = {s_: evaluate_problems(final, ev, s_, a.n, max_new=24, temperature=0.8, stop=("\n",))["accuracy"]
                           for s_ in ("greedy", "self-consistency")}
    print("after RFT:", report["after_rft"])
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
