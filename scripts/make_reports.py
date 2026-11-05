#!/usr/bin/env python3
"""Regenerate the committed reports: benchmark suite + reference training runs + scaling study.

    python3 scripts/make_reports.py            # writes reports/*.md|json (takes a few minutes)
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from titan.multichip import DataParallelTrainer, scaling_study  # noqa: E402
from titan.perf.benchmark import run_suite, write_report  # noqa: E402
from titan.provenance import provenance  # noqa: E402
from titan.train import TrainConfig, Trainer  # noqa: E402

OUT = ROOT / "reports"
RUNS = [
    dict(model="mlp", dataset="spirals", profile="small", precision="fp32", backend="functional"),
    dict(model="mlp", dataset="spirals", profile="small", precision="bf16", backend="functional"),
    dict(model="mlp", dataset="blobs", profile="small", precision="fp32", backend="functional", optimizer="sgd",
         lr=0.05),
    dict(model="transformer", dataset="text", profile="small", precision="fp32", backend="functional"),
    dict(model="mlp", dataset="spirals", profile="small", precision="fp32", backend="cycle", steps=20),
]


def large_models():
    """Speed of the execution paths, real training at increasing size, distributed plans up to 175B."""
    import resource
    import subprocess
    from titan import load_config
    from titan.scale import DeviceSpec, compare_devices, preset
    rows = []
    for name, kw in [
        ("MLP small, spirals, 300 steps", dict(model="mlp", profile="small", steps=300)),
        ("Transformer small (19k params), 300 steps", dict(model="transformer", profile="small", steps=300)),
    ]:
        for be, fast_paths in (("functional", False), ("functional", True), ("fast", True)):
            tc = TrainConfig(backend=be, eval_every=10 ** 9, ckpt_every=0,
                             out_dir=str(ROOT / "runs" / "report" / f"speed-{be}-{fast_paths}-{kw['model']}"), **kw)
            tr = Trainer(tc, log=None)
            tr.dev.fast_paths = fast_paths if hasattr(tr.dev, "fast_paths") else True
            t0 = time.perf_counter()
            tr.run()
            dt = time.perf_counter() - t0
            label = {("functional", False): "ISA level, reference engine (before)",
                     ("functional", True): "ISA level, bit-exact replay", ("fast", True): "TITAN IR kernels (fast)"}
            rows.append({"workload": name, "path": label[(be, fast_paths)], "seconds": dt,
                         "final_loss": tr.history[-1]["loss"]})
            tr.close()
    real = []
    for params in ("1m", "20m", "100m", "250m"):
        out = ROOT / "runs" / "report" / f"params-{params}"
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, "-m", "titan", "quickstart", "--params", params, "--steps", "12",
                            "--eval-every", "12", "--ckpt-every", "0", "--out", str(out)], capture_output=True,
                           text=True, cwd=ROOT)
        s = json.loads((out / "summary.json").read_text()) if r.returncode == 0 else {}
        cfg = json.loads((out / "config.json").read_text()) if r.returncode == 0 else {}
        real.append({"requested": params, "parameters": s.get("parameters"), "backend": s.get("backend"),
                     "arch": cfg.get("arch", {}).get("name"), "steps": s.get("steps"),
                     "initial_loss": s.get("initial_loss"), "eval_loss": s.get("eval_loss"),
                     "seconds": time.perf_counter() - t0, "peak_rss_gb_all_children":
                     resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6, "rc": r.returncode})
        print("real", params, real[-1])
    tgt, b3 = DeviceSpec.from_arch(load_config("target")), DeviceSpec.b300()
    plans = []
    for name, n in (("gpt-1.3b", 8), ("gpt-13b", 16), ("gpt-70b", 64), ("gpt-100b", 64), ("gpt-175b", 128)):
        c = compare_devices(preset(name), tgt, b3, n, 1024)
        plans.append({"model": name, "devices": n, "titan": c["a"]["best"], "b300": c["b"]["best"],
                      "params": c["a"]["model"]["params"], "ratio": c["speedup_a_over_b"],
                      "min_titan": c["a"]["min_devices_for_model_states"],
                      "min_b300": c["b"]["min_devices_for_model_states"]})
    return rows, real, plans


def main():
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    write_report(run_suite("default", "dev"), OUT)
    rows = []
    for spec in RUNS:
        spec = dict(spec)
        tc = TrainConfig(eval_every=100, ckpt_every=0, out_dir=str(ROOT / "runs" / "report" /
                                                                  "-".join(str(v) for v in spec.values())), **spec)
        tr = Trainer(tc, log=None)
        s = tr.run()
        tr.close()
        rows.append({"config": spec, "summary": s, "train_program": tr.train_exe.stats})
        print(spec, s["initial_loss"], s["final_loss_avg10"], s.get("eval_accuracy"))
    dp = DataParallelTrainer(4, global_batch=64, mode="cycle")
    for _ in range(100):
        rec = dp.step()
    dp_row = {"devices": 4, "system": dp.sys.describe(), "steps": 100, "final_loss": rec["loss"],
              "replicas_identical": all(h["replicas_identical"] for h in dp.history),
              "eval_accuracy": dp.evaluate(), "last_step": rec}
    scale = scaling_study((1, 2, 4, 8, 16, 32))
    scale_h = scaling_study((4, 8, 16, 32), chiplets_per_package=4)
    data = {"provenance": provenance(), "training": rows, "data_parallel": dp_row, "scaling_flat": scale,
            "scaling_hierarchical": scale_h, "duration_s": round(time.time() - t0, 1)}
    (OUT / "training_and_scaling.json").write_text(json.dumps(data, indent=2, default=str))
    pv = data["provenance"]
    L = ["# TITAN-X1 training and scaling report", "",
         f"Generated by `python3 scripts/make_reports.py` at commit `{pv['commit']}` ({pv['timestamp_utc']}) on "
         f"{pv['host']['cpu']}, Python {pv['host']['python']}, NumPy {pv['host']['numpy']}.", "",
         "All training below executes forward, loss, backward and optimizer update as TITAN ISA programs on the "
         "virtual accelerator (architecture `titan-x1-dev`). Host work: data generation/shuffling, one-hot "
         "encoding, Adam's per-step learning-rate scalar, evaluation argmax/accuracy from TITAN logits.", "",
         "## Single-device training", "",
         "| Model | Dataset | Precision | Optimizer | Backend | Params | Steps | TITAN instr./step | Initial loss "
         "| Final loss (avg last 10) | Eval loss | Eval accuracy | Host wall (s) | Simulated cycles/step |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        c, s = r["config"], r["summary"]
        L.append(f"| {c['model']} | {c['dataset']} | {c['precision']} | {c.get('optimizer', 'adam')} | "
                 f"{s['backend']} | {s['parameters']} | {s['steps']} | "
                 f"{s.get('titan_instructions_per_step', str(s.get('titan_kernels_per_step')) + ' kernels')} | "
                 f"{s['initial_loss']:.4f} | {s['final_loss_avg10']:.4f} | {s.get('eval_loss', float('nan')):.4f} | "
                 f"{s.get('eval_accuracy', float('nan')):.3f} | {s['host_wall_s']} | "
                 f"{s.get('simulated_cycles_per_step', '-')} |")
    from titan.train.datasets import load_dataset
    vocab = load_dataset("text").classes
    L += ["", f"Chance accuracy: spirals 1/3, blobs 1/4, text (next character over a {vocab}-symbol vocabulary) "
          f"1/{vocab} = {1 / vocab:.3f}.", "",
          "## Data-parallel training (4 virtual devices, ring, assumed links)", "",
          f"100 steps, global batch 64: final loss {dp_row['final_loss']:.4f}, eval accuracy "
          f"{dp_row['eval_accuracy']:.3f}, replicas bit-identical every step: {dp_row['replicas_identical']}. "
          f"Last step: simulated compute {rec['sim_compute_s'] * 1e6:.1f} us, update {rec['sim_update_s'] * 1e6:.1f} "
          f"us, modelled all-reduce {rec['model_comm_s'] * 1e6:.1f} us.", ""]
    for title, sc in (("flat ring", scale), ("hierarchical, 4 chiplets per package", scale_h)):
        L += [f"## Strong scaling — {title}", "", sc["workload"] + ". " + sc["method"] + ".", "",
              "| Devices | System | Compute (µs, simulated) | All-reduce (µs, modelled) | Step (µs) | Speed-up | "
              "Efficiency | Comm. share |", "|---|---|---|---|---|---|---|---|"]
        for x in sc["rows"]:
            L.append(f"| {x['devices']} | {x['system']} | {x['sim_compute_s'] * 1e6:.2f} | "
                     f"{x['model_comm_s'] * 1e6:.2f} | {x['step_s'] * 1e6:.2f} | {x['speedup']:.2f} | "
                     f"{100 * x['efficiency']:.1f}% | {100 * x['comm_fraction']:.1f}% |")
        L.append("")
    L += ["Speed-up saturates because per-device work shrinks while the all-reduce volume (all gradients) stays "
          "constant — more devices do not translate linearly into useful performance.", ""]
    (OUT / "TRAINING_AND_SCALING.md").write_text("\n".join(L) + "\n")
    speed, real, plans = large_models()
    data.update({"speed": speed, "real_large_training": real, "plans": plans})
    (OUT / "training_and_scaling.json").write_text(json.dumps(data, indent=2, default=str))
    M = ["# Speed and large-model report", "",
         f"Generated by `python3 scripts/make_reports.py` at commit `{pv['commit']}` on {pv['host']['cpu']} "
         f"({pv['host']['logical_cpus']} logical CPUs). Host wall-clock times are *measured on the host CPU*; they "
         "describe the emulator, not TITAN hardware.", "",
         "## Execution paths (same compiled program, same results)", "",
         "| Workload | Path | Host time (s) | Final loss |", "|---|---|---|---|"]
    for r in speed:
        M.append(f"| {r['workload']} | {r['path']} | {r['seconds']:.2f} | {r['final_loss']:.4f} |")
    M += ["", "*ISA level* executes every TITAN instruction (replay is bit-identical to the reference engine). "
          "*TITAN IR kernels* runs the same compiled kernels on whole tensors (FP32 summation order differs).", "",
          "## Real training on this host (`titan quickstart --params N`, 12 steps, text dataset)", "",
          "| Requested | Parameters | Backend | Emulation config | Initial loss | Eval loss after 12 steps | "
          "Host time (s) |", "|---|---|---|---|---|---|---|"]
    for r in real:
        if r["rc"] == 0:
            M.append(f"| {r['requested']} | {r['parameters']:,} | {r['backend']} | {r['arch']} | "
                     f"{r['initial_loss']:.3f} | {r['eval_loss']:.3f} | {r['seconds']:.0f} |")
        else:
            M.append(f"| {r['requested']} | failed (rc {r['rc']}) | | | | | |")
    M += ["", f"Peak resident memory of the largest run: {max(r['peak_rss_gb_all_children'] for r in real):.1f} GB. "
          "Models whose estimated host memory exceeds 70 % of available RAM are not trained here; `quickstart` "
          "prints the distributed plan instead.", "",
          "## Distributed training plans up to 175B (analytical, Mode C)", "",
          "Global batch 1024 sequences x 2048 tokens. Same planner, links (TP 900 GB/s, DP/PP 400 GB/s, assumed) "
          "and GEMM efficiency (0.8, assumed) for both devices. TITAN = `titan-x1-target` design point (unbuilt).", "",
          "| Model | Params | Devices | TITAN plan | TITAN step (s) | TITAN MFU | B300 plan | B300 step (s) | "
          "B300 MFU | Step-time ratio | Min. devices for states (TITAN / B300) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda b: f"TP{b['tp']} PP{b['pp']} DP{b['dp']} Z{b['zero']}{' RC' if b['recompute'] else ''}"  # noqa
    for p in plans:
        t, b = p["titan"], p["b300"]
        M.append(f"| {p['model']} | {p['params'] / 1e9:.1f}B | {p['devices']} | {fmt(t) if t else 'infeasible'} | "
                 f"{t['time_s']['step']:.2f} | {100 * t['mfu']:.0f}% | {fmt(b) if b else 'infeasible'} | "
                 f"{b['time_s']['step']:.2f} | {100 * b['mfu']:.0f}% | {p['ratio']:.2f}x | "
                 f"{p['min_titan']} / {p['min_b300']} |")
    M += ["", "The step-time ratio stays below the 10.25x BF16 paper-peak ratio because the target design point "
          "has only 4x the B300's memory bandwidth and both use the same interconnect assumptions.", ""]
    (OUT / "SPEED_AND_LARGE_MODELS.md").write_text("\n".join(M) + "\n")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
