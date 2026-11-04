"""Reproducible benchmark harness.

Every metric carries the method that produced it:
  measured   host wall-clock of the emulator on the host CPU (NOT accelerator speed)
  simulated  TITAN-X1 cycle model at the configured (assumed) clock
  analytical roofline estimate
  verified   numerical comparison against the float64 NumPy reference
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..compiler import Graph, compile_graph, gradients, run_reference
from ..config import load_config
from ..dtypes import DType
from ..nn import MLPClassifier, ParamStore, transformer_block
from ..provenance import provenance
from ..runtime import Device
from .analytical import LABEL, gemm_roofline, program_analysis

M_MEASURED = "measured: host wall clock of the software emulator on the host CPU (not accelerator speed)"
M_SIM = "simulated: TITAN-X1 cycle model (transaction-level, cycle-approximate) at the configured clock"
M_VER = "verified: functional emulation vs float64 NumPy reference"


def _feeds(g: Graph, rng) -> dict:
    out = {}
    for t in g.inputs():
        if t.dtype.is_int:
            lim = 8 if t.dtype == DType.INT4 else 128
            out[t.name] = rng.integers(-lim, lim, t.shape)
        else:
            out[t.name] = rng.standard_normal(t.shape)
    return out


def run_workload(name: str, g: Graph, cfg, precision: str, seed: int = 0, tol: float = 1e-3,
                 analytical: dict | None = None, opts=None) -> dict:
    rng = np.random.default_rng(seed)
    feeds = _feeds(g, rng)
    t0 = time.perf_counter()
    exe = compile_graph(g, cfg, opts)
    compile_s = time.perf_counter() - t0
    ref, _ = run_reference(g, feeds)
    rec = {"workload": name, "precision": precision, "config": cfg.name, "config_hash": cfg.config_hash(),
           "program": {"instructions": exe.stats["instructions"], "kernels": exe.stats["kernels"],
                       "barriers": exe.stats["barriers"]}}
    dev = Device(cfg)
    f = dev.run(exe, feeds, "functional")
    err = max(float(np.abs(f.outputs[k].astype(np.float64) - ref[k]).max() /
                    max(1.0, float(np.abs(ref[k]).max()))) for k in ref)
    rec["verification"] = {"max_rel_error": err, "tolerance": tol, "passed": bool(err <= tol), "method": M_VER}
    rec["functional"] = {"instructions_retired": f.result.instructions, "ops": f.result.ops,
                         "dma_bytes": f.result.dma_bytes, "host_wall_s": round(f.timings["execute_s"], 6),
                         "method": M_MEASURED}
    c = Device(cfg).run(exe, feeds, "cycle").result
    cyc, ctr = c.cycles, c.counters
    secs = c.simulated_seconds
    stalls = ctr["stall_cycles"]
    rec["cycle"] = {
        "simulated_cycles": cyc, "simulated_seconds": secs,
        "simulated_tops": c.ops / secs / 1e12 if secs else 0.0,
        "fraction_of_config_peak": (c.ops / secs) / cfg.peak_ops(DType.parse(precision)) if secs and
        DType.parse(precision) != DType.INT32 else None,
        "unit_utilization": ctr["unit_utilization"], "mxu_tile_efficiency": ctr["mxu_tile_efficiency"],
        "hbm_bytes": ctr["hbm_bytes"], "effective_hbm_gbps": ctr["hbm_bytes"] / secs / 1e9 if secs else 0.0,
        "l2_read_hit_rate": ctr["l2"]["read_hit_rate"],
        "stall_cycles": stalls, "queue_delay_cycles": ctr["queue_delay_cycles"],
        "sync_overhead_fraction": (stalls.get("barrier", 0) + stalls.get("sync_wait", 0)) /
        (cyc * cfg.cores) if cyc else 0.0,
        "method": M_SIM}
    rec["host_overhead"] = {"compile_s": round(compile_s, 6), "host_io_s": round(f.host_io_s, 6),
                            "method": M_MEASURED}
    rec["analytical"] = analytical or program_analysis(exe.program, cfg)
    return rec


def _gemm(m, n, k, dt):
    g = Graph(f"gemm{m}x{n}x{k}")
    a, b = g.input("a", (m, k), dt), g.input("b", (k, n), dt)
    g.output("c", g.matmul(a, b))
    return g


def _ewise(m, n):
    g = Graph("ewise")
    x, y = g.input("x", (m, n)), g.input("y", (m, n))
    g.output("z", g.add(x, y))
    return g


def _mlp_step(batch=64):
    g = Graph("mlp-train-step")
    ps = ParamStore(g, 0)
    _, logits = MLPClassifier(16, 8, [128, 128]).build(g, ps, batch, DType.FP32)
    loss = g.xent(logits, g.input("y", logits.shape))
    params = list(ps.params.values())
    for p, gr in zip(params, gradients(g, loss, params)):
        g.assign(p, g.sub(p, g.scale(gr, 0.01)))
    g.output("loss", loss)
    return g


def _block(batch=4, seq=32, d=64, heads=4):
    g = Graph("transformer-block-fwd")
    ps = ParamStore(g, 0)
    x = g.input("x", (batch * seq, d))
    g.output("y", transformer_block(g, ps, x, batch, seq, heads, 4 * d, "blk"))
    return g


SUITES = {
    "quick": [("gemm", 64, 64, 64, "fp32"), ("gemm", 128, 128, 128, "bf16"), ("gemm", 128, 128, 128, "int8"),
              ("ewise", 256, 256), ("mlp_step",), ("block",)],
    "default": [("gemm", 128, 128, 128, "fp32"), ("gemm", 256, 256, 256, "fp32"), ("gemm", 256, 256, 256, "bf16"),
                ("gemm", 256, 256, 256, "int8"), ("gemm", 512, 512, 512, "bf16"), ("gemm", 512, 64, 512, "bf16"),
                ("gemm", 1024, 1024, 64, "bf16"), ("ewise", 1024, 1024), ("mlp_step",), ("block",)],
}


def run_suite(suite: str = "quick", arch: str = "dev", seed: int = 0, log=print) -> dict:
    cfg = load_config(arch)
    t0 = time.perf_counter()
    records = []
    for spec in SUITES[suite]:
        kind = spec[0]
        if kind == "gemm":
            _, m, n, k, p = spec
            dt = DType.parse(p)
            rec = run_workload(f"GEMM {m}x{n}x{k}", _gemm(m, n, k, dt), cfg, dt.name, seed,
                               tol=0 if dt.is_int else 1e-2 if dt != DType.FP32 else 1e-4,
                               analytical=gemm_roofline(cfg, m, n, k, dt))
        elif kind == "ewise":
            _, m, n = spec
            rec = run_workload(f"element-wise add {m}x{n}", _ewise(m, n), cfg, "FP32", seed, tol=1e-6)
        elif kind == "mlp_step":
            rec = run_workload("MLP training step (16-128-128-8, batch 64, fwd+bwd+SGD)", _mlp_step(), cfg, "FP32",
                               seed, tol=1e-4)
        else:
            rec = run_workload("transformer block forward (B4 T32 d64 h4)", _block(), cfg, "FP32", seed, tol=1e-4)
        records.append(rec)
        if log:
            c = rec["cycle"]
            log(f"{rec['workload']:58s} {rec['precision']:5s} verify={'PASS' if rec['verification']['passed'] else 'FAIL'}"
                f" sim={c['simulated_cycles']:>10.0f} cyc  {c['simulated_tops']:8.3f} TOPS(sim)  "
                f"host={rec['functional']['host_wall_s']:.3f}s")
    return {"suite": suite, "provenance": provenance(arch=cfg.name, arch_hash=cfg.config_hash(), seed=seed),
            "config": cfg.to_dict(), "records": records, "duration_s": round(time.perf_counter() - t0, 3)}


def write_report(res: dict, out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jp = out / f"bench_{res['suite']}_{res['config']['name']}.json"
    jp.write_text(json.dumps(res, indent=2, default=float))
    pv, cfg = res["provenance"], res["config"]
    L = [f"# TITAN-X1 benchmark report — suite `{res['suite']}`", "",
         "All numbers are reproducible with `titan bench --suite %s --arch %s --seed %s`." %
         (res["suite"], cfg["name"], pv["seed"]), "",
         "| Provenance | |", "|---|---|",
         f"| commit | `{pv['commit']}` |", f"| software | titan {pv['titan_version']}, Python {pv['host']['python']}, "
         f"NumPy {pv['host']['numpy']} |", f"| host hardware (for *measured* rows) | {pv['host']['cpu']} "
         f"({pv['host']['logical_cpus']} logical CPUs) |", f"| architecture config | `{cfg['name']}` hash "
         f"`{pv['arch_hash']}`, {cfg['cores']} cores, {cfg['mxu_rows']}x{cfg['mxu_cols']} MXU, "
         f"{cfg['clock_ghz']} GHz (assumed), HBM {cfg['hbm_gbps']} GB/s |",
         f"| timestamp | {pv['timestamp_utc']} |", f"| suite duration | {res['duration_s']} s (host) |", "",
         "## Method legend", "", f"* **verified** — {M_VER}", f"* **simulated** — {M_SIM}",
         f"* **analytical** — {LABEL}", f"* **measured** — {M_MEASURED}", "",
         "No row is a hardware measurement of TITAN silicon or FPGA: neither exists.", "",
         "## Results", "",
         "| Workload | Prec | Verified (max rel err) | Simulated cycles | Simulated TOPS | % of config peak | "
         "MXU util | HBM GB/s (sim) | L2 hit | Sync overhead | Analytical bound | Analytical est. (µs) | "
         "Sim / est. | Host wall (s) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["records"]:
        c, a, v = r["cycle"], r["analytical"], r["verification"]
        est = a.get("estimated_seconds", a.get("lower_bound_seconds"))
        bound = a.get("bound", a.get("bound_by"))
        frac = c["fraction_of_config_peak"]
        L.append(f"| {r['workload']} | {r['precision']} | {'PASS' if v['passed'] else 'FAIL'} "
                 f"({v['max_rel_error']:.1e}) | {c['simulated_cycles']:.0f} | {c['simulated_tops']:.3f} | "
                 f"{'' if frac is None else f'{100 * frac:.1f}%'} | {100 * c['unit_utilization'].get('MXU', 0):.1f}% | "
                 f"{c['effective_hbm_gbps']:.1f} | {100 * c['l2_read_hit_rate']:.0f}% | "
                 f"{100 * c['sync_overhead_fraction']:.1f}% | {bound} | {est * 1e6:.2f} | "
                 f"{c['simulated_seconds'] / est if est else float('nan'):.2f}x | {r['functional']['host_wall_s']:.3f} |")
    L += ["", "*Sim / est.* = simulated time divided by the analytical lower bound (1.0 = roofline-limited).",
          "*MXU util* = busy cycles of all matrix engines / (cores x simulated cycles).", ""]
    mp = out / f"bench_{res['suite']}_{res['config']['name']}.md"
    mp.write_text("\n".join(L) + "\n")
    return jp, mp
