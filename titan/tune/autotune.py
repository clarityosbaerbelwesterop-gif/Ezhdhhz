"""Self-optimization: search compiler / architecture configurations with verification against a baseline.

Rules (enforced here):
  * every candidate is compiled, executed in the cycle model AND numerically verified;
    candidates that fail verification are rejected, never reported as improvements;
  * every result is compared with the baseline measured by the same method in the same run;
  * the tuner only proposes compiler options or configuration parameters. It never edits RTL.
"""
from __future__ import annotations

import itertools
import time

import numpy as np

from ..compiler import CompileOptions, Graph, compile_graph, run_reference
from ..compiler.lower import CompileError
from ..dtypes import DType
from ..provenance import provenance
from ..runtime import Device


def _gemm_graph(m, n, k, dt):
    g = Graph("tune-gemm")
    g.output("c", g.matmul(g.input("a", (m, k), dt), g.input("b", (k, n), dt)))
    return g


def evaluate_candidate(g: Graph, cfg, opts: CompileOptions, feeds: dict, ref: dict, tol: float) -> dict:
    try:
        exe = compile_graph(g, cfg, opts)
    except CompileError as e:
        return {"status": "rejected", "reason": f"compile: {e}"}
    out = Device(cfg).run(exe, feeds, "cycle")
    err = max(float(np.abs(out.outputs[k].astype(np.float64) - v).max() / max(1.0, float(np.abs(v).max())))
              for k, v in ref.items())
    if not err <= tol:
        return {"status": "rejected", "reason": f"verification failed (rel err {err:.2e} > {tol})"}
    return {"status": "verified", "cycles": out.result.cycles, "rel_error": err,
            "instructions": exe.stats["instructions"], "hbm_bytes": out.result.counters["hbm_bytes"]}


def tune_gemm(cfg, m: int, n: int, k: int, dt: DType = DType.FP32, seed: int = 0, max_candidates: int = 64,
              log=None) -> dict:
    g = _gemm_graph(m, n, k, dt)
    rng = np.random.default_rng(seed)
    if dt.is_int:
        feeds = {"a": rng.integers(-128, 128, (m, k)), "b": rng.integers(-128, 128, (k, n))}
        tol = 0.0
    else:
        feeds = {"a": rng.standard_normal((m, k)), "b": rng.standard_normal((k, n))}
        tol = 1e-4 if dt == DType.FP32 else 1e-2
    ref, _ = run_reference(g, feeds)
    t0 = time.perf_counter()
    base = evaluate_candidate(g, cfg, CompileOptions(), feeds, ref, tol)
    if base["status"] != "verified":
        raise RuntimeError(f"baseline failed: {base}")
    R, C = cfg.mxu_rows, cfg.mxu_cols
    tms = sorted({min(m, v) for v in (R, 2 * R, 4 * R, 8 * R)})
    tns = sorted({min(n, v) for v in (C, 2 * C, 4 * C, 8 * C)})
    tks = sorted({min(k, v) for v in (32, 64, 128, 256, k)})
    space = list(itertools.product(tms, tns, tks, (True, False), ("rows", "tiles"), (True, False)))
    rng.shuffle(space)
    results = []
    for tm, tn, tk, res_b, dist, db in space[:max_candidates]:
        opts = CompileOptions(gemm_tiles=(tm, tn, tk), resident_b=res_b, gemm_distribution=dist, double_buffer=db)
        r = evaluate_candidate(g, cfg, opts, feeds, ref, tol)
        r["options"] = {"tiles": (tm, tn, tk), "resident_b": res_b, "distribution": dist, "double_buffer": db}
        results.append(r)
        if log and r["status"] == "verified":
            log(f"  tiles={tm}x{tn}x{tk} resident={res_b} dist={dist} db={db}: {r['cycles']:.0f} cycles")
    ok = [r for r in results if r["status"] == "verified"]
    best = min(ok, key=lambda r: r["cycles"]) if ok else None
    speedup = base["cycles"] / best["cycles"] if best else 1.0
    return {"workload": f"GEMM {m}x{n}x{k} {dt.name}", "config": cfg.name, "baseline": base, "best": best,
            "speedup_vs_baseline": speedup, "accepted": bool(best and speedup > 1.0),
            "candidates_evaluated": len(results), "candidates_rejected": len(results) - len(ok),
            "candidates": results, "method": "simulated (cycle model) + verified (numerical reference)",
            "duration_s": round(time.perf_counter() - t0, 3), "provenance": provenance(arch_hash=cfg.config_hash())}


def sweep_architecture(base_cfg, grid: dict, workload, log=None) -> dict:
    """Evaluate a workload over a grid of configuration parameters (cycle model + verification).

    ``workload(cfg) -> (Graph, feeds)``. Returns rows sorted by simulated time.
    """
    keys = list(grid)
    rows = []
    for vals in itertools.product(*grid.values()):
        cfg = base_cfg.replace(**dict(zip(keys, vals))).validate()
        g, feeds = workload(cfg)
        ref, _ = run_reference(g, feeds)
        r = evaluate_candidate(g, cfg, CompileOptions(), feeds, ref, 1e-4)
        row = {**dict(zip(keys, vals)), **r}
        if r["status"] == "verified":
            row["simulated_us"] = r["cycles"] / (cfg.clock_ghz * 1e3)
            row["peak_bf16_tflops"] = cfg.peak_ops(DType.BF16) / 1e12
        rows.append(row)
        if log:
            log(row)
    rows.sort(key=lambda r: r.get("simulated_us", float("inf")))
    return {"grid": grid, "rows": rows, "base_config": base_cfg.name,
            "method": "simulated (cycle model) + verified (numerical reference)"}
