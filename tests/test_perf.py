"""M9: analytical model, B300 comparison rules, benchmark harness, verified autotuning."""
import json

import pytest

from titan import load_config
from titan.compiler import Graph, compile_graph
from titan.dtypes import DType
from titan.perf import ComparisonError, compare, compare_peak, gemm_roofline, program_analysis, scale_requirements
from titan.perf.benchmark import run_suite, write_report
from titan.tune import sweep_architecture, tune_gemm


def test_roofline_classifies_compute_and_memory_bound(dev):
    big = gemm_roofline(dev, 1024, 1024, 1024, DType.BF16)
    thin = gemm_roofline(dev, 1024, 1024, 1, DType.BF16)
    assert big["bound"] == "compute" and thin["bound"] == "memory"
    assert big["arithmetic_intensity"] > dev.ridge_point(DType.BF16) > thin["arithmetic_intensity"]
    assert big["attainable_ops_per_s"] <= dev.peak_ops(DType.BF16)
    assert "estimate" in big["label"]


def test_program_analysis_counts_work(dev):
    g = Graph()
    g.output("c", g.matmul(g.input("a", (64, 32)), g.input("b", (32, 48))))
    pa = program_analysis(compile_graph(g, dev).program, dev)
    assert pa["mxu_macs_by_dtype"]["FP32"] == 64 * 32 * 48
    assert pa["global_bytes"] >= (64 * 32 + 32 * 48 + 64 * 48) * 4


def test_b300_comparison_rules():
    target = load_config("target")
    c = compare_peak(target, DType.BF16)
    assert c["b300_peak_tflops"] == 2250.0 and c["sparsity"] == "dense"
    assert 10.0 < c["ratio"] < 10.5            # design point is sized for the 10x paper-peak target
    assert c["bandwidth_ratio"] < c["ratio"]    # ...but memory bandwidth is not scaled 10x
    with pytest.raises(ComparisonError):
        compare_peak(target, DType.INT8)        # no sourced like-for-like INT8 figure
    with pytest.raises(ComparisonError):
        compare_peak(target, DType.BF16, sparse=True)
    with pytest.raises(ComparisonError):
        compare(1e15, "simulated", DType.BF16)  # simulated workload vs vendor peak is refused


def test_scale_requirements():
    r = scale_requirements(22500, DType.BF16, 2.0)
    assert r["cores_of_128x128"] == 344
    assert r["memory_tbps_for_balance"] == pytest.approx(80.07, rel=1e-3)


def test_benchmark_report(tmp_path):
    res = run_suite("quick", "tiny", log=None)
    assert all(r["verification"]["passed"] for r in res["records"])
    for r in res["records"]:
        assert r["cycle"]["method"].startswith("simulated")
        assert r["functional"]["method"].startswith("measured")
        assert "estimate" in r["analytical"]["label"]
    jp, mp = write_report(res, tmp_path)
    md = mp.read_text()
    assert res["provenance"]["commit"] in md and "Method legend" in md
    assert json.loads(jp.read_text())["suite"] == "quick"


def test_autotuner_verifies_and_compares_to_baseline(tiny):
    r = tune_gemm(tiny, 64, 48, 40, DType.FP32, max_candidates=12)
    assert r["baseline"]["status"] == "verified"
    for c in r["candidates"]:
        assert c["status"] in ("verified", "rejected")
        if c["status"] == "verified":
            assert c["rel_error"] <= 1e-4
    if r["accepted"]:
        assert r["best"]["cycles"] < r["baseline"]["cycles"]


def test_architecture_sweep(tiny, rng):
    def wl(cfg):
        g = Graph()
        g.output("c", g.matmul(g.input("a", (32, 32)), g.input("b", (32, 32))))
        return g, {"a": rng.standard_normal((32, 32)), "b": rng.standard_normal((32, 32))}
    res = sweep_architecture(tiny, {"mxu_rows": [4, 8], "hbm_gbps": [32.0, 128.0]}, wl)
    assert len(res["rows"]) == 4 and all(r["status"] == "verified" for r in res["rows"])
    best, worst = res["rows"][0], res["rows"][-1]
    assert best["simulated_us"] <= worst["simulated_us"]
    assert best["mxu_rows"] == 8
