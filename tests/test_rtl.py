"""M10/M11: RTL co-simulation against the golden model, hybrid RTL execution, synthesis reports."""
import shutil

import numpy as np
import pytest

from titan.compiler import Graph, compile_graph, run_reference
from titan.dtypes import DType
from titan.runtime import BACKENDS, Device

pytestmark = pytest.mark.rtl
needs_verilator = pytest.mark.skipif(not shutil.which("verilator"), reason="Verilator not installed")
needs_yosys = pytest.mark.skipif(not shutil.which("yosys"), reason="yosys not installed")


@pytest.fixture(scope="module")
def sim8():
    from titan.rtl.cosim import MXUTileSim
    return MXUTileSim(8, 8, 8)


@needs_verilator
def test_rtl_matches_golden_model_bit_exact(sim8):
    from titan.emu.semantics import matmul_values
    rng = np.random.default_rng(7)
    for m, n, k, acc in [(1, 1, 1, False), (8, 8, 8, False), (13, 5, 21, True), (20, 17, 3, True)]:
        A, B = rng.integers(-128, 128, (m, k)), rng.integers(-128, 128, (k, n))
        C0 = rng.integers(-2**31, 2**31, (m, n)) if acc else None
        ref = matmul_values(A.astype(np.int8), B.astype(np.int8), DType.INT8, False, False,
                            None if C0 is None else C0.astype(np.int32))
        np.testing.assert_array_equal(sim8.matmul(A, B, C0), ref)


@needs_verilator
def test_rtl_accumulator_wraparound(sim8):
    A = np.full((8, 4), -128)
    B = np.full((4, 8), -128)
    C0 = np.full((8, 8), 2**31 - 1)
    want = ((2**31 - 1 + 4 * 16384 + 2**31) % 2**32) - 2**31
    assert (sim8.matmul(A, B, C0) == want).all()


@needs_verilator
def test_correspondence_int4():
    from titan.rtl.cosim import correspondence_test
    r = correspondence_test(6, seed=3, rows=4, cols=4, data_w=4)
    assert r["mismatches"] == 0 and r["cases"] == 6


@needs_verilator
def test_hybrid_rtl_backend_runs_quantized_layer(sim8, tiny):
    from titan.rtl.cosim import make_matmul_hook
    rng = np.random.default_rng(0)
    g = Graph("qlinear")
    x = g.input("x", (24, 40))
    w = g.param("w", rng.integers(-127, 128, (40, 16)), DType.INT8)
    g.output("y", g.cast(g.matmul(g.cast(x, DType.INT8, 30.0), w), DType.FP32, 0.001))
    exe = compile_graph(g, tiny)
    feeds = {"x": rng.uniform(-4, 4, (24, 40))}
    hook = make_matmul_hook(sim8)
    rtl = Device(tiny).run(exe, feeds, "functional", matmul_hook=hook).outputs["y"]
    golden = Device(tiny).run(exe, feeds, "functional").outputs["y"]
    np.testing.assert_array_equal(rtl, golden)
    np.testing.assert_allclose(rtl, run_reference(g, feeds)[0]["y"], rtol=1e-6)
    assert hook.stats["rtl_matmuls"] >= 1
    assert BACKENDS["rtl"].status()[0]


@needs_yosys
def test_synthesis_resource_report(tmp_path):
    from titan.rtl.synth import synthesize
    r = synthesize("titan_mxu_tile", {"ROWS": 2, "COLS": 2, "DATA_W": 8}, "xc7", tmp_path)
    s = r["summary"]
    assert s["DSP48E1"] == 4 and s["FF"] == 4 * 32
    assert "pre-place-and-route" in r["method"]
    assert list(tmp_path.glob("*.log"))
