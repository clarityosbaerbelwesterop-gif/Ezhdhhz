"""Replay executor: bit-identical to the reference engine; cached cycle timing equals a full simulation."""
import numpy as np
import pytest

from titan.compiler import Graph, compile_graph, gradients
from titan.dtypes import DType
from titan.emu import CYCLE, Engine
from titan.emu.replay import ReplayProgram
from titan.nn import MLPClassifier, ParamStore, transformer_block
from titan.runtime import Device


def _graphs(rng):
    g1 = Graph("mlp-train")
    ps = ParamStore(g1, 0)
    _, lg = MLPClassifier(6, 3, [32, 16], "gelu").build(g1, ps, 24, DType.FP32)
    loss = g1.xent(lg, g1.input("y", lg.shape))
    params = list(ps.params.values())
    for p, gr in zip(params, gradients(g1, loss, params)):
        g1.assign(p, g1.sub(p, g1.scale(gr, 0.1)))
    g1.output("loss", loss)
    g2 = Graph("block-bf16")
    ps2 = ParamStore(g2, 1)
    x = g2.input("x", (16, 16))
    g2.output("y", transformer_block(g2, ps2, x, 2, 8, 2, 32, "b", True, DType.BF16))
    g3 = Graph("int8")
    a = g3.input("a", (20, 24), DType.INT8)
    w = g3.param("w", rng.integers(-100, 100, (24, 12)), DType.INT8)
    g3.output("y", g3.cast(g3.matmul(a, w), DType.FP32, 0.01))
    g4 = Graph("reduce-transpose")
    z = g4.input("z", (37, 53))
    g4.output("r", g4.sum(g4.transpose(g4.softmax(z)), 2))
    g4.output("m", g4.max(z, 0))
    return [g1, g2, g3, g4]


def _feeds(exe, rng):
    out = {}
    for n in exe.inputs:
        ref = exe.ref(n)
        out[n] = rng.integers(-100, 100, ref.shape) if ref.dtype.is_int else rng.standard_normal(ref.shape)
    return out


@pytest.mark.parametrize("idx", range(4))
def test_replay_is_bit_identical(dev, rng, idx):
    g = _graphs(rng)[idx]
    exe = compile_graph(g, dev)
    feeds = _feeds(exe, rng)
    ref_dev, rep_dev = Device(dev), Device(dev)
    for d in (ref_dev, rep_dev):
        d.load(exe)
        for k, v in feeds.items():
            d.write(exe, k, v)
    for _ in range(3):                      # repeated runs (in-place parameter updates)
        Engine(ref_dev.machine, "functional").run(exe.program, validate_program=False)
    rp = ReplayProgram(rep_dev.machine, exe.program)
    for _ in range(3):
        r = rp.run()
    assert np.array_equal(ref_dev.machine.glob.buf, rep_dev.machine.glob.buf)
    for a, b in zip(ref_dev.machine.local, rep_dev.machine.local):
        assert np.array_equal(a.buf, b.buf)
    assert r.instructions == rp.template.instructions and r.ops > 0


def test_cached_cycle_timing_matches_full_simulation(dev, rng):
    g = _graphs(rng)[0]
    exe = compile_graph(g, dev)
    d = Device(dev)
    d.load(exe)
    first = d.run(exe, _feeds(exe, rng), CYCLE).result          # full cycle simulation
    second = d.run(exe, _feeds(exe, rng), CYCLE).result         # replay data path + cached timing
    d2 = Device(dev)
    d2.fast_paths = False
    d2.load(exe)
    full = d2.run(exe, _feeds(exe, rng), CYCLE).result           # independent full simulation, other data
    assert first.cycles == second.cycles == full.cycles
    assert second.counters["hbm_bytes"] == full.counters["hbm_bytes"]
