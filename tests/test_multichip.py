"""M12: multi-chip system model, collectives, data-parallel training, scaling."""
import numpy as np
import pytest

from titan.multichip import (DataParallelTrainer, allreduce_cost, build_programs, ring_allreduce, scaling_study,
                             system_preset)
from titan.nn.models import MLPClassifier
from titan.runtime import Device


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
def test_ring_allreduce_functional_and_traffic(n, rng):
    bufs = [rng.standard_normal((7, 13)) for _ in range(n)]
    out, st = ring_allreduce(bufs)
    for o in out:
        np.testing.assert_allclose(o, sum(bufs), rtol=1e-12, atol=1e-12)
    if n > 1:
        total = bufs[0].nbytes
        assert st["steps"] == 2 * (n - 1)
        # bandwidth term of the cost model: 2 (n-1)/n of the payload per rank (up to chunk rounding)
        assert st["bytes_sent_per_rank"] <= 2 * (n - 1) * (total // n + 8)
        assert st["bytes_sent_per_rank"] >= 2 * (n - 1) * (total // n) - 8


def test_allreduce_cost_model():
    flat = system_preset(8, "ring", 100.0, 1.0)
    c = allreduce_cost(flat, 1e8)
    assert c["seconds"] == pytest.approx(14e-6 + 2 * 7 / 8 * 1e8 / 100e9)
    hier = system_preset(8, "ring", 100.0, 1.0, chiplets_per_package=4, d2d_gbps=1000.0, d2d_latency_us=0.1)
    assert hier.devices == 8 and len(hier.levels) == 2
    assert allreduce_cost(hier, 1e8)["seconds"] < c["seconds"]   # fast d2d level shrinks the slow-level payload
    fc = allreduce_cost(system_preset(8, "fully_connected", 100.0, 1.0), 1e8)
    assert fc["seconds"] < c["seconds"]
    with pytest.raises(ValueError):
        allreduce_cost(system_preset(8, "mesh2d"), 1e6)


def test_data_parallel_equals_single_device_large_batch(tiny, rng):
    """One SGD step on 2 devices x 16 samples == one step on 1 device x 32 samples."""
    m = MLPClassifier(2, 3, [16])
    x, y = rng.standard_normal((32, 2)), np.eye(3)[rng.integers(0, 3, 32)]
    g1, u1, params = build_programs(m, 32, tiny, lr=0.1)
    d = Device(tiny)
    d.load(g1)
    d.load(u1)
    out = d.run(g1, {"x": x, "y": y}).outputs
    d.run(u1, {"grad:" + p: out["grad:" + p] for p in params}, fetch=[])
    single = {p: d.read(g1, p) for p in params}

    g2, u2, _ = build_programs(m, 16, tiny, lr=0.1)
    devs = [Device(tiny) for _ in range(2)]
    grads = []
    for i, dv in enumerate(devs):
        dv.load(g2)
        dv.load(u2)
        o = dv.run(g2, {"x": x[16 * i:16 * (i + 1)], "y": y[16 * i:16 * (i + 1)]}).outputs
        grads.append(np.concatenate([o["grad:" + p].ravel() for p in params]))
    mean = ring_allreduce(grads)[0][0] / 2
    for dv in devs:
        feeds, off = {}, 0
        for p in params:
            shp = g2.buffers[p].shape
            k = int(np.prod(shp))
            feeds["grad:" + p] = mean[off:off + k].reshape(shp)
            off += k
        dv.run(u2, feeds, fetch=[])
    for p in params:
        np.testing.assert_allclose(devs[0].read(g2, p), single[p], rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(devs[0].read(g2, p), devs[1].read(g2, p))


def test_data_parallel_training_converges_and_replicas_stay_identical():
    tr = DataParallelTrainer(4, arch="tiny", global_batch=32, hidden=(16,), lr=0.1, mode="cycle")
    for _ in range(40):
        rec = tr.step()
        assert rec["replicas_identical"]
    assert tr.history[-1]["loss"] < tr.history[0]["loss"]
    assert rec["model_comm_s"] > 0 and rec["step_s"] > rec["sim_compute_s"]


def test_scaling_is_sublinear():
    s = scaling_study((1, 2, 4, 8), arch="tiny", global_batch=64, hidden=(64,), in_dim=16, classes=8)
    rows = {r["devices"]: r for r in s["rows"]}
    assert rows[1]["efficiency"] == pytest.approx(1.0)
    for n in (2, 4, 8):
        assert rows[n]["speedup"] < n                      # no free linear scaling
        assert rows[n]["comm_fraction"] > 0
    assert rows[8]["efficiency"] < rows[2]["efficiency"]
    assert "ASSUMED" in s["method"]
