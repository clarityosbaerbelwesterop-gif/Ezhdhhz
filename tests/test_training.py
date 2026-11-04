"""M7: training runtime — learning progress, determinism, checkpoint/resume, backend honesty."""
import json

import numpy as np
import pytest

from titan.train import TrainConfig, Trainer


def make(tmp_path, name="run", **kw):
    kw.setdefault("model", "mlp")
    kw.setdefault("profile", "tiny")
    kw.setdefault("eval_every", 1000)
    kw.setdefault("ckpt_every", 0)
    return Trainer(TrainConfig(out_dir=str(tmp_path / name), **kw), log=None)


def test_mlp_learns_spirals(tmp_path):
    tr = make(tmp_path, steps=120, dataset="spirals")
    s = tr.run()
    assert s["final_loss_avg10"] < 0.4 * s["initial_loss"]
    assert s["eval_accuracy"] > 0.85
    cfg = json.loads((tmp_path / "run" / "config.json").read_text())
    assert cfg["placement"]["titan"][0] == "forward pass"
    assert cfg["provenance"]["commit"]


def test_same_seed_is_bitwise_deterministic(tmp_path):
    a = make(tmp_path, "a", steps=15)
    b = make(tmp_path, "b", steps=15)
    a.run()
    b.run()
    assert [h["loss"] for h in a.history] == [h["loss"] for h in b.history]


def test_checkpoint_resume_reproduces_uninterrupted_run(tmp_path):
    full = make(tmp_path, "full", steps=20, optimizer="adam")
    full.run()
    part = make(tmp_path, "part", steps=10, optimizer="adam", ckpt_every=10)
    part.run()
    ckpt = tmp_path / "part" / "checkpoints" / "step_000010.npz"
    assert ckpt.exists()
    resumed = make(tmp_path, "resumed", steps=20, optimizer="adam")
    resumed.load_checkpoint(ckpt)
    resumed.run()
    assert [h["loss"] for h in resumed.history] == [h["loss"] for h in full.history[10:]]
    for n in full.train_exe.params:
        np.testing.assert_array_equal(full.dev.read(full.train_exe, n), resumed.dev.read(resumed.train_exe, n))


def test_checkpoint_corruption_detected(tmp_path):
    tr = make(tmp_path, steps=2, ckpt_every=2)
    tr.run()
    ck = tmp_path / "run" / "checkpoints" / "step_000002.npz"
    data = bytearray(ck.read_bytes())
    data[-10] ^= 0xFF
    ck.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="corrupt"):
        make(tmp_path, "other", steps=2).load_checkpoint(ck)


@pytest.mark.parametrize("opt", ["sgd", "adam"])
def test_bf16_mixed_precision_training(tmp_path, opt):
    tr = make(tmp_path, steps=80, precision="bf16", optimizer=opt, dataset="blobs",
              lr=0.05 if opt == "sgd" else 0.01)
    s = tr.run()
    assert s["final_loss_avg10"] < 0.5 * s["initial_loss"]
    assert any(i.opcode.name == "CONVERT" for i in tr.train_exe.program.instructions)


def test_transformer_lm_learns(tmp_path):
    tr = make(tmp_path, model="transformer", steps=40)
    s = tr.run()
    assert s["initial_loss"] > 2.9
    assert s["final_loss_avg10"] < s["initial_loss"] - 0.5


def test_cycle_backend_reports_simulated_time(tmp_path):
    tr = make(tmp_path, steps=2, backend="cycle")
    s = tr.run()
    assert s["simulated_cycles_per_step"] > 0
    assert all("simulated_cycles" in h for h in tr.history)


@pytest.mark.parametrize("backend,msg", [("fpga", "not operational"), ("hardware", "not operational"),
                                         ("rtl", "INT8")])
def test_unavailable_backends_are_refused(tmp_path, backend, msg):
    import shutil
    if backend == "rtl" and not shutil.which("verilator"):
        msg = "not operational"
    with pytest.raises(RuntimeError, match=msg):
        make(tmp_path, steps=1, backend=backend)


def test_invalid_configs(tmp_path):
    with pytest.raises(ValueError):
        make(tmp_path, precision="int8")
    with pytest.raises(ValueError):
        make(tmp_path, model="transformer", dataset="spirals")
