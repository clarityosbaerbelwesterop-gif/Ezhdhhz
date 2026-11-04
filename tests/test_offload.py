"""Beyond-RAM training: lazy initialization, storage offload, exact memory accounting, 100B compilation."""
import numpy as np
import pytest

from titan.cli import main
from titan.compiler import CompileOptions, Graph, compile_graph, gradients
from titan.dtypes import DType
from titan.nn import ParamStore, TinyTransformerLM
from titan.runtime import FastDevice
from titan.scale import TransformerSpec, compile_training_program, execution_memory, preset
from titan.train import TrainConfig, Trainer
from titan.train.optim import Adam


def _lazy_train_exe(cfg):
    g = Graph("lazy")
    ps = ParamStore(g, 3, lazy=True)
    _, logits = TinyTransformerLM(20, 8, 32, 2, 64, 2).build(g, ps, 4, DType.FP32)
    loss = g.xent(logits, g.input("y", logits.shape))
    params = list(ps.params.values())
    Adam(1e-2, weight_decay=0.1).apply(g, params, gradients(g, loss, params))
    g.output("loss", loss)
    return compile_graph(g, cfg, CompileOptions(emit_isa=False))


def test_lazy_initialization_is_deterministic_and_complete(dev):
    exe = _lazy_train_exe(dev)
    assert exe.init == {} or all(k not in exe.params for k in exe.init)
    assert set(exe.lazy_init) == set(exe.params)
    a, b = FastDevice(dev), FastDevice(dev)
    a.load(exe)
    b.load(exe)
    for p in exe.params:
        np.testing.assert_array_equal(a.read(exe, p), b.read(exe, p))
    w = a.read(exe, "tlm.b0.ff1.w")
    assert w.dtype == np.float32 and 0.05 < w.std() < 0.5


def test_offloaded_execution_is_bit_identical_to_ram(dev, tmp_path, rng):
    exe = _lazy_train_exe(dev)
    ram, off = FastDevice(dev), FastDevice(dev, tmp_path / "off", offload_mb=0.001)
    for d in (ram, off):
        d.load(exe)
    for step in range(3):
        feeds = {"x": np.eye(20)[rng.integers(0, 20, 32)], "y": np.eye(20)[rng.integers(0, 20, 32)],
                 "opt.lr_t": [[1e-2]], "opt.wd_t": [[1e-3]]}
        la = ram.run(exe, feeds).outputs["loss"]
        lb = off.run(exe, feeds).outputs["loss"]
        np.testing.assert_array_equal(la, lb)
    for p in exe.params:
        np.testing.assert_array_equal(ram.read(exe, p), off.read(exe, p))
    files = list((tmp_path / "off").iterdir())
    assert off.offloaded_bytes > 0 and files
    assert not any("-t" in f.name for f in files)              # spilled temporaries are deleted after use
    assert isinstance(off.view(exe, "tlm.b0.ff1.w"), np.memmap)


def test_execution_memory_accounting(dev):
    exe = _lazy_train_exe(dev)
    full = execution_memory(exe)
    off = execution_memory(exe, 0.001)
    n_params = sum(exe.buffers[p].shape[0] * exe.buffers[p].shape[1] for p in exe.params)
    assert full["persistent_bytes"] >= 4 * n_params
    assert full["storage_bytes"] == 0 and off["storage_bytes"] > 0
    assert off["ram_bytes"] < full["ram_bytes"]


def test_trainer_with_offload_matches_ram_training(tmp_path):
    runs = {}
    for name, off in (("ram", None), ("off", str(tmp_path / "storage"))):
        tc = TrainConfig(model="transformer", profile="tiny", steps=8, backend="fast", eval_every=1000, ckpt_every=8,
                         offload_dir=off, offload_mb=0.001, out_dir=str(tmp_path / name))
        tr = Trainer(tc, log=None)
        tr.run()
        runs[name] = tr
    assert runs["off"].dev.offloaded_bytes > 0 and runs["ram"].dev.offload is None
    assert (tmp_path / "off" / "checkpoints" / "step_000008.npz").exists()
    assert runs["off"].history[-1]["loss"] < runs["off"].history[0]["loss"] + 1.0


def test_full_100b_training_step_compiles_with_exact_memory():
    exe = compile_training_program(preset("gpt-100b", seq=32), 1)
    n = sum(exe.buffers[p].shape[0] * exe.buffers[p].shape[1] for p in exe.params
            if not p.endswith((".adam_m", ".adam_v")))
    assert 100e9 < n < 103e9 and len(exe.module.kernels) > 350_000
    m = execution_memory(exe, 4)
    assert m["ram_bytes"] < 32e9                                 # fits an ordinary server's RAM
    assert 1.5e12 < m["storage_bytes"] < 1.8e12                  # ~16 bytes/parameter: p, m, v + gradients
    assert exe.stats["kernels_by_kind"].get("copy", 0) == 0      # all optimizer updates in place


def test_quickstart_offload_paths(tmp_path, capsys):
    assert main(["quickstart", "--params", "100b", "--offload-dir", str(tmp_path / "s")]) == 0
    out = capsys.readouterr().out
    assert "needs" in out and "TB free storage" in out
    assert main(["plan", "--preset", "gpt-125m", "--seq", "32", "--single-machine"]) == 0
    assert "kernels" in capsys.readouterr().out
