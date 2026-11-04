"""Own-environment features: accelerated executor, byte corpora, pretrain -> fine-tune, quality controls."""
import json

import numpy as np
import pytest

from titan.cli import main
from titan.compiler import CompileOptions, compile_graph
from titan.runtime import FastDevice
from titan.runtime.accel import torch_available
from titan.train import TrainConfig, Trainer
from titan.train.tokens import load_bytes, prepare_bytes
from tests.test_scale import _graphs

needs_torch = pytest.mark.skipif(not torch_available(), reason="PyTorch not installed")

TEXT_A = "the accelerator streams tiles through the matrix engine and stores the results. " * 60
TEXT_B = "fine tuning adapts a pretrained model to a new domain with fewer steps. " * 60


def _cfg(tmp_path, name, **kw):
    kw.setdefault("eval_every", 1000)
    kw.setdefault("ckpt_every", 0)
    return TrainConfig(out_dir=str(tmp_path / name), **kw)


@needs_torch
@pytest.mark.parametrize("idx", range(4))
def test_accel_matches_fast_executor(dev, rng, idx):
    from titan.runtime.accel import AccelDevice
    g = _graphs(rng)[idx]
    ir = compile_graph(g, dev, CompileOptions(emit_isa=False))
    feeds = {n: rng.standard_normal(ir.buffers[n].shape) for n in ir.inputs}
    a, b = FastDevice(dev), AccelDevice(dev, "cpu")
    for _ in range(3):
        oa, ob = a.run(ir, feeds), b.run(ir, feeds)
    tol = 2e-3 if idx == 1 else 1e-4
    for k in oa.outputs:
        np.testing.assert_allclose(ob.outputs[k], oa.outputs[k], rtol=tol, atol=tol, err_msg=k)
    for p in ir.params:
        np.testing.assert_allclose(b.read(ir, p), a.read(ir, p), rtol=1e-4, atol=1e-5)


@needs_torch
def test_accel_training_tracks_isa_training(tmp_path):
    runs = {}
    for be in ("functional", "accel"):
        tr = Trainer(_cfg(tmp_path, be, model="transformer", profile="tiny", steps=25, backend=be), log=None)
        tr.run()
        runs[be] = [h["loss"] for h in tr.history]
    np.testing.assert_allclose(runs["accel"], runs["functional"], rtol=3e-3)


def test_byte_corpus_is_memory_mapped(tmp_path):
    (tmp_path / "a.txt").write_text(TEXT_A)
    (tmp_path / "b.md").write_text(TEXT_B)
    meta = prepare_bytes([tmp_path / "a.txt", tmp_path / "b.md"], tmp_path / "corpus.bin")
    assert meta["vocab"] == 256 and meta["tokens"] == len(TEXT_A) + 2 + len(TEXT_B)
    ds = load_bytes(tmp_path / "corpus.bin", seq=32)
    assert isinstance(ds.x_train, np.memmap) and ds.classes == 256 and len(ds.sha256) == 64
    assert bytes(ds.x_train[:13]) == TEXT_A[:13].encode()


def test_pretrain_then_finetune_with_freeze(tmp_path):
    (tmp_path / "a.txt").write_text(TEXT_A)
    (tmp_path / "b.txt").write_text(TEXT_B)
    prepare_bytes([tmp_path / "a.txt"], tmp_path / "a.bin")
    pre = Trainer(_cfg(tmp_path, "pre", data=str(tmp_path / "a.bin"), profile="tiny", steps=30, ckpt_every=30),
                  log=None)
    pre.run()
    ckpt = tmp_path / "pre" / "checkpoints" / "step_000030.npz"
    ft = Trainer(_cfg(tmp_path, "ft", data=str(tmp_path / "b.txt"), tokenizer="byte", profile="tiny", steps=10,
                      init_from=str(ckpt), freeze=["tlm.tok_emb", "tlm.pos_emb"], lr=1e-3), log=None)
    info = json.loads((tmp_path / "ft" / "config.json").read_text())["initialized_from"]
    assert info["loaded"] == len(pre.model_params) and info["skipped"] == []
    emb0 = ft.dev.read(ft.train_exe, "tlm.tok_emb").copy()
    np.testing.assert_array_equal(emb0, pre.dev.read(pre.train_exe, "tlm.tok_emb"))
    head0 = ft.dev.read(ft.train_exe, "tlm.head.w").copy()
    ft.run()
    np.testing.assert_array_equal(ft.dev.read(ft.train_exe, "tlm.tok_emb"), emb0)      # frozen
    assert not np.array_equal(ft.dev.read(ft.train_exe, "tlm.head.w"), head0)            # trained
    scratch = Trainer(_cfg(tmp_path, "scratch", data=str(tmp_path / "b.txt"), tokenizer="byte", profile="tiny",
                           steps=1), log=None)
    scratch.run()
    assert ft.history[0]["loss"] < scratch.history[0]["loss"]        # pretrained weights transfer


def test_finetune_skips_mismatched_shapes(tmp_path):
    (tmp_path / "a.txt").write_text(TEXT_A)
    (tmp_path / "b.txt").write_text(TEXT_B)
    pre = Trainer(_cfg(tmp_path, "pre", data=str(tmp_path / "a.txt"), profile="tiny", steps=2, ckpt_every=2),
                  log=None)
    pre.run()
    ft = Trainer(_cfg(tmp_path, "ft", data=str(tmp_path / "b.txt"), profile="tiny", steps=1,
                      init_from=str(tmp_path / "pre" / "checkpoints" / "step_000002.npz")), log=None)
    info = json.loads((tmp_path / "ft" / "config.json").read_text())["initialized_from"]
    assert set(info["skipped"]) >= {"tlm.tok_emb", "tlm.head.w", "tlm.head.b"}   # char vocabularies differ
    assert info["loaded"] > 0


def test_gradient_clipping_on_titan(tmp_path):
    tr = Trainer(_cfg(tmp_path, "c", model="mlp", profile="tiny", steps=1, optimizer="sgd", lr=0.1, clip_grad=1e-3),
                 log=None)
    before = {p: tr.dev.read(tr.train_exe, p).copy() for p in tr.model_params}
    tr.run()
    assert tr.history[0]["grad_norm"] > 1e-3
    delta = np.sqrt(sum(((tr.dev.read(tr.train_exe, p) - before[p]) ** 2).sum() for p in tr.model_params))
    assert delta <= 0.1 * 1e-3 * 1.001                    # SGD step = lr * clipped gradient


def test_cosine_schedule_weight_decay_and_best_checkpoint(tmp_path):
    tr = Trainer(_cfg(tmp_path, "q", model="mlp", profile="tiny", steps=60, eval_every=10, ckpt_every=10,
                      schedule="cosine", weight_decay=0.05, early_stop=2), log=None)
    lr0 = tr.opt.host_feeds(0)["opt.lr_t"][0, 0]
    lr_mid = tr.opt.host_feeds(30)["opt.lr_t"][0, 0]
    lr_end = tr.opt.host_feeds(59)["opt.lr_t"][0, 0]
    assert lr_mid < max(lr0, tr.tc.lr) and lr_end < lr_mid
    assert "opt.wd_t" in tr.train_exe.inputs
    s = tr.run()
    assert (tmp_path / "q" / "checkpoints" / "best.npz").exists()
    assert s["best_eval_loss"] <= s["eval_loss"] + 1e-12


def test_env_and_prepare_data_commands(tmp_path, capsys):
    assert main(["env", "--path", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "RAM" in out and "max trainable parameters" in out
    (tmp_path / "x.txt").write_text(TEXT_A)
    assert main(["prepare-data", str(tmp_path / "x.txt"), "--out", str(tmp_path / "x.bin")]) == 0
    assert (tmp_path / "x.json").exists()
    assert main(["quickstart", "--data", str(tmp_path / "x.bin"), "--profile", "tiny", "--steps", "5", "--arch",
                 "tiny", "--out", str(tmp_path / "r")]) == 0


def test_gpt2_initialization_for_large_transformers(tmp_path):
    from titan.train import TrainConfig as TC
    big = TC(model="transformer", profile="small", overrides={"d_model": 256, "heads": 2, "d_ff": 1024, "layers": 2,
                                                             "seq": 16}, out_dir=str(tmp_path / "a")).resolve()
    small = TC(model="transformer", profile="tiny", out_dir=str(tmp_path / "b")).resolve()
    assert big.init_scheme == "gpt2" and small.init_scheme == "xavier"
    tr = Trainer(TC(model="transformer", profile="small", steps=1, eval_every=1000, ckpt_every=0, backend="fast",
                    overrides={"d_model": 256, "heads": 2, "d_ff": 1024, "layers": 4, "seq": 16},
                    out_dir=str(tmp_path / "c")), log=None)
    w = tr.dev.read(tr.train_exe, "tlm.b0.ff1.w")
    r = tr.dev.read(tr.train_exe, "tlm.b0.ff2.w")
    assert abs(w.std() - 0.02) < 0.002 and abs(r.std() - 0.02 / np.sqrt(8)) < 0.001   # residual scaling
