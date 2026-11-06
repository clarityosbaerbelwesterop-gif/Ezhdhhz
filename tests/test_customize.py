"""Bring your own transformer and tokenizer; compiled execution plans (bit-identical to the interpreters)."""
import json
import textwrap

import numpy as np
import pytest

from titan.cli import main
from titan.compiler import CompileOptions, Graph, compile_graph, evaluate, gradients, run_reference
from titan.dtypes import DType
from titan.nn import ParamStore, TransformerConfig, TransformerLM
from titan.runtime import Device, FastDevice
from titan.scale import TransformerSpec
from titan.train import TrainConfig, Trainer
from titan.train.datasets import CORPUS
from titan.train.tokenizers import BPETokenizer, ByteTokenizer, CharTokenizer, load_tokenizer
from titan.train.tokens import load_tokens, prepare_tokens

VARIANTS = {
    "gpt2": {},
    "llama": {"norm": "rmsnorm", "activation": "swiglu", "position": "rope", "heads": 4, "kv_heads": 2,
              "bias": False, "tie_embeddings": True},
    "mqa-geglu": {"activation": "geglu", "position": "rope", "heads": 4, "kv_heads": 1, "qkv_bias": True},
    "postln": {"prenorm": False, "position": "sinusoidal", "activation": "relu", "qkv_bias": True},
    "silu-nopos": {"activation": "silu", "position": "none", "norm": "rmsnorm"},
}


def _torch():
    from titan.runtime.accel import torch_available
    return torch_available()


def _model(variant, vocab=13, seq=6, d=16):
    return TransformerLM(TransformerConfig(vocab=vocab, seq=seq, d_model=d, layers=2, **VARIANTS[variant]))


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_variant_isa_and_ir_executors_match_reference(variant, dev, rng):
    g = Graph()
    ps = ParamStore(g, 2)
    _, logits = _model(variant).build(g, ps, 2, DType.FP32)
    g.output("logits", logits)
    feeds = {"x": rng.integers(0, 13, (12, 1))}
    ref, _ = run_reference(g, feeds)
    isa = Device(dev).run(compile_graph(g, dev), feeds, "functional").outputs["logits"]   # one-hot GEMM lowering
    ir = compile_graph(g, dev, CompileOptions(emit_isa=False))
    fast = FastDevice(dev).run(ir, feeds).outputs["logits"]
    for out in [isa, fast] + ([__import__("titan.runtime.accel", fromlist=["x"]).AccelDevice(dev).run(
            ir, feeds).outputs["logits"]] if _torch() else []):
        np.testing.assert_allclose(out, ref["logits"], rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_variant_accounting_is_exact(variant):
    m = _model(variant, vocab=29, seq=8, d=32)
    g = Graph()
    ps = ParamStore(g, 0, meta=True)
    _, logits = m.build(g, ps, 3, DType.FP32)
    spec = TransformerSpec.from_config(m.config)
    assert g.param_count() == spec.params()
    assert g.matmul_flops() == 3 * spec.fwd_flops_per_seq()


def test_token_and_head_ops_gradients(rng):
    """Finite differences on the float64 reference: embedding, split/merge/repeat/sum head layouts."""
    g = Graph()
    table = g.param("table", rng.standard_normal((7, 6)))
    ids = g.const(rng.integers(0, 7, (8, 1)), DType.INT32)
    e = g.embedding(ids, table)                                      # [8, 6], batch 2, T 4
    h = g.split_heads(e, 2, 3)                                       # [2*3*4, 2]
    k = g.repeat_heads(g.split_heads(e, 2, 1), 2, 1, 3)                 # GQA: 1 kv head -> 3 heads
    out = g.merge_heads(g.mul(h, g.split_heads(g.merge_heads(g.split_heads(e, 2, 3), 2, 3), 2, 3)), 2, 3)
    out = g.add(out, g.merge_heads(g.mul(g.split_heads(e, 2, 3), g.split_heads(e, 2, 3)), 2, 3))
    w = g.const(rng.standard_normal(out.shape))
    loss = g.add(g.sum(g.mul(out, w), 2), g.sum(g.mul(k, g.const(rng.standard_normal(k.shape))), 2))
    (gt,) = gradients(g, loss, [table])
    env = evaluate(g, {})
    num = np.zeros(table.shape)
    for idx in np.ndindex(table.shape):
        for sgn in (1, -1):
            v = table.value.copy()
            v[idx] += sgn * 1e-6
            num[idx] += sgn * evaluate(g, {}, {"table": v})[loss.id][0, 0] / 2e-6
    np.testing.assert_allclose(env[gt.id], num, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("variant", ["llama", "postln"])
def test_variants_train_identically_on_all_executors(variant, tmp_path):
    txt = tmp_path / "c.txt"
    txt.write_text((CORPUS + "\n") * 6)
    runs = {}
    for backend in ["functional", "fast"] + (["accel"] if _torch() else []):
        tr = Trainer(TrainConfig(data=str(txt), backend=backend, steps=12, eval_every=12, ckpt_every=0,
                                 overrides=dict(VARIANTS[variant]), out_dir=str(tmp_path / backend)), log=None)
        s = tr.run()
        tr.close()
        runs[backend] = [h["loss"] for h in tr.history]
        assert s["eval_loss"] < s["initial_loss"]
    for b, losses in runs.items():
        np.testing.assert_allclose(losses, runs["fast"], rtol=2e-4, err_msg=b)


@pytest.mark.parametrize("backend,precision", [("fast", "fp32"), ("fast", "bf16"), ("accel", "fp32"),
                                               ("accel", "bf16")])
def test_compiled_plan_is_bit_identical_to_interpreter(backend, precision, tmp_path):
    if backend == "accel" and not _torch():
        pytest.skip("PyTorch not installed")
    res = []
    for compiled in (False, True):
        tr = Trainer(TrainConfig(model="transformer", profile="tiny", backend=backend, precision=precision, steps=6,
                                 eval_every=6, ckpt_every=0, overrides=dict(VARIANTS["llama"]),
                                 out_dir=str(tmp_path / str(compiled))), log=None)
        tr.dev.compiled = compiled
        s = tr.run()
        tr.close()
        res.append(([h["loss"] for h in tr.history], tr.dev.params(tr.train_exe), s["eval_loss"]))
    assert res[0][0] == res[1][0] and res[0][2] == res[1][2]
    for k, v in res[0][1].items():
        np.testing.assert_array_equal(v, res[1][1][k], err_msg=k)


# ------------------------------------------------------------------ tokenizers

def test_tokenizers_roundtrip_and_reload(tmp_path):
    text = "Grüße aus München! TITAN trains transformers. 東京 🚀 " * 20
    bpe = BPETokenizer.train([text], 300)
    assert 256 < bpe.vocab_size <= 300
    ids = bpe.encode(text)
    assert bpe.decode(ids) == text and len(ids) < len(text.encode())
    bpe.save(tmp_path / "bpe.json")
    again = load_tokenizer(str(tmp_path / "bpe.json"))
    assert again.encode(text) == ids and again.fingerprint == load_tokenizer(str(tmp_path / "bpe.json")).fingerprint
    assert ByteTokenizer().decode(ByteTokenizer().encode(text)) == text
    ch = CharTokenizer.fit(text)
    assert ch.decode(ch.encode(text)) == text and load_tokenizer("char", fit_text=text).vocab_size == ch.vocab_size


def test_plugin_and_huggingface_tokenizers(tmp_path):
    plug = tmp_path / "mytok.py"
    plug.write_text(textwrap.dedent("""
        class Lower:
            vocab_size = 27
            def encode(self, s):
                return [ord(c) - 96 if 'a' <= c <= 'z' else 0 for c in s.lower()]
            def decode(self, ids):
                return ''.join(chr(i + 96) if i else ' ' for i in ids)
    """))
    tok = load_tokenizer(f"{plug}:Lower")
    assert tok.kind == "plugin" and tok.vocab_size == 27 and tok.decode(tok.encode("abc xyz")) == "abc xyz"
    hf = pytest.importorskip("tokenizers")
    t = hf.Tokenizer(hf.models.BPE(unk_token="[UNK]"))
    t.pre_tokenizer = hf.pre_tokenizers.Whitespace()
    t.train_from_iterator([CORPUS], hf.trainers.BpeTrainer(vocab_size=120, special_tokens=["[UNK]"]))
    t.save(str(tmp_path / "tokenizer.json"))
    h = load_tokenizer(str(tmp_path / "tokenizer.json"))
    assert h.kind == "huggingface" and h.vocab_size <= 120 and len(h.encode("the cat")) > 0


def test_token_corpus_and_training_with_any_tokenizer(tmp_path):
    txt = tmp_path / "a.txt"
    txt.write_text((CORPUS + "\n") * 30)
    bpe = BPETokenizer.train([txt.read_text()], 320)
    bpe.save(tmp_path / "bpe.json")
    meta = prepare_tokens([txt], tmp_path / "corpus.bin", bpe)
    assert meta["dtype"] == "uint16" and meta["vocab"] == bpe.vocab_size
    ds = load_tokens(tmp_path / "corpus.bin", 16)
    assert isinstance(ds.x_train, np.memmap) and ds.classes == bpe.vocab_size
    for data, tok in ((tmp_path / "corpus.bin", "char"), (txt, str(tmp_path / "bpe.json"))):
        tr = Trainer(TrainConfig(data=str(data), tokenizer=tok, backend="fast", steps=25, eval_every=25, ckpt_every=0,
                                 overrides={"vocab": 384}, out_dir=str(tmp_path / f"r{data.suffix}")), log=None)
        s = tr.run()
        tr.close()
        cfg = json.loads((tr.out / "config.json").read_text())
        assert cfg["model"]["vocab"] == 384 and cfg["tokenizer"]["kind"] == "bpe"   # padded vocab, recorded tokenizer
        assert s["eval_loss"] < s["initial_loss"]


def test_model_config_file_plugin_and_errors(tmp_path):
    txt = tmp_path / "a.txt"
    txt.write_text((CORPUS + "\n") * 6)
    cfg = tmp_path / "arch.toml"
    cfg.write_text('[model]\nnorm = "rmsnorm"\nactivation = "swiglu"\nposition = "rope"\nheads = 4\nkv_heads = 1\n'
                   'd_model = 32\nbias = false\n')
    tr = Trainer(TrainConfig(data=str(txt), model_config=str(cfg), backend="fast", steps=2, eval_every=0,
                             ckpt_every=0, overrides={"layers": 3}, out_dir=str(tmp_path / "toml")), log=None)
    c = tr.model.config
    assert (c.norm, c.activation, c.kv_heads, c.layers, c.d_model) == ("rmsnorm", "swiglu", 1, 3, 32)
    assert "tlm.b2.ff3.w" in tr.train_exe.params and "tlm.b0.attn.q.b" not in tr.train_exe.params
    tr.run()
    tr.close()
    plug = tmp_path / "mymodel.py"
    plug.write_text(textwrap.dedent("""
        from titan.nn import TransformerConfig, TransformerLM
        def build(vocab, seq, d_model=16, layers=1):
            return TransformerLM(TransformerConfig(vocab=vocab, seq=seq, d_model=d_model, layers=layers,
                                                   heads=2, name="mine"))
    """))
    tr = Trainer(TrainConfig(data=str(txt), model_file=f"{plug}:build", backend="fast", steps=3, eval_every=3,
                             ckpt_every=0, out_dir=str(tmp_path / "plug")), log=None)
    assert tr.model.name == "mine" and "mine.head.w" in tr.train_exe.params
    tr.run()
    tr.close()
    with pytest.raises(ValueError, match="unknown transformer settings"):
        Trainer(TrainConfig(data=str(txt), overrides={"nrom": "rmsnorm"}, out_dir=str(tmp_path / "e")), log=None)
    with pytest.raises(ValueError, match="kv_heads"):
        TransformerConfig(vocab=10, heads=4, kv_heads=3, d_model=16)
    with pytest.raises(ValueError, match="smaller than the tokenizer"):
        Trainer(TrainConfig(data=str(txt), overrides={"vocab": 4}, out_dir=str(tmp_path / "v")), log=None)


def test_cli_tokenizer_prepare_and_custom_quickstart(tmp_path, capsys):
    txt = tmp_path / "a.txt"
    txt.write_text((CORPUS + "\n") * 30)
    tok = tmp_path / "tok.json"
    assert main(["tokenizer", "train", str(txt), "--vocab", "300", "--out", str(tok)]) == 0
    assert "characters/token" in capsys.readouterr().out
    assert main(["tokenizer", "info", str(tok), "--text", "hello"]) == 0
    assert '"roundtrip": true' in capsys.readouterr().out
    assert main(["prepare-data", str(txt), "--tokenizer", str(tok), "--out", str(tmp_path / "c.bin")]) == 0
    assert "bpe" in capsys.readouterr().out
    arch = tmp_path / "arch.json"
    arch.write_text(json.dumps({"norm": "rmsnorm", "activation": "swiglu", "position": "rope"}))
    assert main(["quickstart", "--data", str(tmp_path / "c.bin"), "--model-config", str(arch), "--set", "kv_heads=1",
                 "tie_embeddings=true", "--steps", "10", "--eval-every", "10", "--ckpt-every", "0",
                 "--out", str(tmp_path / "run")]) == 0
    cfg = json.loads((tmp_path / "run" / "config.json").read_text())
    assert cfg["model"]["kv_heads"] == 1 and cfg["model"]["tie_embeddings"] is True
    assert cfg["backend_kind"] in ("ir", "ir-accel") or cfg["train_config"]["backend"] in ("fast", "accel")


def test_compile_auto_by_default_and_bf16_program_detection(tmp_path):
    assert TrainConfig(model="transformer").torch_compile is None                   # auto
    if not _torch():
        pytest.skip("PyTorch not installed")
    from titan.runtime.plan import reduced_gemm_program
    tr = Trainer(TrainConfig(model="transformer", profile="tiny", backend="accel", precision="bf16", steps=1,
                             eval_every=0, ckpt_every=0, out_dir=str(tmp_path)), log=None)
    assert reduced_gemm_program(tr.dev, tr.train_exe) == bool(tr.dev.exact_reduced)
