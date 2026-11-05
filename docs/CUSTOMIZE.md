# Bring your own transformer and tokenizer

Every model below trains with the same TITAN program pipeline (graph → autodiff → AdamW/clipping →
compiler → executor), so checkpoints, resume, fine-tuning (`--init-from`, `--freeze`), LoRA, storage
offload and all backends (ISA-level `functional`/`cycle`, compiled `fast`, PyTorch `accel`) work for
custom architectures and tokenizers without extra code.

## 1. Architecture: config file, `--set`, or Python

```bash
python -m titan quickstart --data book.txt --model-config examples/models/llama_style.toml
python -m titan quickstart --data book.txt --set norm=rmsnorm activation=swiglu position=rope kv_heads=2
```

Precedence: profile (`--profile tiny|small|medium`) ← `--model-config` file ← `--set` / `--params`.
JSON or TOML; a `[model]` section is accepted. All keys (`titan.nn.TransformerConfig`):

| key | values | default |
|---|---|---|
| `d_model`, `layers`, `heads`, `seq` | positive ints (`d_model % heads == 0`) | from profile |
| `kv_heads` | divides `heads`; `< heads` = grouped-query / multi-query attention | `heads` |
| `d_ff` | MLP width | `4·d_model`; gated: `8/3·d_model` rounded to 32 |
| `norm` | `layernorm`, `rmsnorm` | `layernorm` |
| `norm_eps` | float | `1e-5` |
| `activation` | `gelu`, `relu`, `silu`, `swiglu`, `geglu` | `gelu` |
| `position` | `learned`, `rope`, `sinusoidal`, `none` | `learned` |
| `rope_base` | float | `10000` |
| `prenorm` | `true` (pre-LN, GPT-2/Llama) / `false` (post-LN, original Transformer) | `true` |
| `bias` | biases on output projections, MLP, LM head | `true` |
| `qkv_bias` | biases on q, k, v | `false` |
| `tie_embeddings` | LM head = token embedding transposed | `false` |
| `causal` | causal mask | `true` |
| `vocab` | optional; ≥ tokenizer vocabulary (pad e.g. to a multiple of 64) | tokenizer size |
| `name` | parameter-name prefix | `tlm` |

Typical presets: GPT-2 = defaults + `qkv_bias=true`; Llama = `rmsnorm, swiglu, rope, bias=false,
kv_heads<heads`; Mistral-like = Llama + GQA; original Transformer decoder = `prenorm=false,
position=sinusoidal, activation=relu`.

Implementation notes (`titan/nn/transformer.py`): attention runs fused across all heads
(`split_heads` → two head-batched matrix products → `merge_heads`); GQA shares key/value heads
with `repeat_heads` (gradient: `sum_head_groups`); token and learned position embeddings are row
gathers (`embedding`; gradient: scatter-add `embedding_grad`); next-token targets are ids and their
one-hot encoding is built on TITAN (`onehot`). On the ISA level each of these lowers to TITAN
instructions (one-hot GEMMs, strided COPY, ACCUMULATE); the IR executors use dedicated kernels.

Parameter/FLOP accounting is exact for every variant: `TransformerSpec.from_config(cfg).params()`
and `fwd_flops_per_seq()` equal the compiled graph (`tests/test_customize.py`), so `titan plan` and
`--params` sizing are valid for custom architectures.

## 2. Your own model code (`--model-file`)

```bash
python -m titan quickstart --data book.txt --model-file examples/models/my_model.py:build
python -m titan quickstart --data book.txt --model-file mypkg.models:make_lm
```

Contract: `factory(vocab=..., seq=..., **settings)` returns an object with `name`, `seq`,
`kind = "lm"` and `build(g, ps, batch, compute) -> (ids_input, logits)` where `ids_input` is
`g.input("x", (batch*seq, 1), DType.INT32)` and `logits` is `[batch*seq, vocab]`. Settings from
`--model-config`/`--set`/profile are passed if the factory declares them (or takes `**kwargs`).
Subclassing `TransformerLM` and overriding `_attention`, `_mlp`, `_norm` or `_block` is the shortest
route (see `examples/models/my_model.py`, a two-branch mixture MLP). Use only TITAN graph ops
(`titan/compiler/graph.py`) and `ParamStore.get` for weights; gradients come from TITAN autodiff.

## 3. Tokenizers (`--tokenizer`)

| spec | tokenizer |
|---|---|
| `char` (default for `.txt`) | characters of the training text (≤ 512) |
| `byte` | UTF-8 bytes, vocabulary 256 (transferable between corpora) |
| `tok.json` from `titan tokenizer train` | byte-level BPE, any vocabulary size, lossless on any UTF-8 |
| `tokenizer.json` | Hugging Face `tokenizers` file (e.g. from a pretrained model; `pip install tokenizers`) |
| `*.model` | SentencePiece model (`pip install sentencepiece`) |
| `file.py:Class` / `pkg.mod:factory` | your own: `encode(str) -> list[int]`, `decode(ids) -> str`, `vocab_size` |

```bash
python -m titan tokenizer train corpus/*.txt --vocab 8192 --out tok.json      # train BPE
python -m titan tokenizer info tok.json --text "Hello TITAN"                  # inspect, round-trip check
python -m titan prepare-data corpus/*.txt --tokenizer tok.json --out corpus.bin   # pre-tokenize (uint16/32)
python -m titan quickstart --data corpus.bin --model-config arch.toml --params 300m
```

`prepare-data` writes a memory-mapped token file plus a JSON sidecar (vocabulary, dtype, token count,
SHA-256 of every source and of the token file, tokenizer description and fingerprint); training
reads only the sampled windows, so the corpus size is bounded by storage. Each run records the
tokenizer in `config.json`. The built-in BPE trainer is pure Python (fine for tens of MB); for
very large corpora train a Hugging Face tokenizer (Rust) and pass its `tokenizer.json`.

## 4. Fine-tuning a custom model

The usual flags apply: `--init-from` copies every weight whose name and shape match (a new
tokenizer/vocabulary re-initializes only the embedding and head), `--freeze` takes name prefixes or
globs (e.g. `tlm.tok_emb 'tlm.b0.*'`), `--lora-rank` adds adapters to every linear layer.
