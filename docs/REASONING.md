# Mixture of experts, Frankenstein models, post-training, reasoning and test-time compute

Everything here runs on the TITAN program pipeline (graph → autodiff → compiler → executors) and is
covered by `tests/test_moe_reasoning.py`. Numbers are measured on this repository's 4-core CPU host
unless stated otherwise; the reasoning numbers come from `examples/reasoning/arith.py`
(report: `reports/reasoning/arith_report.json`).

## 1. Mixture of experts (MoE)

```bash
python -m titan quickstart --data book.txt --set experts=8 top_k=2 capacity_factor=1.25 moe_every=1
```

| key | meaning | default |
|---|---|---|
| `experts` | experts per MoE layer (0 = dense MLP) | 0 |
| `top_k` | experts per token; gates = softmax renormalized over the selected experts | 2 |
| `capacity_factor` | expert capacity = ceil(cf · tokens · k / experts); overflow tokens skip that expert (residual path) | 1.25 |
| `moe_every` | MoE in every n-th block (others dense) | 1 |
| `moe_aux` | Switch load-balancing loss coefficient (E/k · Σ fraction · mean prob) | 0.01 |
| `shared_expert` | always-active dense MLP added to the experts (DeepSeek-style) | false |

Implementation (`titan/nn/transformer.py`): router GEMM → top-k mask (no gradient) → masked softmax
→ `moe_slots` routing table (new TITAN IR op: token ids per expert slot, -1 = empty) → gather →
expert-batched GEMMs (all experts in one batched product) → gate → scatter-add. Evaluation and
generation run without capacity drops. On the TITAN ISA path the same layer uses dense dispatch
(every expert sees every token, gate 0 when unselected): identical math without drops, verified
against the sparse IR path bit for bit (float64 reference) and against the ISA executor.

Verified: sparse = dense without drops (exact), ISA/fast/accel = float64 reference, finite-difference
gradients of router and experts (incl. capacity drops and the auxiliary loss), exact parameter /
active-parameter / FLOP accounting (`TransformerSpec.fwd_flops(batch, dense_moe)`).

Measured (accel, d_model 256, 4 layers, seq 64, batch 8):

| model | parameters | active | forward GFLOP/step | ms/step |
|---|---|---|---|---|
| dense, d_ff 1024 | 3.19 M | 3.19 M | 3.36 | 66.9 |
| dense, d_ff 2560 (same FLOPs as the MoE) | 6.34 M | 6.34 M | 6.58 | 134.2 |
| MoE 8 × 1024, top-2 | 17.91 M | 5.30 M | 6.59 | 155.8 |

The MoE holds 2.8× the parameters of the equal-FLOP dense model for 16 % more step time (routing,
gathers and the host-side routing table).

## 2. Frankenstein models (`titan merge`)

```bash
titan merge average          --models runA runB runC [--weights 2 1 1] --out soup
titan merge slerp            --models runA runB --t 0.3 --out mix
titan merge task-arithmetic  --base pre --models math code --scale 0.8 --out multi
titan merge ties             --base pre --models math code --density 0.2 --out multi
titan merge stack            --models runA runB --layers 0:0-5 1:2-7 --out deep    # depth up-scaling
titan merge upcycle-moe      --models dense --experts 8 --top-k 2 --noise 0.01 --out moe
```

The output is a run directory: `titan generate deep ...` works directly; training continues with
`titan quickstart --data ... --model-config deep/model.json --init-from deep/checkpoints/best.npz
--tokenizer deep/tokenizer.json`. Verified: averaging a model with itself, SLERP at t=0, task
arithmetic with λ=0 and TIES of a single model reproduce the model; stacking all layers in order is
the identity; MoE upcycling preserves the function exactly (identical experts, renormalized gates);
a stacked 4-layer model loads every weight and keeps training. Models must share tokenizer and
widths (stacking may change depth only); LoRA runs (adapters only) cannot be merged.

## 3. Post-training: SFT with response-only loss

```jsonl
{"prompt": "47+38=", "response": "7+8=15,4+3+1=8:85"}
{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}
{"text": "plain pre-training text mixed in"}
```

`titan quickstart --data sft.jsonl [--init-from pretrained.npz --tokenizer run/tokenizer.json]`.
Examples are packed back to back (no padding); training windows start at example boundaries;
prompt tokens are inputs but not targets (`xent(weighted=True)`: gradient
(softmax·rowsum(y) − y)/rows, verified by finite differences). Every run writes `tokenizer.json`, so
fine-tuning and generation use the exact vocabulary.

## 4. Generation and test-time compute

```bash
titan generate RUN --prompt "47+38=" --strategy self-consistency --n 8 --temperature 0.8
titan reason RUN --data heldout.jsonl --strategy greedy self-consistency best-of-n beam --n 8 \
      --verifier examples/reasoning/verifier.py:check
```

Generation uses a compiled one-token decode program with a key/value cache (`build_decode`):
O(t) per token instead of re-running the window. Verified equal to the full forward pass (≤ 2e-5)
for learned/RoPE/sinusoidal positions, GQA, post-LN and MoE, on fast and accel. Sampling:
temperature, top-k, top-p, stop strings; beam search with cache reordering.

Strategies: `greedy`, `sample`, `best-of-n` (verifier `file.py:fn(prompt, completion) -> score`, or the
model's mean token log-probability), `self-consistency` (majority vote over extracted answers),
`beam`. `titan reason` reports accuracy, generated tokens and wall time per strategy.

## 5. Reasoning post-training: rejection-sampling fine-tuning (STaR)

```bash
titan rft RUN --data train_problems.jsonl --eval-data heldout.jsonl --rounds 2 --samples 4 --steps 400
```

Problems need only `{"prompt", "answer"}`. Each round samples solutions, keeps verified ones
(answer match or `--verifier`), fine-tunes on them (response-only loss) and measures held-out
accuracy before and after.

## 6. Measured: `examples/reasoning/arith.py` (2-digit addition with chain of thought)

153 k-parameter transformer (d 64, 3 layers), SFT on 400 worked solutions, 600 steps (24 s).
300 held-out problems (beam: first 100):

| strategy | samples | accuracy | generated tokens | wall |
|---|---|---|---|---|
| greedy | 1 | 75.3 % | 5,593 | 0.30 s |
| sample (T 0.8) | 1 | 70.7 % | 5,586 | 0.31 s |
| self-consistency | 8 | 73.7 % | 44,710 | 2.4 s |
| best-of-n, model log-prob | 8 | 75.3 % | 44,710 | 2.5 s |
| **best-of-n, process verifier** (checks each step, not the answer) | 8 | **82.0 %** | 44,710 | 2.5 s |
| beam | 4 | 77.0 % | 7,292 | 1.7 s |

Decoding: KV cache 17.6–18.6 k tokens/s vs. 2.3–2.4 k tokens/s with full-window recomputation (7.5–7.8×, two runs),
identical outputs. RFT on 2,000 answer-only problems (4 samples each): held-out greedy accuracy
**75.3 % → 95.0 % (round 1) → 98.0 % (round 2)**.

Reading: sampling-based voting does not help this small, deterministic-format model (sampling
noise exceeds its diversity benefit); a verifier turns extra samples into accuracy; training on
self-generated verified solutions (RFT) gives the largest gain. These are small-model results on a
synthetic task, not claims about large language models.
