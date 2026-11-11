# Donor layer (`titan.donor`)

Tools for taking tensors out of a published checkpoint that does not fit on the machine and bringing
them into another model's format, with every step recorded. Nothing here needs a GPU or a local copy of the
donor: weights are read by HTTP range from the Hub, one tensor (or one expert) at a time.

Three kinds of operations are kept apart everywhere (code, manifests, reports):

| class        | what it is                                                                 | evidence                         |
|--------------|----------------------------------------------------------------------------|----------------------------------|
| COPY         | donor payload bytes, unchanged                                             | SHA-256 equals the Hub byte range |
| PROJECTION   | deterministic arithmetic (bridges folded into the weights, padding, unit selection) | formula + SHA-256 of inputs and outputs |
| HEAL         | anything trained                                                           | training log, held-out metric     |

A projected tensor is never called a copy, and an initialisation is never called a result.

## Modules

| module                    | purpose |
|---------------------------|---------|
| `titan.donor.st`          | safetensors by range request: header, tensor table, per-tensor SHA-256, exact BF16<->F32, writer |
| `titan.donor.fp8`         | FP8 E4M3/E5M2 tables, block-scaled checkpoints (`W = fp8 * scale` per 128x128 block) |
| `titan.donor.hf`          | pinned-revision Hub access, provenance (files, sizes, LFS SHA-256), merged parallel range reads |
| `titan.donor.vocab`       | tokenizer readers (seed byte-BPE, HF byte-level BPE, Tekken), vocabulary contract (exact / composed / unknown), row transplantation, byte-boundary alignment of two tokenisations |
| `titan.donor.qwen3moe`    | Qwen3-MoE forward, layer by layer, with routing statistics per token group |
| `titan.donor.mistral_l4`  | Mistral Large 4 text stack (latent attention, LeRoPE, NoPE layers, key sharing, scalable softmax, shared expert, FP8), experts streamed one at a time |
| `titan.donor.calib`       | calibration text with known origin |
| `titan.donor.bridge`      | ridge bridges, data-free anchor bridges from token tables, held-out R^2 |
| `titan.donor.adapt`       | donor expert -> seed MoE slot: folding bridges into `w1/w3/w2`, zero-padding (exact), unit selection (lossy, reported), insertion with selection bias (layer output unchanged) |

Scripts in `tools/donor/`: `qwen_forward.py`, `mistral_probes.py`, `mistral_forward.py`,
`output_bridge.py`, `qwen_expert_acts.py`, `qwen_implant.py`, `mistral_select.py`, `mistral_implant.py`,
`mistral_out_pos.py`, `build_implant.py`, `insert_implant.py`, `transplant_seed.py` (dense seed checkpoint -> the receiving team's own upcycle -> implants -> forward check with the team's own code), `verify_package.py` (checks a delivery package split into byte parts: each part, the whole ZIP, every file against the package manifest).

## Implants

`build_implant.py` stacks the projected experts of a donor into one file in the receiving MoE's tensor layout
(`implant.experts.w1 / w3 [N, fe, d]`, `implant.experts.w2 [N, d, fe]`, `implant.router.weight [N, d]`) with an
index of where every row comes from. `insert_implant.py` puts such files into a checkpoint (`model.safetensors`,
`optim.safetensors`, `state.json`): rows are spread over the target layers in donor-depth order, every MoE layer
grows by the same number of slots (filler slots with zero output where needed), rows are adapted to the layer's
`ffn_norm` gain, new slots get a selection bias that keeps them out of the top-k, everything else is copied byte
for byte, and the file is streamed so memory does not depend on the checkpoint size. After insertion the model
computes the same function; whether a slot becomes useful is training.

## What the tests establish

`tests/test_donor_reference.py` (needs `torch` + `transformers`): the streamed forwards equal the
`transformers` reference implementations on random weights (Qwen3-MoE; Mistral Large 4 including NoPE
layers, key sharing, LeRoPE, the dense first layer and the shared expert), and FP8 dequantisation equals
`torch.float8_e4m3fn`.

`tests/test_donor_implant.py`: implant files are byte copies in the receiving layout; insertion keeps every old
tensor and row bit for bit, keeps routing and outputs, adapts rows to the layer gain, extends the optimizer with
zeros and refuses settings that would change the model.

`tests/test_donor.py`: safetensors round trip and hashes; FP8 table and block scales; vocabulary contract
and boundary alignment; ridge and anchor bridges; folding equals composition (`slot(x) == P_out(expert(x @ P_in))`),
zero-padding adds exact zeros, unit selection is lossy; inserted slots leave routing and old slots untouched (outputs equal up to floating-point rounding).

## What it does not do

- It does not make a donor expert useful in the receiving model. Anchor bridges are initialisations from the
  token tables (depth 0 / output side). Whether they hold at an inner layer has to be measured with paired
  activations of both models, and making inserted experts contribute is training (HEAL).
- It does not judge model quality. Numbers it prints (cross-entropy on a probe set, routing contrasts, R^2)
  describe the donor on that probe set, nothing else.
