# Training on your own environment

ROUGE TITAN runs wherever Python runs. Connect more RAM, larger storage or GPUs and every limit
moves with the hardware: model size (RAM / GPU memory), corpus size (disk), speed (GPU / cores).

## 1. Install

```bash
git clone <this repository> && cd Ezhdhhz
pip install -e ".[test]"            # core (NumPy)
pip install -e ".[accel]"           # + PyTorch: accel backend on GPU (CUDA) or all CPU cores
# or: docker build -t titan --build-arg TORCH=cuda .   (see Dockerfile)
python -m titan env --path /mnt/storage   # what this machine allows
```

`titan env` reports CPU, RAM, free storage at the path you will use, PyTorch/GPUs, the
recommended backend and estimated limits (24 bytes per trained parameter + activations).

Custom architectures and tokenizers (`--model-config`, `--set`, `--model-file`, `--tokenizer`,
`titan tokenizer train`, `prepare-data --tokenizer`): [CUSTOMIZE.md](CUSTOMIZE.md).

## Connected hardware: discover, measure, use all of it

```bash
titan hardware                       # every CPU/GPU/XPU/MPS device, measured GEMM TFLOP/s and bandwidth
titan quickstart ... --accelerators auto        # all GPUs (>= 2) train data parallel automatically
titan quickstart ... --accelerators cuda:0,cuda:2
```

`titan hardware` finds what PyTorch can drive here (CUDA incl. ROCm builds, Intel XPU, Apple MPS,
the CPU with AVX-512/AMX), measures dense FP32 and exact-BF16 GEMM throughput and copy bandwidth
on each, and prints the fastest configuration. With two or more accelerators the trainer runs data
parallel by default (`titan/runtime/multidevice.py`): every device runs the compiled gradient
program on 1/N of the batch (launched together, asynchronous on GPUs), gradients are averaged,
and every device applies the same compiled update — identical replicas, and the same update as a
single large-batch step (tested with 2 and 4 replicas: losses and weights within 4·10⁻⁷).
Defaults pick the fastest measured exact path: `--precision auto` uses BF16 operands with FP32
accumulation where AMX/TF32 exists and it is measured faster (eager: d_model ≥ 1024; with
`--compile on`: d_model ≥ 256; eager BF16 at d_model 256 is slower, 67.9 vs 55.1 ms/step), else FP32;
`--compile auto` compiles runs longer than 10 min. `titan hardware --train-benchmark` measures the
training speed-up of all connected accelerators and of the precision defaults on YOUR machine.
Measured on this 4-core host: FP32 GEMM 0.57 TFLOP/s, exact BF16 GEMM 1.33 TFLOP/s (AMX),
55 GB/s copy; 3.2 M transformer, steady state: FP32 eager 75.8 ms/step → `--precision bf16
--compile on` 25.3 ms/step (3.0×), same loss (0.6179 vs 0.6176). Multi-GPU speed is not measured here (no GPU); data-parallel replicas on one CPU are
correct but not faster (the CPU's cores already run every kernel in parallel).

## 2. Execution backends (choose with `--backend`)

| Backend | Executes | Use for |
|---|---|---|
| `functional` | every TITAN ISA instruction (bit-exact replay) | verification, small models |
| `cycle` | as functional + simulated TITAN-X1 timing | architecture studies |
| `fast` | the compiled TITAN program as ONE generated NumPy function (`titan/runtime/plan.py`) | small/medium models, storage offload |
| `accel` | the same plan via PyTorch on GPU or all CPU cores (exact BF16 GEMMs on AMX / TF32) | large models, GPUs |
| `auto` (default) | `accel` on a CUDA GPU or for `d_model ≥ 128`, else `fast` (measured crossover) | everything |

`--compile auto` (default) runs the FP32 accel program through `torch.compile` once a run's
measured remaining time exceeds 10 min (one-time 0.5–2 min compile, then 1.3–5× faster steps
measured on CPU); `--compile on|off` forces it. With `--precision bf16` the compiled program uses
the exact AMX/TF32 GEMM path: fastest mode measured (100.9 M: 706 vs 1,272 ms/step FP32 eager).

All backends run the same compiled program (autodiff, fusion, optimizer on TITAN) with TITAN
numerics; `fast`/`accel` differ from ISA level only in FP32 summation order. Measured on a
4-core host: 101 M-parameter transformer step 7.2 s (`fast`) → 2.7 s (`accel`, CPU), same loss.

## 3. Data on attached storage

```bash
# tabular: CSV / TSV / NPZ / NPY (classification or regression)
titan quickstart --data /mnt/data/table.csv --target label --out /mnt/runs/tab
# text, small: character vocabulary
titan quickstart --data /mnt/data/book.txt --out /mnt/runs/char-lm
# text, any size: byte tokens, memory-mapped from disk (RAM no longer limits the corpus)
titan prepare-data /mnt/data/corpus/*.txt --out /mnt/data/corpus.bin
```

## 4. Pretrain, continue, fine-tune

```bash
# pretrain from scratch (size by parameter count; fits are checked against this machine)
titan quickstart --data /mnt/data/corpus.bin --params 300m --backend auto --steps 20000 \
    --eval-every 500 --ckpt-every 1000 --out /mnt/runs/pre-300m
# continue training exactly where it stopped (weights, optimizer state, data position)
titan quickstart --data /mnt/data/corpus.bin --params 300m --backend auto --steps 40000 \
    --resume /mnt/runs/pre-300m/checkpoints/step_020000.npz --out /mnt/runs/pre-300m
# fine-tune on new data (weights only, fresh optimizer; optionally freeze parts)
titan quickstart --data /mnt/data/domain.txt --tokenizer byte --params 300m --backend auto --steps 2000 \
    --init-from /mnt/runs/pre-300m/checkpoints/best.npz --lr 1e-4 --freeze tlm.tok_emb tlm.pos_emb \
    --early-stop 5 --out /mnt/runs/ft-domain
```

Byte tokens (vocabulary 256) make every byte-level checkpoint transferable to any other byte
corpus. Weights whose name or shape differ are re-initialized and listed in `config.json`.

## 5. Beyond RAM: storage offload (up to 100B+ on one machine)

```bash
titan plan --params 100b --single-machine          # compiles the full training step, exact RAM/storage
titan quickstart --data /mnt/data/corpus.bin --params 100b --offload-dir /mnt/nvme/titan \
    --steps 1000 --ckpt-every 100 --out /mnt/runs/pre-100b
```

With `--offload-dir`, weights, Adam state and every tensor of at least `--offload-mb` (default 4 MiB)
live in memory-mapped files on that storage; parameters are initialized lazily, one tensor at a
time, directly into those files. RAM holds the working set of one kernel plus the page cache.
Exact accounting of the compiled GPT-100B training step (101.7 B parameters, 14,647 kernels,
sequence 128 or 2048): **10.4 GB RAM and 1.63 TB storage**. That is ~16 bytes per parameter: FP32 weights, Adam m and v (all
updated in place, no copies) plus gradients. Use NVMe: every step reads and writes the optimizer
state, so storage bandwidth sets the speed. If the storage is too small, `quickstart`
prints the exact requirement and the distributed plan instead. Measured in a 16 GB sandbox: a
705 M-parameter transformer (training state larger than the machine's RAM) trained with offload,
3 steps in 370 s on 4 CPU cores and a network disk.

### LoRA fine-tuning of very large models

`--lora-rank R` freezes every base weight and trains rank-R adapters on the MLPs and the LM head.
Frozen, lazily initialized base weights are recomputed from their seeds on every use, so memory is
independent of the base size. Measured in the 16 GB sandbox: a 100.74 B-parameter transformer,
`titan quickstart --params 100b --lora-rank 8 --seq 32 --batch 1 --eval-every 0 --backend accel`:
65.6 M trainable parameters, 12.85 Tera-ops and ~22 minutes per training step on 4 CPU cores (each
step regenerates all base weights twice), peak RAM 7.2 GB. With real pretrained weights, combine
`--init-from` with `--offload-dir` so the frozen base is read from storage instead of regenerated.

## 6. Quality controls (defaults for transformers)

GPT-2 initialization for transformers with d_model ≥ 256 (std 0.02, residual projections scaled by
1/√(2·layers); measured on a 101 M model: eval loss 2.83 vs 3.51 with Xavier after 12 steps, no early
loss spike), AdamW (decoupled weight decay 0.1 on matrices), linear warmup + cosine decay to 10 % of the
learning rate, global gradient-norm clipping at 1.0 computed on TITAN (logged as `grad_norm`),
best-checkpoint saving (`checkpoints/best.npz`) and optional early stopping. Override with
`--init`, `--schedule`, `--warmup`, `--weight-decay`, `--clip-grad`, `--early-stop`, `--lr`.

## 7. Limits, honestly

| Resource | Limit it sets |
|---|---|
| host RAM / GPU memory | in-memory training: parameters ≈ memory / 24 bytes (minus activations) |
| storage | with `--offload-dir`: parameters ≈ storage / 16 bytes (100B ≈ 1.63 TB); corpus (1 byte/token); checkpoints (16 bytes/parameter) |
| GPU / cores | speed; `accel` uses one GPU per process |

Multi-GPU and multi-node training are not implemented in the runtime: one machine trains models up
to its storage (offload) — 100B needs ~1.7 TB of NVMe and is storage-bandwidth bound; faster
large-scale training needs many devices, which `titan plan` models for the TITAN-X1 target system. Host speed is the speed of the host hardware running
TITAN's compiled program, not of TITAN silicon (which does not exist yet).
