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

## 2. Execution backends (choose with `--backend`)

| Backend | Executes | Use for |
|---|---|---|
| `functional` | every TITAN ISA instruction (bit-exact replay) | verification, small models |
| `cycle` | as functional + simulated TITAN-X1 timing | architecture studies |
| `fast` | compiled TITAN IR kernels on whole tensors (NumPy) | large models without PyTorch |
| `accel` | compiled TITAN IR kernels via PyTorch on GPU or all CPU cores | maximum speed and size |
| `auto` | `accel` if PyTorch is installed, else the best of the above | default for big runs |

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

## 5. Quality controls (defaults for transformers)

AdamW (decoupled weight decay 0.1 on matrices), linear warmup + cosine decay to 10 % of the
learning rate, global gradient-norm clipping at 1.0 computed on TITAN (logged as `grad_norm`),
best-checkpoint saving (`checkpoints/best.npz`) and optional early stopping. Override with
`--schedule`, `--warmup`, `--weight-decay`, `--clip-grad`, `--early-stop`, `--lr`.

## 6. Limits, honestly

| Resource | Limit it sets |
|---|---|
| host RAM / GPU memory | trainable parameters ≈ memory / 24 bytes (minus activations) |
| storage | corpus size (1 byte per token) and checkpoints (16 bytes per parameter) |
| GPU / cores | speed; `accel` uses one GPU per process |

Multi-GPU and multi-node training are not implemented in the runtime; models beyond a single
machine (e.g. 100 B parameters ≈ 1.6 TB of weights, gradients and Adam states) are planned with
`titan plan` for the TITAN-X1 target system. Host speed is the speed of the host hardware running
TITAN's compiled program, not of TITAN silicon (which does not exist yet).
