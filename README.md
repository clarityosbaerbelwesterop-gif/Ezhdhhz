# ROUGE TITAN — TITAN-X1 digital AI accelerator

**Status:** working digital accelerator platform — own ISA, virtual hardware (functional +
cycle model), compiler, training runtime, RTL matrix engine with bit-exact co-simulation.
**Not** fabricated silicon, **no** FPGA measurement, **no** B300-class performance.

```
MODEL / FRAMEWORK      titan.nn (MLP, multi-head attention, transformer blocks), titan.train
COMPUTATION GRAPH      titan.compiler.graph  (+ autodiff: backward pass is also a graph)
TITAN IR               titan.compiler.ir     (gemm / ewise / reduce / rownorm / transpose / copy kernels)
GRAPH OPTIMIZER        fusion, liveness memory planning, barrier elimination, stream interleaving
TITAN COMPILER         tiling into LOCAL SRAM, double buffering, multi-core distribution
TITAN ISA v1.0         titan.isa  — 20 opcodes, 64-bit binary encoding, assembler, docs/ISA.md
VIRTUAL HARDWARE       titan.emu  — command processor, per-core queues, DMA/MXU/VPU/SFU, memory hierarchy
EXECUTION RESULTS      verified vs float64 NumPy reference; RTL (rtl/titan_mxu_tile.sv) bit-exact for INT8/INT4
```

## Quick start

Requirements: Python ≥ 3.11 with NumPy. Optional: Verilator (RTL), Yosys + nextpnr-ecp5 (synthesis).

```bash
pip install -e ".[test]"
python -m titan backends                       # what is really available
python -m titan matmul --m 256 --n 192 --k 128 --dtype bf16   # vertical slice, verified
python -m titan quickstart                     # train an MLP on TITAN (or: -i for guided mode)
python -m titan quickstart --data examples/data/machines.csv --target status   # your own data
python -m titan quickstart --data my_book.txt  # your own text -> character-level transformer
python -m titan quickstart --params 100m       # sized transformer, trained for real on this host
python -m titan quickstart --params 100b       # too big for any single host -> distributed TITAN plan
python -m titan plan --params 100b --devices 64 --compare-b300
python -m titan env --path /mnt/storage        # your machine's RAM/GPU/storage limits
python -m titan prepare-data corpus/*.txt --out corpus.bin   # any-size byte corpus (memory-mapped)
python -m titan quickstart --data corpus.bin --params 300m --backend auto        # pretrain
python -m titan quickstart --data domain.txt --tokenizer byte --params 300m \
       --init-from runs/.../checkpoints/best.npz --freeze tlm.tok_emb --lr 1e-4  # fine-tune
python -m titan bench --suite default          # reproducible report (reports/)
python -m pytest -q                            # 315 tests (RTL tests need Verilator)
make lint test                                 # RTL lint (warnings = errors) + 4 RTL simulations
```

## What exists (and how it is verified)

| Layer | Implementation | Evidence |
|---|---|---|
| ISA | 20 opcodes (LOAD … EXECUTE_KERNEL), validation, binary + assembly | encode/decode/asm round-trip of every opcode; spec ↔ code sync test |
| Numerics | INT4, INT8, INT32, FP16, BF16 (RNE), FP32 | per-instruction tests vs NumPy; INT32 wrap identical to RTL |
| Virtual hardware | command processor, kernels, GEMM sequencer, SIGNAL/WAIT/BARRIER, faults | functional ≡ cycle data; deadlock/fault tests |
| Memory system | GLOBAL/SHARED/LOCAL, L2 LRU cache, HBM/L2/SRAM bandwidth calendars, DMA engines | contention, cache-hit, double-buffering tests |
| Compiler | fusion, memory planning, tiling, multi-core, barrier elimination, interleaving | 60 end-to-end tests vs float64 reference |
| NN + autodiff | MLP, attention, transformer block, char-LM; reverse-mode AD | every gradient rule vs finite differences |
| Training | forward+loss+backward+optimizer as one TITAN program; checkpoints | loss reduction, bitwise resume, corruption detection |
| Own data | `.csv/.tsv/.npz/.npy` (classification or regression), `.txt` (language model) | format, validation and end-to-end tests |
| Execution speed | bit-exact replay of the ISA program; `fast` backend runs TITAN IR kernels on whole tensors | replay memory images identical to the reference engine; fast ≈ ISA within FP32 tolerance |
| Large models | meta graphs of any size, distributed planner (TP×PP×DP, ZeRO, recompute) up to 175B | parameter/FLOP formulas equal compiled graphs exactly |
| Own environment | `accel` backend (TITAN IR kernels via PyTorch on GPU/CPU), byte corpora memory-mapped from disk, pretrain / continue / fine-tune (`--init-from`, `--freeze`), AdamW + cosine + clipping + best checkpoint + early stopping, `titan env`, Dockerfile | accel ≈ fast executor; fine-tune transfer and freeze tests; see [docs/DEPLOY.md](docs/DEPLOY.md) |
| Performance | cycle model, roofline, benchmark harness, verified autotuner | `reports/` (method + provenance on every number) |
| RTL | `rtl/titan_mxu_tile.sv` (MXU tile), `rtl/rtl_mac_array.sv` (D0) | directed + 2000-cycle random TBs; Verilator co-sim 0 mismatches |
| FPGA prep | Yosys (generic, xc7, ECP5) + nextpnr-ecp5 P&R | `reports/synth/`, `docs/FPGA.md` |
| Multi-chip | chiplet/package topologies, ring all-reduce, data parallel | DP step ≡ large-batch step; replicas bit-identical |

## Selected results (from `reports/`, commit `f7bd718`)

| Result | Value | Method |
|---|---|---|
| GEMM 512³ BF16 on `titan-x1-dev` | 2.02 TOPS = 98.7 % of config peak | simulated (cycle model, 1 GHz assumed) |
| Transformer char-LM (19 k params), 300 steps | loss 3.99 → 0.91, next-char accuracy 61.7 % (chance 4.2 %) | trained on TITAN (functional) |
| MLP spirals, BF16 mixed precision | 99.6 % test accuracy | trained on TITAN (functional) |
| RTL ↔ emulator | 48 random INT8/INT4 MATMUL cases incl. overflow, 0 mismatches | Verilator co-simulation |
| 8×8 INT8 MXU tile on ECP5-85F | 64 MULT18X18D, 2048 FF, 179.8 MHz | nextpnr post-route STA estimate |
| Data parallel, 16 devices (flat ring) | 2.86× speed-up, 63 % time in all-reduce | simulated compute + modelled links (assumed) |
| MLP training, 300 steps | 5.2 s → 0.24 s (21×), identical loss trajectory | measured on the host (emulator speed) |
| Largest model trained for real on this 16 GB host | 255 M parameters (transformer, fast backend) | measured on the host |
| 101 M-parameter transformer step, 4-core host | 7.2 s (`fast`) → 2.7 s (`accel`, CPU), identical loss | measured on the host |
| Byte-level pretrain (docs) → fine-tune (code), 150 steps | eval loss 2.01 vs 2.37 from scratch | measured, `accel` backend |
| GPT-100B on 64 × `titan-x1-target` | ~2.2 s/step, ~0.96 M tokens/s, MFU 41 %; 7.0× the same plan on 64 × B300 | analytical planner, assumed links (Mode C) |

## Performance honesty

* `titan-x1-target` is a **research design point**: 352 cores × 128×128 MXU at an assumed 2 GHz has
  a BF16 dense *paper peak* of 23.1 PFLOPS = 10.25× the B300's 2.25 PFLOPS (same precision, dense).
  Its assumed memory bandwidth (32 TB/s) is only 4× the B300's, so memory-bound workloads would
  scale by ~4×, not 10×. Nothing of this configuration has been built or measured.
* The host CPU running the emulator is not the accelerator: host wall-clock numbers are labelled
  *measured (host)*; accelerator timing is *simulated* or *analytical*, never mixed.
* B300 figures (`titan/perf/reference.py`) carry sources and retrieval date; the code refuses
  cross-precision, sparse-vs-dense and simulated-vs-peak ratios.
* 100B parameters cannot be trained numerically on a single host (≈1.6 TB of weights, gradients
  and Adam states). Models up to that size and beyond are defined (meta graphs), compiled to TITAN IR
  and planned for the distributed target system; numeric training here is limited by host RAM.

Documentation: [Architecture](docs/ARCHITECTURE.md) · [ISA](docs/ISA.md) · [Performance](docs/PERFORMANCE.md) ·
[Verification](docs/VERIFICATION.md) · [FPGA](docs/FPGA.md) · [Roadmap](docs/ROADMAP.md) ·
[Status](docs/PROJECT_STATUS.md) · [M0 audit](docs/AUDIT_M0.md) · [Speed & large models](reports/SPEED_AND_LARGE_MODELS.md) · [Own environment](docs/DEPLOY.md)
