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
python -m titan quickstart --model transformer --profile small
python -m titan bench --suite default          # reproducible report (reports/)
python -m pytest -q                            # 274 tests (RTL tests need Verilator)
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

## Performance honesty

* `titan-x1-target` is a **research design point**: 352 cores × 128×128 MXU at an assumed 2 GHz has
  a BF16 dense *paper peak* of 23.1 PFLOPS = 10.25× the B300's 2.25 PFLOPS (same precision, dense).
  Its assumed memory bandwidth (32 TB/s) is only 4× the B300's, so memory-bound workloads would
  scale by ~4×, not 10×. Nothing of this configuration has been built or measured.
* The host CPU running the emulator is not the accelerator: host wall-clock numbers are labelled
  *measured (host)*; accelerator timing is *simulated* or *analytical*, never mixed.
* B300 figures (`titan/perf/reference.py`) carry sources and retrieval date; the code refuses
  cross-precision, sparse-vs-dense and simulated-vs-peak ratios.

Documentation: [Architecture](docs/ARCHITECTURE.md) · [ISA](docs/ISA.md) · [Performance](docs/PERFORMANCE.md) ·
[Verification](docs/VERIFICATION.md) · [FPGA](docs/FPGA.md) · [Roadmap](docs/ROADMAP.md) ·
[Status](docs/PROJECT_STATUS.md) · [M0 audit](docs/AUDIT_M0.md)
