# Project status

## Summary

ROUGE TITAN now has a working digital accelerator platform: TITAN ISA v1.0, a virtual TITAN-X1
(functional + cycle model), a compiler from computation graphs to ISA, neural-network and
training runtimes that execute forward, backward and optimizer updates on the virtual hardware,
a QuickStart CLI, a performance engine, an RTL matrix-engine tile verified bit-exact against the
emulator, synthesis/P&R reports and a multi-chip scaling model.

## Evidence status

| Item | Status |
|---|---|
| Software tests | `pytest`: all passing locally (see final commit message); CI runs them on 3.11 / 3.13 |
| RTL simulation | 4 self-checking simulations pass (Verilator 5.020); Icarus cross-check passes |
| RTL lint | clean with `-Wall`, warnings fatal |
| RTL ↔ emulator | 0 mismatches (INT8 and INT4, accumulate, overflow) |
| Synthesis / P&R | Yosys 0.33 (generic, xc7, ECP5), nextpnr-ecp5 0.6 — `reports/synth/` |
| Benchmarks | `reports/bench_default_titan-x1-dev.md` (simulated + analytical + measured-host columns) |
| Training | `reports/TRAINING_AND_SCALING.md` |
| FPGA hardware | not started — no board, no measurement |
| ASIC physical design | not started |
| Fabrication | not started |

## What is incomplete (honest list)

* RTL exists only for the MXU tile and the D0 MAC array. DMA, VPU, SFU, SRAM, command processor
  and interconnect exist only in the emulator and models.
* The cycle model is transaction-level and uncalibrated against hardware or full RTL.
* No FP8/FP4, no sparsity; training is FP32 / BF16 / FP16 mixed precision; INT8/INT4 inference only.
* Graph tensors are 2-D; batched products use row stacking; no convolution operators.
* Datasets are small and synthetic (plus optional scikit-learn digits); training runs at host
  emulation speed (seconds to minutes), which says nothing about accelerator speed.
* Multi-chip results use assumed link parameters; there is no die-to-die or network RTL.
* `titan-x1-target` is an unbuilt design point; its numbers are analytical paper figures.

## Performance honesty

The 10× (package) and 100× (system) B300 goals remain research aspirations. The `titan-x1-target`
configuration reaches 10.25× B300 only as a BF16 dense *paper peak*, with 4× the memory bandwidth;
nothing has been measured on hardware. See `docs/PERFORMANCE.md`.
