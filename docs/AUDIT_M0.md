# M0 — Repository audit (TITAN-X1)

Audit date: 2026-11-04 · Audited commit: `c0bf6425fdc68c439ad6c9f571a5d49fe8db5b03` (main)

## 1. Inventory found

| Path | Content | Assessment |
|---|---|---|
| `rtl/rtl_mac_array.sv` | 16-lane signed INT8 MAC, 32-bit accumulators, async reset, sync clear | Correct, synthesizable, parameterized (`LANES`, `DATA_W`, `ACC_W`). Kept. |
| `tb/tb_rtl_mac_array.sv` | Directed self-checking testbench (5 checks) | Passes. One lint warning (BLKSEQ on clock generator). No randomized coverage. |
| `scripts/perf_model.py` | Peak ops/s estimator with B300 ratio | **Defect:** compares an INT8 MAC estimate against a B300 *FP8* peak (5.0 PF default) — precision mismatch, invalid ratio. |
| `Makefile` | Verilator lint + sim | Works. `-Wno-fatal` lets warnings pass silently. |
| `.github/workflows/rtl.yml` | Verilator lint/sim + perf script | Green on both runs (run 1 `354b766`, run 2 `c0bf642`). No Python tests. |
| `docs/*.md` | Architecture D0, roadmap, verification policy, status | Honest; describes only the D0 tile. |

## 2. Local reproduction (this session)

| Command | Tool | Result |
|---|---|---|
| `make lint` | Verilator 5.020 | clean |
| `make test` | Verilator 5.020 | `PASS: all 16 MAC lanes, accumulation, clear and signed arithmetic` |
| `verilator --lint-only -Wall` on TB | Verilator 5.020 | `%Warning-BLKSEQ tb/tb_rtl_mac_array.sv:16` |

## 3. Gaps against the TITAN-X1 mission

1. No architecture model beyond one MAC row (no memory hierarchy, no interconnect, no command processor).
2. No instruction set, no encoding, no decoder, no emulator.
3. No compiler, IR or runtime; no workload can be expressed or executed.
4. Accumulator overflow behaviour is implicit (two's-complement wrap) and undocumented.
5. Performance tooling is a single peak-formula script with an invalid cross-precision ratio.
6. CI covers RTL only.

## 4. Plan derived from the audit

The existing RTL tile is kept and becomes the reference for the MXU numerics
(INT8 × INT8 → INT32, two's-complement wrap). Everything else is built as a
Python package `titan/` (NumPy only) plus additional RTL:

| Milestone | Deliverable in this repository |
|---|---|
| M1 | `titan/dtypes.py`, `titan/emu/semantics.py` — numerics for INT4/INT8/INT32/FP16/BF16/FP32 |
| M2 | `titan/hw/memory.py`, `titan/emu/cycle.py` — address spaces, L2 cache, bandwidth/latency/contention |
| M3 | `titan/isa/` — TITAN ISA v1.0: opcodes, binary encoding, validation, assembler, spec `docs/ISA.md` |
| M4 | `titan/compiler/` — graph, IR, fusion, memory planning, tiling, ISA generation |
| M5 | `titan/emu/engine.py` — command processor, per-core queues, sync, functional + cycle modes |
| M6 | `titan/nn/` — MLP, attention, transformer block on TITAN, checked against NumPy |
| M7 | `titan/compiler/autodiff.py`, `titan/train/` — training on TITAN ISA, checkpoints |
| M8 | `titan/cli.py` — `titan quickstart` |
| M9 | `titan/perf/` — roofline, benchmark harness with provenance, `titan/tune/` |
| M10 | `rtl/titan_mxu_tile.sv` + Verilator co-simulation against the emulator |
| M11 | Yosys synthesis resource report (pre-P&R; no FPGA hardware available) |
| M12 | `titan/multichip/` — topologies, collectives, data-parallel execution model |

Execution modes are kept separate in every report: functional (A), cycle model
(B), analytical (C), RTL co-simulation (D). FPGA and physical hardware backends
are reported as unavailable.
