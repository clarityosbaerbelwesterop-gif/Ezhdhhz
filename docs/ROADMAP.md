# Roadmap and acceptance gates

Status legend: **done** = implemented, tested, documented; **partial** = deliverable exists with
stated gaps; **open** = not started. Commit SHAs refer to branch `titan-x1/platform`.

| Milestone | Deliverable | Status | Evidence |
|---|---|---|---|
| M0 Audit | repository audit, plan | done | `docs/AUDIT_M0.md` (`35e326b`) |
| M1 Functional compute engine | matmul/vector/activation/normalization numerics, all precisions | done | `tests/test_semantics.py`, `tests/test_dtypes.py` (`ffa9534`) |
| M2 Memory system | address spaces, L2, bandwidth/latency/contention | done | `tests/test_engine.py` (`1a024b5`, `24c737f`) |
| M3 TITAN ISA | opcodes, encoding, validation, assembler, spec | done | `docs/ISA.md`, `tests/test_isa.py` (`ffa9534`) |
| M4 Compiler | graph → IR → ISA; matmul graph executes | done | `tests/test_compiler.py` (`24c737f`, `041f774`, `f65c032`) |
| M5 Hardware emulator | command processor, scheduling, sync, faults, modes A/B | done | `tests/test_engine.py` (`1a024b5`) |
| M6 Neural networks | MLP, attention, transformer block on TITAN | done | `tests/test_nn.py` (`53716a9`) |
| M7 Training runtime | full loop on TITAN, checkpoints, reproducibility | done | `tests/test_training.py`, `reports/TRAINING_AND_SCALING.md` (`f29a697`) |
| M8 QuickStart | one-command training, backend status | done | `titan quickstart`, `tests/test_cli.py` (`52d995d`) |
| M9 Performance & optimization | roofline, benchmarks, verified autotuning | done | `reports/bench_default_titan-x1-dev.md` (`21bb9fb`, `bb5f1ac`) |
| M10 RTL integration | synthesizable MXU tile, co-simulation | partial | INT8/INT4 MATMUL bit-exact (`7c7a783`); DMA, VPU, SFU, command processor not in RTL |
| M11 FPGA preparation | synthesis + P&R reports | partial | `docs/FPGA.md`, `reports/synth/` (`48f95c0`); no board, no measurement |
| M12 Multi-chip | topologies, collectives, data parallel, scaling | done (model) | `tests/test_multichip.py` (`d9ed8af`); links are assumptions |

### Extensions after M12

| Item | Status | Evidence |
|---|---|---|
| Own data (CSV/TSV/NPZ/NPY/TXT; classification, regression, LM) | done | `tests/test_data.py` (`8bf3f5c`) |
| Bit-exact replay executor, cached cycle timing | done | `tests/test_replay.py` (`c43f1e4`) |
| Fast IR executor, meta graphs, host emulation config | done | `tests/test_scale.py` (`58323ae`) |
| Distributed planner up to 175B, `titan plan`, `quickstart --params` | done (analytical) | `tests/test_scale.py` (`2eee40f`) |

## Next gates (in order)

1. **RTL coverage:** VPU and DMA RTL with the same co-simulation harness; command-processor RTL
   decoding the binary format; formal properties (clear > load > step, no X propagation).
2. **Cycle-model calibration:** compare Mode B against RTL simulation cycle counts per instruction
   class once the DMA/VPU RTL exists; publish the error distribution.
3. **FPGA prototype:** ECP5-85F class board, BRAM scratchpad, UART/AXI host link, on-board
   cycle counters; first hardware measurements with full provenance.
4. **Compiler:** attention-specific fusion (flash-style tiling), SHARED-SRAM weight broadcast,
   pipeline/tensor parallelism; autotuner integrated into compilation.
5. **Numerics:** FP8 formats (E4M3/E5M2) and structured sparsity — prerequisite for any
   like-for-like comparison with B300 FP8/FP4 figures.
6. **Physical design study:** area/power model per PE and SRAM macro for the target design point.
