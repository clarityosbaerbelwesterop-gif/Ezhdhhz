# Verification and evidence policy

Every performance or correctness claim must be backed by reproducible evidence. Never label an
estimate as a measurement.

## Verification matrix

| Subject | Method | Command |
|---|---|---|
| Numeric formats (BF16 RNE, INT4 packing, INT32 wrap, saturating convert) | unit tests | `pytest tests/test_dtypes.py` |
| Every data instruction | NumPy reference, all dtypes/flags/broadcast modes | `pytest tests/test_semantics.py` |
| ISA encoding / validation / assembly | round trip of every opcode, 18 negative cases, doc sync | `pytest tests/test_isa.py` |
| Engine: sync, barriers, kernels, faults, deadlock | directed programs, both modes | `pytest tests/test_engine.py` |
| Memory timing: double buffering, contention, L2, queues | cycle-model property tests | `pytest tests/test_engine.py` |
| Compiler end to end | 60 graphs vs float64 reference (shapes, precisions, lowering variants, fusion, barriers, memory reuse, sharing) | `pytest tests/test_compiler.py` |
| Autodiff | every rule vs central finite differences | `pytest tests/test_nn.py` |
| NN forward | vs reference and an independent textbook attention | `pytest tests/test_nn.py` |
| Training | loss reduction, determinism, resume ≡ uninterrupted, checksum | `pytest tests/test_training.py` |
| Replay executor | complete memory images bit-identical to the reference engine (4 programs incl. training step, BF16, INT8); cached cycle timing == full simulation | `pytest tests/test_replay.py` |
| Fast IR executor | outputs and updated parameters vs ISA level (FP32 1e-4, BF16 2e-3); training trajectories agree | `pytest tests/test_scale.py` |
| Own data | CSV/TSV/NPZ/NPY/TXT, header and delimiter detection, categorical features, regression, 5 error classes, end-to-end training | `pytest tests/test_data.py` |
| Accel executor (PyTorch) | outputs and parameters vs fast executor on 4 programs; training trajectory vs ISA level (rtol 3e-3) | `pytest tests/test_environment.py` |
| Pretrain / fine-tune / quality | byte corpus memory-mapped; init-from loads all matching weights; frozen weights unchanged; transfer lowers the first loss; mismatched shapes reported; clipped SGD step = lr·clip; cosine schedule; best checkpoint | `pytest tests/test_environment.py` |
| Large-model accounting | parameter count and forward/training matmul FLOPs of the planner == compiled graphs (exact); 100B-width meta graph | `pytest tests/test_scale.py` |
| Performance tooling | roofline classes, comparison refusals, verified tuning | `pytest tests/test_perf.py` |
| Multi-chip | ring all-reduce exactness + traffic, DP ≡ large batch, replicas identical | `pytest tests/test_multichip.py` |
| RTL D0 MAC array | directed + 2000-cycle randomized self-checking TB (Verilator and Icarus) | `make test` |
| RTL MXU tile | directed TB (INT8 4×4, INT4 8×8) on Verilator and Icarus | `make test` |
| RTL ↔ emulator | random MATMUL incl. accumulate and wrap-around, bit-exact | `pytest -m rtl`, `titan rtl-verify` |
| Lint | Verilator `-Wall`, warnings are errors | `make lint` |
| Synthesis / P&R | Yosys + nextpnr-ecp5 reports | `titan synth --pnr` |

CI (`.github/workflows/rtl.yml`) runs RTL lint, simulation, Icarus cross-check, co-simulation,
synthesis + P&R, and the software suite on Python 3.11 and 3.13, plus a vertical-slice run,
a QuickStart run and the quick benchmark (artifacts uploaded).

## Tolerances

Integer results: exact. FP32 graphs: rtol/atol 1e-4 (reduction order differs from float64
reference). BF16/FP16: 1e-3 … 2e-3 relative, chosen from the 8/11-bit significands.

## Known tool issue

Verilator 5.020: part-select writes to a vector inside a loop of a timed testbench process were
not propagated to the DUT's continuous assigns before the next edge (Icarus behaves correctly).
Testbenches build stimulus in a temporary vector and assign it whole (see `tb/*.sv`).

## Evidence to retain

Tool versions and commands, full logs, commit SHA, configuration hash, synthesis/P&R reports,
benchmark JSON. FPGA measurements (none yet) must be kept separate from estimates.
