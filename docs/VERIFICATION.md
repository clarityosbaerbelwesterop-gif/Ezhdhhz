# Verification and evidence policy

Every performance or correctness claim must be backed by reproducible evidence.

## D0 directed checks
- Reset clears every lane.
- Lane i computes (i+1)×2.
- Repeated enable accumulates correctly.
- Synchronous clear wins over enable.
- Signed multiplication handles -3×7 = -21.

## Commands

    make lint
    make test
    python3 scripts/perf_model.py

## Evidence to retain
- Tool versions and exact commands.
- Full simulator/linter logs.
- Git commit SHA.
- Synthesis reports (cell count, area, timing) when available.
- FPGA measurements kept separate from analytical estimates.
- Workload, precision and sparsity clearly stated for all accelerator comparisons.

Never label an estimate as a measurement.
