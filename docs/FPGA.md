# M11 — FPGA preparation

**Status:** synthesis and place-and-route reports exist for the matrix-engine
tile. **No FPGA board, no bitstream on hardware and no FPGA measurement exist.**
The QuickStart backend "FPGA" is therefore reported as unavailable.

Reproduce: `python -m titan synth --pnr --out reports/synth` (Yosys 0.33,
nextpnr-ecp5 0.6). Raw tool logs and `synth_summary.json` are in `reports/synth/`.

## 1. Technology-mapped resources (Yosys, pre-place-and-route)

| Module | Parameters | Xilinx 7-series (`synth_xilinx`) | Lattice ECP5 (`synth_ecp5`) |
|---|---|---|---|
| `rtl_mac_array` (D0) | 16 lanes, INT8 | 512 LUT, 512 FF, 16 DSP48E1 | 1025 LUT4, 512 FF, 16 MULT18X18D |
| `titan_mxu_tile` | 4×4, INT8 | 512 LUT, 512 FF, 16 DSP48E1 | 1025 LUT4, 512 FF, 16 MULT18X18D, 256 CCU2C |
| `titan_mxu_tile` | 8×8, INT8 | 2048 LUT, 2048 FF, 64 DSP48E1 | 4097 LUT4, 2048 FF, 64 MULT18X18D, 1024 CCU2C |
| `titan_mxu_tile` | 8×8, INT4 | 5790 LUT, 2048 FF, 0 DSP48E1, 640 CARRY4 | 4097 LUT4, 2048 FF, 64 MULT18X18D |

Observations
* Resources scale linearly with the number of processing elements (one
  hardware multiplier + one 32-bit accumulator register per PE).
* On 7-series, Yosys implements the 4-bit multipliers in LUT fabric instead of
  DSP48E1 slices (5790 LUT vs 2048). An INT4 mode on Xilinx should therefore pack
  two INT4 products per DSP (or use LUT multipliers deliberately) — design item for
  the FPGA prototype.
* Accumulators are 32-bit registers; the `load_acc` path costs one LUT per
  accumulator bit.

## 2. Place & route (nextpnr-ecp5, out-of-context, speed grade 6, seed 1)

| Tile | Device | MULT18X18D | Logic (COMB) | FF | Post-route Fmax (STA estimate) |
|---|---|---|---|---|---|
| 4×4 INT8 | LFE5U-25F CABGA381 | 16 / 28 (57 %) | 1603 / 24288 | 512 | 182.6 MHz |
| 8×8 INT8 | LFE5U-85F CABGA381 | 64 / 156 (41 %) | 6403 / 83640 | 2048 | 179.8 MHz |
| 8×8 INT4 | LFE5U-85F CABGA381 | 64 / 156 (41 %) | 6403 / 83640 | 2048 | 172.4 MHz |

These Fmax values are static-timing estimates of an out-of-context routed
block (no I/O, no clock tree to pins). A full design with SRAM, DMA and control
will close at a lower frequency; only on-board measurement establishes it.

## 3. Derived capacity estimate (analytical, unvalidated)

On an LFE5U-85F (156 multipliers), at most two 8×8 tiles fit with DSP mapping:
128 INT8 MAC × 2 ops × ~150 MHz (assumed after integration losses) ≈ 38 GOP/s.
This is a prototyping vehicle for functional validation of the TITAN ISA, not a
performance platform; it is roughly 5 orders of magnitude below a B300.

## 4. What an FPGA prototype still needs

1. BRAM-backed LOCAL scratchpad and a DMA engine (RTL) implementing LOAD/STORE.
2. Command processor RTL: instruction fetch/decode of the binary format in
   `docs/ISA.md`, per-unit queues, SIGNAL/WAIT/BARRIER.
3. Host interface (UART for bring-up; later PCIe/AXI) and the runtime driver.
4. Board-level constraints, clocking, full place & route and timing closure.
5. Measurement protocol: on-board cycle counters for every instruction class,
   compared against the cycle model (Mode B) on the same programs; results
   recorded with board, bitstream hash, tool versions and commit SHA.

## 5. Board selection (deferred)

No board is selected. Candidate families with fully open toolchains (Yosys +
nextpnr) are Lattice ECP5 (e.g. 85F boards); Xilinx 7-series requires Vivado
or the experimental open flow. Current availability, price and toolchain support
must be verified at purchase time.
