"""Execution backends and their honest availability status."""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

MXU_RTL = Path(__file__).resolve().parent.parent.parent / "rtl" / "titan_mxu_tile.sv"


@dataclass(frozen=True)
class Backend:
    key: str
    label: str
    kind: str          # what the numbers produced by this backend mean
    description: str

    def status(self) -> tuple[bool, str]:
        if self.key in ("functional", "cycle"):
            return True, "available (software emulation on the host CPU)"
        if self.key == "rtl":
            if not MXU_RTL.exists():
                return False, "unavailable: rtl/titan_mxu_tile.sv not present"
            if shutil.which("verilator"):
                return True, "available: MATMUL INT8 offloaded to the Verilator model of rtl/titan_mxu_tile.sv; " \
                             "all other instructions execute in the functional model"
            return False, "unavailable: Verilator not installed"
        if self.key == "fpga":
            return False, "unavailable: no FPGA bitstream or board attached (see docs/FPGA.md)"
        return False, "unavailable: no TITAN silicon exists"


BACKENDS = {
    "functional": Backend("functional", "Functional Emulator", "functional (numerical results, no timing)",
                          "Mode A: architectural semantics of TITAN ISA v1.0"),
    "cycle": Backend("cycle", "Cycle Simulator", "simulated cycles (transaction-level, cycle-approximate)",
                     "Mode B: functional results plus the TITAN-X1 cycle model"),
    "rtl": Backend("rtl", "RTL Simulation", "RTL co-simulation (bit-exact INT8 MXU)",
                   "Mode D: Verilator co-simulation of the MXU tile"),
    "fpga": Backend("fpga", "FPGA", "FPGA measurement", "Not available"),
    "hardware": Backend("hardware", "Physical Hardware", "silicon measurement", "Not available"),
}


def backend_status() -> list[tuple[Backend, bool, str]]:
    return [(b, *b.status()) for b in BACKENDS.values()]


def available_backends() -> list[str]:
    return [b.key for b, ok, _ in backend_status() if ok]
