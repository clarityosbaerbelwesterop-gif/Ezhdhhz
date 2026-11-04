"""Yosys synthesis of TITAN RTL for resource estimates (M11).

What this is: logic synthesis + technology mapping (pre-place-and-route) cell counts.
What this is NOT: timing closure, Fmax, power or an FPGA measurement.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

TARGETS = {
    "generic": "synth -top {top} -flatten",
    "xc7": "synth_xilinx -top {top} -family xc7 -flatten",
    "ecp5": "synth_ecp5 -top {top}",
    "ice40": "synth_ice40 -top {top} -dsp",
}


def yosys_version() -> str | None:
    exe = shutil.which("yosys")
    if not exe:
        return None
    return subprocess.run([exe, "-V"], capture_output=True, text=True).stdout.strip()


def synthesize(top: str = "titan_mxu_tile", params: dict | None = None, target: str = "generic",
               out_dir: str | Path | None = None) -> dict:
    if not shutil.which("yosys"):
        raise RuntimeError("yosys is not installed")
    params = params or {}
    src = ROOT / "rtl" / f"{top}.sv"
    chp = "".join(f" -set {k} {v}" for k, v in params.items())
    script = f"read_verilog -sv {src}; " + (f"chparam{chp} {top}; " if chp else "") + \
        TARGETS[target].format(top=top) + "; stat"
    r = subprocess.run(["yosys", "-q", "-p", script.replace("; stat", "; tee -o /dev/stdout stat")],
                       capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(f"yosys failed: {r.stderr[-2000:]}")
    log = r.stdout
    cells = {}
    total = None
    for line in log.splitlines():
        m = re.match(r"\s+Number of cells:\s+(\d+)", line)
        if m:
            total = int(m.group(1))
        m = re.match(r"\s+(\$?[A-Za-z_][\w$]*)\s+(\d+)\s*$", line)
        if m and total is not None:
            cells[m.group(1)] = int(m.group(2))
    res = {"top": top, "params": params, "target": target, "total_cells": total, "cells": cells,
           "tool": yosys_version(), "script": script,
           "method": "Yosys logic synthesis + technology mapping (pre-place-and-route; no timing, no power)"}
    if target == "xc7":
        res["summary"] = {
            "LUT": sum(v for k, v in cells.items() if re.fullmatch(r"LUT\d", k)),
            "FF": sum(v for k, v in cells.items() if k.startswith("FD")),
            "DSP48E1": cells.get("DSP48E1", 0), "CARRY4": cells.get("CARRY4", 0)}
    elif target == "ecp5":
        res["summary"] = {"LUT4": cells.get("LUT4", 0), "FF": sum(v for k, v in cells.items() if k.startswith("TRELLIS_FF")),
                          "MULT18X18D": cells.get("MULT18X18D", 0), "CCU2C": cells.get("CCU2C", 0)}
    elif target == "ice40":
        res["summary"] = {"LUT4": cells.get("SB_LUT4", 0), "FF": sum(v for k, v in cells.items() if k.startswith("SB_DFF")),
                          "MAC16": cells.get("SB_MAC16", 0), "CARRY": cells.get("SB_CARRY", 0)}
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        tag = "_".join(f"{k}{v}" for k, v in params.items()) or "default"
        (out / f"{top}_{target}_{tag}.log").write_text(log)
    return res


def place_and_route_ecp5(top: str = "titan_mxu_tile", params: dict | None = None, device: str = "25k",
                         package: str = "CABGA381", freq_mhz: float = 100.0, seed: int = 1,
                         out_dir: str | Path | None = None) -> dict:
    """Out-of-context place & route with nextpnr-ecp5; returns utilization and post-route STA Fmax.

    The Fmax is a static-timing estimate of the routed netlist for the chosen speed grade,
    not a measurement on a board.
    """
    import json
    import tempfile
    if not (shutil.which("yosys") and shutil.which("nextpnr-ecp5")):
        raise RuntimeError("yosys and nextpnr-ecp5 are required")
    params = params or {}
    src = ROOT / "rtl" / f"{top}.sv"
    chp = "".join(f" -set {k} {v}" for k, v in params.items())
    with tempfile.TemporaryDirectory() as td:
        net, rep = Path(td) / "net.json", Path(td) / "report.json"
        script = f"read_verilog -sv {src}; " + (f"chparam{chp} {top}; " if chp else "") + \
            f"synth_ecp5 -top {top} -json {net}"
        r = subprocess.run(["yosys", "-q", "-p", script], capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            raise RuntimeError(f"yosys failed: {r.stderr[-1500:]}")
        cmd = ["nextpnr-ecp5", f"--{device}", "--package", package, "--json", str(net), "--out-of-context",
               "--freq", str(freq_mhz), "--seed", str(seed), "--report", str(rep)]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        log = p.stdout + p.stderr
        if p.returncode != 0 or not rep.exists():
            return {"top": top, "params": params, "device": f"LFE5U-{device.upper()}", "status": "failed",
                    "reason": log.strip().splitlines()[-1] if log.strip() else "unknown"}
        report = json.loads(rep.read_text())
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        tag = "_".join(f"{k}{v}" for k, v in params.items()) or "default"
        (out / f"{top}_ecp5-{device}_pnr_{tag}.log").write_text(log)
    util = {k: v for k, v in report["utilization"].items() if v["used"]}
    fmax = {k: round(v["achieved"], 2) for k, v in report["fmax"].items()}
    ver = subprocess.run(["nextpnr-ecp5", "--version"], capture_output=True, text=True)
    return {"top": top, "params": params, "device": f"LFE5U-{device.upper()} {package}", "status": "routed",
            "utilization": util, "fmax_mhz": fmax, "constraint_mhz": freq_mhz, "seed": seed,
            "tool": (ver.stdout + ver.stderr).strip(),
            "method": "nextpnr-ecp5 out-of-context place & route; Fmax = post-route static timing estimate "
                      "(not measured on hardware)"}
