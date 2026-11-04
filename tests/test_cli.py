"""M8: command-line interface and QuickStart."""
import json
from pathlib import Path

import pytest

from titan.cli import main

ROOT = Path(__file__).resolve().parent.parent


def test_info_and_backends(capsys):
    assert main(["info", "--arch", "target"]) == 0
    out = capsys.readouterr().out
    assert "estimates, not measurements" in out and "bandwidth ratio" in out
    assert main(["backends"]) == 0
    out = capsys.readouterr().out
    assert "[ ] FPGA" in out and "[ ] Physical Hardware" in out


@pytest.mark.parametrize("dtype", ["fp32", "int8", "bf16"])
def test_matmul_vertical_slice(capsys, dtype):
    assert main(["matmul", "--m", "40", "--n", "24", "--k", "56", "--dtype", dtype, "--arch", "tiny"]) == 0
    out = capsys.readouterr().out
    assert "VERIFICATION: PASS" in out and "== TITAN ISA" in out and "analytical" in out


def test_quickstart_end_to_end(tmp_path, capsys):
    rc = main(["quickstart", "--model", "mlp", "--profile", "tiny", "--steps", "30", "--eval-every", "30",
               "--ckpt-every", "30", "--arch", "tiny", "--out", str(tmp_path / "r")])
    assert rc == 0
    s = json.loads((tmp_path / "r" / "summary.json").read_text())
    assert s["steps"] == 30 and s["final_loss_avg10"] < s["initial_loss"]
    assert (tmp_path / "r" / "checkpoints" / "step_000030.npz").exists()
    rc = main(["quickstart", "--model", "mlp", "--profile", "tiny", "--steps", "35", "--arch", "tiny",
               "--out", str(tmp_path / "r2"), "--resume", str(tmp_path / "r" / "checkpoints" / "step_000030.npz")])
    assert rc == 0
    assert json.loads((tmp_path / "r2" / "summary.json").read_text())["steps"] == 35


def test_quickstart_refuses_unavailable_backend(tmp_path, capsys):
    assert main(["quickstart", "--backend", "hardware", "--out", str(tmp_path / "x")]) == 2
    assert "not operational" in capsys.readouterr().err


def test_quickstart_interactive(tmp_path, monkeypatch, capsys):
    answers = iter(["mlp", "xor", "tiny", "16", "fp32", "functional", "y"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    rc = main(["quickstart", "-i", "--steps", "5", "--arch", "tiny", "--out", str(tmp_path / "i")])
    assert rc == 0
    cfg = json.loads((tmp_path / "i" / "config.json").read_text())
    assert cfg["dataset"]["name"] == "xor" and cfg["train_config"]["batch"] == 16


def test_asm_example_and_disasm(tmp_path, capsys):
    binp = tmp_path / "p.bin"
    assert main(["asm", str(ROOT / "examples" / "gemm_relu.tasm"), "--arch", "tiny", "--run",
                 "--out", str(binp)]) == 0
    assert main(["disasm", str(binp)]) == 0
    assert "activation.relu @0" in capsys.readouterr().out


def test_bench_and_tune_commands(tmp_path, capsys):
    assert main(["bench", "--arch", "tiny", "--out", str(tmp_path)]) == 0
    assert list(tmp_path.glob("bench_quick_*.md"))
    assert main(["tune", "--m", "32", "--n", "32", "--k", "32", "--dtype", "fp32", "--arch", "tiny",
                 "--candidates", "4"]) == 0
    assert "speed-up vs baseline" in capsys.readouterr().out
