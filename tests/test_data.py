"""Own-data loader: formats, task detection, validation errors, end-to-end training on TITAN."""
import json
from pathlib import Path

import numpy as np
import pytest

from titan.cli import main
from titan.train import TrainConfig, Trainer
from titan.train.datasets import DataError, Sampler, eval_batches, load_file

ROOT = Path(__file__).resolve().parent.parent


def _csv(path, header, rows, sep=","):
    path.write_text("\n".join([sep.join(header)] + [sep.join(map(str, r)) for r in rows]) + "\n")
    return path


def test_csv_classification_with_categorical_feature_and_string_labels():
    ds = load_file(ROOT / "examples" / "data" / "machines.csv", target="status")
    assert ds.kind == "classification" and ds.labels == ["fault", "ok", "warning"]
    assert ds.in_dim == 3 + 3                           # 3 numeric + one-hot line A/B/C
    assert len(ds.x_train) + len(ds.x_test) == 600
    assert abs(ds.x_train[:, 0].mean()) < 1e-9          # standardized with training statistics
    assert len(ds.sha256) == 64


def test_target_by_index_semicolon_and_regression(tmp_path, rng):
    x = rng.standard_normal((80, 3))
    y = 2 * x[:, 0] - x[:, 2] + 0.5
    p = _csv(tmp_path / "r.csv", ["y", "a", "b", "c"], [[f"{t:.5f}", *map("{:.5f}".format, r)] for t, r in zip(y, x)],
             ";")
    ds = load_file(p, target=0)
    assert ds.kind == "regression" and ds.in_dim == 3 and ds.outputs == 1
    assert abs(ds.y_train.mean()) < 1e-9 and ds.y_std > 0


def test_npz_npy_and_text(tmp_path, rng):
    x, y = rng.standard_normal((50, 4)), rng.integers(0, 3, 50)
    np.savez(tmp_path / "d.npz", x=x, y=y)
    ds = load_file(tmp_path / "d.npz")
    assert ds.kind == "classification" and ds.classes == 3 and ds.in_dim == 4
    np.save(tmp_path / "d.npy", np.hstack([x, y[:, None]]))
    assert load_file(tmp_path / "d.npy").classes == 3
    (tmp_path / "t.txt").write_text("hello titan, hello tiles. " * 20)
    t = load_file(tmp_path / "t.txt", seq=8)
    assert t.kind == "lm" and t.classes == len(set("hello titan, hello tiles. "))


@pytest.mark.parametrize("content,msg", [
    ("a,b,y\n1,2,0\n3,,1\n" + "1,2,0\n" * 10, "empty cells"),
    ("a,b,y\n1,2,0\n3,4\n" + "1,2,0\n" * 10, "columns"),
    ("a,b,y\n1,2,0\n3,4,0\n", "two rows|at least"),
])
def test_csv_validation_errors(tmp_path, content, msg):
    p = tmp_path / "bad.csv"
    p.write_text(content)
    with pytest.raises(DataError, match=msg):
        load_file(p)


def test_other_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_file(tmp_path / "missing.csv")
    (tmp_path / "x.parquet").write_text("x")
    with pytest.raises(DataError, match="unsupported"):
        load_file(tmp_path / "x.parquet")
    p = _csv(tmp_path / "c.csv", ["a", "y"], [[i, i % 2] for i in range(20)])
    with pytest.raises(DataError, match="not found"):
        load_file(p, target="label")


def test_eval_batches_pad_and_count(tmp_path, rng):
    np.savez(tmp_path / "d.npz", x=rng.standard_normal((50, 2)), y=rng.integers(0, 2, 50))
    ds = load_file(tmp_path / "d.npz")
    got = list(eval_batches(ds, 4))
    assert sum(nv for _, _, nv in got) == len(ds.x_test)
    assert all(xb.shape == (4, 2) for xb, _, _ in got)
    with pytest.raises(DataError, match="batch size"):
        Sampler(ds, 1000, 0).next()


def test_train_on_own_csv_classification(tmp_path):
    tr = Trainer(TrainConfig(data=str(ROOT / "examples" / "data" / "machines.csv"), target="status", profile="small",
                             steps=200, eval_every=1000, ckpt_every=0, out_dir=str(tmp_path / "m")), log=None)
    s = tr.run()
    assert s["eval_accuracy"] > 0.8 and s["final_loss_avg10"] < s["initial_loss"]
    cfg = json.loads((tmp_path / "m" / "config.json").read_text())
    assert cfg["dataset"]["source"].endswith("machines.csv") and len(cfg["dataset"]["sha256"]) == 64


def test_train_regression_on_own_data(tmp_path, rng):
    x = rng.uniform(-2, 2, (400, 2))
    y = np.sin(x[:, 0]) + 0.5 * x[:, 1] ** 2
    np.savez(tmp_path / "reg.npz", x=x, y=y)
    tr = Trainer(TrainConfig(data=str(tmp_path / "reg.npz"), profile="small", steps=300, eval_every=1000,
                             ckpt_every=0, out_dir=str(tmp_path / "r")), log=None)
    s = tr.run()
    assert tr.ds.kind == "regression"
    assert s["eval_r2"] > 0.9 and s["eval_rmse"] < 0.3


def test_quickstart_with_own_text_and_csv(tmp_path, capsys):
    (tmp_path / "book.txt").write_text("the accelerator moves tiles and computes products. " * 40)
    assert main(["quickstart", "--data", str(tmp_path / "book.txt"), "--profile", "tiny", "--steps", "20",
                 "--arch", "tiny", "--out", str(tmp_path / "t")]) == 0
    assert json.loads((tmp_path / "t" / "config.json").read_text())["model"]["kind"] == "lm"
    assert main(["quickstart", "--data", str(ROOT / "examples" / "data" / "machines.csv"), "--target", "status",
                 "--profile", "tiny", "--steps", "20", "--arch", "tiny", "--out", str(tmp_path / "c")]) == 0
    assert main(["quickstart", "--data", str(tmp_path / "book.txt"), "--model", "mlp", "--out",
                 str(tmp_path / "x")]) == 2
