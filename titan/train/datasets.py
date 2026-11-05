"""Small, deterministic datasets for training on the virtual accelerator.

Data loading, shuffling and one-hot encoding are HOST operations by design.
"""
from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CORPUS = (
    "titan computes with tiles. the tiles move from memory to the cores. the cores multiply the tiles and "
    "add the results. a barrier waits for every core. a signal tells one core that data is ready. the compiler "
    "plans the memory and the tiles. the emulator checks every result. the cycle model counts the time. "
    "the roofline shows if a kernel waits for memory or for compute. small models learn first. then larger "
    "models follow. every claim needs a test. every number needs a source. "
) * 4


@dataclass
class Dataset:
    name: str
    kind: str                       # classification | regression | lm
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    classes: int
    in_dim: int = 0
    vocab: list = field(default_factory=list)
    seq: int = 0
    labels: list = field(default_factory=list)      # class names (classification from files)
    y_mean: float = 0.0                              # regression targets are standardized; stats kept for
    y_std: float = 1.0                               # reporting errors in original units
    source: str = ""
    sha256: str = ""

    @property
    def outputs(self) -> int:
        return 1 if self.kind == "regression" else self.classes

    def describe(self) -> dict:
        d = {"name": self.name, "kind": self.kind, "train": int(len(self.x_train)), "test": int(len(self.x_test)),
             "classes": self.classes, "in_dim": self.in_dim, "seq": self.seq}
        if self.source:
            d.update({"source": self.source, "sha256": self.sha256})
        if self.labels:
            d["labels"] = [str(x) for x in self.labels[:50]]
        return d


def _split(x, y, rng, test_frac=0.25):
    idx = rng.permutation(len(x))
    n_test = int(len(x) * test_frac)
    te, tr = idx[:n_test], idx[n_test:]
    mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-8
    return (x[tr] - mu) / sd, y[tr], (x[te] - mu) / sd, y[te]


def spirals(seed=0, n=1200, classes=3, noise=0.15):
    rng = np.random.default_rng(seed)
    per = n // classes
    xs, ys = [], []
    for c in range(classes):
        r = np.linspace(0.05, 1.0, per)
        t = np.linspace(c * 2 * np.pi / classes, c * 2 * np.pi / classes + 3.2, per) + rng.standard_normal(per) * noise
        xs.append(np.stack([r * np.sin(t), r * np.cos(t)], 1))
        ys.append(np.full(per, c))
    return Dataset("spirals", "classification", *_split(np.vstack(xs), np.concatenate(ys), rng), classes, 2)


def blobs(seed=0, n=1200, classes=4, dim=8):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((classes, dim)) * 3
    y = rng.integers(0, classes, n)
    x = centers[y] + rng.standard_normal((n, dim))
    return Dataset("blobs", "classification", *_split(x, y, rng), classes, dim)


def xor(seed=0, n=1000):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (n, 2))
    y = ((x[:, 0] > 0) ^ (x[:, 1] > 0)).astype(int)
    return Dataset("xor", "classification", *_split(x, y, rng), 2, 2)


def digits(seed=0):
    try:
        from sklearn.datasets import load_digits
    except ImportError as e:
        raise RuntimeError("dataset 'digits' requires scikit-learn (pip install scikit-learn)") from e
    d = load_digits()
    rng = np.random.default_rng(seed)
    return Dataset("digits", "classification", *_split(d.data.astype(float), d.target, rng), 10, 64)


def text(seed=0, seq=16):
    vocab = sorted(set(CORPUS))
    ids = np.array([vocab.index(c) for c in CORPUS])
    split = int(len(ids) * 0.9)
    return Dataset("text", "lm", ids[:split], ids[:split], ids[split:], ids[split:], len(vocab), 0, vocab, seq)


DATASETS = {"spirals": spirals, "blobs": blobs, "xor": xor, "digits": digits, "text": text}


class DataError(ValueError):
    pass


def _num(tok: str):
    try:
        return float(tok)
    except ValueError:
        return None


def _read_table(p: Path):
    raw = p.read_text(encoding="utf-8-sig")
    try:
        dialect = csv.Sniffer().sniff(raw[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    rows = [r for r in csv.reader(raw.splitlines(), dialect) if any(c.strip() for c in r)]
    if len(rows) < 2:
        raise DataError(f"{p}: needs at least two rows")
    header = None
    if any(_num(c) is None for c in rows[0]) and any(_num(c) is not None for c in rows[1]):
        header, rows = [c.strip() for c in rows[0]], rows[1:]
    width = len(rows[0])
    for i, r in enumerate(rows):
        if len(r) != width:
            raise DataError(f"{p}: row {i + (2 if header else 1)} has {len(r)} columns, expected {width}")
    return header or [f"col{i}" for i in range(width)], [[c.strip() for c in r] for r in rows]


def _encode_features(names, cols, path):
    """Numeric columns as-is; categorical columns one-hot encoded. Empty cells are rejected."""
    feats, fnames = [], []
    for name, col in zip(names, cols):
        empty = [i for i, v in enumerate(col) if v == ""]
        if empty:
            raise DataError(f"{path}: column {name!r} has empty cells (first at data row {empty[0] + 1})")
        vals = [_num(v) for v in col]
        if all(v is not None for v in vals):
            arr = np.array(vals, dtype=np.float64)
            if not np.isfinite(arr).all():
                raise DataError(f"{path}: column {name!r} contains non-finite values")
            feats.append(arr[:, None])
            fnames.append(name)
        else:
            cats = sorted(set(col))
            if len(cats) > 256:
                raise DataError(f"{path}: categorical column {name!r} has {len(cats)} distinct values (max 256)")
            feats.append(np.array([[1.0 if v == c else 0.0 for c in cats] for v in col]))
            fnames += [f"{name}={c}" for c in cats]
    return np.hstack(feats), fnames


def _target(y_raw, task: str, path):
    nums = [_num(v) if isinstance(v, str) else float(v) for v in y_raw]
    numeric = all(v is not None for v in nums)
    if task == "auto":
        if not numeric:
            task = "classification"
        else:
            arr = np.array(nums)
            task = "classification" if np.all(arr == np.round(arr)) and len(np.unique(arr)) <= 50 else "regression"
    if task == "regression":
        if not numeric:
            raise DataError(f"{path}: regression target must be numeric")
        return task, np.array(nums, dtype=np.float64), []
    keys = [str(int(v)) if numeric and float(v).is_integer() else str(r) for v, r in zip(nums, y_raw)]
    labels = sorted(set(keys), key=lambda k: (_num(k) is None, _num(k) if _num(k) is not None else 0, k))
    if len(labels) < 2:
        raise DataError(f"{path}: classification needs at least two classes")
    index = {k: i for i, k in enumerate(labels)}
    return task, np.array([index[k] for k in keys]), labels


def load_file(path, target=None, task: str = "auto", seed: int = 0, seq: int = 16,
              test_frac: float = 0.2) -> Dataset:
    """Load your own data.

    * ``.csv`` / ``.tsv``: one example per row; ``target`` = column name or index (default: last column);
      numeric feature columns are standardized, text columns one-hot encoded.
    * ``.npz``: arrays ``x`` (or ``X``/``features``/``data``) and ``y`` (or ``Y``/``target``/``labels``).
    * ``.npy``: 2-D array, target in the last column (or ``target`` index).
    * ``.txt``: plain text for character-level language modelling (transformer).
    ``task``: auto | classification | regression.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"data file {p} not found")
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    rng = np.random.default_rng(seed)
    suf = p.suffix.lower()
    if suf in (".txt", ".text", ".md"):
        txt = p.read_text(encoding="utf-8")
        vocab = sorted(set(txt))
        if len(txt) < 8 * seq:
            raise DataError(f"{p}: text too short ({len(txt)} chars) for sequence length {seq}")
        if len(vocab) > 512:
            raise DataError(f"{p}: {len(vocab)} distinct characters (max 512)")
        index = {c: i for i, c in enumerate(vocab)}
        ids = np.array([index[c] for c in txt])
        cut = int(len(ids) * 0.9)
        return Dataset(p.stem, "lm", ids[:cut], ids[:cut], ids[cut:], ids[cut:], len(vocab), 0, vocab, seq,
                       source=str(p), sha256=digest)
    if suf in (".csv", ".tsv"):
        names, rows = _read_table(p)
        cols = list(zip(*rows))
        if target is None:
            ti = len(names) - 1
        elif str(target) in names:
            ti = names.index(str(target))
        else:
            try:
                ti = int(target)
            except ValueError:
                raise DataError(f"{p}: target column {target!r} not found; columns: {names}") from None
        if not -len(names) <= ti < len(names):
            raise DataError(f"{p}: target index {ti} out of range")
        ti %= len(names)
        x, _ = _encode_features([n for i, n in enumerate(names) if i != ti],
                                [c for i, c in enumerate(cols) if i != ti], p)
        y_raw = list(cols[ti])
    elif suf in (".npz", ".npy"):
        if suf == ".npz":
            z = np.load(p, allow_pickle=False)
            xk = next((k for k in ("x", "X", "features", "data") if k in z.files), None)
            yk = next((k for k in ("y", "Y", "target", "labels") if k in z.files), None)
            if xk is None or yk is None:
                raise DataError(f"{p}: needs arrays x and y (found {z.files})")
            x, y_raw = np.asarray(z[xk], dtype=np.float64), np.asarray(z[yk]).ravel().tolist()
        else:
            a = np.load(p, allow_pickle=False)
            if a.ndim != 2 or a.shape[1] < 2:
                raise DataError(f"{p}: expected a 2-D array with at least two columns")
            ti = -1 if target is None else int(target)
            y_raw = a[:, ti].tolist()
            x = np.delete(a, ti % a.shape[1], axis=1).astype(np.float64)
        if x.ndim == 1:
            x = x[:, None]
        x = x.reshape(len(x), -1)
        if not np.isfinite(x).all():
            raise DataError(f"{p}: features contain NaN or inf")
    else:
        raise DataError(f"unsupported data format {suf!r} (use .csv, .tsv, .npz, .npy or .txt)")
    if len(x) != len(y_raw):
        raise DataError(f"{p}: {len(x)} feature rows but {len(y_raw)} targets")
    if len(x) < 10:
        raise DataError(f"{p}: at least 10 examples are required")
    task, y, labels = _target(y_raw, task, p)
    xtr, ytr, xte, yte = _split(x, y, rng, test_frac)
    if task == "regression":
        mu, sd = float(ytr.mean()), float(ytr.std() + 1e-12)
        return Dataset(p.stem, "regression", xtr, (ytr - mu) / sd, xte, (yte - mu) / sd, 1, x.shape[1],
                       y_mean=mu, y_std=sd, source=str(p), sha256=digest)
    return Dataset(p.stem, "classification", xtr, ytr, xte, yte, len(labels), x.shape[1], labels=labels,
                   source=str(p), sha256=digest)


def load_dataset(name: str, seed: int = 0, **kw) -> Dataset:
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    return DATASETS[name](seed=seed, **kw)


def onehot(idx: np.ndarray, n: int) -> np.ndarray:
    return np.eye(n)[idx]


class Sampler:
    """Deterministic mini-batch sampler (host side). Its state is saved in checkpoints."""

    def __init__(self, ds: Dataset, batch: int, seed: int):
        self.ds, self.batch = ds, batch
        self.rng = np.random.default_rng(seed)
        self._perm, self._pos = None, 0

    def next(self) -> dict:
        ds, b = self.ds, self.batch
        if ds.kind == "lm":                       # token ids; one-hot targets are built on the device
            starts = self.rng.integers(0, len(ds.x_train) - ds.seq - 1, b)
            idx = starts[:, None] + np.arange(ds.seq)[None, :]
            return {"x": np.asarray(ds.x_train[idx]).reshape(-1, 1).astype(np.int32),
                    "y": np.asarray(ds.x_train[idx + 1]).reshape(-1, 1).astype(np.int32)}
        if b > len(ds.x_train):
            raise DataError(f"batch size {b} exceeds the {len(ds.x_train)} training examples")
        if self._perm is None or self._pos + b > len(self._perm):
            self._perm, self._pos = self.rng.permutation(len(ds.x_train)), 0
        sel = self._perm[self._pos:self._pos + b]
        self._pos += b
        if ds.kind == "regression":
            return {"x": ds.x_train[sel], "y": ds.y_train[sel].reshape(-1, 1)}
        return {"x": ds.x_train[sel], "y": onehot(ds.y_train[sel], ds.classes)}

    def state(self) -> dict:
        return {"rng": self.rng.bit_generator.state, "perm": None if self._perm is None else self._perm.tolist(),
                "pos": self._pos}

    def restore(self, st: dict) -> None:
        self.rng.bit_generator.state = st["rng"]
        self._perm = None if st["perm"] is None else np.array(st["perm"])
        self._pos = st["pos"]


def eval_batches(ds: Dataset, batch: int):
    """Fixed evaluation batches ``(x, y, n_valid)``; the last batch is padded and only n_valid rows count."""
    if ds.kind == "lm":
        n = (len(ds.x_test) - 1) // ds.seq
        n = min(n, getattr(ds, "eval_windows_max", None) or n)
        windows = [np.arange(i * ds.seq, (i + 1) * ds.seq) for i in range(n)]
        for i in range(0, len(windows), batch):
            chunk = windows[i:i + batch]
            nv = len(chunk) * ds.seq
            idx = np.stack(chunk + [chunk[-1]] * (batch - len(chunk)))
            yield (np.asarray(ds.x_test[idx]).reshape(-1, 1).astype(np.int32),
                   np.asarray(ds.x_test[idx + 1]).ravel(), nv)
        return
    n = len(ds.x_test)
    for i in range(0, n, batch):
        sel = np.arange(i, min(i + batch, n))
        pad = np.concatenate([sel, np.full(batch - len(sel), sel[-1])])
        yield ds.x_test[pad], ds.y_test[pad], len(sel)
