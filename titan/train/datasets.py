"""Small, deterministic datasets for training on the virtual accelerator.

Data loading, shuffling and one-hot encoding are HOST operations by design.
"""
from __future__ import annotations

from dataclasses import dataclass, field

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
    kind: str                       # classification | lm
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    classes: int
    in_dim: int = 0
    vocab: list = field(default_factory=list)
    seq: int = 0

    def describe(self) -> dict:
        return {"name": self.name, "kind": self.kind, "train": int(len(self.x_train)), "test": int(len(self.x_test)),
                "classes": self.classes, "in_dim": self.in_dim, "seq": self.seq}


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
        if ds.kind == "lm":
            starts = self.rng.integers(0, len(ds.x_train) - ds.seq - 1, b)
            idx = starts[:, None] + np.arange(ds.seq)[None, :]
            return {"x": onehot(ds.x_train[idx].ravel(), ds.classes), "y": onehot(ds.x_train[idx + 1].ravel(),
                                                                                  ds.classes)}
        if self._perm is None or self._pos + b > len(self._perm):
            self._perm, self._pos = self.rng.permutation(len(ds.x_train)), 0
        sel = self._perm[self._pos:self._pos + b]
        self._pos += b
        return {"x": ds.x_train[sel], "y": onehot(ds.y_train[sel], ds.classes)}

    def state(self) -> dict:
        return {"rng": self.rng.bit_generator.state, "perm": None if self._perm is None else self._perm.tolist(),
                "pos": self._pos}

    def restore(self, st: dict) -> None:
        self.rng.bit_generator.state = st["rng"]
        self._perm = None if st["perm"] is None else np.array(st["perm"])
        self._pos = st["pos"]


def eval_batches(ds: Dataset, batch: int):
    """Fixed evaluation batches (full batches only; the remainder is dropped and reported)."""
    if ds.kind == "lm":
        n = (len(ds.x_test) - 1) // ds.seq
        windows = [np.arange(i * ds.seq, (i + 1) * ds.seq) for i in range(n)]
        for i in range(0, len(windows) - batch + 1, batch):
            idx = np.stack(windows[i:i + batch])
            yield onehot(ds.x_test[idx].ravel(), ds.classes), ds.x_test[idx + 1].ravel()
        return
    for i in range(0, len(ds.x_test) - batch + 1, batch):
        yield ds.x_test[i:i + batch], ds.y_test[i:i + batch]
