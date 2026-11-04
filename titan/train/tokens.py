"""Byte-level token corpora for pretraining, continued training and fine-tuning at any corpus size.

`titan prepare-data a.txt b.md ... --out corpus.bin` streams the UTF-8 bytes of the inputs into
one token file (vocabulary = 256 byte values) plus a JSON sidecar with sizes and SHA-256 digests.
Training reads it through a memory map: only the sampled windows are paged in, so the corpus
size is bounded by the disk, not by RAM. A fixed byte vocabulary also means every model trained
on bytes can be fine-tuned on any other byte corpus without changing its embedding or head.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .datasets import Dataset, DataError

FORMAT = "titan-bytes-v1"
SEPARATOR = b"\n\n"
CHUNK = 1 << 24


def prepare_bytes(inputs, out, separator: bytes = SEPARATOR) -> dict:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    total = hashlib.sha256()
    sources, n = [], 0
    with open(out, "wb") as fo:
        for i, p in enumerate(Path(x) for x in inputs):
            if not p.is_file():
                raise FileNotFoundError(f"{p} not found")
            h, size = hashlib.sha256(), 0
            with open(p, "rb") as fi:
                while True:
                    block = fi.read(CHUNK)
                    if not block:
                        break
                    fo.write(block)
                    total.update(block)
                    h.update(block)
                    size += len(block)
            if i < len(inputs) - 1 and separator:
                fo.write(separator)
                total.update(separator)
                size += len(separator)
            sources.append({"path": str(p), "bytes": size, "sha256": h.hexdigest()})
            n += size
    if n < 1024:
        raise DataError(f"corpus too small ({n} bytes)")
    meta = {"format": FORMAT, "vocab": 256, "tokens": n, "sha256": total.hexdigest(), "sources": sources}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return meta


def load_bytes(path, seq: int, val_frac: float = 0.01, max_eval_windows: int = 256) -> Dataset:
    """Memory-mapped byte corpus (.bin from prepare_bytes, or any raw file) as a language-model dataset."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"{p} not found")
    meta_p = p.with_suffix(".json")
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    if meta and meta.get("format") != FORMAT:
        raise DataError(f"{meta_p}: unknown token format {meta.get('format')!r}")
    tokens = np.memmap(p, dtype=np.uint8, mode="r")
    n = len(tokens)
    if n < 16 * seq:
        raise DataError(f"{p}: {n} tokens is too small for sequence length {seq}")
    cut = n - max(2 * seq + 2, int(n * val_frac))
    ds = Dataset(p.stem, "lm", tokens[:cut], tokens[:cut], tokens[cut:], tokens[cut:], 256, 0,
                 [bytes([i]) for i in range(256)], seq, source=str(p),
                 sha256=meta.get("sha256", "unhashed (raw file; run titan prepare-data for a digest)"))
    ds.eval_windows_max = max_eval_windows
    return ds


def text_as_bytes(path, seq: int) -> Dataset:
    """A .txt file tokenized as bytes (vocabulary 256) instead of characters."""
    return load_bytes(path, seq, val_frac=0.1)
