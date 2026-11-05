"""Token corpora for pretraining, continued training and fine-tuning at any corpus size.

`titan prepare-data a.txt b.md ... --out corpus.bin` streams the UTF-8 bytes of the inputs into
one token file (vocabulary = 256 byte values) plus a JSON sidecar with sizes and SHA-256 digests;
with `--tokenizer SPEC` (BPE, Hugging Face, SentencePiece, plugin) it writes uint16/uint32 token ids
(format titan-tokens-v1) and records the tokenizer and its fingerprint.
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
TOKENS_FORMAT = "titan-tokens-v1"
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


def prepare_tokens(inputs, out, tokenizer, chunk_chars: int = 1 << 20) -> dict:
    """Encode text files with any tokenizer into a memory-mappable token file (uint16 or uint32)."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.uint16 if tokenizer.vocab_size <= 65535 else np.uint32
    n, sources = 0, []
    with open(out, "wb") as fo:
        for p in (Path(x) for x in inputs):
            if not p.is_file():
                raise FileNotFoundError(f"{p} not found")
            h, count = hashlib.sha256(), 0
            with open(p, encoding="utf-8", errors="replace") as fi:
                while True:
                    block = fi.read(chunk_chars)
                    if not block:
                        break
                    h.update(block.encode("utf-8"))
                    ids = np.asarray(tokenizer.encode(block), dtype=dtype)
                    fo.write(ids.tobytes())
                    count += len(ids)
            sources.append({"path": str(p), "tokens": count, "sha256": h.hexdigest()})
            n += count
    if n < 64:
        raise DataError(f"corpus too small ({n} tokens)")
    meta = {"format": TOKENS_FORMAT, "vocab": tokenizer.vocab_size, "dtype": np.dtype(dtype).name, "tokens": n,
            "tokenizer": tokenizer.describe(), "tokenizer_fingerprint": tokenizer.fingerprint, "sources": sources,
            "sha256": _file_sha(out)}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return meta


def _file_sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                return h.hexdigest()
            h.update(b)


def load_tokens(path, seq: int, val_frac: float = 0.01, max_eval_windows: int = 256) -> Dataset:
    """Token file from prepare_tokens or prepare_bytes (memory-mapped; corpus size bounded by storage)."""
    p = Path(path)
    meta_p = p.with_suffix(".json")
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    if meta.get("format") != TOKENS_FORMAT:
        return load_bytes(path, seq, val_frac, max_eval_windows)
    tokens = np.memmap(p, dtype=np.dtype(meta["dtype"]), mode="r")
    n = len(tokens)
    if n < 16 * seq:
        raise DataError(f"{p}: {n} tokens is too small for sequence length {seq}")
    cut = n - max(2 * seq + 2, int(n * val_frac))
    ds = Dataset(p.stem, "lm", tokens[:cut], tokens[:cut], tokens[cut:], tokens[cut:], int(meta["vocab"]), 0,
                 [], seq, source=str(p), sha256=meta.get("sha256", ""))
    ds.eval_windows_max = max_eval_windows
    ds.tokenizer = meta.get("tokenizer")
    return ds


def text_with_tokenizer(path, seq: int, tokenizer) -> Dataset:
    """A text file encoded in memory with any tokenizer (char, byte, BPE, Hugging Face, plugin)."""
    p = Path(path)
    ids = np.asarray(tokenizer.encode(p.read_text(encoding="utf-8")), dtype=np.int64)
    if len(ids) < 8 * seq:
        raise DataError(f"{p}: {len(ids)} tokens is too small for sequence length {seq}")
    cut = int(len(ids) * 0.9)
    ds = Dataset(p.stem, "lm", ids[:cut], ids[:cut], ids[cut:], ids[cut:], tokenizer.vocab_size, 0, [], seq,
                 source=str(p), sha256=_file_sha(p))
    ds.tokenizer = tokenizer.describe()
    return ds


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
