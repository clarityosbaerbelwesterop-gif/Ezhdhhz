"""Tokenizer / vocabulary contracts between a seed tokenizer and donor tokenizers, at the byte level.

A contract answers, for every seed token: which donor ids spell exactly the same bytes?
    exact     one donor token has the same bytes            -> rows can be copied
    composed  several donor tokens concatenate to the bytes -> rows are combined (mean), an approximation
    unknown   the donor cannot spell the bytes              -> no information
    special   seed control tokens (bos/eos/pad)             -> mapped by role, never by bytes
and, for running text, which token boundaries of both tokenizations fall on the same byte offset
(the only positions where hidden states of two models describe the same prefix).
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np


def _bpe_apply(symbols: list, ranks: dict) -> list:
    """Greedy BPE: repeatedly merge the adjacent pair with the lowest rank."""
    s = list(symbols)
    while len(s) > 1:
        best, bi = None, -1
        for i in range(len(s) - 1):
            r = ranks.get((s[i], s[i + 1]))
            if r is not None and (best is None or r < best):
                best, bi = r, i
        if best is None:
            break
        a, b = s[bi], s[bi + 1]
        out, i = [], 0
        while i < len(s):
            if i < len(s) - 1 and s[i] == a and s[i + 1] == b:
                out.append(a + b)
                i += 2
            else:
                out.append(s[i])
                i += 1
        s = out
    return s


class SeedBPE:
    """Reader for the seed's byte-level BPE file (version 2: pattern, specials, merges [[a, b, id]]).
    ids: 0..255 bytes, then merges in id order, then specials. Encoding = regex pre-tokenisation +
    lowest-merge-id-first BPE inside each piece (same result as the seed's own bpe.py; tested)."""

    def __init__(self, path):
        raw = Path(path).read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        d = json.loads(raw)
        self.pattern = d.get("pattern")
        self._pat = re.compile(self.pattern) if self.pattern else None
        self.specials = list(d.get("specials", ()))
        self.vocab: list[bytes] = [bytes([i]) for i in range(256)]
        self.rank: dict[tuple[bytes, bytes], int] = {}
        for a, b, idx in sorted(d["merges"], key=lambda m: m[2]):
            if idx != len(self.vocab):
                raise ValueError("merge ids are not contiguous")
            self.rank[(self.vocab[a], self.vocab[b])] = idx
            self.vocab.append(self.vocab[a] + self.vocab[b])
        self.n_regular = len(self.vocab)
        self.special_ids = {s: self.n_regular + i for i, s in enumerate(self.specials)}
        self.vocab_size = self.n_regular + len(self.specials)
        self.index = {b: i for i, b in enumerate(self.vocab)}
        self._cache: dict[str, list[int]] = {}

    def token_bytes(self, i: int) -> bytes:
        return self.vocab[i] if i < self.n_regular else b""

    def _piece(self, piece: str) -> list[int]:
        hit = self._cache.get(piece)
        if hit is None:
            syms = _bpe_apply([bytes([b]) for b in piece.encode("utf-8")], self.rank)
            hit = [self.index[s] for s in syms]
            if len(self._cache) < 500_000:
                self._cache[piece] = hit
        return hit

    def encode(self, text: str) -> list[int]:
        pieces = self._pat.findall(text) if self._pat else ([text] if text else [])
        out: list[int] = []
        for p in pieces:
            out.extend(self._piece(p))
        return out

    def decode_bytes(self, ids) -> bytes:
        return b"".join(self.token_bytes(int(i)) for i in ids)

    def end_offsets(self, ids) -> np.ndarray:
        return np.cumsum([len(self.token_bytes(int(i))) for i in ids])

    def describe(self) -> dict:
        return {"kind": "seed-bpe-v2", "sha256": self.sha256, "vocab_size": self.vocab_size, "regular": self.n_regular,
                "specials": self.special_ids, "pattern": self.pattern}


def _gpt2_byte_decoder() -> dict:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(range(ord("\xae"), ord("\xff") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


class HFByteLevelBPE:
    """Hugging Face tokenizer.json with a byte-level BPE model (GPT-2 / Qwen family)."""

    def __init__(self, path):
        from tokenizers import Tokenizer
        raw = Path(path).read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        d = json.loads(raw)
        if d["model"]["type"] != "BPE":
            raise ValueError(f"model type {d['model']['type']} is not BPE")
        self.tok = Tokenizer.from_file(str(path))
        dec = _gpt2_byte_decoder()
        self.added = {t["id"]: t["content"] for t in d.get("added_tokens", [])}
        vocab = d["model"]["vocab"]
        self.n_model = len(vocab)
        self.vocab_size = max(max(vocab.values()), max(self.added, default=0)) + 1
        self.bytes_of: dict[int, bytes] = {}
        self._sym_id: dict[str, int] = vocab
        for tok, i in vocab.items():
            if i in self.added:
                continue
            try:
                self.bytes_of[i] = bytes(dec[c] for c in tok)
            except KeyError:                       # not a byte-level string: only reachable as an added token
                pass
        self.index: dict[bytes, int] = {}
        for i in sorted(self.bytes_of):            # lowest id wins for duplicate byte strings
            self.index.setdefault(self.bytes_of[i], i)
        merges = d["model"]["merges"]
        self.rank = {}
        for r, m in enumerate(merges):
            a, b = m if isinstance(m, list) else m.split(" ")
            self.rank[(a, b)] = r
        self._enc = {b: c for c, b in dec.items()}

    def token_bytes(self, i: int) -> bytes:
        return self.bytes_of.get(i, b"")

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def encode_with_offsets(self, text: str) -> tuple[list[int], np.ndarray]:
        """ids and the byte offset at which every token ends."""
        e = self.tok.encode(text, add_special_tokens=False)
        pre = np.zeros(len(text) + 1, np.int64)
        np.cumsum([len(c.encode("utf-8")) for c in text], out=pre[1:])
        return e.ids, pre[[b for _, b in e.offsets]]

    def encode_bytes(self, b: bytes) -> list[int]:
        """Donor ids spelling ``b``; valid UTF-8 goes through the real tokenizer (pre-tokenisation included),
        other byte strings through plain byte-level BPE."""
        try:
            ids = self.encode(b.decode("utf-8"))
            if b"".join(self.token_bytes(i) for i in ids) == b:
                return ids
        except UnicodeDecodeError:
            pass
        syms = _bpe_apply([self._enc[x] for x in b], self.rank)
        return [self._sym_id[s] for s in syms] if all(s in self._sym_id for s in syms) else []

    def describe(self) -> dict:
        return {"kind": "hf-bytelevel-bpe", "sha256": self.sha256, "vocab_size": self.vocab_size,
                "model_tokens": self.n_model, "added_tokens": len(self.added)}


class Tekken:
    """Mistral tekken.json: tiktoken-style byte BPE (vocab entries carry base64 token bytes and their rank
    = merge priority); control tokens occupy the first ids."""

    def __init__(self, path, hf_tokenizer=None):
        """``hf_tokenizer``: the repository's tokenizer.json; when given, valid UTF-8 strings are encoded by it
        (it applies the pre-tokenisation pattern), and only other byte strings by plain byte-pair merging."""
        raw = Path(path).read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self._hf = None
        if hf_tokenizer:
            from tokenizers import Tokenizer
            self._hf = Tokenizer.from_file(str(hf_tokenizer))
        d = json.loads(raw)
        cfg = d.get("config", {})
        self.n_special = int(cfg.get("default_num_special_tokens", len(d.get("special_tokens", []))))
        self.specials = {int(s["rank"]): s["token_str"] for s in d.get("special_tokens", [])}
        n_vocab = int(cfg.get("default_vocab_size", len(d["vocab"]) + self.n_special)) - self.n_special
        self.pattern = cfg.get("pattern")
        self.bytes_of: dict[int, bytes] = {}
        self.rank: dict[bytes, int] = {}
        for v in d["vocab"][:n_vocab]:
            b = base64.b64decode(v["token_bytes"])
            self.rank[b] = int(v["rank"])
            self.bytes_of[int(v["rank"]) + self.n_special] = b
        self.vocab_size = n_vocab + self.n_special
        self.index = {b: r + self.n_special for b, r in self.rank.items()}

    def token_bytes(self, i: int) -> bytes:
        return self.bytes_of.get(i, b"")

    def encode_bytes(self, b: bytes) -> list[int]:
        """tiktoken byte-pair merge: repeatedly merge the adjacent pair whose concatenation has the lowest rank."""
        if self._hf is not None:
            try:
                ids = self._hf.encode(b.decode("utf-8"), add_special_tokens=False).ids
                if b"".join(self.token_bytes(i) for i in ids) == b:
                    return ids
            except UnicodeDecodeError:
                pass
        parts = [bytes([x]) for x in b]
        while len(parts) > 1:
            best, bi = None, -1
            for i in range(len(parts) - 1):
                r = self.rank.get(parts[i] + parts[i + 1])
                if r is not None and (best is None or r < best):
                    best, bi = r, i
            if best is None:
                break
            parts[bi:bi + 2] = [parts[bi] + parts[bi + 1]]
        return [self.index[p] for p in parts] if all(p in self.index for p in parts) else []

    def describe(self) -> dict:
        return {"kind": "mistral-tekken", "sha256": self.sha256, "vocab_size": self.vocab_size,
                "special_tokens": self.n_special, "pattern": self.pattern}


def vocab_contract(seed: SeedBPE, donor, special_map: dict | None = None) -> dict:
    """Seed token -> donor ids, with the kind of every row and summary statistics."""
    special_map = special_map or {}
    rows, kinds = [], []
    for i in range(seed.vocab_size):
        if i >= seed.n_regular:
            name = seed.specials[i - seed.n_regular]
            rows.append([int(special_map[name])] if name in special_map else [])
            kinds.append("special")
            continue
        b = seed.token_bytes(i)
        if b in donor.index:
            rows.append([donor.index[b]])
            kinds.append("exact")
            continue
        ids = donor.encode_bytes(b)
        ok = bool(ids) and b"".join(donor.token_bytes(j) for j in ids) == b
        rows.append([int(j) for j in ids] if ok else [])
        kinds.append("composed" if ok else "unknown")
    c = Counter(kinds)
    return {"map": rows, "kinds": kinds,
            "stats": {"seed_vocab": seed.vocab_size, "exact": c["exact"], "composed": c["composed"],
                      "unknown": c["unknown"], "special": c["special"],
                      "pieces_hist": dict(sorted(Counter(len(r) for r, k in zip(rows, kinds) if k != "special").items())),
                      "unique_donor_ids": len({j for r in rows for j in r}), "donor_vocab": donor.vocab_size},
            "seed": seed.describe(), "donor": donor.describe()}


def transplant_rows(table: np.ndarray, rows: list[list[int]]) -> tuple[np.ndarray, list[int]]:
    """Donor table [V_donor, d] -> seed rows [V_seed, d]: copy (one id), mean (several), table mean (none).
    Returns the rows and the indices that had no donor id."""
    mean = table.mean(0, dtype=np.float64).astype(np.float32)
    out = np.empty((len(rows), table.shape[1]), np.float32)
    unknown = []
    for i, r in enumerate(rows):
        if not r:
            out[i] = mean
            unknown.append(i)
        elif len(r) == 1:
            out[i] = table[r[0]]
        else:
            out[i] = table[np.asarray(r, np.int64)].mean(0, dtype=np.float64)
    return out, unknown


def align_boundaries(seed_ends: np.ndarray, donor_ends: np.ndarray) -> np.ndarray:
    """Pairs [n, 2] of (seed position, donor position) whose tokens end at the same byte offset."""
    pos = {int(b): i for i, b in enumerate(donor_ends)}
    return np.array([(j, pos[int(b)]) for j, b in enumerate(seed_ends) if int(b) in pos], np.int64).reshape(-1, 2)
