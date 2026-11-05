"""Tokenizers: bring your own, or train one.

    char                       characters of the training text (small vocabulary)
    byte                       UTF-8 bytes, vocabulary 256 (transferable between corpora)
    path/to/titan-bpe.json     byte-level BPE trained with `titan tokenizer train`
    path/to/tokenizer.json     Hugging Face tokenizers file (needs `pip install tokenizers`)
    path/to/model.model        SentencePiece model (needs `pip install sentencepiece`)
    file.py:Factory | pkg.mod:Factory
                               your own class/factory returning an object with
                               encode(str) -> list[int], decode(list[int]) -> str, vocab_size

Every tokenizer reports ``spec`` (how to load it again) and ``fingerprint`` (SHA-256), which are
stored with token files and runs so that a dataset is always decoded with the tokenizer it was
encoded with.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

PRETOKEN = re.compile(r"\s+(?=\S)|\s+$|[A-Za-z0-9_]+|[^\sA-Za-z0-9_]+")


class Tokenizer:
    kind = "base"
    spec = ""

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, ids) -> str:
        raise NotImplementedError

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.describe(), sort_keys=True).encode()).hexdigest()

    def describe(self) -> dict:
        return {"kind": self.kind, "vocab_size": self.vocab_size, "spec": self.spec}


class ByteTokenizer(Tokenizer):
    kind, spec = "byte", "byte"

    @property
    def vocab_size(self) -> int:
        return 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids) -> str:
        return bytes(int(i) for i in ids).decode("utf-8", errors="replace")


class CharTokenizer(Tokenizer):
    kind = "char"

    def __init__(self, chars: list[str]):
        self.chars = list(chars)
        self.index = {c: i for i, c in enumerate(self.chars)}
        self.spec = "char"

    @classmethod
    def fit(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        try:
            return [self.index[c] for c in text]
        except KeyError as e:
            raise ValueError(f"character {e.args[0]!r} is not in the character vocabulary") from None

    def decode(self, ids) -> str:
        return "".join(self.chars[int(i)] for i in ids)

    def describe(self) -> dict:
        return {**super().describe(), "chars": "".join(self.chars)}


class BPETokenizer(Tokenizer):
    """Byte-level BPE: ids 0..255 are bytes, ids >= 256 are learned merges (in merge order)."""
    kind = "bpe"

    def __init__(self, merges: list[tuple[int, int]], spec: str = ""):
        self.merges = [tuple(m) for m in merges]
        self.ranks = {m: i for i, m in enumerate(self.merges)}
        self.pieces: list[bytes] = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self.pieces.append(self.pieces[a] + self.pieces[b])
        self.spec = spec
        self._cache: dict[str, list[int]] = {}

    @classmethod
    def train(cls, texts, vocab_size: int, min_frequency: int = 2) -> "BPETokenizer":
        if vocab_size <= 256:
            raise ValueError("vocab_size must exceed 256 (the byte alphabet)")
        words = Counter()
        for t in texts:
            words.update(PRETOKEN.findall(t))
        seqs = {w: list(w.encode("utf-8")) for w in words}
        merges: list[tuple[int, int]] = []
        pair_counts: Counter = Counter()
        where: dict[tuple, set] = {}
        for w, s in seqs.items():
            for p in zip(s, s[1:]):
                pair_counts[p] += words[w]
                where.setdefault(p, set()).add(w)
        new_id = 256
        while new_id < vocab_size and pair_counts:
            pair, cnt = max(pair_counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
            if cnt < min_frequency:
                break
            merges.append(pair)
            for w in list(where.get(pair, ())):
                s, f = seqs[w], words[w]
                for p in zip(s, s[1:]):
                    pair_counts[p] -= f
                    if pair_counts[p] <= 0:
                        del pair_counts[p]
                out, i = [], 0
                while i < len(s):
                    if i + 1 < len(s) and (s[i], s[i + 1]) == pair:
                        out.append(new_id)
                        i += 2
                    else:
                        out.append(s[i])
                        i += 1
                seqs[w] = out
                for p in zip(out, out[1:]):
                    pair_counts[p] += f
                    where.setdefault(p, set()).add(w)
            where.pop(pair, None)
            new_id += 1
        return cls(merges)

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    def _word(self, w: str) -> list[int]:
        hit = self._cache.get(w)
        if hit is not None:
            return hit
        s = list(w.encode("utf-8"))
        while len(s) > 1:
            best = min(((self.ranks.get(p, 1 << 30), i) for i, p in enumerate(zip(s, s[1:]))), default=None)
            if best is None or best[0] == 1 << 30:
                break
            r, i = best
            s = s[:i] + [256 + r] + s[i + 2:]
        if len(self._cache) < 200_000:
            self._cache[w] = s
        return s

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for w in PRETOKEN.findall(text):
            out.extend(self._word(w))
        return out

    def decode(self, ids) -> str:
        return b"".join(self.pieces[int(i)] for i in ids).decode("utf-8", errors="replace")

    def save(self, path) -> None:
        Path(path).write_text(json.dumps({"format": "titan-bpe-v1", "merges": self.merges}))

    def describe(self) -> dict:
        return {**super().describe(), "merges": len(self.merges)}


class HFTokenizer(Tokenizer):
    kind = "huggingface"

    def __init__(self, path):
        from tokenizers import Tokenizer as _T     # optional dependency
        self.tok = _T.from_file(str(path))
        self.spec = str(path)

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text).ids

    def decode(self, ids) -> str:
        return self.tok.decode([int(i) for i in ids])


class SentencePieceTokenizer(Tokenizer):
    kind = "sentencepiece"

    def __init__(self, path):
        import sentencepiece as spm                # optional dependency
        self.sp = spm.SentencePieceProcessor(model_file=str(path))
        self.spec = str(path)

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    def encode(self, text: str) -> list[int]:
        return list(self.sp.encode(text))

    def decode(self, ids) -> str:
        return self.sp.decode([int(i) for i in ids])


class PluginTokenizer(Tokenizer):
    kind = "plugin"

    def __init__(self, obj, spec: str):
        for need in ("encode", "decode", "vocab_size"):
            if not hasattr(obj, need):
                raise TypeError(f"tokenizer plugin {spec} lacks '{need}'")
        self.obj, self.spec = obj, spec

    @property
    def vocab_size(self) -> int:
        v = self.obj.vocab_size
        return int(v() if callable(v) else v)

    def encode(self, text: str) -> list[int]:
        return list(self.obj.encode(text))

    def decode(self, ids) -> str:
        return self.obj.decode(list(ids))


def load_tokenizer(spec: str, fit_text: str | None = None) -> Tokenizer:
    """Resolve a tokenizer spec (see module docstring). ``char`` needs ``fit_text``."""
    if spec == "byte":
        return ByteTokenizer()
    if spec == "char":
        if fit_text is None:
            raise ValueError("the char tokenizer is fitted on the training text")
        return CharTokenizer.fit(fit_text)
    if ":" in spec and not Path(spec).exists():
        import importlib
        import importlib.util
        mod_name, _, attr = spec.rpartition(":")
        if mod_name.endswith(".py"):
            sp = importlib.util.spec_from_file_location(Path(mod_name).stem, mod_name)
            mod = importlib.util.module_from_spec(sp)
            sp.loader.exec_module(mod)
        else:
            mod = importlib.import_module(mod_name)
        obj = getattr(mod, attr)
        inst = obj() if isinstance(obj, type) or (callable(obj) and not hasattr(obj, "encode")) else obj
        return PluginTokenizer(inst, spec)
    p = Path(spec)
    if not p.is_file():
        raise FileNotFoundError(f"tokenizer {spec!r} not found")
    if p.suffix == ".model":
        return SentencePieceTokenizer(p)
    data = json.loads(p.read_text())
    if data.get("format") == "titan-bpe-v1":
        return BPETokenizer([tuple(m) for m in data["merges"]], spec=str(p))
    if "model" in data and "added_tokens" in data:
        return HFTokenizer(p)
    raise ValueError(f"{p}: unknown tokenizer file format")
