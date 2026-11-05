"""Post-training data: supervised fine-tuning (SFT) on prompt/response pairs with loss masking.

JSON Lines, one example per line:
    {"prompt": "Q: 12+35=", "response": "47\\n"}          loss on the response only
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
                                                         rendered "User: ...\\nAssistant: ...\\n";
                                                         loss on assistant turns only
    {"text": "..."}                                      plain text, loss on every token

Examples are packed back to back into one token stream (no padding); training windows start at
example boundaries, so every window sees its prompt. Prompt tokens are inputs but not targets
(target id -1); the loss is the mean over response tokens.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .datasets import DataError, Dataset


def render(ex: dict) -> list[tuple[str, bool]]:
    """Example -> [(text, trained)] segments."""
    if "messages" in ex:
        segs = []
        for m in ex["messages"]:
            role = m.get("role", "user")
            segs.append((f"{role.capitalize()}: ", False))
            segs.append((m.get("content", "") + "\n", role == "assistant"))
        return segs
    if "prompt" in ex:
        resp = str(ex.get("response", ex.get("completion", "")))
        return [(str(ex["prompt"]), False), (resp if resp.endswith("\n") else resp + "\n", True)]
    if "text" in ex:
        t = str(ex["text"])
        return [(t if t.endswith("\n") else t + "\n", True)]
    raise DataError(f"SFT example needs prompt/response, messages or text: {sorted(ex)}")


def read_jsonl(path) -> list[dict]:
    out = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise DataError(f"{path}:{i + 1}: {e}") from None
    if not out:
        raise DataError(f"{path}: no examples")
    return out


def load_sft(path, seq: int, tokenizer_spec: str = "char", val_frac: float = 0.1, max_eval: int = 256) -> Dataset:
    from .tokenizers import CharTokenizer, load_tokenizer
    p = Path(path)
    examples = [render(ex) for ex in read_jsonl(p)]
    if tokenizer_spec == "char":
        tok = CharTokenizer.fit("".join(t for ex in examples for t, _ in ex))
    else:
        tok = load_tokenizer(tokenizer_spec)
    ids, mask, starts = [], [], []
    n = 0
    for ex in examples:
        starts.append(n)
        for text, trained in ex:
            e = tok.encode(text)
            ids.extend(e)
            mask.extend([trained] * len(e))
            n += len(e)
    ids, mask, starts = np.asarray(ids, dtype=np.int64), np.asarray(mask, dtype=bool), np.asarray(starts)
    if not mask.any():
        raise DataError(f"{p}: no response tokens to train on")
    n_val = max(1, int(len(examples) * val_frac)) if len(examples) > 1 else 0
    cut = int(starts[len(examples) - n_val]) if n_val else len(ids)
    if cut < seq + 2:
        raise DataError(f"{p}: {cut} training tokens is too small for sequence length {seq}")
    ds = Dataset(p.stem, "lm", ids[:cut], ids[:cut], ids[cut:], ids[cut:], tok.vocab_size, 0,
                 list(getattr(tok, "chars", [])), seq, source=str(p),
                 sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    ds.mask_train, ds.mask_test = mask[:cut], mask[cut:]
    ds.starts_train = starts[starts < cut]
    ds.starts_test = starts[starts >= cut] - cut
    ds.eval_windows_max = max_eval
    ds.tokenizer = tok.describe()
    ds.tokenizer_obj = tok
    ds.sft = {"examples": len(examples), "train_tokens": int(cut), "trained_tokens": int(mask[:cut].sum())}
    return ds


def masked_targets(ds: Dataset, split: str, starts: np.ndarray, seq: int):
    """Windows [B, seq] of inputs and targets (-1 where the target is a prompt token)."""
    tokens = ds.x_train if split == "train" else ds.x_test
    mask = ds.mask_train if split == "train" else ds.mask_test
    starts = np.minimum(starts, len(tokens) - seq - 1)
    idx = starts[:, None] + np.arange(seq)[None, :]
    y = np.where(mask[idx + 1], tokens[idx + 1], -1)
    return tokens[idx], y
