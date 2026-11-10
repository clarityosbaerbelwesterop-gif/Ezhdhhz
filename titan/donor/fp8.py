"""FP8 payloads: exact decode of E4M3 (finite-only, as torch.float8_e4m3fn) and E5M2 to float32, and
block scales (one scale per [block_out x block_in] tile, DeepSeek/Mistral FP8 checkpoints)."""
from __future__ import annotations

import numpy as np


def _table_e4m3() -> np.ndarray:
    t = np.zeros(256, np.float64)
    for b in range(256):
        s, e, m = b >> 7, (b >> 3) & 0xF, b & 0x7
        if e == 0xF and m == 0x7:
            v = np.nan
        elif e == 0:
            v = (m / 8.0) * 2.0 ** -6                      # subnormal
        else:
            v = (1.0 + m / 8.0) * 2.0 ** (e - 7)           # max 1.75 * 2^8 = 448
        t[b] = -v if s else v
    return t.astype(np.float32)


def _table_e5m2() -> np.ndarray:
    t = np.zeros(256, np.float64)
    for b in range(256):
        s, e, m = b >> 7, (b >> 2) & 0x1F, b & 0x3
        if e == 0x1F:
            v = np.inf if m == 0 else np.nan
        elif e == 0:
            v = (m / 4.0) * 2.0 ** -14
        else:
            v = (1.0 + m / 4.0) * 2.0 ** (e - 15)
        t[b] = -v if s else v
    return t.astype(np.float32)


TABLES = {"F8_E4M3": _table_e4m3(), "F8_E5M2": _table_e5m2()}


def decode_fp8(raw: bytes | np.ndarray, dtype: str = "F8_E4M3") -> np.ndarray:
    u = np.frombuffer(raw, np.uint8) if isinstance(raw, (bytes, bytearray, memoryview)) else np.asarray(raw, np.uint8)
    return TABLES[dtype][u]


def encode_e4m3(x: np.ndarray) -> np.ndarray:
    """Nearest finite E4M3 code (ties to even code), saturating at +-448. Used for round-trip tests."""
    t = TABLES["F8_E4M3"]
    pos = np.array([i for i in range(127)], np.uint8)            # 0x00..0x7E: non-negative finite, ascending
    vals = t[pos].astype(np.float64)
    a = np.abs(np.asarray(x, np.float64)).clip(max=448.0)
    hi = np.searchsorted(vals, a, side="left").clip(0, len(vals) - 1)
    lo = (hi - 1).clip(0)
    dl, dh = np.abs(a - vals[lo]), np.abs(vals[hi] - a)
    pick = np.where((dl < dh) | ((dl == dh) & (lo % 2 == 0)), lo, hi)
    return (pos[pick] | (np.signbit(x).astype(np.uint8) << 7)).astype(np.uint8)


def apply_block_scale(w: np.ndarray, scale: np.ndarray, block: tuple[int, int] = (128, 128), inverse: bool = False):
    """W[i, j] * scale[i // block_out, j // block_in] (or division when ``inverse``); w is [out, in]."""
    bo, bi = block
    if scale.shape != (-(-w.shape[0] // bo), -(-w.shape[1] // bi)):
        raise ValueError(f"scale {scale.shape} does not tile weight {w.shape} with blocks {block}")
    s = np.repeat(np.repeat(scale.astype(np.float32), bo, 0)[:w.shape[0]], bi, 1)[:, :w.shape[1]]
    return w / s if inverse else w * s
