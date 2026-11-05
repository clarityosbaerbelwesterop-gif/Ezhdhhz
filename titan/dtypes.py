"""TITAN data types and bit-exact numeric conversion rules.

Storage conventions (what lives in emulated memory):
    INT4  two's-complement nibbles, packed two per byte (element 2i in the low nibble)
    INT8  int8            INT32  int32 (accumulator type)
    FP16  IEEE binary16   BF16   bfloat16 bit pattern (uint16)   FP32  IEEE binary32

Logical values (what instruction semantics see after a read):
    INT4/INT8 -> int8, INT32 -> int32, FP16 -> float16, BF16/FP32 -> float32.
"""
from __future__ import annotations

from enum import IntEnum

import numpy as np


class DType(IntEnum):
    INT4 = 0
    INT8 = 1
    INT32 = 2
    FP16 = 3
    BF16 = 4
    FP32 = 5

    @property
    def bits(self) -> int:
        return _BITS[self]

    @property
    def is_int(self) -> bool:
        return self in (DType.INT4, DType.INT8, DType.INT32)

    @property
    def is_float(self) -> bool:
        return not self.is_int

    @property
    def short(self) -> str:
        return self.name.lower()

    @classmethod
    def parse(cls, s: "str | DType") -> "DType":
        if isinstance(s, DType):
            return s
        return cls[str(s).upper()]


_BITS = {DType.INT4: 4, DType.INT8: 8, DType.INT32: 32, DType.FP16: 16, DType.BF16: 16, DType.FP32: 32}
STORAGE = {DType.INT4: np.uint8, DType.INT8: np.int8, DType.INT32: np.int32,
           DType.FP16: np.float16, DType.BF16: np.uint16, DType.FP32: np.float32}
INT_RANGE = {DType.INT4: (-8, 7), DType.INT8: (-128, 127), DType.INT32: (-2**31, 2**31 - 1)}


def elem_bytes(dt: DType) -> float:
    return dt.bits / 8


def nbytes(dt: DType, n_elems: int) -> int:
    return (n_elems * dt.bits + 7) // 8


def accum_dtype(dt: DType) -> DType:
    """Accumulator type of the matrix engine for a given input type."""
    return DType.INT32 if dt.is_int else DType.FP32


# ---------------------------------------------------------------- bf16 / wrap

def f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even FP32 -> BF16 (bit pattern). NaN -> canonical quiet NaN."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    u = x.view(np.uint32).astype(np.uint64)
    bias = ((u >> 16) & 1) + 0x7FFF
    out = ((u + bias) >> 16).astype(np.uint16)
    return np.where(np.isnan(x), np.uint16(0x7FC0), out)


def bf16_bits_to_f32(b: np.ndarray) -> np.ndarray:
    return (np.asarray(b, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def round_bf16(x: np.ndarray) -> np.ndarray:
    return bf16_bits_to_f32(f32_to_bf16_bits(x))


def wrap_int32(x: np.ndarray) -> np.ndarray:
    """Two's-complement wrap of an integer array to 32 bits (matches RTL accumulators)."""
    x = np.asarray(x, dtype=np.int64)
    return (((x + 2**31) % 2**32) - 2**31).astype(np.int32)


# ------------------------------------------------------------- conversions

def to_compute(x: np.ndarray, dt: DType) -> np.ndarray:
    """Logical values -> compute representation (int64 for ints, float32 for floats)."""
    return np.asarray(x).astype(np.int64 if dt.is_int else np.float32)


def round_to(dt: DType, x: np.ndarray) -> np.ndarray:
    """Round float32 compute values to the value set of float type `dt` (still float32)."""
    x = np.asarray(x, dtype=np.float32)
    if dt == DType.FP32:
        return x
    if dt == DType.FP16:
        with np.errstate(over="ignore"):
            return x.astype(np.float16).astype(np.float32)
    if dt == DType.BF16:
        return round_bf16(x)
    raise ValueError(f"{dt} is not a float type")


def saturate_int(dt: DType, x: np.ndarray) -> np.ndarray:
    """Float/int compute values -> integer dtype using round-half-even and saturation. NaN -> 0."""
    lo, hi = INT_RANGE[dt]
    x = np.asarray(x, dtype=np.float64)
    x = np.where(np.isnan(x), 0.0, x)
    return np.clip(np.rint(x), lo, hi).astype(np.int64)


def convert(x: np.ndarray, src: DType, dst: DType, scale: float = 1.0) -> np.ndarray:
    """CONVERT instruction semantics: y = round_dst(x * scale); integers saturate.

    Returns logical values of `dst`.
    """
    xc = to_compute(x, src)
    if dst.is_int:
        y = saturate_int(dst, xc.astype(np.float64) * np.float64(np.float32(scale)))
        return y.astype(np.int32 if dst == DType.INT32 else np.int8)
    y = (xc.astype(np.float32) * np.float32(scale)).astype(np.float32)
    y = round_to(dst, y)
    return y.astype(np.float16) if dst == DType.FP16 else y


def from_compute(x: np.ndarray, dt: DType) -> np.ndarray:
    """Compute values -> logical values of dt. Ints wrap (INT32) / must be in range; floats round."""
    if dt == DType.INT32:
        return wrap_int32(x)
    if dt in (DType.INT8, DType.INT4):
        lo, hi = INT_RANGE[dt]
        x = np.asarray(x, dtype=np.int64)
        if x.size and (x.min() < lo or x.max() > hi):
            raise OverflowError(f"value out of {dt.name} range")
        return x.astype(np.int8)
    y = round_to(dt, x)
    return y.astype(np.float16) if dt == DType.FP16 else y


def to_storage(x: np.ndarray, dt: DType) -> np.ndarray:
    """Logical values -> storage array (INT4 not packed here; packing is done by memory)."""
    x = np.asarray(x)
    if dt == DType.BF16:
        return f32_to_bf16_bits(x.astype(np.float32))
    if dt == DType.INT4:
        return x.astype(np.int8)
    return x.astype(STORAGE[dt])


def from_storage(s: np.ndarray, dt: DType) -> np.ndarray:
    if dt == DType.BF16:
        return bf16_bits_to_f32(s)
    return np.asarray(s)


def pack_int4(v: np.ndarray) -> np.ndarray:
    """(rows, cols) int8 values in [-8,7], cols even -> (rows, cols//2) uint8."""
    v = np.asarray(v, dtype=np.int8).astype(np.uint8) & 0xF
    return (v[:, 0::2] | (v[:, 1::2] << 4)).astype(np.uint8)


def unpack_int4(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=np.uint8)
    lo = (b & 0xF).astype(np.int8)
    hi = (b >> 4).astype(np.int8)
    out = np.empty((b.shape[0], b.shape[1] * 2), dtype=np.int8)
    out[:, 0::2] = np.where(lo > 7, lo - 16, lo)
    out[:, 1::2] = np.where(hi > 7, hi - 16, hi)
    return out


def np_reference_dtype(dt: DType):
    """NumPy dtype used by host-side reference computations."""
    return np.int64 if dt.is_int else np.float64
