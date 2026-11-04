"""Memory hierarchy: byte-addressable emulated memories, L2 cache model and allocators."""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
from numpy.lib.stride_tricks import as_strided

from ..dtypes import STORAGE, DType, from_storage, pack_int4, to_storage, unpack_int4
from ..isa.spec import ErrorCode, ISAError, TensorRef


class Memory:
    """A flat byte-addressable memory holding real data (functional state)."""

    def __init__(self, name: str, size: int):
        self.name = name
        self.size = int(size)
        self.buf = np.zeros(self.size, dtype=np.uint8)

    def _check(self, ref: TensorRef) -> None:
        if ref.addr < 0 or ref.end > self.size:
            raise ISAError(ErrorCode.OUT_OF_BOUNDS, f"{ref} outside {self.name} ({self.size} B)")

    def _view(self, ref: TensorRef) -> np.ndarray:
        e = ref.dtype.bits // 8
        base = self.buf[ref.addr:ref.end].view(STORAGE[ref.dtype])
        return as_strided(base, (ref.rows, ref.cols), (ref.stride * e, e), writeable=True)

    def _int4_rows(self, ref: TensorRef) -> np.ndarray:
        rb, sb = ref.cols // 2, ref.stride // 2
        base = self.buf[ref.addr:ref.end]
        return as_strided(base, (ref.rows, rb), (sb, 1), writeable=True)

    def read(self, ref: TensorRef) -> np.ndarray:
        """Return a copy of the logical values (see titan.dtypes)."""
        self._check(ref)
        if ref.dtype == DType.INT4:
            return unpack_int4(self._int4_rows(ref))
        return from_storage(self._view(ref).copy(), ref.dtype)

    def write(self, ref: TensorRef, values: np.ndarray) -> None:
        self._check(ref)
        v = np.asarray(values)
        if v.shape != ref.shape:
            v = np.broadcast_to(v, ref.shape)
        if ref.dtype == DType.INT4:
            self._int4_rows(ref)[...] = pack_int4(v)
        else:
            self._view(ref)[...] = to_storage(v, ref.dtype)

    def read_bytes(self, addr: int, n: int) -> bytes:
        return self.buf[addr:addr + n].tobytes()

    def fill(self, value: int = 0) -> None:
        self.buf[:] = value


class L2Cache:
    """Set-associative LRU cache model for global-memory traffic (timing only; data lives in Memory).

    Write policy: write-through, no-write-allocate (lines present are updated in place).
    """

    def __init__(self, size_bytes: int, line: int, ways: int):
        self.line, self.ways = line, ways
        self.sets = max(1, size_bytes // (line * ways))
        self._sets = [OrderedDict() for _ in range(self.sets)]
        self.read_hits = self.read_misses = self.write_lines = self.write_hits = 0

    def _lines(self, ref: TensorRef):
        rb = ref.nbytes // ref.rows if ref.rows else 0
        row_bytes = (ref.stride * ref.dtype.bits) // 8
        if ref.contiguous:
            lo, hi = ref.addr, ref.addr + ref.nbytes
            yield from range(lo // self.line, (hi - 1) // self.line + 1)
            return
        seen = -1
        for r in range(ref.rows):
            lo = ref.addr + r * row_bytes
            for ln in range(lo // self.line, (lo + rb - 1) // self.line + 1):
                if ln != seen:
                    seen = ln
                    yield ln

    def access(self, ref: TensorRef, write: bool) -> tuple[int, int]:
        """Return (hit_lines, miss_lines) and update LRU state."""
        hits = misses = 0
        for ln in self._lines(ref):
            s = self._sets[ln % self.sets]
            if ln in s:
                s.move_to_end(ln)
                hits += 1
            else:
                misses += 1
                if not write:
                    s[ln] = True
                    if len(s) > self.ways:
                        s.popitem(last=False)
        if write:
            self.write_lines += hits + misses
            self.write_hits += hits
        else:
            self.read_hits += hits
            self.read_misses += misses
        return hits, misses

    def stats(self) -> dict:
        reads = self.read_hits + self.read_misses
        return {"read_hits": self.read_hits, "read_misses": self.read_misses,
                "read_hit_rate": self.read_hits / reads if reads else 0.0,
                "write_lines": self.write_lines, "write_hits": self.write_hits}


class Allocator:
    """First-fit allocator with coalescing free list (used for global and SRAM planning)."""

    def __init__(self, size: int, base: int = 0, align: int = 64):
        self.base, self.size, self.align = base, size, align
        self.free = [(base, base + size)]
        self.used: dict[int, int] = {}
        self.high_water = base

    def alloc(self, n: int, align: int | None = None) -> int:
        a = align or self.align
        n = max(int(n), 1)
        for i, (lo, hi) in enumerate(self.free):
            start = -(-lo // a) * a
            if start + n <= hi:
                pieces = []
                if start > lo:
                    pieces.append((lo, start))
                if start + n < hi:
                    pieces.append((start + n, hi))
                self.free[i:i + 1] = pieces
                self.used[start] = n
                self.high_water = max(self.high_water, start + n)
                return start
        raise MemoryError(f"allocator: cannot allocate {n} bytes (capacity {self.size})")

    def release(self, addr: int) -> None:
        n = self.used.pop(addr)
        self.free.append((addr, addr + n))
        self.free.sort()
        merged = [self.free[0]]
        for lo, hi in self.free[1:]:
            if lo <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
            else:
                merged.append((lo, hi))
        self.free = merged

    @property
    def peak(self) -> int:
        return self.high_water - self.base
