"""Safetensors by byte range: read headers and single tensors from local files or HTTP without loading
a whole shard, hash every tensor (SHA-256 of its exact payload bytes) and write subset files.

Format: 8 bytes little-endian u64 N, N bytes JSON header {name: {dtype, shape, data_offsets:[a,b]}},
then the payload; a tensor's bytes are payload[a:b] = file[8+N+a : 8+N+b].
"""
from __future__ import annotations

import hashlib
import json
import struct
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

import numpy as np

DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
               "U16": 2, "U32": 4, "U64": 8, "F8_E4M3": 1, "F8_E5M2": 1}
_NP = {"F64": "<f8", "F32": "<f4", "F16": "<f2", "I64": "<i8", "I32": "<i4", "I16": "<i2", "I8": "i1", "U8": "u1",
       "BOOL": "?", "U16": "<u2", "U32": "<u4", "U64": "<u8"}


class LocalReader:
    def __init__(self, path):
        self.path = str(path)
        self.bytes_read = 0

    def read(self, offset: int, length: int) -> bytes:
        with open(self.path, "rb") as f:
            f.seek(offset)
            b = f.read(length)
        if len(b) != length:
            raise IOError(f"{self.path}: short read at {offset}+{length}")
        self.bytes_read += length
        return b


class HttpReader:
    """HTTP range reader with retries (follows redirects; honours HTTPS_PROXY / SSL_CERT_FILE)."""

    def __init__(self, url: str, retries: int = 5, timeout: float = 120.0, chunk: int = 64 << 20):
        self.url, self.retries, self.timeout, self.chunk = url, retries, timeout, chunk
        self.bytes_read = 0

    def _one(self, offset: int, length: int) -> bytes:
        last = None
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={offset}-{offset + length - 1}"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    if r.status != 206:
                        raise IOError(f"range request answered {r.status}, not 206")
                    b = r.read()
                if len(b) != length:
                    raise IOError(f"short read {len(b)} != {length}")
                return b
            except (urllib.error.URLError, IOError, TimeoutError, ConnectionError) as e:
                last = e
                time.sleep(min(30.0, 2.0 ** attempt))
        raise IOError(f"{self.url}: range {offset}+{length} failed after {self.retries} tries: {last}")

    def read(self, offset: int, length: int) -> bytes:
        parts = [self._one(o, min(self.chunk, offset + length - o)) for o in range(offset, offset + length, self.chunk)]
        self.bytes_read += length
        return parts[0] if len(parts) == 1 else b"".join(parts)


@dataclass(frozen=True)
class TensorRef:
    name: str
    dtype: str
    shape: tuple
    start: int           # absolute file offsets of the payload bytes
    end: int
    shard: str = ""

    @property
    def nbytes(self) -> int:
        return self.end - self.start

    @property
    def numel(self) -> int:
        return int(np.prod(self.shape)) if self.shape else 1


def read_header(reader) -> tuple[int, dict, dict]:
    """(header_len, {name: TensorRef-less dict}, __metadata__)."""
    n = struct.unpack("<Q", reader.read(0, 8))[0]
    if not 2 <= n <= 256 << 20:
        raise ValueError(f"implausible safetensors header length {n}")
    hdr = json.loads(reader.read(8, n))
    meta = hdr.pop("__metadata__", {}) or {}
    return n, hdr, meta


def tensor_refs(reader, shard: str = "") -> tuple[dict[str, TensorRef], dict]:
    n, hdr, meta = read_header(reader)
    base = 8 + n
    refs = {}
    for name, t in hdr.items():
        a, b = t["data_offsets"]
        ref = TensorRef(name, t["dtype"], tuple(t["shape"]), base + a, base + b, shard)
        want = ref.numel * DTYPE_BYTES.get(ref.dtype, 0)
        if want and want != ref.nbytes:
            raise ValueError(f"{name}: {ref.nbytes} payload bytes, shape/dtype imply {want}")
        refs[name] = ref
    return refs, {"header_len": n, "metadata": meta}


def fetch(reader, ref: TensorRef) -> bytes:
    return reader.read(ref.start, ref.nbytes)


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def bf16_to_f32(raw: bytes | np.ndarray) -> np.ndarray:
    """Exact widening: a bfloat16 is the upper 16 bits of the float32 with the same value."""
    u = np.frombuffer(raw, dtype="<u2") if isinstance(raw, (bytes, bytearray, memoryview)) else np.asarray(raw, "<u2")
    return (u.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even float32 -> bfloat16 bit patterns (uint16)."""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint32)
    nan = np.isnan(x)
    out = r.astype(np.uint16)
    if nan.any():
        out = np.where(nan, np.uint16(0x7FC0), out)
    return out


def decode(raw: bytes, dtype: str, shape) -> np.ndarray:
    """Payload bytes -> numpy array with the tensor's VALUES (BF16 and FP8 widened exactly to float32)."""
    if dtype == "BF16":
        return bf16_to_f32(raw).reshape(shape)
    if dtype in ("F8_E4M3", "F8_E5M2"):
        from .fp8 import decode_fp8
        return decode_fp8(raw, dtype).reshape(shape)
    return np.frombuffer(raw, dtype=_NP[dtype]).reshape(shape)


def write_safetensors(path, tensors: dict, metadata: dict | None = None) -> dict:
    """tensors: {name: (dtype, shape, payload_bytes)}; names sorted, 8-byte aligned header (as the reference
    writer). Returns {name: {dtype, shape, nbytes, sha256}} plus the file's sha256 under '__file__'."""
    hdr, off, inv = {}, 0, {}
    names = sorted(tensors)
    for name in names:
        dtype, shape, payload = tensors[name]
        if DTYPE_BYTES[dtype] * int(np.prod(shape) if len(shape) else 1) != len(payload):
            raise ValueError(f"{name}: payload size does not match {dtype}{list(shape)}")
        hdr[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + len(payload)]}
        inv[name] = {"dtype": dtype, "shape": list(shape), "nbytes": len(payload), "sha256": sha256(payload)}
        off += len(payload)
    if metadata:
        hdr["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    js = json.dumps(hdr, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 8)
    h = hashlib.sha256()
    with open(path, "wb") as f:
        for part in [struct.pack("<Q", len(js)), js] + [tensors[n][2] for n in names]:
            f.write(part)
            h.update(part)
    inv["__file__"] = {"sha256": h.hexdigest(), "bytes": 8 + len(js) + off}
    return inv
