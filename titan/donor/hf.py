"""Hugging Face repositories pinned to a revision: metadata, resolve URLs, cached safetensors headers
and tensors by HTTP range. No token is used; public repositories only."""
from __future__ import annotations

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import st

HUB = "https://huggingface.co"


def _get_json(url: str, timeout: float = 60.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


class HFRepo:
    def __init__(self, repo: str, revision: str | None = None, cache=None):
        self.repo = repo
        self.info = _get_json(f"{HUB}/api/models/{repo}" + (f"/revision/{revision}" if revision else "") + "?blobs=true")
        self.revision = self.info["sha"]
        if revision and revision != self.revision:
            raise ValueError(f"{repo}: asked for {revision}, hub answered {self.revision}")
        self.files = {s["rfilename"]: s for s in self.info["siblings"]}
        self.cache = Path(cache) if cache else None
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)
        self._refs: dict[str, st.TensorRef] | None = None
        self.bytes_fetched = 0

    # ------------------------------------------------------------------ provenance
    def provenance(self) -> dict:
        i = self.info
        card = i.get("cardData") or {}
        return {"repo": self.repo, "revision": self.revision, "last_modified": i.get("lastModified"),
                "gated": i.get("gated"), "private": i.get("private"),
                "license": card.get("license"), "license_name": card.get("license_name"),
                "license_link": card.get("license_link"), "used_storage_bytes": i.get("usedStorage"),
                "files": [{"path": n, "bytes": s.get("size"), "git_blob": s.get("blobId"),
                           "sha256": (s.get("lfs") or {}).get("sha256")} for n, s in sorted(self.files.items())]}

    def url(self, path: str) -> str:
        return f"{HUB}/{self.repo}/resolve/{self.revision}/{path}"

    def read_bytes(self, path: str) -> bytes:
        """Whole (small) file; cached."""
        c = self.cache / path.replace("/", "__") if self.cache else None
        if c and c.exists():
            return c.read_bytes()
        size = self.files[path]["size"]
        with urllib.request.urlopen(self.url(path), timeout=300) as r:     # plain GET: small files are served
            b = r.read()                                                   # without range support
        if len(b) != size:
            raise IOError(f"{path}: got {len(b)} bytes, hub lists {size}")
        self.bytes_fetched += len(b)
        if c:
            c.write_bytes(b)
        return b

    # ------------------------------------------------------------------ tensors
    def weight_files(self) -> list[str]:
        return sorted(n for n in self.files if n.endswith(".safetensors"))

    def header(self, shard: str) -> tuple[dict, dict]:
        c = self.cache / (shard.replace("/", "__") + ".header.json") if self.cache else None
        if c and c.exists():
            d = json.loads(c.read_text())
            return {n: st.TensorRef(n, t[0], tuple(t[1]), t[2], t[3], shard) for n, t in d["tensors"].items()}, d["info"]
        refs, info = st.tensor_refs(st.HttpReader(self.url(shard)), shard)
        self.bytes_fetched += 8 + info["header_len"]
        if c:
            c.write_text(json.dumps({"info": info, "tensors": {n: [r.dtype, list(r.shape), r.start, r.end]
                                                                for n, r in refs.items()}}))
        return refs, info

    def refs(self, threads: int = 8) -> dict[str, st.TensorRef]:
        """Every tensor of the repository (all shards)."""
        if self._refs is None:
            out = {}
            with ThreadPoolExecutor(threads) as ex:
                for refs, _ in ex.map(self.header, self.weight_files()):
                    dup = set(out) & set(refs)
                    if dup:
                        raise ValueError(f"tensor in two shards: {sorted(dup)[:3]}")
                    out.update(refs)
            self._refs = out
        return self._refs

    def fetch(self, name: str) -> tuple[st.TensorRef, bytes]:
        ref = self.refs()[name]
        b = st.HttpReader(self.url(ref.shard)).read(ref.start, ref.nbytes)
        self.bytes_fetched += len(b)
        return ref, b

    def fetch_span(self, names: list[str], threads: int = 6, max_gap: int = 1 << 20) -> dict[str, bytes]:
        """Several tensors with few requests: tensors that lie (nearly) back to back in a shard are read as
        one range; ranges are downloaded in parallel."""
        refs = self.refs()
        by_shard: dict[str, list] = {}
        for n in names:
            by_shard.setdefault(refs[n].shard, []).append(refs[n])
        spans = []
        for shard, rs in by_shard.items():
            rs.sort(key=lambda r: r.start)
            cur = [rs[0]]
            for r in rs[1:]:
                if r.start - cur[-1].end <= max_gap and r.end - cur[0].start <= 2 << 30:
                    cur.append(r)
                else:
                    spans.append((shard, cur))
                    cur = [r]
            spans.append((shard, cur))

        piece = 48 << 20
        jobs = []                                              # (span index, offset in span, absolute start, length)
        for si, (shard, rs) in enumerate(spans):
            a0, b0 = rs[0].start, rs[-1].end
            jobs += [(si, o - a0, o, min(piece, b0 - o)) for o in range(a0, b0, piece)]

        def get(job):
            si, rel, start, length = job
            return si, rel, st.HttpReader(self.url(spans[si][0])).read(start, length)

        bufs = [bytearray(rs[-1].end - rs[0].start) for _, rs in spans]
        with ThreadPoolExecutor(threads) as ex:
            for si, rel, blob in ex.map(get, jobs):
                bufs[si][rel:rel + len(blob)] = blob
                self.bytes_fetched += len(blob)
        out = {}
        for (shard, rs), buf in zip(spans, bufs):
            a0 = rs[0].start
            mv = memoryview(buf)
            for r in rs:
                out[r.name] = bytes(mv[r.start - a0:r.end - a0])
        return out
