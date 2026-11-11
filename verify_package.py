#!/usr/bin/env python3
"""Check (and optionally unpack) a delivery package that arrives as a ZIP split into byte parts.

    python verify_package.py paket.zip.001 paket.zip.002 ...      # parts in any order; SHA256SUMS.txt is found next to them
    python verify_package.py paket.zip                              # or the joined file
    options: --sums SHA256SUMS.txt   --extract DIR   --join OUT.zip

Checks, in this order, and stops at the first failure:
  1. every part against its SHA-256 in SHA256SUMS.txt (when the file is there)
  2. the joined byte stream against the SHA-256 of the whole ZIP in SHA256SUMS.txt
  3. every file inside the ZIP against MANIFEST.sha256 inside the package; no file missing, none extra
The parts are read in place (no joined copy needed). Standard library only; Python 3.8 or newer.
"""
import argparse
import hashlib
import io
import os
import re
import sys
import zipfile

CHUNK = 1 << 22


class Parts(io.RawIOBase):
    """Several files read as one seekable byte stream."""

    def __init__(self, paths):
        self.paths = list(paths)
        self.sizes = [os.path.getsize(p) for p in self.paths]
        self.starts = [sum(self.sizes[:i]) for i in range(len(self.paths))]
        self.size = sum(self.sizes)
        self.pos = 0
        self.handles = [open(p, "rb") for p in self.paths]

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def readinto(self, b):
        n, done = len(b), 0
        while done < n and self.pos < self.size:
            i = max(j for j, s in enumerate(self.starts) if s <= self.pos)
            f = self.handles[i]
            f.seek(self.pos - self.starts[i])
            want = min(n - done, self.starts[i] + self.sizes[i] - self.pos)
            got = f.readinto(memoryview(b)[done:done + want])
            if not got:
                break
            done += got
            self.pos += got
        return done

    def close(self):
        for f in self.handles:
            f.close()
        super().close()


def sha256_stream(f) -> str:
    h = hashlib.sha256()
    for b in iter(lambda: f.read(CHUNK), b""):
        h.update(b)
    return h.hexdigest()


def read_sums(path) -> dict:
    out = {}
    for line in open(path, encoding="utf-8"):
        m = re.match(r"^([0-9a-f]{64})\s+\*?(.+?)\s*$", line)
        if m:
            out[os.path.basename(m.group(2))] = m.group(1)
    return out


def part_key(p):
    m = re.search(r"\.(\d+)$", p)
    return (0, int(m.group(1))) if m else (1, p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parts", nargs="+")
    ap.add_argument("--sums", default=None)
    ap.add_argument("--extract", default=None)
    ap.add_argument("--join", default=None)
    a = ap.parse_args()
    parts = sorted(a.parts, key=part_key)
    sums_path = a.sums or os.path.join(os.path.dirname(os.path.abspath(parts[0])), "SHA256SUMS.txt")
    sums = read_sums(sums_path) if os.path.exists(sums_path) else {}
    ok = True
    print(f"{len(parts)} file(s); checksums: {sums_path if sums else 'none found'}")
    if len(parts) > 1:
        nums = [part_key(p)[1] for p in parts if part_key(p)[0] == 0]
        if nums != list(range(1, len(parts) + 1)):
            sys.exit(f"FAIL parts are not numbered 1..{len(parts)} without gaps: {nums}")
    # 1. parts
    for p in parts:
        if os.path.basename(p) in sums:
            with open(p, "rb") as f:
                good = sha256_stream(f) == sums[os.path.basename(p)]
            print(f"  {'ok  ' if good else 'FAIL'} {os.path.basename(p)}")
            ok &= good
    if not ok:
        sys.exit("FAIL: a part does not match its checksum (download again)")
    # 2. whole ZIP
    whole = Parts(parts)
    zip_name = re.sub(r"\.\d+$", "", os.path.basename(parts[0]))
    digest = sha256_stream(whole)
    if zip_name in sums:
        good = digest == sums[zip_name]
        print(f"  {'ok  ' if good else 'FAIL'} {zip_name} (joined, {whole.size} bytes) {digest}")
        if not good:
            sys.exit("FAIL: the joined parts are not the original ZIP (missing or reordered part?)")
    else:
        print(f"  --   {zip_name} (joined, {whole.size} bytes) {digest}  (no reference checksum)")
    # 3. files inside
    whole.seek(0)
    z = zipfile.ZipFile(io.BufferedReader(whole, CHUNK))
    names = [n for n in z.namelist() if not n.endswith("/")]
    man = [n for n in names if n.endswith("/MANIFEST.sha256") or n == "MANIFEST.sha256"]
    if len(man) != 1:
        sys.exit("FAIL: no single MANIFEST.sha256 in the package")
    root = man[0][: -len("MANIFEST.sha256")]
    listed = {}
    for line in z.read(man[0]).decode("utf-8").splitlines():
        h, p = line.split("  ", 1)
        listed[root + p] = h
    missing = sorted(set(listed) - set(names))
    extra = sorted(set(names) - set(listed) - {man[0]})
    bad = []
    for i, n in enumerate(sorted(listed)):
        if n in missing:
            continue
        with z.open(n) as f:
            if sha256_stream(f) != listed[n]:
                bad.append(n)
        if (i + 1) % 10 == 0 or i + 1 == len(listed):
            print(f"  checked {i + 1}/{len(listed)} files", end="\r", flush=True)
    print()
    for n in missing:
        print(f"  FAIL missing {n}")
    for n in extra:
        print(f"  FAIL not in MANIFEST.sha256: {n}")
    for n in bad:
        print(f"  FAIL content differs: {n}")
    if missing or extra or bad:
        sys.exit("FAIL: package content does not match MANIFEST.sha256")
    print(f"OK: {len(listed)} files in {root or '.'} match MANIFEST.sha256")
    if a.extract:
        z.extractall(a.extract)
        print(f"extracted to {a.extract}")
    if a.join:
        whole.seek(0)
        with open(a.join, "wb") as f:
            for b in iter(lambda: whole.read(CHUNK), b""):
                f.write(b)
        print(f"joined to {a.join}")
    z.close()
    whole.close()


if __name__ == "__main__":
    main()
