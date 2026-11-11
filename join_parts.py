#!/usr/bin/env python3
"""Rebuild the large files of this branch from their parts and check every file against MANIFEST.sha256.

GitHub refuses files over 100 MB, so every file over 95 MiB is stored as <name>.part01, <name>.part02, ...
    python join_parts.py                  # rebuild all large files, then check everything
    python join_parts.py implants         # only files under implants/ (faster), then check those
    python join_parts.py --delete-parts   # remove the parts after a successful check
Standard library only; Python 3.8 or newer. Run it in the branch's top folder.
"""
import argparse
import hashlib
import os
import re
import sys

PART = re.compile(r"^(.*)\.part(\d{2})$")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("only", nargs="*", help="folders or files to restrict to")
    ap.add_argument("--delete-parts", action="store_true")
    a = ap.parse_args()
    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)
    want = lambda p: not a.only or any(p == o.rstrip("/") or p.startswith(o.rstrip("/") + "/") for o in a.only)
    groups = {}
    for d, _, files in os.walk("."):
        if "/.git" in d or d.startswith("./.git"):
            continue
        for f in files:
            m = PART.match(f)
            if m:
                p = os.path.join(d, m.group(1))[2:]
                if want(p):
                    groups.setdefault(p, []).append((int(m.group(2)), os.path.join(d, f)))
    for p, parts in sorted(groups.items()):
        parts.sort()
        nums = [n for n, _ in parts]
        if nums != list(range(1, len(parts) + 1)):
            sys.exit(f"FAIL {p}: parts {nums} are not 1..{len(parts)} without gaps")
        tmp = p + ".joining"
        with open(tmp, "wb") as out:
            for _, q in parts:
                with open(q, "rb") as f:
                    for b in iter(lambda: f.read(1 << 22), b""):
                        out.write(b)
        os.replace(tmp, p)
        print(f"joined {p} from {len(parts)} parts")
    listed = {}
    for line in open("MANIFEST.sha256", encoding="utf-8"):
        h, p = line.rstrip("\n").split("  ", 1)
        if want(p):
            listed[p] = h
    bad = [p for p, h in sorted(listed.items()) if not os.path.exists(p) or sha256(p) != h]
    for p in bad:
        print(f"FAIL {p}: {'missing' if not os.path.exists(p) else 'content differs'}")
    if bad:
        sys.exit(f"FAIL: {len(bad)} of {len(listed)} files do not match MANIFEST.sha256")
    print(f"OK: {len(listed)} files match MANIFEST.sha256")
    if a.delete_parts:
        for parts in groups.values():
            for _, q in parts:
                os.remove(q)
        print(f"removed the parts of {len(groups)} files")


if __name__ == "__main__":
    main()
