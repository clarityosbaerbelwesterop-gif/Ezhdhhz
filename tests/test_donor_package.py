"""tools/donor/verify_package.py: split ZIP packages are checked part by part, as a whole and file by file."""
import hashlib
import os
import subprocess
import sys
import zipfile

import pytest

TOOL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "donor", "verify_package.py")


def run(*args):
    return subprocess.run([sys.executable, TOOL, *map(str, args)], capture_output=True, text=True)


def build(tmp, files, tamper=None, part=700):
    root = "paket"
    man = "".join(f"{hashlib.sha256(b).hexdigest()}  {p}\n" for p, b in sorted(files.items()))
    zp = tmp / "paket.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
        for p, b in sorted(files.items()):
            z.writestr(f"{root}/{p}", tamper if tamper and p == "a/x.bin" else b)
        z.writestr(f"{root}/MANIFEST.sha256", man)
    data = zp.read_bytes()
    parts = []
    for i in range(0, len(data), part):
        p = tmp / f"paket.zip.{i // part + 1:03d}"
        p.write_bytes(data[i:i + part])
        parts.append(p)
    sums = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}" for p in parts] + [f"{hashlib.sha256(data).hexdigest()}  paket.zip"]
    (tmp / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n")
    zp.unlink()
    return parts


@pytest.fixture()
def files():
    return {"a/x.bin": os.urandom(1500), "b/y.json": b'{"k": 1}', "README.md": b"# r\n"}


def test_good_package_passes_and_extracts(tmp_path, files):
    parts = build(tmp_path, files)
    assert len(parts) > 2
    r = run(*reversed(parts), "--extract", tmp_path / "out", "--join", tmp_path / "joined.zip")   # order of arguments does not matter
    assert r.returncode == 0, r.stdout + r.stderr
    assert "OK: 3 files" in r.stdout
    assert (tmp_path / "out" / "paket" / "a" / "x.bin").read_bytes() == files["a/x.bin"]
    assert zipfile.ZipFile(tmp_path / "joined.zip").testzip() is None


def test_damaged_missing_and_tampered_packages_fail(tmp_path, files):
    parts = build(tmp_path, files)
    b = bytearray(parts[1].read_bytes())
    b[10] ^= 1
    parts[1].write_bytes(bytes(b))
    assert "FAIL" in run(*parts).stderr                                    # damaged part
    t2 = tmp_path / "t2"
    t2.mkdir()
    parts = build(t2, files)
    assert run(parts[0], parts[2]).returncode != 0                          # gap in the numbering
    t3 = tmp_path / "t3"
    t3.mkdir()
    parts = build(t3, files, tamper=os.urandom(1500))                       # content differs from MANIFEST.sha256
    r = run(*parts)
    assert r.returncode != 0 and "content differs: paket/a/x.bin" in r.stdout
