#!/usr/bin/env python3
"""Stack projected donor experts into one implant file in the receiving MoE's tensor layout.

Input: the per-expert slot tensors written by qwen_implant.py / mistral_implant.py
(slot.L{layer}.e{expert}.w1 [fe, d], .w3 [fe, d], .w2 [d, fe], .router [d]) and their manifest.
Output per group:  <prefix>_<group>.safetensors
    implant.experts.w1 [N, fe, d]   implant.experts.w3 [N, fe, d]   implant.experts.w2 [N, d, fe]
    implant.router.weight [N, d]
and <prefix>_<group>.json: one entry per row (donor layer and expert, why it was selected, hashes of
the source tensors, measured loss of the width reduction). Rows are byte copies of the source tensors
(checked by hash), ordered by donor depth. Nothing is computed here; the class stays PROJECTION.
tools/donor/insert_implant.py puts such a file into a checkpoint.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from titan.donor import st                                               # noqa: E402

PARTS = (("w1", "implant.experts.w1"), ("w3", "implant.experts.w3"), ("w2", "implant.experts.w2"), ("router", "implant.router.weight"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--projection", nargs="+", required=True, help="safetensors file(s) with slot.L*.e*.* tensors")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--donor-layers", type=int, required=True, help="number of layers of the donor (for relative depth)")
    ap.add_argument("--group", action="append", default=[], help="NAME=function[,function]; default: one group 'all'")
    ap.add_argument("--default-function", default="code", help="function name for manifests without selected_for")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    man = json.load(open(a.manifest))
    src = {}
    for f in a.projection:
        rd = st.LocalReader(f)
        refs, _ = st.tensor_refs(rd, os.path.basename(f))
        for n, r in refs.items():
            if n.startswith("slot."):
                src[n] = (rd, r)
    experts = []
    for e in man["experts"]:
        L, x = int(e["layer"]), int(e["expert"])
        fns = [w["function"] for w in e.get("selected_for", [])] or [a.default_function]
        key = f"slot.L{L}.e{x}."
        if key + "w1" not in src:
            raise SystemExit(f"{key}w1 is not in the projection file(s)")
        fe, d = src[key + "w1"][1].shape
        rec = {"donor_layer": L, "donor_expert": x, "functions": fns, "relative_depth": L / (a.donor_layers - 1), "key": key}
        if "selected_for" in e:
            rec["selection"] = e["selected_for"]
        us = e.get("unit_selection", {}).get("by_width", {}).get(str(fe))
        fit = e.get("projection", {}).get(f"fe{fe}", {}).get("fit") if isinstance(e.get("projection", {}).get(f"fe{fe}"), dict) else None
        if us:
            rec["width_reduction"] = us
        elif fit:
            rec["width_reduction"] = {k: v for k, v in fit.items() if k != "kept_units"}
        experts.append(rec)
    experts.sort(key=lambda r: (r["donor_layer"], r["donor_expert"]))
    fe, d = src[experts[0]["key"] + "w1"][1].shape
    groups = dict(g.split("=", 1) for g in a.group) or {"all": None}
    for gname, fl in groups.items():
        want = set(fl.split(",")) if fl else None
        rows = [r for r in experts if want is None or want & set(r["functions"])]
        if not rows:
            raise SystemExit(f"group {gname}: no experts")
        N = len(rows)
        shapes = {"w1": (N, fe, d), "w3": (N, fe, d), "w2": (N, d, fe), "router": (N, d)}
        entries = []
        for part, name in PARTS:
            def chunks(part=part):
                for r in rows:
                    rd, ref = src[r["key"] + part]
                    yield from st.read_chunks(rd, ref.start, ref.nbytes)
            for r in rows:
                if tuple(src[r["key"] + part][1].shape) != shapes[part][1:]:
                    raise SystemExit(f"{r['key']}{part}: shape differs")
            entries.append((name, "F32", shapes[part], chunks))
        fn = f"{a.prefix}_{gname}.safetensors"
        prov = man.get("provenance", {})
        inv = st.write_safetensors_stream(os.path.join(a.out, fn), entries,
                                          {"kind": "donor implant (PROJECTION; initialisation, not calibrated on the receiving model)",
                                           "repo": prov.get("repo", ""), "revision": prov.get("revision", ""), "slot_width": fe, "dim": d, "rows": N})
        for i, r in enumerate(rows):
            r["row"] = i
            r["source_sha256"] = {}
            for part, _ in PARTS:
                rd, ref = src[r["key"] + part]
                r["source_sha256"][part] = st.sha256(st.fetch(rd, ref))
        idx = {"file": fn, **inv["__file__"], "class": "PROJECTION", "donor": prov, "donor_layers": a.donor_layers, "slot_width": fe, "dim": d,
               "rows": [{k: v for k, v in r.items() if k != "key"} for r in rows],
               "tensors": {n: inv[n] for _, n in PARTS},
               "assumptions": {"slot_gain": "1.0 - insert_implant.py divides w1, w3 and the router rows by the receiving layer's ffn_norm gain",
                               "seed_residual_rms": "1.0 - a scalar on w2, set at insertion when measured",
                               "bridges": man.get("bridges")}}
        json.dump(idx, open(os.path.join(a.out, f"{a.prefix}_{gname}.json"), "w"), indent=1)
        print(f"{fn}: {N} experts, fe {fe}, d {d}, {inv['__file__']['bytes']} bytes, sha256 {inv['__file__']['sha256']}")


if __name__ == "__main__":
    main()
