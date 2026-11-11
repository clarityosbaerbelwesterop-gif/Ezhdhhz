#!/usr/bin/env python3
"""Insert donor implants into a MoE checkpoint (forge layout: model.safetensors, optim.safetensors, state.json).

Receiving layout (per MoE layer i, one global number of routed experts E):
    blocks.{i}.moe.experts.w1 / w3 [E, fe, d]     blocks.{i}.moe.experts.w2 [E, d, fe]
    blocks.{i}.moe.router.weight [E, d]           blocks.{i}.moe.router.bias [E]   (buffer; selection only)

What happens
  * implant rows are spread over the target layers in donor-depth order; every MoE layer grows by the same
    number of slots (the format has one E), layers with fewer implant rows get filler slots
  * an implant row is adapted to its layer: w1, w3 and the router row are divided by that layer's ffn_norm
    gain (the implant was folded for gain 1), w2 is multiplied by the layer's residual RMS when given
  * filler slot: small random w1/w3 (fixed seed), zero w2, zero router row - zero output, trainable
  * every new slot gets selection bias --off-bias (<= -1): scores are in (0, 1), so no new slot can enter
    the top-k; every token keeps its experts and the layer computes the same function as before
  * everything else is copied byte for byte; old rows of the grown tensors stay byte-identical (new rows
    are appended); the optimizer moments of new rows are zero
The file is streamed, memory stays small whatever the checkpoint size. Nothing is trained here.
"""
import argparse
import hashlib
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from titan.donor import st                                               # noqa: E402

GROWN = re.compile(r"^(v\.)?blocks\.(\d+)\.moe\.(experts\.w1|experts\.w3|experts\.w2|router\.weight|router\.bias)$")


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def f32(reader, ref) -> np.ndarray:
    return np.frombuffer(st.fetch(reader, ref), "<f4").reshape(ref.shape)


def streamed_std(reader, ref) -> float:
    n, s, q = 0, 0.0, 0.0
    for b in st.read_chunks(reader, ref.start, ref.nbytes):
        a = np.frombuffer(b, "<f4").astype(np.float64)
        n, s, q = n + a.size, s + a.sum(), q + (a * a).sum()
    return float(np.sqrt(max(q / n - (s / n) ** 2, 0.0)))


def plan_rows(rows: list, targets: list) -> dict:
    """Rows in donor-depth order, cut into len(targets) consecutive blocks of nearly equal size."""
    order = sorted(range(len(rows)), key=lambda j: (rows[j]["relative_depth"], rows[j]["file"], rows[j]["row"]))
    out = {t: [] for t in targets}
    for pos, j in enumerate(order):
        out[targets[pos * len(targets) // len(order)]].append(j)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="directory with model.safetensors, state.json (and optim.safetensors)")
    ap.add_argument("--implant", nargs="+", required=True, help="implant .safetensors (its .json index next to it)")
    ap.add_argument("--layers", required=True, help="receiving MoE layers, comma separated, shallow to deep")
    ap.add_argument("--seed-rms", default=None, help="JSON {layer: residual RMS at that layer's FFN} (default 1.0)")
    ap.add_argument("--off-bias", type=float, default=-4.0)
    ap.add_argument("--fill-seed", type=int, default=0)
    ap.add_argument("--fill-scale", type=float, default=1.0, help="std of filler w1/w3 relative to the layer's own")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.off_bias > -1.0:
        raise SystemExit("--off-bias must be <= -1: only then no new slot can enter the top-k")
    if os.path.abspath(a.out) == os.path.abspath(a.checkpoint):
        raise SystemExit("--out must differ from --checkpoint (the source is never modified)")
    os.makedirs(a.out, exist_ok=True)
    state = json.load(open(os.path.join(a.checkpoint, "state.json")))
    moe = state["moe"] if "moe" in state else state["config"]["moe"]
    mpath = os.path.join(a.checkpoint, "model.safetensors")
    rd = st.LocalReader(mpath)
    refs, hmeta = st.tensor_refs(rd, "model.safetensors")
    moe_layers = sorted(int(m.group(1)) for n in refs if (m := re.match(r"^blocks\.(\d+)\.moe\.experts\.w1$", n)))
    if not moe_layers:
        raise SystemExit("the checkpoint has no blocks.*.moe.experts.w1 tensor")
    E, fe, d = refs[f"blocks.{moe_layers[0]}.moe.experts.w1"].shape
    if int(moe["n_routed"]) != E or int(moe["expert_ffn"]) != fe:
        raise SystemExit(f"state.json says n_routed {moe['n_routed']}, expert_ffn {moe['expert_ffn']}; the tensors say {E}, {fe}")
    for i in moe_layers:
        for part, shape in (("experts.w1", (E, fe, d)), ("experts.w3", (E, fe, d)), ("experts.w2", (E, d, fe)), ("router.weight", (E, d)), ("router.bias", (E,))):
            if tuple(refs[f"blocks.{i}.moe.{part}"].shape) != shape:
                raise SystemExit(f"blocks.{i}.moe.{part}: shape {refs[f'blocks.{i}.moe.{part}'].shape}, expected {shape}")
    if E < int(moe["top_k"]):
        raise SystemExit("fewer routed experts than top_k")
    targets = [int(x) for x in a.layers.split(",")]
    if any(t not in moe_layers for t in targets) or len(set(targets)) != len(targets):
        raise SystemExit(f"--layers must be distinct MoE layers of the checkpoint ({moe_layers})")
    rms = {int(k): float(v) for k, v in json.load(open(a.seed_rms)).items()} if a.seed_rms else {}
    # ---- implant rows
    imps, rows = [], []
    for fi, path in enumerate(a.implant):
        ird = st.LocalReader(path)
        irefs, _ = st.tensor_refs(ird, os.path.basename(path))
        idx = json.load(open(os.path.splitext(path)[0] + ".json"))
        n, ife, idim = irefs["implant.experts.w1"].shape
        if (ife, idim) != (fe, d):
            raise SystemExit(f"{path}: slots are [{ife}, {idim}], the checkpoint needs [{fe}, {d}]")
        if tuple(irefs["implant.experts.w2"].shape) != (n, d, fe) or tuple(irefs["implant.router.weight"].shape) != (n, d) or len(idx["rows"]) != n:
            raise SystemExit(f"{path}: inconsistent implant")
        imps.append((ird, irefs, idx, path))
        rows += [{"file": fi, "row": r["row"], "relative_depth": r["relative_depth"], "donor_layer": r["donor_layer"],
                  "donor_expert": r["donor_expert"], "functions": r["functions"], "repo": idx.get("donor", {}).get("repo", "")} for r in idx["rows"]]
    plan = plan_rows(rows, targets)
    add = max(len(v) for v in plan.values())
    E2 = E + add
    gains = {}
    for t in targets:
        g = f32(rd, refs[f"blocks.{t}.ffn_norm.weight"]).astype(np.float64)
        if not np.isfinite(g).all() or np.abs(g).min() < 1e-6:
            raise SystemExit(f"blocks.{t}.ffn_norm.weight has a gain near zero; cannot adapt the implant to this layer")
        gains[t] = g
    stds = {(i, p): streamed_std(rd, refs[f"blocks.{i}.moe.experts.{p}"]) for i in moe_layers for p in ("w1", "w3")}

    def implant_row(j: int, part: str) -> np.ndarray:
        ird, irefs, _, _ = imps[rows[j]["file"]]
        ref = irefs["implant.router.weight" if part == "router.weight" else f"implant.{part}"]
        per = ref.nbytes // ref.shape[0]
        return np.frombuffer(ird.read(ref.start + rows[j]["row"] * per, per), "<f4").reshape(ref.shape[1:])

    def new_rows(layer: int, part: str, moments: bool):
        """Bytes of the `add` appended rows of one tensor."""
        mine = plan.get(layer, [])
        for s in range(add):
            shape = {"experts.w1": (fe, d), "experts.w3": (fe, d), "experts.w2": (d, fe), "router.weight": (d,), "router.bias": ()}[part]
            if moments:
                x = np.zeros(shape, "<f4")
            elif part == "router.bias":
                x = np.full(shape, a.off_bias, "<f4")
            elif s < len(mine):
                w = implant_row(mine[s], part).astype(np.float64)
                if part in ("experts.w1", "experts.w3"):
                    w = w / gains[layer][None, :]
                elif part == "router.weight":
                    w = w / gains[layer]
                else:
                    w = w * rms.get(layer, 1.0)
                x = w.astype("<f4")
            elif part in ("experts.w1", "experts.w3"):
                g = np.random.default_rng([a.fill_seed, layer, 1 if part.endswith("w1") else 3, s])
                x = (g.standard_normal(shape) * (stds[(layer, part[-2:])] * a.fill_scale)).astype("<f4")
            else:
                x = np.zeros(shape, "<f4")
            yield np.ascontiguousarray(x).tobytes()

    def grow(reader, all_refs, with_moments: bool):
        entries, untouched, grown = [], {}, {}
        for name, ref in all_refs.items():
            m = GROWN.match(name)
            if m:
                layer, part = int(m.group(2)), m.group(3)
                if ref.dtype != "F32" or ref.shape[0] != E:
                    raise SystemExit(f"{name}: {ref.dtype}{list(ref.shape)} cannot be grown (need F32 with {E} rows)")
                h_old = hashlib.sha256()

                def chunks(ref=ref, layer=layer, part=part, h_old=h_old):
                    for b in st.read_chunks(reader, ref.start, ref.nbytes):
                        h_old.update(b)
                        yield b
                    yield from new_rows(layer, part, with_moments)
                entries.append((name, ref.dtype, (E2,) + tuple(ref.shape[1:]), chunks))
                grown[name] = (ref, h_old)
            else:
                h = hashlib.sha256()

                def chunks(ref=ref, h=h):
                    for b in st.read_chunks(reader, ref.start, ref.nbytes):
                        h.update(b)
                        yield b
                entries.append((name, ref.dtype, tuple(ref.shape), chunks))
                untouched[name] = h
        return entries, untouched, grown

    old_sha = file_sha256(mpath)
    entries, untouched, grown = grow(rd, refs, False)
    inv = st.write_safetensors_stream(os.path.join(a.out, "model.safetensors"), entries, hmeta.get("metadata") or None)
    for n in untouched:                                      # what was written is what was read
        assert inv[n]["sha256"] == untouched[n].hexdigest(), n
    report = {"source": {"checkpoint": os.path.abspath(a.checkpoint), "model_sha256": old_sha},
              "result": {"model_sha256": inv["__file__"]["sha256"], "bytes": inv["__file__"]["bytes"]},
              "n_routed": {"before": E, "after": E2}, "expert_ffn": fe, "dim": d, "top_k": moe["top_k"], "off_bias": a.off_bias,
              "untouched_tensors": {"count": len(untouched), "bytes_identical": True},
              "grown_tensors": {"count": len(grown), "old_rows_bytes_identical": True, "old_rows_sha256": {n: h.hexdigest() for n, (_, h) in sorted(grown.items())}},
              "placement": {str(t): [{"slot": E + s, **{k: rows[j][k] for k in ("repo", "donor_layer", "donor_expert", "functions")},
                                      "implant": os.path.basename(imps[rows[j]["file"]][3]), "implant_row": rows[j]["row"]}
                                     for s, j in enumerate(plan[t])] for t in targets},
              "filler_slots": {str(i): add - len(plan.get(i, [])) for i in moe_layers},
              "parameters_added": {"implant": len(rows) * (3 * fe * d + d), "filler": sum(add - len(plan.get(i, [])) for i in moe_layers) * (3 * fe * d + d)},
              "adaptation": {"ffn_norm_gain": "w1, w3, router row divided by blocks.{layer}.ffn_norm.weight",
                             "residual_rms": {str(t): rms.get(t, 1.0) for t in targets}},
              "implants": [{"file": os.path.basename(p), "sha256": idx["sha256"], "donor": idx.get("donor")} for _, _, idx, p in imps]}
    opath = os.path.join(a.checkpoint, "optim.safetensors")
    if os.path.exists(opath):
        ord_ = st.LocalReader(opath)
        orefs, ometa = st.tensor_refs(ord_, "optim.safetensors")
        oent, _, ogrown = grow(ord_, orefs, True)
        oinv = st.write_safetensors_stream(os.path.join(a.out, "optim.safetensors"), oent, ometa.get("metadata") or None)
        report["optimizer"] = {"sha256": oinv["__file__"]["sha256"], "grown_tensors": len(ogrown), "new_rows": "zero moments"}
    else:
        report["optimizer"] = "no optim.safetensors in the source checkpoint"
    new_state = json.loads(json.dumps(state))
    (new_state["moe"] if "moe" in new_state else new_state["config"]["moe"])["n_routed"] = E2
    if "model_sha256" in new_state:
        new_state["model_sha256"] = inv["__file__"]["sha256"]
    if "parent_sha256" in new_state:
        new_state["parent_sha256"] = old_sha
    with open(os.path.join(a.out, "state.json"), "w") as f:
        f.write(json.dumps(new_state, indent=2, sort_keys=True))
    json.dump(report, open(os.path.join(a.out, "implant_report.json"), "w"), indent=1)
    print(json.dumps({k: report[k] for k in ("n_routed", "parameters_added", "filler_slots", "result")}, indent=1))
    for t in targets:
        print(f"layer {t}: " + ", ".join(f"slot {p['slot']} <- L{p['donor_layer']} e{p['donor_expert']} {p['functions']}" for p in report["placement"][str(t)]))


if __name__ == "__main__":
    main()
