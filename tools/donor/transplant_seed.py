#!/usr/bin/env python3
"""Dense seed checkpoint -> MoE checkpoint (the team's own upcycle) -> implants inserted -> checked.

    python tools/donor/transplant_seed.py --accel /workspace/run2-prep/moe --dense <step-007500> \\
        --implant implants/implant_qwen3coder_fe640_code.safetensors implants/implant_mistral_l4_fe640_*.safetensors \\
        --layers 16,17,18,19,20,21,22,23 --out <work>/transplant-01

--accel is the directory that holds the team's `accel` package (moe.py, model.py, config.py, attn.py, forge_io.py) with
the repository's scripts/rebase.py next to it, exactly as the trainer uses them. Nothing of the team's code is copied
or changed; this script only calls it:
  1. forge_io.load_checkpoint(--dense)                      the dense champion (state.json + model.safetensors)
  2. moe.upcycle(dense, base, n_routed, n_shared, top_k, expert_ffn, noise=0)
     exact when (ffn / expert_ffn) - n_shared == top_k: the MoE computes the dense function at step 0
  3. writes <out>/upcycled  (model, optim with zero moments, state.json with "moe" and parent_sha256)
  4. tools/donor/insert_implant.py  -> <out>/implanted       (new slots switched off by selection bias)
  5. forward check with the team's model.forward / moe.forward on --check-ids (token ids .npy) or random ids:
     largest logit difference dense vs upcycled and upcycled vs implanted, load of the new slots, and the
     mean next-token loss of all three (equal when the transplant keeps the model)
  6. <out>/transplant_report.json: hashes of every input and output, configs, parameter counts, checks
Memory: about twice the dense model in float32 for steps 1-3 (8 GB for the 959.5 M champion); files are streamed;
the forward check holds one model at a time (skip it with --no-forward-check). Nothing is trained.
"""
import argparse
import copy
import gc
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from titan.donor import st                                               # noqa: E402

f32 = np.float32


def sha_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def array_entries(arrays: dict):
    """(name, dtype, shape, chunks) for st.write_safetensors_stream; one tensor in memory at a time."""
    out = []
    for name, a in arrays.items():
        def chunks(a=a):
            yield np.ascontiguousarray(a, "<f4").tobytes()
        out.append((name, "F32", tuple(np.shape(a)), chunks))
    return out


def zero_entries(specs: list, prefix: str = ""):
    out = []
    for name, shape in specs:
        n = int(np.prod(shape)) * 4

        def chunks(n=n):
            block = bytes(1 << 24)
            for o in range(0, n, len(block)):
                yield block[:min(len(block), n - o)]
        out.append((prefix + name, "F32", tuple(shape), chunks))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--accel", required=True, help="directory containing the team's accel package")
    ap.add_argument("--dense", required=True, help="dense forge checkpoint directory (state.json, model.safetensors)")
    ap.add_argument("--implant", nargs="+", required=True)
    ap.add_argument("--layers", required=True, help="receiving layers, comma separated, shallow to deep")
    ap.add_argument("--n-routed", type=int, default=6)
    ap.add_argument("--n-shared", type=int, default=2)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--expert-ffn", type=int, default=640)
    ap.add_argument("--off-bias", type=float, default=-4.0)
    ap.add_argument("--seed-rms", default=None, help="JSON {layer: residual RMS at that layer's FFN}")
    ap.add_argument("--check-ids", default=None, help=".npy int token ids [B, T] for the forward check (real text is best)")
    ap.add_argument("--check-shape", default="2,64", help="B,T of random ids when --check-ids is not given")
    ap.add_argument("--no-forward-check", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out) and os.listdir(a.out):
        raise SystemExit(f"--out {a.out} exists and is not empty")
    os.makedirs(a.out, exist_ok=True)
    sys.path.insert(0, os.path.abspath(a.accel))
    from accel import forge_io as FIO, model as M, moe as MOE                # the team's own code
    from accel.config import ModelConfig
    report = {"tool": "tools/donor/transplant_seed.py", "accel": os.path.abspath(a.accel), "inputs": {}, "steps": {}}
    for f in ("moe.py", "model.py", "config.py", "forge_io.py", "attn.py"):
        p = os.path.join(a.accel, "accel", f)
        if os.path.exists(p):
            report["inputs"][f"accel/{f}"] = sha_file(p)
    # ---- 1. dense champion
    log("load dense checkpoint", a.dense)
    state, dense, dense_sha = FIO.load_checkpoint(a.dense)
    if "moe" in state or "moe" in state.get("config", {}):
        raise SystemExit("--dense is already a MoE checkpoint; use tools/donor/insert_implant.py directly")
    base = ModelConfig.from_dict(state["config"]["model"])
    dense = FIO.check_weights(dense, base.param_specs(), "dense model") if hasattr(base, "param_specs") else dense
    report["inputs"]["dense"] = {"dir": os.path.abspath(a.dense), "model_sha256": dense_sha, "step": state.get("step"),
                                 "state_model_sha256": state.get("model_sha256"), "config": base.to_dict(),
                                 "parameters": int(sum(np.prod(v.shape) for v in dense.values()))}
    if state.get("model_sha256") and state["model_sha256"] != dense_sha:
        raise SystemExit(f"model.safetensors sha256 {dense_sha} differs from state.json model_sha256 {state['model_sha256']}")
    n_seg = base.ffn // a.expert_ffn if base.ffn % a.expert_ffn == 0 else None
    exact = n_seg is not None and n_seg - a.n_shared == a.top_k
    report["steps"]["upcycle"] = {"ffn": base.ffn, "expert_ffn": a.expert_ffn, "segments": n_seg, "n_shared": a.n_shared,
                                  "n_routed": a.n_routed, "top_k": a.top_k, "exact_by_construction": bool(exact)}
    if not exact:
        log(f"WARNING: (ffn/expert_ffn) - n_shared = {None if n_seg is None else n_seg - a.n_shared} != top_k {a.top_k}: the upcycle is not exact")
    ids = None
    if not a.no_forward_check:
        if a.check_ids:
            ids = np.load(a.check_ids).astype(np.int32)
            if ids.ndim == 1:
                ids = ids[None, :]
        else:
            B, T = (int(x) for x in a.check_shape.split(","))
            ids = np.random.default_rng(0).integers(0, base.vocab_size, (B, T)).astype(np.int32)
        if ids.shape[1] > base.max_seq_len or ids.max() >= base.vocab_size:
            raise SystemExit("check ids exceed max_seq_len or vocab_size")
        log(f"forward dense on ids {tuple(ids.shape)}")
        lg_dense = np.asarray(M.forward(dense, base, ids))
    # ---- 2./3. upcycle with the team's code, write the checkpoint (streamed)
    log("upcycle (team code)")
    cfg, params, buffers = MOE.upcycle(dense, base, a.n_routed, a.n_shared, a.top_k, a.expert_ffn, noise=0.0, seed=0)
    del dense
    gc.collect()
    up = os.path.join(a.out, "upcycled")
    os.makedirs(up)
    meta = {"format": "pt", "step": str(state.get("step", 0))}
    inv = st.write_safetensors_stream(os.path.join(up, "model.safetensors"), array_entries({**params, **buffers}), meta)
    st.write_safetensors_stream(os.path.join(up, "optim.safetensors"),
                                zero_entries(cfg.param_specs()) + zero_entries(cfg.param_specs(), "v."), meta)
    st_up = copy.deepcopy(state)
    st_up["moe"] = cfg.to_dict()
    st_up["model_sha256"] = inv["__file__"]["sha256"]
    st_up["parent_sha256"] = dense_sha
    with open(os.path.join(up, "state.json"), "w") as f:
        f.write(FIO.dumps(st_up, pretty=True))
    report["steps"]["upcycle"].update(model_sha256=inv["__file__"]["sha256"], parameters=cfg.num_params(),
                                      active_parameters=cfg.active_params(), moe=cfg.to_dict())
    del params, buffers
    gc.collect()
    # ---- 4. implants
    log("insert implants")
    imp = os.path.join(a.out, "implanted")
    cmd = [sys.executable, os.path.join(HERE, "insert_implant.py"), "--checkpoint", up, "--implant", *a.implant,
           "--layers", a.layers, "--off-bias", str(a.off_bias), "--out", imp]
    if a.seed_rms:
        cmd += ["--seed-rms", a.seed_rms]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit("insert_implant.py failed:\n" + r.stdout + r.stderr)
    st_imp = json.load(open(os.path.join(imp, "state.json")))
    with open(os.path.join(imp, "state.json"), "w") as f:              # the team's own serialisation
        f.write(FIO.dumps(st_imp, pretty=True))
    rep_imp = json.load(open(os.path.join(imp, "implant_report.json")))
    cfg_imp = MOE.model_from_state(st_imp)
    report["inputs"]["implants"] = {os.path.basename(p): sha_file(p) for p in a.implant}
    report["steps"]["implant"] = {k: rep_imp[k] for k in ("n_routed", "parameters_added", "filler_slots", "placement", "off_bias", "result")}
    report["steps"]["implant"]["parameters"] = cfg_imp.num_params()
    report["steps"]["implant"]["active_parameters"] = cfg_imp.active_params()
    # ---- 5. forward check, one model at a time
    if ids is not None:
        def run_moe(d):
            stt = json.load(open(os.path.join(d, "state.json")))
            c = MOE.model_from_state(stt)
            w, _, _ = FIO.read_safetensors(os.path.join(d, "model.safetensors"))
            p = FIO.check_weights(w, c.param_specs(), "moe model")
            b = FIO.check_weights(w, c.buffer_specs(), "moe buffers")
            names = set(p) | set(b)
            if set(w) != names:
                raise SystemExit(f"{d}: tensors not in the team's spec list: {sorted(set(w) - names)[:5]}")
            lg, loads, _ = MOE.forward(p, b, c, ids, dispatch="dense")
            return np.asarray(lg), np.asarray(loads), c

        def ce(lg):
            if ids.shape[1] < 2:
                return None
            x = lg[:, :-1].astype(np.float64)
            m = x.max(-1, keepdims=True)
            lse = (m[..., 0] + np.log(np.exp(x - m).sum(-1)))
            return float((lse - np.take_along_axis(x, ids[:, 1:, None].astype(np.int64), -1)[..., 0]).mean())

        log("forward upcycled")
        lg_up, load_up, c_up = run_moe(up)
        gc.collect()
        log("forward implanted")
        lg_imp, load_imp, c_imp = run_moe(imp)
        E0 = c_up.n_routed
        chk = {"ids": list(ids.shape), "ids_source": a.check_ids or "random (seed 0)",
               "max_abs_logit_diff_upcycled_vs_dense": float(np.abs(lg_up - lg_dense).max()),
               "max_abs_logit_diff_implanted_vs_upcycled": float(np.abs(lg_imp - lg_up).max()),
               "max_abs_logit": float(np.abs(lg_dense).max()),
               "old_slot_loads_equal": bool((load_up == load_imp[:, :E0]).all()),
               "new_slot_load": int(load_imp[:, E0:].sum()),
               "next_token_loss": {"dense": ce(lg_dense), "upcycled": ce(lg_up), "implanted": ce(lg_imp)}}
        tol = 1e-3 * max(1.0, chk["max_abs_logit"])
        chk["passed"] = bool(chk["max_abs_logit_diff_implanted_vs_upcycled"] <= tol and chk["new_slot_load"] == 0 and chk["old_slot_loads_equal"]
                             and (not exact or chk["max_abs_logit_diff_upcycled_vs_dense"] <= tol))
        chk["tolerance"] = tol
        report["steps"]["forward_check"] = chk
    else:
        report["steps"]["forward_check"] = "not run (--no-forward-check)"
    report["outputs"] = {"upcycled": {"dir": up, "model_sha256": sha_file(os.path.join(up, "model.safetensors"))},
                         "implanted": {"dir": imp, "model_sha256": sha_file(os.path.join(imp, "model.safetensors"))}}
    json.dump(report, open(os.path.join(a.out, "transplant_report.json"), "w"), indent=1)
    fc = report["steps"]["forward_check"]
    log("done:", json.dumps({"n_routed": report["steps"]["implant"]["n_routed"], "parameters": report["steps"]["implant"]["parameters"],
                             "forward_check": {k: fc[k] for k in ("max_abs_logit_diff_upcycled_vs_dense", "max_abs_logit_diff_implanted_vs_upcycled",
                                                                   "new_slot_load", "next_token_loss", "passed")} if isinstance(fc, dict) else fc}))
    if isinstance(fc, dict) and not fc["passed"]:
        raise SystemExit("FORWARD CHECK FAILED - see transplant_report.json")


if __name__ == "__main__":
    main()
