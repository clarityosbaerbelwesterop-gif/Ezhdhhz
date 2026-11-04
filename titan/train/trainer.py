"""Training runtime: builds forward + loss + backward + optimizer as ONE TITAN program and runs it per step.

Placement (reported in every run):
  TITAN   forward pass, softmax cross-entropy loss, backward pass, optimizer update
  HOST    dataset generation/shuffling, one-hot encoding, Adam's per-step lr scalar,
          evaluation argmax / accuracy / loss from TITAN-computed logits
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..compiler import CompileOptions, Graph, compile_graph, gradients
from ..config import load_config
from ..dtypes import DType
from ..emu.engine import FUNCTIONAL
from ..nn.layers import ParamStore
from ..nn.models import MLPClassifier, TinyTransformerLM, param_count
from ..provenance import provenance
from ..runtime import BACKENDS, Device, FastDevice
from .datasets import Sampler, eval_batches, load_dataset, load_file
from .optim import make_optimizer

PROFILES = {
    "mlp": {
        "tiny": {"hidden": [32], "lr": 0.01, "steps": 150, "batch": 32},
        "small": {"hidden": [64, 64], "lr": 0.01, "steps": 300, "batch": 64},
        "medium": {"hidden": [128, 128, 128], "lr": 0.005, "steps": 500, "batch": 128},
    },
    "transformer": {
        "tiny": {"seq": 8, "d_model": 16, "heads": 2, "d_ff": 32, "layers": 1, "lr": 0.01, "steps": 150, "batch": 8},
        "small": {"seq": 16, "d_model": 32, "heads": 2, "d_ff": 64, "layers": 2, "lr": 0.005, "steps": 300,
                  "batch": 8},
        "medium": {"seq": 32, "d_model": 64, "heads": 4, "d_ff": 128, "layers": 2, "lr": 0.003, "steps": 500,
                   "batch": 8},
    },
}
DEFAULT_DATASET = {"mlp": "spirals", "transformer": "text"}
PRECISION = {"fp32": DType.FP32, "bf16": DType.BF16, "fp16": DType.FP16}
PLACEMENT = {
    "titan": ["forward pass", "softmax cross-entropy loss", "backward pass (autodiff)", "optimizer update"],
    "host": ["dataset generation and shuffling", "one-hot encoding", "Adam bias-corrected lr scalar (1 value/step)",
             "evaluation argmax, accuracy and loss from TITAN logits", "checkpoint serialization"],
}


@dataclass
class TrainConfig:
    model: str | None = None          # mlp | transformer (default: transformer for text data, else mlp)
    dataset: str | None = None
    profile: str = "small"
    batch: int | None = None
    precision: str = "fp32"
    backend: str = "functional"
    steps: int | None = None
    lr: float | None = None
    optimizer: str = "adam"
    seed: int = 0
    eval_every: int = 50
    ckpt_every: int = 100
    arch: str = "dev"
    out_dir: str | None = None
    overrides: dict = field(default_factory=dict)
    data: str | None = None           # path to your own data (.csv/.tsv/.npz/.npy/.txt)
    target: str | None = None         # target column (name or index) for tabular data
    task: str = "auto"                # auto | classification | regression
    warmup: int | None = None         # Adam learning-rate warmup steps (default: transformers 10% of steps)

    def resolve(self) -> "TrainConfig":
        if self.data:
            text = Path(self.data).suffix.lower() in (".txt", ".text", ".md")
            self.model = self.model or ("transformer" if text else "mlp")
            if (self.model == "transformer") != text:
                raise ValueError("text files (.txt) train the transformer; tabular files (.csv/.npz/.npy) the mlp")
            self.dataset = Path(self.data).stem
        self.model = self.model or "mlp"
        if self.task not in ("auto", "classification", "regression"):
            raise ValueError(f"task must be auto, classification or regression, not {self.task!r}")
        if self.model not in PROFILES:
            raise ValueError(f"unknown model {self.model!r}; choose from {sorted(PROFILES)}")
        if self.profile not in PROFILES[self.model]:
            raise ValueError(f"unknown profile {self.profile!r}; choose from {sorted(PROFILES[self.model])}")
        if self.precision not in PRECISION:
            raise ValueError(f"precision {self.precision!r} not supported for training; choose from "
                             f"{sorted(PRECISION)} (INT8/INT4 are inference-only)")
        p = dict(PROFILES[self.model][self.profile])
        p.update(self.overrides)
        if self.data and self.model == "transformer":
            p.setdefault("seq", PROFILES["transformer"][self.profile]["seq"])
        self.dataset = self.dataset or DEFAULT_DATASET[self.model]
        self.batch = self.batch or p.pop("batch")
        p.pop("batch", None)
        self.steps = self.steps if self.steps is not None else p.pop("steps")
        p.pop("steps", None)
        if self.lr is None and self.model == "transformer":
            d = p.get("d_model", 32)
            if d > 128:                            # larger transformers need smaller steps (standard practice)
                p["lr"] = 1e-3 if d <= 512 else 3e-4
        self.lr = self.lr or p.pop("lr")
        p.pop("lr", None)
        self.overrides = p
        if self.warmup is None:
            self.warmup = max(10, min(200, self.steps // 10)) if self.model == "transformer" else 0
        if self.out_dir is None:
            self.out_dir = f"runs/{self.model}-{self.dataset}-{self.profile}-{self.precision}-{self.backend}-s{self.seed}"
        return self


def check_backend(backend: str, workload: str = "training") -> None:
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; choose from {sorted(BACKENDS)}")
    ok, why = BACKENDS[backend].status()
    if not ok:
        raise RuntimeError(f"backend {BACKENDS[backend].label!r} is not operational: {why}")
    if backend == "rtl" and workload == "training":
        raise RuntimeError("RTL backend supports only INT8 MATMUL offload (inference); training uses FP32/BF16 "
                           "matrix products that have no RTL implementation yet")


class Trainer:
    def __init__(self, tc: TrainConfig, log=print):
        self.tc = tc.resolve()
        check_backend(tc.backend)
        self.log_fn = log
        self.cfg = load_config(tc.arch)
        if tc.data:
            self.ds = load_file(tc.data, tc.target, tc.task, tc.seed,
                                **({"seq": tc.overrides["seq"]} if tc.model == "transformer" else {}))
        else:
            lm = tc.dataset == "text"
            if (tc.model == "transformer") != lm:
                raise ValueError(f"model {tc.model!r} is incompatible with dataset {tc.dataset!r} "
                                 "(transformer <-> text; mlp <-> classification datasets)")
            self.ds = load_dataset(tc.dataset, tc.seed, **({"seq": tc.overrides["seq"]} if lm else {}))
        if tc.model == "mlp":
            if self.ds.kind not in ("classification", "regression"):
                raise ValueError("model 'mlp' needs a classification or regression dataset")
            self.model = MLPClassifier(self.ds.in_dim, self.ds.outputs, list(tc.overrides["hidden"]))
        else:
            if self.ds.kind != "lm":
                raise ValueError("model 'transformer' needs the 'text' dataset")
            o = tc.overrides
            self.model = TinyTransformerLM(self.ds.classes, o["seq"], o["d_model"], o["heads"], o["d_ff"], o["layers"])
        self.compute = PRECISION[tc.precision]
        self.opt = make_optimizer(tc.optimizer, tc.lr, tc.warmup)
        t0 = time.perf_counter()
        self.train_exe, self.n_params = self._build_train()
        self.eval_batch = tc.batch
        self.eval_exe = self._build_eval()
        self.compile_s = time.perf_counter() - t0
        self.dev = FastDevice(self.cfg) if tc.backend == "fast" else Device(self.cfg)
        self.dev.load(self.train_exe)
        self.dev.load(self.eval_exe, init=False)
        self._release_host_init()
        self.sampler = Sampler(self.ds, tc.batch, tc.seed + 1)
        self.step = 0
        self.history: list[dict] = []
        self.out = Path(tc.out_dir)
        (self.out / "checkpoints").mkdir(parents=True, exist_ok=True)
        self._logf = open(self.out / "log.txt", "a")
        self._metf = open(self.out / "metrics.jsonl", "a")
        self._write_config()

    # ------------------------------------------------------------ build
    def _build_train(self):
        g = Graph(f"{self.model.name}-train")
        ps = ParamStore(g, self.tc.seed)
        _, logits = self.model.build(g, ps, self.tc.batch, self.compute)
        y = g.input("y", logits.shape)
        if self.ds.kind == "regression":
            loss = g.mean(g.square(g.sub(logits, y)), 2)          # mean squared error, on TITAN
        else:
            loss = g.xent(logits, y)
        params = list(ps.params.values())
        self.opt.apply(g, params, gradients(g, loss, params))
        g.output("loss", loss)
        self.model_params = [p.name for p in params]
        return compile_graph(g, self.cfg, self._opts()), param_count(ps)

    def _opts(self) -> CompileOptions:
        return CompileOptions(emit_isa=self.tc.backend != "fast")

    def _build_eval(self):
        g = Graph(f"{self.model.name}-eval")
        ps = ParamStore(g, self.tc.seed)
        _, logits = self.model.build(g, ps, self.eval_batch, self.compute)
        g.output("logits", logits)
        return compile_graph(g, self.cfg, self._opts(), share_with=self.train_exe)

    def _release_host_init(self) -> None:
        """Initial values now live on the device; drop the host copies (matters for large models)."""
        for exe in (self.train_exe, self.eval_exe):
            for K in exe.module.kernels:
                for t in K.in_tensors:
                    if t.kind == "param":
                        t.value = None
            exe.init = {k: v for k, v in exe.init.items() if k not in exe.params}

    def _write_config(self):
        info = {"train_config": asdict(self.tc), "arch": self.cfg.to_dict(), "arch_hash": self.cfg.config_hash(),
                "model": self.model.describe(), "parameters": self.n_params, "dataset": self.ds.describe(),
                "optimizer": self.opt.describe(), "backend": BACKENDS[self.tc.backend].label,
                "backend_kind": BACKENDS[self.tc.backend].kind, "placement": PLACEMENT,
                "train_program": self.train_exe.stats, "eval_program": self.eval_exe.stats,
                "compile_s": round(self.compile_s, 4), "provenance": provenance()}
        (self.out / "config.json").write_text(json.dumps(info, indent=2, default=str))

    def log(self, msg: str) -> None:
        self._logf.write(msg + "\n")
        self._logf.flush()
        if self.log_fn:
            self.log_fn(msg)

    # ------------------------------------------------------------ run
    def train_step(self) -> dict:
        feeds = self.sampler.next()
        feeds.update(self.opt.host_feeds(self.step))
        mode = {"rtl": FUNCTIONAL, "fast": "fast"}.get(self.tc.backend, self.tc.backend)
        out = self.dev.run(self.train_exe, feeds, mode, fetch=["loss"])
        r = out.result
        rec = {"step": self.step, "loss": float(out.outputs["loss"][0, 0]), "titan_instructions": r.instructions,
               "titan_ops": r.ops, "host_execute_s": round(out.timings["execute_s"], 6),
               "host_io_s": round(out.host_io_s, 6)}
        if r.mode == "fast":
            rec["titan_kernels"] = r.counts.get("kernels", 0)
        if r.cycles is not None:
            rec["simulated_cycles"] = r.cycles
            rec["simulated_s"] = r.simulated_seconds
        if not np.isfinite(rec["loss"]):
            raise FloatingPointError(f"non-finite loss at step {self.step}")
        self.step += 1
        self.history.append(rec)
        self._metf.write(json.dumps(rec) + "\n")
        self._metf.flush()
        return rec

    def evaluate(self) -> dict:
        preds, targets = [], []
        for xb, yb, nv in eval_batches(self.ds, self.eval_batch):
            ev_mode = "fast" if self.tc.backend == "fast" else FUNCTIONAL
            out = self.dev.run(self.eval_exe, {"x": xb}, ev_mode).outputs["logits"].astype(np.float64)
            preds.append(out[:nv])
            targets.append(np.asarray(yb)[:nv])
        if not preds:
            return {}
        P, Y = np.vstack(preds), np.concatenate(targets)
        res = {"eval_examples": int(len(Y)), "eval_mode": "functional (TITAN outputs, host metrics)"}
        if self.ds.kind == "regression":
            err = P[:, 0] - Y
            mse = float((err ** 2).mean())
            res.update({"eval_loss": mse, "eval_rmse": float(np.sqrt(mse) * self.ds.y_std),
                        "eval_r2": float(1 - mse / max(float(Y.var()), 1e-12))})
            return res
        z = P - P.max(1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(1, keepdims=True))
        res.update({"eval_loss": float(-logp[np.arange(len(Y)), Y].mean()),
                    "eval_accuracy": float((P.argmax(1) == Y).mean())})
        return res

    def save_checkpoint(self) -> Path:
        arrays = {n: self.dev.read(self.train_exe, n) for n in self.train_exe.params}
        path = self.out / "checkpoints" / f"step_{self.step:06d}.npz"
        np.savez(path, **arrays)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        meta = {"step": self.step, "sha256": digest, "arch_hash": self.cfg.config_hash(),
                "params": sorted(arrays), "sampler": self.sampler.state(), "train_config": asdict(self.tc),
                "provenance": provenance()}
        path.with_suffix(".json").write_text(json.dumps(meta, default=str))
        return path

    def load_checkpoint(self, path) -> None:
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
            raise ValueError(f"checkpoint {path} is corrupt (sha256 mismatch)")
        if meta["arch_hash"] != self.cfg.config_hash():
            raise ValueError("checkpoint was produced for a different architecture configuration")
        data = np.load(path)
        if sorted(data.files) != sorted(self.train_exe.params):
            raise ValueError("checkpoint parameters do not match the model")
        for n in data.files:
            self.dev.write(self.train_exe, n, data[n])
        self.sampler.restore(meta["sampler"])
        self.step = meta["step"]
        self.log(f"resumed from {path} at step {self.step}")

    def run(self, steps: int | None = None) -> dict:
        target = steps if steps is not None else self.tc.steps
        t0 = time.perf_counter()
        first = None
        self.log(f"TITAN training | model={self.model.name} params={self.n_params} dataset={self.ds.name} "
                 f"precision={self.tc.precision} backend={BACKENDS[self.tc.backend].label} arch={self.cfg.name}")
        self.log(f"train program: {self.train_exe.stats['instructions']} instructions, "
                 f"{self.train_exe.stats['kernels']} kernels, {self.train_exe.stats['barriers']} barriers")
        while self.step < target:
            rec = self.train_step()
            first = first if first is not None else rec["loss"]
            if self.step % max(1, self.tc.eval_every) == 0 or self.step == target:
                ev = self.evaluate()
                rec.update(ev)
                msg = f"step {self.step:5d} loss {rec['loss']:.4f}"
                if ev:
                    msg += f" | eval loss {ev['eval_loss']:.4f}"
                    msg += (f" acc {ev['eval_accuracy']:.3f}" if "eval_accuracy" in ev else
                            f" rmse {ev['eval_rmse']:.4g} r2 {ev['eval_r2']:.3f}")
                if "simulated_cycles" in rec:
                    msg += f" | {rec['simulated_cycles']:.0f} simulated cycles/step"
                self.log(msg)
            if self.tc.ckpt_every and (self.step % self.tc.ckpt_every == 0 or self.step == target):
                self.save_checkpoint()
        wall = time.perf_counter() - t0
        tail = [h["loss"] for h in self.history[-10:]]
        summary = {"steps": self.step, "initial_loss": first if first is not None else
                   (self.history[0]["loss"] if self.history else None),
                   "final_loss_avg10": float(np.mean(tail)) if tail else None, **self.evaluate(),
                   "host_wall_s": round(wall, 3), "parameters": self.n_params,
                   **({"titan_kernels_per_step": self.history[-1]["titan_kernels"]} if self.history and
                      "titan_kernels" in self.history[-1] else
                      {"titan_instructions_per_step": self.history[-1]["titan_instructions"] if self.history else 0}),
                   "backend": BACKENDS[self.tc.backend].label, "placement": PLACEMENT,
                   "provenance": provenance()}
        sims = [h["simulated_cycles"] for h in self.history if "simulated_cycles" in h]
        if sims:
            summary["simulated_cycles_per_step"] = float(np.mean(sims))
            summary["simulated_step_s_at_clock"] = float(np.mean(sims)) / (self.cfg.clock_ghz * 1e9)
        (self.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        return summary

    def close(self) -> None:
        self._logf.close()
        self._metf.close()
