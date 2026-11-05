"""Training runtime: builds forward + loss + backward + optimizer as ONE TITAN program and runs it per step.

Placement (reported in every run):
  TITAN   forward pass, softmax cross-entropy loss, backward pass, optimizer update
  HOST    dataset generation/shuffling, classifier one-hot targets, Adam's per-step lr scalar,
          evaluation argmax / accuracy / loss from TITAN-computed logits
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

from ..compiler import CompileOptions, Graph, compile_graph, gradients
from ..config import load_config
from ..dtypes import DType
from ..emu.engine import FUNCTIONAL
from ..nn.layers import ParamStore
from ..nn.models import MLPClassifier, param_count
from ..nn.transformer import TransformerConfig, TransformerLM, load_model_plugin, read_model_config
from ..provenance import provenance
from ..runtime import BACKENDS, Device, FastDevice
from .datasets import Sampler, eval_batches, load_dataset, load_file
from .tokenizers import load_tokenizer
from .tokens import load_tokens, text_as_bytes, text_with_tokenizer
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
    "host": ["dataset generation and shuffling", "classifier one-hot targets (LM targets: on TITAN)", "Adam bias-corrected lr scalar (1 value/step)",
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
    # ---- pretrain / continue / fine-tune
    init_from: str | None = None      # fine-tune: start from a checkpoint's weights (fresh optimizer, new data)
    freeze: list = field(default_factory=list)   # fine-tune: parameter name prefixes / globs kept fixed
    tokenizer: str = "char"           # .txt data: char | byte | BPE / tokenizer.json / .model file | file.py:Factory
    # ---- bring your own architecture
    model_config: str | None = None   # JSON/TOML TransformerConfig (norm, activation, position, kv_heads, ...)
    model_file: str | None = None     # your own model: file.py:factory or package.module:factory
    # ---- quality controls (defaults: transformers AdamW + cosine + clipping; MLPs plain Adam)
    schedule: str | None = None       # constant | cosine
    min_lr_ratio: float = 0.1
    weight_decay: float | None = None # decoupled (AdamW), applied to matrices only
    clip_grad: float | None = None    # global gradient-norm clipping, computed on TITAN
    early_stop: int = 0               # stop after N evaluations without improvement (0 = off)
    # ---- beyond RAM: weights, optimizer state and large temporaries on attached storage
    offload_dir: str | None = None    # memory-mapped storage for all large tensors (NVMe/SSD recommended)
    offload_mb: float = 4.0           # tensors of at least this size go to storage
    init_scheme: str = "auto"         # auto | xavier | gpt2 (auto: gpt2 for transformers with d_model >= 256)
    lora_rank: int = 0                # > 0: LoRA fine-tuning; base weights frozen (and regenerated if lazy)
    lora_alpha: float = 16.0

    def resolve(self) -> "TrainConfig":
        if self.data:
            text = Path(self.data).suffix.lower() in (".txt", ".text", ".md", ".bin")
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
        if (self.model_config or self.model_file) and self.model != "transformer":
            raise ValueError("--model-config / --model-file define language models (transformer)")
        p = dict(PROFILES[self.model][self.profile])
        if self.model_config:
            p.update(read_model_config(self.model_config))   # file overrides the profile, explicit overrides win
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
        lm = self.model == "transformer"
        if self.schedule is None:
            self.schedule = "cosine" if lm else "constant"
        if self.schedule not in ("constant", "cosine"):
            raise ValueError("schedule must be constant or cosine")
        if self.weight_decay is None:
            self.weight_decay = 0.1 if lm and self.optimizer == "adam" else 0.0
        if self.clip_grad is None and lm:
            self.clip_grad = 1.0
        if self.init_scheme == "auto":
            self.init_scheme = "gpt2" if lm and self.overrides.get("d_model", 0) >= 256 else "xavier"
        if self.init_scheme not in ("xavier", "gpt2"):
            raise ValueError("init_scheme must be auto, xavier or gpt2")
        if self.tokenizer != "char" and not lm:
            raise ValueError("a tokenizer applies to text data (transformer)")
        if self.offload_dir and self.backend not in ("fast",):
            self.backend = "fast"           # storage offload is implemented by the kernel-level NumPy executor
        if self.lora_rank and self.backend in ("functional", "cycle", "rtl"):
            self.backend = "auto"           # frozen base weights are regenerated by the IR-level executors
        if self.backend == "auto":
            self.backend = auto_backend(lm, self.overrides.get("d_model", 0))
        if self.out_dir is None:
            self.out_dir = f"runs/{self.model}-{self.dataset}-{self.profile}-{self.precision}-{self.backend}-s{self.seed}"
        return self


def auto_backend(lm: bool, d_model: int) -> str:
    """Fastest executor for this host (measured, docs/PERFORMANCE.md): PyTorch on a CUDA GPU, or on CPU for
    transformers with d_model >= 128 (compute-bound; multi-threaded kernels); otherwise the compiled NumPy plan
    (lowest per-kernel overhead for small models)."""
    from ..runtime.accel import default_device, torch_available
    if torch_available() and (default_device() == "cuda" or (lm and d_model >= 128)):
        return "accel"
    return "fast"


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
        suffix = Path(tc.data).suffix.lower() if tc.data else ""
        self.tokenizer = None
        if suffix == ".bin":
            self.ds = load_tokens(tc.data, tc.overrides["seq"])     # titan-tokens-v1 or byte corpus, memory-mapped
        elif tc.data and tc.model == "transformer" and tc.tokenizer == "byte":
            self.ds = text_as_bytes(tc.data, tc.overrides["seq"])
        elif tc.data and tc.model == "transformer" and tc.tokenizer != "char":
            self.tokenizer = load_tokenizer(tc.tokenizer)
            self.ds = text_with_tokenizer(tc.data, tc.overrides["seq"], self.tokenizer)
        elif tc.data:
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
            self.model = self._language_model()
        self.compute = PRECISION[tc.precision]
        self.opt = make_optimizer(tc.optimizer, tc.lr, tc.warmup, schedule=tc.schedule, total_steps=tc.steps,
                                  min_lr_ratio=tc.min_lr_ratio, weight_decay=tc.weight_decay)
        t0 = time.perf_counter()
        self.train_exe, self.n_params = self._build_train()
        self.eval_batch = tc.batch
        self.eval_exe = self._build_eval() if tc.eval_every else None
        self.compile_s = time.perf_counter() - t0
        if tc.backend == "accel":
            from ..runtime.accel import AccelDevice
            self.dev = AccelDevice(self.cfg)
        elif tc.backend == "fast":
            self.dev = FastDevice(self.cfg, tc.offload_dir, tc.offload_mb)
        else:
            self.dev = Device(self.cfg)
        # frozen lazily-initialized weights (LoRA base) are regenerated from their seeds, never stored
        self.regenerated = [n for n in self.frozen if n in self.train_exe.lazy_init] if tc.lora_rank else []
        if self.regenerated:
            self.dev.load(self.train_exe, regenerate=self.regenerated)
        else:
            self.dev.load(self.train_exe)
        if self.eval_exe is not None:
            self.dev.load(self.eval_exe, init=False)
        self._release_host_init()
        self.best = (float("inf"), -1)
        self.stopped_early = False
        self.sampler = Sampler(self.ds, tc.batch, tc.seed + 1)
        self.step = 0
        self.history: list[dict] = []
        self.out = Path(tc.out_dir)
        (self.out / "checkpoints").mkdir(parents=True, exist_ok=True)
        self._logf = open(self.out / "log.txt", "a")
        self._metf = open(self.out / "metrics.jsonl", "a")
        self.init_info = self.init_from_checkpoint(tc.init_from) if tc.init_from else None
        self._write_config()

    # ------------------------------------------------------------ build
    def _build_train(self):
        g = Graph(f"{self.model.name}-train")
        ps = self._store(g)
        _, logits = self.model.build(g, ps, self.tc.batch, self.compute)
        if self.ds.kind == "regression":
            y = g.input("y", logits.shape)
            loss = g.mean(g.square(g.sub(logits, y)), 2)          # mean squared error, on TITAN
        elif self.ds.kind == "lm":                                # next-token ids in, one-hot built on TITAN
            loss = g.xent(logits, g.onehot(g.input("y", (logits.rows, 1), DType.INT32), logits.cols))
        else:
            loss = g.xent(logits, g.input("y", logits.shape))
        params = list(ps.params.values())
        if self.tc.lora_rank:
            self.frozen = [p.name for p in params if ".lora_" not in p.name]
        else:
            self.frozen = [p.name for p in params if _frozen(p.name, self.tc.freeze)]
        trainable = [p for p in params if p.name not in self.frozen]
        if not trainable:
            raise ValueError("every parameter is frozen")
        grads = gradients(g, loss, trainable)
        if self.tc.clip_grad:
            sq = None
            for gr in grads:
                s_ = g.sum(g.square(gr), 2)
                sq = s_ if sq is None else g.add(sq, s_)
            norm = g.sqrt(sq)
            c = g.const(float(self.tc.clip_grad), name="clip_threshold")
            factor = g.div(c, g.maximum(norm, c))                    # min(1, clip / ||g||)
            grads = [g.mul(gr, factor) for gr in grads]
            g.output("grad_norm", norm)
        self.opt.apply(g, trainable, grads)
        g.output("loss", loss)
        self.model_params = [p.name for p in params]
        return compile_graph(g, self.cfg, self._opts()), param_count(ps)

    def _language_model(self):
        """TransformerLM from profile <- --model-config <- overrides, or your own model plugin."""
        tc, ds = self.tc, self.ds
        keys = {f.name for f in fields(TransformerConfig)}
        arch = dict(tc.overrides)
        vocab = int(arch.pop("vocab", 0) or 0)
        if vocab and vocab < ds.classes:
            raise ValueError(f"model vocab {vocab} is smaller than the tokenizer's {ds.classes} tokens")
        vocab = vocab or ds.classes                               # larger vocab = padding (e.g. multiple of 64)
        if tc.model_file:
            return load_model_plugin(tc.model_file, vocab, arch.pop("seq"), **arch)
        unknown = set(arch) - keys
        if unknown:
            raise ValueError(f"unknown transformer settings {sorted(unknown)}; valid: {sorted(keys)}")
        return TransformerLM(TransformerConfig(vocab=vocab, **arch))

    def _store(self, g: Graph) -> ParamStore:
        lazy = bool(self.tc.offload_dir) or bool(self.tc.lora_rank)
        ps = ParamStore(g, self.tc.seed, lazy=lazy)
        ps.scheme = self.tc.init_scheme
        ps.lora_rank, ps.lora_alpha = self.tc.lora_rank, self.tc.lora_alpha
        ps.fast_init = bool(self.tc.lora_rank)
        return ps

    def _opts(self) -> CompileOptions:
        return CompileOptions(emit_isa=self.tc.backend not in ("fast", "accel"))

    def _build_eval(self):
        g = Graph(f"{self.model.name}-eval")
        ps = self._store(g)
        _, logits = self.model.build(g, ps, self.eval_batch, self.compute)
        g.output("logits", logits)
        return compile_graph(g, self.cfg, self._opts(), share_with=self.train_exe)

    def _release_host_init(self) -> None:
        """Initial values now live on the device; drop the host copies (matters for large models)."""
        for exe in (e for e in (self.train_exe, self.eval_exe) if e is not None):
            for K in exe.module.kernels:
                for t in K.in_tensors:
                    if t.kind == "param":
                        t.value = None
            exe.init = {k: v for k, v in exe.init.items() if k not in exe.params}

    def _write_config(self):
        info = {"train_config": asdict(self.tc), "arch": self.cfg.to_dict(), "arch_hash": self.cfg.config_hash(),
                "model": _describe(self.model), "parameters": self.n_params, "dataset": self.ds.describe(),
                "tokenizer": self.tokenizer.describe() if self.tokenizer else getattr(self.ds, "tokenizer", None)
                or (self.tc.tokenizer if self.ds.kind == "lm" else None),
                "optimizer": self.opt.describe(), "backend": BACKENDS[self.tc.backend].label,
                "backend_kind": BACKENDS[self.tc.backend].kind, "placement": PLACEMENT,
                "train_program": self.train_exe.stats,
                "eval_program": self.eval_exe.stats if self.eval_exe is not None else None,
                "lora": {"rank": self.tc.lora_rank, "alpha": self.tc.lora_alpha,
                         "trainable_parameters": int(sum(np.prod(self.train_exe.buffers[n].shape)
                                                         for n in self.model_params if n not in self.frozen)),
                         "regenerated_frozen_tensors": len(self.regenerated)} if self.tc.lora_rank else None,
                "compile_s": round(self.compile_s, 4), "frozen": self.frozen, "initialized_from": self.init_info,
                "offload": {"dir": self.tc.offload_dir, "threshold_mb": self.tc.offload_mb,
                            "bytes_on_storage": getattr(self.dev, "offloaded_bytes", 0)} if self.tc.offload_dir else None,
                "quality": {"init": self.tc.init_scheme, "schedule": self.tc.schedule, "warmup": self.tc.warmup, "weight_decay": self.tc.weight_decay,
                            "clip_grad": self.tc.clip_grad, "early_stop": self.tc.early_stop},
                "provenance": provenance()}
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
        mode = {"rtl": FUNCTIONAL}.get(self.tc.backend, self.tc.backend)
        fetch = ["loss"] + (["grad_norm"] if "grad_norm" in self.train_exe.outputs else [])
        out = self.dev.run(self.train_exe, feeds, mode, fetch=fetch)
        r = out.result
        rec = {"step": self.step, "loss": float(out.outputs["loss"][0, 0]), "titan_instructions": r.instructions,
               "titan_ops": r.ops, "host_execute_s": round(out.timings["execute_s"], 6),
               "host_io_s": round(out.host_io_s, 6)}
        if "grad_norm" in out.outputs:
            rec["grad_norm"] = float(out.outputs["grad_norm"][0, 0])
        if r.mode in ("fast", "accel"):
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
        if self.eval_exe is None:
            return {}
        preds, targets = [], []
        for xb, yb, nv in eval_batches(self.ds, self.eval_batch):
            ev_mode = self.tc.backend if self.tc.backend in ("fast", "accel") else FUNCTIONAL
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

    def save_checkpoint(self, name: str | None = None) -> Path:
        get = getattr(self.dev, "view", self.dev.read)          # no RAM copies for storage-backed tensors
        arrays = {n: get(self.train_exe, n) for n in self.train_exe.params if n not in self.regenerated}
        path = self.out / "checkpoints" / f"{name or f'step_{self.step:06d}'}.npz"
        np.savez(path, **arrays)
        digest = _sha256_file(path)
        meta = {"step": self.step, "sha256": digest, "arch_hash": self.cfg.config_hash(),
                "params": sorted(arrays), "sampler": self.sampler.state(), "train_config": asdict(self.tc),
                "provenance": provenance()}
        path.with_suffix(".json").write_text(json.dumps(meta, default=str))
        return path

    def load_checkpoint(self, path) -> None:
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        if _sha256_file(path) != meta["sha256"]:
            raise ValueError(f"checkpoint {path} is corrupt (sha256 mismatch)")
        if meta["arch_hash"] != self.cfg.config_hash():
            raise ValueError("checkpoint was produced for a different architecture configuration")
        data = np.load(path)
        if sorted(data.files) != sorted(n for n in self.train_exe.params if n not in self.regenerated):
            raise ValueError("checkpoint parameters do not match the model")
        for n in data.files:
            self.dev.write(self.train_exe, n, data[n])
        self.sampler.restore(meta["sampler"])
        self.step = meta["step"]
        self.log(f"resumed from {path} at step {self.step}")

    def init_from_checkpoint(self, path) -> dict:
        """Fine-tuning start: copy every model weight whose name and shape match; report the rest."""
        path = Path(path)
        meta_p = path.with_suffix(".json")
        digest = _sha256_file(path)
        if meta_p.exists() and json.loads(meta_p.read_text()).get("sha256") not in (None, digest):
            raise ValueError(f"checkpoint {path} is corrupt (sha256 mismatch)")
        data = np.load(path)
        loaded, skipped = 0, []
        for name in self.model_params:
            if name in data.files and tuple(data[name].shape) == tuple(self.train_exe.buffers[name].shape):
                self.dev.write(self.train_exe, name, data[name])
                loaded += 1
            else:
                skipped.append(name)
        self.log(f"initialized {loaded} weights from {path}; {len(skipped)} re-initialized (name/shape differ)")
        return {"path": str(path), "sha256": digest, "loaded": loaded, "skipped": skipped}

    def run(self, steps: int | None = None) -> dict:
        target = steps if steps is not None else self.tc.steps
        t0 = time.perf_counter()
        first = None
        self.log(f"TITAN training | model={self.model.name} params={self.n_params} dataset={self.ds.name} "
                 f"precision={self.tc.precision} backend={BACKENDS[self.tc.backend].label} arch={self.cfg.name}")
        self.log(f"train program: {self.train_exe.stats['instructions']} instructions, "
                 f"{self.train_exe.stats['kernels']} kernels, {self.train_exe.stats['barriers']} barriers")
        stall = 0
        while self.step < target:
            rec = self.train_step()
            first = first if first is not None else rec["loss"]
            if self.tc.eval_every and (self.step % self.tc.eval_every == 0 or self.step == target):
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
                if ev and ev["eval_loss"] < self.best[0]:
                    self.best = (ev["eval_loss"], self.step)
                    stall = 0
                    if self.tc.ckpt_every:
                        self.save_checkpoint("best")
                elif ev:
                    stall += 1
                    if self.tc.early_stop and stall >= self.tc.early_stop:
                        self.stopped_early = True
                        self.log(f"early stop at step {self.step}: no improvement in {stall} evaluations "
                                 f"(best {self.best[0]:.4f} at step {self.best[1]})")
                        if self.tc.ckpt_every:
                            self.save_checkpoint()
                        break
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
                   "best_eval_loss": self.best[0] if self.best[1] >= 0 else None, "best_step": self.best[1],
                   "stopped_early": self.stopped_early, "provenance": provenance()}
        sims = [h["simulated_cycles"] for h in self.history if "simulated_cycles" in h]
        if sims:
            summary["simulated_cycles_per_step"] = float(np.mean(sims))
            summary["simulated_step_s_at_clock"] = float(np.mean(sims)) / (self.cfg.clock_ghz * 1e9)
        (self.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        return summary

    def close(self) -> None:
        self._logf.close()
        self._metf.close()


def _describe(model) -> dict:
    d = getattr(model, "describe", None)
    return d() if callable(d) else {"kind": getattr(model, "kind", "lm"), "class": type(model).__name__,
                                    "name": getattr(model, "name", "")}


def _frozen(name: str, patterns) -> bool:
    import fnmatch
    return any(name == p or name.startswith(p + ".") or fnmatch.fnmatch(name, p) for p in patterns)


def _sha256_file(path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)
