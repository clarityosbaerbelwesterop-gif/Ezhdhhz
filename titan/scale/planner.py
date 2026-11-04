"""Distributed training planner for large transformers (up to and beyond 100B parameters).

Scope and honesty
-----------------
This is Mode C (analytical). It plans how a model is split over many accelerator
packages and estimates memory per device and time per training step. It does not
execute the model. All link parameters and the GEMM efficiency are assumptions
and are reported with every plan. The same model is applied to any device spec,
so TITAN-X1 and B300 can be compared like-for-like *within this model*.

Model accounting (validated against compiled TITAN graphs in tests/test_scale.py):
  * parameters and forward matmul FLOPs of the transformer used by titan.nn
    (pre-LN blocks, per-head projections == fused QKV/O, GELU MLP, learned positions,
    untied LM head); token/position embeddings are gathers in the plan
  * training = forward + backward (2x forward) [+ forward again with full recomputation]
  * mixed precision Adam: bf16 weights + bf16 grads + fp32 master/m/v = 16 bytes/param
  * activation memory per layer (bf16, sequence-parallel tensor parallelism):
    t*d*(34 + 5*a*T/d)/TP bytes (Korthikanti et al., 2022); full recomputation keeps 2*t*d
"""
from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass

from ..dtypes import DType

GB = 1e9


@dataclass(frozen=True)
class TransformerSpec:
    name: str
    layers: int
    d_model: int
    heads: int
    d_ff: int
    vocab: int
    seq: int

    def layer_params(self) -> int:
        d, f = self.d_model, self.d_ff
        return 4 * d * d + d + 4 * d + d * f + f + f * d + d

    def params(self) -> int:
        d, V = self.d_model, self.vocab
        return self.layers * self.layer_params() + V * d + self.seq * d + 2 * d + d * V + V

    def fwd_flops_per_seq(self) -> int:
        """Matmul FLOPs of one forward pass over one sequence (embeddings as gathers)."""
        T, d, f = self.seq, self.d_model, self.d_ff
        per_layer = 2 * T * (4 * d * d + 2 * d * f) + 2 * 2 * T * T * d
        return self.layers * per_layer + 2 * T * d * self.vocab

    def onehot_embedding_flops_per_seq(self) -> int:
        """Extra FLOPs when embeddings are one-hot matmuls (as in the small titan.nn graphs)."""
        return 2 * self.seq * self.vocab * self.d_model + 2 * self.seq * self.seq * self.d_model

    def describe(self) -> dict:
        return {**asdict(self), "params": self.params(), "fwd_gflops_per_token": self.fwd_flops_per_seq() / self.seq / 1e9}


PRESETS = {
    "gpt-125m": (12, 768, 12), "gpt-350m": (24, 1024, 16), "gpt-1.3b": (24, 2048, 16), "gpt-2.7b": (32, 2560, 32),
    "gpt-6.7b": (32, 4096, 32), "gpt-13b": (40, 5120, 40), "gpt-30b": (48, 7168, 56), "gpt-70b": (80, 8192, 64),
    "gpt-100b": (80, 10240, 80), "gpt-175b": (96, 12288, 96),
}


def preset(name: str, vocab: int = 50304, seq: int = 2048) -> TransformerSpec:
    L, d, h = PRESETS[name]
    return TransformerSpec(name, L, d, h, 4 * d, vocab, seq)


def parse_count(s: str | int | float) -> int:
    if isinstance(s, (int, float)):
        return int(s)
    t = str(s).strip().lower().replace("_", "")
    mult = {"k": 1e3, "m": 1e6, "b": 1e9, "g": 1e9, "t": 1e12}
    return int(float(t[:-1]) * mult[t[-1]]) if t[-1] in mult else int(float(t))


def spec_for_params(n, vocab: int = 50304, seq: int = 2048) -> TransformerSpec:
    """GPT-style shape with ~n parameters.

    A standard preset within 10 % of n is used as is; otherwise layers are chosen even (so that
    pipeline parallelism can split them) with d_model ~ 128 * layers and head dimension 128.
    """
    n = parse_count(n)
    for name in PRESETS:
        sp = preset(name, vocab, seq)
        if abs(sp.params() - n) <= 0.1 * n:
            return sp
    best = None
    for L in range(2, 161, 2):
        for d in range(128, 32768 + 1, 128):
            s = TransformerSpec(f"auto-{n / 1e9:.3g}b", L, d, max(1, d // 128), 4 * d, vocab, seq)
            err = abs(s.params() - n) / n + 0.05 * abs(math.log((d / L) / 128))
            if best is None or err < best[0]:
                best = (err, s)
            if s.params() > 2 * n:
                break
    return best[1]


@dataclass(frozen=True)
class DeviceSpec:
    name: str
    peak_tflops: float      # dense BF16
    hbm_tbps: float
    hbm_gb: float
    note: str = ""

    @classmethod
    def from_arch(cls, cfg) -> "DeviceSpec":
        return cls(cfg.name, cfg.peak_ops(DType.BF16) / 1e12, cfg.hbm_gbps / 1e3, cfg.global_mem_gib * 1.073741824,
                   "TITAN-X1 configuration (analytical, unbuilt)" if "target" in cfg.name else cfg.description)

    @classmethod
    def b300(cls) -> "DeviceSpec":
        from ..perf.reference import B300
        return cls("NVIDIA B300", B300["dense_tflops"]["BF16"], max(B300["hbm_bandwidth_tbps"]),
                   B300["hbm_capacity_gb"], "vendor-class figures, titan/perf/reference.py")


@dataclass(frozen=True)
class Links:
    tp_gbps: float = 900.0       # ASSUMED per-device bandwidth inside a tensor-parallel group
    tp_latency_us: float = 1.0
    dp_gbps: float = 400.0       # ASSUMED per-device bandwidth across data/pipeline-parallel groups
    dp_latency_us: float = 5.0
    gemm_efficiency: float = 0.8 # ASSUMED achievable fraction of peak for large GEMMs (both devices)
    dp_overlap: float = 0.0      # fraction of the gradient sync hidden behind backward (0 = conservative)
    memory_reserve: float = 0.9  # usable fraction of device memory


def _allreduce(nbytes: float, n: int, gbps: float, lat_us: float) -> float:
    if n <= 1:
        return 0.0
    return 2 * (n - 1) / n * nbytes / (gbps * GB) + 2 * (n - 1) * lat_us * 1e-6


def _gemm(m, n, k, dev: DeviceSpec, eff: float) -> float:
    flops = 2.0 * m * n * k
    traffic = 2.0 * (m * k + k * n + m * n)
    return max(flops / (dev.peak_tflops * 1e12 * eff), traffic / (dev.hbm_tbps * 1e12))


def evaluate_plan(spec: TransformerSpec, dev: DeviceSpec, devices: int, global_batch: int, tp: int, pp: int,
                  micro: int, zero: int, recompute: bool, links: Links = Links()) -> dict:
    dp = devices // (tp * pp)
    T, d, f, V, L, a = spec.seq, spec.d_model, spec.d_ff, spec.vocab, spec.layers, spec.heads
    t = micro * T                                   # tokens per micro-batch
    m = global_batch // (dp * micro)                # micro-batches per step per pipeline
    bw = dev.hbm_tbps * 1e12
    eff = links.gemm_efficiency
    # ---- compute per layer per micro-batch on one device
    gem = [(t, 3 * d // tp, d), (t, d, d // tp), (t, f // tp, d), (t, d, f // tp)]
    t_proj = sum(_gemm(mm, nn, kk, dev, eff) for mm, nn, kk in gem)
    att_flops = 2 * 2 * micro * T * T * d / tp
    att_bytes = 2 * 3 * micro * (a // tp) * T * T          # scores, probabilities, grads (bf16)
    t_att = max(att_flops / (dev.peak_tflops * 1e12 * eff), att_bytes / bw)
    ew_bytes = 2 * t * d * (34 + 5 * a * T / d) / tp        # element-wise traffic ~ activation footprint
    t_fwd = t_proj + t_att + ew_bytes / bw
    t_layer = t_fwd * (3 + (1 if recompute else 0))         # fwd + bwd(2x) [+ recompute]
    act_bytes = t * d * 2
    t_tp = 4 * _allreduce(act_bytes, tp, links.tp_gbps, links.tp_latency_us)
    t_stage = (L // pp) * (t_layer + t_tp)
    t_head = 3 * _gemm(t, V // tp, d, dev, eff)
    t_p2p = 2 * (act_bytes / tp / (links.dp_gbps * GB) + links.dp_latency_us * 1e-6) if pp > 1 else 0.0
    t_pipe = (m + pp - 1) * (t_stage + t_p2p) + m * t_head
    bubble = (pp - 1) / (m + pp - 1)
    # ---- data-parallel gradient synchronization + optimizer
    p_dev = spec.params() / (tp * pp)
    t_dp = _allreduce(2 * p_dev, dp, links.dp_gbps, links.dp_latency_us) * (1 - links.dp_overlap)
    zdiv = dp if zero >= 1 else 1
    t_opt = (p_dev / zdiv) * 16 * 2 / bw
    step = t_pipe + t_dp + t_opt
    # ---- memory per device
    w = 2 * p_dev / (dp if zero >= 3 else 1)
    g = 2 * p_dev / (dp if zero >= 2 else 1)
    o = 12 * p_dev / zdiv
    act_layer = t * d * (34 + 5 * a * T / d) / tp
    inflight = min(pp, m)
    act = (L // pp) * (2 * t * d / tp) * inflight + act_layer if recompute else (L // pp) * act_layer * inflight
    logits = t * V / tp * 4
    mem = w + g + o + act + logits
    cap = dev.hbm_gb * GB * links.memory_reserve
    tokens = global_batch * T
    model_flops = 3 * spec.fwd_flops_per_seq() * global_batch
    return {
        "tp": tp, "pp": pp, "dp": dp, "micro_batch": micro, "micro_batches": m, "zero": zero, "recompute": recompute,
        "fits": mem <= cap, "memory_gb": {"weights": w / GB, "grads": g / GB, "optimizer": o / GB,
                                          "activations": act / GB, "logits": logits / GB, "total": mem / GB,
                                          "capacity_usable": cap / GB},
        "time_s": {"step": step, "pipeline_compute": t_pipe, "tensor_parallel_comm": (m + pp - 1) * (L // pp) * t_tp,
                   "data_parallel_comm": t_dp, "optimizer": t_opt, "pipeline_bubble_fraction": bubble},
        "tokens_per_s": tokens / step, "mfu": model_flops / (devices * dev.peak_tflops * 1e12 * step),
        "params_per_device": p_dev,
    }


def plan(spec: TransformerSpec, dev: DeviceSpec, devices: int, global_batch: int = 1024, links: Links = Links(),
         top: int = 5) -> dict:
    """Search TP x PP x DP, micro-batch, ZeRO stage and recomputation; return the fastest plans that fit."""
    cands, rejected = [], 0
    tps = [x for x in (1, 2, 4, 8, 16, 32, 64) if spec.heads % x == 0 and spec.d_ff % x == 0 and x <= devices]
    pps = [x for x in range(1, spec.layers + 1) if spec.layers % x == 0 and x <= devices]
    for tp, pp, micro, zero, rc in itertools.product(tps, pps, (1, 2, 4, 8), (1, 2, 3), (False, True)):
        if devices % (tp * pp):
            continue
        dp = devices // (tp * pp)
        if global_batch % (dp * micro):
            continue
        r = evaluate_plan(spec, dev, devices, global_batch, tp, pp, micro, zero, rc, links)
        if r["fits"]:
            cands.append(r)
        else:
            rejected += 1
    cands.sort(key=lambda r: r["time_s"]["step"])
    best = cands[0] if cands else None
    p = spec.params()
    min_dev = math.ceil(16 * p / (dev.hbm_gb * GB * links.memory_reserve))
    return {"model": spec.describe(), "device": asdict(dev), "devices": devices, "global_batch": global_batch,
            "tokens_per_step": global_batch * spec.seq, "links": asdict(links), "feasible": best is not None,
            "best": best, "alternatives": cands[1:top], "plans_evaluated": len(cands) + rejected,
            "plans_rejected_memory": rejected, "min_devices_for_model_states": min_dev,
            "training_flops_per_step": 3 * spec.fwd_flops_per_seq() * global_batch,
            "label": "analytical estimate (Mode C): planning model with ASSUMED links and GEMM efficiency; "
                     "not an execution and not a measurement"}


def compare_devices(spec: TransformerSpec, a: DeviceSpec, b: DeviceSpec, devices: int, global_batch: int = 1024,
                    links: Links = Links()) -> dict:
    pa, pb = plan(spec, a, devices, global_batch, links), plan(spec, b, devices, global_batch, links)
    ratio = None
    if pa["best"] and pb["best"]:
        ratio = pb["best"]["time_s"]["step"] / pa["best"]["time_s"]["step"]
    return {"a": pa, "b": pb, "speedup_a_over_b": ratio,
            "label": "same analytical model and assumptions applied to both devices (BF16 dense, same links); "
                     "estimate, not a measurement of either device"}


def host_memory_available() -> int:
    """Bytes of RAM currently available on this host (Linux /proc/meminfo; fallback: unknown -> 0)."""
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def host_training_bytes(spec: TransformerSpec, batch: int) -> int:
    """Estimated host RAM for numeric training with the fast executor (FP32 Adam, all on one host)."""
    t = batch * spec.seq
    states = 16 * spec.params()                      # weights, grads, Adam m and v (FP32)
    transient = 16 * spec.params()                   # update temporaries, init copies
    acts = 4 * spec.layers * t * spec.d_model * (34 + 5 * spec.heads * spec.seq / spec.d_model)
    return int(states + transient + acts + 4 * t * spec.vocab * 3)
