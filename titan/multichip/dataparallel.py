"""Data-parallel training across N virtual TITAN-X1 devices.

Per step:  [device i] forward+backward on its shard (TITAN program)
           -> gradient all-reduce over the system interconnect (functional ring algorithm + cost model)
           -> [device i] optimizer update (TITAN program) on identical averaged gradients
Replica consistency is checked every step (parameters must stay bit-identical).
"""
from __future__ import annotations

import time

import numpy as np

from ..compiler import Graph, compile_graph, gradients
from ..config import load_config
from ..dtypes import DType
from ..nn.layers import ParamStore
from ..nn.models import MLPClassifier
from ..provenance import provenance
from ..runtime import Device
from ..train.datasets import Sampler, eval_batches, load_dataset
from .system import SystemConfig, allreduce_cost, ring_allreduce, system_preset


def build_programs(model, batch: int, cfg, lr: float, momentum: float = 0.9, seed: int = 0):
    g = Graph("dp-grad")
    ps = ParamStore(g, seed)
    _, logits = model.build(g, ps, batch, DType.FP32)
    loss = g.xent(logits, g.input("y", logits.shape))
    params = list(ps.params.values())
    for p, gr in zip(params, gradients(g, loss, params)):
        g.output("grad:" + p.name, gr)
    g.output("loss", loss)
    grad_exe = compile_graph(g, cfg)
    u = Graph("dp-update")
    for p in params:
        pt = u.param(p.name, p.value)
        gin = u.input("grad:" + p.name, p.shape)
        v = u.param(p.name + ".momentum", np.zeros(p.shape))
        v_new = u.add(u.scale(v, momentum), gin)
        u.assign(v, v_new)
        u.assign(pt, u.sub(pt, u.scale(v_new, lr)))
    upd_exe = compile_graph(u, cfg, share_with=grad_exe)
    return grad_exe, upd_exe, [p.name for p in params]


class DataParallelTrainer:
    def __init__(self, devices: int = 2, system: SystemConfig | None = None, arch: str = "dev",
                 dataset: str = "spirals", hidden=(64, 64), global_batch: int = 64, lr: float = 0.05,
                 seed: int = 0, mode: str = "functional"):
        if global_batch % devices:
            raise ValueError("global batch must be divisible by the number of devices")
        self.n, self.mode = devices, mode
        self.sys = system or system_preset(devices)
        if self.sys.devices != devices:
            raise ValueError("system size does not match the number of devices")
        self.cfg = load_config(arch)
        self.ds = load_dataset(dataset, seed)
        self.model = MLPClassifier(self.ds.in_dim, self.ds.classes, list(hidden))
        self.local_batch = global_batch // devices
        self.grad_exe, self.upd_exe, self.params = build_programs(self.model, self.local_batch, self.cfg, lr,
                                                                  seed=seed)
        self.devs = [Device(self.cfg) for _ in range(devices)]
        for d in self.devs:
            d.load(self.grad_exe)
            d.load(self.upd_exe, init=True)
        self.sampler = Sampler(self.ds, global_batch, seed + 1)
        self.grad_bytes = sum(int(np.prod(self.grad_exe.buffers[p].shape)) * 4 for p in self.params)
        self.history: list[dict] = []

    def step(self) -> dict:
        b = self.sampler.next()
        lb = self.local_batch
        losses, grads, cyc_grad = [], [], None
        t0 = time.perf_counter()
        for i, d in enumerate(self.devs):
            feeds = {"x": b["x"][i * lb:(i + 1) * lb], "y": b["y"][i * lb:(i + 1) * lb]}
            mode = self.mode if i == 0 else "functional"     # identical work on every rank
            out = d.run(self.grad_exe, feeds, mode)
            losses.append(float(out.outputs["loss"][0, 0]))
            grads.append(np.concatenate([out.outputs["grad:" + p].ravel() for p in self.params]))
            if i == 0 and out.result.cycles is not None:
                cyc_grad = out.result.cycles
        reduced, traffic = ring_allreduce(grads)
        mean = reduced[0] / self.n
        cyc_upd = None
        for i, d in enumerate(self.devs):
            feeds, off = {}, 0
            for p in self.params:
                shp = self.grad_exe.buffers[p].shape
                k = int(np.prod(shp))
                feeds["grad:" + p] = mean[off:off + k].reshape(shp)
                off += k
            out = d.run(self.upd_exe, feeds, self.mode if i == 0 else "functional", fetch=[])
            if i == 0 and out.result.cycles is not None:
                cyc_upd = out.result.cycles
        rec = {"step": len(self.history), "loss": float(np.mean(losses)), "host_s": time.perf_counter() - t0,
               "allreduce_bytes_per_rank": traffic["bytes_sent_per_rank"], "replicas_identical": self.consistent()}
        if cyc_grad is not None:
            hz = self.cfg.clock_ghz * 1e9
            comm = allreduce_cost(self.sys, self.grad_bytes)["seconds"]
            rec.update({"sim_compute_s": cyc_grad / hz, "sim_update_s": cyc_upd / hz, "model_comm_s": comm,
                        "step_s": cyc_grad / hz + comm + cyc_upd / hz})
        self.history.append(rec)
        return rec

    def consistent(self) -> bool:
        ref = [self.devs[0].read(self.grad_exe, p) for p in self.params]
        return all(np.array_equal(d.read(self.grad_exe, p), r) for d in self.devs[1:] for p, r in zip(self.params, ref))

    def evaluate(self) -> float:
        correct = total = 0
        g = Graph("dp-eval")
        ps = ParamStore(g, 0)
        _, logits = self.model.build(g, ps, self.local_batch, DType.FP32)
        g.output("logits", logits)
        exe = compile_graph(g, self.cfg, share_with=self.grad_exe)
        self.devs[0].load(exe, init=False)
        for xb, yb in eval_batches(self.ds, self.local_batch):
            lg = self.devs[0].run(exe, {"x": xb}).outputs["logits"]
            correct += int((lg.argmax(1) == yb).sum())
            total += len(yb)
        return correct / total if total else float("nan")


def scaling_study(device_counts=(1, 2, 4, 8, 16), topology: str = "ring", arch: str = "dev",
                  global_batch: int = 256, hidden=(256, 256), in_dim: int = 64, classes: int = 16,
                  chiplets_per_package: int = 0, link_gbps: float = 900.0, latency_us: float = 2.0) -> dict:
    """Strong scaling of one data-parallel training step (fixed global batch).

    Compute time per device: cycle model of the gradient + update programs at the local batch size.
    Communication: hierarchical all-reduce cost of the gradient bytes (assumed link parameters).
    """
    cfg = load_config(arch)
    model = MLPClassifier(in_dim, classes, list(hidden))
    rows, base = [], None
    rng = np.random.default_rng(0)
    counts = sorted(set(device_counts) | {1})       # speed-up and efficiency are always relative to 1 device
    for n in counts:
        if global_batch % n:
            continue
        lb = global_batch // n
        gexe, uexe, params = build_programs(model, lb, cfg, 0.01)
        dev = Device(cfg)
        dev.load(gexe)
        dev.load(uexe)
        feeds = {"x": rng.standard_normal((lb, in_dim)), "y": np.eye(classes)[rng.integers(0, classes, lb)]}
        rg = dev.run(gexe, feeds, "cycle").result
        ufeeds = {"grad:" + p: np.zeros(gexe.buffers[p].shape) for p in params}
        ru = dev.run(uexe, ufeeds, "cycle", fetch=[]).result
        hz = cfg.clock_ghz * 1e9
        grad_bytes = sum(int(np.prod(gexe.buffers[p].shape)) * 4 for p in params)
        if chiplets_per_package > 1 and n % chiplets_per_package == 0 and n > chiplets_per_package:
            sysc = system_preset(n, topology, link_gbps, latency_us, chiplets_per_package)
        else:
            sysc = system_preset(n, topology, link_gbps, latency_us)
        comm = allreduce_cost(sysc, grad_bytes)["seconds"] if n > 1 else 0.0
        compute = (rg.cycles + ru.cycles) / hz
        step = compute + comm
        base = base or step
        if n not in device_counts:
            continue
        rows.append({"devices": n, "system": sysc.name, "local_batch": lb, "sim_compute_s": compute,
                     "model_comm_s": comm, "step_s": step, "speedup": base / step,
                     "efficiency": base / (n * step), "comm_fraction": comm / step})
    return {"workload": f"MLP {in_dim}-{'-'.join(map(str, hidden))}-{classes}, global batch {global_batch}, "
                        f"one SGD step", "arch": cfg.name, "rows": rows,
            "method": "compute: simulated (cycle model, one device; ranks are symmetric); communication: "
                      "analytical alpha-beta model with ASSUMED link parameters",
            "provenance": provenance(arch_hash=cfg.config_hash())}
