"""Data-parallel training over every connected accelerator (e.g. all GPUs of a machine).

The training step is compiled as two TITAN programs that share the parameter buffers:
    gradient program  forward + loss + backward on 1/N of the batch     (every device, launched together)
    update program    global-norm clipping + optimizer on averaged grads (every device, identical)
Between them the gradients are averaged across devices (all-reduce through device 0). Every replica
applies the same update to the same parameters, so replicas stay identical without broadcasting
weights; a large-batch step and the data-parallel step compute the same update (tested).
"""
from __future__ import annotations

import numpy as np

from ..compiler import CompileOptions, Graph, compile_graph, gradients


def parse_accelerators(spec, backend: str) -> list[str]:
    """'auto' -> all accelerators of the fastest kind if there are >= 2, else one device;
    N -> the first N accelerators (CPU replicas if none); 'cuda:0,cuda:1' -> exactly these."""
    from .accel import default_device
    from .hardware import devices
    if backend != "accel":
        return []
    accel = [d["device"] for d in devices() if d["kind"] in ("cuda", "xpu")]
    if str(spec).startswith("cores"):
        return []                                             # process-per-core: titan.runtime.multiproc
    if spec in (None, "", "auto"):
        return accel if len(accel) >= 2 else []
    if str(spec).isdigit():
        n = int(spec)
        if n == 1:
            return []
        pool = accel or [default_device()] * n
        return (pool * n)[:n] if n > 1 else []
    return [s.strip() for s in str(spec).split(",") if s.strip()]


class DataParallel:
    def __init__(self, trainer, devices: list[str]):
        from .accel import AccelDevice
        tr, tc = trainer, trainer.tc
        self.tr, self.names = tr, devices
        n = len(devices)
        if tc.batch % n:
            raise ValueError(f"batch {tc.batch} is not divisible by {n} devices")
        if tc.lora_rank or tc.offload_dir:
            raise ValueError("data parallel training does not combine with LoRA or storage offload yet")
        self.shard = tc.batch // n
        # gradient program
        g = Graph(f"{tr.model.name}-dp-grad")
        ps = tr._store(g)
        _, logits = tr.model.build(g, ps, self.shard, tr.compute)
        loss = tr._loss(g, logits)
        params = list(ps.params.values())
        tr.frozen = [p.name for p in params if tr._is_frozen(p.name)]
        trainable = [p for p in params if p.name not in tr.frozen]
        for p, gr in zip(trainable, gradients(g, loss, trainable)):
            g.output(f"grad.{p.name}", gr)
        g.output("loss", loss)
        tr.model_params = [p.name for p in params]
        self.grad_exe = compile_graph(g, tr.cfg, CompileOptions(emit_isa=False))
        # update program (same parameter names -> shared buffers)
        u = Graph(f"{tr.model.name}-dp-update")
        up = [u.param_lazy(p.name, p.shape, lambda s=p.shape: np.zeros(s, np.float32)) for p in trainable]
        gin = [u.input(f"grad.{p.name}", p.shape) for p in trainable]
        tr._clip_and_apply(u, up, gin)
        self.update_exe = compile_graph(u, tr.cfg, CompileOptions(emit_isa=False), share_with=self.grad_exe)
        self.trainable = [p.name for p in trainable]
        self.devs = []
        for name in devices:
            d = AccelDevice(tr.cfg, device=name, torch_compile=bool(tc.torch_compile))
            d.load(self.grad_exe)
            d.load(self.update_exe, init=False)
            for k, v in self.update_exe.init.items():
                if k not in self.grad_exe.buffers:
                    d.write(self.update_exe, k, v)
            for k, fn in self.update_exe.lazy_init.items():
                if k not in self.grad_exe.buffers:
                    d.write(self.update_exe, k, fn())
            self.devs.append(d)

    # -------------------------------------------------------------- one step
    def _split(self, feeds: dict, i: int, n: int) -> dict:
        out = {}
        for k, v in feeds.items():
            v = np.asarray(v)
            if k in ("x", "y") and v.shape[0] % n == 0:
                r = v.shape[0] // n
                out[k] = v[i * r:(i + 1) * r]
            else:
                out[k] = v
        if "lm_w" in feeds:                                   # SFT: global token mean across shards
            y = np.asarray(feeds["y"])
            total = max(1, int((y >= 0).sum()))
            out["lm_w"] = np.array([[n * out["y"].size / total]])
        return out

    def step(self, feeds: dict, host: dict) -> dict:
        n = len(self.devs)
        for i, d in enumerate(self.devs):                     # launch all replicas (asynchronous on GPUs)
            d.run(self.grad_exe, {**self._split(feeds, i, n)}, fetch=[], sync=False)
        torch = self.devs[0].torch
        losses = []
        avg = {}
        with torch.no_grad():
            for d in self.devs:
                d.sync()
                losses.append(float(d.tensor(self.grad_exe, "loss").reshape(-1)[0]))
            for name in self.trainable:
                ts = [d.tensor(self.grad_exe, f"grad.{name}") for d in self.devs]
                acc = ts[0].clone()
                for t in ts[1:]:
                    acc += t.to(acc.device, non_blocking=True)
                avg[name] = acc / n
            outs = None
            for d in self.devs:
                feeds_u = {f"grad.{k}": v.to(d.device, non_blocking=True) for k, v in avg.items()}
                feeds_u.update(host)
                r = d.run(self.update_exe, feeds_u, fetch=["grad_norm"] if "grad_norm" in self.update_exe.outputs else [])
                outs = outs or r
        rec = {"loss": float(np.mean(losses))}
        if "grad_norm" in outs.outputs:
            rec["grad_norm"] = float(outs.outputs["grad_norm"][0, 0])
        rec["result"] = outs.result
        return rec

    def sync_params(self) -> None:
        """Copy device 0's parameters to every replica (after loading a checkpoint)."""
        for name in self.grad_exe.params:
            v = self.devs[0].read(self.grad_exe, name)
            for d in self.devs[1:]:
                d.write(self.grad_exe, name, v)

    def consistent(self) -> bool:
        return all(np.array_equal(self.devs[0].read(self.grad_exe, n), d.read(self.grad_exe, n))
                   for d in self.devs[1:] for n in self.grad_exe.params)
