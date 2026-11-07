"""Process-per-core data parallel: every CPU core (or group of cores) becomes one training device.

Small and medium models do not scale across the threads of one process (per-kernel dispatch and
synchronisation dominate); separate processes do (measured: 4 single-core processes reach 3.7-5x the
aggregate step rate of one 4-thread process). Each worker holds a replica (gradient + update program,
titan.runtime.multidevice), computes gradients on its shard, writes them to shared memory, averages
all workers' gradients and applies the identical update, so replicas stay identical and the step equals
a large-batch step.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import tempfile

import numpy as np


def _worker(rank, n, tc_dict, names_sizes, shm_name, barrier, conn):
    import torch
    torch.set_num_threads(1)
    try:
        cpus = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {cpus[rank % len(cpus)]})
    except (AttributeError, OSError):
        pass
    from multiprocessing import shared_memory
    from ..train import TrainConfig, Trainer
    from .multidevice import DataParallel
    with tempfile.TemporaryDirectory() as d:
        tr = Trainer(TrainConfig(**{**tc_dict, "accelerators": "1", "out_dir": d, "eval_every": 0, "ckpt_every": 0}),
                     log=None)
        dp = DataParallel(tr, ["cpu"])
        dev = dp.devs[0]
        total = sum(s for _, s in names_sizes)
        shm = shared_memory.SharedMemory(name=shm_name)
        buf = np.ndarray((n, total), dtype=np.float32, buffer=shm.buf)
        while True:
            msg = conn.recv()
            if msg[0] == "stop":
                break
            if msg[0] == "params":
                conn.send({k: dev.read(dp.grad_exe, k) for k in dp.grad_exe.params})
                continue
            _, feeds, host = msg
            out = dev.run(dp.grad_exe, feeds, fetch=["loss"])
            off = 0
            for name, size in names_sizes:
                buf[rank, off:off + size] = dev.tensor(dp.grad_exe, f"grad.{name}").reshape(-1).numpy()
                off += size
            barrier.wait()
            avg = buf.mean(0)
            barrier.wait()                                     # nobody overwrites a slot before all have read
            feeds_u, off = dict(host), 0
            for name, size in names_sizes:
                feeds_u[f"grad.{name}"] = torch.from_numpy(avg[off:off + size].copy())
                off += size
            r = dev.run(dp.update_exe, feeds_u, fetch=["grad_norm"] if "grad_norm" in dp.update_exe.outputs else [])
            conn.send((float(out.outputs["loss"][0, 0]), float(r.outputs["grad_norm"][0, 0]) if r.outputs else None))
        shm.close()


class ProcessDataParallel:
    def __init__(self, trainer, n: int):
        from multiprocessing import shared_memory
        from dataclasses import asdict
        tr = trainer
        if tr.tc.batch % n:
            raise ValueError(f"batch {tr.tc.batch} is not divisible by {n} processes")
        self.n, self.shard = n, tr.tc.batch // n
        tc = {**asdict(tr.tc), "batch": self.shard}
        names_sizes = [(p, int(np.prod(tr.train_exe.buffers[p].shape))) for p in tr.model_params
                       if p not in tr.frozen]
        self.shm = shared_memory.SharedMemory(create=True, size=4 * n * max(1, sum(s for _, s in names_sizes)))
        ctx = mp.get_context("spawn")
        barrier = self.barrier = ctx.Barrier(n)            # keep alive until the children have attached
        self.conns, self.procs = [], []
        for r in range(n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(r, n, tc, names_sizes, self.shm.name, barrier, b), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)

    def step(self, feeds: dict, host: dict) -> dict:
        from .multidevice import DataParallel
        for i, c in enumerate(self.conns):
            c.send(("step", DataParallel._split(None, feeds, i, self.n), host))
        res = [c.recv() for c in self.conns]
        return {"loss": float(np.mean([r[0] for r in res])), "grad_norm": res[0][1]}

    def params(self) -> dict:
        self.conns[0].send(("params",))
        return self.conns[0].recv()

    def close(self) -> None:
        for c in self.conns:
            try:
                c.send(("stop",))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=10)
        self.shm.close()
        self.shm.unlink()
