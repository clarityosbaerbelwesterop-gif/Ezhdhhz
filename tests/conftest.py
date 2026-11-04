import numpy as np
import pytest

from titan import load_config
from titan.dtypes import DType, nbytes
from titan.emu import Machine
from titan.isa import Space, TensorRef


@pytest.fixture
def tiny():
    return load_config("tiny")


@pytest.fixture
def dev():
    return load_config("dev")


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


class Placer:
    """Bump allocator for hand-written test programs."""

    def __init__(self, space: Space, base: int = 0):
        self.space, self.off = space, base

    def __call__(self, rows: int, cols: int, dt: DType, stride: int = 0) -> TensorRef:
        st = stride or cols
        ref = TensorRef(self.space, self.off, rows, cols, dt, st)
        self.off += -(-nbytes(dt, (rows - 1) * st + cols) // 64) * 64
        return ref


@pytest.fixture
def placer():
    return Placer


def machine(cfg):
    return Machine(cfg)
