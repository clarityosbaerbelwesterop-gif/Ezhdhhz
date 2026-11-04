"""Verified configuration search (tile sizes, scheduling, architecture parameters)."""
from .autotune import evaluate_candidate, sweep_architecture, tune_gemm

__all__ = ["evaluate_candidate", "sweep_architecture", "tune_gemm"]
