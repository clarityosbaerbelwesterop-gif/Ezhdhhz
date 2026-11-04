"""ROUGE TITAN — TITAN-X1 digital AI accelerator platform."""
__version__ = "0.1.0"

from .config import ArchConfig, load_config  # noqa: E402
from .dtypes import DType  # noqa: E402

__all__ = ["ArchConfig", "DType", "load_config", "__version__"]
