"""TITAN runtime: devices, execution backends and host <-> device transfers."""
from .backends import BACKENDS, Backend, available_backends, backend_status
from .device import Device, RunOutput

__all__ = ["BACKENDS", "Backend", "Device", "RunOutput", "available_backends", "backend_status"]
from .fast import FastDevice  # noqa: E402

__all__.append("FastDevice")
from .accel import AccelDevice, torch_available  # noqa: E402

__all__ += ["AccelDevice", "torch_available"]
