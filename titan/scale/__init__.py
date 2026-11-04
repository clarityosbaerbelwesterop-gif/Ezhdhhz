"""Large-model scaling: model specs up to 100B+ parameters and the distributed training planner."""
from .planner import (PRESETS, compile_training_program, execution_memory, offload_requirements, DeviceSpec, Links, TransformerSpec, compare_devices, evaluate_plan,
                      host_memory_available, host_training_bytes, parse_count, plan, preset, spec_for_params)

__all__ = ["PRESETS", "compile_training_program", "execution_memory", "offload_requirements", "DeviceSpec", "Links", "TransformerSpec", "compare_devices", "evaluate_plan",
           "host_memory_available", "host_training_bytes", "parse_count", "plan", "preset", "spec_for_params"]
