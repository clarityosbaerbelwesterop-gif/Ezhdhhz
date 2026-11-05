"""Performance engine: analytical roofline (Mode C), B300 reference rules, benchmark harness."""
from .analytical import LABEL, gemm_roofline, program_analysis, roofline_points, scale_requirements
from .reference import B300, ComparisonError, compare, compare_peak

__all__ = ["LABEL", "gemm_roofline", "program_analysis", "roofline_points", "scale_requirements", "B300",
           "ComparisonError", "compare", "compare_peak"]
