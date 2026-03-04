"""Semicircle packing challenge."""

from .config import N, RADIUS
from .geometry import Semicircle, semicircle_polygon, semicircles_overlap, semicircle_contained_in_circle
from .scoring import validate_and_score, print_report, compute_mec, ValidationResult
from .baselines import circular_baseline, grid_baseline

__all__ = [
    "N",
    "RADIUS",
    "Semicircle",
    "semicircle_polygon",
    "semicircles_overlap",
    "semicircle_contained_in_circle",
    "validate_and_score",
    "print_report",
    "compute_mec",
    "ValidationResult",
    "circular_baseline",
    "grid_baseline",
]
