"""Backward-compatible aliases for the maintained metric implementation."""

from .computeMetrics import compute_metrics, patchify

__all__ = ["patchify", "compute_metrics"]
