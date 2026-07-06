"""Shared result containers for the high-level HiCAT API.

Algorithm-specific result classes live beside their implementations.  This
module contains only the aggregate result returned by :class:`hicat.HiCAT`,
which keeps this module lightweight and safe to import without optional
scientific dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Tuple


@dataclass
class HiCATResult:
    """Results accumulated while running HiCAT's seven workflow stages.

    A field remains ``None`` until its stage has completed.  Individual fields
    contain the native result returned by the corresponding stage runner; for
    example, ``reference_selection`` is a
    :class:`hicat.reference_selection.ReferenceSelectionResult`.
    """

    preprocessing: Optional[Any] = None
    tree_inference: Optional[Any] = None
    reference_selection: Optional[Any] = None
    hierarchical_features: Optional[Any] = None
    clustering_config: Optional[Any] = None
    label_transfer: Optional[Any] = None
    heterogeneity: Optional[Any] = None

    def completed_stages(self) -> Tuple[str, ...]:
        """Return completed stage names in workflow order."""
        return tuple(
            result_field.name
            for result_field in fields(self)
            if getattr(self, result_field.name) is not None
        )

    def as_dict(self) -> Dict[str, Any]:
        """Return a shallow mapping of stage names to their native results."""
        return {
            result_field.name: getattr(self, result_field.name)
            for result_field in fields(self)
        }

    def __getitem__(self, stage: str) -> Any:
        """Retrieve a stage result by name."""
        if stage not in {result_field.name for result_field in fields(self)}:
            raise KeyError(f"Unknown HiCAT stage: {stage!r}.")
        return getattr(self, stage)


__all__ = ["HiCATResult"]
