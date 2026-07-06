"""High-level, stateful interface to the HiCAT workflow stages."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Mapping, Optional

from .data import HiCATResult


_STAGE_RUNNERS = {
    "preprocessing": (
        ".preprocessing.preprocess_util",
        "preprocess_adata_dic",
    ),
    "tree_inference": (
        ".pipelines.step2_tree_inference",
        "run_tree_inference_stage",
    ),
    "reference_selection": (
        ".pipelines.step3_reference_selection",
        "run_reference_selection_stage",
    ),
    "hierarchical_features": (
        ".pipelines.step4_hierarchical_features",
        "run_hierarchical_feature_stage",
    ),
    "clustering_config": (
        ".pipelines.step5_clustering_config",
        "run_clustering_config_stage",
    ),
    "label_transfer": (
        ".pipelines.step6_label_transfer",
        "run_label_transfer_stage",
    ),
    "heterogeneity": (
        ".pipelines.step7_heterogeneity",
        "run_heterogeneity_stage",
    ),
}


class HiCAT:
    """Run HiCAT stages and retain their results.

    Parameters
    ----------
    config
        Optional mapping from stage name to that stage's configuration object.
        A configuration passed directly to a ``run_*`` method takes precedence.

    Notes
    -----
    HiCAT's transfer frameworks require different input dictionary layouts, so
    this class deliberately exposes the same explicit stage boundaries as the
    functions in :mod:`hicat.pipelines`.
    """

    def __init__(self, config: Optional[Mapping[str, Any]] = None):
        if config is not None and not isinstance(config, Mapping):
            raise TypeError("config must be a mapping from stage names to configs.")
        self.config = dict(config or {})
        self.result = HiCATResult()

    def _resolve_config(self, stage: str, config: Any) -> Any:
        if config is not None:
            return config
        if stage not in self.config:
            raise ValueError(
                f"No configuration was supplied for stage {stage!r}. "
                "Pass config=... or provide it when constructing HiCAT."
            )
        return self.config[stage]

    def run_stage(self, stage: str, *args: Any, **kwargs: Any) -> Any:
        """Run a named stage, store its native result, and return it."""
        if stage not in _STAGE_RUNNERS:
            raise ValueError(
                f"Unknown stage {stage!r}; expected one of {tuple(_STAGE_RUNNERS)}."
            )
        module_name, function_name = _STAGE_RUNNERS[stage]
        runner = getattr(import_module(module_name, __package__), function_name)
        stage_result = runner(*args, **kwargs)
        setattr(self.result, stage, stage_result)
        return stage_result

    def run_preprocessing(
        self,
        adata_dic: Mapping[str, Any],
        **kwargs: Any,
    ) -> Any:
        """Preprocess a section-to-AnnData mapping and store the result."""
        return self.run_stage("preprocessing", adata_dic, **kwargs)

    def run_tree_inference(
        self,
        ref_adata_dic: Mapping[str, Any],
        config: Any = None,
    ) -> Any:
        """Run stage 2."""
        return self.run_stage(
            "tree_inference",
            ref_adata_dic,
            self._resolve_config("tree_inference", config),
        )

    def run_reference_selection(
        self,
        ref_gene_dic: Mapping[str, Any],
        query_gene_dic: Mapping[str, Any],
        config: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Run stage 3."""
        return self.run_stage(
            "reference_selection",
            ref_gene_dic,
            query_gene_dic,
            self._resolve_config("reference_selection", config),
            **kwargs,
        )

    def run_hierarchical_features(
        self,
        ref_adata_by_modality: Mapping[str, Mapping[str, Any]],
        hier_tree: Any,
        config: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Run stage 4."""
        return self.run_stage(
            "hierarchical_features",
            ref_adata_by_modality,
            hier_tree,
            self._resolve_config("hierarchical_features", config),
            **kwargs,
        )

    def run_clustering_config(
        self,
        ref_adata_by_modality: Mapping[str, Mapping[str, Any]],
        feature_stage_result: Any,
        config: Any = None,
    ) -> Any:
        """Run stage 5."""
        return self.run_stage(
            "clustering_config",
            ref_adata_by_modality,
            feature_stage_result,
            self._resolve_config("clustering_config", config),
        )

    def run_label_transfer(
        self,
        jobs: Mapping[str, Mapping[str, Any]],
        config: Any = None,
    ) -> Any:
        """Run stage 6."""
        return self.run_stage(
            "label_transfer",
            jobs,
            self._resolve_config("label_transfer", config),
        )

    def run_heterogeneity(
        self,
        ref_gene_dic: Mapping[str, Any],
        config: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Run stage 7."""
        return self.run_stage(
            "heterogeneity",
            ref_gene_dic,
            self._resolve_config("heterogeneity", config),
            **kwargs,
        )


__all__ = ["HiCAT"]
