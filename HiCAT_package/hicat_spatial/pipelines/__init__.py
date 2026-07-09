"""Stage-level orchestration for the HiCAT workflow.

Stage modules are loaded lazily so one stage cannot make unrelated stages
unimportable when its optional dependencies are unavailable.
"""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "PreprocessConfig": (".step1_preprocessing", "PreprocessConfig"),
    "PreprocessPipelineResult": (
        ".step1_preprocessing",
        "PreprocessPipelineResult",
    ),
    "run_preprocessing_pipeline": (
        ".step1_preprocessing",
        "run_preprocessing_pipeline",
    ),
    "TreeInferenceStageConfig": (
        ".step2_tree_inference",
        "TreeInferenceStageConfig",
    ),
    "construct_tree_reference_adata": (
        ".step2_tree_inference",
        "construct_tree_reference_adata",
    ),
    "run_tree_inference_stage": (
        ".step2_tree_inference",
        "run_tree_inference_stage",
    ),
    "rerun_tree_inference_with_weights": (
        ".step2_tree_inference",
        "rerun_tree_inference_with_weights",
    ),
    "ReferenceSelectionStageConfig": (
        ".step3_reference_selection",
        "ReferenceSelectionStageConfig",
    ),
    "run_reference_selection_stage": (
        ".step3_reference_selection",
        "run_reference_selection_stage",
    ),
    "HierarchicalFeatureStageConfig": (
        ".step4_hierarchical_features",
        "HierarchicalFeatureStageConfig",
    ),
    "HierarchicalFeatureStageResult": (
        ".step4_hierarchical_features",
        "HierarchicalFeatureStageResult",
    ),
    "run_hierarchical_feature_stage": (
        ".step4_hierarchical_features",
        "run_hierarchical_feature_stage",
    ),
    "ClusteringConfigStageConfig": (
        ".step5_clustering_config",
        "ClusteringConfigStageConfig",
    ),
    "ClusteringConfigStageResult": (
        ".step5_clustering_config",
        "ClusteringConfigStageResult",
    ),
    "run_clustering_config_stage": (
        ".step5_clustering_config",
        "run_clustering_config_stage",
    ),
    "LabelTransferStageConfig": (
        ".step6_label_transfer",
        "LabelTransferStageConfig",
    ),
    "LabelTransferStageResult": (
        ".step6_label_transfer",
        "LabelTransferStageResult",
    ),
    "run_label_transfer_stage": (
        ".step6_label_transfer",
        "run_label_transfer_stage",
    ),
    "HeterogeneityStageConfig": (
        ".step7_heterogeneity",
        "HeterogeneityStageConfig",
    ),
    "construct_merged_reference_gene_adata": (
        ".step7_heterogeneity",
        "construct_merged_reference_gene_adata",
    ),
    "run_heterogeneity_stage": (
        ".step7_heterogeneity",
        "run_heterogeneity_stage",
    ),
    "load_stage_result": ("._io", "load_stage_result"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
