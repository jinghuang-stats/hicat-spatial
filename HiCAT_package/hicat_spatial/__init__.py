"""HiCAT public API.

Objects are imported on first access so that lightweight utilities remain
usable when optional dependencies for a later workflow stage are unavailable.
"""

from __future__ import annotations

from importlib import import_module


_PRIMARY_EXPORTS = {
    # High-level workflow
    "HiCAT": (".core", "HiCAT"),
    "HiCATWorkflowConfig": (".main", "HiCATWorkflowConfig"),
    "HiCATWorkflowResult": (".main", "HiCATWorkflowResult"),
    "run_hicat_workflow": (".main", "run_hicat_workflow"),
    # Stage configurations and runners
    "PreprocessConfig": (".pipelines.step1_preprocessing", "PreprocessConfig"),
    "PreprocessPipelineResult": (
        ".pipelines.step1_preprocessing",
        "PreprocessPipelineResult",
    ),
    "run_preprocessing_pipeline": (
        ".pipelines.step1_preprocessing",
        "run_preprocessing_pipeline",
    ),
    "TreeInferenceStageConfig": (
        ".pipelines.step2_tree_inference",
        "TreeInferenceStageConfig",
    ),
    "run_tree_inference_stage": (
        ".pipelines.step2_tree_inference",
        "run_tree_inference_stage",
    ),
    "ReferenceSelectionStageConfig": (
        ".pipelines.step3_reference_selection",
        "ReferenceSelectionStageConfig",
    ),
    "run_reference_selection_stage": (
        ".pipelines.step3_reference_selection",
        "run_reference_selection_stage",
    ),
    "HierarchicalFeatureStageConfig": (
        ".pipelines.step4_hierarchical_features",
        "HierarchicalFeatureStageConfig",
    ),
    "HierarchicalFeatureStageResult": (
        ".pipelines.step4_hierarchical_features",
        "HierarchicalFeatureStageResult",
    ),
    "run_hierarchical_feature_stage": (
        ".pipelines.step4_hierarchical_features",
        "run_hierarchical_feature_stage",
    ),
    "ClusteringConfigStageConfig": (
        ".pipelines.step5_clustering_config",
        "ClusteringConfigStageConfig",
    ),
    "ClusteringConfigStageResult": (
        ".pipelines.step5_clustering_config",
        "ClusteringConfigStageResult",
    ),
    "run_clustering_config_stage": (
        ".pipelines.step5_clustering_config",
        "run_clustering_config_stage",
    ),
    "LabelTransferStageConfig": (
        ".pipelines.step6_label_transfer",
        "LabelTransferStageConfig",
    ),
    "LabelTransferStageResult": (
        ".pipelines.step6_label_transfer",
        "LabelTransferStageResult",
    ),
    "run_label_transfer_stage": (
        ".pipelines.step6_label_transfer",
        "run_label_transfer_stage",
    ),
    "HeterogeneityStageConfig": (
        ".pipelines.step7_heterogeneity",
        "HeterogeneityStageConfig",
    ),
    "run_heterogeneity_stage": (
        ".pipelines.step7_heterogeneity",
        "run_heterogeneity_stage",
    ),
    "load_stage_result": (".pipelines._io", "load_stage_result"),
}

# Advanced and legacy objects remain available by explicit import for backward
# compatibility, but are omitted from ``__all__`` to keep the recommended API
# focused on the workflow and its seven stages.
_COMPAT_EXPORTS = {
    "HiCATResult": (".data", "HiCATResult"),
    "HierTree": (".tree_inference", "HierTree"),
    "build_hier_tree": (".tree_inference", "build_hier_tree"),
    "infer_hier_tree_pipeline": (".tree_inference", "infer_hier_tree_pipeline"),
    "make_split_table": (".tree_inference", "make_split_table"),
    "save_tree_inference_results": (".tree_inference", "save_tree_inference_results"),
    "ReferenceSelectionResult": (".reference_selection", "ReferenceSelectionResult"),
    "select_references_pipeline": (".reference_selection", "select_references_pipeline"),
    "HierarchyRoundResult": (".label_transfer", "HierarchyRoundResult"),
    "HierarchicalTransferResult": (".label_transfer", "HierarchicalTransferResult"),
    "SingleReferenceNNTransferResult": (".label_transfer", "SingleReferenceNNTransferResult"),
    "MultiReferenceNNTransferResult": (".label_transfer", "MultiReferenceNNTransferResult"),
    "QuantileBasedTransferResult": (".label_transfer", "QuantileBasedTransferResult"),
    "single_ref_NN_based_label_transfer": (".label_transfer", "single_ref_NN_based_label_transfer"),
    "multi_ref_NN_based_label_transfer": (".label_transfer", "multi_ref_NN_based_label_transfer"),
    "quantile_based_label_transfer": (".label_transfer", "quantile_based_label_transfer"),
    "pseudo_to_spot_annotation": (".label_assignment", "pseudo_to_spot_annotation"),
    # Historical misspelling retained for compatibility.
    "sudo_to_spot_annotation": (".label_assignment", "sudo_to_spot_annotation"),
    "ReferenceHeterogeneityResult": (".heterogeneity", "ReferenceHeterogeneityResult"),
    "infer_heterogeneity_pipeline": (".heterogeneity", "infer_heterogeneity_pipeline"),
}

_EXPORTS = {**_PRIMARY_EXPORTS, **_COMPAT_EXPORTS}
__all__ = list(_PRIMARY_EXPORTS)


def __getattr__(name: str):
    """Load a public object on first access."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__():
    """Include lazy public objects in interactive completion."""
    return sorted(set(globals()) | set(__all__))
