"""Public preprocessing utilities with optional workflows loaded lazily."""

from __future__ import annotations

from importlib import import_module

_PRIMARY_EXPORTS = {
    "preprocess_molecular_adata": (".preprocess_util", "preprocess_molecular_adata"),
    "preprocess_molecular_sections": (
        ".preprocess_util",
        "preprocess_molecular_sections",
    ),
    "TissueContourResult": (".gene_enhancement", "TissueContourResult"),
    "GeneEnhancementResult": (".gene_enhancement", "GeneEnhancementResult"),
    "enhance_gene_expression": (".gene_enhancement", "enhance_gene_expression"),
    "extract_scribble_labels_pipeline": (
        ".extract_scribble_annotations",
        "extract_scribble_labels_pipeline",
    ),
    "extract_image_features": (".image_features", "extract_image_features"),
}

_COMPAT_EXPORTS = {
    "assign_spot_labels": (".preprocess_util", "assign_spot_labels"),
    "filter_low_exp_genes": (".preprocess_util", "filter_low_exp_genes"),
    "normalize_adata": (".preprocess_util", "normalize_adata"),
    "preprocess_adata": (".preprocess_util", "preprocess_adata"),
    "preprocess_adata_dic": (".preprocess_util", "preprocess_adata_dic"),
    "construct_ref_adata_dic": (".preprocess_util", "construct_ref_adata_dic"),
    "construct_merged_scaled_adata_and_gene_df": (
        ".preprocess_util",
        "construct_merged_scaled_adata_and_gene_df",
    ),
    "subset_adata_dic_by_region": (
        ".preprocess_util",
        "subset_adata_dic_by_region",
    ),
    "make_nonnegative_adata": (".preprocess_util", "make_nonnegative_adata"),
    "detect_he_tissue_mask": (".gene_enhancement", "detect_he_tissue_mask"),
    "detect_he_tissue_contour": (
        ".gene_enhancement",
        "detect_he_tissue_contour",
    ),
    "scan_spot_contour": (".gene_enhancement", "scan_spot_contour"),
    "contour_to_mask": (".gene_enhancement", "contour_to_mask"),
    "scale_tissue_contour": (".gene_enhancement", "scale_tissue_contour"),
    "remove_image_background": (".gene_enhancement", "remove_image_background"),
    "impute_gene_expression": (".gene_enhancement", "impute_gene_expression"),
}

_EXPORTS = {**_PRIMARY_EXPORTS, **_COMPAT_EXPORTS}
__all__ = list(_PRIMARY_EXPORTS)


def __getattr__(name: str):
    """Load preprocessing objects only when requested."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
