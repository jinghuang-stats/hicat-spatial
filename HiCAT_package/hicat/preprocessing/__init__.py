"""Public preprocessing utilities with optional workflows loaded lazily."""

from __future__ import annotations

from importlib import import_module

from .preprocess_util import (
    assign_spot_labels,
    construct_merged_scaled_adata_and_gene_df,
    construct_ref_adata_dic,
    filter_low_exp_genes,
    make_nonnegative_adata,
    normalize_adata,
    preprocess_adata,
    preprocess_adata_dic,
    subset_adata_dic_by_region,
)


_LAZY_EXPORTS = {
    "TissueContourResult": (".gene_enhancement", "TissueContourResult"),
    "GeneEnhancementResult": (".gene_enhancement", "GeneEnhancementResult"),
    "detect_he_tissue_mask": (".gene_enhancement", "detect_he_tissue_mask"),
    "detect_he_tissue_contour": (
        ".gene_enhancement",
        "detect_he_tissue_contour",
    ),
    "scan_spot_contour": (".gene_enhancement", "scan_spot_contour"),
    "contour_to_mask": (".gene_enhancement", "contour_to_mask"),
    "scale_tissue_contour": (".gene_enhancement", "scale_tissue_contour"),
    "remove_image_background": (
        ".gene_enhancement",
        "remove_image_background",
    ),
    "impute_gene_expression": (".gene_enhancement", "impute_gene_expression"),
    "enhance_gene_expression": (".gene_enhancement", "enhance_gene_expression"),
    "extract_scribble_labels_pipeline": (
        ".extract_scribble_annotations",
        "extract_scribble_labels_pipeline",
    ),
    "extract_image_features": (".image_features", "extract_image_features"),
}

__all__ = [
    "assign_spot_labels",
    "filter_low_exp_genes",
    "normalize_adata",
    "preprocess_adata",
    "preprocess_adata_dic",
    "construct_ref_adata_dic",
    "construct_merged_scaled_adata_and_gene_df",
    "subset_adata_dic_by_region",
    "make_nonnegative_adata",
    *_LAZY_EXPORTS,
]


def __getattr__(name: str):
    """Load optional preprocessing workflows only when requested."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
