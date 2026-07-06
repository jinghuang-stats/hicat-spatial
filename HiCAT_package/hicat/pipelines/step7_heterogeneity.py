"""Stage 7: infer reference-region heterogeneity and shared subtypes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import anndata as ad

from ._io import (
    ensure_output_dir,
    logged_stage,
    save_json,
    save_stage_result,
    stage_output_from_config,
)


@dataclass
class HeterogeneityStageConfig:
    """Configuration for reference heterogeneity inference.

    ``parameters`` is forwarded to ``infer_heterogeneity_pipeline``. Common
    settings are ``label_key``, ``sample_key``, ``selection_method``,
    ``hetero_threshold``, ``top_k``, ``run_subtype``, ``n_perm``, clustering
    settings, spatial plotting settings, and ``print_results``.
    """

    output_dir: Path | str | None = None
    dataset_name: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)


def construct_merged_reference_gene_adata(ref_gene_dic, sample_key="sample"):
    """Merge reference gene objects on their ordered common gene set."""
    if not ref_gene_dic:
        raise ValueError("ref_gene_dic cannot be empty.")
    section_list = list(ref_gene_dic)
    first_genes = list(ref_gene_dic[section_list[0]].var_names)
    common = set(first_genes)
    for section in section_list[1:]:
        common.intersection_update(ref_gene_dic[section].var_names)
    common_genes = [gene for gene in first_genes if gene in common]
    if not common_genes:
        raise ValueError("No genes are shared across reference sections.")

    objects = [
        ref_gene_dic[section][:, common_genes].copy() for section in section_list
    ]
    merged = ad.concat(
        objects,
        axis=0,
        join="inner",
        label=sample_key,
        keys=section_list,
        index_unique=None,
        merge="first",
    )
    merged.var["genes"] = merged.var_names.astype(str)
    return merged, common_genes


@logged_stage(
    "heterogeneity",
    stage_output_from_config("results/07_heterogeneity", config_position=1),
)
def run_heterogeneity_stage(
    ref_gene_dic: Mapping[str, Any],
    config: HeterogeneityStageConfig,
    all_adata=None,
    common_genes=None,
):
    """Run stage 7 and save score tables, selected regions, and full results."""
    from ..heterogeneity import infer_heterogeneity_pipeline

    output_dir = ensure_output_dir(config.output_dir or "results/07_heterogeneity")
    parameters = dict(config.parameters)
    sample_key = parameters.get("sample_key", "sample")
    if all_adata is None:
        all_adata, inferred_common_genes = construct_merged_reference_gene_adata(
            ref_gene_dic,
            sample_key=sample_key,
        )
        if common_genes is None:
            common_genes = inferred_common_genes
    parameters["res_dir"] = str(output_dir)
    result = infer_heterogeneity_pipeline(
        ref_adata_dic=dict(ref_gene_dic),
        all_adata=all_adata,
        dataset_name=config.dataset_name,
        common_genes=common_genes,
        **parameters,
    )

    save_stage_result(result, output_dir / "heterogeneity_result.pkl")
    config_record = asdict(config)
    config_record["output_dir"] = str(output_dir)
    save_json(config_record, output_dir / "stage_config.json")
    save_json(result.selected_regions, output_dir / "selected_regions.json")
    if result.hetero_summary is not None:
        result.hetero_summary.to_csv(output_dir / "heterogeneity_summary.csv")
    if result.sta_summary is not None:
        result.sta_summary.to_csv(output_dir / "marker_stability_summary.csv")
    if result.perm_sil_summary is not None:
        result.perm_sil_summary.to_csv(
            output_dir / "permutation_silhouette_summary.csv"
        )
    if result.selected_region_scores is not None:
        result.selected_region_scores.to_csv(output_dir / "selected_region_scores.csv")
    return result


__all__ = [
    "HeterogeneityStageConfig",
    "construct_merged_reference_gene_adata",
    "run_heterogeneity_stage",
]
