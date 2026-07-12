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

    Parameters
    ----------
    output_dir : path-like or None, default=None
        Stage output directory. ``None`` uses
        ``results/07_heterogeneity``.
    dataset_name : str or None, default=None
        Optional identifier stored in the returned result.
    parameters : dict, default={}
        Keywords forwarded to ``infer_heterogeneity_pipeline``. Important
        defaults include:

        - ``label_key="label"`` and ``sample_key="sample"``;
        - ``selection_method="threshold"``, ``hetero_threshold=0.5``,
          ``top_k=None``, and ``score_key="hetero_score_sca"``;
        - ``run_subtype=True`` and ``min_region_spots=10``;
        - ``n_perm=200``, ``pcs_num=30``, and ``random_state=0``;
        - ``section_cluster_method="leiden_clusters"``;
        - ``x_key="pixel_x"`` and ``y_key="pixel_y"``;
        - ``cat_color=None``, ``cnt_color="coolwarm"``,
          ``fig_size=50``, and ``dpi=100``;
        - ``print_results=True``.

        ``res_dir`` is managed by the stage and overwritten with
        ``output_dir``.
    """

    output_dir: Path | str | None = None
    dataset_name: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)


def construct_merged_reference_gene_adata(ref_gene_dic, sample_key="sample"):
    """Merge reference Gene objects on their ordered common gene set.

    Parameters
    ----------
    ref_gene_dic : mapping[str, AnnData]
        Section-keyed reference Gene objects with unique ``obs_names``.
    sample_key : str, default="sample"
        Added ``.obs`` column identifying the source section.

    Returns
    -------
    merged : AnnData
        Row-concatenated object restricted to ordered common genes. Observation
        names receive section suffixes and are unique.
    common_genes : list[str]
        Genes shared by every reference section, ordered as in the first one.
    """
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

    objects = []
    for section in section_list:
        section_adata = ref_gene_dic[section]
        if not section_adata.obs_names.is_unique:
            raise ValueError(
                f"Observation names must be unique within {section!r} before merging."
            )
        objects.append(section_adata[:, common_genes].copy())
    merged = ad.concat(
        objects,
        axis=0,
        join="inner",
        label=sample_key,
        keys=section_list,
        index_unique="-",
        merge="first",
    )
    merged.var["genes"] = merged.var_names.astype(str)
    if not merged.obs_names.is_unique:
        raise RuntimeError("Merged observation names are not unique after section suffixing.")
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
    """Run Stage 7 and save heterogeneity and optional subtype results.

    Parameters
    ----------
    ref_gene_dic : mapping[str, AnnData]
        Reference Gene objects keyed by section. Each object must have unique
        ``obs_names``, the configured label column, and shared genes. Multiple
        reference sections are required for meaningful cross-section scores.
    config : HeterogeneityStageConfig
        Dataset label, output path, scoring, selection, subtype, clustering,
        and plotting parameters.
    all_adata : AnnData or None, default=None
        Optional pre-merged reference object. It must contain
        ``parameters['sample_key']`` in ``.obs``. ``None`` constructs it from
        ``ref_gene_dic`` using ordered common genes.
    common_genes : sequence[str] or None, default=None
        Genes used by the merged analysis. When ``all_adata`` is constructed
        internally, ``None`` uses its inferred common genes.

    Returns
    -------
    ReferenceHeterogeneityResult
        Score tables, selected regions/scores, region markers, optional subtype
        results, sample names, and run parameters.

    Saved files
    -----------
    ``heterogeneity_result.pkl``, ``stage_config.json``,
    ``selected_regions.json``, and available heterogeneity, marker-stability,
    permutation-silhouette, and selected-region CSV summaries. The underlying
    pipeline may also save subtype and spatial figures below ``output_dir``.
    """
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
