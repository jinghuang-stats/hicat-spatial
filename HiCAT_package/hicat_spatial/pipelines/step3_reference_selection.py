"""Stage 3: select query-specific reference sections."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

from ._io import (
    ensure_output_dir,
    logged_stage,
    save_json,
    save_stage_result,
    stage_output_from_config,
)


@dataclass
class ReferenceSelectionStageConfig:
    """Configuration for query-specific reference selection.

    Parameters
    ----------
    output_dir : path-like or None, default=None
        Stage output directory. ``None`` uses
        ``results/03_reference_selection``.
    label_key : str, default="label"
        Reference ``.obs`` column containing tissue-region labels.
    low_exp_thres : float, default=0.05
        Minimum expressing-observation fraction used during optional filtering.
    normalize : bool, default=True
        Apply within-section normalization during optional preprocessing.
    normalization_method : {"min_max", "none"}, default="min_max"
        Normalization forwarded to ``preprocess_adata_dic``.
    min_region_prop : float, default=0.05
        Minimum within-reference proportion for a region to contribute to the
        reference weight.
    exclude_labels : sequence[str], default=("nan", "unknown")
        Reference labels excluded from marker and weight calculations.
    exclude_mode : {"contains", "exact"}, default="contains"
        How ``exclude_labels`` are matched.
    pvals_adj : float, default=0.05
        Adjusted-p-value cutoff for reference marker selection.
    min_in_out_group_ratio : float, default=1.0
        Minimum in-region versus out-region expression ratio.
    min_in_group_fraction : float, default=0.5
        Minimum fraction of in-region observations expressing a marker.
    min_fold_change : float, default=1.10
        Minimum marker fold change.
    gene_num : int, default=10
        Maximum marker genes retained per reference region.
    weight_key : str, default="weight"
        Reference-weight column name.
    similarity_key : str, default="KS_similarity"
        Name assigned to the raw query-reference similarity score.
    summary_key : str, default="average"
        Region-score aggregation statistic.
    similarity_method : {"ks"}, default="ks"
        Query-reference similarity method.
    sort_by : {"weighted", "similarity"}, default="weighted"
        Score used to rank references.
    selection_mode : {"cutoff", "top_k"}, default="cutoff"
        Final reference-selection rule.
    alpha : float, default=0.9
        Relative cutoff from the best score in cutoff mode.
    top_k : int, default=3
        Maximum selected references in top-k mode.
    min_similarity_level : float, default=0.7
        Minimum acceptable similarity used by the selection rule.
    selection_score_key : str or None, default=None
        Explicit score column; ``None`` derives it from ``sort_by``.
    preprocess_ref, preprocess_qry : bool, default=True
        Filter/normalize reference and query dictionaries before comparison.
        Set both to ``False`` only when inputs are already prepared for this
        algorithm.
    print_results : bool, default=True
        Print intermediate summaries.
    """

    output_dir: Path | str | None = None
    label_key: str = "label"
    low_exp_thres: float = 0.05
    normalize: bool = True
    normalization_method: str = "min_max"
    min_region_prop: float = 0.05
    exclude_labels: Sequence[str] = ("nan", "unknown")
    exclude_mode: str = "contains"
    pvals_adj: float = 0.05
    min_in_out_group_ratio: float = 1.0
    min_in_group_fraction: float = 0.5
    min_fold_change: float = 1.10
    gene_num: int = 10
    weight_key: str = "weight"
    similarity_key: str = "KS_similarity"
    summary_key: str = "average"
    similarity_method: str = "ks"
    sort_by: str = "weighted"
    selection_mode: str = "cutoff"
    alpha: float = 0.9
    top_k: int = 3
    min_similarity_level: float = 0.7
    selection_score_key: Optional[str] = None
    preprocess_ref: bool = True
    preprocess_qry: bool = True
    print_results: bool = True


@logged_stage(
    "reference_selection",
    stage_output_from_config("results/03_reference_selection", config_position=2),
)
def run_reference_selection_stage(
    ref_gene_dic,
    query_gene_dic,
    config: ReferenceSelectionStageConfig,
    ref_section_list=None,
    query_section_list=None,
):
    """Run Stage 3 and save query-specific reference rankings.

    Parameters
    ----------
    ref_gene_dic : mapping[str, AnnData]
        Reference Gene objects keyed by section. Each object must have unique
        ``obs_names`` and ``config.label_key`` in ``.obs``.
    query_gene_dic : mapping[str, AnnData]
        Query Gene objects keyed by section. Reference and query objects must
        share usable genes; query labels are not required.
    config : ReferenceSelectionStageConfig
        Preprocessing, marker, similarity, selection, and output settings.
    ref_section_list, query_section_list : sequence[str] or None, default=None
        Optional ordered subsets of dictionary keys. ``None`` uses every key.

    Returns
    -------
    ReferenceSelectionResult
        Contains ``selected_refs_dic``, ``ranked_refs_dic``, query-specific
        similarity/selection tables, weights, markers, processed inputs, and
        parameters. Use ``result.selected_refs_dic[query_section]`` for
        Stages 4–6.

    Saved files
    -----------
    ``reference_selection_result.pkl``, ``selected_references.json``,
    ``stage_config.json``, ``reference_selection_summary.csv``, and per-query
    similarity/selection CSV files.
    """
    from ..reference_selection import select_references_pipeline

    output_dir = ensure_output_dir(
        config.output_dir or "results/03_reference_selection"
    )
    kwargs = asdict(config)
    kwargs.pop("output_dir")
    result = select_references_pipeline(
        ref_adata_dic=dict(ref_gene_dic),
        qry_adata_dic=dict(query_gene_dic),
        ref_section_list=ref_section_list,
        qry_section_list=query_section_list,
        **kwargs,
    )

    save_stage_result(result, output_dir / "reference_selection_result.pkl")
    save_json(result.selected_refs_dic, output_dir / "selected_references.json")
    config_record = asdict(config)
    config_record["output_dir"] = str(output_dir)
    save_json(config_record, output_dir / "stage_config.json")
    result.to_summary_df().to_csv(output_dir / "reference_selection_summary.csv")

    for query_section in result.selected_refs_dic:
        query_dir = ensure_output_dir(output_dir / str(query_section))
        result.similarity_summary_dic[query_section].to_csv(
            query_dir / "similarity_summary.csv"
        )
        result.selection_summary_dic[query_section].to_csv(
            query_dir / "selection_summary.csv"
        )
    return result


__all__ = ["ReferenceSelectionStageConfig", "run_reference_selection_stage"]
