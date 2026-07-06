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
    """Configuration mirroring ``select_references_pipeline``.

    Reference selection uses gene-expression data. Set ``preprocess_ref`` and
    ``preprocess_qry`` to ``False`` when inputs have already been filtered and
    scaled exactly as desired for this stage.
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
    """Run stage 3 and save query-specific scores and selected references.

    Returns
    -------
    ReferenceSelectionResult
        Use ``result.selected_refs_dic[query_section]`` to retrieve the
        references passed to stages 4–6.
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
