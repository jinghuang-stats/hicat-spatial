"""Stage 4: select hierarchy-aware features for each query/reference set."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pandas as pd

from ._io import (
    ensure_output_dir,
    logged_stage,
    save_json,
    save_stage_result,
    stage_output_from_config,
)


@dataclass
class HierarchicalFeatureStageConfig:
    """Configuration for query-specific hierarchical feature selection.

    ``filtering_paras_by_modality`` maps each modality to the parameters used
    by ``select_hierarchical_genes_pipeline``. Every modality mapping must at
    least contain ``label_key``. ``anchor_scenario`` is ``"nn_based"`` for
    single/multi-reference nearest-neighbor transfer and ``"quantile_based"``
    for quantile transfer.
    """

    anchor_scenario: str
    filtering_paras_by_modality: Dict[str, Dict[str, Any]]
    output_dir: Path | str | None = None
    count_num: int = 1
    strict: bool = False
    keep_raw_results: bool = True


@dataclass
class HierarchicalFeatureStageResult:
    """Stage-4 output organized by query and modality."""

    feature_results_by_query: Dict[str, Dict[str, Any]]
    multimodal_results_by_query: Dict[str, Any]
    selected_refs_dic: Dict[str, list]
    params: Dict[str, Any] = field(default_factory=dict)

    def get_modality_result(self, query_section, modality):
        return self.feature_results_by_query[query_section][modality]

    def get_multimodal_result(self, query_section):
        return self.multimodal_results_by_query[query_section]


def _default_selected_references(ref_adata_by_modality):
    first_modality = next(iter(ref_adata_by_modality), None)
    if first_modality is None:
        raise ValueError("ref_adata_by_modality is empty.")
    return {"all_queries": list(ref_adata_by_modality[first_modality])}


@logged_stage(
    "hierarchical_features",
    stage_output_from_config("results/04_hierarchical_features", config_position=2),
)
def run_hierarchical_feature_stage(
    ref_adata_by_modality: Mapping[str, Mapping[str, Any]],
    hier_tree,
    config: HierarchicalFeatureStageConfig,
    selected_refs_dic: Optional[Mapping[str, list]] = None,
):
    """Run stage 4 for every query-specific selected-reference set.

    Parameters
    ----------
    ref_adata_by_modality
        ``{modality: {reference_section: AnnData}}``.
    hier_tree
        Tree returned by stage 2.
    config
        Stage configuration.
    selected_refs_dic
        Stage-3 mapping ``{query_section: [reference sections]}``. If omitted,
        one result named ``"all_queries"`` is constructed using every section.
    """
    from ..hier_feature_selection import (
        construct_multimodal_hierarchical_feature_results,
        select_hierarchical_genes_pipeline,
    )

    if config.anchor_scenario not in {"nn_based", "quantile_based"}:
        raise ValueError("anchor_scenario must be 'nn_based' or 'quantile_based'.")
    if not config.filtering_paras_by_modality:
        raise ValueError("filtering_paras_by_modality cannot be empty.")

    selected_refs_dic = dict(
        selected_refs_dic or _default_selected_references(ref_adata_by_modality)
    )
    output_dir = ensure_output_dir(
        config.output_dir or "results/04_hierarchical_features"
    )
    by_query = {}
    multimodal_by_query = {}
    summary_rows = []

    for query_section, reference_sections in selected_refs_dic.items():
        reference_sections = list(reference_sections)
        if not reference_sections:
            raise ValueError(f"No selected references for query {query_section!r}.")
        modality_results = {}

        for modality, filtering_paras in config.filtering_paras_by_modality.items():
            if modality not in ref_adata_by_modality:
                if config.strict:
                    raise KeyError(f"No reference data supplied for {modality!r}.")
                continue
            missing = [
                section
                for section in reference_sections
                if section not in ref_adata_by_modality[modality]
            ]
            if missing:
                raise KeyError(
                    f"Query {query_section!r}, modality {modality!r} is missing "
                    f"selected references: {missing}."
                )
            modality_ref_dic = {
                section: ref_adata_by_modality[modality][section]
                for section in reference_sections
            }
            modality_result = select_hierarchical_genes_pipeline(
                ref_adata_dic=modality_ref_dic,
                hier_tree=hier_tree,
                anchor_scenario=config.anchor_scenario,
                filtering_paras=dict(filtering_paras),
                ref_section_list=reference_sections,
                count_num=config.count_num,
                strict=config.strict,
                keep_raw_results=config.keep_raw_results,
            )
            modality_results[modality] = modality_result
            for parent_node in modality_result.available_parent_nodes():
                summary_rows.append(
                    {
                        "query_section": query_section,
                        "modality": modality,
                        "parent_node": parent_node,
                        "n_clustering_features": len(
                            modality_result.get_clustering_features(parent_node)
                        ),
                        "reference_sections": ";".join(reference_sections),
                    }
                )

        if not modality_results:
            raise ValueError(
                f"No modality results were generated for {query_section!r}."
            )
        by_query[query_section] = modality_results
        multimodal_by_query[
            query_section
        ] = construct_multimodal_hierarchical_feature_results(
            modality_results_dic=modality_results,
            ref_section_list=reference_sections,
            strict=config.strict,
        )

    config_record = asdict(config)
    config_record["output_dir"] = str(output_dir)
    result = HierarchicalFeatureStageResult(
        feature_results_by_query=by_query,
        multimodal_results_by_query=multimodal_by_query,
        selected_refs_dic={
            key: list(value) for key, value in selected_refs_dic.items()
        },
        params=config_record,
    )
    save_stage_result(result, output_dir / "hierarchical_feature_result.pkl")
    save_json(config_record, output_dir / "stage_config.json")
    save_json(result.selected_refs_dic, output_dir / "selected_references.json")
    pd.DataFrame(summary_rows).to_csv(output_dir / "feature_summary.csv", index=False)
    return result


__all__ = [
    "HierarchicalFeatureStageConfig",
    "HierarchicalFeatureStageResult",
    "run_hierarchical_feature_stage",
]
