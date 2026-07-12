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

SHARED_REFERENCE_KEY = "shared_reference"
LEGACY_SHARED_REFERENCE_KEYS = ("all_queries",)
_SHARED_REFERENCE_KEYS = (SHARED_REFERENCE_KEY, *LEGACY_SHARED_REFERENCE_KEYS)


@dataclass
class HierarchicalFeatureStageConfig:
    """Configuration for query-specific hierarchical feature selection.

    Parameters
    ----------
    anchor_scenario : {"nn_based", "quantile_based"}
        Feature format required by the intended Stage-6 anchor framework.
    filtering_paras_by_modality : dict[str, dict]
        Modality-keyed feature-selection settings. Keys use exact modality
        names (``"Gene"``, ``"Image"``, ``"Protein"``), and every inner
        mapping must include ``label_key``. Common inner keys are
        ``pvals_adj``, ``min_fold_change``, ``min_in_out_group_ratio``,
        ``min_in_group_fraction``, ``gene_num``, ``two_sides``, ``logged``,
        ``verbose``, and ``split_order``.

        Example: ``{"Gene": {"label_key": "label", "gene_num": 20}}``.
    output_dir : path-like or None, default=None
        Stage output directory. ``None`` uses
        ``results/04_hierarchical_features``.
    count_num : int, default=1
        Minimum number of selected reference sections in which a feature must
        appear when aggregating NN-based features.
    strict : bool, default=False
        Raise for unavailable modalities/features instead of skipping them.
    keep_raw_results : bool, default=True
        Retain lower-level feature-selection results for inspection.
    make_image_nonnegative : bool, default=True
        Shift Image feature columns to be non-negative before hierarchical
        feature selection. This is useful for HIPT/UNI embeddings because
        their dimensions can contain negative values, while the current
        rank/filtering logic expects non-negative feature values. The shift is
        applied to copied AnnData objects and does not mutate the input
        ``ref_adata_by_modality``.
    """

    anchor_scenario: str
    filtering_paras_by_modality: Dict[str, Dict[str, Any]]
    output_dir: Path | str | None = None
    count_num: int = 1
    strict: bool = False
    keep_raw_results: bool = True
    make_image_nonnegative: bool = True


@dataclass
class HierarchicalFeatureStageResult:
    """Stage-4 output organized by query and modality.

    Attributes
    ----------
    feature_results_by_query : dict[str, dict[str, HierarchicalFeatureResults]]
        ``{query_section: {modality: result}}``.
    multimodal_results_by_query : dict[str, MultimodalHierarchicalFeatureResults]
        Combined modality result for every query section.
    selected_refs_dic : dict[str, list[str]]
        Reference sections used for each query.
    params : dict
        Serialized stage configuration.
    """

    feature_results_by_query: Dict[str, Dict[str, Any]]
    multimodal_results_by_query: Dict[str, Any]
    selected_refs_dic: Dict[str, list]
    params: Dict[str, Any] = field(default_factory=dict)

    def _resolve_result_key(self, query_section):
        if query_section in self.feature_results_by_query:
            return query_section
        for shared_key in _SHARED_REFERENCE_KEYS:
            if shared_key in self.feature_results_by_query:
                return shared_key
        if len(self.feature_results_by_query) == 1:
            return next(iter(self.feature_results_by_query))
        raise KeyError(
            f"No hierarchical feature result exists for {query_section!r}. "
            f"Available keys: {list(self.feature_results_by_query)}."
        )

    def get_modality_result(self, query_section, modality):
        result_key = self._resolve_result_key(query_section)
        return self.feature_results_by_query[result_key][modality]

    def get_multimodal_result(self, query_section):
        result_key = self._resolve_result_key(query_section)
        return self.multimodal_results_by_query[result_key]


def _default_selected_references(ref_adata_by_modality):
    first_modality = next(iter(ref_adata_by_modality), None)
    if first_modality is None:
        raise ValueError("ref_adata_by_modality is empty.")
    return {SHARED_REFERENCE_KEY: list(ref_adata_by_modality[first_modality])}


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
    """Run Stage 4 for every selected-reference set.

    Parameters
    ----------
    ref_adata_by_modality : mapping[str, mapping[str, AnnData]]
        Modality-first reference data:
        ``{modality: {reference_section: AnnData}}``. Exact modality names are
        required. Every requested object must contain the configured label
        column and features appropriate to that modality.
    hier_tree : HierTree
        ``stage2_result["tree"]``. It must provide internal binary splits and
        region membership.
    config : HierarchicalFeatureStageConfig
        Anchor format, per-modality filters, aggregation, and output settings.
    selected_refs_dic : mapping[str, sequence[str]] or None, default=None
        Stage-3 mapping ``{query_section: [reference_section, ...]}``, or a
        shared mapping such as ``{"shared_reference": reference_sections}``.
        Every referenced section must exist for every requested modality.
        ``None`` creates one ``"shared_reference"`` result using all reference
        sections.

    Returns
    -------
    HierarchicalFeatureStageResult
        Query/modality feature results, combined multimodal results, selected
        references, and configuration metadata.

    Saved files
    -----------
    ``hierarchical_feature_result.pkl``, ``stage_config.json``,
    ``selected_references.json``, and ``feature_summary.csv``.
    """
    from ..hier_feature_selection import (
        construct_multimodal_hierarchical_feature_results,
        select_hierarchical_genes_pipeline,
    )
    from ..preprocessing.preprocess_util import make_nonnegative_adata

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
            if modality == "Image" and config.make_image_nonnegative:
                modality_ref_dic = {
                    section: make_nonnegative_adata(adata, copy=True)
                    for section, adata in modality_ref_dic.items()
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
    "SHARED_REFERENCE_KEY",
    "run_hierarchical_feature_stage",
]
