"""Stage 5: determine informative modalities and embedding configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import pandas as pd

from ._io import (
    ensure_output_dir,
    logged_stage,
    save_json,
    save_stage_result,
    stage_output_from_config,
)


@dataclass
class ClusteringConfigStageConfig:
    """Configuration for multi-modal embedding evaluation.

    Parameters
    ----------
    included_modalities : sequence[str]
        Candidate exact modality names from ``"Gene"``, ``"Image"``, and
        ``"Protein"``. Data and Stage-4 features must exist for each included
        modality.
    output_dir : path-like or None, default=None
        Stage output directory. ``None`` uses
        ``results/05_clustering_config``.
    features_format : {"auto", "section", "modality"}, default="auto"
        Stage-4 feature nesting supplied to the evaluator. ``"auto"`` uses
        section-first features for NN anchors and modality-level features for
        quantile anchors.
    evaluate_all_nodes : bool, default=False
        Evaluate only the hierarchy root by default. ``True`` evaluates every
        available internal parent node for manual inspection.
    label_key : str, default="label"
        Reference ``.obs`` column used as ground truth for ARI evaluation.
    parameters : dict, default={}
        Extra keywords for ``determine_multi_modal_embedding_config``. Common
        keys include ``hard_threshold``, ``alpha``, ``selection_criterion``,
        ``pcs_num_dic``, ``default_pcs_num``, ``candidate_methods``,
        ``min_spots``, ``random_state``, and ``visualization_config``.
    """

    included_modalities: Sequence[str]
    output_dir: Path | str | None = None
    features_format: str = "auto"
    evaluate_all_nodes: bool = False
    label_key: str = "label"
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClusteringConfigStageResult:
    """Embedding-configuration results keyed by query and parent node.

    Attributes
    ----------
    results_by_query : dict[str, dict[str, MultiModalClusteringConfigResult]]
        ``{query_section: {parent_node: result}}``.
    evaluated_nodes_by_query : dict[str, list[str]]
        Ordered parent nodes evaluated for every query.
    params : dict
        Serialized stage configuration.

    Notes
    -----
    The stage selects modalities and dimension reduction but not the final
    KMeans/Leiden choice. Call ``to_clustering_config(query_section, ...)`` to
    create the plain dictionary required by Stage 6.
    """

    results_by_query: Dict[str, Dict[str, Any]]
    evaluated_nodes_by_query: Dict[str, list]
    params: Dict[str, Any] = field(default_factory=dict)

    def get_result(self, query_section, parent_node=None):
        query_results = self.results_by_query[query_section]
        if parent_node is None:
            parent_node = self.evaluated_nodes_by_query[query_section][0]
        return query_results[parent_node]

    def to_clustering_config(self, query_section, parent_node=None, **kwargs):
        """Add user-selected KMeans/Leiden settings to an automatic result."""
        return self.get_result(query_section, parent_node).to_clustering_config(
            **kwargs
        )


def _subset_sections(adata_dic, sections, modality):
    if adata_dic is None:
        return None
    missing = [section for section in sections if section not in adata_dic]
    if missing:
        raise KeyError(f"{modality} reference data are missing sections: {missing}")
    return {section: adata_dic[section] for section in sections}


@logged_stage(
    "clustering_config",
    stage_output_from_config("results/05_clustering_config", config_position=2),
)
def run_clustering_config_stage(
    ref_adata_by_modality: Mapping[str, Mapping[str, Any]],
    feature_stage_result,
    config: ClusteringConfigStageConfig,
):
    """Run Stage 5 for each query's selected references.

    The automatic result chooses modalities and PCA versus selected features.
    It deliberately does not choose KMeans versus Leiden; call
    ``stage_result.to_clustering_config(...)`` afterward.

    Parameters
    ----------
    ref_adata_by_modality : mapping[str, mapping[str, AnnData]]
        Modality-first reference data:
        ``{modality: {reference_section: AnnData}}``. Observation names must
        be unique and modalities within a section must be row-aligned or
        alignable by ``obs_names``.
    feature_stage_result : HierarchicalFeatureStageResult
        Stage-4 result providing selected references and hierarchical features
        for every query.
    config : ClusteringConfigStageConfig
        Candidate modalities, feature format, evaluation scope, and forwarded
        embedding-selection parameters.

    Returns
    -------
    ClusteringConfigStageResult
        Automatic modality/dimension-reduction decisions nested by query and
        parent node.

    Examples
    --------
    Convert one result into a final Stage-6 configuration::

        clustering_config = result.to_clustering_config(
            "query_1",
            clustering_method="leiden",
            resolution=0.5,
            n_neighbors=15,
        )

    Saved files
    -----------
    ``clustering_config_stage_result.pkl``, ``stage_config.json``, the global
    summary CSV, and per-query/node result pickle and ARI CSV files. Optional
    plots are stored under each node directory.
    """
    from ..determine_clustering_config import determine_multi_modal_embedding_config

    if config.features_format not in {"auto", "section", "modality"}:
        raise ValueError("features_format must be 'auto', 'section', or 'modality'.")
    output_dir = ensure_output_dir(config.output_dir or "results/05_clustering_config")
    results_by_query = {}
    nodes_by_query = {}
    summary_rows = []

    for (
        query_section,
        multimodal_result,
    ) in feature_stage_result.multimodal_results_by_query.items():
        included_modalities = list(config.included_modalities)
        reference_sections = feature_stage_result.selected_refs_dic[query_section]
        modality_results = feature_stage_result.feature_results_by_query[query_section]
        first_modality_result = next(iter(modality_results.values()))
        available_nodes = multimodal_result.available_parent_nodes()
        root_node = first_modality_result.root_node
        if config.evaluate_all_nodes:
            target_nodes = available_nodes
        elif root_node in available_nodes:
            target_nodes = [root_node]
        else:
            target_nodes = available_nodes[:1]
        if not target_nodes:
            raise ValueError(f"No parent nodes available for query {query_section!r}.")

        results_by_query[query_section] = {}
        nodes_by_query[query_section] = list(target_nodes)
        for parent_node in target_nodes:
            if config.features_format == "auto":
                features_format = (
                    "section"
                    if first_modality_result.anchor_scenario == "nn_based"
                    else "modality"
                )
            else:
                features_format = config.features_format
            features_dic = multimodal_result.get_features_dic(
                parent_node=parent_node,
                output_format=features_format,
            )

            node_dir = ensure_output_dir(
                output_dir / str(query_section) / str(parent_node)
            )
            parameters = dict(config.parameters)
            visualization = parameters.get("visualization_config")
            if visualization is not None:
                visualization = dict(visualization)
                if visualization.get("plot_modality_clusters") or visualization.get(
                    "plot_dim_reduction_clusters"
                ):
                    visualization.setdefault("output_dir", str(node_dir / "plots"))
                parameters["visualization_config"] = visualization

            result = determine_multi_modal_embedding_config(
                included_modalities=included_modalities,
                ref_section_list=list(reference_sections),
                ref_gene_dic=_subset_sections(
                    ref_adata_by_modality.get("Gene")
                    if "Gene" in included_modalities
                    else None,
                    reference_sections,
                    "Gene",
                ),
                ref_image_dic=_subset_sections(
                    ref_adata_by_modality.get("Image")
                    if "Image" in included_modalities
                    else None,
                    reference_sections,
                    "Image",
                ),
                ref_protein_dic=_subset_sections(
                    ref_adata_by_modality.get("Protein")
                    if "Protein" in included_modalities
                    else None,
                    reference_sections,
                    "Protein",
                ),
                features_dic=features_dic,
                features_format=features_format,
                label_key=config.label_key,
                **parameters,
            )
            results_by_query[query_section][parent_node] = result
            save_stage_result(result, node_dir / "clustering_config_result.pkl")
            result.modality_ari_df.to_csv(node_dir / "modality_ari.csv", index=False)
            result.dim_reduction_summary_df.to_csv(
                node_dir / "dimension_reduction_summary.csv", index=False
            )
            summary_rows.append(
                {
                    "query_section": query_section,
                    "parent_node": parent_node,
                    "selected_modalities": ";".join(result.selected_modalities),
                    "dim_reduction_method": result.dim_reduction_method,
                    "selected_modality_average_ari": result.selected_modality_average_ari,
                    "best_dim_reduction_average_ari": result.best_dim_reduction_average_ari,
                }
            )

    config_record = asdict(config)
    config_record["output_dir"] = str(output_dir)
    stage_result = ClusteringConfigStageResult(
        results_by_query=results_by_query,
        evaluated_nodes_by_query=nodes_by_query,
        params=config_record,
    )
    save_stage_result(stage_result, output_dir / "clustering_config_stage_result.pkl")
    save_json(config_record, output_dir / "stage_config.json")
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "clustering_config_summary.csv", index=False
    )
    return stage_result


__all__ = [
    "ClusteringConfigStageConfig",
    "ClusteringConfigStageResult",
    "run_clustering_config_stage",
]
