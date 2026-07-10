"""Hierarchical multimodal label-transfer workflows.

Reference input formats
-----------------------
Single-reference NN uses a modality-first dictionary::

    {"Gene": ref_gene_sca, "Protein": ref_protein_sca}

Multi-reference NN uses a section-first dictionary::

    {
        "ref_1": {"Gene": ref1_gene_sca, "Protein": ref1_protein_sca},
        "ref_2": {"Gene": ref2_gene_sca, "Protein": ref2_protein_sca},
    }

Quantile transfer uses a modality-first, then section-first dictionary::

    {
        "Gene": {"ref_1": ref1_gene_sca, "ref_2": ref2_gene_sca},
        "Protein": {"ref_1": ref1_protein_sca, "ref_2": ref2_protein_sca},
    }

Multi-reference NN reference guides
-----------------------------------
By default, ``reference_section_guide=None`` uses every section in
``ref_section_list`` for every internal parent-node split. A user may instead
provide one node-specific guide for the current query::

    {
        "node_0": ["ref_1", "ref_2"],
        "node_1": ["ref_2"],
    }

Guide keys are internal parent nodes and values are subsets of
``ref_section_list``. Missing node keys use the complete reference list. An
explicit empty list means no reference is eligible and stops that branch. For
multiple queries, keep a query-first outer dictionary in user code and pass
``reference_section_guide[qry_section]`` to each function call.

For every workflow, query inputs are modality-first dictionaries such as
``{"Gene": gene_adata, "Image": image_adata}``. Required query modalities
must have matching, unique ``obs_names``.

Output formats
--------------
``mode="auto"`` returns a :class:`HierarchicalTransferResult` subclass.
Final labels are available in ``result.final_labels`` and in
``result.query_adata_dic[modality].obs[final_label_key]``. Per-node details
are stored in ``result.round_results[parent_node]``.

``mode="manual"`` returns a session. Its ``run_round`` method returns a
:class:`HierarchyRoundResult`; set ``commit=True`` to update session state,
then call ``session.to_result()`` to materialize the current output.
"""

from __future__ import annotations

from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass, field
import inspect
import pandas as pd
from typing import Any, Dict, List, Mapping, Optional

from .hier_feature_selection import (
    construct_multimodal_hierarchical_feature_results,
    normalize_nn_reference_section_guide,
)

from .query_clustering import (
    QueryClusteringResult,
    query_multi_modal_clustering,
    postprocess_query_clustering_result,
    )

from .anchor_detection import (
    AnchorDetectionResult,
    nn_based_anchor_detection_single_ref_multimodal,
    nn_based_anchor_detection_multiref_multimodal,
    quantile_based_anchor_detection_multimodal,
    )

from .label_assignment import (
    LabelAssignmentResult,
    assign_hierarchical_labels,
    adjust_one_side_binary_assignment,
    refine_labels,
    )

from .visualization import (
    cat_figure,
    )


@dataclass
class HierarchyRoundResult:
    """Output of one binary hierarchy split.

    Attributes
    ----------
    parent_node, child_nodes
        Parent being split and its two child-node names.
    obs_names
        Query observation names processed in this round.
    features_dic, anchor_features_dic
        Features used for clustering and anchor detection, respectively.
    clustering_result, anchor_result, assignment_result
        Detailed results from the three round stages.
    ref_section_list
        Reference sections actually used for this hierarchy round.
    child_obs_names
        Assigned query observation names keyed by child node.
    unresolved_obs_names
        Observations not assigned to either child.
    clustering_config, anchor_config, assignment_config
        Effective settings used for this round.
    skipped, skip_reason
        Whether the round was skipped and, if so, why.
    """

    parent_node: str
    child_nodes: List[str]
    obs_names: List[str]
    features_dic: Dict[str, List[str]] = field(default_factory=dict)
    anchor_features_dic: Dict[str, Any] = field(default_factory=dict)
    clustering_result: Optional[QueryClusteringResult] = None
    anchor_result: Optional[AnchorDetectionResult] = None
    assignment_result: Optional[LabelAssignmentResult] = None
    child_obs_names: Dict[str, List[str]] = field(default_factory=dict)
    unresolved_obs_names: List[str] = field(default_factory=list)
    clustering_config: Dict[str, Any] = field(default_factory=dict)
    anchor_config: Dict[str, Any] = field(default_factory=dict)
    assignment_config: Dict[str, Any] = field(default_factory=dict)
    ref_section_list: List[str] = field(default_factory=list)
    skipped: bool = False
    skip_reason: Optional[str] = None

    def summary(self) -> Dict[str, Any]:
        """Return compact counts and status for this round."""
        return {
            "parent_node": self.parent_node,
            "child_nodes": list(self.child_nodes),
            "n_obs": len(self.obs_names),
            "child_counts": {
                node: len(names) for node, names in self.child_obs_names.items()
            },
            "n_unresolved": len(self.unresolved_obs_names),
            "ref_section_list": list(self.ref_section_list),
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
        }


@dataclass
class HierarchicalTransferResult:
    """Final or partially completed hierarchical transfer output.

    Attributes
    ----------
    query_adata_dic
        Query modality dictionary with ``obs[final_label_key]`` added.
    final_labels
        Label series indexed by query observation name.
    node_obs_names
        Observation names currently associated with each hierarchy node.
    round_results
        Committed :class:`HierarchyRoundResult` objects keyed by parent node.
    terminal_reasons
        Reason each stopped node was not split further.
    pending_nodes
        Internal nodes that can still be processed; empty means complete.
    params
        Run metadata and top-level settings.
    """

    query_adata_dic: Dict[str, Any]
    final_labels: pd.Series
    node_obs_names: Dict[str, List[str]]
    round_results: Dict[str, HierarchyRoundResult]
    terminal_reasons: Dict[str, str] = field(default_factory=dict)
    pending_nodes: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        return len(self.pending_nodes) == 0

    def get_round(self, parent_node: str) -> HierarchyRoundResult:
        """Return the committed round for ``parent_node``."""
        if parent_node not in self.round_results:
            raise KeyError(
                f"No committed round exists for {parent_node!r}. "
                f"Available nodes: {list(self.round_results)}"
            )
        return self.round_results[parent_node]

    def round_summary(self) -> pd.DataFrame:
        """Return one summary row per committed hierarchy round."""
        return pd.DataFrame(
            [result.summary() for result in self.round_results.values()]
        )


@dataclass
class SingleReferenceNNTransferResult(HierarchicalTransferResult):
    """Single-reference NN transfer output."""


@dataclass
class MultiReferenceNNTransferResult(SingleReferenceNNTransferResult):
    """Final or partially completed multi-reference NN transfer output."""


@dataclass
class QuantileBasedTransferResult(HierarchicalTransferResult):
    """Quantile-based transfer output."""


def _filter_kwargs(func, params: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if params is None:
        return {}
    valid_keys = set(inspect.signature(func).parameters)
    return {key: value for key, value in params.items() if key in valid_keys}


def _merge_config(
    base_config: Optional[Mapping[str, Any]],
    overrides: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    merged = deepcopy(dict(base_config or {}))
    if overrides:
        merged.update(deepcopy(dict(overrides)))
    return merged


def _copy_adata_dic(adata_dic: Mapping[str, Any], copy_values: bool) -> Dict[str, Any]:
    if adata_dic is None or len(adata_dic) == 0:
        raise ValueError("AnnData dictionary cannot be None or empty.")
    return {
        modality: adata.copy() if copy_values else adata
        for modality, adata in adata_dic.items()
    }


def _ordered_common_obs(
    query_adata_dic: Mapping[str, Any],
    query_adata_sca_dic: Mapping[str, Any],
    clustering_modalities: List[str],
    anchor_modalities: List[str],
) -> pd.Index:
    base_modality = "Gene" if "Gene" in clustering_modalities else clustering_modalities[0]
    if base_modality not in query_adata_dic:
        raise KeyError(f"query_adata_dic is missing modality {base_modality!r}.")

    base_obs = query_adata_dic[base_modality].obs_names
    if not base_obs.is_unique:
        raise ValueError(f"{base_modality} query obs_names must be unique.")

    required_objects = []
    for modality in clustering_modalities:
        if modality not in query_adata_dic:
            raise KeyError(f"query_adata_dic is missing selected modality {modality!r}.")
        required_objects.append(query_adata_dic[modality])

    for modality in anchor_modalities:
        if modality not in query_adata_sca_dic:
            raise KeyError(
                f"query_adata_sca_dic is missing anchor modality {modality!r}."
            )
        required_objects.append(query_adata_sca_dic[modality])

    common_obs = base_obs
    for adata in required_objects:
        if not adata.obs_names.is_unique:
            raise ValueError("All query modality obs_names must be unique.")
        common_obs = common_obs.intersection(adata.obs_names, sort=False)

    common_obs = base_obs[base_obs.isin(common_obs)]
    if len(common_obs) == 0:
        raise ValueError("No shared query observations remain across required modalities.")
    return common_obs


def _subset_adata_dic(
    adata_dic: Mapping[str, Any],
    obs_names: List[str],
    modalities: List[str],
) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    requested_obs = pd.Index(obs_names)

    for modality in modalities:
        if modality not in adata_dic:
            raise KeyError(f"Missing modality {modality!r} while subsetting query data.")
        missing = requested_obs.difference(adata_dic[modality].obs_names)
        if len(missing) > 0:
            raise ValueError(
                f"Modality {modality!r} is missing active observations. "
                f"Examples: {missing[:5].tolist()}"
            )
        output[modality] = adata_dic[modality][obs_names, :].copy()

    return output


def _normalize_enabled_config(config: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return a config copy with the common ``enables`` typo normalized."""
    if config is None:
        return None
    normalized = dict(config)
    if "enabled" not in normalized and "enables" in normalized:
        normalized["enabled"] = normalized["enables"]
    normalized.pop("enables", None)
    return normalized


def _config_is_enabled(config: Optional[Mapping[str, Any]]) -> bool:
    """Whether an optional config should be applied."""
    if config is None:
        return False
    return bool(config.get("enabled", True))


def _subset_query_dic_for_postprocessing(
    *,
    local_query_dic: Mapping[str, Any],
    full_query_dic: Mapping[str, Any],
    obs_names: List[str],
    boundary_refinement_config: Optional[Mapping[str, Any]],
    gene_subtyping_config: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Add modalities required by optional postprocessing without changing clustering.

    ``local_query_dic`` contains only modalities selected for clustering. Optional
    postprocessing may need additional modalities: Image for HIPT boundary
    refinement and Gene for gene-based subtyping. Those additional modalities are
    subset here only for postprocessing/assignment support.
    """
    output = dict(local_query_dic)
    requested_obs = pd.Index(obs_names)

    required_modalities = []
    if _config_is_enabled(boundary_refinement_config):
        required_modalities.append(("Image", "boundary_refinement_config"))
    if _config_is_enabled(gene_subtyping_config):
        required_modalities.append(("Gene", "gene_subtyping_config"))

    for modality, reason in required_modalities:
        if modality in output:
            continue
        if modality not in full_query_dic:
            raise KeyError(f"{reason} requires query_adata_dic[{modality!r}].")
        missing = requested_obs.difference(full_query_dic[modality].obs_names)
        if len(missing) > 0:
            raise ValueError(
                f"{reason} requires query_adata_dic[{modality!r}] to contain "
                "all active observations. "
                f"Missing examples: {missing[:5].tolist()}"
            )
        output[modality] = full_query_dic[modality][obs_names, :].copy()

    return output


def _build_assignment_adata(
    local_scaled_dic: Mapping[str, Any],
    local_query_dic: Mapping[str, Any],
    clustering_result: QueryClusteringResult,
    anchor_result: AnchorDetectionResult,
    cluster_key: str,
    x_key: str,
    y_key: str,
):
    base_modality = "Gene" if "Gene" in local_scaled_dic else next(iter(local_scaled_dic))
    # Hierarchical assignment uses observation metadata only.
    assignment_adata = local_scaled_dic[base_modality][:, :0].copy()

    assignment_adata.obs[cluster_key] = pd.Categorical(
        clustering_result.labels.reindex(assignment_adata.obs_names).astype(str)
    )

    for column in anchor_result.anchor_df.columns:
        assignment_adata.obs[column] = anchor_result.anchor_df.reindex(
            assignment_adata.obs_names
        )[column].to_numpy()

    coordinate_sources = [base_modality] + [
        modality for modality in local_query_dic if modality != base_modality
    ]
    for coordinate in (x_key, y_key):
        if coordinate in assignment_adata.obs:
            continue
        for modality in coordinate_sources:
            raw_base = local_query_dic.get(modality)
            if raw_base is None or coordinate not in raw_base.obs:
                continue
            assignment_adata.obs[coordinate] = raw_base.obs.reindex(
                assignment_adata.obs_names
            )[coordinate]
            break

    return assignment_adata


class HierarchicalTransferSession:
    """Shared stateful engine for manual or automatic hierarchy traversal.

    Subclasses customize feature preparation, anchor detection, reference
    metadata, and optional one-sided adjustment. Clustering, assignment,
    round commits, descendant invalidation, and recursive traversal remain
    identical across transfer scenarios. Create a scenario-specific session
    through a label-transfer entry point with ``mode="manual"``.
    """

    result_class = HierarchicalTransferResult
    reference_scenario = "base"

    def __init__(
        self,
        ref_adata_sca_dic,
        query_adata_dic,
        query_adata_sca_dic,
        ref_section,
        qry_section,
        hier_tree,
        target_parent_node=None,
        qry_nodes_dic=None,
        gene_feature_results=None,
        image_feature_results=None,
        protein_feature_results=None,
        clustering_config=None,
        boundary_refinement_config=None,
        gene_subtyping_config=None,
        anchor_config=None,
        assignment_config=None,
        fig_paras=None,
        label_key="label",
        cluster_key="query_cluster",
        output_dir=None,
        min_node_prop=0.05,
        min_node_spots=2,
        final_label_key="hicat_label",
        unassigned_label="novel_cluster",
        copy=True,
        print_results=True,
    ):
        if clustering_config is None:
            raise ValueError("clustering_config must be provided.")
        if not 0 <= min_node_prop <= 1:
            raise ValueError("min_node_prop must be between 0 and 1.")
        if min_node_spots < 1:
            raise ValueError("min_node_spots must be at least 1.")

        self.ref_adata_sca_dic = ref_adata_sca_dic
        self.query_adata_dic = _copy_adata_dic(query_adata_dic, copy_values=copy)
        self.query_adata_sca_dic = query_adata_sca_dic
        self.ref_section = ref_section
        self.qry_section = qry_section
        self.hier_tree = hier_tree
        self.start_node = target_parent_node or hier_tree.root_node
        self.feature_results_dic = {
            "Gene": gene_feature_results,
            "Image": image_feature_results,
            "Protein": protein_feature_results,
        }
        available_feature_results = {
            modality: result
            for modality, result in self.feature_results_dic.items()
            if result is not None
        }
        if len(available_feature_results) == 0:
            raise ValueError(
                "At least one of gene_feature_results, image_feature_results, "
                "or protein_feature_results must be provided."
            )
        feature_ref_sections = list(
            getattr(self, "ref_section_list", [self.ref_section])
        )
        self.multimodal_feature_results = (
            construct_multimodal_hierarchical_feature_results(
                modality_results_dic=available_feature_results,
                ref_section_list=feature_ref_sections,
                strict=False,
            )
        )
        self.clustering_config = deepcopy(dict(clustering_config))
        self.boundary_refinement_config = deepcopy(boundary_refinement_config)
        self.gene_subtyping_config = deepcopy(gene_subtyping_config)
        self.anchor_config = deepcopy(dict(anchor_config or {}))
        self.assignment_config = deepcopy(dict(assignment_config or {}))
        self.fig_paras = deepcopy(fig_paras)
        self.label_key = label_key
        self.cluster_key = cluster_key
        self.output_dir = output_dir
        self.min_node_prop = min_node_prop
        self.min_node_spots = min_node_spots
        self.final_label_key = final_label_key
        self.unassigned_label = unassigned_label
        self.print_results = print_results

        if self.start_node not in hier_tree.node_dic:
            raise KeyError(f"Unknown target_parent_node: {self.start_node!r}.")

        selected_modalities = self._selected_modalities(self.clustering_config)
        base_anchor_modalities = self._anchor_modalities(
            self.anchor_config,
            selected_modalities,
        )
        common_obs = _ordered_common_obs(
            query_adata_dic=self.query_adata_dic,
            query_adata_sca_dic=self.query_adata_sca_dic,
            clustering_modalities=selected_modalities,
            anchor_modalities=base_anchor_modalities,
        )

        base_modality = "Gene" if "Gene" in selected_modalities else selected_modalities[0]
        base_obs_names = self.query_adata_dic[base_modality].obs_names.copy()
        self.final_labels = pd.Series(
            unassigned_label,
            index=base_obs_names,
            name=final_label_key,
            dtype=object,
        )

        self.node_obs_names: Dict[str, List[str]] = {}
        if qry_nodes_dic:
            for node, value in qry_nodes_dic.items():
                if hasattr(value, "obs_names"):
                    self.node_obs_names[node] = value.obs_names.tolist()
                else:
                    self.node_obs_names[node] = list(value)

        self.node_obs_names.setdefault(self.start_node, common_obs.tolist())
        self.round_results: Dict[str, HierarchyRoundResult] = {}
        self.terminal_reasons: Dict[str, str] = {}

    @staticmethod
    def _selected_modalities(clustering_config: Mapping[str, Any]) -> List[str]:
        selected = list(
            clustering_config.get(
                "selected_modalities",
                clustering_config.get("informative_modalities", []),
            )
        )
        if len(selected) == 0:
            raise ValueError("clustering_config must specify selected_modalities.")
        return selected

    @staticmethod
    def _anchor_modalities(
        anchor_config: Mapping[str, Any],
        selected_modalities: List[str],
    ) -> List[str]:
        modalities = list(
            anchor_config.get(
                "modalities",
                [m for m in selected_modalities if m in {"Gene", "Protein"}],
            )
        )
        if len(modalities) == 0:
            raise ValueError(
                "Label transfer requires at least one molecular anchor modality "
                "('Gene' or 'Protein')."
            )
        return modalities

    def _output_node_label(self, node: str) -> str:
        if self.hier_tree.is_leaf(node):
            regions = self.hier_tree.get_regions(node)
            return str(regions[0]) if len(regions) == 1 else str(node)
        return str(node)

    def _mark_terminal(self, node: str, obs_names: List[str], reason: str) -> None:
        self.terminal_reasons[node] = reason
        if obs_names:
            self.final_labels.loc[obs_names] = self._output_node_label(node)

    def _subtree_nodes(self, parent_node: str) -> List[str]:
        nodes = [parent_node]
        if not self.hier_tree.is_leaf(parent_node):
            for child in self.hier_tree.get_children(parent_node):
                nodes.extend(self._subtree_nodes(child))
        return nodes

    def _invalidate_subtree(self, parent_node: str) -> None:
        """Remove committed descendants before replacing a parent round."""
        subtree_nodes = self._subtree_nodes(parent_node)
        active_obs = self.node_obs_names.get(parent_node, [])
        if active_obs:
            self.final_labels.loc[active_obs] = self.unassigned_label

        for node in subtree_nodes:
            self.round_results.pop(node, None)
            self.terminal_reasons.pop(node, None)
            if node != parent_node:
                self.node_obs_names.pop(node, None)

    def pending_internal_nodes(self) -> List[str]:
        """Return committed, reachable internal nodes that still need a round."""
        pending = []
        for node, obs_names in self.node_obs_names.items():
            if self.hier_tree.is_leaf(node):
                continue
            if node in self.round_results or node in self.terminal_reasons:
                continue
            if len(obs_names) < max(self.min_node_spots, 2):
                continue
            pending.append(node)
        return pending

    def _build_feature_inputs(
        self,
        parent_node: str,
        child_1: str,
        child_2: str,
        selected_modalities: List[str],
        anchor_modalities: List[str],
        clustering_config: Mapping[str, Any],
    ) -> tuple[Dict[str, List[str]], Dict[str, Any]]:
        """Return clustering and anchor feature dictionaries for one split."""
        del child_1, child_2, clustering_config
        clustering_features = (
            self.multimodal_feature_results.get_clustering_features_dic(
                parent_node=parent_node,
                modalities=selected_modalities,
                ref_section_list=[self.ref_section],
                count_num=1,
                strict=False,
            )
        )
        section_anchor_features = (
            self.multimodal_feature_results.get_nn_anchor_features(
                parent_node=parent_node,
                modalities=anchor_modalities,
                ref_section_list=[self.ref_section],
                strict=True,
            )
        )
        return clustering_features, section_anchor_features[self.ref_section]

    def _missing_anchor_features(
        self,
        anchor_features_dic: Mapping[str, Any],
        anchor_modalities: List[str],
    ) -> List[str]:
        return [
            modality
            for modality in anchor_modalities
            if len(anchor_features_dic.get(modality, [])) == 0
        ]

    def _get_gene_subtyping_features(
        self,
        parent_node: str,
        child_1: str,
        child_2: str,
        count_num: Optional[int] = None,
    ) -> tuple[List[str], List[str]]:
        del count_num
        target_features = (
            self.multimodal_feature_results.get_direction_features_dic(
                parent_node=parent_node,
                direction=f"{child_1}_vs_{child_2}",
                modalities=["Gene"],
                ref_section_list=[self.ref_section],
                count_num=1,
                strict=False,
            )
        )
        nontgt_features = (
            self.multimodal_feature_results.get_direction_features_dic(
                parent_node=parent_node,
                direction=f"{child_2}_vs_{child_1}",
                modalities=["Gene"],
                ref_section_list=[self.ref_section],
                count_num=1,
                strict=False,
            )
        )
        return target_features.get("Gene", []), nontgt_features.get("Gene", [])

    def _detect_anchors(
        self,
        parent_node: str,
        local_scaled_dic: Mapping[str, Any],
        anchor_features_dic: Mapping[str, Any],
        child_1: str,
        child_2: str,
        anchor_modalities: List[str],
        anchor_config: Mapping[str, Any],
    ) -> AnchorDetectionResult:
        del parent_node
        anchor_call_kwargs = _filter_kwargs(
            nn_based_anchor_detection_single_ref_multimodal,
            anchor_config,
        )
        anchor_call_kwargs.update(
            {
                "ref_adata_sca_dic": self.ref_adata_sca_dic,
                "test_adata_sca_dic": local_scaled_dic,
                "features_dic": anchor_features_dic,
                "target_regions": self.hier_tree.get_regions(child_1),
                "nontgt_regions": self.hier_tree.get_regions(child_2),
                "target_node": child_1,
                "nontgt_node": child_2,
                "ref_section": self.ref_section,
                "modalities": anchor_modalities,
                "label_key": self.label_key,
                "copy": True,
                "return_result": True,
                "print_results": self.print_results,
            }
        )
        return nn_based_anchor_detection_single_ref_multimodal(
            **anchor_call_kwargs
        )

    def _gene_reference_for_adjustment(self, parent_node: str):
        """Return the single Gene reference in section-keyed form."""
        del parent_node
        if "Gene" not in self.ref_adata_sca_dic:
            raise KeyError(
                "One-sided adjustment requires ref_adata_sca_dic['Gene']."
            )
        return {self.ref_section: self.ref_adata_sca_dic["Gene"]}

    def _reference_params(self) -> Dict[str, Any]:
        return {
            "reference_scenario": self.reference_scenario,
            "ref_section": self.ref_section,
        }

    def _reference_sections_for_node(self, parent_node: str) -> List[str]:
        """Return reference sections used for one hierarchy round."""
        del parent_node
        return list(getattr(self, "ref_section_list", [self.ref_section]))

    def _compute_round(
        self,
        parent_node: str,
        clustering_config: Mapping[str, Any],
        anchor_config: Mapping[str, Any],
        assignment_config: Mapping[str, Any],
    ) -> HierarchyRoundResult:
        if parent_node not in self.node_obs_names:
            raise KeyError(
                f"No active observations are registered for node {parent_node!r}."
            )
        if self.hier_tree.is_leaf(parent_node):
            raise ValueError(f"Node {parent_node!r} is a leaf and cannot be split.")

        active_obs_names = list(self.node_obs_names[parent_node])
        child_nodes = list(self.hier_tree.get_children(parent_node))
        if len(child_nodes) != 2:
            raise ValueError(
                f"Hierarchy node {parent_node!r} must have exactly two children."
            )

        round_result = HierarchyRoundResult(
            parent_node=parent_node,
            child_nodes=child_nodes,
            obs_names=active_obs_names,
            clustering_config=deepcopy(dict(clustering_config)),
            anchor_config=deepcopy(dict(anchor_config)),
            assignment_config=deepcopy(dict(assignment_config)),
        )
        round_result.ref_section_list = self._reference_sections_for_node(
            parent_node
        )

        if len(active_obs_names) < max(self.min_node_spots, 2):
            round_result.skipped = True
            round_result.skip_reason = "fewer_than_min_node_spots"
            return round_result

        if len(round_result.ref_section_list) == 0:
            round_result.skipped = True
            round_result.skip_reason = "no_eligible_references"
            return round_result

        child_1, child_2 = child_nodes
        selected_modalities = self._selected_modalities(clustering_config)
        anchor_modalities = self._anchor_modalities(
            anchor_config,
            selected_modalities,
        )
        features_dic, anchor_features_dic = self._build_feature_inputs(
            parent_node=parent_node,
            child_1=child_1,
            child_2=child_2,
            selected_modalities=selected_modalities,
            anchor_modalities=anchor_modalities,
            clustering_config=clustering_config,
        )
        round_result.features_dic = features_dic
        round_result.anchor_features_dic = anchor_features_dic

        dim_reduction_method = str(
            clustering_config.get(
                "dim_reduction_method",
                clustering_config.get("reduce_dimension_approach", "pca"),
            )
        ).lower()
        if dim_reduction_method == "selected_features":
            missing_features = [
                modality
                for modality in selected_modalities
                if len(features_dic.get(modality, [])) == 0
            ]
            if missing_features:
                raise ValueError(
                    f"No hierarchical clustering features are available for "
                    f"{missing_features} at parent node {parent_node!r}."
                )

        missing_anchor_features = self._missing_anchor_features(
            anchor_features_dic,
            anchor_modalities,
        )
        if missing_anchor_features:
            raise ValueError(
                f"No NN anchor features are available for {missing_anchor_features} "
                f"at parent node {parent_node!r}."
            )

        local_query_dic = _subset_adata_dic(
            self.query_adata_dic,
            active_obs_names,
            modalities=selected_modalities,
        )
        local_scaled_dic = _subset_adata_dic(
            self.query_adata_sca_dic,
            active_obs_names,
            modalities=anchor_modalities,
        )

        round_clustering_config = deepcopy(dict(clustering_config))
        round_clustering_config["features_dic"] = {
            modality: features_dic[modality]
            for modality in selected_modalities
            if modality in features_dic
        }
        if round_clustering_config.get("clustering_method") == "kmeans":
            requested_clusters = int(round_clustering_config.get("n_clusters", 2))
            round_clustering_config["n_clusters"] = min(
                max(requested_clusters, 2),
                len(active_obs_names),
            )
        round_result.clustering_config = deepcopy(round_clustering_config)

        if self.print_results:
            print()
            print("=" * 78)
            print(
                f"Hierarchy round: {parent_node} -> {child_1} vs {child_2} "
                f"({len(active_obs_names)} observations)"
            )
            print("=" * 78)

        clustering_result = query_multi_modal_clustering(
            query_adata_dic=local_query_dic,
            clustering_config=round_clustering_config,
            pred_key=self.cluster_key,
            query_section=self.qry_section,
            align_by_obs_names=True,
            print_results=self.print_results,
        )

        round_boundary_refinement_config = _normalize_enabled_config(
            deepcopy(self.boundary_refinement_config)
        )
        round_gene_subtyping_config = _normalize_enabled_config(
            deepcopy(self.gene_subtyping_config)
        )
        if round_gene_subtyping_config is not None:
            round_gene_subtyping_config = dict(round_gene_subtyping_config)
            if round_gene_subtyping_config.get("enabled", True):
                target_genes, nontarget_genes = self._get_gene_subtyping_features(
                    parent_node,
                    child_1,
                    child_2,
                    count_num=round_gene_subtyping_config.get("count_num"),
                )
                round_gene_subtyping_config["target_genes"] = target_genes
                round_gene_subtyping_config["nontarget_genes"] = nontarget_genes

        assignment_query_dic = local_query_dic
        if (
            _config_is_enabled(round_boundary_refinement_config)
            or _config_is_enabled(round_gene_subtyping_config)
        ):
            postprocess_query_dic = _subset_query_dic_for_postprocessing(
                local_query_dic=local_query_dic,
                full_query_dic=self.query_adata_dic,
                obs_names=active_obs_names,
                boundary_refinement_config=round_boundary_refinement_config,
                gene_subtyping_config=round_gene_subtyping_config,
            )
            assignment_query_dic = postprocess_query_dic
            clustering_result = postprocess_query_clustering_result(
                result=clustering_result,
                query_adata_dic=postprocess_query_dic,
                pred_key=self.cluster_key,
                boundary_refinement_config=round_boundary_refinement_config,
                gene_subtyping_config=round_gene_subtyping_config,
                print_results=self.print_results,
            )
        round_result.clustering_result = clustering_result

        anchor_result = self._detect_anchors(
            parent_node=parent_node,
            local_scaled_dic=local_scaled_dic,
            anchor_features_dic=anchor_features_dic,
            child_1=child_1,
            child_2=child_2,
            anchor_modalities=anchor_modalities,
            anchor_config=anchor_config,
        )
        round_result.anchor_result = anchor_result

        x_key = assignment_config.get("x_key", "x")
        y_key = assignment_config.get("y_key", "y")
        assignment_adata = _build_assignment_adata(
            local_scaled_dic=local_scaled_dic,
            local_query_dic=assignment_query_dic,
            clustering_result=clustering_result,
            anchor_result=anchor_result,
            cluster_key=self.cluster_key,
            x_key=x_key,
            y_key=y_key,
        )

        child_anchor_keys = [
            anchor_result.get_anchor_key(child_1),
            anchor_result.get_anchor_key(child_2),
        ]
        infer_key = f"{parent_node}__assignment"
        assign_kwargs = _filter_kwargs(
            assign_hierarchical_labels,
            assignment_config,
        )
        assign_kwargs.update(
            {
                "input_adata": assignment_adata,
                "hier_index": child_nodes,
                "hier_anchor_key": child_anchor_keys,
                "infer_key": infer_key,
                "cluster_key": self.cluster_key,
                "unassigned_label": self.unassigned_label,
                "print_results": self.print_results,
            }
        )
        assignment_result = assign_hierarchical_labels(**assign_kwargs)

        if assignment_config.get("adjust_one_side_assignment", False):
            binary_ratio_thres = assignment_config.get("binary_ratio_thres")
            if binary_ratio_thres is None:
                raise ValueError(
                    "binary_ratio_thres is required when "
                    "adjust_one_side_assignment=True."
                )
            adjust_kwargs = _filter_kwargs(
                adjust_one_side_binary_assignment,
                assignment_config,
            )
            adjust_kwargs.update(
                {
                    "test_gene_sca": assignment_adata,
                    "ref_gene_sca_dic": self._gene_reference_for_adjustment(
                        parent_node=parent_node,
                        ),
                    "target_regions": self.hier_tree.get_regions(child_1),
                    "nontgt_regions": self.hier_tree.get_regions(child_2),
                    "binary_ratio_thres": binary_ratio_thres,
                    "assignment_result": assignment_result,
                    "hier_index": child_nodes,
                    "hier_anchor_key": child_anchor_keys,
                    "infer_key": infer_key,
                    "cluster_key": self.cluster_key,
                    "label_key": self.label_key,
                    "unassigned_label": self.unassigned_label,
                    "print_results": self.print_results,
                }
            )
            assignment_result = adjust_one_side_binary_assignment(**adjust_kwargs)

        round_result.assignment_result = assignment_result
        for child_node in child_nodes:
            round_result.child_obs_names[child_node] = (
                assignment_result.labels.index[
                    assignment_result.labels.astype(str) == str(child_node)
                ].tolist()
            )

        assigned_names = {
            obs_name
            for child_names in round_result.child_obs_names.values()
            for obs_name in child_names
        }
        round_result.unresolved_obs_names = [
            name for name in active_obs_names if name not in assigned_names
        ]
        return round_result

    def run_round(
        self,
        parent_node: str,
        clustering_overrides: Optional[Mapping[str, Any]] = None,
        anchor_overrides: Optional[Mapping[str, Any]] = None,
        assignment_overrides: Optional[Mapping[str, Any]] = None,
        commit: bool = False,
    ) -> HierarchyRoundResult:
        """Preview or commit exactly one binary separation round.

        Parameters
        ----------
        parent_node : str
            Active internal hierarchy node to split.
        clustering_overrides, anchor_overrides, assignment_overrides : Mapping, optional
            Settings merged over the session defaults for this round only.
        commit : bool
            If ``True``, update child membership and final-label state.

        Returns
        -------
        HierarchyRoundResult
            Complete intermediate output for inspecting the round.
        """
        clustering_config = _merge_config(
            self.clustering_config,
            clustering_overrides,
        )
        anchor_config = _merge_config(self.anchor_config, anchor_overrides)
        assignment_config = _merge_config(
            self.assignment_config,
            assignment_overrides,
        )

        result = self._compute_round(
            parent_node=parent_node,
            clustering_config=clustering_config,
            anchor_config=anchor_config,
            assignment_config=assignment_config,
        )
        if commit:
            self.commit_round(result)
        return result

    def commit_round(self, round_result: HierarchyRoundResult) -> None:
        """Accept a previewed round and update reachable child-node state.

        ``round_result`` must match the observations currently assigned to its
        parent node. Committing a replacement invalidates older descendants.
        """
        parent_node = round_result.parent_node
        if parent_node not in self.node_obs_names:
            raise KeyError(f"No active state exists for parent node {parent_node!r}.")
        if list(round_result.obs_names) != list(self.node_obs_names[parent_node]):
            raise ValueError(
                "Round observations no longer match the current session state. "
                "Run the round again before committing it."
            )

        self._invalidate_subtree(parent_node)
        self.round_results[parent_node] = round_result
        active_obs_names = list(round_result.obs_names)

        if round_result.skipped:
            self._mark_terminal(
                parent_node,
                active_obs_names,
                round_result.skip_reason or "round_skipped",
            )
            return

        for child_node, child_names in round_result.child_obs_names.items():
            self.node_obs_names[child_node] = list(child_names)

        if round_result.unresolved_obs_names:
            self.final_labels.loc[
                round_result.unresolved_obs_names
            ] = self.unassigned_label

        parent_count = max(len(active_obs_names), 1)
        for child_node in round_result.child_nodes:
            child_names = round_result.child_obs_names.get(child_node, [])
            child_prop = len(child_names) / parent_count

            if self.hier_tree.is_leaf(child_node):
                self._mark_terminal(child_node, child_names, "leaf_node")
            elif len(child_names) == 0:
                self.terminal_reasons[child_node] = "no_assigned_observations"
            elif len(child_names) < max(self.min_node_spots, 2):
                self._mark_terminal(
                    child_node,
                    child_names,
                    "fewer_than_min_node_spots",
                )
            elif child_prop < self.min_node_prop:
                self._mark_terminal(
                    child_node,
                    child_names,
                    "below_min_node_prop",
                )

    def run_auto(self, start_node: Optional[str] = None):
        """Run all eligible internal nodes and return the scenario result.

        Parameters
        ----------
        start_node : str, optional
            Subtree root; defaults to the session's configured start node.
        """
        start_node = start_node or self.start_node
        if start_node not in self.node_obs_names:
            raise KeyError(f"No active observations exist for start node {start_node!r}.")

        allowed_nodes = set(self._subtree_nodes(start_node))
        while True:
            pending = [
                node
                for node in self.pending_internal_nodes()
                if node in allowed_nodes
            ]
            if not pending:
                break
            self.run_round(parent_node=pending[0], commit=True)

        return self.to_result(mode="auto")

    def to_result(self, mode: str = "manual"):
        """Materialize current state as the scenario-specific result object.

        Returned query objects contain the configured final-label column.
        ``mode`` is recorded only as output metadata.
        """
        for adata in self.query_adata_dic.values():
            modality_labels = pd.Series(
                self.unassigned_label,
                index=adata.obs_names,
                dtype=object,
            )
            shared = modality_labels.index.intersection(
                self.final_labels.index,
                sort=False,
            )
            modality_labels.loc[shared] = self.final_labels.reindex(shared).to_numpy()
            adata.obs[self.final_label_key] = pd.Categorical(
                modality_labels.astype(str)
            )

        params = {
            "mode": mode,
            "qry_section": self.qry_section,
            "target_parent_node": self.start_node,
            "cluster_key": self.cluster_key,
            "final_label_key": self.final_label_key,
            "unassigned_label": self.unassigned_label,
            "min_node_prop": self.min_node_prop,
            "min_node_spots": self.min_node_spots,
            "output_dir": self.output_dir,
            "fig_paras": deepcopy(self.fig_paras),
        }
        params.update(self._reference_params())
        return self.result_class(
            query_adata_dic=self.query_adata_dic,
            final_labels=self.final_labels.astype(str).copy(),
            node_obs_names=deepcopy(self.node_obs_names),
            # Round results are treated as committed records; copying the
            # mapping avoids duplicating their potentially large embeddings.
            round_results=dict(self.round_results),
            terminal_reasons=deepcopy(self.terminal_reasons),
            pending_nodes=self.pending_internal_nodes(),
            params=params,
        )


class SingleReferenceNNTransferSession(HierarchicalTransferSession):
    """Manual/automatic NN transfer controller for one reference section."""

    result_class = SingleReferenceNNTransferResult
    reference_scenario = "single_ref"


def run_single_ref_nn_round(
    session: SingleReferenceNNTransferSession,
    parent_node: str,
    clustering_overrides: Optional[Mapping[str, Any]] = None,
    anchor_overrides: Optional[Mapping[str, Any]] = None,
    assignment_overrides: Optional[Mapping[str, Any]] = None,
    commit: bool = False,
) -> HierarchyRoundResult:
    """Preview or commit one single-reference NN round.

    Parameters
    ----------
    session
        Manual session returned by ``single_ref_NN_based_label_transfer``.
    parent_node
        Internal hierarchy node to split.
    clustering_overrides, anchor_overrides, assignment_overrides
        Optional settings merged over the session defaults for this round.
    commit
        If ``True``, save assignments to the session; otherwise only preview.

    Returns
    -------
    HierarchyRoundResult
        Clustering, anchor, assignment, and child-membership outputs.
    """
    if not isinstance(session, SingleReferenceNNTransferSession):
        raise TypeError("session must be a SingleReferenceNNTransferSession.")
    return session.run_round(
        parent_node=parent_node,
        clustering_overrides=clustering_overrides,
        anchor_overrides=anchor_overrides,
        assignment_overrides=assignment_overrides,
        commit=commit,
    )


def single_ref_NN_based_label_transfer(
    ref_adata_sca_dic,
    query_adata_dic,
    query_adata_sca_dic,
    ref_section,
    qry_section,
    hier_tree,
    target_parent_node=None,
    qry_nodes_dic=None,
    gene_feature_results=None,
    image_feature_results=None,
    protein_feature_results=None,
    clustering_config=None,
    boundary_refinement_config=None,
    gene_subtyping_config=None,
    anchor_config=None,
    assignment_config=None,
    fig_paras=None,
    label_key="label",
    cluster_key="query_cluster",
    output_dir=None,
    min_node_prop=0.05,
    min_node_spots=2,
    final_label_key="hicat_label",
    unassigned_label="novel_cluster",
    copy=True,
    print_results=True,
    mode="auto",
):
    """Create a single-reference NN transfer session or run it automatically.

    Parameters
    ----------
    ref_adata_sca_dic : Mapping[str, AnnData]
        Scaled reference data by modality, e.g.
        ``{"Gene": ref_gene_sca, "Protein": ref_protein_sca}``.
    query_adata_dic : Mapping[str, AnnData]
        Query data by modality for clustering (Gene, Protein, and/or Image).
    query_adata_sca_dic : Mapping[str, AnnData]
        Scaled query data by molecular modality for anchor detection.
    ref_section, qry_section : str
        Reference and query section names.
    hier_tree : HierTree
        Binary hierarchy providing root, children, leaves, and region labels.
    target_parent_node : str, optional
        Subtree root to process; defaults to ``hier_tree.root_node``.
    qry_nodes_dic : Mapping[str, Sequence[str] or AnnData], optional
        Initial query membership by hierarchy node.
    gene_feature_results, image_feature_results, protein_feature_results : HierarchicalFeatureResults, optional
        Modality-specific hierarchical features. At least one is required.
    clustering_config : Mapping
        Clustering settings; must include ``selected_modalities``.
    boundary_refinement_config, gene_subtyping_config : Mapping, optional
        Optional post-clustering refinement settings.
    anchor_config : Mapping, optional
        NN anchor settings, including optional ``modalities``.
    assignment_config : Mapping, optional
        Label-assignment settings. Set ``adjust_one_side_assignment=True``
        with ``binary_ratio_thres`` to enable one-sided adjustment.
    fig_paras : Mapping, optional
        Figure metadata retained with the run.
    label_key, cluster_key : str
        Reference label column and query cluster column names.
    output_dir : path-like, optional
        Output path metadata retained with the run.
    min_node_prop : float
        Minimum child proportion required to continue recursion.
    min_node_spots : int
        Minimum child observation count required to continue recursion.
    final_label_key, unassigned_label : str
        Output ``obs`` column and fallback label.
    copy : bool
        Copy ``query_adata_dic`` before adding the output column.
    print_results : bool
        Print stage summaries.
    mode : {"auto", "manual"}
        Run the full subtree or return an unexecuted session.

    Returns
    -------
    SingleReferenceNNTransferResult or SingleReferenceNNTransferSession
        Automatic output, or a session for round-by-round processing.
    """
    if mode not in {"auto", "manual"}:
        raise ValueError("mode must be either 'auto' or 'manual'.")

    session = SingleReferenceNNTransferSession(
        ref_adata_sca_dic=ref_adata_sca_dic,
        query_adata_dic=query_adata_dic,
        query_adata_sca_dic=query_adata_sca_dic,
        ref_section=ref_section,
        qry_section=qry_section,
        hier_tree=hier_tree,
        target_parent_node=target_parent_node,
        qry_nodes_dic=qry_nodes_dic,
        gene_feature_results=gene_feature_results,
        image_feature_results=image_feature_results,
        protein_feature_results=protein_feature_results,
        clustering_config=clustering_config,
        boundary_refinement_config=boundary_refinement_config,
        gene_subtyping_config=gene_subtyping_config,
        anchor_config=anchor_config,
        assignment_config=assignment_config,
        fig_paras=fig_paras,
        label_key=label_key,
        cluster_key=cluster_key,
        output_dir=output_dir,
        min_node_prop=min_node_prop,
        min_node_spots=min_node_spots,
        final_label_key=final_label_key,
        unassigned_label=unassigned_label,
        copy=copy,
        print_results=print_results,
    )

    if mode == "manual":
        return session
    return session.run_auto(start_node=session.start_node)


class MultiReferenceNNTransferSession(SingleReferenceNNTransferSession):
    """Manual/automatic NN transfer controller for multiple references.

    Clustering receives direction-specific features aggregated across the
    requested reference sections. NN anchor detection receives a separate
    feature dictionary for every reference section, matching
    ``nn_based_anchor_detection_multiref_multimodal``.
    """

    result_class = MultiReferenceNNTransferResult
    reference_scenario = "multi_ref"

    def __init__(
        self,
        ref_adata_sca_dic,
        query_adata_dic,
        query_adata_sca_dic,
        ref_section_list,
        qry_section,
        hier_tree,
        reference_section_guide=None,
        strict_reference_guide=True,
        **kwargs,
    ):
        self.ref_section_list = list(ref_section_list or [])
        if len(self.ref_section_list) == 0:
            raise ValueError("ref_section_list cannot be empty.")

        missing_sections = [
            section
            for section in self.ref_section_list
            if section not in ref_adata_sca_dic
        ]
        if missing_sections:
            raise KeyError(
                "ref_adata_sca_dic is missing reference sections: "
                f"{missing_sections}"
            )

        super().__init__(
            ref_adata_sca_dic=ref_adata_sca_dic,
            query_adata_dic=query_adata_dic,
            query_adata_sca_dic=query_adata_sca_dic,
            ref_section=self.ref_section_list[0],
            qry_section=qry_section,
            hier_tree=hier_tree,
            **kwargs,
        )
        self.reference_section_guide = normalize_nn_reference_section_guide(
            reference_section_guide=reference_section_guide,
            selected_references=self.ref_section_list,
            available_parent_nodes=(
                self.multimodal_feature_results.available_parent_nodes()
            ),
            strict=bool(strict_reference_guide),
        )
        self.strict_reference_guide = bool(strict_reference_guide)

    def _reference_sections_for_node(self, parent_node: str) -> List[str]:
        """Return the node-specific references, falling back to all selected."""
        return list(
            self.reference_section_guide.get(
                parent_node,
                self.ref_section_list,
            )
        )

    def _build_feature_inputs(
        self,
        parent_node: str,
        child_1: str,
        child_2: str,
        selected_modalities: List[str],
        anchor_modalities: List[str],
        clustering_config: Mapping[str, Any],
    ) -> tuple[Dict[str, List[str]], Dict[str, Any]]:
        del child_1, child_2
        active_ref_sections = self._reference_sections_for_node(parent_node)
        count_num = clustering_config.get(
            "feature_count_num",
            clustering_config.get("count_num"),
        )
        for modality in selected_modalities:
            modality_result = self.feature_results_dic.get(modality)
            if modality_result is None:
                continue
            required_count = (
                modality_result.count_num
                if count_num is None
                else int(count_num)
            )
            if required_count > len(active_ref_sections):
                raise ValueError(
                    f"Feature count threshold {required_count} exceeds the "
                    f"{len(active_ref_sections)} eligible reference section(s) "
                    f"for modality={modality!r}, parent_node={parent_node!r}. "
                    "Lower clustering_config['feature_count_num'] or retain "
                    "more references for this node."
                )
        clustering_features = (
            self.multimodal_feature_results.get_clustering_features_dic(
                parent_node=parent_node,
                modalities=selected_modalities,
                ref_section_list=active_ref_sections,
                count_num=count_num,
                strict=False,
            )
        )
        anchor_features = self.multimodal_feature_results.get_nn_anchor_features(
            parent_node=parent_node,
            modalities=anchor_modalities,
            ref_section_list=active_ref_sections,
            strict=True,
        )
        return clustering_features, anchor_features

    def _missing_anchor_features(
        self,
        anchor_features_dic: Mapping[str, Any],
        anchor_modalities: List[str],
    ) -> List[str]:
        missing = []
        for section, section_features in anchor_features_dic.items():
            for modality in anchor_modalities:
                if len(section_features.get(modality, [])) == 0:
                    missing.append(f"{section}:{modality}")
        return missing

    def _get_gene_subtyping_features(
        self,
        parent_node: str,
        child_1: str,
        child_2: str,
        count_num: Optional[int] = None,
    ) -> tuple[List[str], List[str]]:
        active_ref_sections = self._reference_sections_for_node(parent_node)
        target_features = (
            self.multimodal_feature_results.get_direction_features_dic(
                parent_node=parent_node,
                direction=f"{child_1}_vs_{child_2}",
                modalities=["Gene"],
                ref_section_list=active_ref_sections,
                count_num=count_num,
                strict=False,
            )
        )
        nontgt_features = (
            self.multimodal_feature_results.get_direction_features_dic(
                parent_node=parent_node,
                direction=f"{child_2}_vs_{child_1}",
                modalities=["Gene"],
                ref_section_list=active_ref_sections,
                count_num=count_num,
                strict=False,
            )
        )
        return target_features.get("Gene", []), nontgt_features.get("Gene", [])

    def _detect_anchors(
        self,
        parent_node: str,
        local_scaled_dic: Mapping[str, Any],
        anchor_features_dic: Mapping[str, Any],
        child_1: str,
        child_2: str,
        anchor_modalities: List[str],
        anchor_config: Mapping[str, Any],
    ) -> AnchorDetectionResult:
        active_ref_sections = self._reference_sections_for_node(parent_node)
        anchor_call_kwargs = _filter_kwargs(
            nn_based_anchor_detection_multiref_multimodal,
            anchor_config,
        )
        anchor_call_kwargs.update(
            {
                "ref_adata_sca_dic": {
                    section: self.ref_adata_sca_dic[section]
                    for section in active_ref_sections
                },
                "test_adata_sca_dic": local_scaled_dic,
                "features_dic": anchor_features_dic,
                "target_regions": self.hier_tree.get_regions(child_1),
                "nontgt_regions": self.hier_tree.get_regions(child_2),
                "target_node": child_1,
                "nontgt_node": child_2,
                "ref_section_list": active_ref_sections,
                "modalities": anchor_modalities,
                "label_key": self.label_key,
                "copy": True,
                "return_result": True,
                "print_results": self.print_results,
            }
        )
        return nn_based_anchor_detection_multiref_multimodal(
            **anchor_call_kwargs
        )

    def _gene_reference_for_adjustment(self, parent_node: str):
        """Convert section-first multimodal references to section -> Gene."""
        references = {}
        for section in self._reference_sections_for_node(parent_node):
            if "Gene" not in self.ref_adata_sca_dic[section]:
                raise KeyError(
                    f"One-sided adjustment requires Gene data for {section!r}."
                )
            references[section] = self.ref_adata_sca_dic[section]["Gene"]
        return references

    def _reference_params(self) -> Dict[str, Any]:
        return {
            "reference_scenario": self.reference_scenario,
            "ref_section_list": list(self.ref_section_list),
            "reference_section_guide": deepcopy(self.reference_section_guide),
            "strict_reference_guide": self.strict_reference_guide,
        }


def run_multi_ref_nn_round(
    session: MultiReferenceNNTransferSession,
    parent_node: str,
    clustering_overrides: Optional[Mapping[str, Any]] = None,
    anchor_overrides: Optional[Mapping[str, Any]] = None,
    assignment_overrides: Optional[Mapping[str, Any]] = None,
    commit: bool = False,
) -> HierarchyRoundResult:
    """Preview or commit one multi-reference NN round.

    Parameters
    ----------
    session : MultiReferenceNNTransferSession
        Manual multi-reference session.
    parent_node : str
        Internal hierarchy node to split.
    clustering_overrides, anchor_overrides, assignment_overrides : Mapping, optional
        Settings merged over the session defaults for this round.
    commit : bool
        Save assignments to the session if ``True``; otherwise only preview.

    Returns
    -------
    HierarchyRoundResult
        Clustering, anchor, assignment, and child-membership outputs.
    """
    if not isinstance(session, MultiReferenceNNTransferSession):
        raise TypeError("session must be a MultiReferenceNNTransferSession.")
    return session.run_round(
        parent_node=parent_node,
        clustering_overrides=clustering_overrides,
        anchor_overrides=anchor_overrides,
        assignment_overrides=assignment_overrides,
        commit=commit,
    )


def multi_ref_NN_based_label_transfer(
    ref_adata_sca_dic,
    query_adata_dic,
    query_adata_sca_dic,
    ref_section_list,
    qry_section,
    hier_tree,
    target_parent_node=None,
    qry_nodes_dic=None,
    gene_feature_results=None,
    image_feature_results=None,
    protein_feature_results=None,
    clustering_config=None,
    boundary_refinement_config=None,
    gene_subtyping_config=None,
    anchor_config=None,
    assignment_config=None,
    fig_paras=None,
    label_key="label",
    cluster_key="query_cluster",
    output_dir=None,
    min_node_prop=0.05,
    min_node_spots=2,
    final_label_key="hicat_label",
    unassigned_label="novel_cluster",
    copy=True,
    print_results=True,
    mode="auto",
    reference_section_guide=None,
    strict_reference_guide=True,
):
    """Create a multi-reference session or traverse it automatically.

    Parameters
    ----------
    ref_adata_sca_dic : Mapping[str, Mapping[str, AnnData]]
        Scaled references, section first and modality second, e.g.
        ``{"ref_1": {"Gene": gene1}, "ref_2": {"Gene": gene2}}``.
    query_adata_dic, query_adata_sca_dic : Mapping[str, AnnData]
        Query modality dictionaries for clustering and anchor detection.
    ref_section_list : Sequence[str]
        Reference sections to use; each must be present in
        ``ref_adata_sca_dic`` and the feature results.
    qry_section : str
        Query section name.
    hier_tree : HierTree
        Binary hierarchy to traverse.
    target_parent_node : str, optional
        Subtree root; defaults to ``hier_tree.root_node``.
    qry_nodes_dic : Mapping[str, Sequence[str] or AnnData], optional
        Initial query membership by hierarchy node.
    gene_feature_results, image_feature_results, protein_feature_results : HierarchicalFeatureResults, optional
        Modality-specific hierarchical features. At least one is required.
    clustering_config : Mapping
        Clustering settings; must include ``selected_modalities``.
    boundary_refinement_config, gene_subtyping_config : Mapping, optional
        Optional post-clustering refinement settings.
    anchor_config, assignment_config : Mapping, optional
        Multi-reference NN anchor and label-assignment settings.
    fig_paras, output_dir : optional
        Figure and output-path metadata retained with the run.
    label_key, cluster_key : str
        Reference label and query cluster column names.
    min_node_prop : float
        Minimum child proportion required to continue recursion.
    min_node_spots : int
        Minimum child observation count required to continue recursion.
    final_label_key, unassigned_label : str
        Output ``obs`` column and fallback label.
    copy, print_results : bool
        Copy query data before annotation and print stage summaries.
    mode : {"auto", "manual"}
        Run the full subtree or return an unexecuted session.
    reference_section_guide : Mapping[str, Sequence[str]], optional
        User-defined node-specific reference subsets based on prior knowledge,
        biological information, or manual inspection. Pass the inner guide for
        the current query, for example
        ``{"node_0": ["ref_1", "ref_2"], "node_1": ["ref_2"]}``.
        Keys are internal parent nodes and values are subsets of
        ``ref_section_list``. If ``None`` or if a parent node is absent, that
        node uses all selected references. An explicit empty list stops that
        branch with ``skip_reason="no_eligible_references"``.
    strict_reference_guide : bool, default=True
        Whether unknown parent nodes or references outside
        ``ref_section_list`` raise an error. If ``False``, invalid entries are
        ignored during guide normalization.

    Returns
    -------
    MultiReferenceNNTransferResult or MultiReferenceNNTransferSession
        Automatic output, or a session for round-by-round processing.
    """
    if mode not in {"auto", "manual"}:
        raise ValueError("mode must be either 'auto' or 'manual'.")

    session = MultiReferenceNNTransferSession(
        ref_adata_sca_dic=ref_adata_sca_dic,
        query_adata_dic=query_adata_dic,
        query_adata_sca_dic=query_adata_sca_dic,
        ref_section_list=ref_section_list,
        qry_section=qry_section,
        hier_tree=hier_tree,
        reference_section_guide=reference_section_guide,
        strict_reference_guide=strict_reference_guide,
        target_parent_node=target_parent_node,
        qry_nodes_dic=qry_nodes_dic,
        gene_feature_results=gene_feature_results,
        image_feature_results=image_feature_results,
        protein_feature_results=protein_feature_results,
        clustering_config=clustering_config,
        boundary_refinement_config=boundary_refinement_config,
        gene_subtyping_config=gene_subtyping_config,
        anchor_config=anchor_config,
        assignment_config=assignment_config,
        fig_paras=fig_paras,
        label_key=label_key,
        cluster_key=cluster_key,
        output_dir=output_dir,
        min_node_prop=min_node_prop,
        min_node_spots=min_node_spots,
        final_label_key=final_label_key,
        unassigned_label=unassigned_label,
        copy=copy,
        print_results=print_results,
    )
    if mode == "manual":
        return session
    return session.run_auto(start_node=session.start_node)


class QuantileBasedTransferSession(HierarchicalTransferSession):
    """Manual/automatic transfer controller using quantile-based anchors.

    Reference inputs use modality-first nesting:

    ``{modality: {reference_section: scaled_adata}}``.

    This differs from the section-first nesting used by the multi-reference NN
    session because quantile thresholds are estimated independently within
    each modality across all selected reference sections.
    """

    result_class = QuantileBasedTransferResult
    reference_scenario = "quantile_based"

    def __init__(
        self,
        ref_adata_sca_dic,
        merged_ref_adata_sca_dic,
        query_adata_dic,
        query_adata_sca_dic,
        ref_section_list,
        qry_section,
        hier_tree,
        **kwargs,
    ):
        if not isinstance(ref_adata_sca_dic, Mapping) or len(ref_adata_sca_dic) == 0:
            raise ValueError("ref_adata_sca_dic cannot be empty.")
        if not isinstance(merged_ref_adata_sca_dic, Mapping):
            raise TypeError("merged_ref_adata_sca_dic must be a modality dictionary.")

        if ref_section_list is None:
            first_modality_refs = next(iter(ref_adata_sca_dic.values()))
            if not isinstance(first_modality_refs, Mapping):
                raise TypeError(
                    "Each ref_adata_sca_dic modality must map reference section "
                    "names to AnnData objects."
                )
            ref_section_list = list(first_modality_refs)

        self.ref_section_list = list(dict.fromkeys(ref_section_list))
        if len(self.ref_section_list) == 0:
            raise ValueError("ref_section_list cannot be empty.")

        self.merged_ref_adata_sca_dic = merged_ref_adata_sca_dic
        super().__init__(
            ref_adata_sca_dic=ref_adata_sca_dic,
            query_adata_dic=query_adata_dic,
            query_adata_sca_dic=query_adata_sca_dic,
            ref_section=self.ref_section_list[0],
            qry_section=qry_section,
            hier_tree=hier_tree,
            **kwargs,
        )

        selected_modalities = self._selected_modalities(self.clustering_config)
        anchor_modalities = self._anchor_modalities(
            self.anchor_config,
            selected_modalities,
        )
        for modality in anchor_modalities:
            if modality not in self.ref_adata_sca_dic:
                raise KeyError(
                    f"ref_adata_sca_dic is missing anchor modality {modality!r}."
                )
            if modality not in self.merged_ref_adata_sca_dic:
                raise KeyError(
                    f"merged_ref_adata_sca_dic is missing modality {modality!r}."
                )
            missing_sections = [
                section
                for section in self.ref_section_list
                if section not in self.ref_adata_sca_dic[modality]
            ]
            if missing_sections:
                raise KeyError(
                    f"ref_adata_sca_dic[{modality!r}] is missing reference "
                    f"sections: {missing_sections}."
                )

    def _build_feature_inputs(
        self,
        parent_node: str,
        child_1: str,
        child_2: str,
        selected_modalities: List[str],
        anchor_modalities: List[str],
        clustering_config: Mapping[str, Any],
    ) -> tuple[Dict[str, List[str]], Dict[str, Any]]:
        count_num = clustering_config.get(
            "feature_count_num",
            clustering_config.get("count_num"),
        )
        clustering_features = (
            self.multimodal_feature_results.get_clustering_features_dic(
                parent_node=parent_node,
                modalities=selected_modalities,
                ref_section_list=self.ref_section_list,
                count_num=count_num,
                strict=False,
            )
        )
        target_genes_dic, nontgt_genes_dic = (
            self.multimodal_feature_results.get_quantile_anchor_features(
                parent_node=parent_node,
                modalities=anchor_modalities,
                target_node=child_1,
                nontgt_node=child_2,
                strict=True,
            )
        )
        return clustering_features, {
            "target_genes_dic": target_genes_dic,
            "nontgt_genes_dic": nontgt_genes_dic,
        }

    def _missing_anchor_features(
        self,
        anchor_features_dic: Mapping[str, Any],
        anchor_modalities: List[str],
    ) -> List[str]:
        target_features = anchor_features_dic.get("target_genes_dic", {})
        nontgt_features = anchor_features_dic.get("nontgt_genes_dic", {})
        missing = []
        for modality in anchor_modalities:
            if len(target_features.get(modality, [])) == 0:
                missing.append(f"target:{modality}")
            if len(nontgt_features.get(modality, [])) == 0:
                missing.append(f"nontarget:{modality}")
        return missing

    def _get_gene_subtyping_features(
        self,
        parent_node: str,
        child_1: str,
        child_2: str,
        count_num: Optional[int] = None,
    ) -> tuple[List[str], List[str]]:
        del count_num
        target_features, nontgt_features = (
            self.multimodal_feature_results.get_quantile_anchor_features(
                parent_node=parent_node,
                modalities=["Gene"],
                target_node=child_1,
                nontgt_node=child_2,
                strict=False,
            )
        )
        return target_features.get("Gene", []), nontgt_features.get("Gene", [])

    def _detect_anchors(
        self,
        parent_node: str,
        local_scaled_dic: Mapping[str, Any],
        anchor_features_dic: Mapping[str, Any],
        child_1: str,
        child_2: str,
        anchor_modalities: List[str],
        anchor_config: Mapping[str, Any],
    ) -> AnchorDetectionResult:
        selected_ref_adata_dic = {
            modality: {
                section: self.ref_adata_sca_dic[modality][section]
                for section in self.ref_section_list
            }
            for modality in anchor_modalities
        }
        selected_merged_ref_dic = {
            modality: self.merged_ref_adata_sca_dic[modality]
            for modality in anchor_modalities
        }
        anchor_call_kwargs = _filter_kwargs(
            quantile_based_anchor_detection_multimodal,
            anchor_config,
        )
        anchor_call_kwargs.update(
            {
                "ref_adata_sca_dic": selected_ref_adata_dic,
                "merged_ref_adata_sca_dic": selected_merged_ref_dic,
                "test_adata_sca_dic": local_scaled_dic,
                "target_node": child_1,
                "nontgt_node": child_2,
                "target_regions": self.hier_tree.get_regions(child_1),
                "nontgt_regions": self.hier_tree.get_regions(child_2),
                "target_genes_dic": anchor_features_dic["target_genes_dic"],
                "nontgt_genes_dic": anchor_features_dic["nontgt_genes_dic"],
                "modalities": anchor_modalities,
                "label_key": self.label_key,
                "copy": True,
                "return_result": True,
                "print_results": self.print_results,
            }
        )
        return quantile_based_anchor_detection_multimodal(**anchor_call_kwargs)

    def _gene_reference_for_adjustment(self, parent_node: str):
        """Extract section -> Gene from modality-first quantile references."""
        if "Gene" not in self.ref_adata_sca_dic:
            raise KeyError(
                "One-sided adjustment requires ref_adata_sca_dic['Gene']."
            )
        return {
            section: self.ref_adata_sca_dic["Gene"][section]
            for section in self.ref_section_list
        }

    def _reference_params(self) -> Dict[str, Any]:
        return {
            "reference_scenario": self.reference_scenario,
            "ref_section_list": list(self.ref_section_list),
        }


def run_quantile_round(
    session: QuantileBasedTransferSession,
    parent_node: str,
    clustering_overrides: Optional[Mapping[str, Any]] = None,
    anchor_overrides: Optional[Mapping[str, Any]] = None,
    assignment_overrides: Optional[Mapping[str, Any]] = None,
    commit: bool = False,
) -> HierarchyRoundResult:
    """Preview or commit one quantile-based hierarchy round.

    Parameters
    ----------
    session : QuantileBasedTransferSession
        Manual quantile-based session.
    parent_node : str
        Internal hierarchy node to split.
    clustering_overrides, anchor_overrides, assignment_overrides : Mapping, optional
        Settings merged over the session defaults for this round.
    commit : bool
        Save assignments to the session if ``True``; otherwise only preview.

    Returns
    -------
    HierarchyRoundResult
        Clustering, anchor, assignment, and child-membership outputs.
    """
    if not isinstance(session, QuantileBasedTransferSession):
        raise TypeError("session must be a QuantileBasedTransferSession.")
    return session.run_round(
        parent_node=parent_node,
        clustering_overrides=clustering_overrides,
        anchor_overrides=anchor_overrides,
        assignment_overrides=assignment_overrides,
        commit=commit,
    )


def quantile_based_label_transfer(
    ref_adata_sca_dic,
    merged_ref_adata_sca_dic,
    query_adata_dic,
    query_adata_sca_dic,
    ref_section_list,
    qry_section,
    hier_tree,
    target_parent_node=None,
    qry_nodes_dic=None,
    gene_feature_results=None,
    image_feature_results=None,
    protein_feature_results=None,
    clustering_config=None,
    boundary_refinement_config=None,
    gene_subtyping_config=None,
    anchor_config=None,
    assignment_config=None,
    fig_paras=None,
    label_key="label",
    cluster_key="query_cluster",
    output_dir=None,
    min_node_prop=0.05,
    min_node_spots=2,
    final_label_key="hicat_label",
    unassigned_label="novel_cluster",
    copy=True,
    print_results=True,
    mode="auto",
):
    """Create a quantile-based session or traverse it automatically.

    Parameters
    ----------
    ref_adata_sca_dic : Mapping[str, Mapping[str, AnnData]]
        Scaled references, modality first and section second, e.g.
        ``{"Gene": {"ref_1": gene1, "ref_2": gene2}}``.
    merged_ref_adata_sca_dic : Mapping[str, AnnData]
        Merged scaled reference per modality. Each object must contain the
        reference-section column named by the anchor configuration.
    query_adata_dic, query_adata_sca_dic : Mapping[str, AnnData]
        Query modality dictionaries for clustering and anchor detection.
    ref_section_list : Sequence[str] or None
        Reference sections to use; ``None`` derives them from the first
        reference modality.
    qry_section : str
        Query section name.
    hier_tree : HierTree
        Binary hierarchy to traverse.
    target_parent_node : str, optional
        Subtree root; defaults to ``hier_tree.root_node``.
    qry_nodes_dic : Mapping[str, Sequence[str] or AnnData], optional
        Initial query membership by hierarchy node.
    gene_feature_results, image_feature_results, protein_feature_results : HierarchicalFeatureResults, optional
        Modality-specific hierarchical features. At least one is required.
    clustering_config : Mapping
        Clustering settings; must include ``selected_modalities``.
    boundary_refinement_config, gene_subtyping_config : Mapping, optional
        Optional post-clustering refinement settings.
    anchor_config, assignment_config : Mapping, optional
        Quantile-anchor and label-assignment settings.
    fig_paras, output_dir : optional
        Figure and output-path metadata retained with the run.
    label_key, cluster_key : str
        Reference label and query cluster column names.
    min_node_prop : float
        Minimum child proportion required to continue recursion.
    min_node_spots : int
        Minimum child observation count required to continue recursion.
    final_label_key, unassigned_label : str
        Output ``obs`` column and fallback label.
    copy, print_results : bool
        Copy query data before annotation and print stage summaries.
    mode : {"auto", "manual"}
        Run the full subtree or return an unexecuted session.

    Returns
    -------
    QuantileBasedTransferResult or QuantileBasedTransferSession
        Automatic output, or a session for round-by-round processing.
    """
    if mode not in {"auto", "manual"}:
        raise ValueError("mode must be either 'auto' or 'manual'.")

    session = QuantileBasedTransferSession(
        ref_adata_sca_dic=ref_adata_sca_dic,
        merged_ref_adata_sca_dic=merged_ref_adata_sca_dic,
        query_adata_dic=query_adata_dic,
        query_adata_sca_dic=query_adata_sca_dic,
        ref_section_list=ref_section_list,
        qry_section=qry_section,
        hier_tree=hier_tree,
        target_parent_node=target_parent_node,
        qry_nodes_dic=qry_nodes_dic,
        gene_feature_results=gene_feature_results,
        image_feature_results=image_feature_results,
        protein_feature_results=protein_feature_results,
        clustering_config=clustering_config,
        boundary_refinement_config=boundary_refinement_config,
        gene_subtyping_config=gene_subtyping_config,
        anchor_config=anchor_config,
        assignment_config=assignment_config,
        fig_paras=fig_paras,
        label_key=label_key,
        cluster_key=cluster_key,
        output_dir=output_dir,
        min_node_prop=min_node_prop,
        min_node_spots=min_node_spots,
        final_label_key=final_label_key,
        unassigned_label=unassigned_label,
        copy=copy,
        print_results=print_results,
    )
    if mode == "manual":
        return session
    return session.run_auto(start_node=session.start_node)


def save_label_transfer_outputs(
    transfer_result,
    transfer_scenario,
    output_dir,
    qry_section,
    x_key="x",
    y_key="y",
    refine=True,
    refined_label_key=None,
    num_nbs=25,
    cat_color=None,
    size=50,
    dpi=100,
    invert_x=False,
    invert_y=True,
):
    """
    Retrieve, optionally refine, save, and visualize Gene label-transfer output.
    """

    final_label_key = transfer_result.params["final_label_key"]

    if "Gene" not in transfer_result.query_adata_dic:
        raise KeyError("transfer_result.query_adata_dic does not contain 'Gene'.")

    query_gene_adata = transfer_result.query_adata_dic["Gene"].copy()

    if final_label_key not in query_gene_adata.obs:
        raise KeyError(
            f"{final_label_key!r} is not found in query Gene adata.obs."
        )

    if refined_label_key is None:
        refined_label_key = f"{final_label_key}_refined"

    if refine:
        query_gene_adata, _ = refine_labels(
            input_adata=query_gene_adata,
            pred_key=final_label_key,
            refined_key=refined_label_key,
            num_nbs=num_nbs,
            x_key=x_key,
            y_key=y_key,
            copy=False,
        )

    sample_dir = Path(output_dir) / qry_section / transfer_scenario

    sample_dir.mkdir(parents=True, exist_ok=True)

    # Save the complete observation table.
    obs_to_save = query_gene_adata.obs.copy()
    obs_to_save.index.name = obs_to_save.index.name or "obs_name"
    obs_to_save.to_csv(sample_dir / "predicted_obs.csv")

    # Save original prediction plot.
    cat_figure(
        input_adata=query_gene_adata,
        x_key=x_key,
        y_key=y_key,
        fig_title=f"{qry_section}: predicted tissue regions",
        fig_path=sample_dir / "predicted_regions.png",
        color_key=final_label_key,
        cat_color=cat_color,
        size=size,
        dpi=dpi,
        invert_x=invert_x,
        invert_y=invert_y,
    )

    # Save refined prediction plot.
    if refine:
        cat_figure(
            input_adata=query_gene_adata,
            x_key=x_key,
            y_key=y_key,
            fig_title=f"{qry_section}: refined tissue regions",
            fig_path=sample_dir / "refined_predicted_regions.png",
            color_key=refined_label_key,
            cat_color=cat_color,
            size=size,
            dpi=dpi,
            invert_x=invert_x,
            invert_y=invert_y,
        )

    return query_gene_adata


__all__ = [
    "HierarchyRoundResult",
    "HierarchicalTransferResult",
    "SingleReferenceNNTransferResult",
    "MultiReferenceNNTransferResult",
    "QuantileBasedTransferResult",
    "HierarchicalTransferSession",
    "SingleReferenceNNTransferSession",
    "MultiReferenceNNTransferSession",
    "QuantileBasedTransferSession",
    "run_single_ref_nn_round",
    "run_multi_ref_nn_round",
    "run_quantile_round",
    "single_ref_NN_based_label_transfer",
    "multi_ref_NN_based_label_transfer",
    "quantile_based_label_transfer",
    "save_label_transfer_outputs",
]
