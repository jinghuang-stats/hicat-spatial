import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional


@dataclass
class LabelAssignmentResult:
    """Index-aligned output from one hierarchical label-assignment round."""

    labels: pd.Series
    cross_table: pd.DataFrame
    adjusted_cross_table: pd.DataFrame
    novel_clusters: List[Any] = field(default_factory=list)
    unassigned_obs_names: List[str] = field(default_factory=list)
    adjustment_info: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.labels, pd.Series):
            raise TypeError("labels must be a pandas Series indexed by obs_names.")
        if not self.labels.index.is_unique:
            raise ValueError("labels.index must contain unique obs_names.")
        self.labels = self.labels.astype(str).copy()

    @property
    def label_key(self) -> str:
        return self.labels.name or self.params.get("infer_key", "inferred_label")

    def apply_to(self, adata, key: Optional[str] = None, copy: bool = True):
        """Attach assigned labels to a compatible AnnData by ``obs_names``."""
        output = adata.copy() if copy or adata.is_view else adata
        missing = self.labels.index.difference(output.obs_names)
        if len(missing) > 0:
            raise ValueError(
                "AnnData is missing assigned observations. "
                f"Examples: {missing[:5].tolist()}"
            )
        output_key = key or self.label_key
        output.obs.loc[self.labels.index, output_key] = self.labels
        output.obs[output_key] = output.obs[output_key].astype("category")
        return output


def _format_cluster_list(clusters: List[Any]) -> str:
    """Return a readable cluster list for diagnostic printing."""
    if len(clusters) == 0:
        return "None"
    return ", ".join(map(str, clusters))


def _anchor_distribution_by_cluster(
    obs: pd.DataFrame,
    *,
    cluster_key: str,
    anchor_key: str,
) -> pd.DataFrame:
    """Summarize where positive anchors fall across query clusters."""
    anchor_values = pd.to_numeric(obs[anchor_key], errors="coerce").fillna(0)
    anchor_obs = obs.loc[anchor_values > 0]

    if anchor_obs.shape[0] == 0:
        return pd.DataFrame(columns=["spots_num", "percentage"])

    counts = anchor_obs[cluster_key].value_counts()
    return pd.DataFrame(
        {
            "spots_num": counts.astype(int),
            "percentage": counts / counts.sum(),
        }
    )


def _print_binary_assignment_adjustment_report(
    *,
    hier_index,
    prop_diff: pd.Series,
    same_prop_clusters: List[Any],
    similar_clusters: List[Any],
    anchor_distributions: Dict[Any, pd.DataFrame],
    cross_table_upd: pd.DataFrame,
    original_assignment: pd.Series,
    adjusted_assignment: pd.Series,
    changed_clusters: List[Any],
) -> None:
    """Print detailed diagnostics for ambiguous binary label assignment."""
    print(
        "The clusters having the same proportions across two hierarchy anchors:",
        _format_cluster_list(same_prop_clusters),
    )
    print(
        "=========== The absolute proportion difference across two hierarchies ==========="
    )
    print(prop_diff.sort_values())
    print(
        "The clusters having similar proportions across two hierarchy anchors:",
        _format_cluster_list(similar_clusters),
    )

    for label in hier_index:
        print(f"------------------- {label} -------------------")
        dist_table = anchor_distributions.get(label)
        if dist_table is None or dist_table.empty:
            print("The number of detected anchors: 0")
            print("The anchors distribution across clusters")
            print(pd.DataFrame(columns=["spots_num", "percentage"]))
            continue

        print(f"The number of detected anchors: {int(dist_table['spots_num'].sum())}")
        print("The anchors distribution across clusters")
        print(dist_table)

    print(
        "========== Updated Cross Table of Anchors "
        "(after assigning the weights of anchors that fall in each cluster) =========="
    )
    print(cross_table_upd.round(2))

    assignment_comparison = pd.DataFrame(
        {
            "original": original_assignment.astype(str),
            "adjusted_by_weights": adjusted_assignment.astype(str),
        }
    )
    print(
        "=========== Clusters Label Assignment Difference "
        "(without/with weights adjustment) ==========="
    )
    print(assignment_comparison)
    print(
        "After adjusted by weights, the clusters that have different label assignments:",
        _format_cluster_list(changed_clusters),
    )


# ============================================================
# Label assignment
# ============================================================
def assign_hierarchical_labels(
    input_adata,
    hier_index=("a", "b"),
    hier_anchor_key=("a_anchors", "b_anchors"),
    infer_key="a_vs_b",
    cluster_key="leiden_clusters",
    x_key="x",
    y_key="y",
    min_cluster_spots=10,
    min_anchor_pct=5,
    unassigned_label="novel_cluster",
    allow_novel_clusters=False,
    prop_diff_cutoff=None,
    anchor_normalizer=1.0,
    reassign_novel=True,
    num_nbs=25,
    print_results=True,
):
    """
    Assign hierarchical labels to spatial spots based on cluster-level anchor enrichment.

    Parameters
    ----------
    input_adata : AnnData
        Input AnnData object.
    hier_index : tuple or list
        Hierarchical labels to assign, for example ("a", "b").
    hier_anchor_key : tuple or list
        Columns in input_adata.obs storing anchor indicators or anchor scores
        for each hierarchy label.
    infer_key : str
        Column name in input_adata.obs for storing inferred labels.
    cluster_key : str
        Column in input_adata.obs storing cluster assignments.
    x_key, y_key : str
        Spatial coordinate columns in input_adata.obs.
    min_cluster_spots : int
        Clusters with fewer than this number of spots are not assigned by
        anchor enrichment and remain as unassigned_label.
    min_anchor_pct : float
        Minimum anchor percentage required for assigning a cluster to a
        hierarchy label.
    unassigned_label : str
        Label used for clusters that cannot be confidently assigned.
    allow_novel_clusters : bool
        If True, small novel clusters are allowed to remain novel. Larger
        novel clusters may still be reassigned by neighborhood smoothing.
    prop_diff_cutoff : float or None
        If not None, clusters whose top two hierarchy anchor percentages are
        within this cutoff are adjusted using anchor distribution weights.
        This is mainly useful for binary splits.
    anchor_normalizer : float or dict
        Normalizer for anchor values. Use 1.0 for binary anchors. For
        multi-modality summed anchors, use the total possible anchor score,
        for example 2.0 if Gene + Protein anchors are summed. 
        anchor_normalizer = (anchor_weight_G + anchor_weight_P) if having different weights
        A dict can also be used, e.g. {"a": 2.0, "b": 2.0}.
    reassign_novel : bool
        Whether to reassign novel clusters based on neighboring assigned spots.
    num_nbs : int
        Number of neighbors used when reassigning novel clusters.
    print_results : bool
        Whether to print summary tables.

    Returns
    -------
    LabelAssignmentResult
        Index-aligned labels, original and adjusted cluster-level anchor tables, 
        novel-cluster information, and assignment parameters.
    """

    if len(hier_index) != len(hier_anchor_key):
        raise ValueError("hier_index and hier_anchor_key must have the same length.")

    if min_cluster_spots < 1:
        raise ValueError("min_cluster_spots must be at least 1.")

    if num_nbs < 1:
        raise ValueError("num_nbs must be at least 1.")

    required_cols = [cluster_key, *hier_anchor_key]
    if reassign_novel:
        required_cols += [x_key, y_key]

    missing_cols = [col for col in required_cols if col not in input_adata.obs.columns]
    if len(missing_cols) > 0:
        raise KeyError(f"Missing columns in input_adata.obs: {missing_cols}")

    # Assignment uses only observation metadata. A zero-feature AnnData keeps
    # the helper interface while avoiding a full copy of the feature matrix.
    adata = input_adata[:, :0].copy()

    obs = adata.obs
    clusters = obs[cluster_key].dropna().unique().tolist()
    cluster_sizes = obs[cluster_key].value_counts()

    keep_clusters = cluster_sizes[cluster_sizes >= min_cluster_spots].index.tolist()
    drop_clusters = cluster_sizes[cluster_sizes < min_cluster_spots].index.tolist()

    if print_results and len(drop_clusters) > 0:
        print("Dropped small clusters:", ", ".join(map(str, drop_clusters)))

    # Initialize all spots as unassigned.
    adata.obs[infer_key] = unassigned_label

    # Build cluster-level anchor percentage table.
    cross_table = pd.DataFrame(
        0.0,
        index=list(hier_index),
        columns=clusters,
    )

    for label, anchor_key in zip(hier_index, hier_anchor_key):
        if isinstance(anchor_normalizer, dict):
            normalizer = anchor_normalizer.get(label, 1.0)
        else:
            normalizer = anchor_normalizer

        normalizer = float(normalizer)
        if normalizer <= 0:
            raise ValueError("anchor_normalizer must be positive.")

        anchor_values = pd.to_numeric(obs[anchor_key], errors="coerce").fillna(0)
        anchor_sum_by_cluster = anchor_values.groupby(obs[cluster_key]).sum()
        prop = anchor_sum_by_cluster / cluster_sizes / normalizer * 100

        cross_table.loc[label, prop.index] = prop

    if print_results:
        print("========== Cross Table of Anchors ==========")
        print(cross_table.round(2))

    # Use only sufficiently large clusters for initial assignment.
    assign_table = cross_table.loc[:, keep_clusters].copy()
    cross_table_upd = assign_table.copy()

    if assign_table.shape[1] == 0:
        adata.obs[infer_key] = adata.obs[infer_key].astype("category")
        labels = adata.obs[infer_key].astype(str).copy()
        labels.name = infer_key
        return LabelAssignmentResult(
            labels=labels,
            cross_table=cross_table,
            adjusted_cross_table=cross_table_upd,
            novel_clusters=[],
            unassigned_obs_names=labels.index.tolist(),
            adjustment_info={"adjustment_triggered": False},
            params={
                "infer_key": infer_key,
                "cluster_key": cluster_key,
                "hier_index": list(hier_index),
                "hier_anchor_key": list(hier_anchor_key),
                "unassigned_label": unassigned_label,
            },
        )

    max_label = assign_table.idxmax(axis=0)
    max_prop = assign_table.max(axis=0)
    adjustment_info = {
        "adjustment_triggered": False,
        "prop_diff_cutoff": prop_diff_cutoff,
        "prop_diff": pd.Series(dtype=float),
        "same_proportion_clusters": [],
        "similar_clusters": [],
        "anchor_distributions": {},
        "original_assignment": max_label.copy(),
        "adjusted_assignment": max_label.copy(),
        "changed_clusters": [],
    }

    # Optional adjustment for ambiguous binary clusters.
    if prop_diff_cutoff is not None and len(hier_index) == 2:
        prop_diff = (assign_table.iloc[0, :] - assign_table.iloc[1, :]).abs()
        similar_clusters = prop_diff[prop_diff <= prop_diff_cutoff].index.tolist()
        same_prop_clusters = prop_diff[np.isclose(prop_diff, 0.0)].index.tolist()
        original_assignment = max_label.copy()
        anchor_distributions = {}

        adjustment_info.update(
            {
                "prop_diff": prop_diff.copy(),
                "same_proportion_clusters": same_prop_clusters,
                "similar_clusters": similar_clusters,
                "original_assignment": original_assignment.copy(),
            }
        )

        if print_results:
            print("========== Absolute Anchor Proportion Difference ==========")
            print(prop_diff.sort_values())

        if len(similar_clusters) > 0:
            for label, anchor_key in zip(hier_index, hier_anchor_key):
                anchor_distribution = _anchor_distribution_by_cluster(
                    obs,
                    cluster_key=cluster_key,
                    anchor_key=anchor_key,
                )
                anchor_distributions[label] = anchor_distribution

                if anchor_distribution.empty:
                    continue

                anchor_cluster_prop = anchor_distribution["percentage"]

                for cluster in similar_clusters:
                    if cluster in anchor_cluster_prop.index:
                        cross_table_upd.loc[label, cluster] = (
                            cross_table.loc[label, cluster]
                            * anchor_cluster_prop.loc[cluster]
                        )

            max_label = cross_table_upd.idxmax(axis=0)
            max_prop = cross_table_upd.max(axis=0)
            changed_clusters = [
                cluster
                for cluster in assign_table.columns
                if str(original_assignment.loc[cluster]) != str(max_label.loc[cluster])
            ]

            adjustment_info.update(
                {
                    "adjustment_triggered": True,
                    "anchor_distributions": anchor_distributions,
                    "adjusted_assignment": max_label.copy(),
                    "changed_clusters": changed_clusters,
                }
            )

            if print_results:
                _print_binary_assignment_adjustment_report(
                    hier_index=hier_index,
                    prop_diff=prop_diff,
                    same_prop_clusters=same_prop_clusters,
                    similar_clusters=similar_clusters,
                    anchor_distributions=anchor_distributions,
                    cross_table_upd=cross_table_upd,
                    original_assignment=original_assignment,
                    adjusted_assignment=max_label,
                    changed_clusters=changed_clusters,
                )

    # Assign clusters by maximum anchor proportion.
    assigned_clusters = []

    for cluster in assign_table.columns:
        if max_prop.loc[cluster] > min_anchor_pct:
            label = max_label.loc[cluster]
            adata.obs.loc[adata.obs[cluster_key] == cluster, infer_key] = label
            assigned_clusters.append(cluster)

    novel_clusters = [
        cluster for cluster in keep_clusters
        if cluster not in assigned_clusters
    ]
    all_novel_clusters = list(novel_clusters)

    if print_results:
        print("Novel / unassigned clusters:", novel_clusters)
        print("========== Before Novel Cluster Reassignment ==========")
        print(adata.obs[infer_key].value_counts())

    # Optionally reassign novel clusters by spatial neighborhood.
    if reassign_novel and len(novel_clusters) > 0:
        if allow_novel_clusters:
            # Keep small novel clusters as novel; only reassign large novel clusters.
            cluster_props = cluster_sizes / cluster_sizes.sum()
            novel_clusters = [
                cluster for cluster in novel_clusters
                if cluster_props.loc[cluster] > 0.1
            ]

        for novel_cluster in novel_clusters:
            reassign_novel_cluster_by_knn(
                adata,
                novel_cluster=novel_cluster,
                cluster_key=cluster_key,
                infer_key=infer_key,
                x_key=x_key,
                y_key=y_key,
                unassigned_label=unassigned_label,
                num_nbs=num_nbs,
                copy=False,
                print_results=print_results,
            )

    adata.obs[infer_key] = adata.obs[infer_key].astype("category")

    if print_results:
        print("========== Inferred Labels ==========")
        print(adata.obs[infer_key].value_counts())

    labels = adata.obs[infer_key].astype(str).copy()
    labels.name = infer_key
    unassigned_obs_names = labels.index[
        labels.astype(str) == str(unassigned_label)
    ].tolist()

    return LabelAssignmentResult(
        labels=labels,
        cross_table=cross_table,
        adjusted_cross_table=cross_table_upd,
        novel_clusters=all_novel_clusters,
        unassigned_obs_names=unassigned_obs_names,
        adjustment_info=adjustment_info,
        params={
            "infer_key": infer_key,
            "cluster_key": cluster_key,
            "hier_index": list(hier_index),
            "hier_anchor_key": list(hier_anchor_key),
            "unassigned_label": unassigned_label,
            "min_cluster_spots": min_cluster_spots,
            "min_anchor_pct": min_anchor_pct,
            "allow_novel_clusters": allow_novel_clusters,
            "prop_diff_cutoff": prop_diff_cutoff,
            "anchor_normalizer": anchor_normalizer,
            "reassign_novel": reassign_novel,
            "num_nbs": num_nbs,
        },
    )


def reassign_novel_cluster_by_knn(
    input_adata,
    novel_cluster,
    cluster_key,
    infer_key,
    x_key="x",
    y_key="y",
    unassigned_label="novel_cluster",
    num_nbs=25,
    metric="euclidean",
    copy=False,
    print_results=True,
):
    """
    Reassign one novel cluster based on the labels of nearby assigned spots.

    Parameters
    ----------
    input_adata : AnnData
        Input AnnData object.
    novel_cluster : str or int
        Cluster ID to reassign.
    cluster_key : str
        Column in input_adata.obs storing cluster assignments.
    infer_key : str
        Column in input_adata.obs storing current inferred labels.
    x_key, y_key : str
        Spatial coordinate columns.
    unassigned_label : str
        Label used for unassigned or novel spots.
    num_nbs : int
        Number of nearest assigned spots used for reassignment.
    metric : str
        Distance metric used by sklearn.neighbors.NearestNeighbors.
    copy : bool
        If True, return a copied AnnData object. If False, modify in place.
    print_results : bool
        Whether to print reassignment summary.

    Returns
    -------
    adata : AnnData
        AnnData object with the novel cluster reassigned if possible.
    """

    if num_nbs < 1:
        raise ValueError("num_nbs must be at least 1.")

    required_cols = [cluster_key, infer_key, x_key, y_key]
    missing_cols = [col for col in required_cols if col not in input_adata.obs.columns]
    if len(missing_cols) > 0:
        raise KeyError(f"Missing columns in input_adata.obs: {missing_cols}")

    adata = input_adata.copy() if copy or input_adata.is_view else input_adata
    obs = adata.obs

    novel_mask = obs[cluster_key] == novel_cluster
    assigned_mask = obs[infer_key].astype(str) != str(unassigned_label)

    if novel_mask.sum() == 0:
        return adata

    if assigned_mask.sum() == 0:
        if print_results:
            print(f"No assigned spots available to reassign cluster {novel_cluster}.")
        return adata

    novel_coords = obs.loc[novel_mask, [x_key, y_key]].apply(pd.to_numeric, errors="coerce").to_numpy()
    assigned_coords = obs.loc[assigned_mask, [x_key, y_key]].apply(pd.to_numeric, errors="coerce").to_numpy()
    assigned_labels = obs.loc[assigned_mask, infer_key].astype(str).to_numpy()

    novel_valid = np.isfinite(novel_coords).all(axis=1)
    assigned_valid = np.isfinite(assigned_coords).all(axis=1)

    if assigned_valid.sum() == 0:
        if print_results:
            print(f"No assigned spots with valid coordinates for cluster {novel_cluster}.")
        return adata

    if novel_valid.sum() == 0:
        if print_results:
            print(f"No novel spots with valid coordinates for cluster {novel_cluster}.")
        return adata

    novel_index = obs.index[novel_mask][novel_valid]
    novel_coords = novel_coords[novel_valid]
    assigned_coords = assigned_coords[assigned_valid]
    assigned_labels = assigned_labels[assigned_valid]

    k = min(num_nbs, assigned_coords.shape[0])

    nbrs = NearestNeighbors(n_neighbors=k, metric=metric)
    nbrs.fit(assigned_coords)

    _, indices = nbrs.kneighbors(novel_coords)

    neighbor_labels = assigned_labels[indices].ravel()
    label_counts = pd.Series(neighbor_labels).value_counts()

    if label_counts.empty:
        return adata

    reassigned_label = label_counts.idxmax()

    adata.obs.loc[novel_index, infer_key] = reassigned_label

    if print_results:
        print(f"---------------- novel cluster {novel_cluster} ----------------")
        print(label_counts)
        print(
            f"Based on neighborhood composition, novel cluster "
            f"{novel_cluster} reassignment: {reassigned_label}"
        )

    return adata


# =====================================================================
# Adjust label assignment when outputting one side binary assignment
# =====================================================================
def calculate_binary_nodes_ratio_multiref(
    ref_gene_sca_dic,
    target_regions,
    nontgt_regions,
    label_key="label",
    ratio_mode="pooled",
    per_section_agg="max",
    skip_empty_sections=True,
):
    """
    Calculate binary-node spot-count ratio from one or multiple reference sections.

    Parameters
    ----------
    ref_gene_sca_dic : Mapping[str, AnnData] or AnnData
        Reference AnnData dictionary, where keys are reference section names and
        values are AnnData objects. Label-transfer sessions always provide this
        section-keyed mapping. A single AnnData is also accepted for backward 
        compatibility with direct function calls. 

    target_regions : sequence of str or str
        Region labels belonging to the target side of the binary split.

    nontgt_regions : sequence of str or str
        Region labels belonging to the non-target side of the binary split.

    label_key : str, default="label"
        Column in each reference AnnData `.obs` containing region labels.

    ratio_mode : {"pooled", "per_section"}, default="pooled"
        Method for calculating the final binary-node ratio.

        - "pooled":
            Sum target spots across all reference sections and sum non-target
            spots across all reference sections, then calculate one ratio.

        - "per_section":
            Calculate one ratio within each reference section, then aggregate
            ratios using `per_section_agg`.

    per_section_agg : {"max", "median", "mean"}, default="max"
        Aggregation method used when `ratio_mode="per_section"`.

    skip_empty_sections : bool, default=True
        If True, sections where both target and non-target spots are zero are
        skipped. If False, such sections raise an error.

    Returns
    -------
    binary_nodes_ratio : float
        Final binary-node ratio, calculated as max(counts) / min(counts).

    ratio_info : dict
        Summary of pooled and per-section spot counts and ratios.
    """

    # Convert target_regions directly.
    if target_regions is None:
        target_regions = []
    elif isinstance(target_regions, str):
        target_regions = [target_regions]
    else:
        target_regions = list(target_regions)

    # Convert nontgt_regions directly.
    if nontgt_regions is None:
        nontgt_regions = []
    elif isinstance(nontgt_regions, str):
        nontgt_regions = [nontgt_regions]
    else:
        nontgt_regions = list(nontgt_regions)

    target_regions = [str(region) for region in target_regions]
    nontgt_regions = [str(region) for region in nontgt_regions]

    if len(target_regions) == 0:
        raise ValueError("`target_regions` must contain at least one region.")

    if len(nontgt_regions) == 0:
        raise ValueError("`nontgt_regions` must contain at least one region.")

    # Normalize reference input directly.
    if isinstance(ref_gene_sca_dic, Mapping):
        ref_gene_sca_dic = dict(ref_gene_sca_dic)
    else:
        ref_gene_sca_dic = {"reference": ref_gene_sca_dic}

    if len(ref_gene_sca_dic) == 0:
        raise ValueError("`ref_gene_sca_dic` must contain at least one reference section.")

    if ratio_mode not in {"pooled", "per_section"}:
        raise ValueError("`ratio_mode` must be either 'pooled' or 'per_section'.")

    if per_section_agg not in {"max", "median", "mean"}:
        raise ValueError("`per_section_agg` must be one of {'max', 'median', 'mean'}.")

    section_ratio_info = {}
    total_target_spots = 0
    total_nontgt_spots = 0

    for ref_section, ref_adata in ref_gene_sca_dic.items():
        if label_key not in ref_adata.obs:
            raise KeyError(
                f"`label_key='{label_key}'` is not found in "
                f"`ref_gene_sca_dic['{ref_section}'].obs`."
            )

        ref_region_spots = ref_adata.obs[label_key].astype(str).value_counts()

        target_spots_num = int(
            ref_region_spots.reindex(target_regions, fill_value=0).sum()
        )
        nontgt_spots_num = int(
            ref_region_spots.reindex(nontgt_regions, fill_value=0).sum()
        )

        if target_spots_num == 0 and nontgt_spots_num == 0:
            if skip_empty_sections:
                section_ratio_info[ref_section] = {
                    "target_spots_num": target_spots_num,
                    "nontgt_spots_num": nontgt_spots_num,
                    "binary_nodes_ratio": None,
                    "used": False,
                    "reason": "Both target and non-target spots are zero.",
                }
                continue

            raise ValueError(
                f"Reference section '{ref_section}' has zero spots for both "
                "`target_regions` and `nontgt_regions`."
            )

        if min(target_spots_num, nontgt_spots_num) == 0:
            section_ratio = np.inf
        else:
            section_ratio = max(target_spots_num, nontgt_spots_num) / min(
                target_spots_num, nontgt_spots_num
            )

        total_target_spots += target_spots_num
        total_nontgt_spots += nontgt_spots_num

        section_ratio_info[ref_section] = {
            "target_spots_num": target_spots_num,
            "nontgt_spots_num": nontgt_spots_num,
            "binary_nodes_ratio": float(section_ratio),
            "used": True,
            "reason": None,
        }

    used_ratios = [
        info["binary_nodes_ratio"]
        for info in section_ratio_info.values()
        if info["used"]
    ]

    if len(used_ratios) == 0:
        raise ValueError(
            "No valid reference sections were available for binary-node ratio calculation."
        )

    if ratio_mode == "pooled":
        if total_target_spots == 0 and total_nontgt_spots == 0:
            raise ValueError(
                "Total target and non-target spots are both zero across reference sections."
            )

        if min(total_target_spots, total_nontgt_spots) == 0:
            binary_nodes_ratio = np.inf
        else:
            binary_nodes_ratio = max(total_target_spots, total_nontgt_spots) / min(
                total_target_spots, total_nontgt_spots
            )

    else:
        if per_section_agg == "max":
            binary_nodes_ratio = np.max(used_ratios)
        elif per_section_agg == "median":
            binary_nodes_ratio = np.median(used_ratios)
        elif per_section_agg == "mean":
            binary_nodes_ratio = np.mean(used_ratios)

    ratio_info = {
        "ratio_mode": ratio_mode,
        "per_section_agg": per_section_agg if ratio_mode == "per_section" else None,
        "target_regions": target_regions,
        "nontgt_regions": nontgt_regions,
        "total_target_spots_num": int(total_target_spots),
        "total_nontgt_spots_num": int(total_nontgt_spots),
        "binary_nodes_ratio": float(binary_nodes_ratio),
        "section_ratio_info": section_ratio_info,
    }

    return float(binary_nodes_ratio), ratio_info


def adjust_one_side_binary_assignment(
    test_gene_sca,
    ref_gene_sca_dic,
    target_regions,
    nontgt_regions,
    binary_ratio_thres,
    assignment_result=None,
    ratio_mode="pooled",
    per_section_agg="max",
    skip_empty_sections=True,
    hier_index=("a", "b"),
    hier_anchor_key=("a_anchors", "b_anchors"),
    infer_key="a_vs_b",
    cluster_key="leiden_clusters",
    label_key="label",
    x_key="x",
    y_key="y",
    min_cluster_spots=10,
    min_anchor_pct=5,
    unassigned_label="novel_cluster",
    allow_novel_clusters=False,
    prop_diff_cutoff_upd=100,
    anchor_normalizer=1.0,
    reassign_novel=True,
    num_nbs=25,
    print_results=True,
    **assign_kwargs,
):
    """
    Adjust a one-sided binary label assignment result.

    This function checks whether binary assignment produced only one non-novel
    category. If so, it compares the target/non-target region ratio across
    reference sections. When the ratio exceeds ``binary_ratio_thres``, label
    assignment is rerun with an updated ``prop_diff_cutoff``.

    The function always works on a private copy of ``test_gene_sca`` and does
    not mutate the input AnnData object.

    Parameters
    ----------
    test_gene_sca : AnnData
        Query AnnData object containing the current assignment labels, anchors,
        clustering results, and spatial coordinates.

    ref_gene_sca_dic : Mapping[str, AnnData]
        Dictionary of reference AnnData objects, keyed by reference section name.
        Used to calculate the target/non-target binary-node ratio.

    target_regions : Sequence[str]
        Region labels corresponding to the target side of the binary split.

    nontgt_regions : Sequence[str]
        Region labels corresponding to the non-target side of the binary split.

    binary_ratio_thres : float
        Threshold for triggering reassignment. Reassignment is performed only
        when the reference-derived binary-node ratio is greater than this value.

    assignment_result : LabelAssignmentResult, optional
        Existing assignment result to adjust. If not provided, labels are read
        from ``test_gene_sca.obs[infer_key]`` for backward compatibility.

    ratio_mode : {"pooled", "per_section"}, default="pooled"
        Method used to calculate the binary-node ratio across reference sections.

    per_section_agg : {"max", "mean", "median"}, default="max"
        Aggregation method used when ``ratio_mode`` calculates section-level ratios.

    skip_empty_sections : bool, default=True
        Whether to skip reference sections that do not contain the target or
        non-target regions.

    hier_index : Sequence[str], default=("a", "b")
        Names of the two binary split branches.

    hier_anchor_key : Sequence[str], default=("a_anchors", "b_anchors")
        Observation column names storing anchor indicators for each split branch.

    infer_key : str, default="a_vs_b"
        Observation column name used for inferred binary labels.

    cluster_key : str, default="leiden_clusters"
        Observation column name storing query cluster labels.

    label_key : str, default="label"
        Observation column name storing reference region labels.

    x_key, y_key : str, default=("x", "y")
        Observation column names storing spatial coordinates.

    min_cluster_spots : int, default=10
        Minimum number of spots required for a cluster to be considered.

    min_anchor_pct : float, default=5
        Minimum anchor percentage required for assigning a cluster to a branch.

    unassigned_label : str, default="novel_cluster"
        Label used for unassigned or novel clusters.

    allow_novel_clusters : bool, default=False
        Whether to allow clusters to remain assigned as novel clusters.

    prop_diff_cutoff_upd : float, default=100
        Updated proportion-difference cutoff used when reassignment is triggered.

    anchor_normalizer : float, default=1.0
        Normalization factor applied to anchor counts or proportions.

    reassign_novel : bool, default=True
        Whether to reassign novel clusters using nearest-neighbor refinement.

    num_nbs : int, default=25
        Number of neighbors used for novel-cluster reassignment.

    print_results : bool, default=True
        Whether to print progress and adjustment messages.

    **assign_kwargs
        Additional keyword arguments passed to ``assign_hierarchical_labels``.

    Returns
    -------
    LabelAssignmentResult
        Assignment result with ``adjustment_info`` added. If adjustment is not
        triggered, the original assignment result is returned with diagnostic
        information. If adjustment is triggered, the updated assignment result
        from ``assign_hierarchical_labels`` is returned.

    Raises
    ------
    ValueError
        If ``binary_ratio_thres`` is not positive, branch/anchor inputs have
        inconsistent lengths, required regions are missing, or observations in
        ``assignment_result`` are absent from ``test_gene_sca``.
    KeyError
        If ``assignment_result`` is not supplied and ``infer_key`` is missing
        from ``test_gene_sca.obs``.
    TypeError
        If ``assignment_result`` is supplied but is not a ``LabelAssignmentResult``.
    """

    if binary_ratio_thres <= 0:
        raise ValueError("binary_ratio_thres must be positive.")

    if len(hier_index) != len(hier_anchor_key):
        raise ValueError("hier_index and hier_anchor_key must have the same length.")

    if target_regions is None or nontgt_regions is None:
        raise ValueError("Both `target_regions` and `nontgt_regions` must be provided.")

    # This adjustment reads and updates only ``obs`` columns.
    adata = test_gene_sca[:, :0].copy()

    if assignment_result is None:
        if infer_key not in adata.obs:
            raise KeyError(
                f"infer_key={infer_key!r} is missing and assignment_result was not supplied."
            )
        labels = adata.obs[infer_key].astype(str).copy()
        labels.name = infer_key
        assignment_result = LabelAssignmentResult(
            labels=labels,
            cross_table=pd.DataFrame(),
            adjusted_cross_table=pd.DataFrame(),
            params={"infer_key": infer_key, "unassigned_label": unassigned_label},
        )
    elif not isinstance(assignment_result, LabelAssignmentResult):
        raise TypeError("assignment_result must be a LabelAssignmentResult.")

    missing_obs = assignment_result.labels.index.difference(adata.obs_names)
    if len(missing_obs) > 0:
        raise ValueError(
            "test_gene_sca is missing observations from assignment_result. "
            f"Examples: {missing_obs[:5].tolist()}"
        )

    assigned_categories = (
        assignment_result.labels.dropna().astype(str).value_counts().index.tolist()
    )

    binary_categories = [
        cate for cate in assigned_categories
        if str(cate) != str(unassigned_label)
    ]

    adjust_info = {
        "num_binary_categories": len(binary_categories),
        "binary_categories": binary_categories,
        "unassigned_label": unassigned_label,
        "binary_nodes_ratio": None,
        "ratio_info": None,
        "adjustment_triggered": False,
        "reason": None,
        "prop_diff_cutoff_used": None,
        "cross_table": None,
        "cross_table_upd": None,
    }

    if len(binary_categories) != 1:
        adjust_info["reason"] = (
            "Binary assignment did not produce exactly one non-novel category."
        )
        return replace(assignment_result, adjustment_info=adjust_info)

    if print_results:
        print("******************************* Outputting 1 side separation *******************************")

    binary_nodes_ratio, ratio_info = calculate_binary_nodes_ratio_multiref(
        ref_gene_sca_dic=ref_gene_sca_dic,
        target_regions=target_regions,
        nontgt_regions=nontgt_regions,
        label_key=label_key,
        ratio_mode=ratio_mode,
        per_section_agg=per_section_agg,
        skip_empty_sections=skip_empty_sections,
    )

    adjust_info["binary_nodes_ratio"] = binary_nodes_ratio
    adjust_info["ratio_info"] = ratio_info

    if print_results:
        print(f"binary nodes ratio: {binary_nodes_ratio}")

    if binary_nodes_ratio <= binary_ratio_thres:
        adjust_info["reason"] = (
            f"Binary nodes ratio <= binary_ratio_thres "
            f"({binary_nodes_ratio} <= {binary_ratio_thres}); skipped adjustment."
        )
        return replace(assignment_result, adjustment_info=adjust_info)

    if print_results:
        print(
            "============================ binary_nodes_ratio is larger than "
            f"{binary_ratio_thres} ============================"
        )
        print("**************************** label assignment adjusted by weights ****************************")

    assignment_params = {
        "input_adata": adata,
        "hier_index": hier_index,
        "hier_anchor_key": hier_anchor_key,
        "infer_key": infer_key,
        "cluster_key": cluster_key,
        "x_key": x_key,
        "y_key": y_key,
        "min_cluster_spots": min_cluster_spots,
        "min_anchor_pct": min_anchor_pct,
        "unassigned_label": unassigned_label,
        "allow_novel_clusters": allow_novel_clusters,
        "prop_diff_cutoff": prop_diff_cutoff_upd,
        "anchor_normalizer": anchor_normalizer,
        "reassign_novel": reassign_novel,
        "num_nbs": num_nbs,
        "print_results": print_results,
    }
    assignment_params.update(assign_kwargs)

    adjusted_result = assign_hierarchical_labels(**assignment_params)

    adjust_info["adjustment_triggered"] = True
    adjust_info["reason"] = (
        "One-sided assignment detected and binary-node ratio exceeded threshold."
    )
    adjust_info["prop_diff_cutoff_used"] = prop_diff_cutoff_upd
    adjust_info["cross_table"] = adjusted_result.cross_table
    adjust_info["cross_table_upd"] = adjusted_result.adjusted_cross_table

    return replace(adjusted_result, adjustment_info=adjust_info)


def refine_labels(
    input_adata,
    pred_key,
    refined_key,
    num_nbs=25,
    x_key="x",
    y_key="y",
    dists_metric="euclidean",
    copy=True,
):
    """
    Refine spot-level labels using spatial nearest-neighbor majority voting.

    Parameters
    ----------
    input_adata : AnnData
        Input AnnData object.
    pred_key : str
        Column in input_adata.obs containing original predicted labels.
    refined_key : str
        Column name for storing refined labels.
    num_nbs : int
        Number of nearest neighbors used for majority voting. The spot itself
        is also included internally when available.
    x_key, y_key : str
        Spatial coordinate columns in input_adata.obs.
    dists_metric : str
        Distance metric used by sklearn.neighbors.NearestNeighbors.
    copy : bool, default=True
        If True, return a copied AnnData object. Set False only when the caller
        owns the object and accepts in-place modification.

    Returns
    -------
    adata : AnnData
        AnnData object with refined labels in adata.obs[refined_key].
    refined_pred : list
        Refined label for each spot, in the same order as adata.obs.
    """

    if num_nbs < 1:
        raise ValueError("num_nbs must be at least 1.")

    required_cols = [pred_key, x_key, y_key]
    missing_cols = [col for col in required_cols if col not in input_adata.obs.columns]
    if len(missing_cols) > 0:
        raise KeyError(f"Missing columns in input_adata.obs: {missing_cols}")

    adata = input_adata.copy() if copy or input_adata.is_view else input_adata

    obs = adata.obs
    pred = obs[pred_key].astype(str).to_numpy()
    coords = obs[[x_key, y_key]].apply(pd.to_numeric, errors="coerce").to_numpy()

    if coords.shape[0] == 0:
        adata.obs[refined_key] = pd.Categorical([])
        return adata, []

    valid_mask = np.isfinite(coords).all(axis=1)
    if valid_mask.sum() == 0:
        adata.obs[refined_key] = pd.Categorical(pred)
        return adata, pred.tolist()

    refined_pred = pred.copy()
    valid_indices = np.where(valid_mask)[0]
    valid_coords = coords[valid_mask]
    valid_pred = pred[valid_mask]

    k = min(num_nbs + 1, valid_coords.shape[0])

    nbrs = NearestNeighbors(n_neighbors=k, metric=dists_metric)
    nbrs.fit(valid_coords)

    _, indices = nbrs.kneighbors(valid_coords)

    for local_i, nb_idx in enumerate(indices):
        nb_labels = valid_pred[nb_idx]
        self_label = valid_pred[local_i]

        label_counts = pd.Series(nb_labels).value_counts()
        top_label = label_counts.idxmax()
        top_count = label_counts.max()

        # Use the effective neighborhood size instead of the requested num_nbs,
        # so the function behaves correctly for small datasets.
        if top_label != self_label and top_count > len(nb_labels) / 2:
            refined_pred[valid_indices[local_i]] = top_label

    adata.obs[refined_key] = pd.Categorical(refined_pred, categories=pd.unique(refined_pred))

    return adata, refined_pred.tolist()


def sudo_to_spot_annotation(
    spot_obs,
    sudo_obs,
    num_nbs,
    spot_x_key,
    spot_y_key,
    sudo_x_key,
    sudo_y_key,
    annotation_key,
    small_region_adjustment=False,
    small_region_thres=0.25,
    dominant_region_thres=0.5,
    novel_label="novel_cluster",
    unknown_label="nan",
    neighbor_mode="knn",
    copy=True,
    print_results=True,
):
    """
    Transfer annotations from sudo-level observations to spot-level observations.

    For each spot, nearby sudo observations are identified. The spot annotation is
    assigned based on the majority annotation among neighboring sudo observations.

    Optionally, this function can adjust for sparse/small regions. If the most common
    local label is a large region but the second most common label is a globally small
    region with sufficient local proportion, the small region can be selected instead.

    Parameters
    ----------
    spot_obs : pd.DataFrame
        Spot-level observation dataframe.
    sudo_obs : pd.DataFrame
        Sudo-level observation dataframe.
    num_nbs : int
        Number of nearest sudo observations used for annotation if neighbor_mode="knn".
        If neighbor_mode="radius_quantile", this controls the global distance quantile.
    spot_x_key, spot_y_key : str
        Column names for spot-level x/y coordinates.
    sudo_x_key, sudo_y_key : str
        Column names for sudo-level x/y coordinates.
    annotation_key : str
        Column name of sudo-level annotations to transfer.
    small_region_adjustment : bool, default=False
        Whether to apply sparse/small-region adjustment.
    small_region_thres : float, default=0.25
        Threshold used to define globally small regions and local small-region support.
    dominant_region_thres : float, default=0.5
        Apply small-region adjustment only if one region dominates the sudo data.
    novel_label : str, default="novel_cluster"
        Label to skip when assigning the final annotation if another label is available.
    unknown_label : str, default="nan"
        Initial annotation for spots without nearby sudo observations.
    neighbor_mode : {"knn", "radius_quantile"}, default="knn"
        - "knn": use exactly num_nbs nearest sudo observations for each spot.
        - "radius_quantile": original global distance-threshold logic.
    copy : bool, default=True
        Whether to copy spot_obs before modification.
    print_results : bool, default=True
        Whether to print annotation summaries.

    Returns
    -------
    spot_obs : pd.DataFrame
        Updated spot-level observation dataframe.
    annotation_key : str
        Name of the transferred annotation column.
    """

    if copy:
        spot_obs = spot_obs.copy()

    required_spot_cols = [spot_x_key, spot_y_key]
    required_sudo_cols = [sudo_x_key, sudo_y_key, annotation_key]

    missing_spot_cols = [col for col in required_spot_cols if col not in spot_obs.columns]
    missing_sudo_cols = [col for col in required_sudo_cols if col not in sudo_obs.columns]

    if len(missing_spot_cols) > 0:
        raise KeyError(f"Missing columns in spot_obs: {missing_spot_cols}")

    if len(missing_sudo_cols) > 0:
        raise KeyError(f"Missing columns in sudo_obs: {missing_sudo_cols}")

    if num_nbs <= 0:
        raise ValueError("num_nbs must be a positive integer.")
    if spot_obs.empty:
        raise ValueError("spot_obs must contain at least one observation.")
    if sudo_obs.empty:
        raise ValueError("sudo_obs must contain at least one observation.")

    # ------------------------------------------------------------
    # Obtain spatial coordinates
    # ------------------------------------------------------------
    spot_coords = spot_obs[[spot_x_key, spot_y_key]].apply(
        pd.to_numeric,
        errors="raise",
    ).to_numpy(dtype=float)
    sudo_coords = sudo_obs[[sudo_x_key, sudo_y_key]].apply(
        pd.to_numeric,
        errors="raise",
    ).to_numpy(dtype=float)

    if not np.isfinite(spot_coords).all() or not np.isfinite(sudo_coords).all():
        raise ValueError("Spot and pseudo-observation coordinates must be finite.")

    # ------------------------------------------------------------
    # Identify tissue regions and initialize probability columns
    # ------------------------------------------------------------
    tissue_sections = (
        pd.Series(sudo_obs[annotation_key])
        .dropna()
        .value_counts()
        .index
        .tolist()
    )

    sections_prob = [f"{section}_prob" for section in tissue_sections]

    for col in sections_prob:
        spot_obs[col] = 0.0

    spot_obs[annotation_key] = unknown_label

    # ------------------------------------------------------------
    # Identify globally small regions, if requested
    # ------------------------------------------------------------
    tissue_prop = pd.Series(sudo_obs[annotation_key]).value_counts(normalize=True)

    if small_region_adjustment and tissue_prop.max() >= dominant_region_thres:
        small_prop_regions = tissue_prop[tissue_prop < small_region_thres].index.tolist()
        small_prop_regions = [
            region for region in small_prop_regions
            if region != novel_label
        ]
    else:
        small_prop_regions = []

    if print_results and small_region_adjustment:
        print(
            f"Based on the proportion threshold of {small_region_thres}, "
            f"small regions include: {', '.join(map(str, small_prop_regions))}"
        )

    # ------------------------------------------------------------
    # Find sudo neighbors for each spot
    # ------------------------------------------------------------
    if neighbor_mode == "knn":
        tree = cKDTree(sudo_coords)

        k = min(num_nbs, sudo_obs.shape[0])
        _, neighbor_indices = tree.query(spot_coords, k=k)

        if k == 1:
            neighbor_indices = neighbor_indices[:, None]

    elif neighbor_mode == "radius_quantile":

        dists = cdist(spot_coords, sudo_coords, metric="euclidean")
        radius_quantile = min(1.0, num_nbs / sudo_obs.shape[0])
        dists_threshold = np.quantile(dists.flatten(), radius_quantile)
        neighbor_indices = [
            np.where(dists[i, :] <= dists_threshold)[0]
            for i in range(spot_obs.shape[0])
        ]

    else:
        raise ValueError("neighbor_mode must be either 'knn' or 'radius_quantile'.")

    # ------------------------------------------------------------
    # Infer spot-level annotations
    # ------------------------------------------------------------
    for i in range(spot_obs.shape[0]):

        if neighbor_mode == "knn":
            sudo_indices = neighbor_indices[i]
        else:
            sudo_indices = neighbor_indices[i]

        if len(sudo_indices) == 0:
            continue

        pred_tmp = sudo_obs[annotation_key].iloc[sudo_indices]
        pred_labels_prop = pred_tmp.value_counts(normalize=True)

        pred_labels = pred_labels_prop.index.tolist()

        # Store local annotation probabilities
        for label in pred_labels:
            prob_col = f"{label}_prob"
            if prob_col in spot_obs.columns:
                spot_obs.loc[spot_obs.index[i], prob_col] = pred_labels_prop[label]

        # Case 1: skip novel_cluster if another candidate exists
        if pred_labels[0] == novel_label and len(pred_labels) > 1:
            final_label = pred_labels[1]

        # Case 2: optionally adjust for sparse/small regions
        elif (
            small_region_adjustment
            and len(pred_labels) > 1
            and pred_labels[0] not in small_prop_regions
            and pred_labels[1] in small_prop_regions
            and pred_labels_prop[pred_labels[1]] > small_region_thres
        ):
            final_label = pred_labels[1]

            if print_results:
                print("************ small region proportion adjustment ************")
                print(
                    f"{final_label} with a local proportion of "
                    f"{round(pred_labels_prop[final_label], 2)}"
                )

        # Case 3: default majority vote
        else:
            final_label = pred_labels[0]

        spot_obs.loc[spot_obs.index[i], annotation_key] = final_label

    spot_obs[annotation_key] = spot_obs[annotation_key].astype("category")

    if print_results:
        print("======================= Tissue region proportions in spot data =======================")
        print(pd.Series(spot_obs[annotation_key]).value_counts(normalize=True))

        print("======================= Tissue region proportions in sudo data =======================")
        print(pd.Series(sudo_obs[annotation_key]).value_counts(normalize=True))

    return spot_obs, annotation_key


def pseudo_to_spot_annotation(
    spot_obs,
    pseudo_obs,
    num_nbs,
    spot_x_key,
    spot_y_key,
    pseudo_x_key,
    pseudo_y_key,
    annotation_key,
    **kwargs,
):
    """Correctly spelled interface for :func:`sudo_to_spot_annotation`.

    The historical function name and ``sudo_*`` parameters remain available for
    backward compatibility. New code should use this interface.
    """
    return sudo_to_spot_annotation(
        spot_obs=spot_obs,
        sudo_obs=pseudo_obs,
        num_nbs=num_nbs,
        spot_x_key=spot_x_key,
        spot_y_key=spot_y_key,
        sudo_x_key=pseudo_x_key,
        sudo_y_key=pseudo_y_key,
        annotation_key=annotation_key,
        **kwargs,
    )
