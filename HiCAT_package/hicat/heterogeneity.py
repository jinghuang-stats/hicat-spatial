import os
import numpy as np
import pandas as pd
import seaborn as sns

from itertools import combinations
from scipy.sparse import issparse
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import fcluster
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

# Local package imports
from .preprocessing import (
    construct_merged_scaled_adata_and_gene_df,
    subset_adata_dic_by_region,
)
from .utils import (
    compute_pca_embedding,
    get_gene_vector,
    get_region_genes,
    kmeans_clustering,
    leiden_clustering,
    select_region_markers_across_samples,
)
from .visualization import cat_figure, con_figure, get_cluster_palette


def _default_cat_color(cat_color):
    """Return a reusable categorical palette when none is provided."""
    if cat_color is None:
        return sns.color_palette("tab20", n_colors=20).as_hex()
    return list(cat_color)


#=======================================================================
# Heterogeneity result objects
#=======================================================================
@dataclass
class HeterogeneousRegionSubtypeResult:
    """
    Store subtype-discovery results for one heterogeneous tissue region.

    Attributes
    ----------
    target_region : str
        Tissue region selected as heterogeneous.
    section_subtype_genes : dict
        Section-level subtype marker gene unions returned by
        section_subtype_DE_genes().
    total_genes_list : list
        Union of subtype marker genes across sections used for shared subtype
        clustering.
    total_clusters_num : int
        Total number of section-level subtype clusters passing filtering.
    subtype_clusters : pandas.DataFrame or None
        Shared subtype cluster assignments for merged spots/cells.
    shared_subtype_genes_merged : dict
        Shared subtype marker genes identified from merged-section DE analysis.
    shared_subtype_genes_individual : dict
        Shared subtype marker genes identified by per-section DE plus overlap.
    parameters : dict
        Parameters used for this region-level subtype analysis.
    """

    target_region: str
    section_subtype_genes: Dict[str, List[str]] = field(default_factory=dict)
    total_genes_list: List[str] = field(default_factory=list)
    total_clusters_num: int = 0
    subtype_clusters: Optional[pd.DataFrame] = None
    shared_subtype_genes_merged: Dict[str, List[str]] = field(default_factory=dict)
    shared_subtype_genes_individual: Dict[str, List[str]] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReferenceHeterogeneityResult:
    """
    Store reference-data heterogeneity results across regions and samples.

    This object is intended as the main return object for the reference
    heterogeneity pipeline. It keeps region-level heterogeneity scores, selected
    heterogeneous regions, and subtype results for each selected region.
    """

    dataset_name: Optional[str] = None
    hetero_summary: Optional[pd.DataFrame] = None
    sta_summary: Optional[pd.DataFrame] = None
    perm_sil_summary: Optional[pd.DataFrame] = None
    selected_regions: List[str] = field(default_factory=list)
    selected_region_scores: Optional[pd.DataFrame] = None
    region_marker_genes: Dict[str, List[str]] = field(default_factory=dict)
    selection_method: str = "threshold"
    selection_params: Dict[str, Any] = field(default_factory=dict)
    subtype_results: Dict[str, HeterogeneousRegionSubtypeResult] = field(default_factory=dict)
    sample_names: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)

    def get_region_subtypes(self, region: str) -> Optional[HeterogeneousRegionSubtypeResult]:
        """Return subtype results for one selected heterogeneous region."""
        return self.subtype_results.get(region)

    def get_region_marker_genes(self, region: str) -> List[str]:
        """Return region-specific marker genes used for heterogeneity scoring."""
        return self.region_marker_genes.get(region, [])


#=======================================================================
# Part 1. Infer region-specific heterogeneous scores across samples 
#=======================================================================
def jaccard(a, b):
    """
    Compute Jaccard similarity between two gene sets.
    """

    a = set(a)
    b = set(b)

    if len(a) == 0 and len(b) == 0:
        return 1.0

    return len(a & b) / len(a | b)


def marker_stability_jaccard(marker_dict):
    """
    Compute marker gene stability across samples using pairwise Jaccard similarity.

    Parameters
    ----------
    marker_dict : dict
        Dictionary of sample-specific marker genes.
        Example:
        {
            "H1": ["geneA", "geneB"],
            "G2": ["geneB", "geneC"]
        }

    Returns
    -------
    result : dict
        {
            "stability": mean pairwise Jaccard similarity,
            "pairwise": array of pairwise Jaccard values,
            "pairs": sample pairs
        }
    """

    samples = list(marker_dict.keys())
    pairs = list(combinations(samples, 2))

    vals = []

    for s1, s2 in pairs:
        vals.append(jaccard(marker_dict[s1], marker_dict[s2]))

    if len(vals) == 0:
        return {
            "stability": np.nan,
            "pairwise": np.array([]),
            "pairs": pairs
        }

    return {
        "stability": float(np.mean(vals)),
        "pairwise": np.array(vals),
        "pairs": pairs
    }


def perm_adjusted_silhouette(
    X_embed,
    labels,
    n_perm=200,
    random_state=0,
    one_sided=True
):
    """
    Compute permutation-adjusted silhouette score.

    Parameters
    ----------
    X_embed : array-like
        Feature matrix.

    labels : array-like
        Group labels, usually sample IDs.

    n_perm : int
        Number of permutations.

    random_state : int
        Random seed.

    one_sided : bool
        Whether to use one-sided permutation p-value.

    Returns
    -------
    result : dict
        Observed silhouette, permutation mean/std, adjusted silhouette,
        z-score, and p-value.
    """

    X_embed = X_embed.toarray() if issparse(X_embed) else np.asarray(X_embed)

    labels = np.asarray(labels).ravel()

    # Need at least two groups and enough observations
    if len(np.unique(labels)) < 2:
        return {
            "obs_sil": np.nan,
            "perm_mean": np.nan,
            "perm_std": np.nan,
            "adj_sil": np.nan,
            "z": np.nan,
            "p_value": np.nan
        }

    if X_embed.shape[0] <= len(np.unique(labels)):
        return {
            "obs_sil": np.nan,
            "perm_mean": np.nan,
            "perm_std": np.nan,
            "adj_sil": np.nan,
            "z": np.nan,
            "p_value": np.nan
        }

    rng = np.random.default_rng(random_state)

    obs = silhouette_score(X_embed, labels)

    perm = np.empty(n_perm, dtype=float)

    for b in range(n_perm):
        perm_labels = rng.permutation(labels)
        perm[b] = silhouette_score(X_embed, perm_labels)

    adj = obs - perm.mean()
    z = (obs - perm.mean()) / (perm.std(ddof=1) + 1e-12)

    if one_sided:
        p = (np.sum(perm >= obs) + 1) / (n_perm + 1)
    else:
        p_upper = (np.sum(perm >= obs) + 1) / (n_perm + 1)
        p_lower = (np.sum(perm <= obs) + 1) / (n_perm + 1)
        p = 2 * min(p_upper, p_lower)

    return {
        "obs_sil": float(obs),
        "perm_mean": float(perm.mean()),
        "perm_std": float(perm.std(ddof=1)),
        "adj_sil": float(adj),
        "z": float(z),
        "p_value": float(p)
    }


def summarize_region_marker_genes(
    d_g_all,
    tissue_region_list=None,
    common_genes=None
):
    """
    Summarize union marker gene sets for each tissue region across samples.

    Parameters
    ----------
    d_g_all : dict
        Output from select_region_markers_across_samples.

    tissue_region_list : list or None
        Tissue regions to summarize.
        If None, all regions appearing in d_g_all are used.

    common_genes : list or None
        Optional common gene set. If provided, region marker genes are restricted
        to these genes.

    Returns
    -------
    d_g_r : dict
        Dictionary:
        {
            region: union marker genes across samples
        }
    """

    if tissue_region_list is None:
        region_set = set()
        for sample_dict in d_g_all.values():
            region_set.update(sample_dict.keys())
        tissue_region_list = sorted(region_set)

    d_g_r = {}

    for region in tissue_region_list:
        gene_list = []

        for sample_name in d_g_all.keys():
            if region in d_g_all[sample_name]:
                gene_list += d_g_all[sample_name][region]

        region_genes = sorted(set(gene_list))

        if common_genes is not None:
            region_genes = sorted(set(region_genes) & set(common_genes))

        d_g_r[region] = region_genes

    return d_g_r


def compute_marker_stability_summary(
    d_g_all,
    d_g_r,
    all_adata,
    tissue_region_list=None,
    label_key="label",
    common_genes=None
):
    """
    Compute region-level marker instability score.

    Parameters
    ----------
    d_g_all : dict
        Sample-specific region marker genes.

    d_g_r : dict
        Region-level union marker genes.

    all_adata : AnnData
        Merged AnnData object containing all samples.

    tissue_region_list : list or None
        Tissue regions to evaluate.

    label_key : str
        Tissue region label column.

    common_genes : list or None
        Optional common gene set.

    Returns
    -------
    sta_summary : pandas.DataFrame
        Region-level gene stability summary.
    """

    if tissue_region_list is None:
        tissue_region_list = list(d_g_r.keys())

    sample_names = list(d_g_all.keys())
    d_r_s = {}

    for region in tissue_region_list:
        marker_dict = {}

        for sample in sample_names:
            if region in d_g_all[sample]:
                marker_dict[sample] = d_g_all[sample][region]

        if len(marker_dict) == 0:
            continue

        gene_max = max(len(v) for v in marker_dict.values())

        region_stability = marker_stability_jaccard(marker_dict)

        union_genes = d_g_r.get(region, [])

        if common_genes is not None:
            union_genes = sorted(set(union_genes) & set(common_genes))

        d_r_s[region] = {
            "stability": region_stability["stability"],
            "union_gene_size": len(union_genes),
            "max_gene_size": gene_max
        }

    sta_summary = pd.DataFrame.from_dict(d_r_s, orient="index")

    sta_summary["gene_score"] = 1 - sta_summary["stability"]

    n_spots = all_adata.obs[label_key].value_counts()
    sta_summary["n_spots_region"] = n_spots.reindex(sta_summary.index)

    sta_summary["log_n_spots_region"] = np.log1p(sta_summary["n_spots_region"])
    sta_summary["log_union_gene_size"] = np.log1p(sta_summary["union_gene_size"])
    sta_summary["log_max_gene_size"] = np.log1p(sta_summary["max_gene_size"])

    sta_summary["adjusted_gene_score"] = (
        sta_summary["gene_score"] *
        sta_summary["log_union_gene_size"] /
        sta_summary["log_max_gene_size"]
    )

    sta_summary = sta_summary.sort_values(
        by="adjusted_gene_score",
        ascending=False
    )

    return sta_summary


def compute_region_silhouette_summary(
    all_adata,
    d_g_r,
    tissue_region_list=None,
    label_key="label",
    sample_key="sample",
    common_genes=None,
    n_perm=200,
    random_state=0,
    one_sided=True,
    min_spots=10,
    print_results=True
):
    """
    Compute permutation-adjusted silhouette scores for each tissue region.

    Parameters
    ----------
    all_adata : AnnData
        Merged AnnData object across samples.

    d_g_r : dict
        Region-level union marker genes.

    tissue_region_list : list or None
        Regions to evaluate.

    label_key : str
        Column in all_adata.obs for tissue region labels.

    sample_key : str
        Column in all_adata.obs for sample IDs.

    common_genes : list or None
        Optional common genes.

    n_perm : int
        Number of permutations.

    random_state : int
        Random seed.

    one_sided : bool
        Whether to use one-sided permutation p-value.

    min_spots : int
        Minimum number of spots required for a region.

    print_results : bool
        Whether to print progress.

    Returns
    -------
    perm_sil_summary : pandas.DataFrame
        Region-level silhouette summary.
    """

    if tissue_region_list is None:
        tissue_region_list = list(d_g_r.keys())

    d_r_p = {}

    for region in tissue_region_list:
        if print_results:
            print(f"=================== {region} ===================")

        region_genes = d_g_r.get(region, [])

        if common_genes is not None:
            region_genes = sorted(set(region_genes) & set(common_genes))
        else:
            region_genes = sorted(set(region_genes) & set(all_adata.var.index))

        if len(region_genes) == 0:
            d_r_p[region] = {
                "obs_sil": np.nan,
                "perm_mean": np.nan,
                "perm_std": np.nan,
                "adj_sil": np.nan,
                "z": np.nan,
                "p_value": np.nan
            }
            continue

        region_adata = all_adata[
            all_adata.obs[label_key].astype(str) == str(region),
            all_adata.var.index.isin(region_genes)
        ].copy()

        if region_adata.n_obs < min_spots:
            d_r_p[region] = {
                "obs_sil": np.nan,
                "perm_mean": np.nan,
                "perm_std": np.nan,
                "adj_sil": np.nan,
                "z": np.nan,
                "p_value": np.nan
            }
            continue

        labels = region_adata.obs[sample_key].to_numpy()

        region_perm = perm_adjusted_silhouette(
            X_embed=region_adata.X,
            labels=labels,
            n_perm=n_perm,
            random_state=random_state,
            one_sided=one_sided
        )

        d_r_p[region] = region_perm

        if print_results:
            print(f"Number of genes: {len(region_genes)}")
            print(region_perm)

    perm_sil_summary = pd.DataFrame.from_dict(d_r_p, orient="index")
    perm_sil_summary = perm_sil_summary.sort_values(
        by="adj_sil",
        ascending=False
    )

    return perm_sil_summary


def compute_final_heterogeneity_score(
    sta_summary,
    perm_sil_summary,
    scale_score=True
):
    """
    Merge marker instability and silhouette separation into final heterogeneity score.

    Parameters
    ----------
    sta_summary : pandas.DataFrame
        Output from compute_marker_stability_summary.

    perm_sil_summary : pandas.DataFrame
        Output from compute_region_silhouette_summary.

    scale_score : bool
        Whether to min-max scale heterogeneity score to [0, 1].

    Returns
    -------
    hetero_summary : pandas.DataFrame
        Final heterogeneity score summary.
    """

    hetero_summary = perm_sil_summary.join(sta_summary, how="inner")

    hetero_summary["hetero_score"] = (
        hetero_summary["adj_sil"] *
        hetero_summary["adjusted_gene_score"]
    )

    if scale_score:
        score_min = hetero_summary["hetero_score"].min()
        score_max = hetero_summary["hetero_score"].max()

        if score_max > score_min:
            hetero_summary["hetero_score_sca"] = (
                (hetero_summary["hetero_score"] - score_min) /
                (score_max - score_min)
            )
        else:
            hetero_summary["hetero_score_sca"] = 0.0

    hetero_summary = hetero_summary.sort_values(
        by="hetero_score_sca" if scale_score else "hetero_score",
        ascending=False
    )

    return hetero_summary


def infer_heterogeneity_scores(
    ref_adata_dic,
    all_adata,
    common_genes=None,
    tissue_region_list=None,
    label_key="label",
    sample_key="sample",
    pvals_adj=0.05,
    min_in_out_group_ratio=1.0,
    min_in_group_fraction=0.5,
    min_fold_change=1.10,
    gene_num=10,
    n_perm=200,
    random_state=0,
    one_sided=True,
    min_spots=10,
    print_results=True
):
    """
    Infer tissue-region heterogeneity scores across multiple reference samples.

    This function assumes that preprocessing has already been performed by
    construct_ref_adata_dic(), including:
        1. low-expression gene filtering,
        2. common-gene selection,
        3. optional min-max normalization,
        4. construction of ref_adata_dic and all_adata.

    This function focuses on:
        1. region-specific marker gene selection,
        2. marker gene stability across samples,
        3. permutation-adjusted silhouette scores,
        4. final heterogeneity score calculation.

    Parameters
    ----------
    ref_adata_dic : dict
        Dictionary of preprocessed sample-specific AnnData objects.

        Example:
        {
            "H1": h1_adata_filtered,
            "G2": g2_adata_filtered,
            "E1": e1_adata_filtered
        }

    all_adata : AnnData
        Merged AnnData object across all reference samples.

    common_genes : list or None
        Common genes shared across samples.
        If None, common genes are inferred from all_adata.var.index.

    tissue_region_list : list or None
        Tissue regions to evaluate.
        If None, tissue regions are inferred from all_adata.obs[label_key].

    label_key : str
        Column in adata.obs containing tissue region labels.

    sample_key : str
        Column in all_adata.obs containing sample IDs.

    pvals_adj : float
        Adjusted p-value cutoff for marker gene selection.

    min_in_out_group_ratio : float
        Minimum in-group/out-group expression ratio.

    min_in_group_fraction : float
        Minimum fraction of spots/cells expressing the gene in the target region.

    min_fold_change : float
        Minimum fold-change cutoff.

    gene_num : int
        Number of top marker genes selected per region per sample.

    n_perm : int
        Number of permutations for adjusted silhouette score.

    random_state : int
        Random seed.

    one_sided : bool
        Whether to use one-sided permutation p-value.

    min_spots : int
        Minimum number of spots required for silhouette score calculation.

    print_results : bool
        Whether to print intermediate results.

    Returns
    -------
    results : dict
        Dictionary containing:
        {
            "hetero_summary": final heterogeneity score summary,
            "sta_summary": marker gene stability summary,
            "perm_sil_summary": permutation-adjusted silhouette summary,
            "d_g_all": sample-level region marker genes,
            "gene_list_all": sample-level union marker genes,
            "d_g_r": region-level union marker genes,
            "all_adata": merged AnnData object,
            "common_genes": common genes across samples,
            "ref_adata_dic": preprocessed sample-level AnnData dictionary
        }
    """

    # ------------------------------------------------------------
    # 0. Basic checks
    # ------------------------------------------------------------
    sample_names = list(ref_adata_dic.keys())

    if len(sample_names) == 0:
        raise ValueError("ref_adata_dic is empty.")

    if label_key not in all_adata.obs.columns:
        raise ValueError(f"{label_key} is not found in all_adata.obs.")

    if sample_key not in all_adata.obs.columns:
        raise ValueError(f"{sample_key} is not found in all_adata.obs.")

    if common_genes is None:
        common_genes = all_adata.var.index.tolist()
    else:
        common_genes = list(common_genes)

    if len(common_genes) == 0:
        raise ValueError("common_genes is empty.")

    if print_results:
        print(f"Number of samples: {len(sample_names)}")
        print(f"Sample names: {sample_names}")
        print(f"Number of common genes: {len(common_genes)}")
        print(f"Merged AnnData shape: {all_adata.shape}")

    # ------------------------------------------------------------
    # 1. Infer tissue_region_list if not provided
    # ------------------------------------------------------------
    if tissue_region_list is None:
        tissue_region_list = all_adata.obs[label_key].value_counts().index.tolist()

        tissue_region_list = [
            str(region) for region in tissue_region_list
            if str(region).lower() not in ["nan", "unknown"]
        ]

        tissue_region_list = sorted(tissue_region_list)

    if print_results:
        print(f"Included tissue regions: {tissue_region_list}")

    # ------------------------------------------------------------
    # 2. Select marker genes across samples
    # ------------------------------------------------------------
    d_g_all, gene_list_all = select_region_markers_across_samples(
        ref_adata_dic=ref_adata_dic,
        label_key=label_key,
        gene_num=gene_num,
        min_fold_change=min_fold_change,
        min_in_out_group_ratio=min_in_out_group_ratio,
        min_in_group_fraction=min_in_group_fraction,
        pvals_adj=pvals_adj,
        print_results=print_results
    )

    # ------------------------------------------------------------
    # 3. Summarize region-level marker gene sets
    # ------------------------------------------------------------
    d_g_r = summarize_region_marker_genes(
        d_g_all=d_g_all,
        tissue_region_list=tissue_region_list,
        common_genes=common_genes
    )

    # ------------------------------------------------------------
    # 4. Compute marker gene stability / instability score
    # ------------------------------------------------------------
    sta_summary = compute_marker_stability_summary(
        d_g_all=d_g_all,
        d_g_r=d_g_r,
        all_adata=all_adata,
        tissue_region_list=tissue_region_list,
        label_key=label_key,
        common_genes=common_genes
    )

    # ------------------------------------------------------------
    # 5. Compute permutation-adjusted silhouette score
    # ------------------------------------------------------------
    perm_sil_summary = compute_region_silhouette_summary(
        all_adata=all_adata,
        d_g_r=d_g_r,
        tissue_region_list=tissue_region_list,
        label_key=label_key,
        sample_key=sample_key,
        common_genes=common_genes,
        n_perm=n_perm,
        random_state=random_state,
        one_sided=one_sided,
        min_spots=min_spots,
        print_results=print_results
    )

    # ------------------------------------------------------------
    # 6. Compute final heterogeneity score
    # ------------------------------------------------------------
    hetero_summary = compute_final_heterogeneity_score(
        sta_summary=sta_summary,
        perm_sil_summary=perm_sil_summary,
        scale_score=True
    )

    # ------------------------------------------------------------
    # 7. Store results
    # ------------------------------------------------------------
    results = {
        "hetero_summary": hetero_summary,
        "sta_summary": sta_summary,
        "perm_sil_summary": perm_sil_summary,
        "d_g_all": d_g_all,
        "gene_list_all": gene_list_all,
        "d_g_r": d_g_r,
        "all_adata": all_adata,
        "common_genes": common_genes,
        "ref_adata_dic": ref_adata_dic,
    }

    return results


#=======================================================================
# Part 2. Identify heterogeneity subtypes
#=======================================================================
# Pipeline:
# clustering within each section and identify marker genes ->
# merge tissue sections to identify shared clusters ->
# identify shared subtype DE genes for functional interpretations

def section_subtype_DE_genes(
    ref_region_adata_dic,
    tissue_section_list,
    target_region,
    res_dir,
    pcs_num=30,
    cluster_method="leiden_clusters",
    n_clusters=2,
    leiden_res=0.5,
    n_neighbors=15,
    random_state=0,
    pvals_adj=0.05,
    min_in_out_group_ratio=1,
    min_in_group_fraction=0.5,
    min_fold_change=1.1,
    gene_num=10,
    min_cluster_fraction=0.05,
    cat_color=None,
    cnt_colormap="coolwarm",
    x_key="pixel_x",
    y_key="pixel_y",
    fig_scale=2500,
    invert_x=False,
    invert_y=False,
):
    """
    Cluster each tissue section within the target region and identify section-level subtype genes.

    Returns
    -------
    d_g : dict
        Section-specific union of subtype marker genes.
    total_genes_list : list
        Union of selected marker genes across all sections.
    total_clusters_num : int
        Total number of non-small clusters across sections.
    """
    cat_color = _default_cat_color(cat_color)
    d_g = {}
    total_clusters_num = 0

    for tissue_section in tissue_section_list:
        print("======================================= " + tissue_section + " =======================================")

        if tissue_section not in ref_region_adata_dic:
            raise KeyError(f"{tissue_section!r} is not present in ref_region_adata_dic.")

        test_gene = ref_region_adata_dic[tissue_section].copy()

        if test_gene.shape[0] < 2:
            raise ValueError(f"{tissue_section} has fewer than 2 spots.")
        if test_gene.shape[1] < 2:
            raise ValueError(f"{tissue_section} has fewer than 2 genes.")

        # ---------------- 1. PCA ----------------
        gene_pcs = compute_pca_embedding(
            input_adata = test_gene, 
            pcs_num = pcs_num, 
            random_state = random_state, 
            sample_name = tissue_section)

        # ---------------- 2. Clustering ----------------
        if cluster_method == "kmeans_clusters":
            cluster_key = "kmeans_clusters"
            pred = kmeans_clustering(
                features_matrix=gene_pcs,
                n_clusters=n_clusters,
                random_state=random_state,
            )
            fig_path = os.path.join(
                res_dir,
                f"{tissue_section}_{target_region}_subtype_{cluster_key}_npcs={pcs_num}_nclusters={n_clusters}.png",
            )

        elif cluster_method == "leiden_clusters":
            cluster_key = "leiden_clusters"
            pred = leiden_clustering(
                features_matrix=gene_pcs,
                resolution=leiden_res,
                n_neighbors=n_neighbors,
                random_state=random_state,
                leiden_key=cluster_key,
            )
            fig_path = os.path.join(
                res_dir,
                f"{tissue_section}_{target_region}_subtype_{cluster_key}_npcs={pcs_num}_res={leiden_res}_nn={n_neighbors}.png",
            )

        else:
            raise ValueError("cluster_method must be 'kmeans_clusters' or 'leiden_clusters'.")

        test_gene.obs[cluster_key] = pred.copy()
        test_gene.obs[cluster_key] = test_gene.obs[cluster_key].astype("category")
        cluster_perct = test_gene.obs[cluster_key].value_counts(normalize=True)
        print(cluster_perct)

        # Check clustering spatial patterns.
        fig_title = f"{tissue_section}: {target_region} subtypes ({cluster_key})"
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        cat_figure(
            input_adata=test_gene,
            x_key=x_key,
            y_key=y_key,
            fig_title=fig_title,
            fig_path=fig_path,
            color_key=cluster_key,
            cat_color=cat_color,
            size=fig_scale / (test_gene.shape[0] ** 0.5),
            invert_x=invert_x,
            invert_y=invert_y,
        )

        # ---------------- 3. One-vs-all DE gene selection ----------------
        d_g_section = {}
        cluster_list = cluster_perct[cluster_perct > min_cluster_fraction].index.tolist()
        total_clusters_num += len(cluster_list)
        print("The total number subtype clusters across tissue section is " + str(total_clusters_num))

        if len(cluster_list) > 1:
            for target_cluster in cluster_list:

                df1_genes, df1_filtered = get_region_genes(
                    input_adata = test_gene,
                    region = target_cluster,
                    label_key = cluster_key,
                    gene_num = gene_num,
                    min_fold_change = min_fold_change,
                    min_in_out_group_ratio = min_in_out_group_ratio,
                    min_in_group_fraction = min_in_group_fraction,
                    pvals_adj = pvals_adj
                    )

                print(df1_filtered.iloc[:gene_num])
                print(df1_genes)

                d_g_section[f"cluster{target_cluster}_vs_others"] = df1_genes

                # Check gene expression patterns.
                for g in df1_genes:
                    test_gene.obs[g] = get_gene_vector(test_gene, g)
                    fig_title = f"{tissue_section}: cluster{target_cluster} ({g})"
                    fig_path = os.path.join(
                        res_dir,
                        "subtype_DE_patterns",
                        tissue_section,
                        f"{tissue_section}_cluster{target_cluster}_{g}.png",
                    )
                    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
                    con_figure(
                        input_adata=test_gene,
                        x_key=x_key,
                        y_key=y_key,
                        fig_title=fig_title,
                        fig_path=fig_path,
                        color_key=g,
                        cnt_color=cnt_colormap,
                        size=fig_scale / (test_gene.shape[0] ** 0.5),
                        invert_x=invert_x,
                        invert_y=invert_y,
                    )

        section_gene_list = sorted(set(gene for genes in d_g_section.values() for gene in genes))
        d_g[tissue_section] = section_gene_list

    print(d_g)

    total_genes_list = sorted(set(gene for gene_list in d_g.values() for gene in gene_list))
    print(f"The total number of genes is: {len(total_genes_list)}")
    print(total_genes_list)

    return d_g, total_genes_list, total_clusters_num


def determine_best_cluster_number(
    gene_df,
    total_clusters_num,
    set_clusters_num=None,
    random_state=0,
    min_k=2,
    print_results=True,
):
    """
    Determine the number of shared subtype clusters.

    If `set_clusters_num` is provided, this function directly uses that value.
    Otherwise, it evaluates candidate cluster numbers using KMeans clustering
    and silhouette scores, then selects the cluster number with the highest
    silhouette score.

    Parameters
    ----------
    gene_df : pandas.DataFrame
        Dense gene-expression dataframe used for clustering.
        Rows are spots/cells and columns are genes/features.

    total_clusters_num : int
        Maximum candidate number of clusters, usually calculated as the total
        number of non-small subtype clusters identified across individual
        tissue sections.

    set_clusters_num : int or None, default=None
        User-specified number of shared subtype clusters. If provided, silhouette
        score evaluation is skipped.

    random_state : int, default=0
        Random seed used for KMeans clustering.

    min_k : int, default=2
        Minimum number of clusters to evaluate.

    print_results : bool, default=True
        Whether to print silhouette scores and selected cluster number.

    Returns
    -------
    best_k : int
        Selected number of shared subtype clusters.

    all_scores : dict
        Dictionary of silhouette scores for each evaluated cluster number.
        If `set_clusters_num` is provided, this returns an empty dictionary.

    Raises
    ------
    ValueError
        If there are not enough observations to evaluate clustering, if no valid
        silhouette score can be calculated, or if the final selected `best_k` is
        invalid.
    """

    n_obs = gene_df.shape[0]

    if set_clusters_num is not None:
        best_k = int(set_clusters_num)
        all_scores = {}

        if print_results:
            print("Use the specified number of clusters:", best_k)

    else:
        max_k = min(total_clusters_num, n_obs - 1)

        if max_k < min_k:
            raise ValueError(
                "Not enough observations/clusters to evaluate silhouette score."
            )

        all_scores = {}
        best_score = -np.inf
        best_k = None

        for k in range(min_k, max_k + 1):
            kmeans = KMeans(
                n_clusters=k,
                random_state=random_state,
                n_init="auto",
            )

            labels = kmeans.fit_predict(gene_df)

            if len(np.unique(labels)) < 2:
                continue

            score = silhouette_score(gene_df, labels)

            if print_results:
                print(f"Clusters: {k}, Silhouette Score: {score}")

            all_scores[k] = score

            if score > best_score:
                best_score = score
                best_k = k

        if best_k is None:
            raise ValueError(
                "Unable to determine best_k from silhouette scores."
            )

        if print_results:
            print(
                f"\nBest number of clusters (by silhouette): "
                f"{best_k}, score={best_score}"
            )

    if best_k < min_k or best_k >= n_obs:
        raise ValueError(
            "Final best_k must be >= 2 and smaller than the number of observations."
        )

    return best_k, all_scores


def identify_shared_subtype(
    ref_adata_dic,
    tissue_section_list,
    target_region,
    total_genes_list,
    total_clusters_num,
    res_dir,
    cat_color,
    set_clusters_num=None,
    merged_key="sample",
    cluster_key="kmeans_clusters",
    random_state=0,
    x_key="pixel_x",
    y_key="pixel_y",
    fig_scale=2500,
    invert_x=False,
    invert_y=False,
    print_results=True,
):
    """
    Identify shared subtype clusters across multiple tissue sections.

    This function merges multiple tissue sections using the union of selected
    subtype marker genes, determines the number of shared subtype clusters,
    performs shared clustering using either KMeans or hierarchical heatmap
    clustering, visualizes subtype patterns in each tissue section, and saves
    subtype cluster assignments.

    Parameters
    ----------
    ref_adata_dic : dict
        Dictionary of reference-tissue-section AnnData objects.

        Example:
        {
            "H1": adata_H1,
            "G2": adata_G2,
            "E1": adata_E1,
        }

    tissue_section_list : list
        List of tissue-section names to include.

    target_region : str
        Name of the target tissue region being analyzed.

    total_genes_list : list
        Union of subtype marker genes selected from individual tissue sections.

    total_clusters_num : int
        Total number of non-small subtype clusters identified across individual
        tissue sections. Used as the upper bound when selecting the shared
        cluster number by silhouette score.

    res_dir : str
        Directory where output figures and subtype results will be saved.

    set_clusters_num : int or None, default=None
        User-specified number of shared subtype clusters. If provided, silhouette
        score selection is skipped.

    merged_key : str, default="sample"
        Column name added to `.obs` of the merged AnnData object to indicate
        the tissue-section/source sample.

    cluster_key : str, default="kmeans_clusters"
        Shared subtype clustering method to use. Supported options are:
            "kmeans_clusters"
            "heatmap_clusters"

    random_state : int, default=0
        Random seed used for KMeans clustering.

    cat_color : list or None, default=None
        Categorical color palette for subtype visualization. If None, the default
        categorical palette from `_default_cat_color` is used.

    x_key : str, default="pixel_x"
        Column in `.obs` containing x coordinates.

    y_key : str, default="pixel_y"
        Column in `.obs` containing y coordinates.

    fig_scale : float, default=2500
        Scaling factor used to determine scatter point size.

    invert_x : bool, default=False
        Whether to invert the x-axis.

    invert_y : bool, default=False
        Whether to invert the y-axis.

    print_results : bool, default=True
        Whether to print intermediate clustering results.

    Returns
    -------
    subtype_clusters : pandas.DataFrame
        DataFrame containing merged `.obs` metadata and shared subtype cluster
        assignments for all spots/cells across tissue sections.
    """

    if total_genes_list is None or len(total_genes_list) == 0:
        raise ValueError(
            "total_genes_list is empty. No subtype marker genes were selected."
        )

    if total_clusters_num < 2 and set_clusters_num is None:
        raise ValueError(
            "total_clusters_num must be >= 2 unless set_clusters_num is provided."
        )

    if cluster_key not in ["kmeans_clusters", "heatmap_clusters"]:
        raise ValueError(
            "cluster_key must be either 'kmeans_clusters' or 'heatmap_clusters'."
        )

    # ---------------- 1. Construct merged AnnData ----------------
    merged_adata_sca, gene_df = construct_merged_scaled_adata_and_gene_df(
        ref_adata_dic=ref_adata_dic,
        tissue_section_list=tissue_section_list,
        total_genes_list=total_genes_list,
        merged_key=merged_key,
        normalize=True,
        print_results=print_results,
    )

    # ---------------- 2. Determine number of shared clusters ----------------
    best_k, all_scores = determine_best_cluster_number(
        gene_df=gene_df,
        total_clusters_num=total_clusters_num,
        set_clusters_num=set_clusters_num,
        random_state=random_state,
        print_results=print_results,
    )

    # ---------------- 3. KMeans clustering ----------------
    if cluster_key == "kmeans_clusters":
        pred = kmeans_clustering(
            features_matrix=gene_df,
            n_clusters=best_k,
            random_state=random_state,
        )

        merged_adata_sca.obs[cluster_key] = pred.copy()
        merged_adata_sca.obs[cluster_key] = merged_adata_sca.obs[cluster_key].astype(
            "category"
        )

        if print_results:
            print(merged_adata_sca.obs[cluster_key].value_counts(normalize=True))

        for tissue_section in tissue_section_list:
            section_adata = merged_adata_sca[
                merged_adata_sca.obs[merged_key] == tissue_section
            ].copy()

            section_cat_colors = get_cluster_palette(
                section_adata.obs[cluster_key],
                cat_color,
            )

            fig_title = (
                f"{tissue_section}: kmeans clustering "
                f"(subtype cluster num={best_k})"
            )

            fig_path = os.path.join(
                res_dir,
                f"{tissue_section}_{cluster_key}_nclusters={best_k}.png",
            )

            os.makedirs(os.path.dirname(fig_path), exist_ok=True)

            cat_figure(
                input_adata=section_adata,
                x_key=x_key,
                y_key=y_key,
                fig_title=fig_title,
                fig_path=fig_path,
                color_key=cluster_key,
                cat_color=section_cat_colors,
                size=fig_scale / (section_adata.shape[0] ** 0.5),
                invert_x=invert_x,
                invert_y=invert_y,
            )

    # ---------------- 4. Heatmap hierarchical clustering ----------------
    elif cluster_key == "heatmap_clusters":
        fig_path = os.path.join(res_dir, f"{cluster_key}.png")
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)

        cluster_grid = sns.clustermap(
            gene_df,
            cmap="coolwarm",
            method="ward",
            figsize=(10, 10),
        )

        spots_linkage = cluster_grid.dendrogram_row.linkage

        spots_clusters = fcluster(
            spots_linkage,
            best_k,
            criterion="maxclust",
        )

        spots_cluster_labels = pd.Series(
            spots_clusters,
            index=gene_df.index,
            name="Spots_clusters",
        )

        merged_adata_sca.obs[cluster_key] = spots_cluster_labels
        merged_adata_sca.obs[cluster_key] = merged_adata_sca.obs[
            cluster_key
        ].astype("category")

        if print_results:
            print(merged_adata_sca.obs[cluster_key].value_counts())

        for tissue_section in tissue_section_list:
            section_adata = merged_adata_sca[
                merged_adata_sca.obs[merged_key] == tissue_section
            ].copy()

            section_cat_colors = get_cluster_palette(
                section_adata.obs[cluster_key],
                cat_color,
            )

            fig_title = f"{tissue_section}: heatmap clustering"

            fig_path = os.path.join(
                res_dir,
                f"{tissue_section}_{cluster_key}_nclusters={best_k}.png",
            )

            os.makedirs(os.path.dirname(fig_path), exist_ok=True)

            cat_figure(
                input_adata=section_adata,
                x_key=x_key,
                y_key=y_key,
                fig_title=fig_title,
                fig_path=fig_path,
                color_key=cluster_key,
                cat_color=section_cat_colors,
                size=fig_scale / (section_adata.shape[0] ** 0.5),
                invert_x=invert_x,
                invert_y=invert_y,
            )

    # ---------------- 5. Save subtype results ----------------
    subtype_clusters = merged_adata_sca.obs.copy()

    subtype_results_dir = os.path.join(res_dir, "subtype_results")
    os.makedirs(subtype_results_dir, exist_ok=True)

    subtype_clusters.to_csv(
        os.path.join(
            subtype_results_dir,
            f"{target_region}_subtype_clusters.csv",
        ),
        index=True,
    )

    for tissue_section in tissue_section_list:
        section_subtype_clusters = subtype_clusters.loc[
            subtype_clusters[merged_key] == tissue_section
        ].copy()

        section_subtype_clusters.to_csv(
            os.path.join(
                subtype_results_dir,
                f"{tissue_section}_{target_region}_subtype_clusters.csv",
            ),
            index=True,
        )

    return subtype_clusters


def identify_overlap_genes_across_sections(
    d_g_section,
    cluster_list,
    tissue_section_list,
    overlap_cutoff,
    print_results=True,
):
    """
    Identify overlapping subtype marker genes across tissue sections.

    For each subtype cluster, this function collects marker genes selected
    from individual tissue sections, counts how many times each gene appears
    across sections, and retains genes whose counts are greater than or equal
    to the final overlap cutoff.

    Parameters
    ----------
    d_g_section : dict
        Nested dictionary containing section-level subtype marker genes.

        Example:
        {
            "H1": {
                "subtype0": ["GeneA", "GeneB"],
                "subtype1": ["GeneC"]
            },
            "G2": {
                "subtype0": ["GeneA", "GeneD"],
                "subtype1": ["GeneC", "GeneE"]
            }
        }

    cluster_list : list
        List of subtype cluster labels to summarize.

    tissue_section_list : list
        List of tissue-section names.

    overlap_cutoff : int
        Minimum number of occurrences required for a gene to be retained.
        The effective cutoff is capped by the maximum observed gene count for
        each subtype cluster.

    print_results : bool, default=True
        Whether to print gene counts and selected overlap genes.

    Returns
    -------
    d_g_ind : dict
        Dictionary of overlap-selected subtype marker genes.

        Example:
        {
            "subtype0": ["GeneA"],
            "subtype1": ["GeneC"]
        }
    """

    if overlap_cutoff < 1:
        raise ValueError("overlap_cutoff must be >= 1.")

    d_g_ind = {}

    for cluster in cluster_list:
        subtype_key = f"subtype{cluster}"

        if print_results:
            print(f"----------------------------- {subtype_key} -----------------------------")

        cluster_union_genes_set = []

        for tissue_section in tissue_section_list:
            if tissue_section not in d_g_section:
                raise KeyError(f"{tissue_section!r} is not present in d_g_section.")

            if subtype_key in d_g_section[tissue_section]:
                cluster_union_genes_set += d_g_section[tissue_section][subtype_key]

        if len(cluster_union_genes_set) == 0:
            if print_results:
                print("No genes found for this subtype across sections.")
            d_g_ind[subtype_key] = []
            continue

        cluster_gene_counts = pd.Series(cluster_union_genes_set).value_counts()

        if print_results:
            print(cluster_gene_counts)

        final_cutoff = min(cluster_gene_counts.max(), overlap_cutoff)

        if print_results:
            print(f"The final overlap counts cut-off is {final_cutoff}")

        overlap_genes = cluster_gene_counts[
            cluster_gene_counts >= final_cutoff
        ].index.tolist()

        if print_results:
            print(overlap_genes)

        d_g_ind[subtype_key] = overlap_genes

    return d_g_ind


def shared_subtype_DE_genes(
    ref_region_adata_dic,
    tissue_section_list,
    target_region,
    res_dir,
    overlap_cutoff,
    subtype_clusters,
    subtype_cluster_key="kmeans_clusters",
    pvals_adj=0.05,
    min_in_out_group_ratio=1,
    min_in_group_fraction=0.5,
    min_fold_change=1.1,
    merged_gene_num=15,
    individual_gene_num=35,
    x_key="pixel_x",
    y_key="pixel_y",
    fig_scale=2500,
    invert_x=False,
    invert_y=False,
    merged_key="sample",
    cnt_colormap="coolwarm",
    min_cluster_fraction=0.05,
    print_results=True,
):
    """
    Identify DE genes for shared subtype clusters using merged and per-section strategies.

    This function performs two complementary DE analyses:

        1. Merged-section strategy:
           Merge all tissue sections, normalize expression, and identify DE genes
           for each shared subtype cluster against the rest.

        2. Per-section overlap strategy:
           Identify subtype DE genes within each tissue section separately, then
           retain genes that recur across sections based on `overlap_cutoff`.

    Parameters
    ----------
    ref_region_adata_dic : dict
        Dictionary of reference-region AnnData objects.

    tissue_section_list : list
        List of tissue-section names to include.

    target_region : str
        Name of the target tissue region being analyzed.

    res_dir : str
        Directory where output figures will be saved.

    overlap_cutoff : int
        Minimum number of sections in which a gene should appear to be retained
        in the per-section overlap strategy. The effective cutoff is capped by
        the maximum observed gene count.

    subtype_clusters : pandas.DataFrame
        DataFrame containing shared subtype cluster assignments. Its index should
        match the observation names of the merged AnnData object.

    subtype_cluster_key : str, default="kmeans_clusters"
        Column in `subtype_clusters` containing shared subtype cluster labels.

    pvals_adj : float, default=0.05
        Maximum adjusted p-value threshold for DE gene filtering.

    min_in_out_group_ratio : float, default=1
        Minimum in/out group expression ratio.

    min_in_group_fraction : float, default=0.5
        Minimum fraction of spots/cells expressing the gene in the target group.

    min_fold_change : float, default=1.1
        Minimum fold-change threshold.

    merged_gene_num : int, default=15
        Maximum number of genes selected per subtype from the merged-section strategy.

    individual_gene_num : int, default=35
        Maximum number of genes selected per subtype per section before overlap.

    x_key : str, default="pixel_x"
        Column in `.obs` containing x coordinates.

    y_key : str, default="pixel_y"
        Column in `.obs` containing y coordinates.

    fig_scale : float, default=2500
        Scaling factor used to determine scatter point size.

    invert_x : bool, default=False
        Whether to invert the x-axis.

    invert_y : bool, default=False
        Whether to invert the y-axis.

    merged_key : str, default="sample"
        Column in merged AnnData `.obs` indicating the source tissue section.

    cnt_colormap : str or matplotlib colormap, default="coolwarm"
        Continuous colormap for gene expression plots.

    min_cluster_fraction : float, default=0.05
        Minimum cluster proportion required to include a subtype cluster.

    print_results : bool, default=True
        Whether to print intermediate DE results.

    Returns
    -------
    d_g_merged : dict
        Dictionary of subtype DE genes identified from the merged-section strategy.

    d_g_ind : dict
        Dictionary of subtype DE genes identified from the per-section overlap strategy.
    """

    if tissue_section_list is None or len(tissue_section_list) == 0:
        raise ValueError("tissue_section_list is empty.")

    if subtype_cluster_key not in subtype_clusters.columns:
        raise KeyError(
            f"{subtype_cluster_key!r} is not present in subtype_clusters."
        )

    if overlap_cutoff < 1:
        raise ValueError("overlap_cutoff must be >= 1.")

    # ------------------------------------------------------------------
    # 1. Construct merged AnnData objects
    # ------------------------------------------------------------------
    # Raw/non-normalized merged object for per-section DE.
    merged_adata, _ = construct_merged_scaled_adata_and_gene_df(
        ref_adata_dic=ref_region_adata_dic,
        tissue_section_list=tissue_section_list,
        total_genes_list=None,
        merged_key=merged_key,
        normalize=False,
        print_results=print_results,
    )

    # Normalized merged object for merged-section DE and visualization.
    merged_adata_sca, _ = construct_merged_scaled_adata_and_gene_df(
        ref_adata_dic=ref_region_adata_dic,
        tissue_section_list=tissue_section_list,
        total_genes_list=None,
        merged_key=merged_key,
        normalize=True,
        print_results=print_results,
    )

    # ------------------------------------------------------------------
    # 2. Add subtype cluster labels
    # ------------------------------------------------------------------
    missing_idx = merged_adata.obs.index.difference(subtype_clusters.index)

    if len(missing_idx) > 0:
        raise KeyError(
            f"subtype_clusters is missing {len(missing_idx)} merged observation IDs. "
            "Please check that obs_names are preserved and unique across sections."
        )

    missing_idx_sca = merged_adata_sca.obs.index.difference(subtype_clusters.index)

    if len(missing_idx_sca) > 0:
        raise KeyError(
            f"subtype_clusters is missing {len(missing_idx_sca)} scaled merged observation IDs. "
            "Please check that obs_names are preserved and unique across sections."
        )

    merged_adata.obs[subtype_cluster_key] = subtype_clusters.loc[
        merged_adata.obs.index,
        subtype_cluster_key,
    ].tolist()

    merged_adata_sca.obs[subtype_cluster_key] = subtype_clusters.loc[
        merged_adata_sca.obs.index,
        subtype_cluster_key,
    ].tolist()

    merged_adata.obs[subtype_cluster_key] = merged_adata.obs[
        subtype_cluster_key
    ].astype("category")

    merged_adata_sca.obs[subtype_cluster_key] = merged_adata_sca.obs[
        subtype_cluster_key
    ].astype("category")

    cluster_perct = merged_adata_sca.obs[subtype_cluster_key].value_counts(
        normalize=True
    )

    cluster_list = cluster_perct[
        cluster_perct > min_cluster_fraction
    ].index.tolist()

    if print_results:
        print("Included shared subtype clusters:")
        print(cluster_perct.loc[cluster_list])

    if len(cluster_list) <= 1:
        print(
            "Fewer than two subtype clusters passed min_cluster_fraction. "
            "DE gene selection will return empty dictionaries."
        )
        return {}, {}

    # ------------------------------------------------------------------
    # 3. Merged-section DE genes
    # ------------------------------------------------------------------
    d_g_merged = {}

    print(
        "----------------------------- Identify shared subtype DE genes "
        "by merging tissue sections -----------------------------"
    )

    for cluster in cluster_list:
        df1_genes, df1_filtered = get_region_genes(
            input_adata=merged_adata_sca,
            region=cluster,
            label_key=subtype_cluster_key,
            gene_num=merged_gene_num,
            min_fold_change=min_fold_change,
            min_in_out_group_ratio=min_in_out_group_ratio,
            min_in_group_fraction=min_in_group_fraction,
            pvals_adj=pvals_adj,
            print_results=print_results,
        )

        if print_results:
            print(df1_filtered.iloc[:merged_gene_num])
            print(df1_genes)

        d_g_merged[f"subtype{cluster}"] = df1_genes

    # Plot merged-section selected genes.
    for tissue_section in tissue_section_list:
        section_adata = merged_adata_sca[
            merged_adata_sca.obs[merged_key] == tissue_section
        ].copy()

        for key, gene_list in d_g_merged.items():
            for g in gene_list:
                if g not in section_adata.var_names:
                    continue

                section_adata.obs[g] = get_gene_vector(section_adata, g)

                fig_title = f"{tissue_section}: {key} ({g})"

                fig_path = os.path.join(
                    res_dir,
                    "subtype_interpretations",
                    "merged_selection",
                    f"{tissue_section}_{target_region}_subtype_DE_genes_merged_version_{key}_{g}.png",
                )

                os.makedirs(os.path.dirname(fig_path), exist_ok=True)

                con_figure(
                    input_adata=section_adata,
                    x_key=x_key,
                    y_key=y_key,
                    fig_title=fig_title,
                    fig_path=fig_path,
                    color_key=g,
                    cnt_color=cnt_colormap,
                    size=fig_scale / (section_adata.shape[0] ** 0.5),
                    invert_x=invert_x,
                    invert_y=invert_y,
                )

    # ------------------------------------------------------------------
    # 4. Per-section DE genes
    # ------------------------------------------------------------------
    d_g_section = {}

    print(
        "----------------------------- Identify shared subtype DE genes "
        "within each individual tissue section -----------------------------"
    )

    for tissue_section in tissue_section_list:
        d_g_cluster = {}

        test_gene = merged_adata[
            merged_adata.obs[merged_key] == tissue_section
        ].copy()

        section_cluster_perct = test_gene.obs[
            subtype_cluster_key
        ].value_counts(normalize=True)

        section_cluster_list = section_cluster_perct[
            section_cluster_perct > min_cluster_fraction
        ].index.tolist()

        if print_results:
            print(f"\n----------- Section: {tissue_section} -----------")
            print(section_cluster_perct)

        if len(section_cluster_list) > 1:
            for cluster in section_cluster_list:
                df1_genes, df1_filtered = get_region_genes(
                    input_adata=test_gene,
                    region=cluster,
                    label_key=subtype_cluster_key,
                    gene_num=individual_gene_num,
                    min_fold_change=min_fold_change,
                    min_in_out_group_ratio=min_in_out_group_ratio,
                    min_in_group_fraction=min_in_group_fraction,
                    pvals_adj=pvals_adj,
                    print_results=print_results,
                )

                if print_results:
                    print(df1_filtered.iloc[:individual_gene_num])
                    print(df1_genes)

                d_g_cluster[f"subtype{cluster}"] = df1_genes

        d_g_section[tissue_section] = d_g_cluster

    # ------------------------------------------------------------------
    # 5. Identify overlap genes across tissue sections
    # ------------------------------------------------------------------
    d_g_ind = identify_overlap_genes_across_sections(
        d_g_section=d_g_section,
        cluster_list=cluster_list,
        tissue_section_list=tissue_section_list,
        overlap_cutoff=overlap_cutoff,
        print_results=print_results
        )

    # ------------------------------------------------------------------
    # 6. Plot per-section overlap genes
    # ------------------------------------------------------------------
    for tissue_section in tissue_section_list:
        section_adata = merged_adata_sca[
            merged_adata_sca.obs[merged_key] == tissue_section
        ].copy()

        for key, gene_list in d_g_ind.items():
            for g in gene_list:
                if g not in section_adata.var_names:
                    continue

                section_adata.obs[g] = get_gene_vector(section_adata, g)

                fig_title = f"{tissue_section}: {key} ({g})"

                fig_path = os.path.join(
                    res_dir,
                    "subtype_interpretations",
                    "individual_selection",
                    "within_target_region",
                    f"{tissue_section}_{target_region}_subtype_DE_genes_individual_version_{key}_{g}.png",
                )

                os.makedirs(os.path.dirname(fig_path), exist_ok=True)

                con_figure(
                    input_adata=section_adata,
                    x_key=x_key,
                    y_key=y_key,
                    fig_title=fig_title,
                    fig_path=fig_path,
                    color_key=g,
                    cnt_color=cnt_colormap,
                    size=fig_scale / (section_adata.shape[0] ** 0.5),
                    invert_x=invert_x,
                    invert_y=invert_y,
                )

    return d_g_merged, d_g_ind


#=======================================================================
# Part 3. End-to-end reference heterogeneity pipeline
#=======================================================================
def select_heterogeneous_regions(
    hetero_summary,
    method="threshold",
    score_key="hetero_score_sca",
    threshold=0.5,
    top_k=None,
    include_ties=True,
    print_results=True,
):
    """
    Select heterogeneous tissue regions from a region-level heterogeneity table.

    Two selection frameworks are supported:
        1. method="threshold": select regions with score >= threshold.
        2. method="top_k": select the top-k highest-scoring regions.

    Parameters
    ----------
    hetero_summary : pandas.DataFrame
        Region-level heterogeneity score table returned by
        compute_final_heterogeneity_score() or infer_heterogeneity_scores().

    method : {"threshold", "top_k"}, default="threshold"
        Selection framework.

    score_key : str, default="hetero_score_sca"
        Column used to rank/select heterogeneous regions.

    threshold : float, default=0.5
        Score cutoff used when method="threshold".

    top_k : int or None, default=None
        Number of top heterogeneous regions selected when method="top_k".

    include_ties : bool, default=True
        If True and method="top_k", include regions tied with the kth score.

    print_results : bool, default=True
        Whether to print selected regions.

    Returns
    -------
    selected_regions : list
        Selected heterogeneous tissue regions.

    selected_region_scores : pandas.DataFrame
        Subset of hetero_summary for selected regions.
    """

    if hetero_summary is None or hetero_summary.shape[0] == 0:
        raise ValueError("hetero_summary is empty.")

    if score_key not in hetero_summary.columns:
        raise KeyError(f"{score_key!r} is not present in hetero_summary.")

    if method not in ["threshold", "top_k"]:
        raise ValueError("method must be either 'threshold' or 'top_k'.")

    score_table = hetero_summary.copy()
    score_table = score_table.loc[score_table[score_key].notna()].copy()
    score_table = score_table.sort_values(by=score_key, ascending=False)

    if method == "threshold":
        selected_region_scores = score_table.loc[
            score_table[score_key] >= threshold
        ].copy()

    else:
        if top_k is None:
            raise ValueError("top_k must be provided when method='top_k'.")

        top_k = int(top_k)
        if top_k < 1:
            raise ValueError("top_k must be >= 1.")

        top_k = min(top_k, score_table.shape[0])

        if include_ties:
            kth_score = score_table.iloc[top_k - 1][score_key]
            selected_region_scores = score_table.loc[
                score_table[score_key] >= kth_score
            ].copy()
        else:
            selected_region_scores = score_table.iloc[:top_k].copy()

    selected_regions = selected_region_scores.index.astype(str).tolist()

    if print_results:
        print("Selected heterogeneous regions:")
        print(selected_region_scores[[score_key]])

    return selected_regions, selected_region_scores


def infer_region_shared_subtypes(
    ref_adata_dic,
    target_region,
    res_dir,
    label_key="label",
    min_region_spots=10,
    pcs_num=30,
    section_cluster_method="leiden_clusters",
    section_n_clusters=2,
    leiden_res=0.5,
    n_neighbors=15,
    shared_cluster_key="kmeans_clusters",
    set_shared_clusters_num=None,
    overlap_cutoff=2,
    random_state=0,
    pvals_adj=0.05,
    min_in_out_group_ratio=1.0,
    min_in_group_fraction=0.5,
    min_fold_change=1.10,
    section_gene_num=10,
    merged_gene_num=15,
    individual_gene_num=35,
    min_cluster_fraction=0.05,
    cat_color=None,
    cnt_colormap="coolwarm",
    x_key="pixel_x",
    y_key="pixel_y",
    fig_scale=2500,
    invert_x=False,
    invert_y=False,
    merged_key="sample",
    print_results=True,
):
    """
    Identify shared heterogeneous subtypes and subtype-specific marker genes
    for one target tissue region.

    This function expects full sample-level reference AnnData objects as input.
    It first subsets each sample to `target_region`, then runs the existing

    The pipeline includes three major steps:

        0. Subset each reference sample to `target_region`.
        1. Cluster spots/cells within each retained sample and identify
           section-level subtype marker genes.
        2. Merge retained samples and identify shared subtypes across samples.
        3. Identify shared subtype-specific marker genes using both:
            - merged-sample differential expression;
            - per-section differential expression followed by overlap selection.

    Parameters
    ----------
    ref_adata_dic : dict
        Dictionary of full sample-level reference AnnData objects.

        Example:
        {
            "H1": adata_H1,
            "G2": adata_G2,
            "E1": adata_E1
        }

        Each AnnData object should contain gene-expression features in `.X`
        and tissue-region labels in `.obs[label_key]`.

    target_region : str
        Tissue region for which heterogeneous subtypes will be inferred.

        Example:
        "Invasive", "CIS", "Immune", or another region label present in
        `adata.obs[label_key]`.

    res_dir : str
        Directory where subtype clustering results, subtype marker gene plots,
        and intermediate output files will be saved.

    label_key : str, default="label"
        Column in `.obs` containing tissue-region annotations.

    min_region_spots : int, default=10
        Minimum number of spots/cells required for a sample to be retained
        after subsetting to `target_region`.

        Samples with fewer than `min_region_spots` observations in the target
        region are excluded from subtype inference for that region.

    pcs_num : int, default=30
        Number of principal components used for section-level subtype clustering.

    section_cluster_method : {"leiden_clusters", "kmeans_clusters"}, default="leiden_clusters"
        Clustering method used within each individual reference sample.

        - "leiden_clusters": graph-based Leiden clustering.
        - "kmeans_clusters": KMeans clustering.

    section_n_clusters : int, default=2
        Number of clusters used for section-level KMeans clustering.
        Only used when `section_cluster_method="kmeans_clusters"`.

    leiden_res : float, default=0.5
        Leiden resolution parameter.
        Only used when `section_cluster_method="leiden_clusters"`.

    n_neighbors : int, default=15
        Number of nearest neighbors used to construct the graph for Leiden
        clustering.

    shared_cluster_key : {"kmeans_clusters", "heatmap_clusters"}, default="kmeans_clusters"
        Clustering method used to identify shared subtypes across merged
        reference samples.

        - "kmeans_clusters": KMeans clustering on the merged gene-expression matrix.
        - "heatmap_clusters": hierarchical clustering based on a heatmap.

    set_shared_clusters_num : int or None, default=None
        User-specified number of shared subtype clusters.

        If None, the number of shared clusters is selected automatically using
        silhouette scores.

    overlap_cutoff : int, default=2
        Minimum number of reference samples in which a subtype marker gene must
        appear to be retained in the per-section overlap strategy.

        The effective cutoff is capped by the maximum observed count for each
        subtype, so the function can still return genes when fewer samples are
        available.

    random_state : int, default=0
        Random seed used for PCA, KMeans, Leiden clustering, and other
        stochastic steps.

    pvals_adj : float, default=0.05
        Adjusted p-value cutoff for subtype marker gene selection.

    min_in_out_group_ratio : float, default=1.0
        Minimum ratio between expression inside the target subtype and outside
        the target subtype.

    min_in_group_fraction : float, default=0.5
        Minimum fraction of spots/cells within the target subtype expressing
        the marker gene.

    min_fold_change : float, default=1.10
        Minimum fold-change required for subtype marker gene selection.

    section_gene_num : int, default=10
        Number of marker genes selected for each subtype cluster within each
        individual reference sample.

    merged_gene_num : int, default=15
        Number of subtype marker genes selected for each shared subtype using
        the merged-sample strategy.

    individual_gene_num : int, default=35
        Number of subtype marker genes selected for each shared subtype within
        each individual section before overlap filtering.

    min_cluster_fraction : float, default=0.05
        Minimum proportion of spots/cells required for a subtype cluster to be
        included in marker gene selection.

        Clusters smaller than this fraction are treated as small clusters and
        excluded from downstream DE gene selection.

    cat_color : list or dict or None, default=None
        Color palette used for categorical subtype-cluster visualization.
        If None, the default categorical palette is used.

    cnt_colormap : str, default="coolwarm"
        Continuous colormap used for subtype marker gene expression plots.

    x_key : str, default="pixel_x"
        Column in `.obs` containing x-coordinates for spatial visualization.

    y_col : str, default="pixel_y"
        Column in `.obs` containing y-coordinates for spatial visualization.

    fig_scale : float, default=2500
        Scaling factor used to determine point size in spatial plots.

    invert_x : bool, default=False
        Whether to invert the x-axis in spatial plots.

    invert_y : bool, default=False
        Whether to invert the y-axis in spatial plots.

    merged_key : str, default="sample"
        Column name used in merged AnnData objects to indicate the source
        reference sample.

    print_results : bool, default=True
        Whether to print intermediate progress, cluster proportions, selected
        genes, and skip messages.

    Returns
    -------
    region_result : HeterogeneousRegionSubtypeResult
        Region-level subtype result object containing shared subtype inference
        results for `target_region`.

        The object includes:

        target_region : str
            Name of the analyzed tissue region.

        section_subtype_genes : dict
            Section-level subtype marker genes.

            Example:
            {
                "H1": ["GeneA", "GeneB", ...],
                "G2": ["GeneC", "GeneD", ...]
            }

        total_genes_list : list
            Union of subtype marker genes selected across retained reference
            samples. These genes are used for shared subtype clustering.

        total_clusters_num : int
            Total number of non-small subtype clusters detected across retained
            reference samples.

        subtype_clusters : pandas.DataFrame or None
            DataFrame containing shared subtype assignments for spots/cells in
            the merged target-region AnnData object.

            This is None if shared subtype clustering is skipped.

        shared_subtype_genes_merged : dict
            Shared subtype marker genes identified from the merged-sample
            differential expression strategy.

            Example:
            {
                "subtype0": ["GeneA", "GeneB"],
                "subtype1": ["GeneC", "GeneD"]
            }

        shared_subtype_genes_individual : dict
            Shared subtype marker genes identified from the per-section
            differential expression and overlap strategy.

            Example:
            {
                "subtype0": ["GeneA"],
                "subtype1": ["GeneC"]
            }

        parameters : dict
            Parameters and metadata used for this region, including retained
            sections, clustering parameters, overlap cutoff, and skip reason
            when applicable.

    Notes
    -----
    If fewer than two reference samples contain at least `min_region_spots`
    observations in `target_region`, the function skips subtype inference and
    returns a `HeterogeneousRegionSubtypeResult` object with a skip reason.

    If section-level subtype clustering does not produce enough subtype genes
    or enough non-small clusters, shared subtype clustering is also skipped.
    """

    region_adata_dic, retained_sections = subset_adata_dic_by_region(
        ref_adata_dic=ref_adata_dic,
        target_region=target_region,
        label_key=label_key,
        min_spots=min_region_spots,
        copy=True,
        print_results=print_results,
    )

    if len(retained_sections) < 2:
        if print_results:
            print(
                f"Skip {target_region}: fewer than two samples have at least "
                f"{min_region_spots} spots."
            )

        return HeterogeneousRegionSubtypeResult(
            target_region=str(target_region),
            parameters={
                "retained_sections": retained_sections,
                "min_region_spots": min_region_spots,
                "skipped_reason": "fewer_than_two_retained_sections",
            },
        )

    region_res_dir = os.path.join(res_dir, str(target_region))
    os.makedirs(region_res_dir, exist_ok=True)

    section_subtype_genes, total_genes_list, total_clusters_num = section_subtype_DE_genes(
        ref_region_adata_dic=region_adata_dic,
        tissue_section_list=retained_sections,
        target_region=target_region,
        res_dir=region_res_dir,
        pcs_num=pcs_num,
        cluster_method=section_cluster_method,
        n_clusters=section_n_clusters,
        leiden_res=leiden_res,
        n_neighbors=n_neighbors,
        random_state=random_state,
        pvals_adj=pvals_adj,
        min_in_out_group_ratio=min_in_out_group_ratio,
        min_in_group_fraction=min_in_group_fraction,
        min_fold_change=min_fold_change,
        gene_num=section_gene_num,
        min_cluster_fraction=min_cluster_fraction,
        cat_color=cat_color,
        cnt_colormap=cnt_colormap,
        x_key=x_key,
        y_key=y_key,
        fig_scale=fig_scale,
        invert_x=invert_x,
        invert_y=invert_y,
    )

    if len(total_genes_list) == 0 or total_clusters_num < 2:
        if print_results:
            print(
                f"Skip shared subtype clustering for {target_region}: "
                "not enough subtype genes or section-level clusters."
            )

        return HeterogeneousRegionSubtypeResult(
            target_region=str(target_region),
            section_subtype_genes=section_subtype_genes,
            total_genes_list=total_genes_list,
            total_clusters_num=total_clusters_num,
            parameters={
                "retained_sections": retained_sections,
                "skipped_reason": "insufficient_subtype_genes_or_clusters",
            },
        )

    subtype_clusters = identify_shared_subtype(
        ref_adata_dic=region_adata_dic,
        tissue_section_list=retained_sections,
        target_region=target_region,
        total_genes_list=total_genes_list,
        total_clusters_num=total_clusters_num,
        res_dir=region_res_dir,
        cat_color=cat_color,
        set_clusters_num=set_shared_clusters_num,
        merged_key=merged_key,
        cluster_key=shared_cluster_key,
        random_state=random_state,
        x_key=x_key,
        y_key=y_key,
        fig_scale=fig_scale,
        invert_x=invert_x,
        invert_y=invert_y,
        print_results=print_results,
    )

    shared_subtype_genes_merged, shared_subtype_genes_individual = shared_subtype_DE_genes(
        ref_region_adata_dic=region_adata_dic,
        tissue_section_list=retained_sections,
        target_region=target_region,
        res_dir=region_res_dir,
        overlap_cutoff=overlap_cutoff,
        subtype_clusters=subtype_clusters,
        subtype_cluster_key=shared_cluster_key,
        pvals_adj=pvals_adj,
        min_in_out_group_ratio=min_in_out_group_ratio,
        min_in_group_fraction=min_in_group_fraction,
        min_fold_change=min_fold_change,
        merged_gene_num=merged_gene_num,
        individual_gene_num=individual_gene_num,
        x_key=x_key,
        y_key=y_key,
        fig_scale=fig_scale,
        invert_x=invert_x,
        invert_y=invert_y,
        merged_key=merged_key,
        cnt_colormap=cnt_colormap,
        min_cluster_fraction=min_cluster_fraction,
        print_results=print_results,
    )

    return HeterogeneousRegionSubtypeResult(
        target_region=str(target_region),
        section_subtype_genes=section_subtype_genes,
        total_genes_list=total_genes_list,
        total_clusters_num=total_clusters_num,
        subtype_clusters=subtype_clusters,
        shared_subtype_genes_merged=shared_subtype_genes_merged,
        shared_subtype_genes_individual=shared_subtype_genes_individual,
        parameters={
            "retained_sections": retained_sections,
            "pcs_num": pcs_num,
            "section_cluster_method": section_cluster_method,
            "section_n_clusters": section_n_clusters,
            "leiden_res": leiden_res,
            "n_neighbors": n_neighbors,
            "shared_cluster_key": shared_cluster_key,
            "set_shared_clusters_num": set_shared_clusters_num,
            "overlap_cutoff": overlap_cutoff,
            "random_state": random_state,
            "min_cluster_fraction": min_cluster_fraction,
        },
    )


def infer_heterogeneity_pipeline(
    ref_adata_dic,
    all_adata,
    dataset_name=None,
    common_genes=None,
    tissue_region_list=None,
    label_key="label",
    sample_key="sample",
    res_dir="heterogeneity_results",
    selection_method="threshold",
    hetero_threshold=0.5,
    top_k=None,
    score_key="hetero_score_sca",
    run_subtype=True,
    min_region_spots=10,
    pvals_adj=0.05,
    min_in_out_group_ratio=1.0,
    min_in_group_fraction=0.5,
    min_fold_change=1.10,
    region_gene_num=10,
    n_perm=200,
    one_sided=True,
    pcs_num=30,
    section_cluster_method="leiden_clusters",
    section_n_clusters=2,
    leiden_res=0.5,
    n_neighbors=15,
    shared_cluster_key="kmeans_clusters",
    set_shared_clusters_num=None,
    overlap_cutoff=2,
    section_gene_num=10,
    merged_gene_num=15,
    individual_gene_num=35,
    min_cluster_fraction=0.05,
    random_state=0,
    cat_color=None,
    cnt_colormap="coolwarm",
    x_key="pixel_x",
    y_key="pixel_y",
    fig_scale=2500,
    invert_x=False,
    invert_y=False,
    merged_key="sample",
    print_results=True,
):
    """
    Run the full reference heterogeneity inference pipeline using gene-expression
    AnnData objects.

    This pipeline is designed for reference data only. It evaluates
    region-specific heterogeneity levels across multiple reference samples,
    selects heterogeneous tissue regions, and optionally identifies shared
    subtypes and subtype-specific marker genes within the selected regions.

    The pipeline includes four main steps:

        1. Infer region-specific heterogeneity scores across reference samples.
        2. Select heterogeneous regions using either:
            - a hard score threshold, or
            - the top-k highest-scoring regions.
        3. For each selected heterogeneous region, identify shared subtypes
           across reference samples.
        4. Identify subtype-specific marker genes using both merged-sample and
           per-section overlap strategies.

    Parameters
    ----------
    ref_adata_dic : dict
        Dictionary of sample-level reference AnnData objects.

        Example:
        {
            "H1": adata_H1,
            "G2": adata_G2,
            "E1": adata_E1
        }

        Each AnnData object should contain gene-expression values in `.X`,
        gene names in `.var_names`, and tissue-region labels in
        `.obs[label_key]`.

    all_adata : AnnData
        Merged reference AnnData object across all samples.

        This object should contain all observations from `ref_adata_dic`, with
        sample identity stored in `.obs[sample_key]` and region labels stored in
        `.obs[label_key]`.

    dataset_name : str or None, default=None
        Optional dataset name stored in the returned result object.

        Example:
        "HER2+BC", "Tonsil", or "Brain_AD".

    common_genes : list or None, default=None
        Common gene set used for heterogeneity score inference.

        If None, genes are inferred from `all_adata.var_names`.

    tissue_region_list : list or None, default=None
        Tissue regions to evaluate.

        If None, regions are inferred from `all_adata.obs[label_key]`, excluding
        missing or unknown labels.

    label_key : str, default="label"
        Column in `.obs` containing tissue-region annotations.

    sample_key : str, default="sample"
        Column in `all_adata.obs` containing sample or section identities.

    res_dir : str, default="heterogeneity_results"
        Directory where subtype clustering results, plots, and intermediate
        outputs are saved.

    selection_method : {"threshold", "top_k"}, default="threshold"
        Method used to select heterogeneous regions.

        - "threshold": select regions whose score is greater than or equal to
          `hetero_threshold`.
        - "top_k": select the top `top_k` regions ranked by `score_key`.

    hetero_threshold : float, default=0.5
        Hard cutoff used when `selection_method="threshold"`.

        Regions with `hetero_summary[score_key] >= hetero_threshold` are
        selected as heterogeneous.

    top_k : int or None, default=None
        Number of top-ranked heterogeneous regions to select when
        `selection_method="top_k"`.

        If `selection_method="top_k"`, this should be a positive integer.

    score_key : str, default="hetero_score_sca"
        Column in `hetero_summary` used for heterogeneous-region selection.

        Usually this is the scaled final heterogeneity score,
        `"hetero_score_sca"`.

    run_subtype : bool, default=True
        Whether to run shared subtype discovery and subtype marker gene
        selection for selected heterogeneous regions.

        If False, only heterogeneity scores and selected regions are returned.

    min_region_spots : int, default=10
        Minimum number of spots/cells required for a region to be evaluated
        within a sample.

        This parameter is used both in heterogeneity score inference and in
        region-level subtype analysis.

    pvals_adj : float, default=0.05
        Adjusted p-value cutoff for marker gene selection.

        Used for both region-specific marker genes and subtype-specific marker
        genes.

    min_in_out_group_ratio : float, default=1.0
        Minimum expression ratio between the target group and the remaining
        groups for marker gene selection.

    min_in_group_fraction : float, default=0.5
        Minimum fraction of spots/cells within the target group expressing a
        candidate marker gene.

    min_fold_change : float, default=1.10
        Minimum fold-change required for marker gene selection.

    region_gene_num : int, default=10
        Number of region-specific marker genes selected per region per sample
        during heterogeneity score inference.

    n_perm : int, default=200
        Number of permutations used to compute permutation-adjusted silhouette
        scores.

    one_sided : bool, default=True
        Whether to use a one-sided permutation p-value when computing adjusted
        silhouette scores.

    pcs_num : int, default=30
        Number of principal components used for section-level subtype clustering.

        Only used when `run_subtype=True`.

    section_cluster_method : {"leiden_clusters", "kmeans_clusters"}, default="leiden_clusters"
        Clustering method used within each individual reference sample for
        subtype discovery.

        Only used when `run_subtype=True`.

    section_n_clusters : int, default=2
        Number of clusters used for section-level KMeans clustering.

        Only used when `section_cluster_method="kmeans_clusters"`.

    leiden_res : float, default=0.5
        Resolution parameter for section-level Leiden clustering.

        Only used when `section_cluster_method="leiden_clusters"`.

    n_neighbors : int, default=15
        Number of neighbors used to construct the nearest-neighbor graph for
        Leiden clustering.

    shared_cluster_key : {"kmeans_clusters", "heatmap_clusters"}, default="kmeans_clusters"
        Method used to identify shared subtype clusters across merged reference
        samples.

        - "kmeans_clusters": KMeans clustering on the merged target-region
          expression matrix.
        - "heatmap_clusters": hierarchical clustering from heatmap linkage.

    set_shared_clusters_num : int or None, default=None
        User-specified number of shared subtype clusters.

        If None, the number of shared subtype clusters is selected
        automatically using silhouette scores.

    overlap_cutoff : int, default=2
        Minimum number of reference samples in which a subtype marker gene must
        appear to be retained in the per-section overlap strategy.

    section_gene_num : int, default=10
        Number of subtype marker genes selected for each subtype cluster within
        each individual reference sample.

    merged_gene_num : int, default=15
        Number of subtype marker genes selected for each shared subtype using
        the merged-sample differential expression strategy.

    individual_gene_num : int, default=35
        Number of subtype marker genes selected for each shared subtype within
        each individual sample before overlap filtering.

    min_cluster_fraction : float, default=0.05
        Minimum proportion of spots/cells required for a subtype cluster to be
        included in marker gene selection.

        Small clusters below this fraction are excluded from downstream subtype
        marker gene selection.

    random_state : int, default=0
        Random seed used for stochastic steps, including PCA, clustering, and
        permutation-based score adjustment.

    cat_color : list or dict or None, default=None
        Color palette used for categorical subtype-cluster visualization.

        If None, the default categorical palette is used.

    cnt_colormap : str, default="coolwarm"
        Continuous colormap used for marker gene expression visualization.

    x_key : str, default="pixel_x"
        Column in `.obs` containing x-coordinates for spatial plots.

    y_col : str, default="pixel_y"
        Column in `.obs` containing y-coordinates for spatial plots.

    fig_scale : float, default=2500
        Scaling factor used to determine point size in spatial plots.

    invert_x : bool, default=False
        Whether to invert the x-axis in spatial plots.

    invert_y : bool, default=False
        Whether to invert the y-axis in spatial plots.

    merged_key : str, default="sample"
        Column name used in merged AnnData objects to indicate the source
        reference sample.

    print_results : bool, default=True
        Whether to print progress messages, intermediate score summaries,
        selected regions, clustering summaries, and selected marker genes.

    Returns
    -------
    result : ReferenceHeterogeneityResult
        Main reference heterogeneity result object.

        The object contains:

        dataset_name : str or None
            Dataset name provided by the user.

        hetero_summary : pandas.DataFrame
            Final region-level heterogeneity score summary.

            This includes marker instability scores, permutation-adjusted
            silhouette scores, and final heterogeneity scores.

        sta_summary : pandas.DataFrame
            Region-level marker gene stability or instability summary.

        perm_sil_summary : pandas.DataFrame
            Region-level permutation-adjusted silhouette score summary.

        selected_regions : list
            Tissue regions selected as heterogeneous according to
            `selection_method`.

        selected_region_scores : pandas.DataFrame
            Score table for selected heterogeneous regions.

        selection_method : str
            Heterogeneous-region selection method used in the pipeline.

        selection_params : dict
            Selection-related parameters, including `score_key`,
            `hetero_threshold`, and `top_k`.

        subtype_results : dict
            Dictionary of region-level subtype results.

            Example:
            {
                "Invasive": RegionSubtypeResult(...),
                "CIS": RegionSubtypeResult(...)
            }

            This dictionary is empty if `run_subtype=False`.

        sample_names : list
            Names of reference samples included in `ref_adata_dic`.

        parameters : dict
            Main pipeline parameters used for heterogeneity score inference,
            region selection, subtype discovery, and output organization.

    Notes
    -----
    This function assumes that preprocessing has already been completed before
    running the pipeline. In particular, `ref_adata_dic` and `all_adata` should
    usually be generated from the same processed gene-expression data, with
    consistent genes and observation metadata.

    This pipeline focuses only on gene-expression AnnData objects for
    heterogeneity inference. Image features or multimodal features are not used
    in this function.
    """

    os.makedirs(res_dir, exist_ok=True)

    score_results = infer_heterogeneity_scores(
        ref_adata_dic=ref_adata_dic,
        all_adata=all_adata,
        common_genes=common_genes,
        tissue_region_list=tissue_region_list,
        label_key=label_key,
        sample_key=sample_key,
        pvals_adj=pvals_adj,
        min_in_out_group_ratio=min_in_out_group_ratio,
        min_in_group_fraction=min_in_group_fraction,
        min_fold_change=min_fold_change,
        gene_num=region_gene_num,
        n_perm=n_perm,
        random_state=random_state,
        one_sided=one_sided,
        min_spots=min_region_spots,
        print_results=print_results,
    )

    selected_regions, selected_region_scores = select_heterogeneous_regions(
        hetero_summary=score_results["hetero_summary"],
        method=selection_method,
        score_key=score_key,
        threshold=hetero_threshold,
        top_k=top_k,
        include_ties=True,
        print_results=print_results,
    )

    subtype_results = {}

    if run_subtype:
        for target_region in selected_regions:
            if print_results:
                print("\n" + "=" * 70)
                print(f"Subtype analysis for heterogeneous region: {target_region}")
                print("=" * 70)

            subtype_results[target_region] = infer_region_shared_subtypes(
                ref_adata_dic=ref_adata_dic,
                target_region=target_region,
                res_dir=os.path.join(res_dir, "heterogeneous_region_subtypes"),
                label_key=label_key,
                min_region_spots=min_region_spots,
                pcs_num=pcs_num,
                section_cluster_method=section_cluster_method,
                section_n_clusters=section_n_clusters,
                leiden_res=leiden_res,
                n_neighbors=n_neighbors,
                shared_cluster_key=shared_cluster_key,
                set_shared_clusters_num=set_shared_clusters_num,
                overlap_cutoff=overlap_cutoff,
                random_state=random_state,
                pvals_adj=pvals_adj,
                min_in_out_group_ratio=min_in_out_group_ratio,
                min_in_group_fraction=min_in_group_fraction,
                min_fold_change=min_fold_change,
                section_gene_num=section_gene_num,
                merged_gene_num=merged_gene_num,
                individual_gene_num=individual_gene_num,
                min_cluster_fraction=min_cluster_fraction,
                cat_color=cat_color,
                cnt_colormap=cnt_colormap,
                x_key=x_key,
                y_key=y_key,
                fig_scale=fig_scale,
                invert_x=invert_x,
                invert_y=invert_y,
                merged_key=merged_key,
                print_results=print_results,
            )

    result = ReferenceHeterogeneityResult(
        dataset_name=dataset_name,
        hetero_summary=score_results["hetero_summary"],
        sta_summary=score_results["sta_summary"],
        perm_sil_summary=score_results["perm_sil_summary"],
        selected_regions=selected_regions,
        selected_region_scores=selected_region_scores,
        region_marker_genes=score_results["d_g_r"],
        selection_method=selection_method,
        selection_params={
            "score_key": score_key,
            "hetero_threshold": hetero_threshold,
            "top_k": top_k,
        },
        subtype_results=subtype_results,
        sample_names=list(ref_adata_dic.keys()),
        parameters={
            "label_key": label_key,
            "sample_key": sample_key,
            "min_region_spots": min_region_spots,
            "pvals_adj": pvals_adj,
            "min_in_out_group_ratio": min_in_out_group_ratio,
            "min_in_group_fraction": min_in_group_fraction,
            "min_fold_change": min_fold_change,
            "region_gene_num": region_gene_num,
            "n_perm": n_perm,
            "one_sided": one_sided,
            "run_subtype": run_subtype,
            "res_dir": res_dir,
        },
    )

    return result


#=======================================================================
# Part 4. Annotate heterogeneity in query samples
#=======================================================================
