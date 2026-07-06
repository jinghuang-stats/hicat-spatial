import math
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

# Local package imports
from .preprocessing.preprocess_util import preprocess_adata_dic
from .utils import get_gene_vector, select_region_markers_across_samples


#=======================================================================
# Reference selection result objects
#=======================================================================
@dataclass
class ReferenceSelectionResult:
    """
    Result object for query-specific reference selection.

    This dataclass stores reference similarity scores, ranked references,
    selected references, selection summaries, and the parameters used for
    reference selection.

    Attributes
    ----------
    selected_refs_dic : dict
        Dictionary mapping each query section to selected reference sections.

        Example
        -------
        selected_refs_dic["H2"] = ["H1", "G2"]

    similarity_summary_dic : dict
        Dictionary mapping each query section to a DataFrame containing
        reference-level similarity scores.

        Each DataFrame is indexed by reference section and usually contains:
            - raw similarity score, e.g. "KS_similarity"
            - reference weight, e.g. "weight"
            - weighted similarity score, e.g. "weighted_KS_similarity"

    selection_summary_dic : dict
        Dictionary mapping each query section to a DataFrame containing
        similarity scores and selection indicators.

        Each DataFrame usually contains:
            - similarity scores
            - selection score
            - whether each reference is above the minimum similarity level
            - whether each reference is selected by the main rule
            - final selected indicator
            - selection cutoff
            - max score
            - minimum similarity level

    ranked_refs_dic : dict
        Dictionary mapping each query section to ranked reference sections.

        Example
        -------
        ranked_refs_dic["H2"] = ["H1", "G2", "E1"]

    weights : pandas.DataFrame, optional
        Reference-section weights used to compute weighted similarity scores.

    region_dic : dict, optional
        Dictionary mapping each reference section to valid tissue regions used
        for reference-weight calculation.

    selection_metric : str
        Score column used for final reference selection.

        Example
        -------
        "weighted_KS_similarity"

    selection_mode : str
        Reference selection mode.

        Usually one of:
            - "cutoff"
            - "top_k"

    selection_cutoffs : dict
        Dictionary recording parameter cutoffs used for selection.

        Example
        -------
        {
            "alpha": 0.9,
            "top_k": 3,
            "min_similarity_level": 0.7
        }

    params : dict
        Full pipeline parameters.

    d_s_All : dict, optional
        Nested dictionary containing gene-level query-reference similarity
        results.

    gene_list_All : dict, optional
        Marker genes used for each reference section.

    d_g_All : dict, optional
        Full marker-gene selection results.

    ref_adata_dic : dict, optional
        Processed reference AnnData dictionary.

    qry_adata_dic : dict, optional
        Processed query AnnData dictionary.
    """

    selected_refs_dic: Dict[str, List[str]]
    similarity_summary_dic: Dict[str, pd.DataFrame]
    selection_summary_dic: Dict[str, pd.DataFrame]
    ranked_refs_dic: Dict[str, List[str]]

    selection_metric: str
    selection_mode: str
    selection_cutoffs: Dict[str, Any]

    weights: Optional[pd.DataFrame] = None
    region_dic: Optional[Dict[str, List[str]]] = None
    params: Dict[str, Any] = field(default_factory=dict)

    d_s_All: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None
    gene_list_All: Optional[Dict[str, List[str]]] = None
    d_g_All: Optional[Dict[str, Any]] = None

    ref_adata_dic: Optional[Dict[str, Any]] = None
    qry_adata_dic: Optional[Dict[str, Any]] = None

    def get_selected_refs(self, query_section: str) -> List[str]:
        """Return selected reference sections for one query section."""
        if query_section not in self.selected_refs_dic:
            raise KeyError(
                f"{query_section!r} is not found in selected_refs_dic."
            )
        return self.selected_refs_dic[query_section]

    def get_similarity_summary(self, query_section: str) -> pd.DataFrame:
        """Return reference similarity summary for one query section."""
        if query_section not in self.similarity_summary_dic:
            raise KeyError(
                f"{query_section!r} is not found in similarity_summary_dic."
            )
        return self.similarity_summary_dic[query_section]

    def get_selection_summary(self, query_section: str) -> pd.DataFrame:
        """Return selection summary for one query section."""
        if query_section not in self.selection_summary_dic:
            raise KeyError(
                f"{query_section!r} is not found in selection_summary_dic."
            )
        return self.selection_summary_dic[query_section]

    def get_ranked_refs(self, query_section: str) -> List[str]:
        """Return ranked reference sections for one query section."""
        if query_section not in self.ranked_refs_dic:
            raise KeyError(
                f"{query_section!r} is not found in ranked_refs_dic."
            )
        return self.ranked_refs_dic[query_section]

    def to_summary_df(self) -> pd.DataFrame:
        """
        Create a compact query-level summary table.

        Returns
        -------
        summary_df : pandas.DataFrame
            One-row-per-query summary with selected references, ranked references,
            selection metric, selection mode, and cutoff parameters.
        """

        rows = []

        for query_section in self.selected_refs_dic:
            selected_refs = self.selected_refs_dic[query_section]
            ranked_refs = self.ranked_refs_dic.get(query_section, [])

            rows.append(
                {
                    "query_section": query_section,
                    "selected_refs": selected_refs,
                    "n_selected_refs": len(selected_refs),
                    "ranked_refs": ranked_refs,
                    "selection_metric": self.selection_metric,
                    "selection_mode": self.selection_mode,
                    **self.selection_cutoffs,
                }
            )

        return pd.DataFrame(rows).set_index("query_section")


# ============================================================
# 1. Reference section weights
# ============================================================
def get_valid_regions(
    input_adata,
    label_key="label",
    min_prop=0.05,
    exclude_labels=("nan", "unknown"),
    exclude_mode="contains",
):
    """
    Identify valid tissue regions in one AnnData object.

    Parameters
    ----------
    input_adata : AnnData
        Input AnnData object.
    label_key : str
        Region annotation column in input_adata.obs.
    min_prop : float
        Minimum region proportion required.
    exclude_labels : tuple
        Labels to exclude.
    exclude_mode : {"contains", "exact"}
        Whether to exclude labels by substring matching or exact matching.

    Returns
    -------
    filtered_regions : list
        Valid tissue regions.
    """

    if label_key not in input_adata.obs.columns:
        raise KeyError(f"{label_key!r} is not in input_adata.obs.")

    region_props = input_adata.obs[label_key].value_counts(normalize=True)

    filtered_regions = []

    for region, prop in region_props.items():
        region_str = str(region).lower()

        if exclude_mode == "contains":
            is_excluded = any(exclude_label in region_str for exclude_label in exclude_labels)
        elif exclude_mode == "exact":
            is_excluded = region_str in exclude_labels
        else:
            raise ValueError("exclude_mode must be either 'contains' or 'exact'.")

        if (not is_excluded) and (prop > min_prop):
            filtered_regions.append(region)

    return filtered_regions


def infer_reference_weights(
    ref_adata_dic,
    ref_section_list=None,
    label_key="label",
    min_prop=0.05,
    exclude_labels=("nan", "unknown"),
    exclude_mode="contains",
    weight_key="weight",
    print_results=True,
):
    """
    Infer reference-section weights based on tissue-region diversity.

    Weight formula:
        weight = sqrt(number of valid regions in section / maximum number of valid regions)

    This generalizes the weighting logic used in the HER2+ BC and Brain Visium scripts.
    """

    if ref_section_list is None:
        ref_section_list = list(ref_adata_dic.keys())

    region_dic = {}
    max_region_num = 0

    for section in ref_section_list:
        if print_results:
            print(f"------------------- {section} -------------------")

        filtered_regions = get_valid_regions(
            input_adata=ref_adata_dic[section],
            label_key=label_key,
            min_prop=min_prop,
            exclude_labels=exclude_labels,
            exclude_mode=exclude_mode,
        )

        region_dic[section] = filtered_regions
        max_region_num = max(max_region_num, len(filtered_regions))

        if print_results:
            print("Filtered tissue regions:")
            print(filtered_regions)
            print(f"Number of regions: {len(filtered_regions)}")

    weights = pd.DataFrame(
        np.zeros((len(ref_section_list), 1)),
        index=ref_section_list,
        columns=[weight_key],
    )

    for section in ref_section_list:
        if max_region_num == 0:
            region_weight = 1.0
        else:
            region_weight = math.sqrt(len(region_dic[section]) / max_region_num)

        weights.loc[section, weight_key] = region_weight

    if print_results:
        print("======================= Reference weights =======================")
        print(weights)

    return weights, region_dic


# ============================================================
# 2. Query-reference similarity
# ============================================================
def compute_gene_distribution_similarity(
    ref_adata,
    qry_adata,
    gene,
    method="ks",
):
    """
    Compute one-gene distribution similarity between reference and query.

    For method="ks":
        Kolmogorov-Smirnov (KS) statistic ranges from 0 to 1.
        measures the maximum difference between the two eCDFs.
        similarity = 1 - KS statistic.
        Larger value means more similar.
    """

    rv_ref = get_gene_vector(ref_adata, gene)
    rv_qry = get_gene_vector(qry_adata, gene)

    if method == "ks":
        ks_stat = stats.ks_2samp(rv_ref, rv_qry).statistic # two-sample Kolmogorov-Smirnov test
        similarity = 1 - ks_stat
    else:
        raise ValueError("Currently only method='ks' is supported.")

    return similarity


def compute_pairwise_ref_query_similarity(
    ref_adata,
    qry_adata,
    gene_list,
    qry_section,
    summary_key="average",
    method="ks"
):
    """
    Compute marker-gene similarity between one reference and one query section.

    Returns
    -------
    similarity_res : pd.DataFrame
        One-row DataFrame with gene-level similarities and average score.
    """

    available_genes = [
        gene for gene in gene_list
        if gene in ref_adata.var_names and gene in qry_adata.var_names
    ]

    similarity_res = pd.DataFrame(
        np.zeros((1, len(available_genes))),
        index=[qry_section],
        columns=available_genes,
    )

    for gene in available_genes:
        similarity_res.loc[qry_section, gene] = compute_gene_distribution_similarity(
            ref_adata=ref_adata,
            qry_adata=qry_adata,
            gene=gene,
            method=method,
        )

    if len(available_genes) > 0:
        similarity_res[summary_key] = similarity_res.loc[qry_section, available_genes].mean()
    else:
        similarity_res[summary_key] = np.nan

    return similarity_res


def compute_reference_similarity(
    ref_adata_dic,
    qry_adata_dic,
    gene_list_All,
    weights=None,
    ref_section_list=None,
    qry_section_list=None,
    weight_key="weight",
    similarity_key="KS_similarity",
    weighted_similarity_key=None,
    summary_key="average",
    method="ks",
    sort_by="weighted",
    print_results=True,
):
    """
    Compute query-reference similarity for reference selection.

    Parameters
    ----------
    ref_adata_dic : dict
        Reference AnnData dictionary.
    qry_adata_dic : dict
        Query AnnData dictionary.
    gene_list_All : dict
        Reference-section marker genes.
    weights : pd.DataFrame or None
        Reference weights from infer_reference_weights().
        If None, all references receive weight 1.
    sort_by : {"similarity", "weighted"}
        Whether to sort references by raw similarity or weighted similarity.

    Returns
    -------
    d_s_All : dict
        Nested dictionary containing gene-level similarity results.
    similarity_summary_dic : dict
        Summary DataFrame for each query section.
    ranked_refs_dic : dict
        Ranked reference sections for each query section.
    """

    if ref_section_list is None:
        ref_section_list = list(ref_adata_dic.keys())

    if qry_section_list is None:
        qry_section_list = list(qry_adata_dic.keys())

    if weighted_similarity_key is None:
        weighted_similarity_key = "weighted_" + similarity_key

    if weights is None:
        weights = pd.DataFrame(
            np.ones((len(ref_section_list), 1)),
            index=ref_section_list,
            columns=[weight_key],
        )

    d_s_All = {}
    similarity_summary_dic = {}
    ranked_refs_dic = {}

    for qry_section in qry_section_list:
        if print_results:
            print(f"==================== Query section: {qry_section} ====================")

        d_s = {}

        similarity_summary = pd.DataFrame(
            np.zeros((len(ref_section_list), 3)),
            index=ref_section_list,
            columns=[similarity_key, weight_key, weighted_similarity_key],
        )

        qry_adata_test = qry_adata_dic[qry_section]

        for ref_section in ref_section_list:
            ref_adata_test = ref_adata_dic[ref_section]
            gene_list = gene_list_All[ref_section]

            similarity_res = compute_pairwise_ref_query_similarity(
                ref_adata=ref_adata_test,
                qry_adata=qry_adata_test,
                gene_list=gene_list,
                qry_section=qry_section,
                summary_key=summary_key,
                method=method,
            )

            d_s[ref_section] = similarity_res

            raw_similarity = similarity_res.loc[qry_section, summary_key]
            ref_weight = weights.loc[ref_section, weight_key]

            similarity_summary.loc[ref_section, similarity_key] = raw_similarity
            similarity_summary.loc[ref_section, weight_key] = ref_weight
            similarity_summary.loc[ref_section, weighted_similarity_key] = (
                raw_similarity * ref_weight
            )

            if print_results:
                print(f"========== {qry_section} vs. {ref_section} ==========")
                print(similarity_res)

        d_s_All[qry_section] = d_s

        if sort_by == "similarity":
            similarity_summary = similarity_summary.sort_values(
                by=similarity_key,
                ascending=False,
            )
        elif sort_by == "weighted":
            similarity_summary = similarity_summary.sort_values(
                by=weighted_similarity_key,
                ascending=False,
            )
        else:
            raise ValueError("sort_by must be either 'similarity' or 'weighted'.")

        similarity_summary_dic[qry_section] = similarity_summary
        ranked_refs_dic[qry_section] = similarity_summary.index.tolist()

        if print_results:
            print(f"========== Similarity summary for {qry_section} ==========")
            print(similarity_summary)

    return d_s_All, similarity_summary_dic, ranked_refs_dic


# ============================================================
# 3. Select query-specific references
# ============================================================
def select_refs_for_each_query(
    similarity_summary_dic,
    selection_mode="cutoff",
    alpha=0.9,
    top_k=3,
    min_similarity_level=0.7,
    score_key=None,
    similarity_key="KS_similarity",
    weighted_similarity_key=None,
    sort_by="weighted",
    print_results=True,
):
    """
    Select reference sections for each query section based on similarity scores.

    Selection logic
    ---------------
    A reference is selected only if it satisfies BOTH:

        1. score >= min_similarity_level

    and

        2. the main selection rule:
            - if selection_mode="cutoff":
                score >= alpha * max_score
            - if selection_mode="top_k":
                reference is among the top_k ranked references

    If no reference has score >= min_similarity_level, the function reports this
    and keeps only the reference with the largest similarity score.

    Parameters
    ----------
    similarity_summary_dic : dict
        Dictionary of similarity summary DataFrames.
        Example:
            similarity_summary_dic[query_section] = DataFrame indexed by ref sections.

    selection_mode : {"cutoff", "top_k"}
        Reference selection mode.

    alpha : float, default=0.9
        Used when selection_mode="cutoff".
        The cutoff is alpha * maximum similarity score for each query.

    top_k : int, default=3
        Used when selection_mode="top_k".
        Number of top-ranked references to keep.

    min_similarity_level : float, default=0.7
        Lowest acceptable similarity level.

    score_key : str or None
        Column used for reference selection.
        If None, the function determines it from sort_by.

    similarity_key : str
        Raw similarity column name.

    weighted_similarity_key : str or None
        Weighted similarity column name.

    sort_by : {"weighted", "similarity"}
        Determines default score_key when score_key=None.

    print_results : bool
        Whether to print selected references.

    Returns
    -------
    selected_refs_dic : dict
        Dictionary mapping each query section to selected reference sections.

    selection_summary_dic : dict
        Dictionary of updated similarity summary DataFrames with selection indicators.
    """

    if weighted_similarity_key is None:
        weighted_similarity_key = "weighted_" + similarity_key

    if score_key is None:
        if sort_by == "weighted":
            score_key = weighted_similarity_key
        elif sort_by == "similarity":
            score_key = similarity_key
        else:
            raise ValueError("sort_by must be either 'weighted' or 'similarity'.")

    if selection_mode not in ["cutoff", "top_k"]:
        raise ValueError("selection_mode must be either 'cutoff' or 'top_k'.")

    if selection_mode == "cutoff":
        if alpha <= 0:
            raise ValueError("alpha must be positive when selection_mode='cutoff'.")

    if selection_mode == "top_k":
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer when selection_mode='top_k'.")

    selected_refs_dic = {}
    selection_summary_dic = {}

    for qry_section, similarity_summary in similarity_summary_dic.items():

        summary_df = similarity_summary.copy()

        if score_key not in summary_df.columns:
            raise KeyError(
                f"{score_key!r} is not in similarity_summary_dic[{qry_section!r}].columns."
            )

        # Sort references by selected score.
        summary_df = summary_df.sort_values(by=score_key, ascending=False)

        scores = summary_df[score_key]
        valid_scores = scores.dropna()

        if valid_scores.shape[0] == 0:
            raise ValueError(
                f"No valid similarity scores found for query section {qry_section!r}."
            )

        max_score = valid_scores.max()
        best_ref = valid_scores.idxmax()

        # --------------------------------------------------------
        # References passing the lowest similarity level
        # --------------------------------------------------------
        refs_above_min_level = valid_scores[
            valid_scores >= min_similarity_level
        ].index.tolist()

        # --------------------------------------------------------
        # Main selection rule
        # --------------------------------------------------------
        if selection_mode == "cutoff":
            similarity_cutoff = alpha * max_score

            refs_by_main_rule = valid_scores[
                valid_scores >= similarity_cutoff
            ].index.tolist()

        elif selection_mode == "top_k":
            similarity_cutoff = np.nan

            refs_by_main_rule = valid_scores.head(
                min(top_k, valid_scores.shape[0])
            ).index.tolist()

        # --------------------------------------------------------
        # Final selected references: intersection, not union
        # --------------------------------------------------------
        if len(refs_above_min_level) == 0:
            selected_refs = [best_ref]
            no_ref_above_min_level = True

            if print_results:
                print(
                    f"[Warning] Query {qry_section}: no reference has "
                    f"{score_key} >= {min_similarity_level}. "
                    f"Only keeping the best reference: {best_ref} "
                    f"with score {round(max_score, 4)}."
                )

        else:
            selected_refs = [
                ref for ref in refs_by_main_rule
                if ref in refs_above_min_level
            ]

            no_ref_above_min_level = False

            # This should rarely happen if top_k >= 1 or alpha <= 1,
            # but this fallback makes the function safer.
            if len(selected_refs) == 0:
                best_ref_above_min = valid_scores.loc[refs_above_min_level].idxmax()
                selected_refs = [best_ref_above_min]

                if print_results:
                    print(
                        f"[Warning] Query {qry_section}: no reference passes both "
                        f"the minimum similarity level and the main selection rule. "
                        f"Keeping the best reference above the minimum level: "
                        f"{best_ref_above_min}."
                    )

        # --------------------------------------------------------
        # Add selection indicators to summary table
        # --------------------------------------------------------
        summary_df["above_min_similarity_level"] = summary_df.index.isin(
            refs_above_min_level
        )

        summary_df["selected_by_main_rule"] = summary_df.index.isin(
            refs_by_main_rule
        )

        summary_df["selected"] = summary_df.index.isin(selected_refs)

        summary_df["selection_cutoff"] = similarity_cutoff
        summary_df["max_score"] = max_score
        summary_df["min_similarity_level"] = min_similarity_level
        summary_df["no_ref_above_min_level"] = no_ref_above_min_level

        selected_refs_dic[qry_section] = selected_refs
        selection_summary_dic[qry_section] = summary_df

        if print_results:
            print(f"==================== {qry_section} selected references ====================")
            print(f"Selection score key: {score_key}")
            print(f"Selection mode: {selection_mode}")

            if selection_mode == "cutoff":
                print(f"Alpha: {alpha}")
                print(f"Maximum score: {round(max_score, 4)}")
                print(f"Alpha cutoff: {round(similarity_cutoff, 4)}")

            if selection_mode == "top_k":
                print(f"Top k: {top_k}")

            print(f"Minimum similarity level: {min_similarity_level}")
            print(f"References above minimum level: {refs_above_min_level}")
            print(f"References selected by main rule: {refs_by_main_rule}")
            print(f"Final selected references: {selected_refs}")

    return selected_refs_dic, selection_summary_dic


def select_references_pipeline(
    ref_adata_dic,
    qry_adata_dic,
    ref_section_list=None,
    qry_section_list=None,
    label_key="label",
    low_exp_thres=0.05,
    normalize=True,
    normalization_method="min_max",
    min_region_prop=0.05,
    exclude_labels=("nan", "unknown"),
    exclude_mode="contains",
    pvals_adj=0.05,
    min_in_out_group_ratio=1.0,
    min_in_group_fraction=0.5,
    min_fold_change=1.10,
    gene_num=10,
    weight_key="weight",
    similarity_key="KS_similarity",
    summary_key="average",
    similarity_method="ks",
    sort_by="weighted",
    selection_mode="cutoff",
    alpha=0.9,
    top_k=3,
    min_similarity_level=0.7,
    selection_score_key=None,
    preprocess_ref=True,
    preprocess_qry=True,
    print_results=True,
):
    """
    Run the full reference-selection pipeline for choosing suitable reference
    sections for each query section.

    This pipeline performs the following steps:

    ```
    1. Resolves reference and query section lists.
    2. Optionally preprocesses reference and query AnnData objects.
    3. Computes reference-section weights based on tissue-region diversity.
    4. Selects region-specific marker genes from reference sections.
    5. Computes query-reference similarity using the selected marker genes.
    6. Ranks and selects candidate reference sections for each query section.
    ```

    ## Parameters

    ref_adata_dic : dict[str, AnnData]
    Dictionary of reference AnnData objects. Keys are reference section names,
    and values are AnnData objects containing reference-section data. Each
    AnnData object must contain tissue-region labels in `adata.obs[label_key]`.

    qry_adata_dic : dict[str, AnnData]
    Dictionary of query AnnData objects. Keys are query section names, and
    values are AnnData objects containing query-section data. Query objects do
    not need known tissue-region labels for reference selection.

    ref_section_list : list[str], optional
    Reference sections to include in the pipeline. Each section name must be a
    key in `ref_adata_dic`. If None, all sections in `ref_adata_dic` are used.        
    
    qry_section_list : list[str], optional
    Query sections to include in the pipeline. Each section name must be a key
    in `qry_adata_dic`. If None, all sections in `qry_adata_dic` are used.        
    
    label_key : str, default="label"
    Column name in `adata.obs` containing tissue-region or annotation labels
    for reference sections. This label is used for region filtering, reference
    weight calculation, and marker-gene selection.        
    
    low_exp_thres : float, default=0.05
    Low-expression filtering threshold used during AnnData preprocessing. Genes
    expressed in fewer than this fraction of spots/cells may be removed,
    depending on the implementation of `preprocess_adata_dic`.        
    
    normalize : bool, default=True
    Whether to normalize expression values during preprocessing.        
    
    normalization_method : str, default="min_max"
    Normalization method passed to `preprocess_adata_dic`. For example,
    `"min_max"` indicates min-max scaling.        
    
    min_region_prop : float, default=0.05
    Minimum proportion of spots/cells required for a tissue region to be
    considered valid within a reference section when computing reference
    weights. Regions below this proportion are ignored.        
    
    exclude_labels : tuple[str, ...], default=("nan", "unknown")
    Labels to exclude from region-weight calculation and marker-gene selection.
    Common examples include missing, background, or unknown labels.        
    
    exclude_mode : {"contains", "exact"}, default="contains"
    Strategy for matching labels in `exclude_labels`.        
    
    ```
    - `"contains"`: exclude labels containing any excluded string.
    - `"exact"`: exclude labels exactly matching one of the excluded strings.
    ```        
    
    pvals_adj : float, default=0.05
    Adjusted p-value threshold used for marker-gene selection.        
    
    min_in_out_group_ratio : float, default=1.0
    Minimum required ratio between within-region expression and out-of-region
    expression for a gene to be considered region-specific.        
    
    min_in_group_fraction : float, default=0.5
    Minimum fraction of spots/cells within a tissue region that must express a
    gene for it to be considered as a marker.        
    
    min_fold_change : float, default=1.10
    Minimum fold-change threshold used for marker-gene selection.        
    
    gene_num : int, default=10
    Maximum number of marker genes to select for each tissue region in each
    reference section.        
    
    weight_key : str, default="weight"
    Column name used to store or retrieve reference-section weights in the
    reference-weight table.        
    
    similarity_key : str, default="KS_similarity"
    Name of the unweighted similarity score produced by
    `compute_reference_similarity`.        
    
    summary_key : str, default="average"
    Summary statistic used to aggregate region-level similarity scores into a
    section-level query-reference similarity score.        
    
    similarity_method : str, default="ks"
    Method used to compute query-reference similarity. For example, `"ks"`
    indicates a Kolmogorov-Smirnov-based similarity calculation.        
    
    sort_by : {"weighted", "similarity"}, default="weighted"
    Similarity score used for ranking reference sections.        
    
    ```
    - `"weighted"`: rank by weighted similarity.
    - `"similarity"`: rank by unweighted similarity.
    ```        
    
    selection_mode : str, default="cutoff"
    Strategy used to select final reference sections from the ranked
    query-reference similarity table. Common choices include cutoff-based
    selection and top-k selection, depending on the implementation of
    `select_refs_for_each_query`.        
    
    alpha : float, default=0.9
    Relative cutoff used when `selection_mode` is cutoff-based. References with
    scores greater than or equal to `alpha` times the best score may be selected.        
    
    top_k : int, default=3
    Maximum number of top-ranked reference sections to select for each query
    when top-k selection is used.        
    
    min_similarity_level : float, default=0.7
    Minimum acceptable similarity score. This can be used as a fallback or
    lower-bound filter to avoid selecting weakly matched reference sections.        
    
    selection_score_key : str, optional
    Specific score column to use for final reference selection. If None, the
    score is chosen automatically based on `sort_by`.        
    
    preprocess_ref : bool, default=True
    Whether to preprocess reference AnnData objects before reference selection.        
    
    preprocess_qry : bool, default=True
    Whether to preprocess query AnnData objects before reference selection.        
    
    print_results : bool, default=True
    Whether to print progress messages and intermediate summaries.        
    
    ## Returns        
    
    result : ReferenceSelectionResult
    Dataclass object containing all major intermediate and final outputs from
    the reference-selection pipeline.        
    
    ```
    The returned object contains the following fields:        
    
    selected_refs_dic : dict[str, list[str]]
        Dictionary mapping each query section to its selected reference
        sections.        
    
    similarity_summary_dic : dict
        Query-reference similarity summaries. Usually contains section-level
        similarity scores for each query-reference pair, including weighted and
        unweighted similarity scores.        
    
    selection_summary_dic : dict
        Summary of the final reference-selection result for each query section,
        including selected references and the score cutoffs used.        
    
    ranked_refs_dic : dict
        Ranked reference sections for each query section before final
        selection.        
    
    weights : pandas.DataFrame or dict
        Reference-section weights inferred from tissue-region diversity.        
    
    region_dic : dict
        Dictionary describing valid tissue regions retained for each reference
        section after filtering by `min_region_prop` and `exclude_labels`.        
    
    selection_metric : str
        Name of the score used for final reference selection.        
    
    selection_mode : str
        Reference-selection mode used in the final selection step.        
    
    selection_cutoffs : dict
        Dictionary storing selection cutoff parameters, including `alpha`,
        `top_k`, and `min_similarity_level`.        
    
    params : dict
        Dictionary containing all major input parameters used to run the
        pipeline.        
    
    d_s_All : dict
        Detailed query-reference similarity results, typically including
        region-level or marker-gene-level similarity scores.        
   
    gene_list_All : dict
        Selected marker genes for each reference section and tissue region.        
    
    d_g_All : dict
        Detailed marker-gene selection results for reference sections.        
    
    ref_adata_dic : dict[str, AnnData]
        Reference AnnData objects used in the pipeline after optional
        preprocessing and section filtering.        
    
    qry_adata_dic : dict[str, AnnData]
        Query AnnData objects used in the pipeline after optional preprocessing
        and section filtering.
    ```        
    
    """

    # --------------------------------------------------------
    # Step 0. Resolve section lists
    # --------------------------------------------------------
    if ref_section_list is None:
        ref_section_list = list(ref_adata_dic.keys())

    if qry_section_list is None:
        qry_section_list = list(qry_adata_dic.keys())

    missing_ref_sections = [
        section for section in ref_section_list
        if section not in ref_adata_dic
    ]

    missing_qry_sections = [
        section for section in qry_section_list
        if section not in qry_adata_dic
    ]

    if len(missing_ref_sections) > 0:
        raise KeyError(
            f"The following ref_section_list entries are not in ref_adata_dic: "
            f"{missing_ref_sections}"
        )

    if len(missing_qry_sections) > 0:
        raise KeyError(
            f"The following qry_section_list entries are not in qry_adata_dic: "
            f"{missing_qry_sections}"
        )

    # Keep only requested sections.
    ref_adata_dic = {
        section: ref_adata_dic[section]
        for section in ref_section_list
    }

    qry_adata_dic = {
        section: qry_adata_dic[section]
        for section in qry_section_list
    }

    # --------------------------------------------------------
    # Step 1. Preprocess reference and query AnnData objects
    # --------------------------------------------------------
    if preprocess_ref:
        ref_adata_dic = preprocess_adata_dic(
            adata_dic=ref_adata_dic,
            section_list=ref_section_list,
            low_exp_thres=low_exp_thres,
            normalize=normalize,
            normalization_method=normalization_method,
            print_results=print_results,
        )

    if preprocess_qry:
        qry_adata_dic = preprocess_adata_dic(
            adata_dic=qry_adata_dic,
            section_list=qry_section_list,
            low_exp_thres=low_exp_thres,
            normalize=normalize,
            normalization_method=normalization_method,
            print_results=print_results,
        )

    # --------------------------------------------------------
    # Step 2. Infer reference weights
    # --------------------------------------------------------
    weights, region_dic = infer_reference_weights(
        ref_adata_dic=ref_adata_dic,
        ref_section_list=ref_section_list,
        label_key=label_key,
        min_prop=min_region_prop,
        exclude_labels=exclude_labels,
        exclude_mode=exclude_mode,
        weight_key=weight_key,
        print_results=print_results,
    )

    # --------------------------------------------------------
    # Step 3. Select reference-section marker genes
    # --------------------------------------------------------
    d_g_All, gene_list_All = select_region_markers_across_samples(
        ref_adata_dic=ref_adata_dic,
        label_key=label_key,
        gene_num=gene_num,
        min_fold_change=min_fold_change,
        min_in_out_group_ratio=min_in_out_group_ratio,
        min_in_group_fraction=min_in_group_fraction,
        pvals_adj=pvals_adj,
        exclude_labels=exclude_labels,
        exclude_mode=exclude_mode,
        print_results=print_results,
    )

    # --------------------------------------------------------
    # Step 4. Compute query-reference similarity
    # --------------------------------------------------------
    d_s_All, similarity_summary_dic, ranked_refs_dic = compute_reference_similarity(
        ref_adata_dic=ref_adata_dic,
        qry_adata_dic=qry_adata_dic,
        gene_list_All=gene_list_All,
        weights=weights,
        ref_section_list=ref_section_list,
        qry_section_list=qry_section_list,
        weight_key=weight_key,
        similarity_key=similarity_key,
        summary_key=summary_key,
        method=similarity_method,
        sort_by=sort_by,
        print_results=print_results,
    )

    # --------------------------------------------------------
    # Step 5. Select final references for each query
    # --------------------------------------------------------
    weighted_similarity_key = "weighted_" + similarity_key

    selected_refs_dic, selection_summary_dic = select_refs_for_each_query(
        similarity_summary_dic=similarity_summary_dic,
        selection_mode=selection_mode,
        alpha=alpha,
        top_k=top_k,
        min_similarity_level=min_similarity_level,
        score_key=selection_score_key,
        similarity_key=similarity_key,
        weighted_similarity_key=weighted_similarity_key,
        sort_by=sort_by,
        print_results=print_results,
    )

    # --------------------------------------------------------
    # Step 6. Store results
    # --------------------------------------------------------
    params = {
        "ref_section_list": ref_section_list,
        "qry_section_list": qry_section_list,
        "low_exp_thres": low_exp_thres,
        "normalize": normalize,
        "normalization_method": normalization_method,
        "min_region_prop": min_region_prop,
        "exclude_labels": exclude_labels,
        "exclude_mode": exclude_mode,
        "min_fold_change": min_fold_change,
        "min_in_out_group_ratio": min_in_out_group_ratio,
        "min_in_group_fraction": min_in_group_fraction,
        "pvals_adj": pvals_adj,
        "gene_num": gene_num,
        "weight_key": weight_key,
        "similarity_key": similarity_key,
        "summary_key": summary_key,
        "similarity_method": similarity_method,
        "sort_by": sort_by,
        "selection_mode": selection_mode,
        "alpha": alpha,
        "top_k": top_k,
        "min_similarity_level": min_similarity_level,
        "selection_score_key": selection_score_key,
        "preprocess_ref": preprocess_ref,
        "preprocess_qry": preprocess_qry,
    }

    if selection_score_key is None:
        if sort_by == "weighted":
            final_selection_metric = weighted_similarity_key
        elif sort_by == "similarity":
            final_selection_metric = similarity_key
    else:
        final_selection_metric = selection_score_key    
    
    selection_cutoffs = {
        "alpha": alpha,
        "top_k": top_k,
        "min_similarity_level": min_similarity_level,
    }    
    
    result = ReferenceSelectionResult(
        selected_refs_dic=selected_refs_dic,
        similarity_summary_dic=similarity_summary_dic,
        selection_summary_dic=selection_summary_dic,
        ranked_refs_dic=ranked_refs_dic,
        weights=weights,
        region_dic=region_dic,
        selection_metric=final_selection_metric,
        selection_mode=selection_mode,
        selection_cutoffs=selection_cutoffs,
        params=params,
        d_s_All=d_s_All,
        gene_list_All=gene_list_All,
        d_g_All=d_g_All,
        ref_adata_dic=ref_adata_dic,
        qry_adata_dic=qry_adata_dic,
    )

    return result




