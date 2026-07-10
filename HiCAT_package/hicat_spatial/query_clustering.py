"""
Query multi-modal clustering utilities for HiCAT / label-transfer workflows.

This module performs query-level clustering after modality selection. For each
selected modality, it reduces features by either PCA or selected hierarchical
features, concatenates the reduced modality embeddings, and clusters query spots
using KMeans or Leiden.

Expected inputs
---------------
query_adata_dic
    Dictionary of modality-specific AnnData objects. Keys must use exact modality
    names from {'Gene', 'Image', 'Protein'}.

clustering_config
    Dictionary controlling feature reduction and clustering. Required keys:

    selected_modalities : list, tuple, or set
        Modalities used for clustering. Accepted values are 'Gene', 'Image', and
        'Protein'. Example: {'Gene', 'Image', 'Protein'}.

    reduce_dimension_approach : {'pca', 'selected_features'}
        If 'PCA', each modality is reduced by PCA.
        If 'selected_features', selected features are extracted directly from X.

    clustering_method : {'kmeans', 'leiden'}
        If 'kmeans', n_clusters must be provided.
        If 'leiden', resolution and n_neighbors can be provided.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .label_assignment import refine_labels
from .multi_modal_integration import (
    cluster_integrated_embedding,
    integrate_modalities_for_adata_dic,
    )

SUPPORTED_MODALITIES = ("Gene", "Image", "Protein")
SUPPORTED_REDUCTION_APPROACHES = ("pca", "selected_features")
SUPPORTED_CLUSTERING_METHODS = ("kmeans", "leiden")


def _dense_string_labels(labels: Sequence[Any]) -> pd.Series:
    """Return labels as dense string codes ordered by descending cluster size."""
    labels = pd.Series(labels, dtype="object").astype(str)
    counts = labels.value_counts()
    mapping = {old: str(i) for i, old in enumerate(counts.index)}
    return labels.map(mapping).astype(str)


def _merge_excess_clusters_by_centroid(
    labels: pd.Series,
    embedding: np.ndarray,
    max_clusters: int,
    min_cluster_spots: int = 1,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Merge excess or tiny clusters into nearest retained cluster centroids."""
    counts = labels.value_counts()
    if counts.empty:
        return labels, {"merge_map": {}}

    max_clusters = max(1, min(int(max_clusters), int(counts.shape[0])))
    min_cluster_spots = max(1, int(min_cluster_spots))

    large_labels = counts[counts >= min_cluster_spots].index.tolist()
    if len(large_labels) >= 2:
        keep_labels = large_labels[:max_clusters]
    else:
        keep_labels = counts.index[:max_clusters].tolist()

    if len(keep_labels) == 0:
        keep_labels = [counts.index[0]]

    merge_labels = [label for label in counts.index if label not in keep_labels]
    if len(merge_labels) == 0:
        return labels.copy(), {"merge_map": {}}

    embedding = np.asarray(embedding)
    centroid_by_label = {
        label: embedding[labels.to_numpy() == label].mean(axis=0)
        for label in counts.index
    }
    kept_centroids = np.vstack([centroid_by_label[label] for label in keep_labels])
    kept_labels_array = np.asarray(keep_labels, dtype=object)

    merge_map: Dict[str, str] = {}
    for label in merge_labels:
        centroid = centroid_by_label[label][None, :]
        distances = np.linalg.norm(kept_centroids - centroid, axis=1)
        nearest_idx = int(np.argmin(distances))
        merge_map[str(label)] = str(kept_labels_array[nearest_idx])

    merged = labels.copy()
    merged = merged.replace(merge_map)
    return merged.astype(str), {"merge_map": merge_map}


def apply_cluster_control(
    labels: pd.Series,
    embedding: np.ndarray,
    config: Optional[Mapping[str, Any]],
    print_results: bool = True,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Optionally control excessive query clusters after initial clustering.

    This is intended mainly for HIPT/Image-only Leiden clustering in newer
    Scanpy/Leiden environments where the neighborhood graph can fragment into
    many connected components. It does not change the embedding; it only
    replaces the initial cluster labels with a controlled label set.
    """
    if config is None:
        return labels, {"enabled": False, "performed": False}

    control = dict(config)
    if "enabled" not in control and "enables" in control:
        control["enabled"] = control["enables"]
    control.pop("enables", None)

    if not bool(control.get("enabled", True)):
        return labels, {"enabled": False, "performed": False}

    if not isinstance(labels, pd.Series):
        labels = pd.Series(labels)
    labels = labels.astype(str).copy()
    initial_counts = labels.value_counts()
    initial_n_clusters = int(initial_counts.shape[0])

    max_clusters = control.get("max_clusters", None)
    if max_clusters is None:
        return labels, {
            "enabled": True,
            "performed": False,
            "reason": "max_clusters_not_set",
            "initial_n_clusters": initial_n_clusters,
        }

    max_clusters = int(max_clusters)
    if max_clusters < 2:
        raise ValueError("cluster_control['max_clusters'] must be >= 2.")
    max_clusters = min(max_clusters, len(labels))

    min_cluster_spots = int(control.get("min_cluster_spots", 1))
    tiny_clusters = initial_counts[initial_counts < max(1, min_cluster_spots)]
    needs_control = (
        initial_n_clusters > max_clusters
        or int(tiny_clusters.shape[0]) > 0
    )

    if not needs_control:
        return labels, {
            "enabled": True,
            "performed": False,
            "reason": "within_limits",
            "initial_n_clusters": initial_n_clusters,
            "max_clusters": max_clusters,
            "min_cluster_spots": max(1, min_cluster_spots),
        }

    method = str(control.get("method", control.get("mode", "merge"))).lower()
    random_state = int(control.get("random_state", 0))

    if method in {"kmeans", "kmeans_fallback"}:
        kmeans = KMeans(
            n_clusters=max_clusters,
            random_state=random_state,
            n_init=int(control.get("n_init", 10)),
        )
        controlled = pd.Series(
            kmeans.fit_predict(np.asarray(embedding)).astype(str),
            index=labels.index,
            name=labels.name,
        )
        merge_info: Dict[str, Any] = {"merge_map": {}}
    elif method in {"merge", "centroid", "centroid_merge"}:
        controlled, merge_info = _merge_excess_clusters_by_centroid(
            labels=labels,
            embedding=np.asarray(embedding),
            max_clusters=max_clusters,
            min_cluster_spots=min_cluster_spots,
        )
        controlled.index = labels.index
        controlled.name = labels.name
    else:
        raise ValueError(
            "cluster_control['method'] must be 'merge' or 'kmeans_fallback'."
        )

    if bool(control.get("relabel_dense", True)):
        controlled = _dense_string_labels(controlled.to_numpy())
        controlled.index = labels.index
        controlled.name = labels.name

    final_counts = controlled.value_counts()
    info = {
        "enabled": True,
        "performed": True,
        "method": method,
        "initial_n_clusters": initial_n_clusters,
        "final_n_clusters": int(final_counts.shape[0]),
        "max_clusters": max_clusters,
        "min_cluster_spots": max(1, min_cluster_spots),
        "initial_cluster_sizes": initial_counts.astype(int).to_dict(),
        "final_cluster_sizes": final_counts.astype(int).to_dict(),
        **merge_info,
    }

    if print_results:
        print("========== Cluster control results ==========")
        print(
            f"Controlled clusters from {initial_n_clusters} to "
            f"{info['final_n_clusters']} (max={max_clusters}, method={method})."
        )
        print(final_counts)

    return controlled.astype(str), info

#===============================================================================================
# Part 3. Clustering based on informative modalities and determined dimension reduction approach
#===============================================================================================
@dataclass
class QueryClusteringResult:
    """
    Container for query multi-modal clustering results.

    This dataclass stores index-aligned labels, the integrated embedding, 
    modality-specific embeddings, and the clustering configuration. It does not
    own a complete AnnData object; use ``apply_to`` when labels need to be
    attached to one

    Attributes
    ----------
    labels : pandas.Series
        Cluster labels indexed by the exact query ``obs_names`` used during
        clustering.

    pred_df : pandas.DataFrame
        DataFrame containing spot/cell-level clustering predictions.

        The index matches ``labels.index``. It usually contains at least one
        column storing the predicted cluster labels, such as `"query_cluster"`.

    cluster_labels : numpy.ndarray
        One-dimensional array of predicted cluster labels.

        The length equals the number of query spots/cells after modality
        alignment. Labels are generated by the specified clustering method,
        such as KMeans or Leiden.

    integrated_embedding : numpy.ndarray
        Concatenated multi-modal feature matrix used for clustering.

        Each selected modality is first reduced using either PCA or selected
        hierarchical features. The reduced modality-specific embeddings are then
        concatenated column-wise.

        Shape
        -----
        `(n_obs, n_integrated_features)`

        where `n_integrated_features` is the sum of reduced feature dimensions
        across selected modalities.

    modality_embedding_dic : dict
        Dictionary storing reduced embeddings for each selected modality.

        Keys are modality names, such as `"Gene"`, `"Image"`, and `"Protein"`.
        Values are NumPy arrays containing the reduced embedding for that
        modality.

        Example
        -------
        {
            "Gene": gene_embedding,
            "Image": image_embedding,
            "Protein": protein_embedding,
        }

        Each embedding has shape:

        `(n_obs, n_reduced_features_for_that_modality)`

    selected_modalities : list of str
        List of informative modalities used for clustering.

        Example
        -------
        `["Gene", "Image", "Protein"]`

        These modalities are selected from `query_adata_dic` according to
        `clustering_config["selected_modalities"]`.

    config : dict
        Clustering configuration used to generate the result.

        This stores the user-provided or internally standardized clustering
        settings, including selected modalities, dimension reduction approach,
        PCA dimensions, selected features, clustering method, and clustering
        parameters.

        Example
        -------
        {
            "selected_modalities": ["Gene", "Image"],
            "dim_reduction_method": "pca",
            "pcs_num_dic": {
                "Gene": 30,
                "Image": 20,
            },
            "clustering_method": "leiden",
            "resolution": 0.5,
            "n_neighbors": 15,
            "random_state": 0,
        }
    """

    labels: pd.Series
    integrated_embedding: np.ndarray
    modality_embedding_dic: Dict[str, np.ndarray]
    selected_modalities: List[str]
    config: Dict[str, Any]

    def __post_init__(self) -> None:
        """Normalize labels to an index-aligned string Series."""
        if not isinstance(self.labels, pd.Series):
            raise TypeError("labels must be a pandas Series indexed by obs_names.")
        if not self.labels.index.is_unique:
            raise ValueError("labels.index must contain unique obs_names.")

        self.labels = self.labels.astype(str).copy()
        self.labels.name = self.config.get("final_pred_key", self.config.get("pred_key", self.labels.name))

        if self.integrated_embedding.shape[0] != len(self.labels):
            raise ValueError(
                "integrated_embedding and labels must contain the same number "
                "of observations."
            )

    @property
    def cluster_labels(self) -> np.ndarray:
        """Backward-compatible NumPy representation of ``labels``."""
        return self.labels.to_numpy(copy=True)

    @property
    def obs_names(self) -> pd.Index:
        """Observation names in the exact order used for clustering."""
        return self.labels.index

    @property
    def pred_df(self) -> pd.DataFrame:
        """Return an index-aligned prediction table."""
        pred_key = self.labels.name or self.config.get("pred_key", "query_cluster")
        return pd.DataFrame(
            {
                "spot_id": self.labels.index,
                pred_key: self.labels.to_numpy(),
            },
            index=self.labels.index,
        )

    def apply_to(self, adata: Any, key: Optional[str] = None, copy: bool = True) -> Any:
        """Attach labels to a compatible AnnData object by ``obs_names``."""
        output = adata.copy() if copy or adata.is_view else adata
        missing = self.labels.index.difference(output.obs_names)
        if len(missing) > 0:
            raise ValueError(
                "AnnData is missing clustered observations. "
                f"Examples: {missing[:5].tolist()}"
            )

        output_key = key or self.labels.name or self.config.get("pred_key", "query_cluster")
        output.obs.loc[self.labels.index, output_key] = self.labels
        output.obs[output_key] = output.obs[output_key].astype("category")
        return output


def _normalize_selected_modalities(selected_modalities: Sequence[str]) -> List[str]:
    selected_modalities = list(selected_modalities)

    invalid_modalities = [
        modality for modality in selected_modalities
        if modality not in SUPPORTED_MODALITIES
    ]
    if invalid_modalities:
        raise ValueError(
            f"Unsupported modalities: {invalid_modalities}. "
            f"Supported modalities are {SUPPORTED_MODALITIES}."
        )

    return [modality for modality in SUPPORTED_MODALITIES if modality in selected_modalities]


def compute_query_integrated_embedding(
    query_adata_dic: Mapping[str, Any],
    clustering_config: Mapping[str, Any],
    align_by_obs_names: bool = True,
    query_section: Optional[str] = None,
) -> Tuple[Any, np.ndarray, Dict[str, np.ndarray], Sequence[str], Dict[str, Any]]:

    """
    Compute integrated multi-modal embedding for one query section.

    Query-specific input requirement
    --------------------------------
    ``query_adata_dic`` must use modality-level format only::

        query_adata_dic = {
            "Gene": query_gene_adata,
            "Image": query_image_adata,
            "Protein": query_protein_adata,
        }

    When selected features are used, ``features_dic`` must also use
    modality-level format::

        features_dic = {
            "Gene": gene_feature_list,
            "Image": image_feature_list,
            "Protein": protein_feature_list,
        }

    Parameters
    ----------
    query_adata_dic : dict
        Dictionary of modality-specific query AnnData objects. Keys must be
        exact modality names from ``{'Gene', 'Image', 'Protein'}``.

    clustering_config : dict
        Configuration dictionary.

        Required keys:

        ``selected_modalities`` : list, tuple, or set
            Modalities used for clustering. Accepted values are ``'Gene'``,
            ``'Image'``, and ``'Protein'``.

        ``dim_reduction_approach`` : {'pca', 'selected_features'}
            Dimension reduction approach.

        Required when ``dim_reduction_approach='pca'``:

        ``pcs_num_dic`` : dict, optional
            Number of PCs for each modality, such as
            ``{'Gene': 30, 'Image': 20, 'Protein': 10}``. If omitted,
            ``default_pcs_num`` is used.

        Required when ``dim_reduction_approach='selected_features'``:

        ``features_dic`` : dict
            Selected hierarchical feature dictionary.

        Optional keys:

        ``default_pcs_num`` : int, default=30
            Default number of PCs for modalities not included in ``pcs_num_dic``.

        ``feature_section`` : str or None, default=None
            Reference section used for section-specific selected features.

        ``combine_feature_sections`` : bool, default=True
            Whether to union selected features across sections when
            ``feature_section`` is None.

        ``scale_embedding`` : bool, default=True
            Whether to standardize each modality embedding before concatenation.

        ``random_state`` : int, default=0
            Random seed.

    align_by_obs_names : bool, default=True
        Whether to align modalities by shared ``obs_names`` before integration.

    qry_section: str, default=None
        the name of the target query section.

    Returns
    -------
    base_adata : AnnData
        Copy of the first selected modality AnnData after alignment.

    integrated_embedding : numpy.ndarray
        Concatenated multi-modal embedding.

    modality_embedding_dic : dict
        Reduced embedding for each selected modality.

    selected_modalities : list of str
        Ordered selected modalities.

    normalized_config : dict
        Configuration after filling defaults and recording resolved settings.
    """

    selected_modalities = clustering_config.get(
        "selected_modalities",
        clustering_config.get("informative_modalities", None),
    )

    if selected_modalities is None:
        raise KeyError(
            "clustering_config must contain 'selected_modalities' "
            "or 'informative_modalities'."
        )

    selected_modalities = _normalize_selected_modalities(selected_modalities)

    missing_modalities = [
        modality for modality in selected_modalities
        if modality not in query_adata_dic
    ]
    if missing_modalities:
        raise KeyError(
            f"query_adata_dic is missing selected modalities: {missing_modalities}"
        )

    dim_reduction_method = clustering_config.get(
        "dim_reduction_method",
        clustering_config.get("reduce_dimension_approach", "pca"),
    )
    dim_reduction_method = str(dim_reduction_method).lower().strip()

    if dim_reduction_method not in SUPPORTED_REDUCTION_APPROACHES:
        raise ValueError(
            f"dim_reduction_method must be one of {SUPPORTED_REDUCTION_APPROACHES}, "
            f"got {dim_reduction_method!r}."
        )

    features_dic = clustering_config.get("features_dic", None)
    if dim_reduction_method == "selected_features" and features_dic is None:
        raise KeyError(
            "features_dic is required when dim_reduction_method='selected_features'."
        )

    pcs_num_dic = clustering_config.get("pcs_num_dic", None)
    default_pcs_num = int(clustering_config.get("default_pcs_num", 30))
    scale_embedding = bool(clustering_config.get("scale_embedding", True))
    random_state = int(clustering_config.get("random_state", 0))

    base_adata, integrated_embedding, modality_embedding_dic, _ = integrate_modalities_for_adata_dic(
        modality_adata_dic=query_adata_dic,
        selected_modalities=selected_modalities,
        dim_reduction_method=dim_reduction_method,
        features_dic=features_dic,
        features_format="modality",
        feature_section=None,
        pcs_num_dic=pcs_num_dic,
        default_pcs_num=default_pcs_num,
        scale_embedding=scale_embedding,
        random_state=random_state,
        align_by_obs_names=align_by_obs_names,
        sample_name=query_section,
    )

    final_config = dict(clustering_config)
    final_config.update(
        {
            "selected_modalities": selected_modalities,
            "dim_reduction_method": dim_reduction_method,
            "features_format": "modality",
            "default_pcs_num": default_pcs_num,
            "scale_embedding": scale_embedding,
            "random_state": random_state,
            "align_by_obs_names": align_by_obs_names,
        }
    )

    return base_adata, integrated_embedding, modality_embedding_dic, selected_modalities, final_config


def query_multi_modal_clustering(
    query_adata_dic: Mapping[str, Any],
    clustering_config: Mapping[str, Any],
    pred_key: str = "query_cluster",
    query_section: Optional[str] = None,
    align_by_obs_names: bool = True,
    print_results: bool = True,
) -> QueryClusteringResult:
    """
    Run multi-modal clustering for one query section.

    This function performs query-level unsupervised clustering using selected
    modalities. For each selected modality, it computes a reduced embedding
    using either pca or selected features, concatenates the modality-specific
    embeddings, clusters the integrated embedding by KMeans or Leiden, and
    stores the predicted cluster labels in the returned AnnData object.

    This function is query-specific. The shared steps, including modality
    alignment, modality embedding, multi-modal integration, and clustering,
    are handled by the shared helper functions:

        - compute_query_integrated_embedding
        - integrate_modalities_for_adata_dic
        - compute_modality_embedding
        - cluster_integrated_embedding

    Parameters
    ----------
    query_adata_dic : Mapping[str, AnnData]
        Dictionary containing modality-specific AnnData objects for one query
        section.

        Required format::

            {
                "Gene": query_gene_adata,
                "Image": query_image_adata,
                "Protein": query_protein_adata,
            }

        Only modalities listed in ``clustering_config["selected_modalities"]``
        are used. Therefore, this dictionary only needs to contain the selected
        modalities.

        Requirements:

        - Keys must be exact modality names: ``"Gene"``, ``"Image"``,
          and/or ``"Protein"``.
        - Values must be AnnData objects.
        - If ``align_by_obs_names=False``, all selected AnnData objects must
          already have the same ``obs_names`` in the same order.
        - If ``align_by_obs_names=True``, selected modalities are aligned by
          shared ``obs_names`` before integration.
        - For ``dim_reduction_method="selected_features"``, the selected
          features for each modality must be present in the corresponding
          AnnData ``var_names``.

    clustering_config : Mapping[str, Any]
        Configuration dictionary controlling modality selection, dimension
        reduction, and clustering.

        Required keys:

        ``selected_modalities`` : list of str
            Modalities used for query clustering.

            Example::

                ["Gene", "Image", "Protein"]

        ``dim_reduction_method`` : {"pca", "selected_features"}
            Method used to obtain modality-specific embeddings.

            - ``"pca"``: compute pca embedding from each modality's ``.X``.
            - ``"selected_features"``: directly extract selected features
              from each modality's ``.X``.

        ``clustering_method`` : {"kmeans", "leiden"}
            Clustering method applied to the integrated multi-modal embedding.

        Required when ``dim_reduction_method="selected_features"``:

        ``features_dic`` : dict
            Modality-level selected feature dictionary.

            For query clustering, this must use modality-level format only::

                {
                    "Gene": gene_feature_list,
                    "Image": image_feature_list,
                    "Protein": protein_feature_list,
                }

            Section-level feature dictionaries are not used for query clustering.

        Required when ``clustering_method="kmeans"``:

        ``n_clusters`` : int
            Number of KMeans clusters. Must be at least 2 and cannot exceed the
            number of query observations.

        Optional keys:

        ``pcs_num_dic`` : dict, optional
            Modality-specific number of PCs when
            ``dim_reduction_method="pca"``.

            Example::

                {
                    "Gene": 30,
                    "Image": 20,
                    "Protein": 10,
                }

        ``default_pcs_num`` : int, default=30
            Default number of PCs for modalities not listed in ``pcs_num_dic``.

        ``scale_embedding`` : bool, default=True
            Whether to standardize each modality-specific embedding before
            concatenating embeddings across modalities.

        ``resolution`` : float, default=0.5
            Leiden resolution. Used only when
            ``clustering_method="leiden"``.

        ``n_neighbors`` : int, default=15
            Number of neighbors used to build the Leiden neighborhood graph.
            Internally adjusted to at most ``n_obs - 1``.

        ``neighbors_method`` : str or None, default="umap"
            Neighbor graph backend forwarded to ``scanpy.pp.neighbors`` when
            supported.

        ``neighbors_metric`` : str, default="euclidean"
            Distance metric forwarded to ``scanpy.pp.neighbors``.

        ``leiden_flavor`` : {"leidenalg", "igraph"} or None, default="leidenalg"
            Leiden backend forwarded to ``scanpy.tl.leiden`` when supported.
            The default makes the historical Scanpy 1.9-style backend explicit.

        ``leiden_directed`` : bool or None, default=None
            Optional ``directed`` argument for ``scanpy.tl.leiden``. Use
            ``False`` when intentionally using ``leiden_flavor="igraph"``.

        ``leiden_n_iterations`` : int or None, default=None
            Optional number of Leiden iterations.

        ``cluster_control`` : dict or None, default=None
            Optional post-clustering control for excessive Leiden clusters.
            Useful when an Image/HIPT neighbor graph fragments into many
            connected components. Lower-level query clustering requires
            ``cluster_control["max_clusters"]`` to be an integer. Stage 6 also
            supports hierarchy-aware string modes such as ``"auto"`` and
            ``"legacy"``.

            Recommended keys are:

            - ``enabled``: bool, default=True when provided.
            - ``max_clusters``: maximum final cluster count.
            - ``method``: ``"merge"`` or ``"kmeans_fallback"``, default
              ``"merge"``.
            - ``min_cluster_spots``: merge clusters smaller than this size.

            Advanced keys such as ``relabel_dense``, ``random_state``, and
            ``n_init`` are available but usually do not need to be specified.

        ``random_state`` : int, default=0
            Random seed used for pca, KMeans, and Leiden.

    pred_key : str, default="query_cluster"
        Name assigned to the returned index-aligned label Series.

    query_section : str or None, default=None
        Optional query section name.

        This is used only for clearer printed messages and is stored in
        ``result.config["query_section"]``. It does not affect clustering.

    align_by_obs_names : bool, default=True
        Whether to align modality-specific AnnData objects by shared
        ``obs_names`` before integration.

        - If True, the intersection of ``obs_names`` across selected modalities
          is used, preserving the order of the first selected modality.
        - If False, all selected modalities must already have identical
          ``obs_names`` in the same order.

    print_results : bool, default=True
        Whether to print a summary of selected modalities, dimension reduction
        method, clustering method, integrated embedding shape, and cluster sizes.

    Returns
    -------
    QueryClusteringResult
        Dataclass containing query clustering results.

        Fields:
        
        labels : pandas.Series
            Predicted labels indexed by aligned query obs_names.

        ``pred_df`` : pandas.DataFrame
            Spot-level prediction table.

            Index:
                Same as ``adata.obs_names``.

            Columns:
                - ``"spot_id"``: spot/cell IDs from ``adata.obs_names``.
                - ``pred_key``: predicted query cluster labels as strings.

        ``cluster_labels`` : numpy.ndarray
            One-dimensional array of predicted cluster labels. The length equals
            the number of aligned query observations.

        ``integrated_embedding`` : numpy.ndarray
            Concatenated multi-modal embedding used for clustering.

            Shape:
                ``(n_obs, n_integrated_features)``

        ``modality_embedding_dic`` : dict
            Dictionary storing reduced embeddings for each selected modality.

            Example::

                {
                    "Gene": gene_embedding,
                    "Image": image_embedding,
                    "Protein": protein_embedding,
                }

        ``selected_modalities`` : list of str
            Ordered list of modalities used for clustering.

        ``config`` : dict
            Final clustering configuration, including user-provided settings,
            clustering metadata, ``pred_key``, ``query_section``, ``n_obs``, and
            ``n_integrated_features``.
    """

    (
        base_adata,
        integrated_embedding,
        modality_embedding_dic,
        selected_modalities,
        final_config,
    ) = compute_query_integrated_embedding(
        query_adata_dic=query_adata_dic,
        clustering_config=clustering_config,
        align_by_obs_names=align_by_obs_names,
        query_section=query_section,
    )

    cluster_labels, cluster_info = cluster_integrated_embedding(
        integrated_embedding=integrated_embedding,
        clustering_config=final_config,
        cluster_key=pred_key,
    )

    labels = pd.Series(
        cluster_labels.astype(str),
        index=base_adata.obs_names.copy(),
        name=pred_key,
    )

    cluster_control_config = final_config.get(
        "cluster_control",
        final_config.get(
            "cluster_control_config",
            final_config.get("hipt_cluster_control_config", None),
        ),
    )
    labels, cluster_control_info = apply_cluster_control(
        labels=labels,
        embedding=integrated_embedding,
        config=cluster_control_config,
        print_results=print_results,
    )

    final_config.update(cluster_info)
    final_config.update(
        {
            "pred_key": pred_key,
            "query_section": query_section,
            "n_obs": int(base_adata.n_obs),
            "n_integrated_features": int(integrated_embedding.shape[1]),
            "cluster_control": cluster_control_info,
        }
    )

    if print_results:
        section_msg = f" for {query_section}" if query_section is not None else ""
        print(f"========== Query clustering results{section_msg} ==========")
        print(f"Selected modalities: {selected_modalities}")
        print(f"Reduction: {final_config['dim_reduction_method']}")
        print(f"Clustering: {cluster_info['clustering_method']}")
        print(f"Integrated embedding shape: {integrated_embedding.shape}")
        print(labels.value_counts())

    return QueryClusteringResult(
        labels=labels,
        integrated_embedding=integrated_embedding,
        modality_embedding_dic=modality_embedding_dic,
        selected_modalities=selected_modalities,
        config=final_config,
    )


def query_multi_modal_clustering_for_sections(
    query_section_adata_dic: Mapping[str, Mapping[str, Any]],
    clustering_config: Mapping[str, Any],
    pred_key: str = "query_cluster",
    align_by_obs_names: bool = True,
    print_results: bool = True,
) -> Dict[str, QueryClusteringResult]:
    """
    Run query multi-modal clustering for multiple query sections.

    ``query_section_adata_dic`` must be::

        {
            "query_section1": {
                "Gene": gene_adata,
                "Image": image_adata,
                "Protein": protein_adata,
            },
            "query_section2": {
                "Gene": gene_adata,
                "Image": image_adata,
                "Protein": protein_adata,
            },
        }
    """

    result_dic = {}

    for query_section, query_adata_dic in query_section_adata_dic.items():
        result_dic[query_section] = query_multi_modal_clustering(
            query_adata_dic=query_adata_dic,
            clustering_config=clustering_config,
            pred_key=pred_key,
            query_section=query_section,
            align_by_obs_names=align_by_obs_names,
            print_results=print_results,
        )

    return result_dic


#==============================================================================================
# HIPT boundary-cluster detection and reassignment
#==============================================================================================
_DEFAULT_HIPT_BOUNDARY_FEATURE_SETS = (
    ("rgb_0", "rgb_1", "rgb_2"),
    ("hipt_576", "hipt_577", "hipt_578"),
)


def _resolve_boundary_feature_names(
    image_adata: Any,
    requested_features: Sequence[str],
) -> Tuple[List[str], List[str]]:
    """Resolve feature names against var_names and optional source_name aliases."""
    if isinstance(requested_features, str):
        raise TypeError("Boundary features must be a sequence of names, not a string.")

    var_names = [str(feature) for feature in image_adata.var_names]
    var_name_set = set(var_names)
    source_to_var: Dict[str, Optional[str]] = {}

    if "source_name" in image_adata.var.columns:
        for var_name, source_name in zip(
            var_names,
            image_adata.var["source_name"],
        ):
            if pd.isna(source_name):
                continue
            source_name = str(source_name)
            if source_name in source_to_var and source_to_var[source_name] != var_name:
                source_to_var[source_name] = None
            else:
                source_to_var[source_name] = var_name

    resolved_features: List[str] = []
    missing_features: List[str] = []
    for feature in requested_features:
        feature = str(feature)
        if feature in var_name_set:
            resolved_features.append(feature)
        elif source_to_var.get(feature) is not None:
            resolved_features.append(source_to_var[feature])
        else:
            missing_features.append(feature)

    return resolved_features, missing_features


def identify_boundary_cluster(
    image_adata: Any,
    cluster_key: str,
    boundary_features: Optional[Sequence[str]] = None,
    candidate_feature_sets: Sequence[
        Sequence[str]
    ] = _DEFAULT_HIPT_BOUNDARY_FEATURE_SETS,
    min_cluster_size: int = 1,
    max_boundary_score_ratio: Optional[float] = None,
    print_results: bool = True,
) -> Tuple[Optional[str], pd.DataFrame]:
    """
    Identify a likely HIPT boundary/background cluster using RGB-like features.

    HIPT boundary artifacts often have near-zero RGB-like feature values.
    This function computes mean RGB-like values for each cluster and selects
    the cluster with the smallest absolute RGB-like feature sum.

    Parameters
    ----------
    image_adata : AnnData
        Image-feature AnnData object. It must contain cluster labels in
        ``image_adata.obs[cluster_key]`` and RGB-like features in
        ``image_adata.var_names``.
    cluster_key : str
        Cluster label column in ``image_adata.obs``.
    boundary_features : sequence of str or None, default=None
        Explicit feature names used to detect the boundary cluster. If None,
        the function tries ``("rgb_0", "rgb_1", "rgb_2")`` first and then
        ``("hipt_576", "hipt_577", "hipt_578")``. Names may match either
        ``image_adata.var_names`` or aliases in
        ``image_adata.var["source_name"]``.
    candidate_feature_sets : sequence of sequence of str
        Candidate RGB-like feature-name sets used when ``boundary_features``
        is not provided.
    min_cluster_size : int, default=1
        Clusters smaller than this are ignored when selecting the boundary
        cluster.
    max_boundary_score_ratio : float or None, default=None
        Optional safeguard. If provided, the candidate boundary cluster must
        have ``boundary_score <= median(other_cluster_scores) * max_boundary_score_ratio``.
        If this condition fails, the function returns ``None`` as the boundary
        cluster.
    print_results : bool, default=True
        Whether to print the cluster-level RGB-like summary.

    Returns
    -------
    boundary_cluster : str or None
        Label of the likely boundary cluster. Returns None if no cluster passes
        the optional safeguard.
    boundary_summary : pandas.DataFrame
        Cluster-level table containing mean RGB-like values, cluster size, and
        boundary score.
    """

    if cluster_key not in image_adata.obs.columns:
        raise KeyError(f"Missing column in image_adata.obs: {cluster_key}")

    if boundary_features is None:
        selected_features = None
        requested_selected_features = None
        for feature_set in candidate_feature_sets:
            requested_features = list(feature_set)
            resolved_features, missing_features = _resolve_boundary_feature_names(
                image_adata=image_adata,
                requested_features=requested_features,
            )
            if len(missing_features) == 0:
                selected_features = resolved_features
                requested_selected_features = requested_features
                break

        if selected_features is None:
            raise ValueError(
                "Cannot find RGB-like boundary features. Provide boundary_features "
                "explicitly or add one expected feature set to image_adata.var_names "
                "or image_adata.var['source_name']."
            )
    else:
        requested_selected_features = list(boundary_features)
        selected_features, missing_features = _resolve_boundary_feature_names(
            image_adata=image_adata,
            requested_features=requested_selected_features,
        )
        if len(missing_features) > 0:
            raise KeyError(
                "boundary_features not found in image_adata.var_names or "
                f"image_adata.var['source_name']: {missing_features}"
            )

    labels = image_adata.obs[cluster_key].astype(str)
    cluster_sizes = labels.value_counts()
    clusters = cluster_sizes.index.astype(str).tolist()

    summary = pd.DataFrame(index=clusters, columns=selected_features, dtype=float)
    summary["cluster_size"] = cluster_sizes.reindex(clusters).astype(int)

    for cluster in clusters:
        cluster_mask = (labels == cluster).to_numpy()
        X_tmp = image_adata[cluster_mask, selected_features].X
        if issparse(X_tmp):
            X_tmp = X_tmp.toarray()
        else:
            X_tmp = np.asarray(X_tmp)
        summary.loc[cluster, selected_features] = np.nanmean(X_tmp.astype(float), axis=0)

    summary["boundary_score"] = summary[selected_features].abs().sum(axis=1)
    summary.attrs["requested_boundary_features"] = requested_selected_features
    summary.attrs["resolved_boundary_features"] = selected_features

    eligible_summary = summary[summary["cluster_size"] >= min_cluster_size]
    if eligible_summary.empty:
        raise ValueError("No cluster has enough spots after applying min_cluster_size.")

    boundary_cluster = str(eligible_summary["boundary_score"].idxmin())

    if max_boundary_score_ratio is not None and eligible_summary.shape[0] > 1:
        boundary_score = float(eligible_summary.loc[boundary_cluster, "boundary_score"])
        other_scores = eligible_summary.drop(index=boundary_cluster)["boundary_score"]
        reference_score = float(other_scores.median())

        if reference_score > 0 and boundary_score > reference_score * float(max_boundary_score_ratio):
            boundary_cluster = None

    if print_results:
        print("========================= mean HIPT/RGB values by cluster =========================")
        print(summary.sort_values("boundary_score"))
        print(f"Boundary cluster: {boundary_cluster}")

    return boundary_cluster, summary


def reassign_boundary_cluster(
    image_adata: Any,
    boundary_cluster: str,
    cluster_key: str,
    x_key: str = "pixel_x",
    y_key: str = "pixel_y",
    refined_key: Optional[str] = None,
    num_nbs: int = 25,
    metric: str = "euclidean",
    weighted_vote: bool = False,
    copy: bool = True,
    print_results: bool = True,
) -> Tuple[Any, str]:
    """
    Reassign boundary-cluster spots to nearby non-boundary clusters.

    This step is different from general label smoothing. It only changes spots
    currently assigned to ``boundary_cluster`` and only uses non-boundary spots
    as candidate neighbors.

    Parameters
    ----------
    image_adata : AnnData
        AnnData object containing spatial coordinates and cluster labels.
    boundary_cluster : str
        Boundary/background cluster label to reassign.
    cluster_key : str
        Existing cluster label column in ``image_adata.obs``.
    x_key, y_key : str, default=("pixel_x", "pixel_y")
        Spatial coordinate columns in ``image_adata.obs``.
    refined_key : str or None, default=None
        Output column name. If None, uses ``f"{cluster_key}_bd_reassigned"``.
    num_nbs : int, default=25
        Number of nearest non-boundary neighbors used to reassign each boundary
        spot.
    metric : str, default="euclidean"
        Distance metric passed to ``NearestNeighbors``.
    weighted_vote : bool, default=False
        If True, use inverse-distance weighted voting instead of simple majority
        voting.
    copy : bool, default=True
        Whether to copy ``image_adata`` before writing the new label column.
    print_results : bool, default=True
        Whether to print refined label counts.

    Returns
    -------
    adata_out : AnnData
        AnnData object with ``adata_out.obs[refined_key]`` added.
    refined_key : str
        Name of the boundary-reassigned label column.
    """

    required_cols = [cluster_key, x_key, y_key]
    missing_cols = [col for col in required_cols if col not in image_adata.obs.columns]
    if len(missing_cols) > 0:
        raise KeyError(f"Missing columns in image_adata.obs: {missing_cols}")

    if num_nbs < 1:
        raise ValueError("num_nbs must be at least 1.")

    adata_out = image_adata.copy() if copy or image_adata.is_view else image_adata
    refined_key = refined_key or f"{cluster_key}_bd_reassigned"

    labels = adata_out.obs[cluster_key].astype(str)
    boundary_cluster = str(boundary_cluster)
    boundary_mask = labels == boundary_cluster
    other_mask = ~boundary_mask

    if boundary_mask.sum() == 0:
        raise ValueError(f"boundary_cluster={boundary_cluster!r} is not present in {cluster_key}.")
    if other_mask.sum() == 0:
        raise ValueError("All spots belong to the boundary cluster; cannot reassign.")

    refined_labels = labels.copy().astype(str)

    boundary_coords = adata_out.obs.loc[boundary_mask, [x_key, y_key]].to_numpy(dtype=float)
    other_coords = adata_out.obs.loc[other_mask, [x_key, y_key]].to_numpy(dtype=float)
    other_labels = labels.loc[other_mask].astype(str).to_numpy()

    k = min(int(num_nbs), other_coords.shape[0])
    nbrs = NearestNeighbors(n_neighbors=k, metric=metric)
    nbrs.fit(other_coords)
    dists, indices = nbrs.kneighbors(boundary_coords)

    reassigned_labels = []
    for nb_idx, nb_dist in zip(indices, dists):
        nb_labels = other_labels[nb_idx].astype(str)

        if weighted_vote:
            weights = 1.0 / (nb_dist.astype(float) + 1e-8)
            score_df = pd.DataFrame({"label": nb_labels, "weight": weights})
            label_scores = score_df.groupby("label")["weight"].sum().sort_values(ascending=False)
            top_score = label_scores.iloc[0]
            tied_labels = set(label_scores[np.isclose(label_scores, top_score)].index.astype(str))
        else:
            label_counts = pd.Series(nb_labels).value_counts()
            top_count = label_counts.iloc[0]
            tied_labels = set(label_counts[label_counts == top_count].index.astype(str))

        # Neighbor labels are already ordered by distance, so this tie-breaks
        # by the nearest neighbor among tied labels.
        selected_label = None
        for label in nb_labels:
            if label in tied_labels:
                selected_label = str(label)
                break
        reassigned_labels.append(selected_label)

    refined_labels.loc[boundary_mask] = reassigned_labels
    adata_out.obs[refined_key] = pd.Categorical(refined_labels.astype(str))

    if print_results:
        print("========================= boundary-reassigned clusters =========================")
        print(adata_out.obs[refined_key].value_counts())

    return adata_out, refined_key


def refine_hipt_boundary_clusters(
    image_adata: Any,
    cluster_key: str,
    x_key: str = "pixel_x",
    y_key: str = "pixel_y",
    boundary_cluster: Optional[str] = None,
    boundary_features: Optional[Sequence[str]] = None,
    candidate_feature_sets: Sequence[
        Sequence[str]
    ] = _DEFAULT_HIPT_BOUNDARY_FEATURE_SETS,
    min_cluster_size: int = 1,
    max_boundary_score_ratio: Optional[float] = None,
    bd_num_nbs: int = 25,
    smooth_after_reassign: bool = True,
    smooth_num_nbs: int = 15,
    refined_boundary_key: Optional[str] = None,
    final_cluster_key: Optional[str] = None,
    metric: str = "euclidean",
    weighted_vote: bool = False,
    copy: bool = True,
    print_results: bool = True,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Full HIPT boundary-refinement workflow.

    Steps
    -----
    1. Identify the likely boundary/background cluster using RGB-like features.
    2. Reassign only boundary-cluster spots to nearby non-boundary clusters.
    3. Optionally apply the existing ``refine_labels`` function to spatially
       smooth all cluster labels after boundary reassignment.

    Parameters
    ----------
    image_adata : AnnData
        Image-feature AnnData object containing image features, coordinates,
        and the cluster column to refine.
    cluster_key : str
        Existing cluster label column in ``image_adata.obs``.
    x_key, y_key : str, default=("pixel_x", "pixel_y")
        Spatial coordinate columns in ``image_adata.obs``.
    boundary_cluster : str or None, default=None
        Boundary cluster label. If None, it is inferred using
        ``identify_boundary_cluster``.
    boundary_features : sequence of str or None, default=None
        RGB-like features used to identify the boundary cluster.
    candidate_feature_sets : sequence of sequence of str
        Candidate feature-name sets passed to ``identify_boundary_cluster``
        when ``boundary_features`` is None.
    min_cluster_size : int, default=1
        Minimum cluster size considered during boundary cluster detection.
    max_boundary_score_ratio : float or None, default=None
        Optional safeguard for automatic boundary-cluster detection.
    bd_num_nbs : int, default=25
        Number of nearest non-boundary neighbors used for boundary reassignment.
    smooth_after_reassign : bool, default=True
        Whether to call ``refine_labels`` after boundary reassignment.
    smooth_num_nbs : int, default=15
        Number of neighbors used by ``refine_labels`` when
        ``smooth_after_reassign=True``.
    refined_boundary_key : str or None, default=None
        Column name for labels after boundary reassignment.
    final_cluster_key : str or None, default=None
        Column name for the final refined labels.
    metric : str, default="euclidean"
        Spatial distance metric.
    weighted_vote : bool, default=False
        Whether boundary reassignment uses inverse-distance weighted voting.
    copy : bool, default=True
        Whether to copy ``image_adata`` before writing new columns.
    print_results : bool, default=True
        Whether to print intermediate summaries.

    Returns
    -------
    adata_out : AnnData
        AnnData object with refined cluster columns added.
    info : dict
        Metadata containing the boundary cluster, boundary summary, added keys,
        and final cluster key.
    """

    adata_out = image_adata.copy() if copy or image_adata.is_view else image_adata

    boundary_summary = None
    if boundary_cluster is None:
        boundary_cluster, boundary_summary = identify_boundary_cluster(
            image_adata=adata_out,
            cluster_key=cluster_key,
            boundary_features=boundary_features,
            candidate_feature_sets=candidate_feature_sets,
            min_cluster_size=min_cluster_size,
            max_boundary_score_ratio=max_boundary_score_ratio,
            print_results=print_results,
        )

    if boundary_cluster is None:
        final_cluster_key = final_cluster_key or f"{cluster_key}_bd_refined"
        adata_out.obs[final_cluster_key] = pd.Categorical(adata_out.obs[cluster_key].astype(str))

        return adata_out, {
            "boundary_cluster": None,
            "boundary_summary": boundary_summary,
            "refined_boundary_key": None,
            "final_cluster_key": final_cluster_key,
            "added_keys": [final_cluster_key],
            "boundary_refinement_performed": False,
        }

    adata_out, refined_boundary_key = reassign_boundary_cluster(
        image_adata=adata_out,
        boundary_cluster=str(boundary_cluster),
        cluster_key=cluster_key,
        x_key=x_key,
        y_key=y_key,
        refined_key=refined_boundary_key or f"{cluster_key}_bd_reassigned",
        num_nbs=bd_num_nbs,
        metric=metric,
        weighted_vote=weighted_vote,
        copy=False,
        print_results=print_results,
    )

    if smooth_after_reassign:
        final_cluster_key = final_cluster_key or f"{cluster_key}_bd_refined"
        adata_out, _ = refine_labels(
            input_adata=adata_out,
            pred_key=refined_boundary_key,
            refined_key=final_cluster_key,
            num_nbs=smooth_num_nbs,
            x_key=x_key,
            y_key=y_key,
            dists_metric=metric,
            copy=False,
        )
        added_keys = [refined_boundary_key, final_cluster_key]

        if print_results:
            print("========================= final boundary-refined clusters =========================")
            print(adata_out.obs[final_cluster_key].value_counts())
    else:
        final_cluster_key = refined_boundary_key
        added_keys = [refined_boundary_key]

    return adata_out, {
        "boundary_cluster": str(boundary_cluster),
        "boundary_summary": boundary_summary,
        "refined_boundary_key": refined_boundary_key,
        "final_cluster_key": final_cluster_key,
        "added_keys": added_keys,
        "boundary_refinement_performed": True,
    }


#==============================================================================================
# Optional gene-feature subtyping within image-derived or boundary-refined clusters
#==============================================================================================
def subtype_clusters_by_gene_features(
    clustered_adata: Any,
    gene_adata: Any,
    cluster_key: str,
    target_genes: Optional[Sequence[str]] = None,
    nontarget_genes: Optional[Sequence[str]] = None,
    subtype_genes: Optional[Sequence[str]] = None,
    subtype_gene_num: int = 10,
    subtype_min_cluster_prop: float = 0.05,
    min_cluster_size: int = 30,
    min_genes: int = 2,
    clustering_method: str = "leiden",
    resolution: float = 0.5,
    n_neighbors: int = 15,
    neighbors_method: Optional[str] = "umap",
    neighbors_metric: Optional[str] = "euclidean",
    leiden_flavor: Optional[str] = "leidenalg",
    leiden_directed: Optional[bool] = None,
    leiden_n_iterations: Optional[int] = None,
    n_clusters: Optional[int] = None,
    max_subtypes: int = 5,
    scale_gene_features: bool = True,
    subtype_key: Optional[str] = None,
    encoded_subtype_key: Optional[str] = None,
    smooth_subtypes: bool = False,
    final_subtype_key: Optional[str] = None,
    subtype_num_nbs: int = 10,
    x_key: str = "pixel_x",
    y_key: str = "pixel_y",
    random_state: int = 0,
    copy: bool = True,
    print_results: bool = True,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Further subtype image-derived clusters using selected gene-expression features.

    Parameters
    ----------
    clustered_adata : AnnData
        AnnData object containing image-derived or boundary-refined cluster
        labels in ``clustered_adata.obs[cluster_key]``.
    gene_adata : AnnData
        Gene-expression AnnData object for the same query section. The function
        uses the shared ``obs_names`` between ``clustered_adata`` and
        ``gene_adata``.
    cluster_key : str
        Existing cluster column used as parent clusters for gene-based subtyping.
    target_genes, nontarget_genes : sequence of str, optional
        Ranked gene lists. When ``subtype_genes`` is None, the function uses
        the first ``subtype_gene_num`` genes from each list.
    subtype_genes : sequence of str, optional
        Explicit genes used for subtyping. If provided, this overrides
        ``target_genes`` and ``nontarget_genes``.
    subtype_gene_num : int, default=10
        Number of genes taken from each of ``target_genes`` and
        ``nontarget_genes`` when ``subtype_genes`` is not provided.
    subtype_min_cluster_prop : float, default=0.05
        Only parent clusters with at least this fraction of all spots are
        eligible for subtyping.
    min_cluster_size : int, default=30
        Minimum number of shared spots required in a parent cluster before
        subtyping.
    min_genes : int, default=2
        Minimum number of available genes required for subtyping.
    clustering_method : {"leiden", "kmeans"}, default="leiden"
        Clustering method used within each eligible parent cluster. This uses
        the existing ``cluster_integrated_embedding`` function in your package.
    resolution : float, default=0.5
        Initial Leiden resolution for gene-based subtyping.
    n_neighbors : int, default=15
        Number of neighbors used for Leiden graph construction. Internally
        adjusted to at most ``n_parent_spots - 1``.
    neighbors_method : str or None, default="umap"
        Neighbor graph backend passed to Scanpy when Leiden subtyping is used.
        Set to ``None`` to use Scanpy's own default.
    neighbors_metric : str or None, default="euclidean"
        Distance metric passed to Scanpy when Leiden subtyping is used. Set to
        ``None`` to use Scanpy's own default.
    leiden_flavor : {"leidenalg", "igraph"} or None, default="leidenalg"
        Leiden backend passed to Scanpy when supported. Set to ``None`` for
        older Scanpy compatibility or to use Scanpy's own default.
    leiden_directed : bool or None, default=None
        Optional ``directed`` setting passed to Scanpy when supported.
    leiden_n_iterations : int or None, default=None
        Optional ``n_iterations`` setting passed to Scanpy when supported.
    n_clusters : int or None, default=None
        Number of KMeans clusters when ``clustering_method="kmeans"``.
    max_subtypes : int, default=5
        Maximum desired number of subtypes per parent cluster. For Leiden, the
        function retries lower resolutions if too many subtypes are produced.
    scale_gene_features : bool, default=True
        Whether to standardize selected gene features before subclustering.
    subtype_key : str or None, default=None
        Output string subtype column. If None, uses
        ``f"{cluster_key}_gene_subtype"``.
    encoded_subtype_key : str or None, default=None
        Integer-coded subtype column. If None, uses ``f"{subtype_key}_code"``.
    smooth_subtypes : bool, default=False
        Whether to apply ``refine_labels`` to subtype labels.
    final_subtype_key : str or None, default=None
        Final subtype column. If smoothing is disabled, this equals
        ``encoded_subtype_key``. If smoothing is enabled, this defaults to
        ``f"{subtype_key}_refined"``.
    subtype_num_nbs : int, default=10
        Number of neighbors used by ``refine_labels`` when
        ``smooth_subtypes=True``.
    x_key, y_key : str, default=("pixel_x", "pixel_y")
        Spatial coordinate columns required only when ``smooth_subtypes=True``.
    random_state : int, default=0
        Random seed passed to ``cluster_integrated_embedding``.
    copy : bool, default=True
        Whether to copy ``clustered_adata`` before writing subtype columns.
    print_results : bool, default=True
        Whether to print subtype summaries.

    Returns
    -------
    adata_out : AnnData
        AnnData object with gene-subtype columns added.
    info : dict
        Metadata containing selected genes, missing genes, subtype keys, and a
        per-parent-cluster subtyping summary.
    """

    if cluster_key not in clustered_adata.obs.columns:
        raise KeyError(f"Missing column in clustered_adata.obs: {cluster_key}")

    if smooth_subtypes:
        missing_cols = [col for col in [x_key, y_key] if col not in clustered_adata.obs.columns]
        if len(missing_cols) > 0:
            raise KeyError(f"Missing columns in clustered_adata.obs: {missing_cols}")

    if subtype_genes is None:
        target_genes = list(target_genes or [])
        nontarget_genes = list(nontarget_genes or [])
        subtype_genes = target_genes[:subtype_gene_num] + nontarget_genes[:subtype_gene_num]
    else:
        subtype_genes = list(subtype_genes)

    subtype_genes = list(dict.fromkeys(subtype_genes))
    gene_var_names = set(gene_adata.var_names.astype(str))
    available_genes = [gene for gene in subtype_genes if gene in gene_var_names]
    missing_genes = [gene for gene in subtype_genes if gene not in gene_var_names]

    if len(available_genes) < min_genes:
        raise ValueError(
            f"Only {len(available_genes)} subtype genes are present in gene_adata.var_names; "
            f"at least {min_genes} are required."
        )

    adata_out = clustered_adata.copy() if copy or clustered_adata.is_view else clustered_adata
    subtype_key = subtype_key or f"{cluster_key}_gene_subtype"
    encoded_subtype_key = encoded_subtype_key or f"{subtype_key}_code"

    gene_obs_set = set(gene_adata.obs_names)
    shared_obs = [obs for obs in adata_out.obs_names if obs in gene_obs_set]
    if len(shared_obs) == 0:
        raise ValueError("clustered_adata and gene_adata do not share obs_names.")

    parent_labels = adata_out.obs[cluster_key].astype(str)
    adata_out.obs[subtype_key] = parent_labels.to_numpy()

    parent_props = parent_labels.value_counts(normalize=True)
    parent_clusters = parent_props.index.astype(str).tolist()
    subtype_summary: Dict[str, Any] = {}

    if print_results:
        print("========================= parent cluster proportions =========================")
        print(parent_props)
        print("Subtype genes used for gene-based subclustering:")
        print(available_genes)

    shared_obs_set = set(shared_obs)

    for parent_cluster in parent_clusters:
        parent_obs = [
            obs
            for obs in adata_out.obs_names
            if obs in shared_obs_set and str(parent_labels.loc[obs]) == parent_cluster
        ]

        parent_prop = float(parent_props.loc[parent_cluster])
        subtype_summary[parent_cluster] = {
            "n_spots": len(parent_obs),
            "parent_prop": parent_prop,
            "subtyped": False,
            "n_subtypes": 1,
        }

        if parent_prop < subtype_min_cluster_prop or len(parent_obs) < min_cluster_size:
            continue

        n_neighbors_use = min(int(n_neighbors), len(parent_obs) - 1)
        if n_neighbors_use < 1:
            continue

        gene_tmp = gene_adata[parent_obs, available_genes].copy()
        X = gene_tmp.X
        if issparse(X):
            X = X.toarray()
        else:
            X = np.asarray(X)
        X = X.astype(float)

        if scale_gene_features:
            X = StandardScaler().fit_transform(X)

        method = clustering_method.lower().strip()

        if method == "kmeans":
            if n_clusters is None:
                n_clusters_use = min(int(max_subtypes), X.shape[0])
            else:
                n_clusters_use = min(int(n_clusters), int(max_subtypes), X.shape[0])

            if n_clusters_use < 2:
                continue

            cluster_config = {
                "clustering_method": "kmeans",
                "n_clusters": n_clusters_use,
                "random_state": int(random_state),
            }
            subtype_pred, _ = cluster_integrated_embedding(
                integrated_embedding=X,
                clustering_config=cluster_config,
                cluster_key="gene_subtype",
            )
            subtype_pred = np.asarray(subtype_pred).astype(str)
        else:
            resolution_candidates = []
            for res in [resolution, resolution / 2, resolution / 4, 0.05, 0.01]:
                res = float(res)
                if res > 0 and res not in resolution_candidates:
                    resolution_candidates.append(res)

            subtype_pred = None
            for res in resolution_candidates:
                cluster_config = {
                    "clustering_method": method,
                    "resolution": float(res),
                    "n_neighbors": int(n_neighbors_use),
                    "neighbors_method": neighbors_method,
                    "neighbors_metric": neighbors_metric,
                    "leiden_flavor": leiden_flavor,
                    "leiden_directed": leiden_directed,
                    "leiden_n_iterations": leiden_n_iterations,
                    "random_state": int(random_state),
                }
                subtype_pred, _ = cluster_integrated_embedding(
                    integrated_embedding=X,
                    clustering_config=cluster_config,
                    cluster_key="gene_subtype",
                )
                subtype_pred = np.asarray(subtype_pred).astype(str)

                if pd.Series(subtype_pred).nunique() <= max_subtypes:
                    break

        n_subtypes = int(pd.Series(subtype_pred).nunique())
        subtype_summary[parent_cluster].update(
            {
                "subtyped": n_subtypes > 1,
                "n_subtypes": n_subtypes,
            }
        )

        if n_subtypes <= 1:
            continue

        adata_out.obs.loc[parent_obs, subtype_key] = [
            f"{parent_cluster}_{sub_label}" for sub_label in subtype_pred.astype(str)
        ]

    adata_out.obs[subtype_key] = pd.Categorical(adata_out.obs[subtype_key].astype(str))
    adata_out.obs[encoded_subtype_key] = adata_out.obs[subtype_key].cat.codes.astype(int)

    if smooth_subtypes:
        final_subtype_key = final_subtype_key or f"{subtype_key}_refined"
        adata_out, _ = refine_labels(
            input_adata=adata_out,
            pred_key=encoded_subtype_key,
            refined_key=final_subtype_key,
            num_nbs=subtype_num_nbs,
            x_key=x_key,
            y_key=y_key,
            dists_metric="euclidean",
            copy=False,
        )
    else:
        final_subtype_key = encoded_subtype_key

    if print_results:
        print("========================= gene-subtyped clusters =========================")
        print(adata_out.obs[final_subtype_key].value_counts())

    return adata_out, {
        "subtype_key": subtype_key,
        "encoded_subtype_key": encoded_subtype_key,
        "final_subtype_key": final_subtype_key,
        "available_genes": available_genes,
        "missing_genes": missing_genes,
        "subtype_summary": subtype_summary,
    }


#==============================================================================================
# Optional postprocessing after query_multi_modal_clustering
#==============================================================================================
def postprocess_query_clustering_result(
    result: Any,
    query_adata_dic: Mapping[str, Any],
    pred_key: str = "query_cluster",
    boundary_refinement_config: Optional[Mapping[str, Any]] = None,
    gene_subtyping_config: Optional[Mapping[str, Any]] = None,
    print_results: bool = True,
) -> Any:
    """
    Apply optional HIPT boundary refinement and optional gene subtyping after
    ``query_multi_modal_clustering`` has already been run.

    This function is intentionally separated from ``query_multi_modal_clustering``.
    The base clustering function remains modality-general, while this function
    handles HIPT-specific boundary cleanup and optional gene-based subtyping.

    Parameters
    ----------
    result : QueryClusteringResult
        Output from ``query_multi_modal_clustering``.
    query_adata_dic : Mapping[str, AnnData]
        Query modality dictionary used for clustering. ``"Image"`` is required
        when ``boundary_refinement_config`` is provided. ``"Gene"`` is required
        when ``gene_subtyping_config`` is provided.
    pred_key : str, default="query_cluster"
        Original cluster column produced by ``query_multi_modal_clustering``.
    boundary_refinement_config : mapping or None, default=None
        Optional configuration passed to ``refine_hipt_boundary_clusters``.
        Set to None to skip HIPT boundary refinement.

        example setting:
        boundary_refinement_config = {
            "enabled": True,
            "x_key": "pixel_x",
            "y_key": "pixel_y",
            "boundary_cluster": None,
            "boundary_features": None,
            "min_cluster_size": 5,
            "max_boundary_score_ratio": 0.2,
            "bd_num_nbs": 25,
            "smooth_after_reassign": True,
            "smooth_num_nbs": 15,
        }

    gene_subtyping_config : mapping or None, default=None
        Optional configuration passed to ``subtype_clusters_by_gene_features``.
        Set to None to skip gene-based subtyping.

        example setting
        gene_subtyping_config = {
            "enabled": True,
            "target_genes": target_genes,
            "nontarget_genes": nontgt_genes,
            "subtype_genes": None,
            "subtype_gene_num": 10,
            "subtype_min_cluster_prop": 0.05,
            "min_cluster_size": 30,
            "min_genes": 2,
            "clustering_method": "leiden",
            "resolution": 0.5,
            "n_neighbors": 15,
            "max_subtypes": 5,
            "scale_gene_features": True,
            "smooth_subtypes": False,
            "x_key": "pixel_x",
            "y_key": "pixel_y",
            "subtype_num_nbs": 10,
            "random_state": 0,
        }

    print_results : bool, default=True
        Whether to print postprocessing summaries.

    Returns
    -------
    QueryClusteringResult
        A new result object of the same class as ``result``. The original
        ``pred_key`` is kept, and the final active prediction column is stored
        in ``result.config["final_pred_key"]``.
    """

    if not isinstance(result, QueryClusteringResult):
        raise TypeError("result must be a QueryClusteringResult.")

    base_modality = "Gene" if "Gene" in query_adata_dic else result.selected_modalities[0]
    if base_modality not in query_adata_dic:
        raise KeyError(
            f"query_adata_dic is missing the base modality {base_modality!r}."
        )

    base_adata = query_adata_dic[base_modality]
    missing_obs = result.obs_names.difference(base_adata.obs_names)
    if len(missing_obs) > 0:
        raise ValueError(
            "The base query AnnData is missing clustered observations. "
            f"Examples: {missing_obs[:5].tolist()}"
        )

    # Postprocessing needs a temporary AnnData container, but it is deliberately
    # local to this function and is not stored in the result object.
    adata_out = base_adata[result.obs_names, :].copy()
    adata_out.obs[pred_key] = pd.Categorical(
        result.labels.reindex(adata_out.obs_names).astype(str)
    )
    active_pred_key = pred_key
    postprocess_info: Dict[str, Any] = {}

    if pred_key not in adata_out.obs.columns:
        raise KeyError(f"Temporary postprocessing data is missing column: {pred_key}")

    if (
        boundary_refinement_config is not None
        and boundary_refinement_config.get(
            "enabled",
            boundary_refinement_config.get("enables", True),
        )
    ):
        if "Image" not in query_adata_dic:
            raise KeyError("boundary_refinement_config requires query_adata_dic['Image'].")

        bd_cfg = dict(boundary_refinement_config)
        bd_cfg.pop("enabled", None)
        bd_cfg.pop("enables", None)
        x_key = bd_cfg.pop("x_key", "pixel_x")
        y_key = bd_cfg.pop("y_key", "pixel_y")

        image_adata = query_adata_dic["Image"]
        image_obs_set = set(image_adata.obs_names)
        missing_obs = [obs for obs in adata_out.obs_names if obs not in image_obs_set]
        if len(missing_obs) > 0:
            raise ValueError(
                "query_adata_dic['Image'] does not contain all clustered observations. "
                f"Missing examples: {missing_obs[:5]}"
            )

        image_tmp = image_adata[list(adata_out.obs_names), :].copy()
        image_tmp.obs[active_pred_key] = adata_out.obs[active_pred_key].astype(str).to_numpy()

        for coord_col in [x_key, y_key]:
            if coord_col not in image_tmp.obs.columns:
                if coord_col in adata_out.obs.columns:
                    image_tmp.obs[coord_col] = adata_out.obs[coord_col].to_numpy()
                else:
                    raise KeyError(
                        f"Missing coordinate column {coord_col!r} in both image AnnData and result AnnData."
                    )

        image_tmp, bd_info = refine_hipt_boundary_clusters(
            image_adata=image_tmp,
            cluster_key=active_pred_key,
            x_key=x_key,
            y_key=y_key,
            print_results=print_results,
            **bd_cfg,
        )

        for key in bd_info["added_keys"]:
            adata_out.obs[key] = pd.Categorical(image_tmp.obs.loc[adata_out.obs_names, key].astype(str))

        active_pred_key = bd_info["final_cluster_key"]
        postprocess_info["boundary_refinement"] = bd_info

    if (
        gene_subtyping_config is not None
        and gene_subtyping_config.get(
            "enabled",
            gene_subtyping_config.get("enables", True),
        )
    ):
        if "Gene" not in query_adata_dic:
            raise KeyError("gene_subtyping_config requires query_adata_dic['Gene'].")

        gene_cfg = dict(gene_subtyping_config)
        gene_cfg.pop("enabled", None)
        gene_cfg.pop("enables", None)
        gene_cfg.setdefault("print_results", print_results)

        adata_out, subtype_info = subtype_clusters_by_gene_features(
            clustered_adata=adata_out,
            gene_adata=query_adata_dic["Gene"],
            cluster_key=active_pred_key,
            copy=False,
            **gene_cfg,
        )

        active_pred_key = subtype_info["final_subtype_key"]
        postprocess_info["gene_subtyping"] = subtype_info

    final_config = dict(getattr(result, "config", {}))
    final_config.update(
        {
            "pred_key": pred_key,
            "final_pred_key": active_pred_key,
            "postprocess_info": postprocess_info,
        }
    )

    labels = adata_out.obs[active_pred_key].astype(str).copy()
    labels.name = active_pred_key

    return QueryClusteringResult(
        labels=labels,
        integrated_embedding=result.integrated_embedding,
        modality_embedding_dic=result.modality_embedding_dic,
        selected_modalities=result.selected_modalities,
        config=final_config,
    )
