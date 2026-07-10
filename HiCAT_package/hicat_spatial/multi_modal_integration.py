from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .utils import (
    compute_modality_embedding,
    get_ref_modality_adata,
    kmeans_clustering,
    leiden_clustering,
    )


SUPPORTED_MODALITIES = ("Gene", "Image", "Protein")
SUPPORTED_REDUCTION_METHODS = ("pca", "selected_features")
SUPPORTED_CLUSTERING_METHODS = ("kmeans", "leiden")

def align_modalities_by_obs_names(
    modality_adata_dic: Mapping[str, Any],
    selected_modalities: Sequence[str],
    align_by_obs_names: bool = True,
    sample_name: Optional[str] = None,
    copy: bool = False,
) -> Dict[str, Any]:
    """
    Align modality-specific AnnData objects by obs_names.

    Shared by both reference and query workflows.

    Parameters
    ----------
    modality_adata_dic : dict
        Dictionary with exact modality keys, for example::

            {
                "Gene": gene_adata,
                "Image": image_adata,
                "Protein": protein_adata,
            }

    selected_modalities : sequence of str
        Ordered selected modalities. Values must be exact modality names.

    align_by_obs_names : bool, default=True
        If True, keep shared obs_names across selected modalities and preserve
        the order of the first selected modality.
        If False, all selected modalities must already have identical obs_names
        in the same order.

    sample_name : str or None, default=None
        Optional section/query name used only for clearer error messages.

    copy : bool, default=False
        Copy aligned AnnData objects. The default returns the original objects
        or AnnData views because integration only reads them.

    Returns
    -------
    aligned_adata_dic : dict
        Dictionary of aligned AnnData objects for selected modalities.
    """

    sample_msg = f" for {sample_name}" if sample_name is not None else ""

    if modality_adata_dic is None or len(modality_adata_dic) == 0:
        raise ValueError(f"modality_adata_dic cannot be None or empty{sample_msg}.")

    if selected_modalities is None or len(selected_modalities) == 0:
        raise ValueError(f"selected_modalities cannot be None or empty{sample_msg}.")

    selected_modalities = list(selected_modalities)

    for modality in selected_modalities:
        if modality not in SUPPORTED_MODALITIES:
            raise ValueError(
                f"Unsupported modality {modality!r}{sample_msg}. "
                f"Expected one of {list(SUPPORTED_MODALITIES)}."
            )

        if modality not in modality_adata_dic:
            raise KeyError(
                f"Selected modality {modality!r} is not found in modality_adata_dic"
                f"{sample_msg}. Available keys: {list(modality_adata_dic.keys())}."
            )

        if not modality_adata_dic[modality].obs_names.is_unique:
            raise ValueError(
                f"{modality}{sample_msg}: obs_names must be unique before "
                "multi-modal alignment."
            )

    first_modality = selected_modalities[0]
    first_obs_names = modality_adata_dic[first_modality].obs_names

    if not align_by_obs_names:
        for modality in selected_modalities[1:]:
            if not modality_adata_dic[modality].obs_names.equals(first_obs_names):
                raise ValueError(
                    f"obs_names are not aligned between {first_modality} and "
                    f"{modality}{sample_msg}. Set align_by_obs_names=True or "
                    "align AnnData objects before integration."
                )

        return {
            modality: (
                modality_adata_dic[modality].copy()
                if copy
                else modality_adata_dic[modality]
            )
            for modality in selected_modalities
        }

    common_obs = first_obs_names
    for modality in selected_modalities[1:]:
        common_obs = common_obs.intersection(modality_adata_dic[modality].obs_names)

    common_obs = first_obs_names[first_obs_names.isin(common_obs)]

    if len(common_obs) == 0:
        raise ValueError(f"No shared obs_names were found across modalities{sample_msg}.")

    n_obs_before = {m: modality_adata_dic[m].n_obs for m in selected_modalities}

    aligned_adata_dic = {}
    for modality in selected_modalities:
        aligned = modality_adata_dic[modality][common_obs, :]
        aligned_adata_dic[modality] = aligned.copy() if copy else aligned

    n_obs_after = len(common_obs)
    if any(n != n_obs_after for n in n_obs_before.values()):
        print(
            f"Aligned modalities{sample_msg} by shared obs_names: "
            f"kept {n_obs_after} spots/cells from original sizes {n_obs_before}."
        )

    return aligned_adata_dic


def integrate_modalities_for_adata_dic(
    modality_adata_dic: Mapping[str, Any],
    selected_modalities: Sequence[str],
    dim_reduction_method: str = "pca",
    features_dic: Optional[Mapping[str, Any]] = None,
    features_format: str = "modality",
    feature_section: Optional[str] = None,
    pcs_num_dic: Optional[Mapping[str, int]] = None,
    default_pcs_num: int = 30,
    scale_embedding: bool = True,
    random_state: int = 0,
    align_by_obs_names: bool = True,
    sample_name: Optional[str] = None,
    copy: bool = False,
) -> Tuple[Any, np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Reduce and concatenate modality embeddings for one sample/section.

    This is the central shared integration function. It is used by:

    1. reference-section workflows, after constructing ``modality_adata_dic``
       from ``ref_gene_dic/ref_image_dic/ref_protein_dic``;
    2. query-section workflows, where ``query_adata_dic`` already has modality
       keys.

    Parameters
    ----------
    modality_adata_dic : dict
        Exact modality-keyed AnnData dictionary.

    selected_modalities : sequence of str
        Exact modality names to use.

    dim_reduction_method : {"pca", "selected_features"}, default="pca"
        Dimension-reduction method applied to every selected modality.

    features_dic : dict or None
        Required only when ``dim_reduction_method="selected_features"``.

        If ``features_format="modality"``::

            features_dic[modality]

        If ``features_format="section"``::

            features_dic[feature_section][modality]

    features_format : {"modality", "section"}, default="modality"
        Feature dictionary format.
        Use ``"modality"`` for query clustering.
        Use either ``"modality"`` or ``"section"`` for reference clustering.

    feature_section : str or None, default=None
        Required when ``features_format="section"``.

    pcs_num_dic : dict or None, default=None
        Modality-specific PC numbers.

    default_pcs_num : int, default=30
        Default number of PCs for modalities not included in ``pcs_num_dic``.

    scale_embedding : bool, default=True
        Whether to scale each modality embedding before concatenation.

    random_state : int, default=0
        Random seed.

    align_by_obs_names : bool, default=True
        Whether to align modalities by shared obs_names before integration.

    sample_name : str or None, default=None
        Optional section/query name for clearer messages.

    copy : bool, default=False
        Copy aligned AnnData objects. This is unnecessary for the built-in
        integration and clustering paths, which only read them.

    Returns
    -------
    base_adata : AnnData
        First selected modality AnnData after alignment.

    integrated_embedding : numpy.ndarray
        Concatenated multi-modal embedding.

    modality_embedding_dic : dict
        Reduced embedding for each selected modality.

    aligned_adata_dic : dict
        Aligned modality AnnData objects.
    """

    sample_msg = f" for {sample_name}" if sample_name is not None else ""

    if dim_reduction_method not in SUPPORTED_REDUCTION_METHODS:
        raise ValueError(
            "dim_reduction_method must be either 'pca' or 'selected_features'."
        )

    if features_format not in {"modality", "section"}:
        raise ValueError("features_format must be either 'modality' or 'section'.")

    if dim_reduction_method == "selected_features" and features_dic is None:
        raise ValueError(
            "features_dic is required when dim_reduction_method='selected_features'."
        )

    if features_format == "section" and feature_section is None:
        raise ValueError("feature_section is required when features_format='section'.")

    aligned_adata_dic = align_modalities_by_obs_names(
        modality_adata_dic=modality_adata_dic,
        selected_modalities=selected_modalities,
        align_by_obs_names=align_by_obs_names,
        sample_name=sample_name,
        copy=copy,
    )

    pcs_num_dic = {} if pcs_num_dic is None else dict(pcs_num_dic)
    modality_embedding_dic = {}

    for modality in selected_modalities:
        modality_adata = aligned_adata_dic[modality]
        pcs_num = int(pcs_num_dic.get(modality, default_pcs_num))

        if dim_reduction_method == "selected_features":
            if features_format == "modality":
                if modality not in features_dic:
                    raise KeyError(
                        f"features_dic does not contain modality {modality!r}{sample_msg}."
                    )
                selected_features = list(features_dic[modality])

            else:
                if feature_section not in features_dic:
                    raise KeyError(
                        f"features_dic does not contain feature_section "
                        f"{feature_section!r}{sample_msg}."
                    )
                if modality not in features_dic[feature_section]:
                    raise KeyError(
                        f"features_dic[{feature_section!r}] does not contain "
                        f"modality {modality!r}{sample_msg}."
                    )
                selected_features = list(features_dic[feature_section][modality])
        else:
            selected_features = None

        embedding = compute_modality_embedding(
            input_adata=modality_adata,
            dim_reduction_method=dim_reduction_method,
            selected_features=selected_features,
            pcs_num=pcs_num,
            scale_embedding=scale_embedding,
            random_state=random_state,
            sample_name=sample_name,
            modality_name=modality,
        )

        modality_embedding_dic[modality] = embedding

    integrated_embedding = np.concatenate(
        [modality_embedding_dic[m] for m in selected_modalities],
        axis=1,
    )

    base_adata = aligned_adata_dic[selected_modalities[0]]

    return base_adata, integrated_embedding, modality_embedding_dic, aligned_adata_dic


def integrate_modalities_for_section(
    ref_section: str,
    selected_modalities: Sequence[str],
    dim_reduction_method: str = "pca",
    ref_gene_dic: Optional[Mapping[str, Any]] = None,
    ref_image_dic: Optional[Mapping[str, Any]] = None,
    ref_protein_dic: Optional[Mapping[str, Any]] = None,
    features_dic: Optional[Mapping[str, Any]] = None,
    features_format: str = "section",
    pcs_num_dic: Optional[Mapping[str, int]] = None,
    default_pcs_num: int = 30,
    scale_embedding: bool = True,
    random_state: int = 0,
    align_by_obs_names: bool = False,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Reference-section wrapper around ``integrate_modalities_for_adata_dic``.

    This function is reference-specific only because it retrieves modality AnnData
    objects from ``ref_gene_dic/ref_image_dic/ref_protein_dic``. The actual
    integration logic is shared.

    Parameters
    ----------
    features_format : {"section", "modality"}, default="section"
        For reference-section clustering, selected features can be stored as
        either ``features_dic[ref_section][modality]`` or ``features_dic[modality]``.

    align_by_obs_names : bool, default=False
        Default is False to preserve the original reference-section behavior:
        modalities are expected to already have aligned rows. Set True only if
        you intentionally want to keep the intersection of obs_names.

    Returns
    -------
    integrated_embedding : numpy.ndarray
        Concatenated embedding for this reference section.

    modality_embedding_dic : dict
        Reduced modality embeddings for this reference section.
    """

    modality_adata_dic = {}

    for modality in selected_modalities:
        modality_adata_dic[modality] = get_ref_modality_adata(
            ref_section=ref_section,
            modality=modality,
            ref_gene_dic=ref_gene_dic,
            ref_image_dic=ref_image_dic,
            ref_protein_dic=ref_protein_dic,
        )

    _, integrated_embedding, modality_embedding_dic, _ = integrate_modalities_for_adata_dic(
        modality_adata_dic=modality_adata_dic,
        selected_modalities=selected_modalities,
        dim_reduction_method=dim_reduction_method,
        features_dic=features_dic,
        features_format=features_format,
        feature_section=ref_section,
        pcs_num_dic=pcs_num_dic,
        default_pcs_num=default_pcs_num,
        scale_embedding=scale_embedding,
        random_state=random_state,
        align_by_obs_names=align_by_obs_names,
        sample_name=ref_section,
    )

    return integrated_embedding, modality_embedding_dic


def cluster_integrated_embedding(
    integrated_embedding: np.ndarray,
    clustering_config: Mapping[str, Any],
    cluster_key: str = "clusters",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Cluster an integrated embedding using KMeans or Leiden.

    Shared by reference and query workflows.

    Parameters
    ----------
    integrated_embedding : numpy.ndarray
        Integrated feature matrix with shape ``(n_obs, n_integrated_features)``.

    clustering_config : dict
        Clustering configuration.

        Required key:

        ``clustering_method`` : {'kmeans', 'leiden'}
            Clustering method.

        Required when ``clustering_method='kmeans'``:

        ``n_clusters`` : int
            Number of KMeans clusters. Must be at least 2 and cannot exceed
            ``n_obs``.

        Optional when ``clustering_method='leiden'``:

        ``resolution`` : float, default=0.5
            Leiden resolution parameter.

        ``n_neighbors`` : int, default=15
            Number of neighbors used to construct the graph. Internally adjusted
            to at most ``n_obs - 1``.

        Optional for both methods:

        ``random_state`` : int, default=0
            Random seed.

    cluster_key: str
        clustering prediction column name

    Returns
    -------
    cluster_labels : numpy.ndarray
        One-dimensional array of cluster labels.

    cluster_info : dict
        Metadata for the clustering run.
    """

    if "clustering_method" not in clustering_config:
        raise KeyError("clustering_config must contain 'clustering_method'.")

    clustering_method = clustering_config["clustering_method"]
    if clustering_method not in SUPPORTED_CLUSTERING_METHODS:
        raise ValueError("clustering_method must be either 'kmeans' or 'leiden'.")

    integrated_embedding = np.asarray(integrated_embedding)
    random_state = int(clustering_config.get("random_state", 0))

    if integrated_embedding.shape[0] < 2:
        raise ValueError("Clustering requires at least 2 observations.")

    if clustering_method == "kmeans":
        if "n_clusters" not in clustering_config:
            raise ValueError("n_clusters is required when clustering_method='kmeans'.")

        n_clusters = int(clustering_config["n_clusters"])

        cluster_labels = kmeans_clustering(
            features_matrix=integrated_embedding,
            n_clusters=n_clusters,
            random_state=random_state,
        )

        cluster_info = {
            "clustering_method": "kmeans",
            "n_clusters": n_clusters,
            "random_state": random_state,
        }

    else:
        cluster_labels, cluster_info = leiden_clustering(
            features_matrix=integrated_embedding,
            resolution=clustering_config.get("resolution", 0.5),
            n_neighbors=clustering_config.get("n_neighbors", 15),
            random_state=random_state,
            leiden_key=cluster_key,
            neighbors_method=clustering_config.get("neighbors_method", "umap"),
            neighbors_metric=clustering_config.get(
                "neighbors_metric",
                clustering_config.get("metric", "euclidean"),
            ),
            leiden_flavor=clustering_config.get("leiden_flavor", "leidenalg"),
            leiden_directed=clustering_config.get("leiden_directed", None),
            leiden_n_iterations=clustering_config.get(
                "leiden_n_iterations",
                None,
            ),
            return_info=True,
        )

    return cluster_labels, cluster_info
