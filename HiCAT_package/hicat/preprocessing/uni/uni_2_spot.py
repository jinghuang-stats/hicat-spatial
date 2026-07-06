from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import scanpy as sc

from ..image_utils import aggregate_grid_to_spots, visualize_img_clusters

__all__ = ["uni_patch_2_spot"]

def _to_dense_array(X) -> np.ndarray:
    """Convert sparse or array-like matrix to a dense NumPy array."""
    if hasattr(X, "toarray"):
        return X.toarray()
    return np.asarray(X)


def _build_uni_feature_grid(
    patch_adata,
    spot_coordinates: pd.DataFrame,
    spot_pixel_x: str,
    spot_pixel_y: str,
    patch_size_spot: int,
    patch_size_emb: int,
    spatial_key: str = "spatial",
    spatial_coords_are_pixel: bool = False,
) -> np.ndarray:
    """
    Reconstruct a dense UNI feature grid from patch-level AnnData.

    Parameters
    ----------
    patch_adata : anndata.AnnData
        UNI patch embedding AnnData. Embeddings are expected in ``.X``.
    spot_coordinates : pandas.DataFrame
        Spot-level coordinate table.
    spot_pixel_x : str
        Column in ``spot_coordinates`` storing spot x pixel coordinates.
    spot_pixel_y : str
        Column in ``spot_coordinates`` storing spot y pixel coordinates.
    patch_size_spot : int
        Spot window size in pixels.
    patch_size_emb : int
        Pixel size represented by one embedding-grid cell.
    spatial_key : str, default="spatial"
        Key in ``patch_adata.obsm`` storing patch coordinates.
    spatial_coords_are_pixel : bool, default=False
        If False, ``patch_adata.obsm[spatial_key]`` is treated as grid indices.
        If True, it is treated as pixel coordinates and divided by
        ``patch_size_emb`` before grid construction.

    Returns
    -------
    numpy.ndarray
        Dense feature grid with shape ``(n_features, n_x, n_y)``.
    """
    if spatial_key not in patch_adata.obsm:
        raise KeyError(f'UNI embedding AnnData must contain `.obsm["{spatial_key}"]`.')

    spatial = np.asarray(patch_adata.obsm[spatial_key])
    if spatial.ndim != 2 or spatial.shape[1] < 2:
        raise ValueError(f'`.obsm["{spatial_key}"]` must have shape (n_patches, >=2).')

    if spatial_coords_are_pixel:
        cols = np.floor(spatial[:, 0] / patch_size_emb).astype(int)
        rows = np.floor(spatial[:, 1] / patch_size_emb).astype(int)
    else:
        cols = spatial[:, 0].astype(int)
        rows = spatial[:, 1].astype(int)

    if np.any(cols < 0) or np.any(rows < 0):
        raise ValueError("UNI grid coordinates must be non-negative.")

    X = _to_dense_array(patch_adata.X)
    if X.ndim != 2:
        raise ValueError(f"patch_adata.X must be 2-dimensional, got shape {X.shape}.")
    if X.shape[0] != cols.shape[0]:
        raise ValueError(
            "Number of UNI embedding rows does not match number of spatial coordinates: "
            f"X has {X.shape[0]} rows, spatial has {cols.shape[0]} rows."
        )

    n_features = X.shape[1]

    # Build a grid large enough for both extracted patch locations and requested
    # spot windows. This avoids the old fixed +1000 margin while protecting
    # boundary spots.
    max_spot_x = float(pd.to_numeric(spot_coordinates[spot_pixel_x]).max())
    max_spot_y = float(pd.to_numeric(spot_coordinates[spot_pixel_y]).max())
    half = patch_size_spot / 2.0

    n_x_from_spots = int(np.ceil((max_spot_x + half) / patch_size_emb)) + 2
    n_y_from_spots = int(np.ceil((max_spot_y + half) / patch_size_emb)) + 2

    n_x = max(int(cols.max()) + 1, n_x_from_spots)
    n_y = max(int(rows.max()) + 1, n_y_from_spots)

    feature_grid = np.zeros((n_features, n_x, n_y), dtype=X.dtype)
    feature_grid[:, cols, rows] = X.T
    feature_grid = np.nan_to_num(
        feature_grid,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
        copy=False,
    )

    return feature_grid


def uni_patch_2_spot(
    sample,
    spot_coordinates: pd.DataFrame,
    output_dir=None,
    spot_pixel_x: str = "pixel_x",
    spot_pixel_y: str = "pixel_y",
    patch_size_spot: int = 200,
    patch_size_emb: int = 16,
    aggregation_method: str = "weighted",
    normalize_by: str = "overlap",
    ignore_zero_features: bool = False,
    zero_tol: float = 1e-8,
    ncluster_list: Sequence[int] | int | None = None,
    npcs: int = 50,
    plot_clusters: bool = True,
    plot_spot_size: int = 100,
    cat_color=None,
    dpi: int = 200,
    invert_x: bool = False,
    invert_y: bool = True,
    save_h5ad: bool = True,
    spatial_key: str = "spatial",
    spatial_coords_are_pixel: bool = False,
    random_state: int = 42,
):
    """
    Convert UNI patch/grid-level embeddings to spot-level image features.

    Parameters
    ----------
    sample : str or pathlib.Path
        Sample directory containing ``uni_super_emb.h5ad``.
    spot_coordinates : pandas.DataFrame
        Spot-level coordinate table. Must contain ``spot_pixel_x`` and
        ``spot_pixel_y`` columns in image pixel units.
    output_dir : str or pathlib.Path, optional
        Directory for saving ``spot_image_features.h5ad`` and cluster figures.
        Defaults to ``sample / "results"``.
    spot_pixel_x : str, default="pixel_x"
        Column name for spot x-coordinate.
    spot_pixel_y : str, default="pixel_y"
        Column name for spot y-coordinate.
    patch_size_spot : int, default=200
        Spot window size in pixels.
    patch_size_emb : int, default=16
        Spatial resolution of one UNI embedding grid cell in pixels.
    aggregation_method : {"weighted", "mean", "median"}, default="weighted"
        Method used to aggregate grid-level features into each spot.
    normalize_by : {"overlap", "spot"}, default="overlap"
        Normalization rule for weighted aggregation.
    ignore_zero_features : bool, default=False
        Ignore all-zero grid cells during spot aggregation.
    zero_tol : float, default=1e-8
        Tolerance used when ``ignore_zero_features=True``.
    ncluster_list : int, sequence of int, or None, default=None
        KMeans cluster numbers. If None, clustering and cluster plots are skipped.
    npcs : int, default=50
        Number of principal components used before KMeans clustering.
    plot_clusters : bool, default=True
        Whether to run KMeans and save spatial cluster plots.
    plot_spot_size : int, default=100
        Marker size for cluster visualization.
    cat_color : list, tuple, or None
        Categorical color palette used for plotting. Each category will be assigned
        one color from this palette. If `None`, Scanpy will use its default
        categorical color palette.
    dpi : int, default=100
        Resolution of the saved figure.
    invert_x : bool, default=False
        Whether to invert the x-axis.
    invert_y : bool, default=True
        Whether to invert the y-axis. This is often useful for spatial transcriptomics
        or image-based coordinates where the image origin is in the upper-left corner.
    save_h5ad : bool, default=True
        Whether to save ``spot_image_features.h5ad``.
    spatial_key : str, default="spatial"
        Key in ``uni_super_emb.h5ad`` ``.obsm`` containing patch coordinates.
    spatial_coords_are_pixel : bool, default=False
        Whether ``.obsm[spatial_key]`` stores pixel coordinates instead of grid indices.
    random_state : int, default=42
        Random seed for PCA and KMeans.

    Returns
    -------
    anndata.AnnData
        AnnData object containing spot-level UNI image features in ``.X``.
        Features are indexed as ``uni_0``, ..., ``uni_{n_features - 1}`` in the
        same order as the UNI embedding dimensions.
        If clustering is requested, PCA is stored in ``.obsm["X_pca"]`` and
        KMeans labels are stored in ``.obs["kmeans_{k}"]``.
    """
    sample = Path(sample)
    results_dir = Path(output_dir) if output_dir is not None else sample / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not isinstance(spot_coordinates, pd.DataFrame):
        raise TypeError("spot_coordinates must be a pandas.DataFrame.")

    missing_cols = {spot_pixel_x, spot_pixel_y} - set(spot_coordinates.columns)
    if missing_cols:
        raise ValueError(f"spot_coordinates is missing columns: {missing_cols}")

    emb_file = sample / "uni_super_emb.h5ad"
    if not emb_file.exists():
        raise FileNotFoundError(emb_file)

    patch_adata = sc.read_h5ad(emb_file)
    feature_grid = _build_uni_feature_grid(
        patch_adata=patch_adata,
        spot_coordinates=spot_coordinates,
        spot_pixel_x=spot_pixel_x,
        spot_pixel_y=spot_pixel_y,
        patch_size_spot=patch_size_spot,
        patch_size_emb=patch_size_emb,
        spatial_key=spatial_key,
        spatial_coords_are_pixel=spatial_coords_are_pixel,
    )

    spot_feature_matrix = aggregate_grid_to_spots(
        feature_grid=feature_grid,
        spot_coordinates=spot_coordinates,
        patch_size_spot=patch_size_spot,
        patch_size_emb=patch_size_emb,
        spot_x_key=spot_pixel_x,
        spot_y_key=spot_pixel_y,
        method=aggregation_method,
        normalize_by=normalize_by,
        ignore_zero_features=ignore_zero_features,
        zero_tol=zero_tol,
    )

    spot_adata = sc.AnnData(spot_feature_matrix)
    spot_adata.obs = spot_coordinates.copy()
    spot_adata.var_names = [f"uni_{i}" for i in range(spot_adata.n_vars)]
    spot_adata.var["name"] = list(spot_adata.var_names)

    do_clustering = bool(plot_clusters) and ncluster_list is not None
    spot_adata = visualize_img_clusters(
        spot_adata=spot_adata,
        output_dir=results_dir,
        sample_name=sample.name,
        do_pca=True,
        n_pcs=npcs,
        do_clustering=do_clustering,
        n_clusters=ncluster_list if do_clustering else None,
        cluster_method="kmeans",
        plot_results=bool(plot_clusters),
        plot_spot_size=plot_spot_size,
        plot_x_key=spot_pixel_x,
        plot_y_key=spot_pixel_y,
        cat_color=cat_color,
        dpi=dpi,
        invert_x=invert_x,
        invert_y=invert_y,
        save_h5ad=save_h5ad,
        random_state=random_state,
    )

    print("----------Finished extracting UNI spot-level image features----------")
    return spot_adata
