from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from ..visualization import cat_figure


__all__ = [
    "aggregate_grid_to_spots",
    "visualize_img_clusters",
]


def _to_dense_array(X) -> np.ndarray:
    """Return a dense NumPy array from a dense or sparse matrix."""
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X)


def _normalise_cluster_list(n_clusters: int | Sequence[int] | None) -> list[int]:
    """Convert one cluster value or a sequence of cluster values to a clean list."""
    if n_clusters is None:
        return []

    if isinstance(n_clusters, Iterable) and not isinstance(n_clusters, (str, bytes)):
        cluster_list = list(n_clusters)
    else:
        cluster_list = [n_clusters]

    cluster_list = [int(k) for k in cluster_list]
    if any(k < 2 for k in cluster_list):
        raise ValueError("All KMeans cluster numbers must be >= 2.")
    return cluster_list


def aggregate_grid_to_spots(
    feature_grid,
    spot_coordinates,
    patch_size_spot: int = 200,
    patch_size_emb: int = 16,
    spot_x_key: str = "pixel_x",
    spot_y_key: str = "pixel_y",
    method: str = "weighted",
    normalize_by: str = "overlap",
    ignore_zero_features: bool = False,
    zero_tol: float = 1e-8,
):
    """
    Aggregate grid-level image embeddings into spot-level image features.

    This function is shared by the HIPT and UNI backends. It expects a dense
    grid with shape ``(n_features, n_x, n_y)``, where ``n_x`` and ``n_y`` are
    grid indices in image-pixel order. Each spot is represented by a square
    window centered at ``spot_x_key`` and ``spot_y_key``.

    Parameters
    ----------
    feature_grid : numpy.ndarray
        Feature grid with shape ``(n_features, n_x, n_y)``.
    spot_coordinates : pandas.DataFrame
        Spot-level coordinate table.
    patch_size_spot : int, default=200
        Spot window size in pixels.
    patch_size_emb : int, default=16
        Pixel size represented by each embedding-grid cell.
    spot_x_key : str, default="pixel_x"
        Column in ``spot_coordinates`` storing x pixel coordinates.
    spot_y_key : str, default="pixel_y"
        Column in ``spot_coordinates`` storing y pixel coordinates.
    method : {"weighted", "mean", "median"}, default="weighted"
        Aggregation method. ``weighted`` uses exact area overlap between the
        spot window and each embedding-grid cell.
    normalize_by : {"overlap", "spot"}, default="overlap"
        For weighted aggregation, divide by the observed overlap area or by the
        full spot area. ``overlap`` is usually safer for edge spots after image
        padding/cropping. ``spot`` preserves the old behavior where missing or
        clipped area effectively contributes zero.
    ignore_zero_features : bool, default=False
        If True, cells with all feature values close to zero are ignored. This
        can reduce artificial boundary/background clusters when masked
        background embeddings are exactly zero.
    zero_tol : float, default=1e-8
        Tolerance used by ``ignore_zero_features``.

    Returns
    -------
    numpy.ndarray
        Spot-level feature matrix with shape ``(n_spots, n_features)``.
    """
    if feature_grid is None:
        raise ValueError("feature_grid must be provided.")
    if not isinstance(spot_coordinates, pd.DataFrame):
        raise TypeError("spot_coordinates must be a pandas.DataFrame.")

    feature_grid = np.asarray(feature_grid)
    if feature_grid.ndim != 3:
        raise ValueError(
            "feature_grid must have shape (n_features, n_x, n_y); "
            f"got {feature_grid.shape}."
        )

    if patch_size_spot <= 0 or patch_size_emb <= 0:
        raise ValueError("patch_size_spot and patch_size_emb must be positive.")

    method = method.lower().strip()
    if method not in {"weighted", "mean", "median"}:
        raise ValueError("method must be one of {'weighted', 'mean', 'median'}.")

    normalize_by = normalize_by.lower().strip()
    if normalize_by not in {"overlap", "spot"}:
        raise ValueError("normalize_by must be one of {'overlap', 'spot'}.")

    missing_cols = {spot_x_key, spot_y_key} - set(spot_coordinates.columns)
    if missing_cols:
        raise ValueError(f"spot_coordinates is missing columns: {missing_cols}")

    feature_grid = np.nan_to_num(feature_grid, nan=0.0, posinf=0.0, neginf=0.0)
    n_features, n_x, n_y = feature_grid.shape
    spot_feature_matrix = np.zeros((len(spot_coordinates), n_features), dtype=np.float32)

    half = patch_size_spot / 2.0
    full_spot_area = float(patch_size_spot * patch_size_spot)

    for out_i, (_, row) in enumerate(spot_coordinates.iterrows()):
        x_center = float(row[spot_x_key])
        y_center = float(row[spot_y_key])

        x_min = x_center - half
        x_max = x_center + half
        y_min = y_center - half
        y_max = y_center + half

        gx0 = max(0, int(np.floor(x_min / patch_size_emb)))
        gx1 = min(n_x, int(np.ceil(x_max / patch_size_emb)))
        gy0 = max(0, int(np.floor(y_min / patch_size_emb)))
        gy1 = min(n_y, int(np.ceil(y_max / patch_size_emb)))

        if gx0 >= gx1 or gy0 >= gy1:
            continue

        patch = feature_grid[:, gx0:gx1, gy0:gy1]
        if patch.size == 0:
            continue

        if method in {"mean", "median"}:
            flat = patch.reshape(n_features, -1)
            if ignore_zero_features:
                valid = np.any(np.abs(flat) > zero_tol, axis=0)
                flat = flat[:, valid]
                if flat.shape[1] == 0:
                    continue

            if method == "mean":
                spot_feature_matrix[out_i] = np.nanmean(flat, axis=1)
            else:
                spot_feature_matrix[out_i] = np.nanmedian(flat, axis=1)
            continue

        weighted_sum = np.zeros(n_features, dtype=np.float64)
        area_sum = 0.0

        for gx in range(gx0, gx1):
            cell_x0 = gx * patch_size_emb
            cell_x1 = (gx + 1) * patch_size_emb
            x_overlap = max(0.0, min(x_max, cell_x1) - max(x_min, cell_x0))
            if x_overlap == 0:
                continue

            for gy in range(gy0, gy1):
                cell_y0 = gy * patch_size_emb
                cell_y1 = (gy + 1) * patch_size_emb
                y_overlap = max(0.0, min(y_max, cell_y1) - max(y_min, cell_y0))
                if y_overlap == 0:
                    continue

                vec = feature_grid[:, gx, gy]
                if ignore_zero_features and not np.any(np.abs(vec) > zero_tol):
                    continue

                area = x_overlap * y_overlap
                weighted_sum += vec * area
                area_sum += area

        if area_sum == 0:
            continue

        denom = area_sum if normalize_by == "overlap" else full_spot_area
        spot_feature_matrix[out_i] = (weighted_sum / denom).astype(np.float32)

    return spot_feature_matrix


def visualize_img_clusters(
    spot_adata,
    output_dir=None,
    sample_name: str | None = None,
    do_pca: bool = True,
    n_pcs: int = 50,
    do_clustering: bool = True,
    n_clusters: int | Sequence[int] | None = 10,
    cluster_method: str = "kmeans",
    plot_results: bool = True,
    plot_spot_size: int = 100,
    plot_x_key: str = "pixel_x",
    plot_y_key: str = "pixel_y",
    cat_color=None,
    dpi: int = 200,
    invert_x: bool = False,
    invert_y: bool = True,
    save_h5ad: bool = True,
    random_state: int = 42,
    copy: bool = False,
):
    """
    Optionally run PCA, KMeans clustering, spatial visualization, and saving.

    By default this updates ``spot_adata`` in place because the UNI/HIPT
    wrappers pass a newly created object. Set ``copy=True`` to preserve an
    independently owned input object.
    """
    adata = spot_adata.copy() if copy or spot_adata.is_view else spot_adata

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    X = np.nan_to_num(_to_dense_array(adata.X), nan=0.0, posinf=0.0, neginf=0.0)
    if X.ndim != 2:
        raise ValueError(f"spot_adata.X must be 2-dimensional, got shape {X.shape}.")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("spot_adata.X must contain at least one spot and one feature.")
    adata.X = X

    if do_pca:
        n_components = min(int(n_pcs), X.shape[0], X.shape[1])
        if n_components < 1:
            raise ValueError("n_pcs must be >= 1 after checking data dimensions.")

        pca = PCA(n_components=n_components, random_state=random_state)
        adata.obsm["X_pca"] = pca.fit_transform(X)
        embedding = adata.obsm["X_pca"]
    else:
        embedding = X

    cluster_list = _normalise_cluster_list(n_clusters) if do_clustering else []
    if cluster_list:
        if cluster_method.lower().strip() != "kmeans":
            raise ValueError("Only cluster_method='kmeans' is currently supported.")

        for k in cluster_list:
            if k > adata.n_obs:
                raise ValueError(
                    f"KMeans n_clusters={k} cannot exceed n_spots={adata.n_obs}."
                )

            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            adata.obs[f"kmeans_{k}"] = kmeans.fit_predict(embedding).astype(str)

            if plot_results and output_dir is not None:
                missing_plot_cols = {plot_x_key, plot_y_key} - set(adata.obs.columns)
                if missing_plot_cols:
                    raise ValueError(
                        "Cannot plot image clusters because .obs is missing "
                        f"columns: {missing_plot_cols}."
                    )

                title_prefix = f"{sample_name}: " if sample_name else ""

                cat_figure(
                    input_adata=adata,
                    x_key=plot_x_key,
                    y_key=plot_y_key,
                    fig_title=f"{title_prefix}Image Features Clustering (K={k})",
                    fig_path=output_dir / f"spot_image_features_kmeans_{k}.png",
                    color_key=f"kmeans_{k}",
                    cat_color=cat_color,
                    fig_size=plot_spot_size,
                    dpi=dpi,
                    invert_x=invert_x,
                    invert_y=invert_y,
                )

    if output_dir is not None and save_h5ad:
        adata.write_h5ad(output_dir / "spot_image_features.h5ad")

    return adata
