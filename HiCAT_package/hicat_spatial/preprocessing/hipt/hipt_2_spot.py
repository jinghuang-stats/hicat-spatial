from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import scanpy as sc

from ..image_utils import aggregate_grid_to_spots, visualize_img_clusters


__all__ = ["patch_2_spot"]


def _load_pickle(path: str | Path):
    """Load a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def _combine_hipt_embedding_grids(embedding_obj: dict) -> tuple[np.ndarray, list[str]]:
    """
    Combine HIPT embedding grids into one feature grid.

    The usual HIPT keys are ``"cls"``, ``"sub"``, and ``"rgb"``. Any additional
    keys are appended after these preferred keys.

    HIPT stores each channel in image-array order ``(feature, row_y, col_x)``.
    The shared spot aggregator expects ``(feature, x, y)``, so this function
    transposes the two spatial axes before returning the combined grid.
    """
    if not isinstance(embedding_obj, dict) or len(embedding_obj) == 0:
        raise ValueError("HIPT embedding object must be a non-empty dictionary.")

    preferred_keys = ["cls", "sub", "rgb"]
    keys = [k for k in preferred_keys if k in embedding_obj]
    keys += [k for k in embedding_obj.keys() if k not in keys]

    grids = []
    feature_names = []
    expected_shape = None

    for key in keys:
        grid = np.asarray(embedding_obj[key])
        if grid.ndim != 3:
            raise ValueError(
                f"HIPT embedding for key '{key}' must have shape "
                f"(n_features, n_y, n_x), got {grid.shape}."
            )

        raw_spatial_shape = grid.shape[1:]
        if expected_shape is None:
            expected_shape = raw_spatial_shape
        elif raw_spatial_shape != expected_shape:
            raise ValueError(
                "All HIPT embedding grids must have the same spatial shape. "
                f"Expected {expected_shape}, got {raw_spatial_shape} for key '{key}'."
            )

        grid = np.swapaxes(grid, 1, 2)
        grids.append(grid)
        feature_names.extend([f"{key}_{i}" for i in range(grid.shape[0])])

    combined_feature_grid = np.concatenate(grids, axis=0)
    combined_feature_grid = np.nan_to_num(
        combined_feature_grid,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
        copy=False,
    )

    return combined_feature_grid, feature_names


def patch_2_spot(
    sample,
    spot_coordinates: pd.DataFrame,
    output_dir=None,
    patch_size_spot: int = 200,
    patch_size_emb: int = 16,
    spot_pixel_x: str = "pixel_x",
    spot_pixel_y: str = "pixel_y",
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
    random_state: int = 42,
):
    """Convert HIPT grid-level embeddings to spot-level image features.

    Output features are indexed as ``hipt_0``, ...,
    ``hipt_{n_features - 1}``. The original HIPT group/channel identifiers are
    retained in ``spot_adata.var["source_name"]``.
    """
    sample = Path(sample)
    results_dir = Path(output_dir) if output_dir is not None else sample / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not isinstance(spot_coordinates, pd.DataFrame):
        raise TypeError("spot_coordinates must be a pandas.DataFrame.")

    missing_cols = {spot_pixel_x, spot_pixel_y} - set(spot_coordinates.columns)
    if missing_cols:
        raise ValueError(f"spot_coordinates is missing columns: {missing_cols}")

    emb_path = sample / "embeddings-hist.pickle"
    if not emb_path.exists():
        raise FileNotFoundError(emb_path)

    embedding_obj = _load_pickle(emb_path)
    feature_grid, feature_names = _combine_hipt_embedding_grids(embedding_obj)

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
    spot_adata.var_names = [f"hipt_{i}" for i in range(spot_adata.n_vars)]
    spot_adata.var["name"] = list(spot_adata.var_names)
    spot_adata.var["source_name"] = feature_names

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

    print("----------Finished extracting HIPT spot-level image features----------")
    return spot_adata

