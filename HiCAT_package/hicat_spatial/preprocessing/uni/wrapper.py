from __future__ import annotations

from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

from .extract_features import uni_extract_features
from .histosweep import uni_generate_mask
from .preprocess import uni_preprocess_image
from .uni_2_spot import uni_patch_2_spot


__all__ = ["extract_spatial_image_features"]


def _resolve_uni_checkpoint(checkpoint_path):
    """Resolve an optional UNI checkpoint path."""
    if checkpoint_path is None:
        return None

    checkpoint_path = Path(checkpoint_path).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"UNI checkpoint_path does not exist: {checkpoint_path}")
    if checkpoint_path.is_dir():
        raise ValueError(
            "For model='uni', checkpoint_path should usually be the checkpoint "
            "file, for example checkpoints/pytorch_model.bin, not a directory."
        )
    return str(checkpoint_path)


def extract_spatial_image_features(
    sample,
    spot_coordinates,
    output_dir=None,
    checkpoint_path=None,
    patch_size_spot: int = 200,
    patch_size_emb: int = 16,
    aggregation_method: str = "weighted",
    normalize_by: str = "overlap",
    ignore_zero_features: bool = False,
    zero_tol: float = 1e-8,
    scale_value: float = 1.0,
    pad_value: int = 16,
    mask_save_dir: str = "mask",
    density_thresh: int = 100,
    clean_background_flag: bool = False,
    min_size: int = 10,
    device: str = "cuda",
    batch_size: int = 128,
    stride: int = 112,
    num_workers: int = 8,
    spot_pixel_x: str = "pixel_x",
    spot_pixel_y: str = "pixel_y",
    cat_color=None,
    dpi: int = 200,
    invert_x: bool = False,
    invert_y: bool = True,
    ncluster_list=None,
    npcs: int = 50,
    plot_clusters: bool = True,
    plot_spot_size: int = 100,
    save_h5ad: bool = True,
    spatial_key: str = "spatial",
    spatial_coords_are_pixel: bool = False,
    random_state: int = 42,
):
    """
    Run UNI preprocessing, mask generation, grid embedding extraction, and spot aggregation.

    Parameters
    ----------
    checkpoint_path : str or pathlib.Path, optional
        UNI pretrained checkpoint file, usually ``pytorch_model.bin``.
    """
    if sample is None or not isinstance(sample, (str, Path)) or str(sample).strip() == "":
        raise ValueError("sample must be a non-empty path-like value.")

    sample = Path(sample)
    if not sample.exists():
        raise FileNotFoundError(f"Sample folder not found: {sample}")

    checkpoint_path = _resolve_uni_checkpoint(checkpoint_path)
    output_dir = Path(output_dir) if output_dir is not None else sample / "results"

    uni_preprocess_image(
        sample=str(sample),
        scale_value=scale_value,
        pad_value=pad_value,
    )

    uni_generate_mask(
        sample=str(sample),
        save_dir=mask_save_dir,
        density_thresh=density_thresh,
        clean_background_flag=clean_background_flag,
        min_size=min_size,
        patch_size=patch_size_emb,
    )

    uni_extract_features(
        sample=str(sample),
        device=device,
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        stride=stride,
        num_workers=num_workers,
    )

    return uni_patch_2_spot(
        sample=sample,
        spot_coordinates=spot_coordinates,
        spot_pixel_x=spot_pixel_x,
        spot_pixel_y=spot_pixel_y,
        patch_size_spot=patch_size_spot,
        patch_size_emb=patch_size_emb,
        aggregation_method=aggregation_method,
        normalize_by=normalize_by,
        ignore_zero_features=ignore_zero_features,
        zero_tol=zero_tol,
        ncluster_list=ncluster_list,
        npcs=npcs,
        plot_clusters=plot_clusters,
        plot_spot_size=plot_spot_size,
        cat_color=cat_color,
        dpi=dpi,
        invert_x=invert_x,
        invert_y=invert_y,
        output_dir=output_dir,
        save_h5ad=save_h5ad,
        spatial_key=spatial_key,
        spatial_coords_are_pixel=spatial_coords_are_pixel,
        random_state=random_state,
    )

