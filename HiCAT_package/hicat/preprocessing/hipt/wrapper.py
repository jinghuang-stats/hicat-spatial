from __future__ import annotations

from pathlib import Path

from .extract_features import hipt_extract_features
from .hipt_2_spot import patch_2_spot
from .preprocess import preprocess_image


__all__ = ["extract_spatial_image_features"]


def _resolve_hipt_checkpoints(checkpoint_path):
    """Resolve HIPT checkpoint inputs."""
    if checkpoint_path is None:
        return None, None, None

    checkpoint_path = Path(checkpoint_path).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"HIPT checkpoint_path does not exist: {checkpoint_path}")

    if checkpoint_path.is_dir():
        vit256_path = checkpoint_path / "vit256_small_dino.pth"
        vit4k_path = checkpoint_path / "vit4k_xs_dino.pth"
        missing = [str(p) for p in [vit256_path, vit4k_path] if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "HIPT checkpoint directory is missing required file(s): "
                f"{missing}."
            )
        return str(checkpoint_path), str(vit256_path), str(vit4k_path)

    return str(checkpoint_path), None, None


def extract_spatial_image_features(
    sample,
    spot_coordinates,
    output_dir=None,
    checkpoint_path=None,
    mask=None,
    patch_size_spot: int = 200,
    patch_size_emb: int = 16,
    aggregation_method: str = "weighted",
    normalize_by: str = "overlap",
    ignore_zero_features: bool = False,
    zero_tol: float = 1e-8,
    pad_size: int = 256,
    device: str = "cuda",
    reduction_method=None,
    n_components=None,
    smoothen_method: str = "cv",
    random_weights: bool = False,
    no_shift: bool = False,
    use_cache: bool = True,
    ncluster_list=None,
    npcs: int = 50,
    plot_clusters: bool = True,
    plot_spot_size: int = 100,
    cat_color=None,
    dpi: int = 200,
    invert_x: bool = False,
    invert_y: bool = True,
    save_h5ad: bool = True,
    spot_pixel_x: str = "pixel_x",
    spot_pixel_y: str = "pixel_y",
    random_state: int = 42,
):
    """
    Run HIPT preprocessing, grid embedding extraction, and spot aggregation.

    Parameters
    ----------
    checkpoint_path : str or pathlib.Path, optional
        HIPT checkpoint directory containing ``vit256_small_dino.pth`` and
        ``vit4k_xs_dino.pth``. It may also be a custom checkpoint file if your
        local ``hipt_extract_features`` implementation supports that.
    """
    if sample is None or not isinstance(sample, (str, Path)) or str(sample).strip() == "":
        raise ValueError("sample must be a non-empty path-like value.")

    sample = Path(sample)
    if not sample.exists():
        raise FileNotFoundError(f"Sample folder not found: {sample}")

    output_dir = Path(output_dir) if output_dir is not None else sample / "results"

    checkpoint_dir, vit256_checkpoint_path, vit4k_checkpoint_path = _resolve_hipt_checkpoints(
        checkpoint_path
    )

    preprocess_image(
        sample=str(sample),
        mask=mask,
        pad_size=pad_size,
    )

    hipt_extract_features(
        sample=str(sample),
        device=device,
        checkpoint_path=checkpoint_dir,
        vit256_checkpoint_path=vit256_checkpoint_path,
        vit4k_checkpoint_path=vit4k_checkpoint_path,
        reduction_method=reduction_method,
        n_components=n_components,
        smoothen_method=smoothen_method,
        random_weights=random_weights,
        no_shift=no_shift,
        use_cache=use_cache,
    )

    return patch_2_spot(
        sample=sample,
        spot_coordinates=spot_coordinates,
        output_dir=output_dir,
        patch_size_spot=patch_size_spot,
        patch_size_emb=patch_size_emb,
        spot_pixel_x=spot_pixel_x,
        spot_pixel_y=spot_pixel_y,
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
        save_h5ad=save_h5ad,
        random_state=random_state,
    )

