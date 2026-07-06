from __future__ import annotations

from pathlib import Path
import shutil
from typing import Sequence
import warnings

import pandas as pd

__all__ = ["extract_image_features"]


_HIPT_SPECIFIC_KWARGS = {
    "mask",
    "pad_size",
    "reduction_method",
    "n_components",
    "smoothen_method",
    "random_weights",
    "no_shift",
    "use_cache",
}

_UNI_SPECIFIC_KWARGS = {
    "scale_value",
    "pad_value",
    "mask_save_dir",
    "density_thresh",
    "clean_background_flag",
    "min_size",
    "batch_size",
    "stride",
    "num_workers",
    "spatial_key",
    "spatial_coords_are_pixel",
}


def _validate_model_specific_kwargs(model: str, kwargs: dict) -> None:
    """Check whether user-supplied extra keyword arguments are valid."""
    allowed = _HIPT_SPECIFIC_KWARGS if model == "hipt" else _UNI_SPECIFIC_KWARGS

    invalid = set(kwargs) - allowed
    if invalid:
        raise ValueError(
            f"Invalid keyword argument(s) for model='{model}': {sorted(invalid)}. "
            f"Allowed model-specific arguments are: {sorted(allowed)}."
        )


def _prepare_sample_dir(
    image_path,
    output_dir=None,
    sample_name: str | None = None,
    raw_image_name: str = "he-raw.jpg",
    overwrite: bool = False,
) -> Path:
    """
    Create a per-sample working directory and place the raw H&E image there.

    The HIPT and UNI preprocessing wrappers usually expect the raw image to be
    named ``he-raw.jpg`` inside the sample folder, so non-JPEG inputs are
    converted to RGB JPEG by default.
    """
    raw_image_path = Path(raw_image_name)
    if raw_image_path.name != raw_image_name or raw_image_path.stem != "he-raw":
        raise ValueError(
            "raw_image_name must be a filename with stem 'he-raw', such as "
            "'he-raw.jpg'; the HIPT and UNI backends resolve that fixed stem."
        )

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if output_dir is None:
        output_dir = image_path.parent / "results"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_name = sample_name or image_path.stem
    sample_dir = output_dir / sample_name
    sample_dir.mkdir(parents=True, exist_ok=True)

    target_image = sample_dir / raw_image_name
    if overwrite or not target_image.exists():
        if image_path.suffix.lower() in {".jpg", ".jpeg"} and raw_image_name.lower().endswith(
            (".jpg", ".jpeg")
        ):
            shutil.copy2(image_path, target_image)
        else:
            try:
                from PIL import Image

                Image.MAX_IMAGE_PIXELS = None
                with Image.open(image_path) as img:
                    img.convert("RGB").save(target_image, quality=95)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to copy/convert {image_path} to {target_image}. "
                    "Provide a JPEG image or check that Pillow can read the file."
                ) from exc

    return sample_dir


def _load_spot_coordinates(
    spot_coordinates_path=None,
    spot_coordinates=None,
    spot_x_key: str = "pixel_x",
    spot_y_key: str = "pixel_y",
) -> pd.DataFrame:
    """Load and standardize spot coordinates to pixel_x/pixel_y columns."""
    if spot_coordinates is None:
        if spot_coordinates_path is None:
            raise ValueError("Either spot_coordinates or spot_coordinates_path must be provided.")
        spot_coordinates = pd.read_csv(spot_coordinates_path)

    if not isinstance(spot_coordinates, pd.DataFrame):
        raise TypeError("spot_coordinates must be a pandas.DataFrame.")

    missing_cols = {spot_x_key, spot_y_key} - set(spot_coordinates.columns)
    if missing_cols:
        raise ValueError(f"spot_coordinates is missing columns: {missing_cols}")

    spot_coordinates = spot_coordinates.copy()
    rename_dic = {}
    if spot_x_key != "pixel_x":
        rename_dic[spot_x_key] = "pixel_x"
    if spot_y_key != "pixel_y":
        rename_dic[spot_y_key] = "pixel_y"
    if rename_dic:
        spot_coordinates = spot_coordinates.rename(columns=rename_dic)

    spot_coordinates["pixel_x"] = pd.to_numeric(spot_coordinates["pixel_x"], errors="raise")
    spot_coordinates["pixel_y"] = pd.to_numeric(spot_coordinates["pixel_y"], errors="raise")
    return spot_coordinates


def _resolve_checkpoint_path(checkpoint_path, model: str) -> str | None:
    """
    Resolve an optional user-supplied checkpoint path.

    For ``model='uni'``, this is usually a checkpoint file such as
    ``pytorch_model.bin``. For ``model='hipt'``, this is usually a checkpoint
    directory containing ``vit256_small_dino.pth`` and ``vit4k_xs_dino.pth``.
    """
    if checkpoint_path is None:
        return None

    checkpoint_path = Path(checkpoint_path).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"checkpoint_path for model='{model}' does not exist: {checkpoint_path}"
        )

    if model == "hipt" and not checkpoint_path.is_dir():
        raise ValueError(
            "For model='hipt', checkpoint_path must be a directory containing "
            "vit256_small_dino.pth and vit4k_xs_dino.pth."
        )

    if model == "hipt":
        expected = [
            checkpoint_path / "vit256_small_dino.pth",
            checkpoint_path / "vit4k_xs_dino.pth",
        ]
        missing = [str(p) for p in expected if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "HIPT checkpoint directory is missing required file(s): "
                f"{missing}. Expected both vit256_small_dino.pth and vit4k_xs_dino.pth."
            )

    if model == "uni" and checkpoint_path.is_dir():
        raise ValueError(
            "For model='uni', checkpoint_path should usually be a checkpoint file, "
            "for example pytorch_model.bin, not a directory."
        )

    return str(checkpoint_path)


def _resolve_device(device: str) -> str:
    """Use CPU when CUDA was requested but is unavailable."""
    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must be a non-empty string.")

    device = device.strip()
    if not device.lower().startswith("cuda"):
        return device

    try:
        import torch
    except ImportError:
        return device

    if not torch.cuda.is_available():
        warnings.warn(
            "CUDA was requested but is unavailable; falling back to CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "cpu"

    if ":" in device:
        try:
            device_index = int(device.rsplit(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid CUDA device string: {device!r}.") from exc
        if device_index < 0 or device_index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device index {device_index} is unavailable; "
                f"detected {torch.cuda.device_count()} CUDA device(s)."
            )

    return device


def _validate_spot_coordinates_against_image(
    spot_coordinates: pd.DataFrame,
    image_path: Path,
) -> tuple[int, int]:
    """Validate that spot centers use the raw image's pixel coordinate system."""
    from PIL import Image

    with Image.open(image_path) as image:
        width, height = image.size

    invalid = (
        (spot_coordinates["pixel_x"] < 0)
        | (spot_coordinates["pixel_x"] >= width)
        | (spot_coordinates["pixel_y"] < 0)
        | (spot_coordinates["pixel_y"] >= height)
    )
    if invalid.any():
        examples = spot_coordinates.loc[invalid, ["pixel_x", "pixel_y"]].head(5)
        raise ValueError(
            "Spot centers must be in raw-image pixel coordinates within "
            f"x=[0, {width}) and y=[0, {height}). Invalid examples:\n{examples}"
        )

    return width, height


def extract_image_features(
    image_path,
    spot_coordinates_path=None,
    spot_coordinates=None,
    model: str = "uni",
    output_dir=None,
    sample_name: str | None = None,
    patch_size_spot: int = 280,
    patch_size_emb: int = 16,
    aggregation_method: str = "weighted",
    normalize_by: str = "overlap",
    ignore_zero_features: bool = False,
    zero_tol: float = 1e-8,
    npcs: int = 50,
    n_clusters: int | Sequence[int] | None = (5, 10),
    ncluster_list: int | Sequence[int] | None = None,
    plot_clusters: bool = True,
    plot_spot_size: int = 200,
    spot_x_key: str = "pixel_x",
    spot_y_key: str = "pixel_y",
    cat_color=None,
    dpi: int = 200,
    invert_x: bool = False,
    invert_y: bool = True,
    random_state: int = 42,
    device: str = "cuda",
    checkpoint_path=None,
    raw_image_name: str = "he-raw.jpg",
    overwrite_raw_image: bool = False,
    save_h5ad: bool = True,
    **kwargs,
):
    """
    Extract spot-level image features from a raw H&E image.

    Parameters
    ----------
    image_path : str or pathlib.Path
        Path to the raw H&E image.
    spot_coordinates_path : str or pathlib.Path, optional
        CSV file containing spot coordinates.
    spot_coordinates : pandas.DataFrame, optional
        Spot coordinate table. Must contain ``spot_x_key`` and ``spot_y_key``.
    model : {"hipt", "uni"}, default="uni"
        Image feature extraction backend.
    output_dir : str or pathlib.Path, optional
        Root directory for per-sample outputs. A subfolder named
        ``sample_name`` is created inside this directory.
    sample_name : str, optional
        Per-sample folder name. Defaults to the image file stem.
    patch_size_spot : int, default=280
        Spot window size in pixels.
    patch_size_emb : int, default=16
        Pixel size represented by each embedding-grid cell.
    aggregation_method : {"weighted", "mean", "median"}, default="weighted"
        Method used to aggregate grid embeddings to spot embeddings.
    normalize_by : {"overlap", "spot"}, default="overlap"
        Normalization rule for weighted aggregation.
    ignore_zero_features : bool, default=False
        Ignore all-zero grid cells during spot aggregation.
    zero_tol : float, default=1e-8
        Numerical tolerance used when ``ignore_zero_features=True``.
    npcs : int, default=50
        Number of principal components used for optional clustering.
    n_clusters : int, sequence of int, or None, default=(5, 10)
        KMeans cluster number(s). Ignored if ``ncluster_list`` is provided.
    ncluster_list : int, sequence of int, or None, optional
        Backward-compatible alias for ``n_clusters``.
    plot_clusters : bool, default=True
        Whether to run KMeans and save spatial cluster plots.
    plot_spot_size : int, default=200
        Marker size used in cluster visualizations.
    spot_x_key : str, default="pixel_x"
        Input column name for x pixel coordinate.
    spot_y_key : str, default="pixel_y"
        Input column name for y pixel coordinate.
    cat_color : list, tuple, or None
        Categorical color palette used for plotting. Each category will be assigned
        one color from this palette. If `None`, Scanpy will use its default
        categorical color palette.
    dpi : int, default=200
        Resolution of the saved figure.
    invert_x : bool, default=False
        Whether to invert the x-axis.
    invert_y : bool, default=True
        Whether to invert the y-axis. This is often useful for spatial transcriptomics
        or image-based coordinates where the image origin is in the upper-left corner.
    random_state : int, default=42
        Random seed used for PCA and KMeans visualization steps. It does not
        change the ordering of the extracted embedding dimensions.
    device : str, default="cuda"
        Device used by the image model wrapper.
    checkpoint_path : str or pathlib.Path, optional
        Path to pretrained checkpoint file or directory. For ``model='uni'``,
        this is usually the UNI checkpoint file, such as ``pytorch_model.bin``.
        For ``model='hipt'``, this is usually the HIPT checkpoint directory
        containing ``vit256_small_dino.pth`` and ``vit4k_xs_dino.pth``.
    raw_image_name : str, default="he-raw.jpg"
        Name used for the copied/converted image inside the sample folder.
    overwrite_raw_image : bool, default=False
        Whether to overwrite an existing copied raw image.
    save_h5ad : bool, default=True
        Whether to save ``results/spot_image_features.h5ad``.
    **kwargs
        Additional model-specific keyword arguments passed to the selected
        backend wrapper.

        HIPT-specific options include: ``mask``, ``pad_size``,
        ``reduction_method``, ``n_components``, ``smoothen_method``,
        ``random_weights``, ``no_shift``, and ``use_cache``.

        UNI-specific options include: ``scale_value``, ``pad_value``,
        ``mask_save_dir``, ``density_thresh``, ``clean_background_flag``,
        ``min_size``, ``batch_size``, ``stride``, ``num_workers``,
        ``spatial_key``, and ``spatial_coords_are_pixel``.

    Returns
    -------
    anndata.AnnData
        Spot-level image-feature object with the following structure:

        - ``.X``: dense array of shape ``(n_spots, n_image_features)``.
          Each row follows the input spot order.
        - ``.obs``: a copy of the supplied spot metadata. The coordinate
          columns selected by ``spot_x_key`` and ``spot_y_key`` are stored
          under the standardized names ``pixel_x`` and ``pixel_y``.
          If clustering is enabled, it also contains categorical columns such
          as ``kmeans_5`` and ``kmeans_10``.
        - ``.var_names``: zero-based feature identifiers ``uni_0``,
          ``uni_1``, ... or ``hipt_0``, ``hipt_1``, ....
        - ``.var["name"]``: copy of ``.var_names``. HIPT output additionally
          contains ``.var["source_name"]`` with the original feature identity,
          such as ``cls_0``, ``sub_0``, or ``rgb_0``.
        - ``.obsm["X_pca"]``: PCA scores computed from ``.X``. The number of
          components is at most ``min(npcs, n_spots, n_image_features)``.

        The object is returned in memory whether or not ``save_h5ad`` is
        enabled.

    Files created
    -------------
    Common final files are written below
    ``<output_dir>/<sample_name>/results/``:

    - ``spot_image_features.h5ad`` when ``save_h5ad=True``. This contains the
      same final AnnData structure described above.
    - ``spot_image_features_kmeans_<k>.png`` for each requested cluster number
      when ``plot_clusters=True`` and ``n_clusters`` (or ``ncluster_list``) is
      not ``None``.

    Model preprocessing also creates intermediate images, masks, and
    patch-level embeddings in ``<output_dir>/<sample_name>/``. See
    ``hicat_spatial/preprocessing/README.md`` for the complete UNI and HIPT file trees.
    
    """
    model = model.lower().strip()
    if model not in {"hipt", "uni"}:
        raise ValueError("model must be one of {'hipt', 'uni'}.")

    if patch_size_spot <= 0 or patch_size_emb <= 0:
        raise ValueError("patch_size_spot and patch_size_emb must be positive.")

    _validate_model_specific_kwargs(model, kwargs)
    checkpoint_path = _resolve_checkpoint_path(checkpoint_path, model=model)
    device = _resolve_device(device)

    spot_coordinates = _load_spot_coordinates(
        spot_coordinates_path=spot_coordinates_path,
        spot_coordinates=spot_coordinates,
        spot_x_key=spot_x_key,
        spot_y_key=spot_y_key,
    )

    sample_dir = _prepare_sample_dir(
        image_path=image_path,
        output_dir=output_dir,
        sample_name=sample_name,
        raw_image_name=raw_image_name,
        overwrite=overwrite_raw_image,
    )
    raw_image_size = _validate_spot_coordinates_against_image(
        spot_coordinates,
        sample_dir / raw_image_name,
    )

    backend_coordinates = spot_coordinates
    backend_patch_size_spot = patch_size_spot
    coordinate_scale = 1.0
    if model == "uni":
        coordinate_scale = float(kwargs.get("scale_value", 1.0))
        if not 0 < coordinate_scale <= 1.0:
            raise ValueError("UNI scale_value must be greater than 0 and at most 1.")
        if coordinate_scale != 1.0:
            backend_coordinates = spot_coordinates.copy()
            backend_coordinates[["pixel_x", "pixel_y"]] *= coordinate_scale
            backend_patch_size_spot = patch_size_spot * coordinate_scale

    clusters = ncluster_list if ncluster_list is not None else n_clusters

    common_kwargs = dict(
        sample=str(sample_dir),
        spot_coordinates=backend_coordinates,
        checkpoint_path=checkpoint_path,
        patch_size_spot=backend_patch_size_spot,
        patch_size_emb=patch_size_emb,
        aggregation_method=aggregation_method,
        normalize_by=normalize_by,
        ignore_zero_features=ignore_zero_features,
        zero_tol=zero_tol,
        device=device,
        npcs=npcs,
        ncluster_list=clusters,
        plot_clusters=plot_clusters,
        plot_spot_size=plot_spot_size,
        cat_color=cat_color,
        dpi=dpi,
        invert_x=invert_x,
        invert_y=invert_y,
        save_h5ad=save_h5ad,
        random_state=random_state,
    )
    common_kwargs.update(kwargs)

    if model == "hipt":
        from .hipt.wrapper import extract_spatial_image_features as _extract
    else:
        from .uni.wrapper import extract_spatial_image_features as _extract
        # UNI resolves relative mask directories inside ``sample_dir``.
        common_kwargs.setdefault("mask_save_dir", "mask")

    result = _extract(**common_kwargs)

    if result.n_obs != len(spot_coordinates):
        raise RuntimeError(
            "Image backend returned a different number of observations than the "
            "input spot table."
        )

    # UNI may operate on a scaled image. Restore raw-image coordinates in the
    # public result after aggregation so downstream modalities remain aligned.
    result.obs["pixel_x"] = spot_coordinates["pixel_x"].to_numpy()
    result.obs["pixel_y"] = spot_coordinates["pixel_y"].to_numpy()
    result.uns["image_coordinate_system"] = {
        "raw_image_width": int(raw_image_size[0]),
        "raw_image_height": int(raw_image_size[1]),
        "backend_coordinate_scale": float(coordinate_scale),
        "boundary_normalization": normalize_by,
    }

    if save_h5ad:
        results_dir = sample_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        result.write_h5ad(results_dir / "spot_image_features.h5ad")

    return result
