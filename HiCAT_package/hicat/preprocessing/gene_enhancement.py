"""H&E contour detection and spatial gene-expression enhancement.

The high-level workflow detects the tissue boundary, removes image background,
and interpolates selected genes onto a dense pseudo-spot grid.

Example
-------
::

    from hicat.preprocessing import enhance_gene_expression

    result = enhance_gene_expression(
        image=img,
        raw_adata=adata_upd,
        genes=adata_upd.var_names,
        contour_method="auto",
        x_key="pixel_x",
        y_key="pixel_y",
        resolution=50,
        n_neighbors=10,
        qc_path="outputs/section_contour.jpg",
    )

    enhanced_adata = result.enhanced_adata
    contour = result.contour_result.contour
    result.contour_result.candidate_scores
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict

import anndata as ad
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


@dataclass
class TissueContourResult:
    """Automatic H&E tissue-contour detection output.

    Attributes
    ----------
    contour
        Selected full-resolution OpenCV contour with shape ``(n, 1, 2)``.
        It can be passed directly to :func:`impute_gene_expression`.
    mask
        Full-resolution binary tissue mask with values 0 and 1.
    image_no_background
        Copy of the input image with pixels outside ``mask`` replaced.
    enlarged_contour, enlarged_mask
        Contour and mask after applying ``contour_scale``.
    selected_method
        Selected candidate: ``color``, ``cv2``, ``scan_x``, ``scan_y``, or
        ``spot_hull``.
    candidate_scores
        Candidate metrics sorted by the automatic selection score.
    qc_image
        Resized image with the selected contour drawn for quick inspection.
    params
        Resizing and detection parameters used by the workflow.
    """

    contour: np.ndarray
    mask: np.ndarray
    image_no_background: np.ndarray
    enlarged_contour: np.ndarray
    enlarged_mask: np.ndarray
    selected_method: str
    candidate_scores: pd.DataFrame
    qc_image: np.ndarray
    params: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """Return the candidate metrics used for method selection."""
        return self.candidate_scores.copy()


def _as_uint8_image(image):
    """Validate an image and convert it to uint8 without changing its shape."""
    image = np.asarray(image)
    if image.ndim not in {2, 3}:
        raise ValueError("image must be a 2D grayscale or 3D color array.")
    if image.ndim == 3 and image.shape[2] not in {1, 3, 4}:
        raise ValueError("A color image must have 1, 3, or 4 channels.")
    if image.size == 0:
        raise ValueError("image cannot be empty.")

    if image.ndim == 3 and image.shape[2] == 4:
        image = image[..., :3]
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[..., 0]

    if image.dtype == np.uint8:
        return image.copy()

    image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
    if np.issubdtype(image.dtype, np.floating) and image.max() <= 1:
        image = image * 255
    return np.clip(image, 0, 255).astype(np.uint8)


def _gray_and_saturation(image, color_order="rgb"):
    """Return grayscale and HSV saturation channels from an uint8 image."""
    if color_order not in {"rgb", "bgr"}:
        raise ValueError("color_order must be either 'rgb' or 'bgr'.")

    if image.ndim == 2:
        return image, np.zeros_like(image)

    if color_order == "rgb":
        gray_code = cv2.COLOR_RGB2GRAY
        hsv_code = cv2.COLOR_RGB2HSV
    else:
        gray_code = cv2.COLOR_BGR2GRAY
        hsv_code = cv2.COLOR_BGR2HSV

    gray = cv2.cvtColor(image, gray_code)
    saturation = cv2.cvtColor(image, hsv_code)[..., 1]
    return gray, saturation


def _clean_tissue_mask(
    mask,
    min_component_fraction=0.001,
    morph_kernel_size=None,
):
    """Close gaps, fill holes, and remove small foreground components."""
    if not 0 <= min_component_fraction < 1:
        raise ValueError("min_component_fraction must be in [0, 1).")

    mask = (np.asarray(mask) > 0).astype(np.uint8)
    if morph_kernel_size is None:
        morph_kernel_size = max(3, int(round(min(mask.shape) * 0.005)))
    morph_kernel_size = max(1, int(morph_kernel_size))
    if morph_kernel_size % 2 == 0:
        morph_kernel_size += 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (morph_kernel_size, morph_kernel_size),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = binary_fill_holes(mask > 0).astype(np.uint8)

    component_num, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )
    if component_num <= 1:
        return mask

    minimum_area = max(1, int(round(mask.size * min_component_fraction)))
    component_areas = stats[1:, cv2.CC_STAT_AREA]
    keep_labels = np.flatnonzero(component_areas >= minimum_area) + 1
    if len(keep_labels) == 0:
        keep_labels = np.array([int(np.argmax(component_areas)) + 1])
    return np.isin(labels, keep_labels).astype(np.uint8)


def detect_he_tissue_mask(
    image,
    color_order="rgb",
    min_component_fraction=0.001,
    morph_kernel_size=None,
):
    """Detect H&E tissue from color saturation and distance from white.

    This lightweight mask follows the same general idea as the HIPT/UNI
    preprocessing: work on image-derived foreground evidence, clean small
    components, close gaps, and fill holes. The returned mask has the same
    height and width as ``image``.

    Parameters
    ----------
    image : numpy.ndarray
        RGB/BGR or grayscale H&E image.
    color_order : {"rgb", "bgr"}
        Channel order for a color image. PIL images are normally RGB and
        ``cv2.imread`` images are BGR.
    min_component_fraction : float
        Components smaller than this fraction of the image are removed.
    morph_kernel_size : int, optional
        Morphological kernel width. By default it scales with image size.

    Returns
    -------
    numpy.ndarray
        Binary uint8 mask with tissue equal to 1.
    """
    image_u8 = _as_uint8_image(image)
    gray, saturation = _gray_and_saturation(image_u8, color_order=color_order)

    darkness = 255 - gray
    tissue_evidence = np.maximum(darkness, saturation)
    tissue_evidence = cv2.GaussianBlur(tissue_evidence, (5, 5), 0)
    _, mask = cv2.threshold(
        tissue_evidence,
        0,
        1,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return _clean_tissue_mask(
        mask,
        min_component_fraction=min_component_fraction,
        morph_kernel_size=morph_kernel_size,
    )


def contour_to_mask(contour, image_shape):
    """Convert one OpenCV contour to a binary mask."""
    if contour is None:
        raise ValueError("contour cannot be None.")
    height, width = tuple(image_shape)[:2]
    if height < 1 or width < 1:
        raise ValueError("image_shape must contain positive height and width.")

    contour = np.asarray(contour)
    if contour.ndim == 2 and contour.shape[1] == 2:
        contour = contour[:, None, :]
    if contour.ndim != 3 or contour.shape[1:] != (1, 2):
        raise ValueError("contour must have shape (n, 1, 2) or (n, 2).")
    if len(contour) < 3:
        raise ValueError("contour must contain at least three points.")

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(mask, [contour.astype(np.int32)], -1, 1, thickness=-1)
    return mask


def _mask_to_contour(mask, strategy="largest"):
    """Construct one contour from a binary mask."""
    contours, _ = cv2.findContours(
        (np.asarray(mask) > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    contours = [contour for contour in contours if cv2.contourArea(contour) > 0]
    if len(contours) == 0:
        raise ValueError("No non-empty tissue contour was detected.")

    if strategy == "largest":
        return max(contours, key=cv2.contourArea).astype(np.int32)
    if strategy == "convex_hull":
        return cv2.convexHull(np.vstack(contours)).astype(np.int32)
    raise ValueError("strategy must be either 'largest' or 'convex_hull'.")


def scale_tissue_contour(contour, scale=1.05, image_shape=None):
    """Scale an OpenCV contour around its centroid and optionally clip it."""
    if scale <= 0:
        raise ValueError("scale must be positive.")
    contour = np.asarray(contour, dtype=np.float64)
    if contour.ndim == 2 and contour.shape[1] == 2:
        contour = contour[:, None, :]
    if contour.ndim != 3 or contour.shape[1:] != (1, 2):
        raise ValueError("contour must have shape (n, 1, 2) or (n, 2).")

    moments = cv2.moments(contour.astype(np.float32))
    if moments["m00"] == 0:
        center = contour[:, 0, :].mean(axis=0)
    else:
        center = np.array(
            [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]
        )
    scaled = (contour - center) * float(scale) + center

    if image_shape is not None:
        height, width = tuple(image_shape)[:2]
        scaled[..., 0] = np.clip(scaled[..., 0], 0, width - 1)
        scaled[..., 1] = np.clip(scaled[..., 1], 0, height - 1)
    return np.rint(scaled).astype(np.int32)


def remove_image_background(image, tissue_mask, background_value=0, copy=True):
    """Replace pixels outside a tissue mask with a constant background."""
    image_out = np.asarray(image).copy() if copy else np.asarray(image)
    tissue_mask = np.asarray(tissue_mask) > 0
    if tissue_mask.shape != image_out.shape[:2]:
        raise ValueError("tissue_mask must match the image height and width.")
    image_out[~tissue_mask] = background_value
    return image_out


def _get_spot_dataframe(spots):
    """Accept an AnnData object or a spot-metadata DataFrame."""
    if spots is None:
        return None
    if hasattr(spots, "obs"):
        spots = spots.obs
    if not isinstance(spots, pd.DataFrame):
        raise TypeError("spots must be an AnnData object or pandas DataFrame.")
    return spots.copy()


def scan_spot_contour(
    spots,
    scan_axis="x",
    x_key="pixel_x",
    y_key="pixel_y",
    array_x_key="array_x",
    array_y_key="array_y",
    scale=1.0,
    image_shape=None,
):
    """Construct a boundary by scanning spatial-array rows or columns.

    ``scan_axis="x"`` joins the minimum and maximum y coordinate within each
    array-x group. ``scan_axis="y"`` performs the complementary scan. This is
    the general equivalent of manually choosing ``scan_x=True/False``.
    """
    spots = _get_spot_dataframe(spots)
    if scan_axis not in {"x", "y"}:
        raise ValueError("scan_axis must be either 'x' or 'y'.")

    required = [x_key, y_key]
    missing = [column for column in required if column not in spots]
    if missing:
        raise KeyError(f"spots is missing coordinate columns: {missing}.")

    group_key = array_x_key if scan_axis == "x" else array_y_key
    if group_key not in spots:
        raise KeyError(
            f"{group_key!r} is required for scan_axis={scan_axis!r}; "
            "use the spot_hull candidate when array coordinates are unavailable."
        )

    columns = [x_key, y_key, group_key]
    work = spots.loc[:, columns].apply(pd.to_numeric, errors="coerce").dropna()
    if len(work) < 3:
        raise ValueError("At least three finite spots are required.")

    first_side = []
    second_side = []
    for _, group in work.groupby(group_key, sort=True):
        if scan_axis == "x":
            axis_value = float(group[x_key].median())
            first_side.append([axis_value, float(group[y_key].min())])
            second_side.append([axis_value, float(group[y_key].max())])
        else:
            axis_value = float(group[y_key].median())
            first_side.append([float(group[x_key].min()), axis_value])
            second_side.append([float(group[x_key].max()), axis_value])

    points = np.rint(
        np.asarray(first_side + second_side[::-1], dtype=np.float64)
    ).astype(np.int32)
    _, unique_indices = np.unique(points, axis=0, return_index=True)
    points = points[np.sort(unique_indices)]
    if len(points) < 3:
        raise ValueError(f"The {scan_axis}-scan produced fewer than three points.")

    contour = points[:, None, :]
    if scale != 1:
        contour = scale_tissue_contour(
            contour,
            scale=scale,
            image_shape=image_shape,
        )
    return contour.astype(np.int32)


def _cv2_edge_contour(image, color_order="rgb", aperture_size=5):
    """Detect the largest Canny edge contour using automatic thresholds."""
    if aperture_size not in {3, 5, 7}:
        raise ValueError("aperture_size must be 3, 5, or 7.")
    gray, _ = _gray_and_saturation(image, color_order=color_order)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(gray))
    lower = int(max(0, 0.67 * median))
    upper = int(min(255, max(lower + 1, 1.33 * median)))
    edges = cv2.Canny(
        gray,
        lower,
        upper,
        apertureSize=aperture_size,
        L2gradient=True,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [contour for contour in contours if cv2.contourArea(contour) > 0]
    if len(contours) == 0:
        raise ValueError("OpenCV Canny detection did not find a contour.")
    return max(contours, key=cv2.contourArea).astype(np.int32)


def _candidate_metrics(mask, contour, image_mask, spot_points):
    """Score a candidate against image foreground and spatial spots."""
    mask_bool = np.asarray(mask) > 0
    image_bool = np.asarray(image_mask) > 0
    intersection = int(np.logical_and(mask_bool, image_bool).sum())
    union = int(np.logical_or(mask_bool, image_bool).sum())
    candidate_area = int(mask_bool.sum())
    image_area = int(image_bool.sum())

    image_iou = intersection / max(union, 1)
    image_precision = intersection / max(candidate_area, 1)
    image_recall = intersection / max(image_area, 1)
    area_fraction = candidate_area / mask_bool.size

    perimeter = float(cv2.arcLength(contour, closed=True))
    compactness = (
        min(1.0, 4 * np.pi * candidate_area / max(perimeter ** 2, 1.0))
    )

    if spot_points is None or len(spot_points) == 0:
        spot_coverage = np.nan
        score = 0.60 * image_iou + 0.25 * image_precision + 0.15 * compactness
    else:
        x = np.rint(spot_points[:, 0]).astype(int)
        y = np.rint(spot_points[:, 1]).astype(int)
        valid = (
            (x >= 0)
            & (x < mask_bool.shape[1])
            & (y >= 0)
            & (y < mask_bool.shape[0])
        )
        if not valid.any():
            spot_coverage = 0.0
        else:
            spot_coverage = float(mask_bool[y[valid], x[valid]].mean())
        score = (
            0.45 * spot_coverage
            + 0.25 * image_iou
            + 0.15 * image_precision
            + 0.10 * image_recall
            + 0.05 * compactness
        )

    return {
        "score": float(score),
        "spot_coverage": float(spot_coverage),
        "image_iou": float(image_iou),
        "image_precision": float(image_precision),
        "image_recall": float(image_recall),
        "area_fraction": float(area_fraction),
        "compactness": float(compactness),
    }


def detect_he_tissue_contour(
    image,
    spots=None,
    method="auto",
    x_key="pixel_x",
    y_key="pixel_y",
    array_x_key="array_x",
    array_y_key="array_y",
    color_order="rgb",
    target_min_side=1000,
    min_component_fraction=0.001,
    morph_kernel_size=None,
    image_contour_strategy="largest",
    aperture_size=5,
    contour_scale=1.05,
    background_value=0,
    qc_path=None,
    print_results=True,
):
    """Automatically select a tissue contour and remove H&E background.

    Candidate contours are created from color foreground, OpenCV Canny edges,
    x/y spot scans, and the spot convex hull. ``method="auto"`` scores every
    available candidate by H&E-mask agreement, spot coverage, and contour
    compactness, avoiding a manual scan-x versus scan-y decision.

    Parameters
    ----------
    image : numpy.ndarray
        Full-resolution H&E image.
    spots : AnnData or pandas.DataFrame, optional
        Spatial observations. Pixel columns are required for spot scoring;
        array columns additionally enable x/y scan candidates.
    method : {"auto", "color", "cv2", "scan_x", "scan_y", "spot_hull"}
        Automatic selection or a specific candidate.
    x_key, y_key : str
        Conventional image x/column and y/row coordinate columns.
    array_x_key, array_y_key : str
        Spatial-grid columns used by scan candidates.
    color_order : {"rgb", "bgr"}
        Image channel order.
    target_min_side : int
        Detection resolution. Large images are downsampled so their shorter
        side is at most this value; small images are not enlarged.
    min_component_fraction, morph_kernel_size
        Color-mask cleanup settings.
    image_contour_strategy : {"largest", "convex_hull"}
        How multiple image-mask components are converted to one contour.
    aperture_size : {3, 5, 7}
        Canny aperture used by the ``cv2`` candidate.
    contour_scale : float
        Scale used for ``enlarged_contour``; 1.05 adds a 5% margin.
    background_value : scalar or sequence
        Value assigned outside the selected tissue contour.
    qc_path : path-like, optional
        If provided, save the resized contour overlay.
    print_results : bool
        Print the selected method and candidate metrics.

    Returns
    -------
    TissueContourResult
        Full-resolution contour, masks, background-removed image, QC overlay,
        and candidate scores. Use ``result.contour`` for local enhancement::

            result = detect_he_tissue_contour(img, spots=adata)
            enhanced = impute_gene_expression(
                image=img,
                raw_adata=adata,
                contour=result.contour,
            )
    """
    valid_methods = {"auto", "color", "cv2", "scan_x", "scan_y", "spot_hull"}
    if method not in valid_methods:
        raise ValueError(f"method must be one of {sorted(valid_methods)}.")
    if image_contour_strategy not in {"largest", "convex_hull"}:
        raise ValueError(
            "image_contour_strategy must be 'largest' or 'convex_hull'."
        )
    if target_min_side < 50:
        raise ValueError("target_min_side must be at least 50.")

    image_u8 = _as_uint8_image(image)
    original_height, original_width = image_u8.shape[:2]
    resize_factor = min(1.0, target_min_side / min(original_height, original_width))
    resized_width = max(1, int(round(original_width * resize_factor)))
    resized_height = max(1, int(round(original_height * resize_factor)))
    if (resized_height, resized_width) == (original_height, original_width):
        image_small = image_u8.copy()
    else:
        image_small = cv2.resize(
            image_u8,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )

    image_mask = detect_he_tissue_mask(
        image_small,
        color_order=color_order,
        min_component_fraction=min_component_fraction,
        morph_kernel_size=morph_kernel_size,
    )

    candidates = {}
    candidate_errors = {}
    try:
        candidates["color"] = _mask_to_contour(
            image_mask,
            strategy=image_contour_strategy,
        )
    except ValueError as error:
        candidate_errors["color"] = str(error)
    try:
        candidates["cv2"] = _cv2_edge_contour(
            image_small,
            color_order=color_order,
            aperture_size=aperture_size,
        )
    except ValueError as error:
        candidate_errors["cv2"] = str(error)

    spot_df = _get_spot_dataframe(spots)
    spot_points = None
    if spot_df is not None:
        missing_pixel_columns = [
            key for key in (x_key, y_key) if key not in spot_df
        ]
        if missing_pixel_columns:
            raise KeyError(
                f"spots is missing pixel coordinate columns: {missing_pixel_columns}."
            )
        small_spots = spot_df.copy()
        small_spots[x_key] = pd.to_numeric(
            small_spots[x_key], errors="coerce"
        ) * (resized_width / original_width)
        small_spots[y_key] = pd.to_numeric(
            small_spots[y_key], errors="coerce"
        ) * (resized_height / original_height)
        spot_points = small_spots[[x_key, y_key]].dropna().to_numpy(dtype=float)

        for candidate_method, scan_axis in (("scan_x", "x"), ("scan_y", "y")):
            try:
                candidates[candidate_method] = scan_spot_contour(
                    small_spots,
                    scan_axis=scan_axis,
                    x_key=x_key,
                    y_key=y_key,
                    array_x_key=array_x_key,
                    array_y_key=array_y_key,
                    image_shape=image_small.shape,
                )
            except (KeyError, ValueError) as error:
                candidate_errors[candidate_method] = str(error)

        if spot_points is not None and len(spot_points) >= 3:
            hull_points = np.rint(spot_points).astype(np.int32)[:, None, :]
            spot_hull = cv2.convexHull(hull_points)
            if cv2.contourArea(spot_hull) > 0:
                candidates["spot_hull"] = spot_hull
            else:
                candidate_errors["spot_hull"] = "Spot coordinates are collinear."

    if len(candidates) == 0:
        raise ValueError(
            "No valid contour candidates were generated. "
            f"Candidate errors: {candidate_errors}"
        )
    if method != "auto" and method not in candidates:
        reason = candidate_errors.get(method, "required spot data were not provided")
        raise ValueError(f"Contour method {method!r} is unavailable: {reason}.")

    candidate_masks = {
        name: contour_to_mask(contour, image_small.shape)
        for name, contour in candidates.items()
    }
    metrics = {
        name: _candidate_metrics(
            mask=candidate_masks[name],
            contour=contour,
            image_mask=image_mask,
            spot_points=spot_points,
        )
        for name, contour in candidates.items()
    }
    scores = pd.DataFrame.from_dict(metrics, orient="index")
    scores.index.name = "method"
    scores = scores.sort_values("score", ascending=False)
    selected_method = scores.index[0] if method == "auto" else method
    selected_small_contour = candidates[selected_method]

    x_scale = original_width / resized_width
    y_scale = original_height / resized_height
    contour = selected_small_contour.astype(np.float64)
    contour[..., 0] *= x_scale
    contour[..., 1] *= y_scale
    contour[..., 0] = np.clip(contour[..., 0], 0, original_width - 1)
    contour[..., 1] = np.clip(contour[..., 1], 0, original_height - 1)
    contour = np.rint(contour).astype(np.int32)

    mask = contour_to_mask(contour, image_u8.shape)
    enlarged_contour = scale_tissue_contour(
        contour,
        scale=contour_scale,
        image_shape=image_u8.shape,
    )
    enlarged_mask = contour_to_mask(enlarged_contour, image_u8.shape)
    image_no_background = remove_image_background(
        image_u8,
        mask,
        background_value=background_value,
        copy=True,
    )

    qc_image = image_small.copy()
    if qc_image.ndim == 2:
        qc_image = cv2.cvtColor(qc_image, cv2.COLOR_GRAY2BGR)
    qc_thickness = max(1, int(round(min(qc_image.shape[:2]) / 250)))
    cv2.drawContours(
        qc_image,
        [selected_small_contour],
        -1,
        (255, 255, 255),
        thickness=qc_thickness,
    )
    if qc_path is not None:
        qc_path = os.fspath(qc_path)
        qc_parent = os.path.dirname(qc_path)
        if qc_parent:
            os.makedirs(qc_parent, exist_ok=True)
        qc_image_to_save = qc_image
        if color_order == "rgb" and qc_image.ndim == 3:
            qc_image_to_save = cv2.cvtColor(qc_image, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(qc_path, qc_image_to_save):
            raise OSError(f"Failed to save contour QC image: {qc_path}")

    if print_results:
        print(f"Selected H&E contour method: {selected_method}")
        print(scores.round(3))

    return TissueContourResult(
        contour=contour,
        mask=mask,
        image_no_background=image_no_background,
        enlarged_contour=enlarged_contour,
        enlarged_mask=enlarged_mask,
        selected_method=selected_method,
        candidate_scores=scores,
        qc_image=qc_image,
        params={
            "method": method,
            "resize_factor": resize_factor,
            "resized_shape": (resized_height, resized_width),
            "color_order": color_order,
            "target_min_side": target_min_side,
            "min_component_fraction": min_component_fraction,
            "morph_kernel_size": morph_kernel_size,
            "image_contour_strategy": image_contour_strategy,
            "aperture_size": aperture_size,
            "contour_scale": contour_scale,
            "candidate_errors": candidate_errors,
        },
    )


@dataclass
class GeneEnhancementResult:
    """Combined contour-detection and gene-enhancement output.

    Attributes
    ----------
    enhanced_adata
        Dense-grid AnnData containing imputed expression for selected genes.
    contour_result
        Tissue contour, masks, background-removed image, and QC information.
    params
        High-level enhancement parameters.
    """

    enhanced_adata: ad.AnnData
    contour_result: TissueContourResult
    params: Dict[str, Any] = field(default_factory=dict)


def _ordered_genes(raw_adata, genes, strict=True):
    """Validate genes and preserve the requested order."""
    if genes is None:
        selected = raw_adata.var_names.tolist()
    else:
        selected = list(dict.fromkeys(genes))

    if len(selected) == 0:
        raise ValueError("At least one gene must be selected.")
    missing = [gene for gene in selected if gene not in raw_adata.var_names]
    if missing and strict:
        raise KeyError(
            f"{len(missing)} selected genes are absent from raw_adata. "
            f"Examples: {missing[:5]}"
        )
    selected = [gene for gene in selected if gene in raw_adata.var_names]
    if len(selected) == 0:
        raise ValueError("None of the selected genes are present in raw_adata.")
    return selected, missing


def _known_spot_coordinates(raw_adata, x_key, y_key, image_shape):
    """Return validated spot coordinates in conventional OpenCV x/y order."""
    missing = [key for key in (x_key, y_key) if key not in raw_adata.obs]
    if missing:
        raise KeyError(f"raw_adata.obs is missing coordinate columns: {missing}.")

    coordinates = raw_adata.obs[[x_key, y_key]].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)
    finite = np.isfinite(coordinates).all(axis=1)
    if not finite.all():
        raise ValueError(
            f"{int((~finite).sum())} observations have non-finite coordinates."
        )

    height, width = tuple(image_shape)[:2]
    in_bounds = (
        (coordinates[:, 0] >= 0)
        & (coordinates[:, 0] < width)
        & (coordinates[:, 1] >= 0)
        & (coordinates[:, 1] < height)
    )
    if not in_bounds.all():
        raise ValueError(
            f"{int((~in_bounds).sum())} spot coordinates lie outside the image."
        )
    return coordinates


def _pseudo_spot_coordinates(mask, resolution, max_pseudo_spots):
    """Generate regularly spaced x/y grid centers inside a binary mask."""
    if resolution <= 0:
        raise ValueError("resolution must be positive.")
    height, width = mask.shape[:2]
    offset = float(resolution) / 2
    x_values = np.unique(
        np.clip(np.rint(np.arange(offset, width, resolution)), 0, width - 1)
        .astype(int)
    )
    y_values = np.unique(
        np.clip(np.rint(np.arange(offset, height, resolution)), 0, height - 1)
        .astype(int)
    )
    grid_x, grid_y = np.meshgrid(x_values, y_values, indexing="xy")
    coordinates = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    inside = mask[coordinates[:, 1], coordinates[:, 0]] > 0
    coordinates = coordinates[inside]

    if len(coordinates) == 0:
        raise ValueError(
            "No pseudo spots fall inside the contour; reduce resolution or "
            "check the contour coordinates."
        )
    if max_pseudo_spots is not None and len(coordinates) > max_pseudo_spots:
        raise ValueError(
            f"The grid contains {len(coordinates)} pseudo spots, exceeding "
            f"max_pseudo_spots={max_pseudo_spots}. Increase resolution or the limit."
        )
    return coordinates.astype(float)


def _local_channel_means(image, coordinates, patch_size):
    """Compute local channel means at x/y coordinates using integral images."""
    patch_size = max(1, int(round(patch_size)))
    image = _as_uint8_image(image)
    if image.ndim == 2:
        image = image[..., None]

    x = np.rint(coordinates[:, 0]).astype(int)
    y = np.rint(coordinates[:, 1]).astype(int)
    half = patch_size // 2
    x0 = np.clip(x - half, 0, image.shape[1])
    y0 = np.clip(y - half, 0, image.shape[0])
    x1 = np.clip(x0 + patch_size, 0, image.shape[1])
    y1 = np.clip(y0 + patch_size, 0, image.shape[0])
    x0 = np.maximum(0, x1 - patch_size)
    y0 = np.maximum(0, y1 - patch_size)
    areas = np.maximum((x1 - x0) * (y1 - y0), 1)

    means = np.empty((len(coordinates), image.shape[2]), dtype=np.float64)
    for channel in range(image.shape[2]):
        integral = cv2.integral(image[..., channel], sdepth=cv2.CV_64F)
        sums = (
            integral[y1, x1]
            - integral[y0, x1]
            - integral[y1, x0]
            + integral[y0, x0]
        )
        means[:, channel] = sums / areas
    return means


def _histology_signatures(pseudo_means, known_means):
    """Project local channel means using data-driven channel variances."""
    combined = np.vstack([pseudo_means, known_means])
    channel_variances = np.var(combined, axis=0)
    variance_sum = float(channel_variances.sum())
    if variance_sum <= np.finfo(float).eps:
        channel_weights = np.full(
            combined.shape[1],
            1 / combined.shape[1],
            dtype=float,
        )
    else:
        channel_weights = channel_variances / variance_sum
    return (
        pseudo_means @ channel_weights,
        known_means @ channel_weights,
        channel_weights,
    )


def _neighbor_weights(
    distances,
    neighbor_indices,
    known_count,
    weighting,
    distance_power,
    distance_epsilon,
    distance_threshold,
):
    """Construct a sparse pseudo-to-known interpolation matrix."""
    valid = distances <= distance_threshold
    if weighting == "inverse_distance":
        raw_weights = 1 / np.power(distances + distance_epsilon, distance_power)
    elif weighting == "exponential":
        positive = distances[distances > 0]
        scale = float(np.median(positive)) if positive.size else 1.0
        raw_weights = np.exp(-distances / max(scale, distance_epsilon))
    elif weighting == "uniform":
        raw_weights = np.ones_like(distances, dtype=float)
    else:
        raise ValueError(
            "weighting must be 'inverse_distance', 'exponential', or 'uniform'."
        )

    raw_weights[~valid] = 0
    row_sums = raw_weights.sum(axis=1, keepdims=True)
    weights = np.divide(
        raw_weights,
        row_sums,
        out=np.zeros_like(raw_weights, dtype=float),
        where=row_sums > 0,
    )

    row_indices = np.repeat(np.arange(weights.shape[0]), weights.shape[1])
    col_indices = neighbor_indices.ravel()
    values = weights.ravel()
    nonzero = values > 0
    matrix = csr_matrix(
        (values[nonzero], (row_indices[nonzero], col_indices[nonzero])),
        shape=(weights.shape[0], int(known_count)),
    )
    return matrix, valid, weights


def impute_gene_expression(
    image,
    raw_adata,
    contour,
    genes=None,
    resolution=50,
    histology_scale=1.0,
    n_neighbors=10,
    distance_power=2.0,
    weighting="inverse_distance",
    neighbor_distance_quantile=0.95,
    max_neighbor_distance=None,
    contour_scale=1.05,
    color_patch_size=None,
    x_key="pixel_x",
    y_key="pixel_y",
    strict_genes=True,
    max_pseudo_spots=250_000,
    n_jobs=None,
    print_results=True,
):
    """Interpolate spot expression onto a dense histology-aware spatial grid.

    This standalone implementation queries only the requested nearest neighbors
    instead of constructing a pseudo-spot-by-known-spot dense distance matrix.
    Coordinates consistently follow OpenCV convention: x is the image column
    and y is the image row.

    Parameters
    ----------
    image : numpy.ndarray
        Full-resolution H&E image.
    raw_adata : AnnData
        Observed spot-level expression with pixel coordinates in ``.obs``.
    contour : numpy.ndarray
        OpenCV tissue contour with shape ``(n, 1, 2)``.
    genes : sequence, optional
        Genes to enhance, in the desired output order. Defaults to all genes.
    resolution : float
        Pixel spacing between pseudo spots.
    histology_scale : float
        Strength of the local-color coordinate. Set to zero for spatial-only
        interpolation.
    n_neighbors : int
        Maximum known spots used for each pseudo spot.
    distance_power : float
        Inverse-distance exponent when ``weighting="inverse_distance"``.
    weighting : {"inverse_distance", "exponential", "uniform"}
        Neighbor weighting scheme.
    neighbor_distance_quantile : float or None
        Automatic cutoff based on the selected-neighbor distances. ``None``
        disables the automatic cutoff.
    max_neighbor_distance : float, optional
        Explicit cutoff overriding ``neighbor_distance_quantile``.
    contour_scale : float
        Margin applied before generating the pseudo-spot grid.
    color_patch_size : int, optional
        Local H&E window width; defaults to ``resolution``.
    x_key, y_key : str
        Pixel-coordinate columns in ``raw_adata.obs``.
    strict_genes : bool
        Raise for missing requested genes. If false, silently omit them.
    max_pseudo_spots : int or None
        Safety limit for dense-grid size.
    n_jobs : int, optional
        Parallel workers passed to ``NearestNeighbors``.
    print_results : bool
        Print grid and interpolation summaries.

    Returns
    -------
    AnnData
        Pseudo-spot expression with pixel coordinates in ``.obs`` and
        ``obsm["spatial"]``.
    """
    image_u8 = _as_uint8_image(image)
    if not hasattr(raw_adata, "obs") or not hasattr(raw_adata, "var_names"):
        raise TypeError("raw_adata must be an AnnData-like object.")
    if raw_adata.n_obs == 0:
        raise ValueError("raw_adata must contain at least one observation.")
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be at least 1.")
    if distance_power <= 0:
        raise ValueError("distance_power must be positive.")
    if histology_scale < 0:
        raise ValueError("histology_scale cannot be negative.")
    if neighbor_distance_quantile is not None and not (
        0 < neighbor_distance_quantile <= 1
    ):
        raise ValueError("neighbor_distance_quantile must be in (0, 1].")

    selected_genes, missing_genes = _ordered_genes(
        raw_adata,
        genes,
        strict=strict_genes,
    )
    known_adata = raw_adata[:, selected_genes].copy()
    known_coordinates = _known_spot_coordinates(
        known_adata,
        x_key=x_key,
        y_key=y_key,
        image_shape=image_u8.shape,
    )

    enlarged_contour = scale_tissue_contour(
        contour,
        scale=contour_scale,
        image_shape=image_u8.shape,
    )
    enlarged_mask = contour_to_mask(enlarged_contour, image_u8.shape)
    pseudo_coordinates = _pseudo_spot_coordinates(
        enlarged_mask,
        resolution=resolution,
        max_pseudo_spots=max_pseudo_spots,
    )

    if color_patch_size is None:
        color_patch_size = max(1, int(round(resolution)))
    pseudo_means = _local_channel_means(
        image_u8,
        pseudo_coordinates,
        patch_size=color_patch_size,
    )
    known_means = _local_channel_means(
        image_u8,
        known_coordinates,
        patch_size=color_patch_size,
    )
    pseudo_color, known_color, channel_weights = _histology_signatures(
        pseudo_means,
        known_means,
    )

    combined_color = np.concatenate([pseudo_color, known_color])
    color_std = float(np.std(combined_color))
    if color_std <= np.finfo(float).eps or histology_scale == 0:
        pseudo_z = np.zeros_like(pseudo_color)
        known_z = np.zeros_like(known_color)
    else:
        color_mean = float(np.mean(combined_color))
        all_coordinates = np.vstack([pseudo_coordinates, known_coordinates])
        spatial_scale = max(
            float(np.std(all_coordinates[:, 0])),
            float(np.std(all_coordinates[:, 1])),
        ) * histology_scale
        pseudo_z = (pseudo_color - color_mean) / color_std * spatial_scale
        known_z = (known_color - color_mean) / color_std * spatial_scale

    pseudo_features = np.column_stack([pseudo_coordinates, pseudo_z])
    known_features = np.column_stack([known_coordinates, known_z])
    effective_neighbors = min(int(n_neighbors), known_adata.n_obs)
    neighbor_model = NearestNeighbors(
        n_neighbors=effective_neighbors,
        metric="euclidean",
        n_jobs=n_jobs,
    )
    neighbor_model.fit(known_features)
    distances, neighbor_indices = neighbor_model.kneighbors(pseudo_features)

    if max_neighbor_distance is not None:
        if max_neighbor_distance <= 0:
            raise ValueError("max_neighbor_distance must be positive.")
        distance_threshold = float(max_neighbor_distance)
    elif neighbor_distance_quantile is None:
        distance_threshold = np.inf
    else:
        distance_threshold = float(
            np.quantile(distances[:, -1], neighbor_distance_quantile)
        )

    interpolation_matrix, valid_neighbors, normalized_weights = _neighbor_weights(
        distances=distances,
        neighbor_indices=neighbor_indices,
        known_count=known_adata.n_obs,
        weighting=weighting,
        distance_power=float(distance_power),
        distance_epsilon=0.1,
        distance_threshold=distance_threshold,
    )
    enhanced_matrix = interpolation_matrix @ known_adata.X

    pseudo_obs = pd.DataFrame(
        index=pd.Index(
            [f"enhanced_{i}" for i in range(len(pseudo_coordinates))],
            name="spot_id",
        )
    )
    pseudo_obs[x_key] = pseudo_coordinates[:, 0]
    pseudo_obs[y_key] = pseudo_coordinates[:, 1]
    pseudo_obs["x"] = pseudo_coordinates[:, 0]
    pseudo_obs["y"] = pseudo_coordinates[:, 1]
    pseudo_obs["histology_color"] = pseudo_color
    pseudo_obs["histology_z"] = pseudo_z
    pseudo_obs["nearest_distance"] = distances[:, 0]
    pseudo_obs["neighbor_count"] = valid_neighbors.sum(axis=1)
    pseudo_obs["weight_sum"] = normalized_weights.sum(axis=1)

    enhanced_adata = ad.AnnData(
        X=enhanced_matrix,
        obs=pseudo_obs,
        var=known_adata.var.copy(),
    )
    enhanced_adata.obsm["spatial"] = pseudo_coordinates.copy()
    enhanced_adata.uns["gene_enhancement"] = {
        "resolution": float(resolution),
        "histology_scale": float(histology_scale),
        "n_neighbors": effective_neighbors,
        "distance_power": float(distance_power),
        "weighting": weighting,
        "distance_threshold": distance_threshold,
        "neighbor_distance_quantile": neighbor_distance_quantile,
        "contour_scale": float(contour_scale),
        "color_patch_size": int(color_patch_size),
        "channel_weights": channel_weights.tolist(),
        "missing_genes": list(missing_genes),
        "x_key": x_key,
        "y_key": y_key,
    }

    if print_results:
        assigned = int((pseudo_obs["neighbor_count"] > 0).sum())
        print(
            f"Enhanced {len(selected_genes)} genes at "
            f"{enhanced_adata.n_obs} pseudo spots; "
            f"{assigned} received neighbor-weighted expression."
        )
    return enhanced_adata


def enhance_gene_expression(
    image,
    raw_adata,
    genes=None,
    contour_result=None,
    contour_method="auto",
    contour_kwargs=None,
    resolution=50,
    histology_scale=1.0,
    n_neighbors=10,
    distance_power=2.0,
    weighting="inverse_distance",
    neighbor_distance_quantile=0.95,
    max_neighbor_distance=None,
    contour_scale=1.05,
    color_patch_size=None,
    x_key="pixel_x",
    y_key="pixel_y",
    array_x_key="array_x",
    array_y_key="array_y",
    color_order="rgb",
    strict_genes=True,
    max_pseudo_spots=250_000,
    n_jobs=None,
    qc_path=None,
    print_results=True,
):
    """Detect the H&E contour and enhance genes in one reproducible workflow.

    Pass a previously inspected ``contour_result`` to reuse it, or leave it as
    ``None`` to run automatic contour selection. Additional contour settings
    can be supplied through ``contour_kwargs``.

    Parameters
    ----------
    image : numpy.ndarray
        Full-resolution RGB/BGR or grayscale H&E image.
    raw_adata : AnnData
        Observed spot-level gene expression. ``.obs`` must contain the columns
        named by ``x_key`` and ``y_key``.

    genes : sequence, optional
        Genes to enhance in the requested output order; defaults to all genes.

    contour_result : TissueContourResult, optional
        Previously inspected contour result. If omitted, a contour is detected.

    contour_method : {"auto", "color", "cv2", "scan_x", "scan_y", "spot_hull"}
        Contour candidate to use. ``"auto"`` scores all available candidates.
        - "color": Detects dark or saturated H&E pixels. Uses Otsu thresholding, 
        morphological cleanup, hole filing, and component filtering.
        - "cv2": Uses Gaussian smoothing and Cany edge detection with automatically
        estimated thresholds. Selects the largest detected contour.
        - "scan_x": Builds a contour from minimum/maximum y coordinates across 
        spatial array-x groups.
        - "scan_y": Builds a contour from minimum/maximum x coordinates across
        spatial array-y groups.
        - "spot_hull": constructs a convex hull around all spatial spots.

    contour_kwargs : mapping, optional
        Extra keyword arguments for :func:`detect_he_tissue_contour`.

    resolution : float
        Pixel spacing between generated pseudo spots. Smaller values produce
        more spots and require more memory.

    histology_scale : float
        Strength of H&E color in neighbor matching; zero uses spatial distance
        only.

    n_neighbors : int
        Maximum observed spots used to interpolate each pseudo spot.

    distance_power : float
        Inverse-distance exponent when ``weighting="inverse_distance"``.

    weighting : {"inverse_distance", "exponential", "uniform"}
        Method used to combine neighboring observed spots.

    neighbor_distance_quantile : float or None
        Automatic neighbor-distance cutoff. ``None`` keeps all selected
        neighbors.

    max_neighbor_distance : float, optional
        Explicit distance cutoff overriding ``neighbor_distance_quantile``.

    contour_scale : float
        Contour enlargement factor used before creating the pseudo-spot grid.

    color_patch_size : int, optional
        H&E window width around each coordinate; defaults to ``resolution``.

    x_key, y_key : str
        Pixel-coordinate column names in ``raw_adata.obs``.

    array_x_key, array_y_key : str
        Spatial-array columns used by the x/y scan contour candidates.

    color_order : {"rgb", "bgr"}
        Use ``"bgr"`` for images loaded by ``cv2.imread``.

    strict_genes : bool
        Raise if requested genes are absent; otherwise omit missing genes.

    max_pseudo_spots : int or None
        Safety ceiling for generated pseudo spots, not a requested spot count.
        The default 250,000 prevents accidentally creating still larger grids;
        lower it for limited memory, or use ``None`` to disable the ceiling.

    n_jobs : int, optional
        Parallel workers used by nearest-neighbor search.

    qc_path : path-like, optional
        Path for saving the selected-contour QC image.

    print_results : bool
        Print contour scores and enhancement summary.

    Returns
    -------
    GeneEnhancementResult
        ``enhanced_adata`` contains pseudo-spot expression and coordinates;
        ``contour_result`` contains the contour, masks, QC image, and scores.

    Notes
    -----
    The approximate pseudo-spot count is ``tissue_area / resolution**2``.
    Dense expression storage requires roughly
    ``n_pseudo_spots * n_genes * bytes_per_value`` bytes, so 250,000 is not a
    universally safe target when enhancing thousands of genes.

    """
    if contour_result is None:
        contour_parameters = dict(contour_kwargs or {})
        contour_parameters.setdefault("method", contour_method)
        contour_parameters.setdefault("x_key", x_key)
        contour_parameters.setdefault("y_key", y_key)
        contour_parameters.setdefault("array_x_key", array_x_key)
        contour_parameters.setdefault("array_y_key", array_y_key)
        contour_parameters.setdefault("color_order", color_order)
        contour_parameters.setdefault("contour_scale", contour_scale)
        contour_parameters.setdefault("qc_path", qc_path)
        contour_parameters.setdefault("print_results", print_results)
        contour_result = detect_he_tissue_contour(
            image=image,
            spots=raw_adata,
            **contour_parameters,
        )
    elif not isinstance(contour_result, TissueContourResult):
        raise TypeError("contour_result must be a TissueContourResult.")

    enhanced_adata = impute_gene_expression(
        image=image,
        raw_adata=raw_adata,
        contour=contour_result.contour,
        genes=genes,
        resolution=resolution,
        histology_scale=histology_scale,
        n_neighbors=n_neighbors,
        distance_power=distance_power,
        weighting=weighting,
        neighbor_distance_quantile=neighbor_distance_quantile,
        max_neighbor_distance=max_neighbor_distance,
        contour_scale=contour_scale,
        color_patch_size=color_patch_size,
        x_key=x_key,
        y_key=y_key,
        strict_genes=strict_genes,
        max_pseudo_spots=max_pseudo_spots,
        n_jobs=n_jobs,
        print_results=print_results,
    )
    return GeneEnhancementResult(
        enhanced_adata=enhanced_adata,
        contour_result=contour_result,
        params={
            "contour_method": contour_result.selected_method,
            "resolution": resolution,
            "histology_scale": histology_scale,
            "n_neighbors": n_neighbors,
            "distance_power": distance_power,
            "weighting": weighting,
            "contour_scale": contour_scale,
            "color_order": color_order,
        },
    )


__all__ = [
    "TissueContourResult",
    "GeneEnhancementResult",
    "detect_he_tissue_mask",
    "detect_he_tissue_contour",
    "scan_spot_contour",
    "contour_to_mask",
    "scale_tissue_contour",
    "remove_image_background",
    "impute_gene_expression",
    "enhance_gene_expression",
]
