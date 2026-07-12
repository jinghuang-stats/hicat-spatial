import os
import re
import cv2
import numpy as np
import pandas as pd

from ..visualization import cat_figure


# ============================================================
# coordinates QC mapping
# ============================================================
def transform_spot_coordinates(
    obs,
    image_shape,
    x_key="pixel_x",
    y_key="pixel_y",
    coord_transform="xy",
    round_coords=True,
):
    """
    Transform AnnData spot coordinates into image pixel coordinates.

    Parameters
    ----------
    obs : pd.DataFrame
        AnnData .obs table.

    image_shape : tuple
        Shape of image or mask.
        Expected as (height, width) or (height, width, channels).

    x_key, y_key : str
        Coordinate columns in obs.

    coord_transform : str
        Coordinate transformation to apply.

        Supported transforms:

        "xy":
            image_x = x
            image_y = y

        "xy_flip_x":
            image_x = W - 1 - x
            image_y = y

        "xy_flip_y":
            image_x = x
            image_y = H - 1 - y

        "xy_flip_xy":
            image_x = W - 1 - x
            image_y = H - 1 - y

        "yx":
            image_x = y
            image_y = x

        "yx_flip_x":
            image_x = W - 1 - y
            image_y = x

        "yx_flip_y":
            image_x = y
            image_y = H - 1 - x

        "yx_flip_xy":
            image_x = W - 1 - y
            image_y = H - 1 - x

    round_coords : bool
        Whether to round transformed coordinates to integers.

    Returns
    -------
    image_x : np.ndarray
        Transformed image x-coordinate.

    image_y : np.ndarray
        Transformed image y-coordinate.

    in_bounds : np.ndarray
        Whether each transformed coordinate lies inside the image.
    """

    if image_shape is None:
        raise ValueError("image_shape must be provided.")

    if x_key not in obs.columns:
        raise KeyError(f"{x_key!r} is not found in obs.")

    if y_key not in obs.columns:
        raise KeyError(f"{y_key!r} is not found in obs.")

    image_height, image_width = image_shape[:2]

    x = obs[x_key].astype(float).to_numpy()
    y = obs[y_key].astype(float).to_numpy()

    if coord_transform == "xy":
        image_x = x
        image_y = y

    elif coord_transform == "xy_flip_x":
        image_x = image_width - 1 - x
        image_y = y

    elif coord_transform == "xy_flip_y":
        image_x = x
        image_y = image_height - 1 - y

    elif coord_transform == "xy_flip_xy":
        image_x = image_width - 1 - x
        image_y = image_height - 1 - y

    elif coord_transform == "yx":
        image_x = y
        image_y = x

    elif coord_transform == "yx_flip_x":
        image_x = image_width - 1 - y
        image_y = x

    elif coord_transform == "yx_flip_y":
        image_x = y
        image_y = image_height - 1 - x

    elif coord_transform == "yx_flip_xy":
        image_x = image_width - 1 - y
        image_y = image_height - 1 - x

    else:
        raise ValueError(
            "coord_transform must be one of: "
            "'xy', 'xy_flip_x', 'xy_flip_y', 'xy_flip_xy', "
            "'yx', 'yx_flip_x', 'yx_flip_y', 'yx_flip_xy'."
        )

    if round_coords:
        image_x = np.round(image_x).astype(int)
        image_y = np.round(image_y).astype(int)

    in_bounds = (
        (image_x >= 0) &
        (image_x < image_width) &
        (image_y >= 0) &
        (image_y < image_height)
    )

    return image_x, image_y, in_bounds


def plot_spots_on_image_with_transform(
    image_path,
    adata,
    x_key="pixel_x",
    y_key="pixel_y",
    coord_transform="xy",
    output_path=None,
    spot_radius=20,
    spot_color=(0, 0, 255),
    spot_thickness=-1,
    max_spots=None,
    random_state=0,
):
    """
    Overlay AnnData spot coordinates on histology image using a specified
    coordinate transformation.
    """

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Cannot read image_path: {image_path}")

    img_overlay = img.copy()

    image_x, image_y, in_bounds = transform_spot_coordinates(
        obs=adata.obs,
        x_key=x_key,
        y_key=y_key,
        image_shape=img.shape,
        coord_transform=coord_transform,
        round_coords=True,
    )

    idx = np.arange(adata.n_obs)

    if max_spots is not None and max_spots < adata.n_obs:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(idx, size=max_spots, replace=False)

    for i in idx:
        if not in_bounds[i]:
            continue

        cv2.circle(
            img_overlay,
            center=(int(image_x[i]), int(image_y[i])),
            radius=spot_radius,
            color=spot_color,
            thickness=spot_thickness,
        )

    qc_df = adata.obs[[x_key, y_key]].copy()
    qc_df["image_x"] = image_x
    qc_df["image_y"] = image_y
    qc_df["in_image_bounds"] = in_bounds
    qc_df["coord_transform"] = coord_transform

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img_overlay)

    return img_overlay, qc_df


def save_all_coordinate_qc_images(
    image_path,
    adata,
    output_dir,
    section_name,
    x_key="pixel_x",
    y_key="pixel_y",
    coord_transforms=None,
    spot_radius=20,
    max_spots=None,
    random_state=0,
):
    """
    Save coordinate QC images for all common coordinate transformations.

    Users should inspect the output images and choose the transformation where
    the spots correctly overlay the tissue image.
    """

    if coord_transforms is None:
        coord_transforms = [
            "xy",
            "xy_flip_x",
            "xy_flip_y",
            "xy_flip_xy",
            "yx",
            "yx_flip_x",
            "yx_flip_y",
            "yx_flip_xy",
        ]

    os.makedirs(output_dir, exist_ok=True)

    qc_results = {}

    for coord_transform in coord_transforms:

        output_path = os.path.join(
            output_dir,
            f"{section_name}_spots_{coord_transform}.png"
        )

        _, qc_df = plot_spots_on_image_with_transform(
            image_path=image_path,
            adata=adata,
            x_key=x_key,
            y_key=y_key,
            coord_transform=coord_transform,
            output_path=output_path,
            spot_radius=spot_radius,
            max_spots=max_spots,
            random_state=random_state,
        )

        qc_results[coord_transform] = {
            "output_path": output_path,
            "qc_df": qc_df,
            "in_bounds_count": int(qc_df["in_image_bounds"].sum()),
            "in_bounds_fraction": float(qc_df["in_image_bounds"].mean()),
        }

    print(f"\nSaved coordinate QC images for {section_name}:")
    for coord_transform, res in qc_results.items():
        print(
            f"  {coord_transform:12s} "
            f"in-bounds: {res['in_bounds_count']} / {adata.n_obs} "
            f"({res['in_bounds_fraction']:.3f}) "
            f"-> {res['output_path']}"
        )

    return qc_results


# ============================================================
# Extract pathologist scribbles
# ============================================================
def extract_scribble_masks(
    image_path,
    annotated_image_path,
    label_color_dict,
    selected_labels=None,
    color_tolerance=30,
    resize_min_size=1000,
    min_contour_area=1000,
    output_dir=None,
    save_individual_masks=True,
):
    """
    Extract pathologist scribble masks from an annotated histology image.

    Parameters
    ----------
    image_path : str
        Path to the original histology image.

    annotated_image_path : str
        Path to the annotated image containing pathologist scribbles.

    label_color_dict : dict
        Dictionary mapping label names to RGB colors.
        Example:
        {
            "invasive_cancer": [236, 28, 36],
            "connective_tissue": [63, 72, 203]
        }

    selected_labels : list or None
        Labels to extract for this specific tissue section.
        If None, all labels in label_color_dict will be used.

    color_tolerance : int or dict
        Allowed RGB deviation when matching scribble colors.
        If int, the same tolerance is used for all labels and channels.
        If dict, it should map label names to RGB tolerances.
        Example:
        {
            "invasive_cancer": [30, 30, 30],
            "connective_tissue": [25, 25, 25]
        }

    resize_min_size : int
        Resize annotated image so that its shorter side equals this value.
        This speeds up contour detection.

    min_contour_area : int
        Minimum contour area to keep. Smaller detected regions are removed.

    output_dir : str or None
        Directory to save masks. If None, masks are not saved.

    save_individual_masks : bool
        Whether to save one binary mask per label.

    Returns
    -------
    ref_mask : np.ndarray
        Final label mask resized to the original image size.
        Shape: original image height x original image width.
        Values:
            0 = background / unlabeled
            1, 2, 3, ... = tissue labels

    label_id_dict : dict
        Dictionary mapping integer mask values to label names.
        Example:
        {
            0: "nan",
            1: "invasive_cancer",
            2: "connective_tissue"
        }

    d_mask : dict
        Dictionary mapping label names to binary masks at resized annotated-image resolution.
    """

    # ------------------------------------------------------------
    # Read original and annotated images
    # ------------------------------------------------------------
    img = cv2.imread(image_path)
    img_annotated = cv2.imread(annotated_image_path)

    if img is None:
        raise ValueError(f"Cannot read image_path: {image_path}")

    if img_annotated is None:
        raise ValueError(f"Cannot read annotated_image_path: {annotated_image_path}")

    # cv2 reads images as BGR, but user-provided colors are usually RGB
    original_height, original_width = img.shape[:2]

    # ------------------------------------------------------------
    # Decide which labels to extract
    # ------------------------------------------------------------
    if selected_labels is None:
        selected_labels = list(label_color_dict.keys())

    missing_labels = [label for label in selected_labels if label not in label_color_dict]
    if len(missing_labels) > 0:
        raise ValueError(f"These selected labels are not in label_color_dict: {missing_labels}")

    # ------------------------------------------------------------
    # Resize annotated image for faster processing
    # ------------------------------------------------------------
    resize_factor = resize_min_size / np.min(img.shape[:2])
    resize_width = int(img.shape[1] * resize_factor)
    resize_height = int(img.shape[0] * resize_factor)

    img_annotated_resized = cv2.resize(
        img_annotated,
        (resize_width, resize_height),
        interpolation=cv2.INTER_AREA
    )

    # ------------------------------------------------------------
    # Prepare output folder
    # ------------------------------------------------------------
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Extract binary mask for each label
    # ------------------------------------------------------------
    d_mask = {}

    for label in selected_labels:
        r, g, b = label_color_dict[label]

        if isinstance(color_tolerance, dict):
            r_tol, g_tol, b_tol = color_tolerance.get(label, [30, 30, 30])
        else:
            r_tol = g_tol = b_tol = color_tolerance

        # Because OpenCV image is BGR:
        b_channel = img_annotated_resized[:, :, 0]
        g_channel = img_annotated_resized[:, :, 1]
        r_channel = img_annotated_resized[:, :, 2]

        color_mask = (
            (b_channel > b - b_tol) & (b_channel < b + b_tol) &
            (g_channel > g - g_tol) & (g_channel < g + g_tol) &
            (r_channel > r - r_tol) & (r_channel < r + r_tol)
        ).astype(np.uint8)

        # Find contours from the binary color mask
        contours, _ = cv2.findContours(
            color_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Keep only sufficiently large contours
        mask = np.zeros(color_mask.shape, dtype=np.uint8)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_contour_area:
                cv2.drawContours(mask, [cnt], -1, 1, thickness=-1)

        d_mask[label] = mask

        if output_dir is not None and save_individual_masks:
            save_path = os.path.join(output_dir, f"{label}_mask.jpg")
            cv2.imwrite(save_path, mask * 255)

    # ------------------------------------------------------------
    # Merge individual masks into one reference mask
    # ------------------------------------------------------------
    ref_mask_resized = np.zeros(img_annotated_resized.shape[:2], dtype=np.uint8)

    label_id_dict = {0: "nan"}

    for idx, label in enumerate(selected_labels, start=1):
        mask = d_mask[label]
        ref_mask_resized[mask != 0] = idx
        label_id_dict[idx] = label

    # ------------------------------------------------------------
    # Resize final mask back to original image size
    # Important: use INTER_NEAREST for label masks
    # ------------------------------------------------------------
    ref_mask = cv2.resize(
        ref_mask_resized,
        (original_width, original_height),
        interpolation=cv2.INTER_NEAREST
    )

    if output_dir is not None:
        ref_mask_path = os.path.join(output_dir, "ref_mask.jpg")
        cv2.imwrite(ref_mask_path, ref_mask * 40)

    return ref_mask, label_id_dict, d_mask


def assign_mask_labels_to_adata(
    ref_adata,
    ref_mask,
    label_id_dict,
    x_key="pixel_x",
    y_key="pixel_y",
    label_key="label",
    coord_scale_x=1.0,
    coord_scale_y=1.0,
    background_label="nan",
    copy=True,
):
    """
    Assign labels from an image-space label mask back to AnnData observations.

    Important
    ---------
    Image masks are indexed as ref_mask[y, x], not ref_mask[x, y].

    Parameters
    ----------
    ref_adata : AnnData
        Reference AnnData object.

    ref_mask : np.ndarray
        Label mask with shape (image_height, image_width).
        Values should correspond to keys in label_id_dict.

    label_id_dict : dict
        Mapping from integer mask IDs to label names.
        Example:
        {
            0: "nan",
            1: "invasive_cancer",
            2: "connective_tissue"
        }

    x_key : str
        Column in ref_adata.obs containing image x-coordinate / pixel column.

    y_key : str
        Column in ref_adata.obs containing image y-coordinate / pixel row.

    label_key : str
        Name of the output label column added to ref_adata.obs.

    coord_scale_x : float
        Scale factor applied to x_key before indexing the mask.
        Use 1.0 if adata coordinates are already in the same image pixel space.

    coord_scale_y : float
        Scale factor applied to y_key before indexing the mask.
        Use 1.0 if adata coordinates are already in the same image pixel space.

    background_label : str
        Label assigned to mask value 0 or out-of-bound spots.

    copy : bool
        Whether to copy ref_adata before modifying.

    Returns
    -------
    ref_adata : AnnData
        Updated AnnData object with ref_adata.obs[label_key].
    """

    if copy:
        ref_adata = ref_adata.copy()

    if x_key not in ref_adata.obs.columns:
        raise KeyError(f"{x_key!r} is not found in ref_adata.obs.")

    if y_key not in ref_adata.obs.columns:
        raise KeyError(f"{y_key!r} is not found in ref_adata.obs.")

    mask_height, mask_width = ref_mask.shape[:2]

    labels = []

    for _, row in ref_adata.obs.iterrows():
        x = int(round(float(row[x_key]) * coord_scale_x))
        y = int(round(float(row[y_key]) * coord_scale_y))

        # Important:
        # NumPy image indexing is [y, x], not [x, y].
        if 0 <= x < mask_width and 0 <= y < mask_height:
            mask_id = int(ref_mask[y, x])
            label = label_id_dict.get(mask_id, background_label)
        else:
            label = background_label

        labels.append(label)

    ref_adata.obs[label_key] = pd.Categorical(labels)

    return ref_adata


def extract_scribble_labels_pipeline(
    ref_adata_dic,
    ref_section_list,
    data_path,
    label_color_dict,
    image_template="{section}.jpg",
    annotated_image_template="{section}_annotated.jpg",
    selected_labels_dic=None,
    selected_labels=None,
    x_key="pixel_x",
    y_key="pixel_y",
    label_key="label",
    output_dir="./extract_scribbles",
    color_tolerance=30,
    resize_min_size=1000,
    min_contour_area=1000,
    save_individual_masks=True,
    coord_scale_x=1.0,
    coord_scale_y=1.0,
    background_label="nan",
    cat_color=None,
    fig_size=50,
    fig_region_size=50,
    dpi=100,
    invert_x=False,
    invert_y=True,
    plot_results=True,
    copy=True,
):
    """
    Extract pathologist scribble annotations from annotated images and assign
    the extracted labels back to reference AnnData objects.

    This pipeline performs the following steps for each reference section:

        1. Load the original image and annotated image.
        2. Extract color-based scribble masks from the annotated image.
        3. Assign extracted mask labels to spatial spots in the corresponding
           AnnData object using spot coordinates.
        4. Save assigned labels as a CSV file.
        5. Optionally generate QC plots for each extracted label and all labels.

    Parameters
    ----------
    ref_adata_dic : dict
        Dictionary of reference AnnData objects.

        Example:
        {
            "H1": adata_H1,
            "G2": adata_G2,
            "E1": adata_E1
        }

        Each AnnData object must contain spatial coordinate columns in
        `adata.obs`, specified by `x_key` and `y_key`.

    ref_section_list : list of str
        List of reference section names to process. Each section name must be
        a key in `ref_adata_dic`.

    data_path : str
        Directory containing the original images and annotated images.

    label_color_dict : dict
        Dictionary mapping tissue label names to RGB colors used in the
        annotated image.

        Example:
        {
            "invasive_cancer": (255, 0, 0),
            "cancer_in_situ": (0, 255, 0),
            "connective_tissue": (0, 0, 255)
        }

        The RGB colors should match the scribble colors in the annotated image.

    image_template : str, default="{section}.jpg"
        File name template for the original image. The template must contain
        "{section}", which will be replaced by each section name.

        Example:
        "{section}.jpg" gives "H1.jpg" for section "H1".

    annotated_image_template : str, default="{section}_annotated.jpg"
        File name template for the annotated image. The template must contain
        "{section}", which will be replaced by each section name.

        Example:
        "{section}_annotated.jpg" gives "H1_annotated.jpg" for section "H1".

    selected_labels_dic : dict or None, default=None
        Optional dictionary specifying section-specific labels to extract.

        Example:
        {
            "H1": ["invasive_cancer", "connective_tissue"],
            "G2": ["cancer_in_situ", "breast_glands"]
        }

        If provided, labels are selected separately for each section. If a
        section is missing from this dictionary, the function falls back to
        `selected_labels`.

    selected_labels : list of str or None, default=None
        Global list of labels to extract from all sections.

        If `selected_labels_dic` is None, this list is used for all sections.
        If both `selected_labels_dic` and `selected_labels` are None, all labels
        in `label_color_dict` are extracted.

    x_key : str, default="pixel_x"
        Column name in `adata.obs` containing x coordinates.

    y_key : str, default="pixel_y"
        Column name in `adata.obs` containing y coordinates.

    label_key : str, default="label"
        Column name in `adata.obs` where assigned scribble labels will be saved.

    output_dir : str, default="./extract_scribbles"
        Base output directory. For each reference section, a section-specific
        subfolder will be created under this directory.

        Example:
            If ``output_dir="./extract_scribbles"`` and ``ref_section="H1"``,
            outputs will be saved under::

                ./extract_scribbles/H1

            The assigned-label CSV will be saved as::

                ./extract_scribbles/H1/H1_ref_labels.csv

            Extracted masks and QC figures will be saved under::

                ./extract_scribbles/H1/masks/
                ./extract_scribbles/H1/labels/

    color_tolerance : int or float, default=30
        RGB color tolerance used when extracting scribbles from the annotated
        image. Larger values allow more deviation from the exact RGB color in
        `label_color_dict`.

    resize_min_size : int, default=1000
        Minimum image size used internally during scribble mask extraction.
        This can help stabilize contour detection if the image is very small.
        The exact behavior depends on `extract_scribble_masks`.

    min_contour_area : int or float, default=1000
        Minimum contour area retained during scribble extraction. Smaller
        connected components below this area are removed as noise.

    save_individual_masks : bool, default=True
        Whether to save one binary mask per selected label.

    coord_scale_x : float, default=1.0
        Scale factor used to map AnnData x coordinates to image pixel
        coordinates.

        For example, if `adata.obs[x_key]` is measured at half resolution
        relative to the image, use `coord_scale_x=2.0`.

    coord_scale_y : float, default=1.0
        Scale factor used to map AnnData y coordinates to image pixel
        coordinates.

    background_label : str, default="nan"
        Label assigned to spots that do not fall inside any extracted scribble
        mask.

    cat_color : dict, list, or None, default=None
        Color mapping passed to `cat_figure` for plotting categorical labels.
        If None, the default color handling inside `cat_figure` is used.

    fig_size : int or float, default=50
        Point size or figure-related size parameter passed to `cat_figure`
        when plotting all extracted labels together.

    fig_region_size : int or float, default=50
        Point size or figure-related size parameter passed to `cat_figure`
        when plotting each extracted region separately.

    dpi : int, default=100
        Resolution of saved QC figures.

    invert_x : bool, default=False
        Whether to invert the x-axis in QC plots.

    invert_y : bool, default=True
        Whether to invert the y-axis in QC plots. This is often useful for
        image pixel coordinates, where the origin is usually at the top-left.

    plot_results : bool, default=True
        Whether to save QC plots for each extracted label and all labels
        together.

    copy : bool, default=True
        Whether to copy each AnnData object before modifying it.

        If True, the original objects in `ref_adata_dic` are not modified.
        If False, labels are assigned in place.

    Returns
    -------
    updated_ref_adata_dic : dict
        Dictionary of updated AnnData objects.

        Each AnnData object contains a new or updated column:

            adata.obs[label_key]

        storing the assigned scribble label for each spot.

    results_dic : dict
        Dictionary containing section-level extraction results.

        For each section, `results_dic[section]` contains:

            "ref_mask" : np.ndarray
                Combined integer mask where each label is represented by a
                numeric label ID.

            "label_id_dict" : dict
                Mapping from integer mask ID to label name.

            "d_mask" : dict
                Dictionary of individual binary masks for each label.

            "label_counts" : pandas.Series
                Counts of assigned labels in `adata.obs[label_key]`.

            "label_csv_path" : str
                Path to the saved CSV file containing assigned labels.
    """

    updated_ref_adata_dic = {}
    results_dic = {}

    def _safe_filename(x):
        """Convert label name to a filesystem-safe string."""
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x))

    for ref_section in ref_section_list:

        if ref_section not in ref_adata_dic:
            raise KeyError(f"{ref_section!r} is not found in ref_adata_dic.")

        print(f"\nExtracting scribble labels for section: {ref_section}")

        image_path = os.path.join(
            data_path,
            image_template.format(section=ref_section)
        )

        annotated_image_path = os.path.join(
            data_path,
            annotated_image_template.format(section=ref_section)
        )

        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"Original image for section {ref_section!r} was not found: "
                f"{image_path}"
            )

        if not os.path.exists(annotated_image_path):
            raise FileNotFoundError(
                f"Annotated image for section {ref_section!r} was not found: "
                f"{annotated_image_path}"
            )

        # ------------------------------------------------------------
        # Section-specific output folders
        # ------------------------------------------------------------
        section_output_dir = os.path.join(output_dir, ref_section)
        section_mask_output_dir = os.path.join(section_output_dir, "masks")
        section_fig_output_dir = os.path.join(section_output_dir, "labels")

        os.makedirs(section_output_dir, exist_ok=True)
        os.makedirs(section_mask_output_dir, exist_ok=True)

        if plot_results:
            os.makedirs(section_fig_output_dir, exist_ok=True)

        # ------------------------------------------------------------
        # Determine selected labels for this section
        # ------------------------------------------------------------
        if selected_labels_dic is not None:
            section_selected_labels = selected_labels_dic.get(
                ref_section,
                selected_labels
            )
        else:
            section_selected_labels = selected_labels

        # ------------------------------------------------------------
        # Step 1. Extract scribble masks from annotated image
        # ------------------------------------------------------------
        ref_mask, label_id_dict, d_mask = extract_scribble_masks(
            image_path=image_path,
            annotated_image_path=annotated_image_path,
            label_color_dict=label_color_dict,
            selected_labels=section_selected_labels,
            color_tolerance=color_tolerance,
            resize_min_size=resize_min_size,
            min_contour_area=min_contour_area,
            output_dir=section_mask_output_dir,
            save_individual_masks=save_individual_masks,
        )

        # ------------------------------------------------------------
        # Step 2. Assign mask labels back to adata.obs
        # ------------------------------------------------------------
        ref_adata = assign_mask_labels_to_adata(
            ref_adata=ref_adata_dic[ref_section],
            ref_mask=ref_mask,
            label_id_dict=label_id_dict,
            x_key=x_key,
            y_key=y_key,
            label_key=label_key,
            coord_scale_x=coord_scale_x,
            coord_scale_y=coord_scale_y,
            background_label=background_label,
            copy=copy,
        )

        # Make sure the label column is categorical for stable plotting.
        if not pd.api.types.is_categorical_dtype(ref_adata.obs[label_key]):
            ref_adata.obs[label_key] = ref_adata.obs[label_key].astype("category")

        updated_ref_adata_dic[ref_section] = ref_adata

        # ------------------------------------------------------------
        # Step 3. Save assigned labels
        # ------------------------------------------------------------
        label_pred = ref_adata.obs.copy()

        label_csv_path = os.path.join(
            section_output_dir,
            f"{ref_section}_ref_labels.csv"
        )

        label_pred.to_csv(label_csv_path, index=True)

        label_counts = ref_adata.obs[label_key].value_counts(dropna=False)

        results_dic[ref_section] = {
            "ref_mask": ref_mask,
            "label_id_dict": label_id_dict,
            "d_mask": d_mask,
            "label_counts": label_counts,
            "label_csv_path": label_csv_path,
        }

        print(label_counts)

        # ------------------------------------------------------------
        # Step 4. QC visualization
        # ------------------------------------------------------------
        if plot_results:

            # Plot each extracted region separately.
            for region in ref_adata.obs[label_key].cat.categories:

                if str(region) == str(background_label):
                    continue

                sub = ref_adata[ref_adata.obs[label_key] == region].copy()

                if sub.n_obs == 0:
                    continue

                safe_region = _safe_filename(region)

                fig_title = f"{ref_section}: {region}"
                fig_path = os.path.join(
                    section_fig_output_dir,
                    f"{ref_section}_{safe_region}_label.png"
                )

                cat_figure(
                    input_adata=sub,
                    x_key=x_key,
                    y_key=y_key,
                    fig_title=fig_title,
                    fig_path=fig_path,
                    color_key=label_key,
                    cat_color=cat_color,
                    fig_size=fig_region_size,
                    dpi=dpi,
                    invert_x=invert_x,
                    invert_y=invert_y,
                )

            # Plot all extracted labels together.
            fig_title = f"{ref_section}: extracted labels"
            fig_path = os.path.join(
                section_fig_output_dir,
                f"{ref_section}_all_extracted_labels.png"
            )

            cat_figure(
                input_adata=ref_adata,
                x_key=x_key,
                y_key=y_key,
                fig_title=fig_title,
                fig_path=fig_path,
                color_key=label_key,
                cat_color=cat_color,
                fig_size=fig_size,
                dpi=dpi,
                invert_x=invert_x,
                invert_y=invert_y,
            )

    return updated_ref_adata_dic, results_dic
