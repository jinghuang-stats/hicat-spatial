import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from copy import deepcopy
from scipy.sparse import issparse
from sklearn.preprocessing import MinMaxScaler


@dataclass(frozen=True)
class PreprocessPaths:
    """Standard folders used for one reference or query cohort.

    Attributes
    ----------
    cohort_dir
        ``<preprocess_dir>/<cohort>``.
    raw_dir
        User-provided raw inputs. Pipeline outputs are never written here.
    preprocessed_dir
        Final ``.h5ad`` files and all derived outputs.
    contour_dir
        Contour quality-control images created during enhancement.
    scribble_dir
        Extracted reference masks, labels, and scribble QC plots.
    image_spot_dir, image_enhanced_dir
        Model workspaces for observed-spot and enhanced-grid image features.
    """

    cohort_dir: Path
    raw_dir: Path
    preprocessed_dir: Path
    contour_dir: Path
    scribble_dir: Path
    image_spot_dir: Path
    image_enhanced_dir: Path


# ============================================================
# Data preprocessing
# ============================================================
def assign_spot_labels(
    obs_df,
    ref_mask,
    label_id_dict,
    x_col="x",
    y_col="y",
    spot_id_col=None,
):
    """
    Assign tissue labels to spots based on their x/y coordinates and ref_mask.

    Parameters
    ----------
    obs_df : pandas.DataFrame
        Spot metadata dataframe, such as adata.obs.

    ref_mask : np.ndarray
        Final annotation mask at the original image size.
        Shape: height x width.

    label_id_dict : dict
        Dictionary mapping integer mask IDs to label names.
        Example:
        {
            0: "nan",
            1: "invasive_cancer",
            2: "connective_tissue"
        }

    x_col : str
        Column name for x coordinates.

    y_col : str
        Column name for y coordinates.

    spot_id_col : str or None
        Optional column containing spot IDs.
        If None, obs_df.index is used.

    Returns
    -------
    spot_label_df : pandas.DataFrame
        Dataframe with spot ID, x, y, label ID, and label name.
    """

    results = []

    height, width = ref_mask.shape[:2]

    for spot_idx, row in obs_df.iterrows():
        x = int(row[x_col])
        y = int(row[y_col])

        # Check whether coordinates are inside the image
        if x < 0 or x >= width or y < 0 or y >= height:
            label_id = 0
            label_name = "nan"
        else:
            label_id = int(ref_mask[y, x]) # check x and y coordinates
            label_name = label_id_dict.get(label_id, "nan")

        if spot_id_col is None:
            spot_id = spot_idx
        else:
            spot_id = row[spot_id_col]

        results.append({
            "spot_id": spot_id,
            "x": x,
            "y": y,
            "label_id": label_id,
            "label": label_name,
        })

    spot_label_df = pd.DataFrame(results)

    return spot_label_df


def filter_low_exp_genes(input_adata, low_exp_thres=0.02):
    """
    Filter genes based on the fraction of spots/cells with non-zero expression.

    Parameters
    ----------
    input_adata : AnnData
        Input AnnData object.

    low_exp_thres : float
        Minimum fraction of spots/cells where a gene must be expressed.

    Returns
    -------
    filtered_genes : list
        Genes passing the expression frequency threshold.
    """

    X = input_adata.X

    if issparse(X):
        nonzero_exp_frac = np.asarray((X > 0).mean(axis=0)).ravel()
    else:
        nonzero_exp_frac = (X > 0).mean(axis=0)

    gene_names = input_adata.var.index.to_numpy()
    filtered_genes = gene_names[nonzero_exp_frac >= low_exp_thres].tolist()

    return filtered_genes


def normalize_adata(input_adata, method="min_max", copy=True):
    """
    Normalize AnnData.X within one sample.

    Parameters
    ----------
    input_adata : AnnData
        Input AnnData object.

    method : str
        Normalization method.
        Currently supports:
            "min_max" : scale each gene to [0, 1] within the sample
            None or "none" : no normalization

    copy : bool
        Whether to return a copied AnnData object.

    Returns
    -------
    output_adata : AnnData
        Normalized AnnData object.
    """

    if copy:
        output_adata = input_adata.copy()
    else:
        output_adata = input_adata

    if method is None or method == "none":
        return output_adata

    if method == "min_max":
        X = output_adata.X

        # MinMaxScaler expects a dense matrix. Make this cost visible because
        # large spatial matrices can otherwise exhaust memory unexpectedly.
        if issparse(X):
            warnings.warn(
                "min_max normalization converts sparse AnnData.X to a dense "
                "array; use this method only when the matrix fits in memory.",
                RuntimeWarning,
                stacklevel=2,
            )
            X = X.toarray()
        else:
            X = np.asarray(X)

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        output_adata.X = X_scaled

        return output_adata

    else:
        raise ValueError(f"Unsupported normalization method: {method}")


def preprocess_adata(
    input_adata,
    low_exp_thres=0.05,
    normalize=True,
    normalization_method="min_max",
):
    """
    Filter low-expression genes and optionally normalize AnnData.
    """

    filtered_genes = filter_low_exp_genes(
        input_adata=input_adata,
        low_exp_thres=low_exp_thres,
    )

    adata = input_adata[:, input_adata.var_names.isin(filtered_genes)].copy()

    if normalize:
        adata = normalize_adata(
            input_adata=adata,
            method=normalization_method,
            copy=False,
        )

    return adata


def preprocess_adata_dic(
    adata_dic,
    section_list=None,
    low_exp_thres=0.05,
    normalize=True,
    normalization_method="min_max",
    print_results=True,
):
    """
    Preprocess a dictionary of AnnData objects.

    Parameters
    ----------
    adata_dic : dict
        Dictionary of AnnData objects.
        Example:
            {
                "A1": adata_A1,
                "B1": adata_B1,
            }
    section_list : list or None
        Sections to preprocess. If None, use all keys in adata_dic.

    Returns
    -------
    adata_sca_dic : dict
        Preprocessed AnnData dictionary.
    """

    if section_list is None:
        section_list = list(adata_dic.keys())

    adata_sca_dic = {}

    for section in section_list:
        if print_results:
            print(f"=============== {section} ===============")

        adata_sca_dic[section] = preprocess_adata(
            input_adata=adata_dic[section],
            low_exp_thres=low_exp_thres,
            normalize=normalize,
            normalization_method=normalization_method,
        )

    return adata_sca_dic


def construct_ref_adata_dic(
    ref_section_list,
    data_path,
    dataset_name,
    file_template="{section}.h5ad",
    label_key="label",
    sample_key="sample",
    low_exp_thres=0.02,
    filter_low_exp=True,
    normalize=True,
    normalize_method="min_max",
    integrate_filtered=True,
    print_results=True
):
    """
    Construct filtered reference AnnData dictionary and merged AnnData object.

    This function performs:
        1. read reference AnnData objects,
        2. filter lowly expressed genes within each sample,
        3. identify common genes shared across samples,
        4. subset each sample to common genes,
        5. optionally perform min-max normalization within each sample,
        6. integrate all reference samples into one AnnData object.

    Parameters
    ----------
    ref_section_list : list
        List of reference tissue section/sample names.
        Example:
        ["H1", "G2", "E1"]

    data_path : str
        Directory containing reference AnnData files.

    dataset_name : str
        The name of analyzed dataset

    file_template : str
        Template for AnnData file names.
        Use "{section}" as placeholder.

        Example:
        "{section}.h5ad"
        "sudo_HER2+BC_{section}_log_s=1_res=50_nbr=10_k=2.h5ad"
        "Brain_Visium_{section}_normalize+log_with_labels.h5ad"

    label_key : str
        Column in adata.obs containing tissue region labels.

    sample_key : str
        Column name used to store sample IDs after merging.

    low_exp_thres : float
        Minimum fraction of spots/cells with non-zero expression.

    filter_low_exp : bool
        Whether to filter lowly expressed genes within each sample.

    normalize : bool
        Whether to normalize each sample after gene filtering and common-gene selection.

    normalize_method : str
        Normalization method passed to normalize_adata().
        Default is "min_max".

    integrate_filtered : bool
        If True, all_adata is constructed from filtered and normalized reference samples.
        If False, all_adata is constructed from original reference samples restricted to common genes (after filtering).

    print_results : bool
        Whether to print processing information.

    Returns
    -------
    results : dict
        Dictionary containing:

        {
            "ref_adata_dic_raw": original reference AnnData dictionary,
            "ref_adata_dic_filtered": filtered and optionally normalized AnnData dictionary,
            "all_adata": merged AnnData object,
            "common_genes": common genes after filtering,
            "sample_names": sample names
        }
    """

    # ------------------------------------------------------------
    # 1. Read reference AnnData objects
    # ------------------------------------------------------------
    ref_adata_dic_raw = {}

    for section in ref_section_list:
        file_name = file_template.format(section=section, dataset_name=dataset_name)
        file_path = os.path.join(data_path, file_name)

        if print_results:
            print(f"Reading: {file_path}")

        adata = sc.read(file_path)

        if label_key not in adata.obs.columns:
            raise ValueError(
                f"{label_key} is not found in adata.obs for section: {section}"
            )

        ref_adata_dic_raw[section] = adata

        if print_results:
            print(f"{section}: raw shape = {adata.shape}")

    # ------------------------------------------------------------
    # 2. Filter lowly expressed genes within each sample
    # ------------------------------------------------------------
    ref_adata_dic_tmp = {}

    for section, adata in ref_adata_dic_raw.items():

        # Keep the input read-only until a filtered/common-gene subset is made.
        adata_tmp = adata

        if filter_low_exp:
            filtered_genes = filter_low_exp_genes(
                input_adata=adata_tmp,
                low_exp_thres=low_exp_thres
            )

            adata_tmp = adata_tmp[
                :,
                adata_tmp.var.index.isin(filtered_genes)
            ].copy()

            if print_results:
                print(
                    f"{section}: {len(filtered_genes)} genes retained "
                    f"after low-expression filtering."
                )

        ref_adata_dic_tmp[section] = adata_tmp

    # ------------------------------------------------------------
    # 3. Identify common genes shared across filtered samples
    # ------------------------------------------------------------
    common_genes = None

    for section in ref_section_list:
        genes = set(ref_adata_dic_tmp[section].var.index.tolist())

        if common_genes is None:
            common_genes = genes
        else:
            common_genes = common_genes & genes

    common_genes = sorted(common_genes)

    if len(common_genes) == 0:
        raise ValueError("No common genes found across reference samples.")

    if print_results:
        print(f"Number of common genes across samples: {len(common_genes)}")

    # ------------------------------------------------------------
    # 4. Subset to common genes and normalize within each sample
    # ------------------------------------------------------------
    ref_adata_dic_filtered = {}

    for section in ref_section_list:

        # Subset and order in one allocation.
        adata_tmp = ref_adata_dic_tmp[section][:, common_genes].copy()

        if normalize:
            adata_tmp = normalize_adata(
                input_adata=adata_tmp,
                method=normalize_method,
                copy=False
            )

        ref_adata_dic_filtered[section] = adata_tmp

        if print_results:
            print(f"{section}: filtered shape = {adata_tmp.shape}")

    # ------------------------------------------------------------
    # 5. Construct all_adata
    # ------------------------------------------------------------
    all_adata_list = []

    if integrate_filtered:
        # Use filtered and normalized data
        for section in ref_section_list:
            all_adata_list.append(ref_adata_dic_filtered[section])
    else:
        # Use raw data, but restricted to common genes
        for section in ref_section_list:
            adata_raw = ref_adata_dic_raw[section][:, common_genes].copy()

            all_adata_list.append(adata_raw)

    all_adata = ad.concat(
        all_adata_list,
        axis=0,
        join="inner",
        label=sample_key,
        keys=ref_section_list
    )

    all_adata.var["genes"] = all_adata.var.index.tolist()

    if print_results:
        print(f"Merged all_adata shape = {all_adata.shape}")
        print(f"Sample key added to all_adata.obs: {sample_key}")

    results = {
        "ref_adata_dic_raw": ref_adata_dic_raw,
        "ref_adata_dic_filtered": ref_adata_dic_filtered,
        "all_adata": all_adata,
        "common_genes": common_genes,
        "sample_names": ref_section_list
    }

    return results


def construct_merged_scaled_adata_and_gene_df(
    ref_adata_dic,
    tissue_section_list,
    total_genes_list=None,
    merged_key="sample",
    normalize=True,
    normalize_method="min_max",
    print_results=True,
):
    """
    Construct a merged AnnData object and gene-expression dataframe across tissue sections.

    This function can work in two modes:

        1. Marker-gene mode:
           If `total_genes_list` is provided and non-empty, each tissue section is
           subset to the available genes from `total_genes_list`.

        2. Shared-gene mode:
           If `total_genes_list` is None or empty, the function automatically
           identifies genes shared by all tissue sections and uses those genes.

    For each tissue section, this function:
        1. checks whether the section exists in `ref_adata_dic`,
        2. selects marker genes or shared genes,
        3. optionally normalizes the selected gene-expression matrix within each section,
        4. concatenates all AnnData objects across sections,
        5. converts the merged expression matrix into a pandas DataFrame.

    Parameters
    ----------
    ref_adata_dic : dict
        Dictionary of reference-tissue-section AnnData objects.

        Example:
        {
            "H1": adata_H1,
            "G2": adata_G2,
            "E1": adata_E1,
        }

    tissue_section_list : list
        List of tissue-section names to merge.

    total_genes_list : list or None, default=None
        Union of selected subtype marker genes across tissue sections.

        If provided and non-empty, the function uses genes from this list that are
        available in each tissue section.

        If None or empty, the function automatically identifies genes shared by
        all tissue sections and uses those shared genes.

    merged_key : str, default="sample"
        Column name added to `.obs` of the merged AnnData object to record
        the tissue-section/source sample.

    normalize : bool, default=True
        Whether to normalize each sample after gene filtering.

    normalize_method : str or None, default="min_max"
        Normalization method passed to `normalize_adata`.

    print_results : bool, default=True
        Whether to print merged AnnData variable information and sample counts.

    Returns
    -------
    merged_adata : AnnData
        Concatenated AnnData object across all tissue sections.

    gene_df : pandas.DataFrame
        Dense gene-expression dataframe from `merged_adata.X`.
        Rows are spots/cells and columns are genes.

    Raises
    ------
    KeyError
        If a tissue section in `tissue_section_list` is not present in `ref_adata_dic`.

    ValueError
        If `tissue_section_list` is empty, if a section has no genes, or if no
        shared genes can be found across tissue sections.
    """

    if tissue_section_list is None or len(tissue_section_list) == 0:
        raise ValueError("tissue_section_list is empty.")

    for tissue_section in tissue_section_list:
        if tissue_section not in ref_adata_dic:
            raise KeyError(f"{tissue_section!r} is not present in ref_adata_dic.")

        if ref_adata_dic[tissue_section].shape[1] == 0:
            raise ValueError(f"{tissue_section!r} has no genes/features.")

    # ------------------------------------------------------------------
    # Determine which genes to use
    # ------------------------------------------------------------------
    if total_genes_list is None or len(total_genes_list) == 0:
        # Use genes shared across all tissue sections.
        shared_genes = set(ref_adata_dic[tissue_section_list[0]].var_names)

        for tissue_section in tissue_section_list[1:]:
            shared_genes = shared_genes.intersection(
                set(ref_adata_dic[tissue_section].var_names)
            )

        if len(shared_genes) == 0:
            raise ValueError(
                "No shared genes are available across all tissue sections."
            )

        # Preserve the gene order from the first tissue section.
        genes_to_use = [
            g for g in ref_adata_dic[tissue_section_list[0]].var_names
            if g in shared_genes
        ]

        if print_results:
            print(
                "total_genes_list is empty. "
                f"Using {len(genes_to_use)} shared genes across tissue sections."
            )

    else:
        # Use user-provided marker genes, removing duplicates while preserving order.
        genes_to_use = list(dict.fromkeys(total_genes_list))

        if print_results:
            print(f"Using {len(genes_to_use)} genes from total_genes_list.")

    # ------------------------------------------------------------------
    # Subset, optionally normalize, and concatenate AnnData objects
    # ------------------------------------------------------------------
    adata_list = []

    for tissue_section in tissue_section_list:
        test_gene = ref_adata_dic[tissue_section]

        if not test_gene.obs_names.is_unique:
            raise ValueError(
                f"Observation names must be unique within {tissue_section!r} "
                "before merging sections."
            )

        available_genes = [
            g for g in genes_to_use
            if g in test_gene.var_names
        ]

        if len(available_genes) == 0:
            raise ValueError(
                f"No selected/shared genes are available in {tissue_section}."
            )

        test_gene_sub = test_gene[:, available_genes].copy()

        if normalize:
            test_gene_sub = normalize_adata(
                test_gene_sub,
                method=normalize_method,
                copy=False,
            )

        adata_list.append(test_gene_sub)

    merged_adata = ad.concat(
        adata_list,
        axis=0,
        join="inner",
        label=merged_key,
        keys=tissue_section_list,
        index_unique="-",
    )

    if not merged_adata.obs_names.is_unique:
        raise RuntimeError("Merged observation names are not unique after section suffixing.")

    merged_adata.var["genes"] = merged_adata.var.index.tolist()

    if print_results:
        print(merged_adata.var)
        print(merged_adata.obs[merged_key].value_counts())
        print(f"Final merged gene number: {merged_adata.shape[1]}")

    if issparse(merged_adata.X):
        X = merged_adata.X.toarray()
    else:
        X = np.asarray(merged_adata.X)

    gene_df = pd.DataFrame(
        X,
        index=merged_adata.obs.index,
        columns=merged_adata.var_names,
    )

    return merged_adata, gene_df


def subset_adata_dic_by_region(
    ref_adata_dic,
    target_region,
    label_key="label",
    min_spots=10,
    copy=True,
    print_results=True,
):
    """
    Subset each sample AnnData to one target tissue region.

    Parameters
    ----------
    ref_adata_dic : dict
        Dictionary of reference-sample-level AnnData objects.

    target_region : str
        Tissue region to keep.

    label_key : str, default="label"
        Column in `.obs` containing tissue-region labels.

    min_spots : int, default=10
        Minimum number of spots required for a sample to be retained.

    copy : bool, default=True
        Whether to return copied AnnData objects.

    print_results : bool, default=True
        Whether to print retained samples and spot counts.

    Returns
    -------
    region_adata_dic : dict
        Dictionary of sample-level AnnData objects restricted to target_region.

    retained_sections : list
        Samples retained for subtype analysis.
    """

    region_adata_dic = {}
    retained_sections = []

    for sample_name, sample_adata in ref_adata_dic.items():
        if label_key not in sample_adata.obs.columns:
            raise KeyError(f"{sample_name}: {label_key!r} is not found in adata.obs.")

        mask = sample_adata.obs[label_key].astype(str) == str(target_region)
        n_region_spots = int(mask.sum())

        if n_region_spots >= min_spots:
            region_adata = sample_adata[mask].copy() if copy else sample_adata[mask]
            region_adata_dic[sample_name] = region_adata
            retained_sections.append(sample_name)

        if print_results:
            print(f"{sample_name}: {target_region}, n_spots={n_region_spots}")

    return region_adata_dic, retained_sections


def make_nonnegative_adata(input_adata, copy=True):
    """
    Shift AnnData.X to be non-negative feature-wise.

    For each gene/feature, subtract its minimum value across all spots/cells.
    This preserves relative differences within each feature while ensuring
    all values are >= 0.

    Parameters
    ----------
    input_adata : AnnData
        Input AnnData object.

    copy : bool, default=True
        Whether to return a copied AnnData object.

    Returns
    -------
    output_adata : AnnData
        AnnData object with non-negative X.
    """

    output_adata = input_adata.copy() if copy else input_adata

    X = output_adata.X

    if issparse(X):
        X = X.toarray()
    else:
        X = np.asarray(X)

    X_min = X.min(axis=0)
    X_nonneg = X - X_min

    output_adata.X = X_nonneg

    return output_adata


# ============================================================
# Raw-data preprocessing pipeline helpers
# ============================================================
def create_preprocess_output_dirs(preprocess_dir, cohort):
    """Create and return the standard folders for one cohort.

    Parameters
    ----------
    preprocess_dir : str or pathlib.Path
        Root preprocessing directory.
    cohort : {"reference", "query"}
        Cohort whose folders should be created.

    Returns
    -------
    PreprocessPaths
        Paths to the raw-input, final-output, QC, annotation, and image-feature
        directories. Existing directories are kept.
    """
    cohort = str(cohort).strip().lower()
    if cohort not in {"reference", "query"}:
        raise ValueError("cohort must be either 'reference' or 'query'.")

    cohort_dir = Path(preprocess_dir).expanduser() / cohort
    raw_dir = cohort_dir / "raw"
    preprocessed_dir = cohort_dir / "preprocessed"
    paths = PreprocessPaths(
        cohort_dir=cohort_dir,
        raw_dir=raw_dir,
        preprocessed_dir=preprocessed_dir,
        contour_dir=preprocessed_dir / "contours",
        scribble_dir=preprocessed_dir / "extracted_scribbles",
        image_spot_dir=preprocessed_dir / "image_features" / "spot",
        image_enhanced_dir=preprocessed_dir / "image_features" / "enhanced",
    )
    for path in (
        paths.raw_dir,
        paths.preprocessed_dir,
        paths.contour_dir,
        paths.scribble_dir,
        paths.image_spot_dir,
        paths.image_enhanced_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return paths


def resolve_section_file(
    input_dir,
    section,
    file_template,
    extensions=(".jpg", ".jpeg", ".png", ".tif", ".tiff"),
):
    """Resolve a section-level input file, optionally trying extensions.

    ``file_template`` must contain ``{section}``. It may also contain
    ``{ext}``, for example ``"{section}_image{ext}"``. When ``{ext}`` is
    present, extensions are tried in the supplied order.

    Returns
    -------
    pathlib.Path
        Existing resolved file.
    """
    input_dir = Path(input_dir).expanduser()
    if "{section}" not in file_template:
        raise ValueError("file_template must contain the '{section}' placeholder.")

    if "{ext}" in file_template:
        candidates = [
            input_dir / file_template.format(section=section, ext=extension)
            for extension in extensions
        ]
    else:
        candidates = [input_dir / file_template.format(section=section)]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    tried = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"No input file was found for section {section!r}. Tried:\n  - {tried}"
    )


def read_he_image(image_path, color_order="bgr"):
    """Read an H&E image as a NumPy array.

    Parameters
    ----------
    image_path : str or pathlib.Path
        Image readable by OpenCV.
    color_order : {"bgr", "rgb"}, default="bgr"
        Desired channel order in the returned array.

    Returns
    -------
    numpy.ndarray
        Three-channel uint8 image.
    """
    color_order = str(color_order).lower()
    if color_order not in {"bgr", "rgb"}:
        raise ValueError("color_order must be either 'bgr' or 'rgb'.")

    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "Reading H&E images requires the optional image dependencies. "
            "Install HiCAT with `pip install -e '.[image]'`."
        ) from exc

    image_path = Path(image_path).expanduser()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")
    if color_order == "rgb":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def standardize_feature_names(
    input_adata,
    uppercase=True,
    feature_key="genes",
    make_unique=True,
    copy=True,
):
    """Standardize AnnData feature names while preserving their order.

    Parameters
    ----------
    input_adata : AnnData
        Input molecular feature object.
    uppercase : bool, default=True
        Convert feature names to uppercase. This is usually useful for gene
        and protein matching across sections.
    feature_key : str or None, default="genes"
        Optional ``.var`` column receiving the finalized feature names.
    make_unique : bool, default=True
        Make duplicated names unique using AnnData's standard suffixes.
    copy : bool, default=True
        Return a copy instead of modifying the input.

    Returns
    -------
    AnnData
        Object with standardized ``.var_names``.
    """
    output_adata = input_adata.copy() if copy else input_adata
    names = pd.Index(output_adata.var_names.astype(str))
    if uppercase:
        names = pd.Index([name.upper() for name in names])
    output_adata.var_names = names
    if make_unique:
        output_adata.var_names_make_unique()
    if feature_key is not None:
        output_adata.var[feature_key] = output_adata.var_names.astype(str)
    return output_adata


def normalize_log1p_adata(
    input_adata,
    target_sum=10_000,
    log1p=True,
    copy=True,
):
    """Normalize every observation to a common total and optionally log1p.

    Parameters
    ----------
    input_adata : AnnData
        Raw non-negative count/intensity object.
    target_sum : float or None, default=10000
        Target total per observation passed to ``scanpy.pp.normalize_total``.
        Use ``None`` to skip total normalization.
    log1p : bool, default=True
        Apply ``log(1 + x)`` after total normalization.
    copy : bool, default=True
        Return a copy instead of modifying the input.

    Returns
    -------
    AnnData
        Normalized object. Sparse input remains sparse unless another step
        explicitly requires a dense matrix.
    """
    output_adata = input_adata.copy() if copy else input_adata
    if target_sum is not None:
        if target_sum <= 0:
            raise ValueError("target_sum must be positive or None.")
        sc.pp.normalize_total(output_adata, target_sum=float(target_sum))
    if log1p:
        sc.pp.log1p(output_adata)
    return output_adata


def replace_zeros_with_small_values(
    input_adata,
    scale=0.01,
    random_state=0,
    copy=True,
):
    """Optionally replace molecular zeros with small positive random values.

    For each feature, replacement values are sampled uniformly between zero
    and ``scale * smallest_positive_value`` for that feature. All-zero
    features remain zero. This operation densifies sparse matrices and should
    generally be reserved for modest protein panels whose downstream method
    cannot accept exact zeros.

    Parameters
    ----------
    input_adata : AnnData
        Input molecular data.
    scale : float, default=0.01
        Upper bound relative to each feature's smallest positive value.
    random_state : int or numpy.random.Generator, default=0
        Reproducible random-number source.
    copy : bool, default=True
        Return a copy instead of modifying the input.

    Returns
    -------
    AnnData
        Object with eligible zero entries replaced in ``.X``.
    """
    if not 0 < scale <= 1:
        raise ValueError("scale must be in (0, 1].")
    output_adata = input_adata.copy() if copy else input_adata
    if issparse(output_adata.X):
        warnings.warn(
            "replace_zeros_with_small_values densifies sparse AnnData.X.",
            RuntimeWarning,
            stacklevel=2,
        )
        X = output_adata.X.toarray().astype(float, copy=False)
    else:
        X = np.asarray(output_adata.X, dtype=float).copy()

    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    for feature_index in range(X.shape[1]):
        values = X[:, feature_index]
        positive = values[values > 0]
        zero_mask = values == 0
        if positive.size and zero_mask.any():
            upper = float(positive.min()) * float(scale)
            values[zero_mask] = rng.uniform(0.0, upper, size=int(zero_mask.sum()))
    output_adata.X = X
    return output_adata


def preprocess_molecular_adata(
    input_adata,
    target_sum=10_000,
    log1p=True,
    uppercase_features=True,
    feature_key="genes",
    replace_zeros=False,
    zero_replacement_scale=0.01,
    random_state=0,
    copy=True,
):
    """Apply the standard raw gene/protein preprocessing workflow.

    The operation order is total normalization, optional log1p, optional zero
    replacement, and feature-name standardization. Zero replacement is off by
    default and is intended mainly for small protein panels.

    Returns
    -------
    AnnData
        Preprocessed object with normalized ``.X`` and standardized
        ``.var_names``/``.var[feature_key]``.
    """
    output_adata = normalize_log1p_adata(
        input_adata,
        target_sum=target_sum,
        log1p=log1p,
        copy=copy,
    )
    if replace_zeros:
        output_adata = replace_zeros_with_small_values(
            output_adata,
            scale=zero_replacement_scale,
            random_state=random_state,
            copy=False,
        )
    return standardize_feature_names(
        output_adata,
        uppercase=uppercase_features,
        feature_key=feature_key,
        make_unique=True,
        copy=False,
    )


def preprocess_molecular_sections(
    section_list,
    raw_dir,
    modality,
    file_template=None,
    output_dir=None,
    **preprocess_kwargs,
):
    """Load and preprocess one molecular modality across sections.

    Parameters
    ----------
    section_list : sequence of str
        Section identifiers.
    raw_dir : str or pathlib.Path
        Directory containing raw ``.h5ad`` inputs.
    modality : {"Gene", "Protein"}
        Molecular modality used in default input/output filenames.
    file_template : str, optional
        Raw filename template. Defaults to
        ``"{section}_<modality>_raw.h5ad"``.
    output_dir : str or pathlib.Path, optional
        If supplied, save each processed object as
        ``"{section}_<modality>.h5ad"``.
    **preprocess_kwargs
        Parameters forwarded to :func:`preprocess_molecular_adata`.

    Returns
    -------
    dict[str, AnnData]
        Section-to-AnnData mapping in ``section_list`` order.
    """
    modality_name = str(modality).strip().lower()
    if modality_name not in {"gene", "protein"}:
        raise ValueError("modality must be either 'Gene' or 'Protein'.")
    if file_template is None:
        file_template = f"{{section}}_{modality_name}_raw.h5ad"
    if "{section}" not in file_template:
        raise ValueError("file_template must contain the '{section}' placeholder.")

    raw_dir = Path(raw_dir).expanduser()
    output_dir = Path(output_dir).expanduser() if output_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    result = {}
    for section in section_list:
        input_path = raw_dir / file_template.format(section=section)
        if not input_path.is_file():
            raise FileNotFoundError(f"Raw {modality_name} AnnData not found: {input_path}")
        output_adata = preprocess_molecular_adata(
            sc.read_h5ad(input_path),
            **preprocess_kwargs,
        )
        result[section] = output_adata
        if output_dir is not None:
            output_adata.write_h5ad(
                output_dir / f"{section}_{modality_name}.h5ad"
            )
    return result


def transfer_obs_columns(source_adata, target_adata, columns, copy=True):
    """Copy observation columns between aligned AnnData objects by obs name.

    The target observation order is preserved. A clear error is raised if any
    target observation is absent from the source.
    """
    output_adata = target_adata.copy() if copy else target_adata
    columns = [columns] if isinstance(columns, str) else list(columns)
    missing_columns = [column for column in columns if column not in source_adata.obs]
    if missing_columns:
        raise KeyError(f"Source adata.obs is missing columns: {missing_columns}")
    missing_obs = output_adata.obs_names.difference(source_adata.obs_names)
    if len(missing_obs):
        raise ValueError(
            f"{len(missing_obs)} target observations are absent from the source; "
            f"examples: {missing_obs[:5].tolist()}"
        )
    aligned = source_adata.obs.reindex(output_adata.obs_names)
    for column in columns:
        output_adata.obs[column] = aligned[column].copy()
    return output_adata


def transfer_labels_by_nearest_spot(
    source_adata,
    target_adata,
    label_key="label",
    x_key="pixel_x",
    y_key="pixel_y",
    copy=True,
):
    """Transfer labels to new coordinates from the nearest observed spot.

    This is a practical fallback for enhanced pseudo spots when reference
    labels already exist in ``source_adata.obs`` but no image annotation mask
    is available. Coordinates in both objects must use the same pixel space.

    Returns
    -------
    AnnData
        Target object with a categorical ``.obs[label_key]`` column.
    """
    from sklearn.neighbors import NearestNeighbors

    if label_key not in source_adata.obs:
        raise KeyError(f"Source adata.obs does not contain {label_key!r}.")
    for name, adata_obj in (("source", source_adata), ("target", target_adata)):
        missing = [key for key in (x_key, y_key) if key not in adata_obj.obs]
        if missing:
            raise KeyError(f"{name} adata.obs is missing coordinate columns: {missing}")
    if source_adata.n_obs == 0:
        raise ValueError("source_adata has no observations.")

    output_adata = target_adata.copy() if copy else target_adata
    source_xy = source_adata.obs.loc[:, [x_key, y_key]].to_numpy(dtype=float)
    target_xy = output_adata.obs.loc[:, [x_key, y_key]].to_numpy(dtype=float)
    if not np.isfinite(source_xy).all() or not np.isfinite(target_xy).all():
        raise ValueError("Spatial coordinates must contain only finite values.")

    model = NearestNeighbors(n_neighbors=1).fit(source_xy)
    nearest_indices = model.kneighbors(target_xy, return_distance=False).ravel()
    labels = source_adata.obs[label_key].iloc[nearest_indices].to_numpy()
    output_adata.obs[label_key] = pd.Categorical(labels)
    return output_adata


def save_spot_coordinates(
    input_adata,
    output_path,
    columns=None,
    index_label="spot_id",
):
    """Save spot metadata/coordinates while retaining observation identifiers.

    Parameters
    ----------
    input_adata : AnnData
        Source object.
    output_path : str or pathlib.Path
        Destination CSV path.
    columns : sequence of str, optional
        Columns to save. By default all ``.obs`` columns are retained.
    index_label : str, default="spot_id"
        CSV name for ``adata.obs_names``.

    Returns
    -------
    pathlib.Path
        Written CSV path.
    """
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    obs = input_adata.obs.copy()
    if columns is not None:
        columns = list(columns)
        missing = [column for column in columns if column not in obs]
        if missing:
            raise KeyError(f"adata.obs is missing coordinate columns: {missing}")
        obs = obs.loc[:, columns]
    obs.to_csv(output_path, index=True, index_label=index_label)
    return output_path


def remove_obs_columns_by_prefix(input_adata, prefixes=("kmeans_",), copy=True):
    """Remove temporary ``.obs`` columns whose names start with a prefix.

    Returns a copied AnnData by default. This is useful for dropping image
    feature QC clusters before storing a clean pipeline input.
    """
    output_adata = input_adata.copy() if copy else input_adata
    prefixes = (prefixes,) if isinstance(prefixes, str) else tuple(prefixes)
    columns = [
        column
        for column in output_adata.obs.columns
        if str(column).startswith(prefixes)
    ]
    if columns:
        output_adata.obs.drop(columns=columns, inplace=True)
    return output_adata


def restrict_hierarchical_genes(
    gene_select_res,
    provided_genes,
    inplace=False,
):
    """
    Restrict hierarchical gene-selection results to an allowed gene set.

    The function filters genes stored in both the direction-specific anchor
    feature lists and the combined clustering feature lists. If raw results
    are retained, ``hier_genes_dic`` and ``hier_genenum`` are also updated.

    Parameters
    ----------
    gene_select_res : HierarchicalFeatureResults
        Result returned by ``select_hierarchical_genes_pipeline()``.

        The following contents are filtered:

        - ``split_features_dic[parent_node].anchor_features_dic``
        - ``split_features_dic[parent_node].clustering_features_list``
        - ``raw_results_dic["hier_genes_dic"]``, when available
        - ``raw_results_dic["hier_genenum"]``, when available

    provided_genes : sequence of str or set of str
        Allowed genes. A selected hierarchical gene is retained only when it
        appears in this collection. Matching is exact and case-sensitive.

    inplace : bool, default=False
        Whether to modify ``gene_select_res`` directly.

        - If ``False``, create and return a deep copy.
        - If ``True``, update and return the original object.

    Returns
    -------
    filtered_res : HierarchicalFeatureResults
        Filtered hierarchical feature-selection result with the same hierarchy,
        split metadata, reference sections, and anchor scenario as the input.

        Retrieve filtered features using:

        ```python
        filtered_res.get_anchor_features_dic(parent_node)
        filtered_res.get_clustering_features(parent_node)
        filtered_res.get_direction_features(
            parent_node=parent_node,
            direction=direction,
            section=section,
        )
        ```

    Notes
    -----
    The order of retained genes follows their original selected-feature order.

    If a ``MultimodalHierarchicalFeatureResults`` object was constructed before
    filtering, reconstruct it afterward so its cached feature dictionaries use
    the filtered gene lists.
    """

    result = gene_select_res if inplace else deepcopy(gene_select_res)
    allowed_genes = set(provided_genes)

    # Update organized ParentSplitFeatures results
    for split_res in result.split_features_dic.values():

        if split_res.anchor_scenario == "nn_based":
            # section -> direction -> genes
            for direction_dic in split_res.anchor_features_dic.values():
                for direction, genes in direction_dic.items():
                    direction_dic[direction] = [
                        gene for gene in genes if gene in allowed_genes
                    ]

        else:  # quantile_based
            # direction -> genes
            for direction, genes in split_res.anchor_features_dic.items():
                split_res.anchor_features_dic[direction] = [
                    gene for gene in genes if gene in allowed_genes
                ]

        split_res.clustering_features_list = [
            gene
            for gene in split_res.clustering_features_list
            if gene in allowed_genes
        ]

    # Keep optional raw results consistent
    raw = result.raw_results_dic

    if raw is not None:
        raw_genes = raw.get("hier_genes_dic", {})

        for section_dic in raw_genes.values():
            for direction, genes in section_dic.items():
                section_dic[direction] = [
                    gene for gene in genes if gene in allowed_genes
                ]

        # Update raw gene-count summaries
        split_info_dic = raw.get("split_info", {})
        gene_count_dic = raw.get("hier_genenum", {})

        for section, section_counts in gene_count_dic.items():
            section_genes = raw_genes.get(section, {})

            for parent_node, split_info in split_info_dic.items():
                split_key_1 = split_info["split_key_1"]
                split_key_2 = split_info["split_key_2"]

                section_counts[parent_node] = (
                    len(section_genes.get(split_key_1, [])),
                    len(section_genes.get(split_key_2, [])),
                )

    return result
