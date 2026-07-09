"""End-to-end preprocessing entry point for HiCAT.

This module turns flat-folder raw gene/protein AnnData objects and H&E images
into consistently named preprocessing outputs.

Typical use
-----------
::

    from hicat_spatial import (
        PreprocessConfig,
        run_preprocessing_pipeline,
    )

    config = PreprocessConfig(
        data_dir="./data",
        preprocess_dir="./results/01_preprocessing",
        reference_sections=["ref_1", "ref_2"],
        query_sections=["query_1"],
        modalities=("Gene", "Image", "Protein"),
        raw_file_mode="copy",
        gene_enhancement=True,
        label_color_dict={"tumor": (255, 0, 0), "stroma": (0, 255, 0)},
        image_feature_levels=("spot", "enhanced"),
        image_feature_kwargs={
            "model": "uni",
            "checkpoint_path": "./checkpoints/pytorch_model.bin",
            "n_clusters": (5, 10, 15),
        },
        image_feature_level_kwargs={
            "spot": {"patch_size_spot": 250},
            "enhanced": {"patch_size_spot": 50},
        },
    )
    result = run_preprocessing_pipeline(config)
"""

from __future__ import annotations

import shutil
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import anndata as ad
import numpy as np

from ..preprocessing.preprocess_util import (
    PreprocessPaths,
    create_preprocess_output_dirs,
    preprocess_molecular_sections,
    read_he_image,
    remove_obs_columns_by_prefix,
    resolve_section_file,
    save_spot_coordinates,
    transfer_labels_by_nearest_spot,
    transfer_obs_columns,
)
from ._io import logged_stage, save_json, save_stage_result, stage_output_from_config


_MODALITY_ORDER = ("Gene", "Image", "Protein")
_IMAGE_FEATURE_LEVELS = ("spot", "enhanced")
_IMAGE_FEATURE_AGGREGATION_OVERRIDE_KEYS = {
    "patch_size_spot",
    "aggregation_method",
    "normalize_by",
    "ignore_zero_features",
    "zero_tol",
    "npcs",
    "n_clusters",
    "ncluster_list",
    "plot_clusters",
    "plot_spot_size",
    "cat_color",
    "dpi",
    "invert_x",
    "invert_y",
    "random_state",
    "save_h5ad",
    "spatial_key",
    "spatial_coords_are_pixel",
}
_IMAGE_FEATURE_MODEL_SPECIFIC_KEYS = {
    "mask",
    "pad_size",
    "reduction_method",
    "n_components",
    "smoothen_method",
    "random_weights",
    "no_shift",
    "use_cache",
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
_IMAGE_FEATURE_WORKSPACE_KEYS = {
    "model",
    "checkpoint_path",
    "patch_size_emb",
    "device",
    "raw_image_name",
    "overwrite_raw_image",
} | _IMAGE_FEATURE_MODEL_SPECIFIC_KEYS
_IMAGE_FEATURE_AGGREGATION_KEYS = {
    "patch_size_spot",
    "patch_size_emb",
} | _IMAGE_FEATURE_AGGREGATION_OVERRIDE_KEYS | _IMAGE_FEATURE_MODEL_SPECIFIC_KEYS


@dataclass
class PreprocessConfig:
    """Parameters for :func:`run_preprocessing_pipeline`.

    Parameters
    ----------
    data_dir : path-like
        User-created flat folder containing raw input files. Expected molecular
        defaults are ``{section}_ref_gene_raw.h5ad`` and
        ``{section}_query_gene_raw.h5ad``; images use
        ``{section}_image{ext}``.
    preprocess_dir : path-like
        Package-managed preprocessing/output folder. Stage 1 creates
        ``reference/raw``, ``query/raw``, ``preprocessed`` and related
        subfolders automatically.
    reference_sections, query_sections : sequence[str]
        Unique section IDs used in all input filenames.
    modalities : sequence[str], default=("Gene", "Image")
        Exact available modalities from ``"Gene"``, ``"Image"``, and
        ``"Protein"``. Outputs use that canonical order.
    raw_file_mode : {"copy", "symlink", "none"}, default="copy"
        How flat raw files from ``data_dir`` are handled. ``"copy"`` copies
        them into ``preprocess_dir/<cohort>/raw``. ``"symlink"`` creates
        symbolic links there. ``"none"`` reads directly from ``data_dir`` and
        leaves the package-created raw folders empty.
    target_sum : float or None, default=10_000
        Per-observation molecular total. ``None`` skips total normalization.
        To keep molecular values on their original scale, use
        ``target_sum=None`` together with ``log1p=False``.
    log1p : bool, default=True
        Apply ``log1p`` after optional total normalization. Set to ``False``
        to skip log transformation.
    uppercase_features : bool, default=True
        Uppercase and uniquify molecular feature names.
    protein_replace_zeros : bool, default=False
        Replace protein zeros with small values. Off by default because it
        densifies ``.X`` and changes the meaning of exact zeros.
    zero_replacement_scale : float, default=0.01
        Maximum replacement value relative to the smallest positive feature
        value.
    random_state : int, default=0
        Seed used by zero replacement and forwarded stochastic operations.
    gene_enhancement, protein_enhancement : bool, default=False
        Generate dense pseudo-spot molecular data from H&E and observed spots.
    enhancement_kwargs : dict, default={}
        Additional arguments for ``enhance_gene_expression``, such as
        ``resolution``, ``contour_method``, ``n_neighbors``,
        ``histology_scale``, or ``max_pseudo_spots``.
    label_color_dict : mapping[str, tuple[int, int, int]] or None, default=None
        Optional mapping of reference scribble label to RGB color. When
        provided, Stage 1 extracts labels from annotated reference images.
        When ``None``, no scribble extraction is performed; at least one
        reference molecular modality should already contain
        ``adata.obs[label_key]`` for every reference section.
    scribble_kwargs : dict, default={}
        Additional arguments for ``extract_scribble_labels_pipeline``, such as
        ``color_tolerance``, ``selected_labels_dic``, or plotting settings.
    image_feature_mode : {"extract", "load"}, default="extract"
        How to create Image modality objects. ``"extract"`` runs UNI/HIPT from
        raw H&E images. ``"load"`` reads pre-extracted image-feature ``.h5ad``
        files from ``data_dir`` and saves them into the standard preprocessing
        output folders.
    image_feature_kwargs : dict, default={}
        Arguments for ``extract_image_features`` when
        ``image_feature_mode="extract"``, including ``model``,
        ``checkpoint_path``, patch settings, clustering, and plot settings.
    image_feature_levels : {"auto", "spot", "enhanced"} or sequence[str], default="auto"
        Final Image feature levels to create when ``image_feature_mode="extract"``.
        ``"auto"`` creates enhanced Image features when molecular enhancement
        is enabled and enhanced coordinates are available; otherwise it creates
        spot-level Image features. Use ``("spot", "enhanced")`` to save both.
    image_feature_level_kwargs : dict, default={}
        Per-level aggregation overrides used after the shared HIPT/UNI
        embedding grid is extracted once. For example,
        ``{"spot": {"patch_size_spot": 250}, "enhanced": {"patch_size_spot": 50}}``.
    x_key, y_key : str, default=("pixel_x", "pixel_y")
        Image-pixel coordinate columns in every molecular ``adata.obs``.
        ``x_key`` should be the horizontal image coordinate (column, bounded
        by image width), and ``y_key`` should be the vertical coordinate (row,
        bounded by image height). Some datasets store coordinates as
        ``(row, column)`` or ``(y, x)``; in that case switch the keys.
    array_x_key, array_y_key : str, default=("array_x", "array_y")
        Array-grid coordinates used by scan-based contour candidates.
    label_key : str, default="label"
        Reference tissue-region annotation column.
    reference_gene_template, query_gene_template : str or None
        Flat-folder spot-level Gene input filename templates. Set to ``None``
        to skip spot-level Gene loading when only enhanced Gene objects are
        available.
    reference_protein_template, query_protein_template : str or None
        Flat-folder spot-level Protein input filename templates. Set to
        ``None`` to skip spot-level Protein loading.
    reference_enhanced_gene_template, query_enhanced_gene_template : str or None
        Optional flat-folder precomputed enhanced Gene ``.h5ad`` templates.
    reference_enhanced_protein_template, query_enhanced_protein_template : str or None
        Optional flat-folder precomputed enhanced Protein ``.h5ad`` templates.
    reference_image_template, query_image_template : str
        Flat-folder raw H&E filename templates. ``{ext}``, if present, tries
        jpg, jpeg, png, tif, and tiff.
    reference_annotated_image_template : str
        Flat-folder reference annotated-H&E filename template used when
        ``label_color_dict`` is supplied.
    reference_image_feature_template, query_image_feature_template : str
        Flat-folder pre-extracted spot-level Image feature ``.h5ad`` templates
        used when ``image_feature_mode="load"``.
    reference_enhanced_image_feature_template, query_enhanced_image_feature_template : str or None
        Optional flat-folder pre-extracted enhanced-grid Image feature
        ``.h5ad`` templates used when ``image_feature_mode="load"``. Leave as
        ``None`` to skip enhanced Image features.
    color_order : {"bgr", "rgb"}, default="bgr"
        H&E channel order supplied to enhancement. ``"bgr"`` matches OpenCV.
    """

    data_dir: Path | str
    preprocess_dir: Path | str
    reference_sections: Sequence[str]
    query_sections: Sequence[str]
    modalities: Sequence[str] = ("Gene", "Image")
    raw_file_mode: str = "copy"
    target_sum: Optional[float] = 10_000
    log1p: bool = True
    uppercase_features: bool = True
    protein_replace_zeros: bool = False
    zero_replacement_scale: float = 0.01
    random_state: int = 0
    gene_enhancement: bool = False
    protein_enhancement: bool = False
    enhancement_kwargs: Dict[str, Any] = field(default_factory=dict)
    label_color_dict: Optional[Mapping[str, Tuple[int, int, int]]] = None
    scribble_kwargs: Dict[str, Any] = field(default_factory=dict)
    image_feature_mode: str = "extract"
    image_feature_kwargs: Dict[str, Any] = field(default_factory=dict)
    image_feature_levels: str | Sequence[str] = "auto"
    image_feature_level_kwargs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    x_key: str = "pixel_x"
    y_key: str = "pixel_y"
    array_x_key: str = "array_x"
    array_y_key: str = "array_y"
    label_key: str = "label"
    reference_gene_template: Optional[str] = "{section}_ref_gene_raw.h5ad"
    query_gene_template: Optional[str] = "{section}_query_gene_raw.h5ad"
    reference_protein_template: Optional[str] = "{section}_ref_protein_raw.h5ad"
    query_protein_template: Optional[str] = "{section}_query_protein_raw.h5ad"
    reference_enhanced_gene_template: Optional[str] = None
    query_enhanced_gene_template: Optional[str] = None
    reference_enhanced_protein_template: Optional[str] = None
    query_enhanced_protein_template: Optional[str] = None
    reference_image_template: str = "{section}_image{ext}"
    query_image_template: str = "{section}_image{ext}"
    reference_annotated_image_template: str = "{section}_annotated_image{ext}"
    reference_image_feature_template: str = "{section}_ref_image_features.h5ad"
    query_image_feature_template: str = "{section}_query_image_features.h5ad"
    reference_enhanced_image_feature_template: Optional[str] = None
    query_enhanced_image_feature_template: Optional[str] = None
    color_order: str = "bgr"


@dataclass
class PreprocessPipelineResult:
    """In-memory preprocessing outputs and their disk locations.

    ``reference`` and ``query`` each use this nested structure::

        {
            "spot": {
                "Gene": {section: AnnData},
                "Protein": {section: AnnData},
                "Image": {section: AnnData},
            },
            "enhanced": {
                "Gene": {section: AnnData},
                "Protein": {section: AnnData},
                "Image": {section: AnnData},
            },
        }

    Attributes
    ----------
    reference, query : dict
        Nested objects shown above.
    annotation_results : dict[str, dict]
        Reference scribble masks and label mappings keyed by section; empty
        when color-based annotations were not extracted.
    paths : dict[str, PreprocessPaths]
        ``{"reference": paths, "query": paths}`` with resolved raw, final,
        contour, scribble, and image-feature directories.
    """

    reference: Dict[str, Dict[str, Dict[str, ad.AnnData]]]
    query: Dict[str, Dict[str, Dict[str, ad.AnnData]]]
    annotation_results: Dict[str, Dict[str, Any]]
    paths: Dict[str, PreprocessPaths]

    def get_adata(self, cohort, level, modality, section):
        """Return one AnnData using nested output keys.

        Parameters
        ----------
        cohort : {"reference", "query"}
        level : {"spot", "enhanced"}
        modality : {"Gene", "Image", "Protein"}
        section : str

        Returns
        -------
        AnnData
            The requested processed object. Missing keys raise ``KeyError``.
        """
        cohort = str(cohort).lower()
        if cohort not in {"reference", "query"}:
            raise ValueError("cohort must be 'reference' or 'query'.")
        cohort_result = self.reference if cohort == "reference" else self.query
        return cohort_result[level][modality][section]


def _canonical_modalities(modalities: Sequence[str]) -> Tuple[str, ...]:
    requested = {str(modality).strip().lower() for modality in modalities}
    unknown = requested - {name.lower() for name in _MODALITY_ORDER}
    if unknown:
        raise ValueError(f"Unsupported modalities: {sorted(unknown)}")
    return tuple(name for name in _MODALITY_ORDER if name.lower() in requested)


def _normalize_image_feature_levels(levels) -> Tuple[str, ...] | None:
    """Return explicit image-feature levels, or None for auto mode."""
    if isinstance(levels, str):
        value = levels.strip().lower()
        if value == "auto":
            return None
        requested = (value,)
    else:
        requested = tuple(str(level).strip().lower() for level in levels)

    if not requested:
        raise ValueError("image_feature_levels must not be empty.")

    unknown = set(requested) - set(_IMAGE_FEATURE_LEVELS)
    if unknown:
        raise ValueError(
            "image_feature_levels must be 'auto', 'spot', 'enhanced', or a "
            f"sequence containing those levels. Unsupported: {sorted(unknown)}."
        )

    ordered = tuple(level for level in _IMAGE_FEATURE_LEVELS if level in requested)
    return ordered


def _validate_image_feature_level_kwargs(config: PreprocessConfig) -> None:
    if not isinstance(config.image_feature_level_kwargs, Mapping):
        raise TypeError("image_feature_level_kwargs must be a mapping.")

    for level, kwargs in config.image_feature_level_kwargs.items():
        normalized = str(level).strip().lower()
        if normalized not in _IMAGE_FEATURE_LEVELS:
            raise ValueError(
                "image_feature_level_kwargs keys must be 'spot' or 'enhanced'. "
                f"Unsupported: {level!r}."
            )
        if not isinstance(kwargs, Mapping):
            raise TypeError(
                f"image_feature_level_kwargs[{level!r}] must be a mapping."
            )
        invalid = set(kwargs) - _IMAGE_FEATURE_AGGREGATION_OVERRIDE_KEYS
        if invalid:
            raise ValueError(
                f"Invalid per-level image feature option(s) for {level!r}: "
                f"{sorted(invalid)}. Per-level overrides are only for aggregation "
                "and plotting options such as patch_size_spot, normalize_by, "
                "n_clusters, or plot_clusters. Put model/checkpoint/preprocessing "
                "options in image_feature_kwargs."
            )


def _validate_image_feature_kwargs(config: PreprocessConfig) -> None:
    if not isinstance(config.image_feature_kwargs, Mapping):
        raise TypeError("image_feature_kwargs must be a mapping.")

    allowed = _IMAGE_FEATURE_WORKSPACE_KEYS | _IMAGE_FEATURE_AGGREGATION_KEYS
    invalid = set(config.image_feature_kwargs) - allowed
    if invalid:
        raise ValueError(
            "Invalid image_feature_kwargs option(s): "
            f"{sorted(invalid)}. Stage 1 manages image_path, spot_coordinates, "
            "output_dir, sample_name, spot_x_key, and spot_y_key directly."
        )


def _has_coordinate_source(result, section, level) -> bool:
    try:
        _preferred_coordinate_source(result, section, level=level)
    except KeyError:
        return False
    return True


def _resolved_image_feature_levels(config, result, section) -> Tuple[str, ...]:
    explicit = _normalize_image_feature_levels(config.image_feature_levels)
    if explicit is not None:
        for level in explicit:
            if not _has_coordinate_source(result, section, level):
                raise KeyError(
                    f"image_feature_levels requested {level!r} image features for "
                    f"section {section!r}, but no {level}-level Gene or Protein "
                    "coordinate source is available."
                )
        return explicit

    has_spot = _has_coordinate_source(result, section, "spot")
    has_enhanced = _has_coordinate_source(result, section, "enhanced")
    if (config.gene_enhancement or config.protein_enhancement) and has_enhanced:
        return ("enhanced",)
    if has_spot:
        return ("spot",)
    if has_enhanced:
        return ("enhanced",)
    return ()


def _image_feature_kwargs_for_level(config, level: str) -> Dict[str, Any]:
    kwargs = dict(config.image_feature_kwargs)
    level_kwargs = {}
    for key, value in config.image_feature_level_kwargs.items():
        if str(key).strip().lower() == level:
            level_kwargs = dict(value)
            break
    kwargs.update(level_kwargs)

    if (
        level == "enhanced"
        and "patch_size_spot" not in level_kwargs
        and "resolution" in config.enhancement_kwargs
    ):
        kwargs["patch_size_spot"] = config.enhancement_kwargs["resolution"]
    return kwargs


def _select_image_feature_kwargs(kwargs: Mapping[str, Any], keys: set[str]) -> Dict[str, Any]:
    return {key: value for key, value in kwargs.items() if key in keys}


def _image_feature_level_output_dir(paths, level: str, section: str) -> Path:
    root = paths.image_spot_dir if level == "spot" else paths.image_enhanced_dir
    return root / section / "results"


def _validate_config(config: PreprocessConfig) -> Tuple[str, ...]:
    modalities = _canonical_modalities(config.modalities)
    if not modalities:
        raise ValueError("At least one modality must be supplied.")
    data_dir = Path(config.data_dir).expanduser()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data_dir does not exist or is not a folder: {data_dir}")
    if config.raw_file_mode not in {"copy", "symlink", "none"}:
        raise ValueError("raw_file_mode must be 'copy', 'symlink', or 'none'.")
    if str(config.image_feature_mode).lower() not in {"extract", "load"}:
        raise ValueError("image_feature_mode must be 'extract' or 'load'.")
    _normalize_image_feature_levels(config.image_feature_levels)
    _validate_image_feature_kwargs(config)
    _validate_image_feature_level_kwargs(config)
    if "Image" in modalities and not ({"Gene", "Protein"} & set(modalities)):
        raise ValueError(
            "Image preprocessing needs Gene or Protein AnnData to provide spot coordinates."
        )
    if config.gene_enhancement and "Gene" not in modalities:
        raise ValueError("gene_enhancement=True requires the Gene modality.")
    if config.protein_enhancement and "Protein" not in modalities:
        raise ValueError("protein_enhancement=True requires the Protein modality.")
    molecular_templates = (
        ("reference_gene_template", config.reference_gene_template),
        ("query_gene_template", config.query_gene_template),
        ("reference_protein_template", config.reference_protein_template),
        ("query_protein_template", config.query_protein_template),
        (
            "reference_enhanced_gene_template",
            config.reference_enhanced_gene_template,
        ),
        ("query_enhanced_gene_template", config.query_enhanced_gene_template),
        (
            "reference_enhanced_protein_template",
            config.reference_enhanced_protein_template,
        ),
        (
            "query_enhanced_protein_template",
            config.query_enhanced_protein_template,
        ),
    )
    templates = (
        *molecular_templates,
        ("reference_image_template", config.reference_image_template),
        ("query_image_template", config.query_image_template),
        (
            "reference_annotated_image_template",
            config.reference_annotated_image_template,
        ),
    )
    for template_name, template in templates:
        if template is not None and "{section}" not in template:
            raise ValueError(f"{template_name} must contain '{{section}}'.")
    image_feature_templates = (
        ("reference_image_feature_template", config.reference_image_feature_template),
        ("query_image_feature_template", config.query_image_feature_template),
        (
            "reference_enhanced_image_feature_template",
            config.reference_enhanced_image_feature_template,
        ),
        (
            "query_enhanced_image_feature_template",
            config.query_enhanced_image_feature_template,
        ),
    )
    for template_name, template in image_feature_templates:
        if template is not None and "{section}" not in template:
            raise ValueError(f"{template_name} must contain '{{section}}'.")
    for template_name, template in (
        ("reference_image_template", config.reference_image_template),
        ("query_image_template", config.query_image_template),
        (
            "reference_annotated_image_template",
            config.reference_annotated_image_template,
        ),
    ):
        if "{ext}" not in template:
            raise ValueError(f"{template_name} must contain '{{ext}}'.")
    for cohort_name, sections in (
        ("reference", config.reference_sections),
        ("query", config.query_sections),
    ):
        if len(set(sections)) != len(sections):
            raise ValueError(f"{cohort_name}_sections contains duplicate IDs.")
    if "Gene" in modalities:
        if (
            config.reference_gene_template is None
            and config.reference_enhanced_gene_template is None
        ):
            raise ValueError(
                "Gene modality requires reference_gene_template or "
                "reference_enhanced_gene_template."
            )
        if (
            config.query_gene_template is None
            and config.query_enhanced_gene_template is None
        ):
            raise ValueError(
                "Gene modality requires query_gene_template or "
                "query_enhanced_gene_template."
            )
    if "Protein" in modalities:
        if (
            config.reference_protein_template is None
            and config.reference_enhanced_protein_template is None
        ):
            raise ValueError(
                "Protein modality requires reference_protein_template or "
                "reference_enhanced_protein_template."
            )
        if (
            config.query_protein_template is None
            and config.query_enhanced_protein_template is None
        ):
            raise ValueError(
                "Protein modality requires query_protein_template or "
                "query_enhanced_protein_template."
            )
    if config.gene_enhancement and (
        config.reference_gene_template is None or config.query_gene_template is None
    ):
        raise ValueError(
            "gene_enhancement=True requires spot-level reference/query Gene templates."
        )
    if config.protein_enhancement and (
        config.reference_protein_template is None
        or config.query_protein_template is None
    ):
        raise ValueError(
            "protein_enhancement=True requires spot-level reference/query Protein "
            "templates."
        )
    if config.gene_enhancement and (
        config.reference_enhanced_gene_template is not None
        or config.query_enhanced_gene_template is not None
    ):
        raise ValueError(
            "Use either gene_enhancement=True or precomputed enhanced Gene "
            "templates, not both."
        )
    if config.protein_enhancement and (
        config.reference_enhanced_protein_template is not None
        or config.query_enhanced_protein_template is not None
    ):
        raise ValueError(
            "Use either protein_enhancement=True or precomputed enhanced Protein "
            "templates, not both."
        )
    if config.label_color_dict is not None and (
        config.reference_gene_template is None
        and config.reference_protein_template is None
    ):
        raise ValueError(
            "label_color_dict requires at least one spot-level reference molecular "
            "template because scribbles are assigned to observed spots."
        )
    if "Image" in modalities and str(config.image_feature_mode).lower() == "load":
        for cohort_name, spot_template, enhanced_template in (
            (
                "reference",
                config.reference_image_feature_template,
                config.reference_enhanced_image_feature_template,
            ),
            (
                "query",
                config.query_image_feature_template,
                config.query_enhanced_image_feature_template,
            ),
        ):
            if spot_template is None and enhanced_template is None:
                raise ValueError(
                    f"Image load mode requires at least one {cohort_name} "
                    "spot-level or enhanced image-feature template."
                )
    return modalities


def _data_dir(config):
    return Path(config.data_dir).expanduser()


def _active_raw_dir(config, paths):
    return _data_dir(config) if config.raw_file_mode == "none" else paths.raw_dir


def _molecular_template(config, cohort, modality, level="spot"):
    cohort = str(cohort).lower()
    modality = str(modality).lower()
    level = str(level).lower()
    if cohort == "reference" and modality == "gene" and level == "spot":
        return config.reference_gene_template
    if cohort == "query" and modality == "gene" and level == "spot":
        return config.query_gene_template
    if cohort == "reference" and modality == "protein" and level == "spot":
        return config.reference_protein_template
    if cohort == "query" and modality == "protein" and level == "spot":
        return config.query_protein_template
    if cohort == "reference" and modality == "gene" and level == "enhanced":
        return config.reference_enhanced_gene_template
    if cohort == "query" and modality == "gene" and level == "enhanced":
        return config.query_enhanced_gene_template
    if cohort == "reference" and modality == "protein" and level == "enhanced":
        return config.reference_enhanced_protein_template
    if cohort == "query" and modality == "protein" and level == "enhanced":
        return config.query_enhanced_protein_template
    raise ValueError(
        f"Unsupported cohort/modality/level combination: {cohort}, {modality}, {level}"
    )


def _image_template(config, cohort, annotated=False):
    cohort = str(cohort).lower()
    if annotated:
        if cohort != "reference":
            raise ValueError("Annotated images are only supported for references.")
        return config.reference_annotated_image_template
    if cohort == "reference":
        return config.reference_image_template
    if cohort == "query":
        return config.query_image_template
    raise ValueError(f"Unsupported cohort: {cohort}")


def _image_feature_template(config, cohort, level="spot"):
    cohort = str(cohort).lower()
    level = str(level).lower()
    if level == "spot":
        if cohort == "reference":
            return config.reference_image_feature_template
        if cohort == "query":
            return config.query_image_feature_template
    elif level == "enhanced":
        if cohort == "reference":
            return config.reference_enhanced_image_feature_template
        if cohort == "query":
            return config.query_enhanced_image_feature_template
    raise ValueError(f"Unsupported cohort/level combination: {cohort}, {level}")


def _resolve_data_file(config, section, template):
    return resolve_section_file(_data_dir(config), section, template)


def _iter_required_data_files(config, modalities, cohort, sections):
    extracts_image_features = (
        "Image" in modalities and str(config.image_feature_mode).lower() == "extract"
    )
    loads_image_features = (
        "Image" in modalities and str(config.image_feature_mode).lower() == "load"
    )
    needs_raw_images = (
        extracts_image_features or config.gene_enhancement or config.protein_enhancement
    )
    for section in sections:
        if "Gene" in modalities:
            for level in ("spot", "enhanced"):
                template = _molecular_template(config, cohort, "Gene", level=level)
                if template is not None:
                    yield _resolve_data_file(config, section, template)
        if "Protein" in modalities:
            for level in ("spot", "enhanced"):
                template = _molecular_template(config, cohort, "Protein", level=level)
                if template is not None:
                    yield _resolve_data_file(config, section, template)
        if needs_raw_images or (
            cohort == "reference" and config.label_color_dict is not None
        ):
            yield _resolve_data_file(
                config,
                section,
                _image_template(config, cohort, annotated=False),
            )
        if cohort == "reference" and config.label_color_dict is not None:
            yield _resolve_data_file(
                config,
                section,
                _image_template(config, cohort, annotated=True),
            )
        if loads_image_features:
            spot_template = _image_feature_template(config, cohort, level="spot")
            if spot_template is not None:
                yield _resolve_data_file(config, section, spot_template)
            enhanced_template = _image_feature_template(
                config,
                cohort,
                level="enhanced",
            )
            if enhanced_template is not None:
                yield _resolve_data_file(config, section, enhanced_template)


def _place_raw_file(source, raw_dir, mode):
    source = Path(source).expanduser().resolve()
    destination = Path(raw_dir) / source.name
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "symlink":
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(source)
    else:
        raise ValueError("mode must be 'copy' or 'symlink'.")
    return destination


def _organize_raw_inputs(config, paths, modalities):
    if config.raw_file_mode == "none":
        return {"reference": {}, "query": {}}

    organized = {"reference": {}, "query": {}}
    for cohort, sections in (
        ("reference", config.reference_sections),
        ("query", config.query_sections),
    ):
        seen = set()
        for source in _iter_required_data_files(config, modalities, cohort, sections):
            if source in seen:
                continue
            seen.add(source)
            destination = _place_raw_file(
                source,
                paths[cohort].raw_dir,
                config.raw_file_mode,
            )
            organized[cohort][source.name] = str(destination)
    return organized


def _empty_cohort_result(modalities):
    return {
        "spot": {modality: {} for modality in modalities},
        "enhanced": {modality: {} for modality in modalities},
    }


def _load_molecular_cohort(config, paths, sections, modalities, cohort):
    result = _empty_cohort_result(modalities)
    raw_dir = _active_raw_dir(config, paths)
    common_kwargs = dict(
        target_sum=config.target_sum,
        log1p=config.log1p,
        uppercase_features=config.uppercase_features,
        feature_key="genes",
        random_state=config.random_state,
        # These objects have just been loaded and are owned by this stage.
        copy=False,
    )
    if "Gene" in modalities:
        for level in ("spot", "enhanced"):
            template = _molecular_template(config, cohort, "Gene", level=level)
            if template is None:
                continue
            result[level]["Gene"] = preprocess_molecular_sections(
                sections,
                raw_dir,
                modality="Gene",
                file_template=template,
                **common_kwargs,
            )
    if "Protein" in modalities:
        for level in ("spot", "enhanced"):
            template = _molecular_template(config, cohort, "Protein", level=level)
            if template is None:
                continue
            result[level]["Protein"] = preprocess_molecular_sections(
                sections,
                raw_dir,
                modality="Protein",
                file_template=template,
                replace_zeros=config.protein_replace_zeros,
                zero_replacement_scale=config.zero_replacement_scale,
                **common_kwargs,
            )
    return result


def _resolved_images(config, paths, sections, cohort, annotated=False):
    template = _image_template(config, cohort, annotated=annotated)
    raw_dir = _active_raw_dir(config, paths)
    return {
        section: resolve_section_file(raw_dir, section, template)
        for section in sections
    }


def _image_size(image_path, config):
    """Return image width and height without loading pixels when possible."""
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            return image.size
    except Exception:
        image = read_he_image(image_path, color_order=config.color_order)
        height, width = image.shape[:2]
        return width, height


def _preview_columns(columns, max_items=12):
    columns = [str(column) for column in columns]
    if len(columns) <= max_items:
        return columns
    return columns[:max_items] + [f"... ({len(columns) - max_items} more)"]


def _coordinate_range_text(coordinates):
    if coordinates.shape[0] == 0:
        return "no observations"
    x_min, y_min = np.nanmin(coordinates, axis=0)
    x_max, y_max = np.nanmax(coordinates, axis=0)
    return f"x=[{x_min:.3g}, {x_max:.3g}], y=[{y_min:.3g}, {y_max:.3g}]"


def _outside_image_count(coordinates, width, height):
    in_bounds = (
        (coordinates[:, 0] >= 0)
        & (coordinates[:, 0] < width)
        & (coordinates[:, 1] >= 0)
        & (coordinates[:, 1] < height)
    )
    return int((~in_bounds).sum())


def _validate_pixel_coordinates_for_image(
    *,
    adata_obj,
    image_path,
    config,
    context,
):
    """Validate configured x/y columns against one image and explain failures."""
    missing = [
        key for key in (config.x_key, config.y_key) if key not in adata_obj.obs
    ]
    if missing:
        raise KeyError(
            f"Coordinate check failed for {context}: missing obs columns "
            f"{missing}. Stage 1 image-based steps require x_key/y_key to be "
            "image-pixel coordinates. Available obs columns include "
            f"{_preview_columns(adata_obj.obs.columns)}. If your data stores "
            "coordinates as row/column, set x_key to the horizontal column "
            "coordinate and y_key to the vertical row coordinate. "
            "array_x_key/array_y_key are only for scan-based contour "
            "detection, not replacements for x_key/y_key."
        )

    array_keys = {config.array_x_key, config.array_y_key}
    pixel_keys = {config.x_key, config.y_key}
    if array_keys.intersection(pixel_keys):
        warnings.warn(
            f"Coordinate check for {context}: x_key/y_key overlap with "
            f"array_x_key/array_y_key ({sorted(array_keys.intersection(pixel_keys))}). "
            "x_key/y_key should be image-pixel coordinates. array_x_key/"
            "array_y_key should be array-grid coordinates used only by "
            "scan_x/scan_y contour detection.",
            UserWarning,
            stacklevel=2,
        )

    try:
        coordinates = adata_obj.obs[[config.x_key, config.y_key]].to_numpy(
            dtype=float
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Coordinate check failed for {context}: x_key={config.x_key!r} "
            f"and y_key={config.y_key!r} must contain numeric image-pixel "
            "coordinates."
        ) from exc

    if coordinates.shape[0] == 0:
        return

    finite = np.isfinite(coordinates).all(axis=1)
    if not finite.all():
        raise ValueError(
            f"Coordinate check failed for {context}: {int((~finite).sum())} "
            "observations have non-finite x/y coordinates."
        )

    width, height = _image_size(image_path, config)
    outside_count = _outside_image_count(coordinates, width=width, height=height)
    if outside_count == 0:
        return

    swapped_coordinates = coordinates[:, [1, 0]]
    swapped_outside_count = _outside_image_count(
        swapped_coordinates,
        width=width,
        height=height,
    )
    message = (
        f"Coordinate check failed for {context}: {outside_count} spot "
        "coordinates lie outside the image.\n"
        f"Image size: width={width}, height={height}.\n"
        f"Using x_key={config.x_key!r}, y_key={config.y_key!r}: "
        f"{_coordinate_range_text(coordinates)}.\n"
        "Stage 1 expects x_key to be the horizontal image coordinate "
        "(bounded by image width) and y_key to be the vertical image "
        "(bounded by image height)."
    )
    if swapped_outside_count < outside_count:
        message += (
            "\nThe switched interpretation fits better: "
            f"{swapped_outside_count} coordinates would be outside after "
            f"using x_key={config.y_key!r}, y_key={config.x_key!r}. If your "
            "data stores coordinates as row/column or y/x, switch x_key and "
            "y_key in PreprocessConfig."
        )
    else:
        message += (
            "\nSwitching x_key/y_key does not appear to resolve the mismatch. "
            "This may indicate that array/grid coordinates were used instead "
            "of image-pixel coordinates, or that the AnnData coordinates and "
            "image file use different resolution, cropping, rotation, or "
            "scaling."
        )
    message += (
        "\nIf you use scan_x or scan_y contour detection, keep array/grid "
        "columns in array_x_key/array_y_key separately from pixel x_key/y_key."
    )
    raise ValueError(message)


def _validate_scan_array_coordinates_for_enhancement(
    *,
    adata_obj,
    config,
    context,
):
    contour_method = str(config.enhancement_kwargs.get("contour_method", "auto"))
    if contour_method not in {"scan_x", "scan_y"}:
        return

    missing = [
        key
        for key in (config.array_x_key, config.array_y_key)
        if key not in adata_obj.obs
    ]
    if missing:
        raise KeyError(
            f"Coordinate check failed for {context}: contour_method="
            f"{contour_method!r} requires array-grid coordinate columns "
            f"{config.array_x_key!r} and {config.array_y_key!r}, but missing "
            f"{missing}. Keep x_key/y_key as image-pixel coordinates and set "
            "array_x_key/array_y_key to the array-grid columns used for "
            "scan-based contour detection."
        )


def _iter_spot_molecular_sources(result, section):
    for modality in ("Gene", "Protein"):
        adata_obj = result["spot"].get(modality, {}).get(section)
        if adata_obj is not None:
            yield modality, adata_obj


def _preflight_image_coordinate_inputs(
    *,
    config,
    reference,
    query,
    reference_images,
    query_images,
    modalities,
):
    """Validate pixel and scan-coordinate settings before image-dependent work."""
    extracts_image_features = (
        "Image" in modalities and str(config.image_feature_mode).lower() == "extract"
    )

    cohorts = (
        ("reference", reference, config.reference_sections, reference_images),
        ("query", query, config.query_sections, query_images),
    )
    for cohort, result, sections, images in cohorts:
        if not images:
            continue
        for section in sections:
            image_path = images.get(section)
            if image_path is None:
                continue

            checked_pixel_sources = set()
            should_check_annotation = (
                cohort == "reference" and config.label_color_dict is not None
            )
            should_check_image_features = extracts_image_features

            for modality, enabled in (
                ("Gene", config.gene_enhancement),
                ("Protein", config.protein_enhancement),
            ):
                if modality not in modalities or not enabled:
                    continue
                adata_obj = result["spot"].get(modality, {}).get(section)
                if adata_obj is None:
                    continue
                context = (
                    f"{cohort} section {section!r}, spot-level {modality} "
                    "used for molecular enhancement"
                )
                _validate_scan_array_coordinates_for_enhancement(
                    adata_obj=adata_obj,
                    config=config,
                    context=context,
                )
                _validate_pixel_coordinates_for_image(
                    adata_obj=adata_obj,
                    image_path=image_path,
                    config=config,
                    context=context,
                )
                checked_pixel_sources.add(id(adata_obj))

            if should_check_annotation:
                for modality, adata_obj in _iter_spot_molecular_sources(
                    result,
                    section,
                ):
                    if id(adata_obj) in checked_pixel_sources:
                        continue
                    context = (
                        f"{cohort} section {section!r}, spot-level {modality} "
                        "used for scribble annotation"
                    )
                    _validate_pixel_coordinates_for_image(
                        adata_obj=adata_obj,
                        image_path=image_path,
                        config=config,
                        context=context,
                    )
                    checked_pixel_sources.add(id(adata_obj))
                    break

            if should_check_image_features:
                levels_to_check = (
                    _normalize_image_feature_levels(config.image_feature_levels)
                    or _IMAGE_FEATURE_LEVELS
                )
                for level in levels_to_check:
                    try:
                        adata_obj = _preferred_coordinate_source(
                            result,
                            section,
                            level=level,
                        )
                    except KeyError:
                        continue
                    if id(adata_obj) in checked_pixel_sources:
                        continue
                    context = (
                        f"{cohort} section {section!r}, {level}-level "
                        "coordinates used for image feature extraction"
                    )
                    _validate_pixel_coordinates_for_image(
                        adata_obj=adata_obj,
                        image_path=image_path,
                        config=config,
                        context=context,
                    )
                    checked_pixel_sources.add(id(adata_obj))


def _add_reference_annotations(config, result, paths, images, sections):
    """Extract optional scribbles and propagate reference labels."""
    annotation_results = {}
    molecular_modalities = [
        modality for modality in ("Gene", "Protein") if result["spot"].get(modality)
    ]
    if not molecular_modalities:
        return annotation_results

    if config.label_color_dict is None:
        labeled_modalities = [
            modality
            for modality in molecular_modalities
            if all(
                config.label_key in result["spot"][modality][section].obs
                for section in sections
            )
        ]
    else:
        labeled_modalities = []
    annotation_source = (
        labeled_modalities[0]
        if labeled_modalities
        else ("Gene" if "Gene" in molecular_modalities else "Protein")
    )
    if config.label_color_dict is not None:
        from ..preprocessing.extract_scribble_annotations import (
            assign_mask_labels_to_adata,
            extract_scribble_labels_pipeline,
        )

        annotated_images = _resolved_images(
            config,
            paths,
            sections,
            cohort="reference",
            annotated=True,
        )
        for section in sections:
            kwargs = dict(config.scribble_kwargs)
            kwargs.update(
                ref_adata_dic={section: result["spot"][annotation_source][section]},
                ref_section_list=[section],
                data_path=str(images[section].parent),
                label_color_dict=dict(config.label_color_dict),
                image_template=images[section].name,
                annotated_image_template=annotated_images[section].name,
                x_key=config.x_key,
                y_key=config.y_key,
                label_key=config.label_key,
                output_dir=str(paths.scribble_dir),
                copy=False,
            )
            updated, section_results = extract_scribble_labels_pipeline(**kwargs)
            result["spot"][annotation_source][section] = updated[section]
            annotation_results[section] = section_results[section]

    # Apply labels from the selected annotation source to other modalities.
    for section in sections:
        source = result["spot"][annotation_source][section]
        if config.label_key not in source.obs:
            continue
        for modality in molecular_modalities:
            if modality == annotation_source:
                continue
            target = result["spot"][modality][section]
            if section in annotation_results:
                extraction = annotation_results[section]
                target = assign_mask_labels_to_adata(
                    ref_adata=target,
                    ref_mask=extraction["ref_mask"],
                    label_id_dict=extraction["label_id_dict"],
                    x_key=config.x_key,
                    y_key=config.y_key,
                    label_key=config.label_key,
                    background_label=config.scribble_kwargs.get(
                        "background_label", "nan"
                    ),
                    copy=False,
                )
            elif target.obs_names.isin(source.obs_names).all():
                target = transfer_obs_columns(
                    source,
                    target,
                    columns=config.label_key,
                    copy=False,
                )
            else:
                target = transfer_labels_by_nearest_spot(
                    source,
                    target,
                    label_key=config.label_key,
                    x_key=config.x_key,
                    y_key=config.y_key,
                    copy=False,
                )
            result["spot"][modality][section] = target
    return annotation_results


def _enhance_cohort(
    config,
    result,
    paths,
    images,
    sections,
    cohort,
    modalities,
    annotation_results=None,
):
    annotation_results = annotation_results or {}
    if not config.gene_enhancement and not config.protein_enhancement:
        return

    from ..preprocessing.extract_scribble_annotations import assign_mask_labels_to_adata
    from ..preprocessing.gene_enhancement import enhance_gene_expression

    for modality, enabled in (
        ("Gene", config.gene_enhancement),
        ("Protein", config.protein_enhancement),
    ):
        if modality not in modalities or not enabled:
            continue
        for section in sections:
            source = result["spot"][modality][section]
            kwargs = dict(config.enhancement_kwargs)
            contour_method = kwargs.get("contour_method", "auto")
            kwargs.update(
                image=read_he_image(images[section], color_order=config.color_order),
                raw_adata=source,
                genes=source.var_names,
                x_key=config.x_key,
                y_key=config.y_key,
                array_x_key=config.array_x_key,
                array_y_key=config.array_y_key,
                color_order=config.color_order,
                qc_path=str(
                    paths.contour_dir
                    / f"{section}_{modality.lower()}_{contour_method}.jpg"
                ),
            )
            enhanced = enhance_gene_expression(**kwargs).enhanced_adata

            # Reference enhanced grids use the extracted image mask when one
            # exists; otherwise they inherit the nearest observed-spot label.
            if cohort == "reference" and config.label_key in source.obs:
                if section in annotation_results:
                    extraction = annotation_results[section]
                    enhanced = assign_mask_labels_to_adata(
                        ref_adata=enhanced,
                        ref_mask=extraction["ref_mask"],
                        label_id_dict=extraction["label_id_dict"],
                        x_key=config.x_key,
                        y_key=config.y_key,
                        label_key=config.label_key,
                        background_label=config.scribble_kwargs.get(
                            "background_label", "nan"
                        ),
                        copy=False,
                    )
                else:
                    enhanced = transfer_labels_by_nearest_spot(
                        source,
                        enhanced,
                        label_key=config.label_key,
                        x_key=config.x_key,
                        y_key=config.y_key,
                        copy=False,
                    )
            result["enhanced"][modality][section] = enhanced


def _preferred_coordinate_source(result, section, level="spot"):
    for modality in ("Gene", "Protein"):
        if section in result[level].get(modality, {}):
            return result[level][modality][section]
    raise KeyError(f"No molecular coordinate source is available for {section!r}.")


def _save_molecular_outputs(result, paths, sections):
    """Save molecular h5ad files and one coordinate table per spatial level."""
    for level in ("spot", "enhanced"):
        suffix = "" if level == "spot" else "enhanced_"
        for modality in ("Gene", "Protein"):
            for section, adata_obj in result[level].get(modality, {}).items():
                adata_obj.write_h5ad(
                    paths.preprocessed_dir
                    / f"{section}_{suffix}{modality.lower()}.h5ad"
                )
        for section in sections:
            try:
                source = _preferred_coordinate_source(result, section, level=level)
            except KeyError:
                continue
            filename = (
                f"{section}_spot_coordinates.csv"
                if level == "spot"
                else f"{section}_enhanced_spot_coordinates.csv"
            )
            save_spot_coordinates(source, paths.preprocessed_dir / filename)


def _extract_image_cohort(config, result, paths, images, sections, modalities):
    if "Image" not in modalities:
        return
    from ..preprocessing.image_features import (
        _aggregate_image_features_to_spots,
        _prepare_image_feature_workspace,
    )

    for section in sections:
        levels = _resolved_image_feature_levels(config, result, section)
        if not levels:
            continue

        for level in levels:
            coordinates = _preferred_coordinate_source(result, section, level=level)
            _validate_pixel_coordinates_for_image(
                adata_obj=coordinates,
                image_path=images[section],
                config=config,
                context=(
                    f"section {section!r}, {level}-level coordinates used for "
                    "image feature extraction"
                ),
            )

        workspace_kwargs = _select_image_feature_kwargs(
            config.image_feature_kwargs,
            _IMAGE_FEATURE_WORKSPACE_KEYS,
        )
        workspace = _prepare_image_feature_workspace(
            image_path=str(images[section]),
            output_dir=str(paths.image_grid_dir),
            sample_name=section,
            **workspace_kwargs,
        )

        for level in levels:
            coordinates = _preferred_coordinate_source(result, section, level=level)
            level_kwargs = _image_feature_kwargs_for_level(config, level)
            aggregation_kwargs = _select_image_feature_kwargs(
                level_kwargs,
                _IMAGE_FEATURE_AGGREGATION_KEYS,
            )
            output_dir = _image_feature_level_output_dir(paths, level, section)
            image_adata = _aggregate_image_features_to_spots(
                workspace=workspace,
                spot_coordinates=coordinates.obs.copy(),
                output_dir=str(output_dir),
                spot_x_key=config.x_key,
                spot_y_key=config.y_key,
                **aggregation_kwargs,
            )
            image_adata.var["image"] = image_adata.var_names.astype(str)
            image_adata = remove_obs_columns_by_prefix(
                image_adata,
                prefixes="kmeans_",
                copy=False,
            )
            result[level]["Image"][section] = image_adata
            suffix = "" if level == "spot" else "enhanced_"
            image_adata.write_h5ad(
                paths.preprocessed_dir / f"{section}_{suffix}image.h5ad"
            )


def _prepare_loaded_image_adata(image_adata, coordinates, config, section, level):
    """Validate and standardize a pre-extracted image-feature AnnData object."""
    if image_adata.n_obs != coordinates.n_obs:
        raise ValueError(
            f"Loaded {level} image features for section {section!r} contain "
            f"{image_adata.n_obs} observations, but the molecular coordinate "
            f"source contains {coordinates.n_obs} observations."
        )

    if image_adata.obs_names.is_unique and coordinates.obs_names.is_unique:
        missing_obs = coordinates.obs_names.difference(image_adata.obs_names)
        if len(missing_obs) > 0:
            preview = list(missing_obs[:5])
            raise ValueError(
                f"Loaded {level} image features for section {section!r} are "
                f"missing molecular observation IDs, for example {preview}."
            )
        if not image_adata.obs_names.equals(coordinates.obs_names):
            image_adata = image_adata[coordinates.obs_names].copy()

    for coordinate_key in (config.x_key, config.y_key):
        if coordinate_key not in image_adata.obs and coordinate_key in coordinates.obs:
            image_adata.obs[coordinate_key] = coordinates.obs[coordinate_key].to_numpy()

    if "image" not in image_adata.var:
        image_adata.var["image"] = image_adata.var_names.astype(str)
    image_adata = remove_obs_columns_by_prefix(
        image_adata,
        prefixes="kmeans_",
        copy=False,
    )
    return image_adata


def _load_image_cohort(config, result, paths, sections, cohort, modalities):
    if "Image" not in modalities:
        return

    raw_dir = _active_raw_dir(config, paths)
    for level in ("spot", "enhanced"):
        template = _image_feature_template(config, cohort, level=level)
        if template is None:
            continue
        for section in sections:
            try:
                coordinates = _preferred_coordinate_source(result, section, level=level)
            except KeyError:
                continue
            image_path = resolve_section_file(
                raw_dir,
                section,
                template,
                extensions=(".h5ad",),
            )
            image_adata = ad.read_h5ad(image_path)
            image_adata = _prepare_loaded_image_adata(
                image_adata,
                coordinates,
                config,
                section,
                level,
            )
            result[level]["Image"][section] = image_adata
            suffix = "" if level == "spot" else "enhanced_"
            image_adata.write_h5ad(
                paths.preprocessed_dir / f"{section}_{suffix}image.h5ad"
            )


def _process_image_cohort(config, result, paths, images, sections, cohort, modalities):
    image_feature_mode = str(config.image_feature_mode).lower()
    if image_feature_mode == "extract":
        _extract_image_cohort(config, result, paths, images, sections, modalities)
    elif image_feature_mode == "load":
        _load_image_cohort(config, result, paths, sections, cohort, modalities)
    else:
        raise ValueError("image_feature_mode must be 'extract' or 'load'.")


@logged_stage(
    "preprocessing",
    stage_output_from_config(
        default_output_dir=None,
        config_position=0,
        output_attribute="preprocess_dir",
    ),
)
def run_preprocessing_pipeline(config: PreprocessConfig) -> PreprocessPipelineResult:
    """Preprocess all reference and query sections.

    Processing steps
    ----------------
    1. Create the standard folder tree.
    2. Load gene/protein ``.h5ad`` files; normalize, log1p, and standardize
       feature names.
    3. Optionally extract reference scribble annotations and propagate labels.
    4. Optionally detect contours and generate enhanced molecular pseudo spots.
    5. Save molecular objects and coordinate CSV files.
    6. Optionally extract observed/enhanced UNI or HIPT image features, or load
       pre-extracted image-feature ``.h5ad`` files, and save clean final Image
       AnnData objects.

    Required raw inputs
    -------------------
    For every listed section, place the selected files in ``data_dir``:

    - ``{section}_ref_gene_raw.h5ad`` for spot-level reference Gene unless
      ``reference_gene_template=None``;
    - ``{section}_query_gene_raw.h5ad`` for spot-level query Gene unless
      ``query_gene_template=None``;
    - ``{section}_ref_protein_raw.h5ad`` for spot-level reference Protein unless
      ``reference_protein_template=None``;
    - ``{section}_query_protein_raw.h5ad`` for spot-level query Protein unless
      ``query_protein_template=None``;
    - optional precomputed enhanced Gene/Protein ``.h5ad`` files matching the
      enhanced molecular templates when supplied;
    - an H&E matching the cohort image template when Image features are
      extracted from raw H&E or molecular enhancement is enabled;
    - pre-extracted Image feature ``.h5ad`` files matching
      ``reference_image_feature_template`` and ``query_image_feature_template``
      when ``image_feature_mode="load"``;
    - a reference annotated H&E matching ``reference_annotated_image_template`` when
      ``label_color_dict`` is supplied.

    Molecular ``.obs`` must contain ``x_key``/``y_key``. Scan-based contour
    candidates additionally use ``array_x_key``/``array_y_key``.

    Parameters
    ----------
    config : PreprocessConfig
        Input folders, section IDs, modalities, preprocessing options, image
        templates, coordinates, and optional enhancement/model settings.

    Returns
    -------
    PreprocessPipelineResult
        Nested in-memory reference/query AnnData dictionaries, optional
        scribble extraction results, and resolved output paths. Final files
        are also written below each cohort's ``preprocessed`` directory.

    Saved files
    -----------
    ``preprocessing_result.pkl`` and ``stage_config.json`` at
    ``preprocess_dir``; molecular/image ``.h5ad`` files and coordinate CSVs
    below each cohort's ``preprocessed`` directory; optional contour,
    scribble, image-feature, and clustering-QC artifacts in its subfolders.
    """
    modalities = _validate_config(config)

    # Step 1: establish package-managed preprocessing folders and optionally
    # organize flat raw inputs into their cohort-specific raw folders.
    paths = {
        cohort: create_preprocess_output_dirs(config.preprocess_dir, cohort)
        for cohort in ("reference", "query")
    }
    raw_input_manifest = _organize_raw_inputs(config, paths, modalities)

    # Step 2: preprocess raw molecular data.
    reference = _load_molecular_cohort(
        config, paths["reference"], config.reference_sections, modalities, "reference"
    )
    query = _load_molecular_cohort(
        config, paths["query"], config.query_sections, modalities, "query"
    )

    extracts_image_features = (
        "Image" in modalities and str(config.image_feature_mode).lower() == "extract"
    )
    needs_raw_images = (
        extracts_image_features or config.gene_enhancement or config.protein_enhancement
    )
    reference_images = (
        _resolved_images(
            config,
            paths["reference"],
            config.reference_sections,
            cohort="reference",
        )
        if needs_raw_images or config.label_color_dict is not None
        else {}
    )
    query_images = (
        _resolved_images(
            config,
            paths["query"],
            config.query_sections,
            cohort="query",
        )
        if needs_raw_images
        else {}
    )

    # Step 2b: fail early on common coordinate/image mismatches before running
    # scribble extraction, molecular enhancement, or image-feature extraction.
    _preflight_image_coordinate_inputs(
        config=config,
        reference=reference,
        query=query,
        reference_images=reference_images,
        query_images=query_images,
        modalities=modalities,
    )

    # Step 3: extract/propagate optional reference annotations.
    annotation_results = _add_reference_annotations(
        config,
        reference,
        paths["reference"],
        reference_images,
        config.reference_sections,
    )

    # Step 4: optionally create dense molecular pseudo spots.
    _enhance_cohort(
        config,
        reference,
        paths["reference"],
        reference_images,
        config.reference_sections,
        "reference",
        modalities,
        annotation_results=annotation_results,
    )
    _enhance_cohort(
        config,
        query,
        paths["query"],
        query_images,
        config.query_sections,
        "query",
        modalities,
    )

    # Step 5: write molecular outputs and coordinate tables.
    _save_molecular_outputs(reference, paths["reference"], config.reference_sections)
    _save_molecular_outputs(query, paths["query"], config.query_sections)

    # Step 6: optionally create or load spot/enhanced image-feature objects.
    _process_image_cohort(
        config,
        reference,
        paths["reference"],
        reference_images,
        config.reference_sections,
        "reference",
        modalities,
    )
    _process_image_cohort(
        config,
        query,
        paths["query"],
        query_images,
        config.query_sections,
        "query",
        modalities,
    )

    result = PreprocessPipelineResult(
        reference=reference,
        query=query,
        annotation_results=annotation_results,
        paths=paths,
    )
    stage_root = Path(config.preprocess_dir).expanduser()
    save_stage_result(result, stage_root / "preprocessing_result.pkl")
    stage_config = asdict(config)
    stage_config["organized_raw_inputs"] = raw_input_manifest
    save_json(stage_config, stage_root / "stage_config.json")
    return result
