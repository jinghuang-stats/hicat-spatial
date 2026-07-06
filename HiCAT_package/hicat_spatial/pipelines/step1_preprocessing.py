"""End-to-end preprocessing entry point for HiCAT.

This module turns raw section-level gene/protein AnnData objects and H&E
images into consistently named preprocessing outputs. See
``PREPROCESSING_PIPELINE_GUIDE.md`` for input and output file trees.

Typical use
-----------
::

    from hicat_spatial import (
        PreprocessConfig,
        run_preprocessing_pipeline,
    )

    config = PreprocessConfig(
        preprocess_dir="./preprocess",
        reference_sections=["ref_1", "ref_2"],
        query_sections=["query_1"],
        modalities=("Gene", "Image", "Protein"),
        gene_enhancement=True,
        label_color_dict={"tumor": (255, 0, 0), "stroma": (0, 255, 0)},
        image_feature_kwargs={
            "model": "uni",
            "checkpoint_path": "./checkpoints/pytorch_model.bin",
            "n_clusters": (5, 10, 15),
        },
    )
    result = run_preprocessing_pipeline(config)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import anndata as ad

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


@dataclass
class PreprocessConfig:
    """Parameters for :func:`run_preprocessing_pipeline`.

    Parameters
    ----------
    preprocess_dir : path-like
        Root folder containing ``reference/raw`` and ``query/raw``.
    reference_sections, query_sections : sequence[str]
        Unique section IDs used in all input filenames.
    modalities : sequence[str], default=("Gene", "Image")
        Exact available modalities from ``"Gene"``, ``"Image"``, and
        ``"Protein"``. Outputs use that canonical order.
    target_sum : float or None, default=10_000
        Per-observation molecular total. ``None`` skips total normalization.
    log1p : bool, default=True
        Apply ``log1p`` after total normalization.
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
        Optional mapping of reference scribble label to RGB color. When None,
        existing ``adata.obs[label_key]`` labels are retained and no masks are
        extracted.
    scribble_kwargs : dict, default={}
        Additional arguments for ``extract_scribble_labels_pipeline``, such as
        ``color_tolerance``, ``selected_labels_dic``, or plotting settings.
    image_feature_kwargs : dict, default={}
        Arguments for ``extract_image_features``, including ``model``,
        ``checkpoint_path``, patch settings, clustering, and plot settings.
    x_key, y_key : str, default=("pixel_x", "pixel_y")
        Image-pixel coordinate columns in every molecular ``adata.obs``.
    array_x_key, array_y_key : str, default=("array_x", "array_y")
        Array-grid coordinates used by scan-based contour candidates.
    label_key : str, default="label"
        Reference tissue-region annotation column.
    image_template : str, default="{section}_image{ext}"
        Raw H&E filename template.
    annotated_image_template : str, default="{section}_annotated_image{ext}"
        Reference annotated-H&E filename template. Both templates require
        ``{section}``; ``{ext}`` tries jpg, jpeg, png, tif, and tiff.
    color_order : {"bgr", "rgb"}, default="bgr"
        H&E channel order supplied to enhancement. ``"bgr"`` matches OpenCV.
    """

    preprocess_dir: Path | str
    reference_sections: Sequence[str]
    query_sections: Sequence[str]
    modalities: Sequence[str] = ("Gene", "Image")
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
    image_feature_kwargs: Dict[str, Any] = field(default_factory=dict)
    x_key: str = "pixel_x"
    y_key: str = "pixel_y"
    array_x_key: str = "array_x"
    array_y_key: str = "array_y"
    label_key: str = "label"
    image_template: str = "{section}_image{ext}"
    annotated_image_template: str = "{section}_annotated_image{ext}"
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


def _validate_config(config: PreprocessConfig) -> Tuple[str, ...]:
    modalities = _canonical_modalities(config.modalities)
    if not modalities:
        raise ValueError("At least one modality must be supplied.")
    if "Image" in modalities and not ({"Gene", "Protein"} & set(modalities)):
        raise ValueError(
            "Image preprocessing needs Gene or Protein AnnData to provide spot coordinates."
        )
    if config.gene_enhancement and "Gene" not in modalities:
        raise ValueError("gene_enhancement=True requires the Gene modality.")
    if config.protein_enhancement and "Protein" not in modalities:
        raise ValueError("protein_enhancement=True requires the Protein modality.")
    for template_name, template in (
        ("image_template", config.image_template),
        ("annotated_image_template", config.annotated_image_template),
    ):
        if "{section}" not in template:
            raise ValueError(f"{template_name} must contain '{{section}}'.")
    for cohort_name, sections in (
        ("reference", config.reference_sections),
        ("query", config.query_sections),
    ):
        if len(set(sections)) != len(sections):
            raise ValueError(f"{cohort_name}_sections contains duplicate IDs.")
    return modalities


def _empty_cohort_result(modalities):
    return {
        "spot": {modality: {} for modality in modalities},
        "enhanced": {modality: {} for modality in modalities},
    }


def _load_molecular_cohort(config, paths, sections, modalities):
    result = _empty_cohort_result(modalities)
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
        result["spot"]["Gene"] = preprocess_molecular_sections(
            sections,
            paths.raw_dir,
            modality="Gene",
            **common_kwargs,
        )
    if "Protein" in modalities:
        result["spot"]["Protein"] = preprocess_molecular_sections(
            sections,
            paths.raw_dir,
            modality="Protein",
            replace_zeros=config.protein_replace_zeros,
            zero_replacement_scale=config.zero_replacement_scale,
            **common_kwargs,
        )
    return result


def _resolved_images(config, paths, sections, annotated=False):
    template = config.annotated_image_template if annotated else config.image_template
    return {
        section: resolve_section_file(paths.raw_dir, section, template)
        for section in sections
    }


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

        annotated_images = _resolved_images(config, paths, sections, annotated=True)
        for section in sections:
            kwargs = dict(config.scribble_kwargs)
            kwargs.update(
                ref_adata_dic={section: result["spot"][annotation_source][section]},
                ref_section_list=[section],
                data_path=str(paths.raw_dir),
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
    from ..preprocessing.image_features import extract_image_features

    for level, output_root in (
        ("spot", paths.image_spot_dir),
        ("enhanced", paths.image_enhanced_dir),
    ):
        for section in sections:
            try:
                coordinates = _preferred_coordinate_source(result, section, level=level)
            except KeyError:
                continue
            kwargs = dict(config.image_feature_kwargs)
            kwargs.update(
                image_path=str(images[section]),
                spot_coordinates=coordinates.obs.copy(),
                output_dir=str(output_root),
                sample_name=section,
                spot_x_key=config.x_key,
                spot_y_key=config.y_key,
            )
            image_adata = extract_image_features(**kwargs)
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
    6. Optionally extract observed/enhanced UNI or HIPT image features, retain
       KMeans QC plots, and save clean final image AnnData objects.

    Required raw inputs
    -------------------
    For every listed section, place the selected modality files under
    ``<preprocess_dir>/<cohort>/raw``:

    - ``{section}_gene_raw.h5ad`` for Gene;
    - ``{section}_protein_raw.h5ad`` for Protein;
    - an H&E matching ``image_template`` for Image or enhancement;
    - a reference annotated H&E matching ``annotated_image_template`` when
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

    # Step 1: establish input/output folders.
    paths = {
        cohort: create_preprocess_output_dirs(config.preprocess_dir, cohort)
        for cohort in ("reference", "query")
    }

    # Step 2: preprocess raw molecular data.
    reference = _load_molecular_cohort(
        config, paths["reference"], config.reference_sections, modalities
    )
    query = _load_molecular_cohort(
        config, paths["query"], config.query_sections, modalities
    )

    needs_images = (
        "Image" in modalities or config.gene_enhancement or config.protein_enhancement
    )
    reference_images = (
        _resolved_images(config, paths["reference"], config.reference_sections)
        if needs_images or config.label_color_dict is not None
        else {}
    )
    query_images = (
        _resolved_images(config, paths["query"], config.query_sections)
        if needs_images
        else {}
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

    # Step 6: optionally create spot/enhanced image-feature objects.
    _extract_image_cohort(
        config,
        reference,
        paths["reference"],
        reference_images,
        config.reference_sections,
        modalities,
    )
    _extract_image_cohort(
        config,
        query,
        paths["query"],
        query_images,
        config.query_sections,
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
    save_json(asdict(config), stage_root / "stage_config.json")
    return result
