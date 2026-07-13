"""Stage 2: infer the reference tissue-region hierarchy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import anndata as ad

from ..preprocessing.preprocess_util import make_nonnegative_adata
from ._io import (
    ensure_output_dir,
    logged_stage,
    save_json,
    save_stage_result,
    stage_output_from_config,
)


@dataclass
class TreeInferenceStageConfig:
    """Configuration for reference hierarchy inference.

    Parameters
    ----------
    output_dir : path-like or None, default=None
        Stage output directory. ``None`` uses
        ``results/02_tree_inference``.
    label_key : str, default="label"
        Reference ``.obs`` column containing tissue-region annotations.
    x_key, y_key : str, default=("pixel_x", "pixel_y")
        Reference ``.obs`` columns containing spatial coordinates.
    image_available : bool, default=False
        Whether the combined Stage-2 objects include image features.
    image_feature_key : str, default="uni"
        Substring identifying image features in ``adata.var_names``. Features
        without this substring are treated as gene features.
    gene_filtering_paras, image_filtering_paras : dict or None, default=None
        Feature-selection keyword dictionaries forwarded to the tree
        algorithm. ``None`` uses its built-in thresholds. Common keys are
        ``pvals_adj``, ``min_fold_change``, ``min_in_out_group_ratio``,
        ``min_in_group_fraction``, and ``gene_num``.
    weights : dict[str, float] or None, default=None
        Distance weights using keys ``w_G`` (gene), ``w_I`` (image), and
        ``w_S`` (spatial). ``None`` uses ``1`` for Gene/Spatial and ``1`` for
        Image only when ``image_available=True``.
    neighbors : int or None, default=None
        Spatial-neighbor count. ``None`` derives it from ``shape``.
    shape : {"hexagon", "square"}, default="hexagon"
        Assumed spatial array geometry.
    scale : bool, default=True
        Min-max scale modality-specific distance matrices before integration.
    show_tree : bool, default=False
        Display the inferred tree interactively in addition to saving it.
    exclude_regions : sequence[str], default=("nan", "unknown")
        Labels excluded from tree inference.
    exclude_mode : {"contains", "exact"}, default="contains"
        How ``exclude_regions`` are matched.
    print_results : bool, default=True
        Print intermediate summaries.
    """

    output_dir: Path | str | None = None
    label_key: str = "label"
    x_key: str = "pixel_x"
    y_key: str = "pixel_y"
    image_available: bool = False
    image_feature_key: str = "uni"
    gene_filtering_paras: Optional[Dict[str, Any]] = None
    image_filtering_paras: Optional[Dict[str, Any]] = None
    weights: Optional[Dict[str, float]] = None
    neighbors: Optional[int] = None
    shape: str = "hexagon"
    scale: bool = True
    show_tree: bool = False
    exclude_regions: Sequence[str] = ("nan", "unknown")
    exclude_mode: str = "contains"
    print_results: bool = True


def construct_tree_reference_adata(
    preprocess_result,
    modalities=("Gene",),
    level="spot",
    make_image_nonnegative=True,
):
    """Combine preprocessed reference modalities feature-wise for stage 2.

    Parameters
    ----------
    preprocess_result
        ``PreprocessPipelineResult`` returned by stage 1.
    modalities
        ``("Gene",)`` or ``("Gene", "Image")``. Protein is intentionally
        excluded because the current tree algorithm distinguishes gene and
        image features only.
    level
        ``"spot"`` or ``"enhanced"``.
    make_image_nonnegative
        Shift each image feature to be non-negative before concatenation.

    Returns
    -------
    dict[str, AnnData]
        Section-level reference objects with aligned observations.
    """
    modalities = tuple(modalities)
    invalid = set(modalities) - {"Gene", "Image"}
    if invalid:
        raise ValueError(
            "Tree inference currently supports Gene and Image features only; "
            f"received {sorted(invalid)}."
        )
    if level not in {"spot", "enhanced"}:
        raise ValueError("level must be 'spot' or 'enhanced'.")
    if not modalities:
        raise ValueError("At least one modality is required.")

    source = preprocess_result.reference[level]
    section_sets = [set(source.get(modality, {})) for modality in modalities]
    if any(not sections for sections in section_sets):
        missing = [
            modality
            for modality, sections in zip(modalities, section_sets)
            if not sections
        ]
        raise ValueError(f"No stage-1 reference data are available for {missing}.")
    shared_sections = set.intersection(*section_sets)

    combined_dic = {}
    for section in source[modalities[0]]:
        if section not in shared_sections:
            continue
        objects = []
        for modality in modalities:
            adata_obj = source[modality][section]
            if modality == "Image" and make_image_nonnegative:
                adata_obj = make_nonnegative_adata(adata_obj, copy=True)
            else:
                adata_obj = adata_obj.copy()
            objects.append(adata_obj)

        base_obs = objects[0].obs.copy()
        if len(objects) == 1:
            combined = objects[0]
        else:
            combined = ad.concat(
                objects,
                axis=1,
                join="inner",
                merge="first",
                index_unique=None,
            )
            combined.obs = base_obs.reindex(combined.obs_names).copy()
        combined_dic[section] = combined

    if not combined_dic:
        raise ValueError(
            "No reference sections are shared across requested modalities."
        )
    return combined_dic


@logged_stage(
    "tree_inference",
    stage_output_from_config("results/02_tree_inference", config_position=1),
)
def run_tree_inference_stage(
    ref_adata_dic: Mapping[str, Any],
    config: TreeInferenceStageConfig,
):
    """Run Stage 2 and save human-readable and resumable outputs.

    Parameters
    ----------
    ref_adata_dic : mapping[str, AnnData]
        Reference objects keyed by section, for example
        ``{"ref_1": adata_1, "ref_2": adata_2}``. Every object must have
        unique ``obs_names``, ``config.label_key`` and coordinate columns in
        ``.obs``, and gene/image features in ``.var_names``. Use
        :func:`construct_tree_reference_adata` to build these objects from
        Stage 1.
    config : TreeInferenceStageConfig
        Tree-inference and output settings.

    Returns
    -------
    dict[str, Any]
        A dictionary with ``tree`` (:class:`HierTree`),
        ``integrated_dists``, ``integrated_ranks``, ``sample_dists_dic``,
        ``split_df``, selected-feature dictionaries, and ``metadata``.

    Saved files
    -----------
    ``tree_inference_result.pkl`` and ``stage_config.json``, plus tree text,
    PNG/pickle, split table, distance matrices, and metadata produced by
    ``infer_hier_tree_pipeline``.
    """
    from ..tree_inference import infer_hier_tree_pipeline

    output_dir = ensure_output_dir(config.output_dir or "results/02_tree_inference")
    kwargs = asdict(config)
    kwargs["output_dir"] = output_dir
    kwargs["return_results"] = True
    result = infer_hier_tree_pipeline(ref_adata_dic=dict(ref_adata_dic), **kwargs)
    save_stage_result(result, output_dir / "tree_inference_result.pkl")
    config_record = asdict(config)
    config_record["output_dir"] = str(output_dir)
    save_json(config_record, output_dir / "stage_config.json")
    return result


def rerun_tree_inference_with_weights(
    tree_result: Mapping[str, Any],
    weights: Optional[Mapping[str, float]] = None,
    reference_sections: Sequence[str] | str | None = None,
    output_dir: Path | str | None = None,
    show_tree: bool = False,
):
    """Infer a new Stage-2 tree from cached component distances.

    Parameters
    ----------
    tree_result : mapping
        Result returned by :func:`run_tree_inference_stage` with the current
        package version. It must contain ``sample_component_dists_dic`` and
        ``metadata["spot_counts"]``.
    weights : mapping[str, float] or None, default=None
        New modality weights with keys ``w_G``, ``w_I``, and ``w_S``. If
        ``None``, reuse ``tree_result["metadata"]["weights"]``.
    reference_sections : sequence[str], str, or None, default=None
        Optional subset of cached reference sections to reuse. ``None`` uses
        every reference section from the original Stage-2 result. This avoids
        rerunning feature selection and component-distance calculation when you
        want to compare trees from a subset of a larger reference cohort.
    output_dir : path-like or None, default=None
        Optional output directory for the reweighted tree result. When
        provided, this function saves ``tree_inference_result.pkl``,
        ``stage_config.json``, distance matrices, split table, and tree files.
    show_tree : bool, default=False
        Display the inferred tree interactively.

    Returns
    -------
    dict[str, Any]
        New tree-inference result using cached selected features/component
        distances, optionally restricted to ``reference_sections``, and the
        supplied weights.
    """
    from ..tree_inference import reweight_tree_inference_result

    resolved_output_dir = ensure_output_dir(output_dir) if output_dir is not None else None
    result = reweight_tree_inference_result(
        previous_result=tree_result,
        weights=weights,
        reference_sections=reference_sections,
        output_dir=resolved_output_dir,
        show_tree=show_tree,
        return_results=True,
    )

    if resolved_output_dir is not None:
        save_stage_result(result, resolved_output_dir / "tree_inference_result.pkl")
        metadata = dict(tree_result.get("metadata", {}))
        resolved_weights = result.get("metadata", {}).get("weights", weights)
        save_json(
            {
                "output_dir": str(resolved_output_dir),
                "weights": dict(resolved_weights) if resolved_weights is not None else None,
                "reference_sections": (
                    [reference_sections]
                    if isinstance(reference_sections, str)
                    else list(reference_sections)
                    if reference_sections is not None
                    else None
                ),
                "source": "precomputed_tree_component_distances",
                "previous_weights": metadata.get("weights"),
                "previous_sample_names": metadata.get("sample_names"),
            },
            resolved_output_dir / "stage_config.json",
        )

    return result


__all__ = [
    "TreeInferenceStageConfig",
    "construct_tree_reference_adata",
    "run_tree_inference_stage",
    "rerun_tree_inference_with_weights",
]
