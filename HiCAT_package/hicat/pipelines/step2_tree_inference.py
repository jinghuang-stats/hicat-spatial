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

    The fields mirror ``infer_hier_tree_pipeline``. ``output_dir`` receives
    matrices, the split table, tree text/PNG/pickle files, metadata, and a
    complete ``tree_inference_result.pkl``.
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
    """Run stage 2 and save both human-readable and resumable outputs."""
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


__all__ = [
    "TreeInferenceStageConfig",
    "construct_tree_reference_adata",
    "run_tree_inference_stage",
]
