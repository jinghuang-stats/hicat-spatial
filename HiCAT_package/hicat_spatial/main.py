"""Thin coordinator for the seven-stage HiCAT analysis workflow.

Each stage can be run independently from ``hicat_spatial.pipelines``. This module
connects configured stages in order and passes only well-defined result
objects between them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .pipelines._io import ensure_output_dir, load_stage_result, save_json

from .pipelines.step1_preprocessing import (
    PreprocessConfig,
    PreprocessPipelineResult,
    run_preprocessing_pipeline,
)

from .pipelines.step2_tree_inference import (
    TreeInferenceStageConfig,
    construct_tree_reference_adata,
    run_tree_inference_stage,
)

from .pipelines.step3_reference_selection import (
    ReferenceSelectionStageConfig,
    run_reference_selection_stage,
)

from .pipelines.step4_hierarchical_features import (
    HierarchicalFeatureStageConfig,
    run_hierarchical_feature_stage,
)

from .pipelines.step5_clustering_config import (
    ClusteringConfigStageConfig,
    run_clustering_config_stage,
)

from .pipelines.step6_label_transfer import (
    LabelTransferStageConfig,
    run_label_transfer_stage,
)

from .pipelines.step7_heterogeneity import (
    HeterogeneityStageConfig,
    run_heterogeneity_stage,
)


STAGE_NAMES = {
    1: "preprocessing",
    2: "tree_inference",
    3: "reference_selection",
    4: "hierarchical_features",
    5: "clustering_config",
    6: "label_transfer",
    7: "heterogeneity",
}


@dataclass
class HiCATWorkflowConfig:
    """Configuration for an ordered HiCAT run.

    ``preprocessing`` is required. Later stage configurations are optional;
    omitted stages are skipped. ``output_root`` contains stages 2–7. Stage 1
    uses ``preprocessing.data_dir`` for flat raw inputs and
    ``preprocessing.preprocess_dir`` for package-managed preprocessing outputs.
    """

    preprocessing: PreprocessConfig # required
    output_root: Path | str = "results"
    tree_inference: Optional[TreeInferenceStageConfig] = None
    reference_selection: Optional[ReferenceSelectionStageConfig] = None
    hierarchical_features: Optional[HierarchicalFeatureStageConfig] = None
    clustering_config: Optional[ClusteringConfigStageConfig] = None
    label_transfer: Optional[LabelTransferStageConfig] = None
    heterogeneity: Optional[HeterogeneityStageConfig] = None
    tree_modalities: Sequence[str] = ("Gene",)


@dataclass
class HiCATWorkflowResult:
    """Results from configured stages; skipped stages remain ``None``."""

    preprocessing: PreprocessPipelineResult
    tree_inference: Optional[Any] = None
    reference_selection: Optional[Any] = None
    hierarchical_features: Optional[Any] = None
    clustering_config: Optional[Any] = None
    label_transfer: Optional[Any] = None
    heterogeneity: Optional[Any] = None
    output_dirs: Dict[str, str] = field(default_factory=dict)


def _stage_config_with_default_output(stage_config, output_path):
    if stage_config is None or stage_config.output_dir is not None:
        return stage_config
    return replace(stage_config, output_dir=output_path)


def _inject_label_transfer_dependencies(jobs, tree, feature_result):
    """Fill tree/feature inputs while preserving user job overrides."""
    updated_jobs = {}
    for query_section, original_job in jobs.items():
        job = dict(original_job)
        job.setdefault("hier_tree", tree)
        if feature_result is not None:
            modality_results = {}
            if hasattr(feature_result, "get_modality_result"):
                for modality in ("Gene", "Image", "Protein"):
                    try:
                        modality_results[modality] = feature_result.get_modality_result(
                            query_section,
                            modality,
                        )
                    except (KeyError, AttributeError):
                        continue

            if not modality_results:
                feature_results_by_query = getattr(
                    feature_result, "feature_results_by_query", {}
                )
                modality_results = feature_results_by_query.get(query_section)
            if modality_results is None:
                raise KeyError(
                    f"No hierarchical feature result exists for {query_section!r}."
                )
            job.setdefault("gene_feature_results", modality_results.get("Gene"))
            job.setdefault("image_feature_results", modality_results.get("Image"))
            job.setdefault("protein_feature_results", modality_results.get("Protein"))
        updated_jobs[query_section] = job
    return updated_jobs


def run_hicat_workflow(
    config: HiCATWorkflowConfig,
    label_transfer_jobs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> HiCATWorkflowResult:
    """Run configured HiCAT stages in their required order.

    Parameters
    ----------
    config
        Stage configurations. A later configured stage requires the outputs of
        its prerequisite stages.

    label_transfer_jobs : dict[str, dict[str, Any]], optional
        Query-specific input dictionaries for stage 6 label transfer.   

        The outer dictionary is keyed by query section name:    

            {
                "query_section": {
                    ...
                }
            }   

        Each inner dictionary contains the inputs required by the selected
        label-transfer framework. The exact required keys depend on whether the
        stage uses single-reference NN, multi-reference NN, or quantile-based
        transfer.   

        The workflow coordinator automatically injects the following keys when
        they are omitted and the corresponding upstream results are available:  

            - "hier_tree"
            - "gene_feature_results"
            - "image_feature_results"
            - "protein_feature_results" 

        Users must still provide query/reference scaled AnnData inputs and the
        final clustering configuration explicitly.

    Returns
    -------
    HiCATWorkflowResult
        In-memory result objects for every configured stage. Each stage also
        saves independently reloadable outputs in its numbered directory.
    """
    output_root = ensure_output_dir(config.output_root)
    output_dirs = {
        "preprocessing": str(Path(config.preprocessing.preprocess_dir)),
        **{
            STAGE_NAMES[index]: str(output_root / f"{index:02d}_{STAGE_NAMES[index]}")
            for index in range(2, 8)
        },
    }

    # 1. Preprocessing
    preprocess_result = run_preprocessing_pipeline(config.preprocessing)
    workflow_result = HiCATWorkflowResult(
        preprocessing=preprocess_result,
        output_dirs=output_dirs,
    )
    reference_by_modality = preprocess_result.reference["spot"]

    # 2. Tree inference
    if config.tree_inference is not None:
        tree_config = _stage_config_with_default_output(
            config.tree_inference, output_dirs["tree_inference"]
        )
        uses_image = "Image" in config.tree_modalities
        if bool(tree_config.image_available) != uses_image:
            raise ValueError(
                "tree_inference.image_available must match whether "
                "tree_modalities contains 'Image'."
            )
        tree_inputs = construct_tree_reference_adata(
            preprocess_result,
            modalities=config.tree_modalities,
            level="spot",
        )
        workflow_result.tree_inference = run_tree_inference_stage(
            tree_inputs, tree_config
        )

    # 3. Query-specific reference selection
    if config.reference_selection is not None:
        if (
            "Gene" not in reference_by_modality
            or "Gene" not in preprocess_result.query["spot"]
        ):
            raise ValueError(
                "Reference selection requires reference and query Gene data."
            )
        reference_config = _stage_config_with_default_output(
            config.reference_selection, output_dirs["reference_selection"]
        )
        workflow_result.reference_selection = run_reference_selection_stage(
            reference_by_modality["Gene"],
            preprocess_result.query["spot"]["Gene"],
            reference_config,
        )

    # 4. Hierarchical feature selection
    if config.hierarchical_features is not None:
        if workflow_result.tree_inference is None:
            raise ValueError("Stage 4 requires a configured stage-2 tree result.")
        if workflow_result.reference_selection is None:
            raise ValueError("Stage 4 requires configured stage-3 selected references.")
        feature_config = _stage_config_with_default_output(
            config.hierarchical_features, output_dirs["hierarchical_features"]
        )
        workflow_result.hierarchical_features = run_hierarchical_feature_stage(
            ref_adata_by_modality=reference_by_modality,
            hier_tree=workflow_result.tree_inference["tree"],
            config=feature_config,
            selected_refs_dic=workflow_result.reference_selection.selected_refs_dic,
        )

    # 5. Clustering/embedding configuration
    if config.clustering_config is not None:
        if workflow_result.hierarchical_features is None:
            raise ValueError("Stage 5 requires configured stage-4 feature results.")
        clustering_config = _stage_config_with_default_output(
            config.clustering_config, output_dirs["clustering_config"]
        )
        workflow_result.clustering_config = run_clustering_config_stage(
            ref_adata_by_modality=reference_by_modality,
            feature_stage_result=workflow_result.hierarchical_features,
            config=clustering_config,
        )

    # 6. Hierarchical label transfer
    if config.label_transfer is not None:
        if workflow_result.tree_inference is None:
            raise ValueError("Stage 6 requires a configured stage-2 tree result.")
        if not label_transfer_jobs:
            raise ValueError(
                "Stage 6 is configured but label_transfer_jobs was not supplied."
            )
        transfer_config = _stage_config_with_default_output(
            config.label_transfer, output_dirs["label_transfer"]
        )
        jobs = _inject_label_transfer_dependencies(
            label_transfer_jobs,
            workflow_result.tree_inference["tree"],
            workflow_result.hierarchical_features,
        )
        workflow_result.label_transfer = run_label_transfer_stage(
            jobs=jobs,
            config=transfer_config,
        )

    # 7. Reference heterogeneity inference
    if config.heterogeneity is not None:
        if "Gene" not in reference_by_modality:
            raise ValueError("Stage 7 requires reference Gene data.")
        heterogeneity_config = _stage_config_with_default_output(
            config.heterogeneity, output_dirs["heterogeneity"]
        )
        workflow_result.heterogeneity = run_heterogeneity_stage(
            reference_by_modality["Gene"],
            heterogeneity_config,
        )

    save_json(
        {
            "configured_stages": {
                name: getattr(config, name) is not None
                for name in STAGE_NAMES.values()
                if name != "preprocessing"
            },
            "output_dirs": output_dirs,
            "workflow_config": asdict(config),
        },
        output_root / "workflow_manifest.json",
    )
    return workflow_result


def main():
    """Show the library entry-point guidance when invoked as a module."""
    raise SystemExit(
        "HiCAT uses explicit stage configuration. Import HiCATWorkflowConfig "
        "and run_hicat_workflow, or run functions from hicat_spatial.pipelines. "
        "See HICAT_WORKFLOW_GUIDE.md."
    )


if __name__ == "__main__":
    main()


__all__ = [
    "HiCATWorkflowConfig",
    "HiCATWorkflowResult",
    "run_hicat_workflow",
    "load_stage_result",
]
