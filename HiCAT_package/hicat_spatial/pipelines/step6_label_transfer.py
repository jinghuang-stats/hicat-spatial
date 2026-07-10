"""Stage 6: run one of the three hierarchical label-transfer frameworks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import re
from time import perf_counter
from typing import Any, Dict, Mapping

import pandas as pd

from ._io import (
    ensure_output_dir,
    logged_stage,
    save_json,
    save_stage_result,
    stage_output_from_config,
)


@dataclass
class LabelTransferStageConfig:
    """Configuration shared across query-specific transfer jobs.

    Parameters
    ----------
    scenario : {"single_ref_nn", "multi_ref_nn", "quantile"}
        Transfer framework used for every query job in this stage invocation.
        Long-form function-name aliases are also accepted.
    output_dir : path-like or None, default=None
        Stage output directory. ``None`` uses ``results/06_label_transfer``.
    mode : {"auto", "manual"}, default="auto"
        ``"auto"`` recursively processes all eligible hierarchy nodes and
        returns finalized result objects. ``"manual"`` returns sessions for
        explicit round-by-round commits.
    parameters : dict, default={}
        Default transfer-function keywords shared by all jobs, such as
        ``anchor_config``, ``assignment_config``, ``min_node_prop``, or
        ``print_results``. Values inside an individual job take precedence.
    postprocess : bool, default=False
        If True, run ``save_label_transfer_outputs`` after each automatic
        finalized transfer result. This writes prediction tables and spatial
        plots under ``output_dir/<query>/<scenario>/``.
    postprocess_parameters : dict, default={}
        Extra keywords for ``save_label_transfer_outputs``. Stage 6 manages
        ``transfer_result``, ``transfer_scenario``, ``output_dir``, and
        ``qry_section``. Useful keys include ``x_key``, ``y_key``, ``refine``,
        ``num_nbs``, ``cat_color``, ``size``, ``dpi``, ``invert_x``, and
        ``invert_y``. Defaults are ``x_key="pixel_x"``, ``y_key="pixel_y"``,
        ``refine=True``, and ``num_nbs=25``.
    save_postprocessed_h5ad : bool, default=True
        When postprocessing is enabled, save the copied Gene AnnData returned
        by ``save_label_transfer_outputs`` as
        ``output_dir/<query>/<scenario>/<query>_gene_postprocessed.h5ad``.
    save_intermediate_figures : bool, default=False
        If True, save one folder per committed hierarchy round with clustering,
        anchor-detection, and label-assignment spatial plots.
    intermediate_figure_parameters : dict, default={}
        Optional settings for intermediate round plots. Common keys are
        ``x_key``, ``y_key``, ``cat_color``, ``anchor_cat_color``, ``size``,
        ``dpi``, ``invert_x``, ``invert_y``, ``base_modality``, ``subdir``,
        ``plot_clustering``, ``plot_anchors``, ``plot_assignment``, and
        ``save_tables``.
    """

    scenario: str
    output_dir: Path | str | None = None
    mode: str = "auto"
    parameters: Dict[str, Any] = field(default_factory=dict)
    postprocess: bool = False
    postprocess_parameters: Dict[str, Any] = field(default_factory=dict)
    save_postprocessed_h5ad: bool = True
    save_intermediate_figures: bool = False
    intermediate_figure_parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LabelTransferStageResult:
    """Transfer results or manual sessions keyed by query section.

    Attributes
    ----------
    scenario : str
        Transfer scenario requested in the stage configuration.
    results_by_query : dict[str, Any]
        ``{query_section: transfer_result_or_manual_session}``.
    params : dict
        Serialized stage configuration.
    """

    scenario: str
    results_by_query: Dict[str, Any]
    params: Dict[str, Any] = field(default_factory=dict)

    def get_result(self, query_section):
        return self.results_by_query[query_section]


def _resolve_transfer_function(scenario):
    from ..label_transfer import (
        multi_ref_NN_based_label_transfer,
        quantile_based_label_transfer,
        single_ref_NN_based_label_transfer,
    )

    aliases = {
        "single_ref_nn": single_ref_NN_based_label_transfer,
        "single_ref_NN_based": single_ref_NN_based_label_transfer,
        "single_ref_NN_based_label_transfer": single_ref_NN_based_label_transfer,
        "multi_ref_nn": multi_ref_NN_based_label_transfer,
        "multi_ref_NN_based": multi_ref_NN_based_label_transfer,
        "multi_ref_NN_based_label_transfer": multi_ref_NN_based_label_transfer,
        "quantile": quantile_based_label_transfer,
        "quantile_based": quantile_based_label_transfer,
        "quantile_based_label_transfer": quantile_based_label_transfer,
    }
    if scenario not in aliases:
        raise ValueError(
            "scenario must be 'single_ref_nn', 'multi_ref_nn', or 'quantile'."
        )
    return aliases[scenario]


def _canonical_transfer_scenario(scenario):
    aliases = {
        "single_ref_nn": "single_ref_nn",
        "single_ref_NN_based": "single_ref_nn",
        "single_ref_NN_based_label_transfer": "single_ref_nn",
        "multi_ref_nn": "multi_ref_nn",
        "multi_ref_NN_based": "multi_ref_nn",
        "multi_ref_NN_based_label_transfer": "multi_ref_nn",
        "quantile": "quantile",
        "quantile_based": "quantile",
        "quantile_based_label_transfer": "quantile",
    }
    if scenario not in aliases:
        raise ValueError(
            "scenario must be 'single_ref_nn', 'multi_ref_nn', or 'quantile'."
        )
    return aliases[scenario]


def _is_finalized_transfer_result(result):
    return (
        hasattr(result, "final_labels")
        and hasattr(result, "query_adata_dic")
        and callable(getattr(result, "round_summary", None))
    )


def _save_finalized_transfer_outputs(result, query_dir, query_section):
    final_label_name = result.final_labels.name or getattr(
        result, "params", {}
    ).get("final_label_key", "final_label")
    result.final_labels.rename(final_label_name).to_csv(
        query_dir / "final_labels.csv", index=True
    )
    result.round_summary().to_csv(query_dir / "round_summary.csv", index=False)
    for modality, adata_obj in result.query_adata_dic.items():
        adata_obj.write_h5ad(
            query_dir / f"{query_section}_{str(modality).lower()}_annotated.h5ad"
        )


_DEFAULT_INTERMEDIATE_FIGURE_PARAMETERS = {
    "subdir": "intermediate_round_figures",
    "x_key": "pixel_x",
    "y_key": "pixel_y",
    "base_modality": None,
    "cat_color": None,
    "clustering_cat_color": None,
    "assignment_cat_color": None,
    "anchor_cat_color": ["#D1D1D1", "#FD2B5C"],
    "size": 50,
    "dpi": 100,
    "invert_x": False,
    "invert_y": True,
    "plot_clustering": True,
    "plot_anchors": True,
    "plot_assignment": True,
    "save_tables": True,
}


def _safe_path_component(value):
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return text or "unnamed"


def _normalize_intermediate_figure_parameters(parameters):
    config = dict(_DEFAULT_INTERMEDIATE_FIGURE_PARAMETERS)
    if parameters is None:
        return config
    if not isinstance(parameters, Mapping):
        raise TypeError("intermediate_figure_parameters must be a mapping or None.")

    invalid_keys = set(parameters) - set(config)
    if invalid_keys:
        raise ValueError(
            "Unknown intermediate_figure_parameters key(s): "
            f"{sorted(invalid_keys)}. Allowed keys: {sorted(config)}."
        )
    config.update(dict(parameters))

    config["size"] = float(config["size"])
    if config["size"] <= 0:
        raise ValueError("intermediate_figure_parameters['size'] must be positive.")
    config["dpi"] = int(config["dpi"])
    if config["dpi"] < 1:
        raise ValueError("intermediate_figure_parameters['dpi'] must be at least 1.")
    for key in ("invert_x", "invert_y", "plot_clustering", "plot_anchors", "plot_assignment", "save_tables"):
        config[key] = bool(config[key])

    return config


def _plotting_base_adata(result, obs_names, config):
    query_adata_dic = getattr(result, "query_adata_dic", {})
    if not query_adata_dic:
        return None, None

    requested_obs = pd.Index(obs_names)
    x_key = config["x_key"]
    y_key = config["y_key"]
    candidate_modalities = []
    if config["base_modality"] is not None:
        candidate_modalities.append(config["base_modality"])
    candidate_modalities.extend(
        modality for modality in query_adata_dic if modality not in candidate_modalities
    )

    for modality in candidate_modalities:
        adata = query_adata_dic.get(modality)
        if adata is None:
            continue
        if x_key not in adata.obs.columns or y_key not in adata.obs.columns:
            continue
        missing = requested_obs.difference(adata.obs_names)
        if len(missing) > 0:
            continue
        return adata[list(requested_obs), :].copy(), modality

    return None, None


def _series_to_plot_obs(plot_adata, key, series, fill_value="missing"):
    aligned = series.reindex(plot_adata.obs_names)
    plot_adata.obs[key] = aligned.fillna(fill_value).astype(str)


def _save_cat_plot(
    *,
    plot_adata,
    color_key,
    fig_title,
    fig_path,
    cat_color,
    config,
):
    from ..visualization import cat_figure

    cat_figure(
        input_adata=plot_adata,
        x_key=config["x_key"],
        y_key=config["y_key"],
        fig_title=fig_title,
        fig_path=fig_path,
        color_key=color_key,
        cat_color=cat_color,
        size=config["size"],
        dpi=config["dpi"],
        invert_x=config["invert_x"],
        invert_y=config["invert_y"],
    )


def _save_intermediate_transfer_figures(
    *,
    result,
    query_dir,
    query_section,
    figure_parameters,
):
    config = _normalize_intermediate_figure_parameters(figure_parameters)
    round_results = getattr(result, "round_results", {})
    if not round_results:
        print(
            f"[label_transfer] query={query_section!r}: no committed hierarchy "
            "rounds available for intermediate figure saving."
        )
        return

    output_root = ensure_output_dir(query_dir / config["subdir"])
    default_cat_color = config["cat_color"]
    clustering_cat_color = config["clustering_cat_color"] or default_cat_color
    assignment_cat_color = config["assignment_cat_color"] or default_cat_color

    for parent_node, round_result in round_results.items():
        if getattr(round_result, "skipped", False):
            continue

        child_nodes = list(getattr(round_result, "child_nodes", []))
        child_text = "_vs_".join(_safe_path_component(node) for node in child_nodes)
        round_name = "_".join(
            part
            for part in [_safe_path_component(parent_node), child_text]
            if part
        )
        round_dir = ensure_output_dir(output_root / round_name)

        plot_adata, base_modality = _plotting_base_adata(
            result,
            getattr(round_result, "obs_names", []),
            config,
        )
        if plot_adata is None:
            print(
                f"[label_transfer] query={query_section!r}, round={parent_node!r}: "
                "skipped intermediate figures because no query AnnData contains "
                f"coordinates {config['x_key']!r}/{config['y_key']!r} for all "
                "round observations."
            )
            continue

        if config["plot_clustering"] and round_result.clustering_result is not None:
            try:
                clustering_key = (
                    round_result.clustering_result.labels.name
                    or getattr(round_result, "clustering_config", {}).get(
                        "pred_key", "query_cluster"
                    )
                )
                _series_to_plot_obs(
                    plot_adata,
                    clustering_key,
                    round_result.clustering_result.labels.astype(str),
                )
                _save_cat_plot(
                    plot_adata=plot_adata,
                    color_key=clustering_key,
                    fig_title=(
                        f"{query_section}: {parent_node} clustering "
                        f"({base_modality} coordinates)"
                    ),
                    fig_path=round_dir / "01_clustering.png",
                    cat_color=clustering_cat_color,
                    config=config,
                )
            except Exception as exc:
                print(
                    f"[label_transfer] query={query_section!r}, round={parent_node!r}: "
                    f"could not save clustering plot: {exc}"
                )

        if config["plot_anchors"] and round_result.anchor_result is not None:
            anchor_df = round_result.anchor_result.anchor_df
            for child_node in child_nodes:
                try:
                    anchor_key = round_result.anchor_result.get_anchor_key(child_node)
                    if anchor_key not in anchor_df.columns:
                        raise KeyError(anchor_key)
                    _series_to_plot_obs(
                        plot_adata,
                        anchor_key,
                        anchor_df[anchor_key].astype(str),
                        fill_value="0",
                    )
                    _save_cat_plot(
                        plot_adata=plot_adata,
                        color_key=anchor_key,
                        fig_title=f"{query_section}: {parent_node} {child_node} anchors",
                        fig_path=round_dir / f"02_anchor_{_safe_path_component(child_node)}.png",
                        cat_color=config["anchor_cat_color"],
                        config=config,
                    )
                except Exception as exc:
                    print(
                        f"[label_transfer] query={query_section!r}, "
                        f"round={parent_node!r}, child={child_node!r}: "
                        f"could not save anchor plot: {exc}"
                    )

        if config["plot_assignment"] and round_result.assignment_result is not None:
            try:
                assignment_key = round_result.assignment_result.label_key
                _series_to_plot_obs(
                    plot_adata,
                    assignment_key,
                    round_result.assignment_result.labels.astype(str),
                )
                _save_cat_plot(
                    plot_adata=plot_adata,
                    color_key=assignment_key,
                    fig_title=f"{query_section}: {parent_node} label assignment",
                    fig_path=round_dir / "03_assignment.png",
                    cat_color=assignment_cat_color,
                    config=config,
                )
            except Exception as exc:
                print(
                    f"[label_transfer] query={query_section!r}, round={parent_node!r}: "
                    f"could not save assignment plot: {exc}"
                )

        if config["save_tables"]:
            try:
                if round_result.clustering_result is not None:
                    round_result.clustering_result.pred_df.to_csv(
                        round_dir / "clustering_labels.csv"
                    )
                if round_result.anchor_result is not None:
                    round_result.anchor_result.anchor_df.to_csv(
                        round_dir / "anchor_df.csv"
                    )
                if round_result.assignment_result is not None:
                    round_result.assignment_result.labels.rename(
                        round_result.assignment_result.label_key
                    ).to_csv(round_dir / "assignment_labels.csv")
                    round_result.assignment_result.cross_table.to_csv(
                        round_dir / "assignment_cross_table.csv"
                    )
                    round_result.assignment_result.adjusted_cross_table.to_csv(
                        round_dir / "assignment_adjusted_cross_table.csv"
                    )
            except Exception as exc:
                print(
                    f"[label_transfer] query={query_section!r}, round={parent_node!r}: "
                    f"could not save intermediate tables: {exc}"
                )


def _postprocess_finalized_transfer_result(
    *,
    result,
    output_dir,
    query_section,
    scenario,
    postprocess_parameters,
    save_postprocessed_h5ad,
):
    from ..label_transfer import save_label_transfer_outputs

    reserved = {"transfer_result", "transfer_scenario", "output_dir", "qry_section"}
    provided_reserved = reserved.intersection(postprocess_parameters)
    if provided_reserved:
        raise ValueError(
            "postprocess_parameters must not include stage-managed keys: "
            f"{sorted(provided_reserved)}."
        )

    kwargs = {
        "x_key": "pixel_x",
        "y_key": "pixel_y",
        "refine": True,
        "num_nbs": 25,
    }
    kwargs.update(dict(postprocess_parameters))
    refined_gene = save_label_transfer_outputs(
        transfer_result=result,
        transfer_scenario=scenario,
        output_dir=output_dir,
        qry_section=query_section,
        **kwargs,
    )
    if save_postprocessed_h5ad:
        postprocess_dir = ensure_output_dir(output_dir / query_section / scenario)
        refined_gene.write_h5ad(
            postprocess_dir / f"{query_section}_gene_postprocessed.h5ad"
        )
    return refined_gene


@logged_stage(
    "label_transfer",
    stage_output_from_config("results/06_label_transfer", config_position=1),
)
def run_label_transfer_stage(
    jobs: Mapping[str, Mapping[str, Any]],
    config: LabelTransferStageConfig,
):
    """Run Stage 6 for explicitly prepared query jobs.

    Parameters
    ----------
    jobs : mapping[str, mapping[str, Any]]
        Outer keys are query-section names. Each inner mapping follows the
        selected framework's function signature. ``qry_section`` defaults to
        the outer key. Exact modality names are ``"Gene"``, ``"Image"``, and
        ``"Protein"``; all selected query modalities must share/overlap unique
        ``obs_names``.

        Common required keys are ``query_adata_dic`` (modality to preprocessed
        AnnData), ``query_adata_sca_dic`` (molecular modality to scaled AnnData),
        ``hier_tree``, at least one modality-specific hierarchical feature
        result, and ``clustering_config``. A minimal clustering configuration
        contains ``selected_modalities``, ``dim_reduction_method``, and
        ``clustering_method``; KMeans also requires ``n_clusters``, while
        Leiden accepts ``resolution`` and ``n_neighbors``.

        Reference nesting differs by scenario:

        - ``single_ref_nn``: ``ref_adata_sca_dic`` is
          ``{modality: AnnData}`` and ``ref_section`` is required.
        - ``multi_ref_nn``: ``ref_adata_sca_dic`` is
          ``{section: {modality: AnnData}}`` and ``ref_section_list`` is
          required. Optional ``reference_section_guide`` maps parent nodes to
          reference subsets; missing nodes use all references and ``[]`` stops
          that branch.
        - ``quantile``: ``ref_adata_sca_dic`` is
          ``{modality: {section: AnnData}}`` and
          ``merged_ref_adata_sca_dic`` is ``{modality: merged_AnnData}``.
          Merged objects must contain the section column configured by
          ``anchor_config['merged_key']`` (default ``"sample"``).

        Example multi-reference job::

            {
                "query_1": {
                    "ref_section_list": ["ref_1", "ref_2"],
                    "ref_adata_sca_dic": {
                        "ref_1": {"Gene": ref_1_gene_scaled},
                        "ref_2": {"Gene": ref_2_gene_scaled},
                    },
                    "query_adata_dic": {"Gene": query_1_gene},
                    "query_adata_sca_dic": {
                        "Gene": query_1_gene_scaled,
                    },
                    "hier_tree": tree,
                    "gene_feature_results": gene_features,
                    "clustering_config": clustering_config,
                    "reference_section_guide": {
                        "node_0": ["ref_1", "ref_2"],
                    },
                }
            }

    config : LabelTransferStageConfig
        Scenario, automatic/manual mode, common defaults, and output path.

    Returns
    -------
    LabelTransferStageResult
        Per-query automatic transfer results or manual sessions.

    Saved files
    -----------
    Always saves timing, stage configuration, the aggregate stage pickle, and
    one per-query result/session pickle. Automatic finalized results also save
    labels, round summaries, and annotated modality ``.h5ad`` files. When
    ``config.postprocess=True``, finalized results also run
    ``save_label_transfer_outputs`` and save the returned copied Gene AnnData
    as ``<query>_gene_postprocessed.h5ad``.

    Manual-mode sessions are saved as pickles, but finalized CSV, H5AD, and
    postprocessing outputs are skipped until the user materializes a result.
    """
    if config.mode not in {"auto", "manual"}:
        raise ValueError("mode must be 'auto' or 'manual'.")
    if not jobs:
        raise ValueError("jobs cannot be empty.")

    transfer_function = _resolve_transfer_function(config.scenario)
    canonical_scenario = _canonical_transfer_scenario(config.scenario)
    output_dir = ensure_output_dir(config.output_dir or "results/06_label_transfer")
    results_by_query = {}
    timing_rows = []
    timing_path = output_dir / "label_transfer_timing.csv"

    for query_section, job in jobs.items():
        query_dir = ensure_output_dir(output_dir / str(query_section))
        job = dict(job)
        run_postprocess = job.pop("postprocess", config.postprocess)
        postprocess_parameters = dict(config.postprocess_parameters)
        postprocess_parameters.update(dict(job.pop("postprocess_parameters", {})))
        save_intermediate_figures = job.pop(
            "save_intermediate_figures", config.save_intermediate_figures
        )
        intermediate_figure_parameters = dict(config.intermediate_figure_parameters)
        intermediate_figure_parameters.update(
            dict(job.pop("intermediate_figure_parameters", {}))
        )
        save_postprocessed_h5ad = job.pop(
            "save_postprocessed_h5ad", config.save_postprocessed_h5ad
        )
        kwargs = dict(config.parameters)
        kwargs.update(job)
        kwargs.setdefault("qry_section", query_section)
        kwargs["mode"] = config.mode
        kwargs.setdefault("output_dir", str(query_dir))
        query_started_at = datetime.now(timezone.utc)
        query_started_clock = perf_counter()
        print(
            f"[label_transfer] query={query_section!r} started at "
            f"{query_started_at.isoformat()}"
        )
        try:
            result = transfer_function(**kwargs)
            results_by_query[query_section] = result
            save_stage_result(result, query_dir / "label_transfer_result.pkl")

            if _is_finalized_transfer_result(result):
                _save_finalized_transfer_outputs(result, query_dir, query_section)
                if save_intermediate_figures:
                    _save_intermediate_transfer_figures(
                        result=result,
                        query_dir=query_dir,
                        query_section=str(query_section),
                        figure_parameters=intermediate_figure_parameters,
                    )
                if run_postprocess:
                    _postprocess_finalized_transfer_result(
                        result=result,
                        output_dir=output_dir,
                        query_section=str(query_section),
                        scenario=canonical_scenario,
                        postprocess_parameters=postprocess_parameters,
                        save_postprocessed_h5ad=save_postprocessed_h5ad,
                    )
            else:
                print(
                    f"[label_transfer] query={query_section!r} returned a manual "
                    "session or unfinished object; skipped finalized CSV/H5AD "
                    "and postprocessing outputs."
                )
        except BaseException as exc:
            query_ended_at = datetime.now(timezone.utc)
            elapsed_seconds = perf_counter() - query_started_clock
            timing_rows.append(
                {
                    "query_section": query_section,
                    "scenario": config.scenario,
                    "mode": config.mode,
                    "status": "failed",
                    "started_at": query_started_at.isoformat(),
                    "ended_at": query_ended_at.isoformat(),
                    "elapsed_seconds": elapsed_seconds,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            pd.DataFrame(timing_rows).to_csv(timing_path, index=False)
            raise
        query_ended_at = datetime.now(timezone.utc)
        elapsed_seconds = perf_counter() - query_started_clock
        timing_rows.append(
            {
                "query_section": query_section,
                "scenario": config.scenario,
                "mode": config.mode,
                "status": "completed",
                "started_at": query_started_at.isoformat(),
                "ended_at": query_ended_at.isoformat(),
                "elapsed_seconds": elapsed_seconds,
                "error": None,
            }
        )
        pd.DataFrame(timing_rows).to_csv(timing_path, index=False)
        print(
            f"[label_transfer] query={query_section!r} completed in "
            f"{elapsed_seconds:.3f} seconds"
        )

    config_record = asdict(config)
    config_record["output_dir"] = str(output_dir)
    stage_result = LabelTransferStageResult(
        scenario=config.scenario,
        results_by_query=results_by_query,
        params=config_record,
    )
    save_stage_result(stage_result, output_dir / "label_transfer_stage_result.pkl")
    save_json(config_record, output_dir / "stage_config.json")
    return stage_result


__all__ = [
    "LabelTransferStageConfig",
    "LabelTransferStageResult",
    "run_label_transfer_stage",
]
