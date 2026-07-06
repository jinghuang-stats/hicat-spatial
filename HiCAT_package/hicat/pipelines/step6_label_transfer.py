"""Stage 6: run one of the three hierarchical label-transfer frameworks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
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
    scenario
        ``"single_ref_nn"``, ``"multi_ref_nn"``, or ``"quantile"``.
    mode
        ``"auto"`` recursively processes the subtree; ``"manual"`` returns
        sessions for round-by-round inspection and adjustment.
    parameters
        Default keyword arguments merged into every job. Query-specific values
        in ``jobs`` take precedence.
    """

    scenario: str
    output_dir: Path | str | None = None
    mode: str = "auto"
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LabelTransferStageResult:
    """Transfer results or manual sessions keyed by query section."""

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


@logged_stage(
    "label_transfer",
    stage_output_from_config("results/06_label_transfer", config_position=1),
)
def run_label_transfer_stage(
    jobs: Mapping[str, Mapping[str, Any]],
    config: LabelTransferStageConfig,
):
    """Run stage 6 for explicitly prepared query jobs.

    ``jobs`` is keyed by query section. Each inner mapping follows the selected
    framework's function signature, excluding ``qry_section`` which is filled
    from the outer key when omitted.

    Single-reference jobs require ``ref_adata_sca_dic``, ``query_adata_dic``,
    ``query_adata_sca_dic``, ``ref_section``, ``hier_tree``, hierarchical
    feature results, and ``clustering_config``.

    Multi-reference jobs replace ``ref_section`` with ``ref_section_list`` and
    use section-first references. Quantile jobs use modality-first references
    plus ``merged_ref_adata_sca_dic``. This explicit job boundary prevents the
    three incompatible reference dictionary layouts from being confused.
    """
    if config.mode not in {"auto", "manual"}:
        raise ValueError("mode must be 'auto' or 'manual'.")
    if not jobs:
        raise ValueError("jobs cannot be empty.")

    transfer_function = _resolve_transfer_function(config.scenario)
    output_dir = ensure_output_dir(config.output_dir or "results/06_label_transfer")
    results_by_query = {}
    timing_rows = []
    timing_path = output_dir / "label_transfer_timing.csv"

    for query_section, job in jobs.items():
        query_dir = ensure_output_dir(output_dir / str(query_section))
        kwargs = dict(config.parameters)
        kwargs.update(dict(job))
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
        results_by_query[query_section] = result
        save_stage_result(result, query_dir / "label_transfer_result.pkl")

        if hasattr(result, "final_labels"):
            final_label_name = result.final_labels.name or getattr(
                result, "params", {}
            ).get("final_label_key", "final_label")
            result.final_labels.rename(final_label_name).to_csv(
                query_dir / "final_labels.csv", index=True
            )
            result.round_summary().to_csv(query_dir / "round_summary.csv", index=False)
            for modality, adata_obj in result.query_adata_dic.items():
                adata_obj.write_h5ad(
                    query_dir
                    / f"{query_section}_{str(modality).lower()}_annotated.h5ad"
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
