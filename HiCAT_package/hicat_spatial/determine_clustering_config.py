import numpy as np
import pandas as pd
import re

from sklearn.metrics import adjusted_rand_score

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# Local import
from .utils import (
    compute_pca_embedding,
    get_ref_modality_adata,
    get_valid_label_mask,
    kmeans_clustering,
)
from .multi_modal_integration import (
    cluster_integrated_embedding,
    integrate_modalities_for_section,
)
from .visualization import cat_figure


SUPPORTED_REDUCTION_METHODS = ("pca", "selected_features")


_DEFAULT_VISUALIZATION_CONFIG = {
    "plot_modality_clusters": False,
    "plot_dim_reduction_clusters": False,
    "output_dir": None,
    "x_key": "x",
    "y_key": "y",
    "cat_color": None,
    "size": 50,
    "dpi": 100,
    "invert_x": False,
    "invert_y": True,
}


_DEFAULT_HIPT_BOUNDARY_REFINEMENT_CONFIG = {
    "enabled": False,
    "image_modality": "Image",
    "extra_boundary_cluster": True,
    "x_key": None,
    "y_key": None,
    "boundary_cluster": None,
    "boundary_features": None,
    "candidate_feature_sets": None,
    "min_cluster_size": 1,
    "max_boundary_score_ratio": None,
    "bd_num_nbs": 25,
    "smooth_after_reassign": True,
    "smooth_num_nbs": 15,
    "metric": "euclidean",
    "weighted_vote": False,
    "use_refined_ari": True,
    "strict": False,
}


def _normalize_visualization_config(
    visualization_config: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Validate and fill optional clustering-visualization settings."""
    config = dict(_DEFAULT_VISUALIZATION_CONFIG)
    if visualization_config is None:
        return config
    if not isinstance(visualization_config, Mapping):
        raise TypeError("visualization_config must be a mapping or None.")

    invalid_keys = set(visualization_config) - set(config)
    if invalid_keys:
        raise ValueError(
            "Unknown visualization_config key(s): "
            f"{sorted(invalid_keys)}. Allowed keys: {sorted(config)}."
        )
    config.update(dict(visualization_config))

    plotting_enabled = bool(config["plot_modality_clusters"]) or bool(
        config["plot_dim_reduction_clusters"]
    )
    if plotting_enabled and config["output_dir"] is None:
        raise ValueError(
            "visualization_config['output_dir'] is required when clustering "
            "visualization is enabled."
        )
    if float(config["size"]) <= 0:
        raise ValueError("visualization_config['size'] must be positive.")
    if int(config["dpi"]) < 1:
        raise ValueError("visualization_config['dpi'] must be at least 1.")

    return config


def _normalize_hipt_boundary_refinement_config(
    hipt_boundary_refinement_config: Optional[Mapping[str, Any]],
    visualization_config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate and fill optional HIPT boundary-refinement settings."""
    config = dict(_DEFAULT_HIPT_BOUNDARY_REFINEMENT_CONFIG)

    if hipt_boundary_refinement_config is None:
        return config
    if not isinstance(hipt_boundary_refinement_config, Mapping):
        raise TypeError("hipt_boundary_refinement_config must be a mapping or None.")

    invalid_keys = set(hipt_boundary_refinement_config) - set(config)
    if invalid_keys:
        raise ValueError(
            "Unknown hipt_boundary_refinement_config key(s): "
            f"{sorted(invalid_keys)}. Allowed keys: {sorted(config)}."
        )

    config.update(dict(hipt_boundary_refinement_config))
    config["enabled"] = bool(config["enabled"])
    config["extra_boundary_cluster"] = bool(config["extra_boundary_cluster"])
    config["smooth_after_reassign"] = bool(config["smooth_after_reassign"])
    config["weighted_vote"] = bool(config["weighted_vote"])
    config["use_refined_ari"] = bool(config["use_refined_ari"])
    config["strict"] = bool(config["strict"])

    for key in ("min_cluster_size", "bd_num_nbs", "smooth_num_nbs"):
        config[key] = int(config[key])
        if config[key] < 1:
            raise ValueError(f"hipt_boundary_refinement_config['{key}'] must be >= 1.")

    if config["candidate_feature_sets"] is not None:
        config["candidate_feature_sets"] = tuple(
            tuple(feature_set) for feature_set in config["candidate_feature_sets"]
        )

    if config["enabled"]:
        viz_config = _normalize_visualization_config(visualization_config)
        config["x_key"] = config["x_key"] or viz_config["x_key"]
        config["y_key"] = config["y_key"] or viz_config["y_key"]

    return config


def _apply_hipt_boundary_refinement_to_clusters(
    image_adata,
    cluster_labels,
    cluster_key: str,
    config: Mapping[str, Any],
    print_results: bool = True,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Refine HIPT boundary clusters for a modality-ARI evaluation."""
    from .query_clustering import refine_hipt_boundary_clusters

    eval_adata = image_adata.copy()
    eval_adata.obs[cluster_key] = pd.Categorical(np.asarray(cluster_labels).astype(str))

    kwargs = {
        "image_adata": eval_adata,
        "cluster_key": cluster_key,
        "x_key": config["x_key"],
        "y_key": config["y_key"],
        "boundary_cluster": config["boundary_cluster"],
        "boundary_features": config["boundary_features"],
        "min_cluster_size": config["min_cluster_size"],
        "max_boundary_score_ratio": config["max_boundary_score_ratio"],
        "bd_num_nbs": config["bd_num_nbs"],
        "smooth_after_reassign": config["smooth_after_reassign"],
        "smooth_num_nbs": config["smooth_num_nbs"],
        "metric": config["metric"],
        "weighted_vote": config["weighted_vote"],
        "copy": False,
        "print_results": print_results,
    }
    if config["candidate_feature_sets"] is not None:
        kwargs["candidate_feature_sets"] = config["candidate_feature_sets"]

    refined_adata, info = refine_hipt_boundary_clusters(**kwargs)
    final_cluster_key = info["final_cluster_key"]
    refined_labels = refined_adata.obs[final_cluster_key].astype(str).to_numpy()

    return refined_labels, info


def _safe_path_component(value: Any) -> str:
    """Convert a user/sample label into a portable filename component."""
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return text or "unnamed"


def _save_clustering_pattern(
    input_adata,
    cluster_labels,
    cluster_key: str,
    stage: str,
    group_name: str,
    section: str,
    title: str,
    visualization_config: Optional[Mapping[str, Any]],
    print_results: bool = True,
) -> Optional[str]:
    """Save one categorical cluster map without modifying the source AnnData."""
    config = _normalize_visualization_config(visualization_config)
    stage_flag = {
        "informative_modalities": "plot_modality_clusters",
        "dimension_reduction": "plot_dim_reduction_clusters",
    }.get(stage)
    if stage_flag is None:
        raise ValueError(f"Unsupported visualization stage: {stage!r}.")
    if not config[stage_flag]:
        return None

    figure_path = None
    try:
        plot_adata = input_adata.copy()
        labels = np.asarray(cluster_labels).astype(str)
        if labels.ndim != 1 or labels.shape[0] != plot_adata.n_obs:
            raise ValueError(
                "cluster_labels must be one-dimensional and match "
                "input_adata.n_obs."
            )
        plot_adata.obs[cluster_key] = pd.Categorical(labels)

        figure_path = (
            Path(config["output_dir"])
            / stage
            / _safe_path_component(group_name)
            / f"{_safe_path_component(section)}_clusters.png"
        )
        cat_figure(
            input_adata=plot_adata,
            x_key=config["x_key"],
            y_key=config["y_key"],
            fig_title=title,
            fig_path=figure_path,
            color_key=cluster_key,
            cat_color=config["cat_color"],
            size=config["size"],
            dpi=config["dpi"],
            invert_x=bool(config["invert_x"]),
            invert_y=bool(config["invert_y"]),
        )
    except Exception as exc:
        if print_results:
            target = figure_path or f"{stage}/{group_name}/{section}"
            print(f"Could not save clustering plot {target}: {exc}")
        return None

    if print_results:
        print(f"Saved clustering plot: {figure_path}")
    return str(figure_path)


@dataclass
class MultiModalClusteringConfigResult:
    """
    Store automatically determined multi-modal clustering configuration.

    This dataclass records two decisions:

    1. Which modalities are informative enough to use.
    2. Which dimension-reduction method performs better.

    It does NOT automatically choose the final clustering method.
    The user should later specify whether to use KMeans or Leiden.

    When optional visualization is enabled, saved paths are recorded in
    ``modality_ari_df["plot_path"]`` and in each DataFrame stored under
    ``dim_reduction_ari_df_dic[method]["plot_path"]``.
    """

    # ------------------------------------------------------------
    # 1. Automatically selected modalities
    # ------------------------------------------------------------
    selected_modalities: List[str]

    modality_ari_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    modality_avg_ari: pd.DataFrame = field(default_factory=pd.DataFrame)
    ranked_modalities: List[str] = field(default_factory=list)

    selected_modalities_hard: List[str] = field(default_factory=list)
    selected_modalities_relative: List[str] = field(default_factory=list)

    modality_selection_criterion: Optional[str] = None
    modality_selection_info: Dict[str, Any] = field(default_factory=dict)
    modality_selection_params: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------
    # 2. Automatically selected dimension-reduction method
    # ------------------------------------------------------------
    dim_reduction_method: str = "pca"

    dim_reduction_summary_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    dim_reduction_ari_df_dic: Dict[str, pd.DataFrame] = field(default_factory=dict)
    dim_reduction_average_ari_dic: Dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------
    # 3. Embedding/integration parameters
    # ------------------------------------------------------------
    features_format: str = "section"
    pcs_num_dic: Dict[str, int] = field(default_factory=dict)
    default_pcs_num: int = 30
    scale_embedding: bool = True
    align_by_obs_names: bool = False
    random_state: int = 0
    visualization_config: Dict[str, Any] = field(default_factory=dict)

    @property
    def selected_modality_average_ari(self) -> Optional[float]:
        """
        Average ARI of the final selected modalities.
        """
        if self.modality_avg_ari is None or self.modality_avg_ari.empty:
            return None

        required_cols = {"modality", "average_ari"}
        if not required_cols.issubset(self.modality_avg_ari.columns):
            return None

        mask = self.modality_avg_ari["modality"].isin(self.selected_modalities)

        if mask.sum() == 0:
            return None

        return float(self.modality_avg_ari.loc[mask, "average_ari"].mean())

    @property
    def best_dim_reduction_average_ari(self) -> Optional[float]:
        """
        Average ARI of the selected dimension-reduction method.
        """
        if self.dim_reduction_method in self.dim_reduction_average_ari_dic:
            return float(
                self.dim_reduction_average_ari_dic[self.dim_reduction_method]
            )

        if (
            self.dim_reduction_summary_df is not None
            and not self.dim_reduction_summary_df.empty
            and {"dim_reduction_method", "average_ari"}.issubset(
                self.dim_reduction_summary_df.columns
            )
        ):
            row = self.dim_reduction_summary_df[
                self.dim_reduction_summary_df["dim_reduction_method"]
                == self.dim_reduction_method
            ]

            if not row.empty:
                return float(row.iloc[0]["average_ari"])

        return None

    @classmethod
    def from_result_dics(
        cls, # means the class itself
        modality_result: Mapping[str, Any],
        dim_reduction_result: Mapping[str, Any],
        scale_embedding: bool = True,
        pcs_num_dic: Optional[Mapping[str, int]] = None,
        default_pcs_num: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> "MultiModalClusteringConfigResult":
        """
        Create the dataclass from the outputs of:

        1. select_informative_modalities()
        2. determine_dimension_reduction_method()
        """

        modality_selection_params = dict(
            modality_result.get("selection_params", {})
        )

        if pcs_num_dic is None:
            pcs_num_dic = modality_selection_params.get("pcs_num_dic", {})

        if default_pcs_num is None:
            default_pcs_num = modality_selection_params.get(
                "default_pcs_num", 30
            )

        if random_state is None:
            random_state = modality_selection_params.get("random_state", 0)

        return cls(
            selected_modalities=list(modality_result["selected_modalities"]),

            modality_ari_df=modality_result.get(
                "ari_df", pd.DataFrame()
            ),
            modality_avg_ari=modality_result.get(
                "avg_ari", pd.DataFrame()
            ),
            ranked_modalities=list(
                modality_result.get("ranked_modalities", [])
            ),
            selected_modalities_hard=list(
                modality_result.get("selected_modalities_hard", [])
            ),
            selected_modalities_relative=list(
                modality_result.get("selected_modalities_relative", [])
            ),
            modality_selection_criterion=modality_result.get(
                "selection_criterion"
            ),
            modality_selection_info=dict(
                modality_result.get("selection_info", {})
            ),
            modality_selection_params=modality_selection_params,

            dim_reduction_method=dim_reduction_result["best_method"],
            dim_reduction_summary_df=dim_reduction_result.get(
                "summary_df", pd.DataFrame()
            ),
            dim_reduction_ari_df_dic=dict(
                dim_reduction_result.get("ari_df_dic", {})
            ),
            dim_reduction_average_ari_dic=dict(
                dim_reduction_result.get("average_ari_dic", {})
            ),

            features_format=dim_reduction_result.get(
                "features_format", "section"
            ),
            align_by_obs_names=dim_reduction_result.get(
                "align_by_obs_names", False
            ),

            pcs_num_dic=dict(pcs_num_dic),
            default_pcs_num=int(default_pcs_num),
            scale_embedding=scale_embedding,
            random_state=int(random_state),
            visualization_config=dict(
                dim_reduction_result.get(
                    "visualization_config",
                    modality_selection_params.get("visualization_config", {}),
                )
            ),
        )

    def to_clustering_config(
        self,
        clustering_method: str,
        n_clusters: Optional[int] = None,
        resolution: Optional[float] = None,
        n_neighbors: Optional[int] = None,
        random_state: Optional[int] = None,
        features_format: Optional[str] = None,
        **extra_params: Any,
    ) -> Dict[str, Any]:
        """
        Convert stored modality/dimension-reduction decisions into a plain
        clustering_config dictionary.

        The user must specify the final clustering method here.
        """

        if clustering_method not in {"kmeans", "leiden"}:
            raise ValueError(
                "clustering_method must be either 'kmeans' or 'leiden'."
            )

        config = {
            "selected_modalities": list(self.selected_modalities),
            "dim_reduction_method": self.dim_reduction_method,
            "features_format": features_format or self.features_format,
            "pcs_num_dic": dict(self.pcs_num_dic),
            "default_pcs_num": self.default_pcs_num,
            "scale_embedding": self.scale_embedding,
            "align_by_obs_names": self.align_by_obs_names,
            "clustering_method": clustering_method,
            "random_state": (
                self.random_state if random_state is None else int(random_state)
            ),
        }

        if clustering_method == "kmeans":
            if n_clusters is None:
                raise ValueError(
                    "n_clusters must be specified when clustering_method='kmeans'."
                )

            config["n_clusters"] = int(n_clusters)

        else:
            config["resolution"] = 0.5 if resolution is None else float(resolution)
            config["n_neighbors"] = 15 if n_neighbors is None else int(n_neighbors)

        config.update(extra_params)

        return config

    def summary(self) -> Dict[str, Any]:
        """
        Return a compact summary of the automatically determined configuration.
        """

        return {
            "selected_modalities": list(self.selected_modalities),
            "selected_modality_average_ari": self.selected_modality_average_ari,
            "ranked_modalities": list(self.ranked_modalities),
            "dim_reduction_method": self.dim_reduction_method,
            "best_dim_reduction_average_ari": self.best_dim_reduction_average_ari,
            "features_format": self.features_format,
            "pcs_num_dic": dict(self.pcs_num_dic),
            "default_pcs_num": self.default_pcs_num,
            "scale_embedding": self.scale_embedding,
            "align_by_obs_names": self.align_by_obs_names,
            "random_state": self.random_state,
            "visualization_config": dict(self.visualization_config),
        }


#=======================================================================
# Part 1. Select the informative modalities
#=======================================================================
def evaluate_modality_ari(
    modality_ref_dic,
    pcs_num_dic=None,
    default_pcs_num=30,
    label_key="label",
    min_spots=10,
    exclude_regions=("nan", "unknown"),
    exclude_mode="exact",
    random_state=0,
    print_results=True,
    visualization_config=None,
    hipt_boundary_refinement_config=None,
):
    """
    Evaluate each modality by computing clustering ARI across reference sections.

    When ``visualization_config['plot_modality_clusters']`` is true, save the
    successful section-level KMeans patterns under the configured output root.

    For each modality and reference section, this function:
        1. filters spots/cells with invalid labels,
        2. computes PCA embeddings,
        3. performs K-means clustering using the number of annotated labels,
        4. computes adjusted Rand index between true labels and clusters,
        5. summarizes ARI scores across reference sections.

    Parameters
    ----------
    modality_ref_dic : dict
        Dictionary of modality-specific reference AnnData dictionaries.

        Example:
        {
            "gene": {"H1": adata_gene_H1, "G2": adata_gene_G2},
            "image": {"H1": adata_img_H1, "G2": adata_img_G2},
        }

    pcs_num_dic : dict or None, optional
        Dictionary specifying the number of PCs for each modality.
        If a modality is not found, `default_pcs_num` is used.

    default_pcs_num : int, optional
        Default number of PCs used for PCA embedding.

    label_key : str, optional
        Column in `adata.obs` containing annotated tissue labels.

    min_spots : int, optional
        Minimum number of spots/cells required to evaluate a section.

    exclude_regions : tuple, optional
        Labels to exclude before ARI calculation.

    exclude_mode : {"exact", "contains"}, optional
        Whether to exclude labels by exact matching or substring matching.

    random_state : int, optional
        Random seed for PCA and K-means.

    print_results : bool, optional
        Whether to print progress and ARI results.

    visualization_config : mapping or None, optional
        Shared plotting configuration. Set ``plot_modality_clusters=True``
        and provide ``output_dir`` to save one KMeans spatial pattern per
        modality and reference section.
    hipt_boundary_refinement_config : mapping or None, optional
        Optional HIPT-specific boundary correction for Image modality ARI.
        When ``{"enabled": True}``, Image KMeans uses one extra cluster by
        default, identifies the HIPT boundary/background cluster, reassigns it
        to nearby non-boundary clusters, and uses the refined ARI for modality
        selection. Raw and refined ARIs are both stored in the output table.

    Returns
    -------
    ari_df : pandas.DataFrame
        Section-level ARI evaluation results.

    avg_ari : pandas.DataFrame
        Modality-level ARI summary across successful reference sections.

    ranked_modalities : list
        Modalities ranked by average ARI in descending order.
    """
    visualization_config = _normalize_visualization_config(
        visualization_config
    )
    hipt_boundary_config = _normalize_hipt_boundary_refinement_config(
        hipt_boundary_refinement_config,
        visualization_config=visualization_config,
    )

    if pcs_num_dic is None:
        pcs_num_dic = {}

    ari_records = []

    for modality, ref_dic in modality_ref_dic.items():

        pcs_num = pcs_num_dic.get(modality, default_pcs_num)

        if print_results:
            print("\n============================================================")
            print(f"Evaluating modality: {modality}")
            print("============================================================")

        for section, adata in ref_dic.items():

            if label_key not in adata.obs.columns:
                raise ValueError(
                    f"{modality} - {section}: label_key='{label_key}' "
                    "is not found in adata.obs."
                )

            if adata.n_obs < min_spots:
                ari_records.append(
                    {
                        "modality": modality,
                        "section": section,
                        "n_obs": adata.n_obs,
                        "n_features": adata.n_vars,
                        "n_labels": np.nan,
                        "pcs_num": pcs_num,
                        "ari": np.nan,
                        "status": f"skipped: fewer than {min_spots} spots/cells",
                    }
                )
                continue

            labels = adata.obs[label_key].astype(str)

            valid_mask = get_valid_label_mask(
                labels = labels,
                exclude_regions = exclude_regions,
                exclude_mode = exclude_mode
                )

            if valid_mask.sum() < min_spots:
                ari_records.append(
                    {
                        "modality": modality,
                        "section": section,
                        "n_obs": valid_mask.sum(),
                        "n_features": adata.n_vars,
                        "n_labels": np.nan,
                        "pcs_num": pcs_num,
                        "ari": np.nan,
                        "status": "skipped: insufficient valid labels",
                    }
                )
                continue

            # Evaluation only reads the selected rows; plotting makes its own
            # private copy when enabled.
            adata_use = adata[valid_mask.values, :]
            y_true = adata_use.obs[label_key].astype(str).values

            unique_labels = pd.Series(y_true).unique()
            n_clusters = len(unique_labels)
            boundary_refinement_requested = (
                hipt_boundary_config["enabled"]
                and modality == hipt_boundary_config["image_modality"]
            )
            n_clusters_eval = n_clusters + int(
                boundary_refinement_requested
                and hipt_boundary_config["extra_boundary_cluster"]
            )

            if n_clusters < 2:
                ari_records.append(
                    {
                        "modality": modality,
                        "section": section,
                        "n_obs": adata_use.n_obs,
                        "n_features": adata_use.n_vars,
                        "n_labels": n_clusters,
                        "n_clusters": np.nan,
                        "pcs_num": pcs_num,
                        "ari": np.nan,
                        "ari_raw": np.nan,
                        "ari_refined": np.nan,
                        "boundary_refinement_requested": boundary_refinement_requested,
                        "boundary_refinement_applied": False,
                        "boundary_cluster": None,
                        "boundary_final_cluster_key": None,
                        "boundary_refinement_status": "not_evaluated",
                        "status": "skipped: fewer than 2 labels",
                    }
                )
                continue

            if n_clusters_eval >= adata_use.n_obs:
                ari_records.append(
                    {
                        "modality": modality,
                        "section": section,
                        "n_obs": adata_use.n_obs,
                        "n_features": adata_use.n_vars,
                        "n_labels": n_clusters,
                        "n_clusters": n_clusters_eval,
                        "pcs_num": pcs_num,
                        "ari": np.nan,
                        "ari_raw": np.nan,
                        "ari_refined": np.nan,
                        "boundary_refinement_requested": boundary_refinement_requested,
                        "boundary_refinement_applied": False,
                        "boundary_cluster": None,
                        "boundary_final_cluster_key": None,
                        "boundary_refinement_status": "not_evaluated",
                        "status": "skipped: n_clusters >= n_obs",
                    }
                )
                continue

            try:
                modality_pcs = compute_pca_embedding(
                    input_adata=adata_use,
                    pcs_num=pcs_num,
                    random_state=random_state,
                    sample_name=f"{modality} - {section}",
                )

                y_pred = kmeans_clustering(
                    features_matrix=modality_pcs,
                    n_clusters=n_clusters_eval,
                    random_state=random_state,
                )

                ari_raw = adjusted_rand_score(y_true, y_pred)
                ari = ari_raw
                ari_refined = np.nan
                final_labels = y_pred
                boundary_cluster = None
                boundary_final_cluster_key = None
                boundary_refinement_applied = False
                boundary_refinement_status = "not_requested"

                if boundary_refinement_requested:
                    try:
                        refined_labels, boundary_info = (
                            _apply_hipt_boundary_refinement_to_clusters(
                                image_adata=adata_use,
                                cluster_labels=y_pred,
                                cluster_key="modality_eval_cluster",
                                config=hipt_boundary_config,
                                print_results=print_results,
                            )
                        )
                        final_labels = refined_labels
                        ari_refined = adjusted_rand_score(y_true, refined_labels)
                        if hipt_boundary_config["use_refined_ari"]:
                            ari = ari_refined
                        boundary_cluster = boundary_info.get("boundary_cluster")
                        boundary_final_cluster_key = boundary_info.get("final_cluster_key")
                        boundary_refinement_applied = bool(
                            boundary_info.get("boundary_refinement_performed", False)
                        )
                        boundary_refinement_status = (
                            "applied"
                            if boundary_refinement_applied
                            else "no_boundary_detected"
                        )
                    except Exception as boundary_exc:
                        if hipt_boundary_config["strict"]:
                            raise
                        boundary_refinement_status = f"skipped: {boundary_exc}"
                        if print_results:
                            print(
                                f"{section}: HIPT boundary refinement skipped - "
                                f"{boundary_exc}"
                            )

                plot_path = _save_clustering_pattern(
                    input_adata=adata_use,
                    cluster_labels=final_labels,
                    cluster_key="modality_eval_cluster",
                    stage="informative_modalities",
                    group_name=modality,
                    section=section,
                    title=(
                        f"{section}: {modality} KMeans clusters "
                        f"(ARI={ari:.3f})"
                    ),
                    visualization_config=visualization_config,
                    print_results=print_results,
                )

                ari_records.append(
                    {
                        "modality": modality,
                        "section": section,
                        "n_obs": adata_use.n_obs,
                        "n_features": adata_use.n_vars,
                        "n_labels": n_clusters,
                        "n_clusters": n_clusters_eval,
                        "pcs_num": pcs_num,
                        "ari": ari,
                        "ari_raw": ari_raw,
                        "ari_refined": ari_refined,
                        "boundary_refinement_requested": boundary_refinement_requested,
                        "boundary_refinement_applied": boundary_refinement_applied,
                        "boundary_cluster": boundary_cluster,
                        "boundary_final_cluster_key": boundary_final_cluster_key,
                        "boundary_refinement_status": boundary_refinement_status,
                        "plot_path": plot_path,
                        "status": "success",
                    }
                )

                if print_results:
                    print(
                        f"{section}: n_labels={n_clusters}, "
                        f"n_clusters={n_clusters_eval}, "
                        f"ARI={ari:.4f}"
                    )

            except Exception as e:
                ari_records.append(
                    {
                        "modality": modality,
                        "section": section,
                        "n_obs": adata_use.n_obs,
                        "n_features": adata_use.n_vars,
                        "n_labels": n_clusters,
                        "n_clusters": n_clusters_eval,
                        "pcs_num": pcs_num,
                        "ari": np.nan,
                        "ari_raw": np.nan,
                        "ari_refined": np.nan,
                        "boundary_refinement_requested": boundary_refinement_requested,
                        "boundary_refinement_applied": False,
                        "boundary_cluster": None,
                        "boundary_final_cluster_key": None,
                        "boundary_refinement_status": "failed",
                        "status": f"failed: {str(e)}",
                    }
                )

                if print_results:
                    print(f"{section}: failed - {str(e)}")

    ari_df = pd.DataFrame(ari_records)

    success_df = ari_df[ari_df["status"] == "success"].copy()

    if success_df.empty:
        raise ValueError(
            "No modality was successfully evaluated. Please check the input "
            "reference AnnData dictionaries and label_key."
        )

    avg_ari = (
        success_df
        .groupby("modality", as_index=False)
        .agg(
            average_ari=("ari", "mean"),
            average_ari_raw=("ari_raw", "mean"),
            average_ari_refined=("ari_refined", "mean"),
            median_ari=("ari", "median"),
            min_ari=("ari", "min"),
            max_ari=("ari", "max"),
            n_sections=("section", "nunique"),
        )
        .sort_values("average_ari", ascending=False)
        .reset_index(drop=True)
    )

    ranked_modalities = avg_ari["modality"].tolist()

    return ari_df, avg_ari, ranked_modalities


def select_modalities_by_ari(
    avg_ari,
    ranked_modalities,
    included_modalities=None,
    hard_threshold=0.3,
    alpha=0.8,
    selection_criterion="relative",
):
    """
    Select informative modalities based on modality-level ARI summaries.

    This function applies three possible selection rules:
        1. hard threshold,
        2. relative threshold,
        3. overlap between hard and relative thresholds.

    Parameters
    ----------
    avg_ari : pandas.DataFrame
        Modality-level ARI summary. Must contain columns:
        "modality" and "average_ari".

    ranked_modalities : list
        Modalities ranked by average ARI in descending order.

    included_modalities : list or None, optional
        Modalities included in the current evaluation.

    hard_threshold : float, optional
        Minimum average ARI required for the hard-threshold rule.

    alpha : float, optional
        Relative threshold multiplier. A modality is selected if its average
        ARI is at least `alpha * max_average_ari`.

    selection_criterion : {"hard", "relative", "both"}, optional
        Criterion used for the final modality selection.

    Returns
    -------
    selected_modalities : list
        Final selected modalities.

    selected_modalities_hard : list
        Modalities selected by the hard-threshold rule.

    selected_modalities_relative : list
        Modalities selected by the relative-threshold rule.

    selection_info : dict
        Dictionary recording selection parameters and thresholds.
    """

    if included_modalities is None:
        included_modalities = avg_ari["modality"].tolist()

    if selection_criterion not in ["hard", "relative", "both"]:
        raise ValueError(
            "selection_criterion must be one of "
            "{'hard', 'relative', 'both'}."
        )

    if avg_ari.empty:
        raise ValueError("avg_ari is empty. No modalities can be selected.")

    if len(ranked_modalities) == 0:
        raise ValueError("ranked_modalities is empty. No fallback is available.")

    # ------------------------------------------------------------
    # Selection rule 1: hard-threshold rule
    # ------------------------------------------------------------
    selected_modalities_hard = avg_ari.loc[
        avg_ari["average_ari"] >= hard_threshold,
        "modality",
    ].tolist()

    if len(selected_modalities_hard) == 0:
        selected_modalities_hard = ["Gene"]

    # ------------------------------------------------------------
    # Selection rule 2: relative-threshold rule
    # ------------------------------------------------------------
    max_average_ari = avg_ari["average_ari"].max()
    relative_threshold = max_average_ari * alpha

    selected_modalities_relative = avg_ari.loc[
        avg_ari["average_ari"] >= relative_threshold,
        "modality",
    ].tolist()

    # ------------------------------------------------------------
    # Final modality selection
    # ------------------------------------------------------------
    if selection_criterion == "hard":
        selected_modalities = selected_modalities_hard

    elif selection_criterion == "relative":
        selected_modalities = selected_modalities_relative

    elif selection_criterion == "both":
        selected_modalities = [
            modality for modality in selected_modalities_hard
            if modality in selected_modalities_relative
        ]

        if len(selected_modalities) == 0:
            selected_modalities = ["Gene"]

    selection_info = {
        "selection_criterion": selection_criterion,
        "hard_threshold": hard_threshold,
        "alpha": alpha,
        "relative_threshold": relative_threshold,
        "max_average_ari": max_average_ari,
    }

    return (
        selected_modalities,
        selected_modalities_hard,
        selected_modalities_relative,
        selection_info,
    )


def select_informative_modalities(
    included_modalities,
    ref_gene_dic=None,
    ref_image_dic=None,
    ref_protein_dic=None,
    label_key="label",
    hard_threshold=0.3,
    alpha=0.8,
    selection_criterion="both",
    pcs_num_dic=None,
    default_pcs_num=30,
    random_state=0,
    min_spots=10,
    exclude_regions=("nan", "unknown"),
    exclude_mode="exact",
    print_results=True,
    visualization_config=None,
    hipt_boundary_refinement_config=None,
):
    """
    Select informative modalities based on unsupervised clustering agreement
    with reference ground-truth labels.

    For each included modality and each reference section, this function:
        1. computes PCA embeddings from the modality-specific AnnData.X,
        2. runs KMeans clustering with the number of clusters equal to the
           number of true labels,
        3. evaluates clustering agreement using adjusted Rand index, ARI,
        4. averages ARI across reference sections for each modality,
        5. ranks modalities by average ARI,
        6. selects informative modalities using the specified criterion.

    Parameters
    ----------
    included_modalities : list of str
        Modalities included in the target dataset.

        Supported values are:
        - "Gene"
        - "Image"
        - "Protein"

        Example
        -------
        included_modalities = ["Gene", "Image", "Protein"]

    ref_gene_dic : dict or None, optional
        Dictionary of gene-expression reference AnnData objects.

        Example
        -------
        {
            "H1": gene_adata_H1,
            "G2": gene_adata_G2,
            "E1": gene_adata_E1,
        }

    ref_image_dic : dict or None, optional
        Dictionary of image-feature reference AnnData objects.

    ref_protein_dic : dict or None, optional
        Dictionary of protein-feature reference AnnData objects.

    label_key : str, default="label"
        Column in `adata.obs` containing ground-truth tissue labels.

    hard_threshold : float, default=0.2
        Minimum average ARI required for a modality to be selected under the
        hard-threshold rule.

    alpha : float, default=0.8
        Relative threshold factor. Under the relative-threshold rule, modalities
        with average ARI >= max_average_ARI * alpha are selected.

    selection_criterion : {"hard", "relative", "both"}, default="both"
        Criterion used to select informative modalities.

        - "hard":
            Select modalities passing `hard_threshold`.

        - "relative":
            Select modalities with average ARI >= max_average_ARI * alpha.

        - "both":
            Select the overlap between the hard-threshold and relative-threshold
            selected modalities.

    pcs_num_dic : dict or None, optional
        Number of PCs used for each modality.

        Example
        -------
        {
            "Gene": 30,
            "Image": 20,
            "Protein": 10,
        }

        If a modality is not included in `pcs_num_dic`, `default_pcs_num` is used.

    default_pcs_num : int, default=30
        Default number of PCs for modalities not specified in `pcs_num_dic`.

    random_state : int, default=0
        Random seed used for PCA and KMeans.

    min_spots : int, default=10
        Minimum number of spots/cells required in a reference section to evaluate
        a modality.

    print_results : bool, default=True
        Whether to print summary results.

    visualization_config : mapping or None, optional
        Optional settings passed to ``evaluate_modality_ari``. Enable
        ``plot_modality_clusters`` and provide ``output_dir`` to save the
        section-level clustering patterns used for modality ARI evaluation.
    hipt_boundary_refinement_config : mapping or None, optional
        Optional HIPT-specific correction for Image modality ARI. When enabled,
        Image KMeans uses one extra boundary cluster by default; the likely
        HIPT boundary/background cluster is detected, reassigned to nearby
        non-boundary clusters, and the refined ARI is used for modality
        selection.

    Returns
    -------
    results_dic : dict
        Dictionary containing modality evaluation and selection results.

        Keys include:

        - "ari_df":
            Per-section and per-modality ARI results.

        - "avg_ari":
            Average ARI summary for each modality.

        - "ranked_modalities":
            Modalities ranked by decreasing average ARI.

        - "selected_modalities_hard":
            Modalities selected by the hard-threshold rule.

        - "selected_modalities_relative":
            Modalities selected by the relative-threshold rule.

        - "selected_modalities":
            Final selected modalities according to `selection_criterion`.

        - "selection_params":
            Parameters used for modality selection.
    """

    # ------------------------------------------------------------
    # Check inputs
    # ------------------------------------------------------------
    supported_modalities = ["Gene", "Image", "Protein"]

    included_modalities = list(included_modalities)

    invalid_modalities = [
        modality for modality in included_modalities
        if modality not in supported_modalities
    ]

    if len(invalid_modalities) > 0:
        raise ValueError(
            f"Unsupported modalities found: {invalid_modalities}. "
            f"Supported modalities are {supported_modalities}."
        )

    if selection_criterion not in ["hard", "relative", "both"]:
        raise ValueError(
            "selection_criterion must be one of: "
            "'hard', 'relative', or 'both'."
        )

    if pcs_num_dic is None:
        pcs_num_dic = {}

    visualization_config = _normalize_visualization_config(
        visualization_config
    )
    hipt_boundary_refinement_config = _normalize_hipt_boundary_refinement_config(
        hipt_boundary_refinement_config,
        visualization_config=visualization_config,
    )

    modality_ref_dic = {
        "Gene": ref_gene_dic,
        "Image": ref_image_dic,
        "Protein": ref_protein_dic,
    }

    # Keep only included modalities
    modality_ref_dic = {
        modality: modality_ref_dic[modality]
        for modality in included_modalities
    }

    for modality, ref_dic in modality_ref_dic.items():
        if ref_dic is None:
            raise ValueError(
                f"{modality} is included in included_modalities, "
                f"but its reference dictionary is None."
            )

        if not isinstance(ref_dic, dict) or len(ref_dic) == 0:
            raise ValueError(
                f"{modality} reference dictionary must be a non-empty dict."
            )

    # ------------------------------------------------------------
    # Compute ARI for each modality and each reference section
    # ------------------------------------------------------------
    ari_df, avg_ari, ranked_modalities = evaluate_modality_ari(
        modality_ref_dic=modality_ref_dic,
        pcs_num_dic=pcs_num_dic,
        default_pcs_num=default_pcs_num,
        label_key=label_key,
        min_spots=min_spots,
        exclude_regions=exclude_regions,
        exclude_mode=exclude_mode,
        random_state=random_state,
        print_results=print_results,
        visualization_config=visualization_config,
        hipt_boundary_refinement_config=hipt_boundary_refinement_config,
    )

    # ------------------------------------------------------------
    # Modality selection
    # ------------------------------------------------------------
    (
        selected_modalities,
        selected_modalities_hard,
        selected_modalities_relative,
        selection_info,
    ) = select_modalities_by_ari(
        avg_ari=avg_ari,
        ranked_modalities=ranked_modalities,
        included_modalities=included_modalities,
        hard_threshold=hard_threshold,
        alpha=alpha,
        selection_criterion=selection_criterion,
    )

    # ------------------------------------------------------------
    # Store results
    # ------------------------------------------------------------
    results_dic = {
        "ari_df": ari_df,
        "avg_ari": avg_ari,
        "ranked_modalities": ranked_modalities,
        "selected_modalities_hard": selected_modalities_hard,
        "selected_modalities_relative": selected_modalities_relative,
        "selected_modalities": selected_modalities,
        "selection_criterion": selection_criterion,
        "selection_info": selection_info,
        "selection_params": {
            "included_modalities": included_modalities,
            "label_key": label_key,
            "hard_threshold": hard_threshold,
            "alpha": alpha,
            "relative_threshold": selection_info["relative_threshold"],
            "selection_criterion": selection_criterion,
            "pcs_num_dic": pcs_num_dic,
            "default_pcs_num": default_pcs_num,
            "random_state": random_state,
            "min_spots": min_spots,
            "exclude_regions": exclude_regions,
            "exclude_mode": exclude_mode,
            "visualization_config": visualization_config,
            "hipt_boundary_refinement_config": hipt_boundary_refinement_config,
        },
    }

    if print_results:
        print("\n============================================================")
        print("Modality informativeness summary")
        print("============================================================")
        print(avg_ari)

        print("\nRanked modalities:")
        print(ranked_modalities)

        print("\nSelected by hard-threshold rule:")
        print(selected_modalities_hard)

        print("\nSelected by relative-threshold rule:")
        print(selected_modalities_relative)

        print(f"\nFinal selected modalities using criterion='{selection_criterion}':")
        print(selected_modalities)

    return results_dic


#============================================================================
# Part 2. Determine dimension reduction approach (PCA vs. selected features)
#============================================================================
def evaluate_dim_reduction_for_section(
    ref_section: str,
    selected_modalities: Sequence[str],
    dim_reduction_method: str = "pca",
    ref_gene_dic: Optional[Mapping[str, Any]] = None,
    ref_image_dic: Optional[Mapping[str, Any]] = None,
    ref_protein_dic: Optional[Mapping[str, Any]] = None,
    features_dic: Optional[Mapping[str, Any]] = None,
    features_format: str = "section",
    pcs_num_dic: Optional[Mapping[str, int]] = None,
    default_pcs_num: int = 30,
    label_key: str = "label",
    exclude_regions: Sequence[str] = ("nan", "unknown"),
    exclude_mode: str = "contains",
    scale_embedding: bool = True,
    random_state: int = 0,
    align_by_obs_names: bool = False,
    visualization_config: Optional[Mapping[str, Any]] = None,
    print_results: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate one dimension-reduction method for one reference section.

    The logic is unchanged from the original implementation:

    1. get true labels from the reference Gene AnnData;
    2. exclude invalid labels;
    3. set ``n_clusters`` to the number of unique valid labels;
    4. integrate selected modalities using PCA or selected features;
    5. run KMeans on the valid spots;
    6. compute ARI against the true labels.

    The only organizational update is that modality integration and clustering
    are now delegated to shared helpers.

    ``visualization_config`` may enable ``plot_dim_reduction_clusters`` to
    save the evaluated KMeans pattern for this section and method.
    """

    visualization_config = _normalize_visualization_config(
        visualization_config
    )

    label_adata = get_ref_modality_adata(
        ref_section=ref_section,
        modality="Gene",
        ref_gene_dic=ref_gene_dic,
        ref_image_dic=ref_image_dic,
        ref_protein_dic=ref_protein_dic,
    )

    if label_key not in label_adata.obs.columns:
        raise KeyError(
            f"{ref_section}: label_key={label_key!r} is not found in adata.obs."
        )

    labels = label_adata.obs[label_key].astype(str)

    valid_mask = get_valid_label_mask(
        labels=labels,
        exclude_regions=exclude_regions,
        exclude_mode=exclude_mode,
    )

    true_labels = labels.loc[valid_mask].to_numpy()
    n_clusters = len(np.unique(true_labels))

    if n_clusters < 2:
        raise ValueError(
            f"{ref_section}: at least two unique labels are required for ARI."
        )

    integrated_embedding, modality_embedding_dic = integrate_modalities_for_section(
        ref_section=ref_section,
        selected_modalities=selected_modalities,
        dim_reduction_method=dim_reduction_method,
        ref_gene_dic=ref_gene_dic,
        ref_image_dic=ref_image_dic,
        ref_protein_dic=ref_protein_dic,
        features_dic=features_dic,
        features_format=features_format,
        pcs_num_dic=pcs_num_dic,
        default_pcs_num=default_pcs_num,
        scale_embedding=scale_embedding,
        random_state=random_state,
        align_by_obs_names=align_by_obs_names,
    )

    if integrated_embedding.shape[0] != label_adata.n_obs:
        raise ValueError(
            f"{ref_section}: integrated embedding has "
            f"{integrated_embedding.shape[0]} observations, but label AnnData has "
            f"{label_adata.n_obs} observations. Please check row alignment."
        )

    integrated_embedding_eval = integrated_embedding[valid_mask.to_numpy(), :]

    pred_labels, cluster_info = cluster_integrated_embedding(
        integrated_embedding=integrated_embedding_eval,
        clustering_config={
            "clustering_method": "kmeans",
            "n_clusters": n_clusters,
            "random_state": random_state,
        },
        cluster_key="dim_reduction_eval_cluster",
    )

    ari = adjusted_rand_score(true_labels, pred_labels)

    plot_path = _save_clustering_pattern(
        input_adata=label_adata[valid_mask.to_numpy(), :],
        cluster_labels=pred_labels,
        cluster_key="dim_reduction_eval_cluster",
        stage="dimension_reduction",
        group_name=dim_reduction_method,
        section=ref_section,
        title=(
            f"{ref_section}: {dim_reduction_method} KMeans clusters "
            f"(ARI={ari:.3f})"
        ),
        visualization_config=visualization_config,
        print_results=print_results,
    )

    result = {
        "ref_section": ref_section,
        "dim_reduction_method": dim_reduction_method,
        "features_format": features_format,
        "selected_modalities": list(selected_modalities),
        "n_clusters": n_clusters,
        "n_valid_obs": int(valid_mask.sum()),
        "n_integrated_features": int(integrated_embedding.shape[1]),
        "ari": ari,
        "cluster_info": cluster_info,
        "plot_path": plot_path,
    }

    return result


def evaluate_dim_reduction_method(
    ref_section_list: Sequence[str],
    selected_modalities: Sequence[str],
    dim_reduction_method: str = "pca",
    ref_gene_dic: Optional[Mapping[str, Any]] = None,
    ref_image_dic: Optional[Mapping[str, Any]] = None,
    ref_protein_dic: Optional[Mapping[str, Any]] = None,
    features_dic: Optional[Mapping[str, Any]] = None,
    features_format: str = "section",
    pcs_num_dic: Optional[Mapping[str, int]] = None,
    default_pcs_num: int = 30,
    label_key: str = "label",
    exclude_regions: Sequence[str] = ("nan", "unknown"),
    exclude_mode: str = "contains",
    scale_embedding: bool = True,
    random_state: int = 0,
    align_by_obs_names: bool = False,
    print_results: bool = True,
    visualization_config: Optional[Mapping[str, Any]] = None,
) -> Tuple[pd.DataFrame, float]:

    """
    Evaluate one dimension-reduction method across reference sections.

    Returns
    -------
    ari_df : pandas.DataFrame
        Section-level ARI table.

    average_ari : float
        Average ARI across reference sections.

    Notes
    -----
    When ``visualization_config['plot_dim_reduction_clusters']`` is true, one
    spatial clustering plot is saved for every successfully evaluated section.
    """

    visualization_config = _normalize_visualization_config(
        visualization_config
    )

    result_list = []

    for ref_section in ref_section_list:
        result = evaluate_dim_reduction_for_section(
            ref_section=ref_section,
            selected_modalities=selected_modalities,
            dim_reduction_method=dim_reduction_method,
            ref_gene_dic=ref_gene_dic,
            ref_image_dic=ref_image_dic,
            ref_protein_dic=ref_protein_dic,
            features_dic=features_dic,
            features_format=features_format,
            pcs_num_dic=pcs_num_dic,
            default_pcs_num=default_pcs_num,
            label_key=label_key,
            exclude_regions=exclude_regions,
            exclude_mode=exclude_mode,
            scale_embedding=scale_embedding,
            random_state=random_state,
            align_by_obs_names=align_by_obs_names,
            visualization_config=visualization_config,
            print_results=print_results,
        )

        result_list.append(result)

        if print_results:
            print(
                f"{ref_section} | {dim_reduction_method} | "
                f"ARI = {result['ari']:.4f} | "
                f"n_clusters = {result['n_clusters']} | "
                f"n_features = {result['n_integrated_features']}"
            )

    ari_df = pd.DataFrame(result_list)
    average_ari = ari_df["ari"].mean()

    if print_results:
        print(
            f"\nAverage ARI for {dim_reduction_method}: "
            f"{average_ari:.4f}"
        )

    return ari_df, average_ari


def determine_dimension_reduction_method(
    ref_section_list: Sequence[str],
    selected_modalities: Sequence[str],
    ref_gene_dic: Optional[Mapping[str, Any]] = None,
    ref_image_dic: Optional[Mapping[str, Any]] = None,
    ref_protein_dic: Optional[Mapping[str, Any]] = None,
    features_dic: Optional[Mapping[str, Any]] = None,
    features_format: str = "section",
    pcs_num_dic: Optional[Mapping[str, int]] = None,
    default_pcs_num: int = 30,
    candidate_methods: Sequence[str] = SUPPORTED_REDUCTION_METHODS,
    label_key: str = "label",
    exclude_regions: Sequence[str] = ("nan", "unknown"),
    exclude_mode: str = "contains",
    scale_embedding: bool = True,
    random_state: int = 0,
    align_by_obs_names: bool = False,
    print_results: bool = True,
    visualization_config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compare candidate dimension-reduction methods and select the best method
    based on average ARI across reference sections.

    For each candidate method, this function integrates the selected modalities
    for every reference section, clusters the integrated embedding using KMeans,
    compares the clusters with reference labels using adjusted Rand index (ARI),
    and selects the method with the highest average ARI.

    Parameters
    ----------
    ref_section_list : sequence of str
        Reference section names used for method evaluation. Each section should
        be present in the required modality-specific reference dictionaries.

    selected_modalities : sequence of str
        Modalities to include in the integrated embedding, such as
        ``["Gene"]``, ``["Gene", "Image"]``, or
        ``["Gene", "Image", "Protein"]``.

    ref_gene_dic : mapping or None, default=None
        Dictionary of gene-expression reference AnnData objects, keyed by
        section name. Required when ``"Gene"`` is included in
        ``selected_modalities``. Gene AnnData is also used to retrieve
        reference labels.

    ref_image_dic : mapping or None, default=None
        Dictionary of image-feature reference AnnData objects, keyed by
        section name. Required when ``"Image"`` is included in
        ``selected_modalities``.

    ref_protein_dic : mapping or None, default=None
        Dictionary of protein-feature reference AnnData objects, keyed by
        section name. Required when ``"Protein"`` is included in
        ``selected_modalities``.

    features_dic : mapping or None, default=None
        Selected feature dictionary used when evaluating
        ``dim_reduction_method="selected_features"``. If
        ``"selected_features"`` is included in ``candidate_methods``,
        this argument must be provided.

        If ``features_format="section"``, the expected structure is::

            features_dic[section][modality] = feature_list

        If ``features_format="modality"``, the expected structure is::

            features_dic[modality] = feature_list

    features_format : {"section", "modality"}, default="section"
        Format of ``features_dic``. Use ``"section"`` for section-specific
        feature lists and ``"modality"`` for one shared feature list per
        modality.

    pcs_num_dic : mapping or None, default=None
        Number of principal components to use for each modality when evaluating
        ``dim_reduction_method="pca"``. If a modality is not found in this
        dictionary, ``default_pcs_num`` is used.

    default_pcs_num : int, default=30
        Default number of PCs used for modalities not specified in
        ``pcs_num_dic``.

    candidate_methods : sequence of str, default=SUPPORTED_REDUCTION_METHODS
        Dimension-reduction methods to compare. Supported values are
        ``"pca"`` and ``"selected_features"``.

    label_key : str, default="label"
        Column in ``adata.obs`` containing reference labels used for ARI
        evaluation.

    exclude_regions : sequence of str, default=("nan", "unknown")
        Labels to exclude before clustering evaluation.

    exclude_mode : {"exact", "contains"}, default="contains"
        Rule used to remove labels in ``exclude_regions``. Use ``"exact"``
        for exact label matching and ``"contains"`` for substring matching.

    scale_embedding : bool, default=True
        Whether to standardize each modality embedding before concatenating
        modalities.

    random_state : int, default=0
        Random seed used for PCA, KMeans, and other stochastic steps.

    align_by_obs_names : bool, default=False
        Whether to align modality AnnData objects by shared ``obs_names`` before
        integration. If False, all modalities are assumed to have the same spot
        order.

    print_results : bool, default=True
        Whether to print section-level ARI values and the final method summary.

    visualization_config : mapping or None, optional
        Shared plotting settings. Enable ``plot_dim_reduction_clusters`` and
        provide ``output_dir`` to save one spatial clustering pattern for each
        candidate method and reference section.

    Returns
    -------
    result_dic : dict
        Dictionary containing the selected dimension-reduction method and
        evaluation results.

        Keys include:

        ``"best_method"``
            Dimension-reduction method with the highest average ARI.

        ``"summary_df"``
            DataFrame summarizing each candidate method, including average ARI,
            number of evaluated sections, and evaluation status.

        ``"ari_df_dic"``
            Dictionary mapping each candidate method to its section-level ARI
            DataFrame.

        ``"average_ari_dic"``
            Dictionary mapping each candidate method to its average ARI.

        ``"features_format"``
            Feature dictionary format used during evaluation.

        ``"align_by_obs_names"``
            Whether modalities were aligned by ``obs_names`` before integration.
    """

    visualization_config = _normalize_visualization_config(
        visualization_config
    )

    candidate_methods = list(candidate_methods)

    if len(candidate_methods) == 0:
        raise ValueError("candidate_methods must contain at least one method.")

    invalid_methods = [
        method for method in candidate_methods
        if method not in SUPPORTED_REDUCTION_METHODS
    ]
    if len(invalid_methods) > 0:
        raise ValueError(
            f"Unsupported dimension-reduction methods: {invalid_methods}. "
            f"Supported methods are {SUPPORTED_REDUCTION_METHODS}."
        )

    ari_df_dic = {}
    average_ari_dic = {}
    summary_records = []

    for method_order, method in enumerate(candidate_methods):
        if method == "selected_features" and features_dic is None:
            ari_df_dic[method] = pd.DataFrame()
            average_ari_dic[method] = np.nan
            summary_records.append(
                {
                    "dim_reduction_method": method,
                    "average_ari": np.nan,
                    "n_sections": 0,
                    "method_order": method_order,
                    "status": "skipped: features_dic is required",
                }
            )
            if print_results:
                print(
                    "\nSkipping selected_features because features_dic is None."
                )
            continue

        try:
            ari_df, average_ari = evaluate_dim_reduction_method(
                ref_section_list=ref_section_list,
                selected_modalities=selected_modalities,
                dim_reduction_method=method,
                ref_gene_dic=ref_gene_dic,
                ref_image_dic=ref_image_dic,
                ref_protein_dic=ref_protein_dic,
                features_dic=features_dic,
                features_format=features_format,
                pcs_num_dic=pcs_num_dic,
                default_pcs_num=default_pcs_num,
                label_key=label_key,
                exclude_regions=exclude_regions,
                exclude_mode=exclude_mode,
                scale_embedding=scale_embedding,
                random_state=random_state,
                align_by_obs_names=align_by_obs_names,
                print_results=print_results,
                visualization_config=visualization_config,
            )
            status = "success"

        except Exception as exc:
            ari_df = pd.DataFrame()
            average_ari = np.nan
            status = f"failed: {exc}"

            if print_results:
                print(f"\n{method} failed: {exc}")

        ari_df_dic[method] = ari_df
        average_ari_dic[method] = average_ari
        summary_records.append(
            {
                "dim_reduction_method": method,
                "average_ari": average_ari,
                "n_sections": 0 if ari_df.empty else int(ari_df.shape[0]),
                "method_order": method_order,
                "status": status,
            }
        )

    summary_df = pd.DataFrame(summary_records)
    success_df = summary_df[np.isfinite(summary_df["average_ari"])].copy()

    if success_df.empty:
        raise ValueError(
            "No dimension-reduction method was successfully evaluated. "
            "Please check selected modalities, features_dic, labels, and row alignment."
        )

    success_df = success_df.sort_values(
        ["average_ari", "method_order"],
        ascending=[False, True],
    )
    best_method = str(success_df.iloc[0]["dim_reduction_method"])

    summary_df = summary_df.sort_values(
        ["average_ari", "method_order"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)

    if print_results:
        print("\n============================================================")
        print("Dimension-reduction method summary")
        print("============================================================")
        print(summary_df.drop(columns=["method_order"]))
        print(f"\nSelected dimension-reduction method: {best_method}")

    return {
        "best_method": best_method,
        "summary_df": summary_df.drop(columns=["method_order"]),
        "ari_df_dic": ari_df_dic,
        "average_ari_dic": average_ari_dic,
        "features_format": features_format,
        "align_by_obs_names": align_by_obs_names,
        "visualization_config": visualization_config,
    }


# This step determines the embedding configuration only:
#   1. selected modalities
#   2. pca vs selected features
# The final clustering method and clustering parameters are specified afterward.
def determine_multi_modal_embedding_config(
    included_modalities,
    ref_section_list,
    ref_gene_dic=None,
    ref_image_dic=None,
    ref_protein_dic=None,
    features_dic=None,
    features_format="section",
    label_key="label",
    hard_threshold=0.3,
    alpha=0.8,
    selection_criterion="both",
    pcs_num_dic=None,
    default_pcs_num=30,
    candidate_methods=("pca", "selected_features"),
    min_spots=10,
    exclude_regions=("nan", "unknown"),
    modality_exclude_mode="exact",
    dim_reduction_exclude_mode="exact",
    scale_embedding=True,
    random_state=0,
    align_by_obs_names=False,
    print_results=True,
    visualization_config=None,
    hipt_boundary_refinement_config=None,
) -> MultiModalClusteringConfigResult:
    """
    Determine the multi-modal embedding configuration from reference sections.

    This pipeline makes two automatic decisions before query clustering:

    1. Select informative modalities using reference-label ARI.
    2. Select the better dimension-reduction method, either PCA or selected features.

    The final clustering method is not selected here. After this function returns,
    users should call ``config_result.to_clustering_config(...)`` and specify
    ``clustering_method="kmeans"`` or ``clustering_method="leiden"`` together
    with the corresponding clustering parameters.

    Parameters
    ----------
    included_modalities : list of str
        Modalities to evaluate and potentially include in the final embedding.
        Supported values are ``"Gene"``, ``"Image"``, and ``"Protein"``.

    ref_section_list : list of str
        Reference section names used to compare dimension-reduction methods.
        Each section should be present in the corresponding reference AnnData
        dictionaries.

    ref_gene_dic : dict or None, default=None
        Dictionary of gene-expression reference AnnData objects, keyed by
        section name.

    ref_image_dic : dict or None, default=None
        Dictionary of image-feature reference AnnData objects, keyed by
        section name. Required when ``"Image"`` is included in
        ``included_modalities``.

    ref_protein_dic : dict or None, default=None
        Dictionary of protein-feature reference AnnData objects, keyed by
        section name. Required when ``"Protein"`` is included in
        ``included_modalities``.

    features_dic : dict or None, default=None
        Selected feature dictionary used when evaluating
        ``dim_reduction_method="selected_features"``. Required if
        ``candidate_methods`` contains ``"selected_features"``.

        If ``features_format="section"``, the expected format is::

            features_dic[section][modality] = feature_list

        If ``features_format="modality"``, the expected format is::

            features_dic[modality] = feature_list

    features_format : {"section", "modality"}, default="section"
        Format of ``features_dic``. Use ``"section"`` when selected features
        are section-specific. Use ``"modality"`` when one shared feature list is
        used per modality.

    label_key : str, default="label"
        Column in ``adata.obs`` containing reference tissue labels.

    hard_threshold : float, default=0.3
        Minimum average modality ARI required for the hard-threshold selection
        rule.

    alpha : float, default=0.8
        Relative threshold factor. A modality passes the relative rule if its
        average ARI is at least ``alpha * max_average_ARI``.

    selection_criterion : {"hard", "relative", "both"}, default="both"
        Rule used to select final informative modalities.

        - ``"hard"``: keep modalities passing ``hard_threshold``.
        - ``"relative"``: keep modalities close to the best modality.
        - ``"both"``: keep modalities passing both rules.

    pcs_num_dic : dict or None, default=None
        Modality-specific number of PCs used for PCA embeddings.

        Example::

            {
                "Gene": 30,
                "Image": 20,
                "Protein": 10,
            }

    default_pcs_num : int, default=30
        Default number of PCs for modalities not specified in ``pcs_num_dic``.

    candidate_methods : sequence of str, default=("pca", "selected_features")
        Dimension-reduction methods to compare. Supported values are
        ``"pca"`` and ``"selected_features"``.

    min_spots : int, default=10
        Minimum number of valid spots/cells required for modality ARI
        evaluation.

    exclude_regions : sequence of str, default=("nan", "unknown")
        Labels excluded from ARI evaluation.

    modality_exclude_mode : {"exact", "contains"}, default="exact"
        Label exclusion mode used during modality informativeness evaluation.

    dim_reduction_exclude_mode : {"exact", "contains"}, default="contains"
        Label exclusion mode used during dimension-reduction method evaluation.

    scale_embedding : bool, default=True
        Whether to scale each modality embedding before concatenating
        modalities.

    random_state : int, default=0
        Random seed used for PCA, KMeans, and other stochastic steps.

    align_by_obs_names : bool, default=False
        Whether to align modalities by shared ``obs_names`` before integration.
        If False, modalities are assumed to already have matched observation
        order.

    print_results : bool, default=True
        Whether to print progress and summary tables.

    visualization_config : mapping or None, optional
        Optional categorical spatial-plot settings. Supported keys are:

        - ``plot_modality_clusters``: plot each modality/section evaluation.
        - ``plot_dim_reduction_clusters``: plot each method/section evaluation.
        - ``output_dir``: root output directory; required if either plot flag
          is true.
        - ``x_key``, ``y_key``: coordinate columns, default ``"x"`` and
          ``"y"``.
        - ``cat_color``: categorical palette passed to ``cat_figure``.
        - ``size``, ``dpi``, ``invert_x``, ``invert_y``: plot settings.
        Example:
        visualization_config={
            "plot_modality_clusters": True,
            "plot_dim_reduction_clusters": True,
            "output_dir": "results/clustering_plots",
            "x_key": "pixel_x",
            "y_key": "pixel_y",
            "cat_color": [
                "#E64B35",
                "#4DBBD5",
                "#00A087",
                "#3C5488",
            ],
            "size": 80,
            "dpi": 300,
            "invert_x": False,
            "invert_y": True,

        Plots are saved below ``output_dir/informative_modalities`` and
        ``output_dir/dimension_reduction``. The configured coordinate columns
        must be present in the relevant reference AnnData ``.obs``. Plots show
        the same valid-label subset used for ARI evaluation. If a figure cannot
        be generated, ARI evaluation continues and its ``plot_path`` is null.

        results/clustering_plots/
        ├── informative_modalities/
        │   ├── Gene/
        │   │   └── <section>_clusters.png
        │   ├── Image/
        │   │   └── <section>_clusters.png
        │   └── Protein/
        │       └── <section>_clusters.png
        └── dimension_reduction/
            ├── pca/
            │   └── <section>_clusters.png
            └── selected_features/
                └── <section>_clusters.png

    hipt_boundary_refinement_config : mapping or None, optional
        Optional HIPT-specific correction used only during Image-modality ARI
        evaluation. Set ``{"enabled": True}`` to cluster Image features with
        one extra candidate boundary cluster by default, reassign the likely
        boundary/background cluster to nearby non-boundary clusters, and use
        the refined ARI for modality selection. Raw and refined ARIs are both
        stored in ``config_result.modality_ari_df``.

    Returns
    -------
    config_result : MultiModalClusteringConfigResult
        Dataclass storing the automatically determined embedding configuration.

        Main fields include:

        ``config_result.selected_modalities``
            Final selected informative modalities.

        ``config_result.modality_avg_ari``
            Average ARI summary for each evaluated modality.

        ``config_result.ranked_modalities``
            Modalities ranked by average ARI.

        ``config_result.dim_reduction_method``
            Selected dimension-reduction method, either ``"pca"`` or
            ``"selected_features"``.

        ``config_result.dim_reduction_summary_df``
            Average ARI comparison across candidate dimension-reduction methods.

        ``config_result.modality_ari_df["plot_path"]``
            Saved modality-clustering figure paths when visualization is enabled.

        ``config_result.dim_reduction_ari_df_dic[method]["plot_path"]``
            Saved figure paths for each evaluated dimension-reduction method.

        ``config_result.to_clustering_config(...)``
            Helper method for constructing the final clustering configuration
            after the user specifies KMeans or Leiden parameters.

    Examples
    --------
    First determine the embedding configuration:

    >>> config_result = determine_multi_modal_embedding_config(
    ...     included_modalities=["Gene", "Image", "Protein"],
    ...     ref_section_list=ref_section_list,
    ...     ref_gene_dic=ref_gene_dic,
    ...     ref_image_dic=ref_image_dic,
    ...     ref_protein_dic=ref_protein_dic,
    ...     features_dic=features_dic,
    ...     features_format="section",
    ...     label_key="label",
    ...     visualization_config={
    ...         "plot_modality_clusters": True,
    ...         "plot_dim_reduction_clusters": True,
    ...         "output_dir": "clustering_config_plots",
    ...         "x_key": "pixel_x",
    ...         "y_key": "pixel_y",
    ...     },
    ... )

    Then specify Leiden clustering parameters:

    >>> clustering_config = config_result.to_clustering_config(
    ...     clustering_method="leiden",
    ...     resolution=0.5,
    ...     n_neighbors=15,
    ... )

    Or specify KMeans clustering parameters:

    >>> clustering_config = config_result.to_clustering_config(
    ...     clustering_method="kmeans",
    ...     n_clusters=6,
    ... )

    Notes
    -----
    This function does not choose ``clustering_method`` automatically.

    The automatically determined part is:

    - ``selected_modalities``
    - ``dim_reduction_method``

    The user-specified clustering part is:

    - for KMeans: ``n_clusters``
    - for Leiden: ``resolution`` and ``n_neighbors``
    """

    visualization_config = _normalize_visualization_config(
        visualization_config
    )

    # ------------------------------------------------------------
    # Step 1. Select informative modalities
    # ------------------------------------------------------------
    modality_result = select_informative_modalities(
        included_modalities=included_modalities,
        ref_gene_dic=ref_gene_dic,
        ref_image_dic=ref_image_dic,
        ref_protein_dic=ref_protein_dic,
        label_key=label_key,
        hard_threshold=hard_threshold,
        alpha=alpha,
        selection_criterion=selection_criterion,
        pcs_num_dic=pcs_num_dic,
        default_pcs_num=default_pcs_num,
        random_state=random_state,
        min_spots=min_spots,
        exclude_regions=exclude_regions,
        exclude_mode=modality_exclude_mode,
        print_results=print_results,
        visualization_config=visualization_config,
        hipt_boundary_refinement_config=hipt_boundary_refinement_config,
    )

    selected_modalities = modality_result["selected_modalities"]

    # ------------------------------------------------------------
    # Step 2. Determine best dimension-reduction method
    # ------------------------------------------------------------
    dim_reduction_result = determine_dimension_reduction_method(
        ref_section_list=ref_section_list,
        selected_modalities=selected_modalities,
        ref_gene_dic=ref_gene_dic,
        ref_image_dic=ref_image_dic,
        ref_protein_dic=ref_protein_dic,
        features_dic=features_dic,
        features_format=features_format,
        pcs_num_dic=pcs_num_dic,
        default_pcs_num=default_pcs_num,
        candidate_methods=candidate_methods,
        label_key=label_key,
        exclude_regions=exclude_regions,
        exclude_mode=dim_reduction_exclude_mode,
        scale_embedding=scale_embedding,
        random_state=random_state,
        align_by_obs_names=align_by_obs_names,
        print_results=print_results,
        visualization_config=visualization_config,
    )

    # ------------------------------------------------------------
    # Step 3. Store both decisions in a dataclass
    # ------------------------------------------------------------
    config_result = MultiModalClusteringConfigResult.from_result_dics(
        modality_result=modality_result,
        dim_reduction_result=dim_reduction_result,
        scale_embedding=scale_embedding,
        pcs_num_dic=pcs_num_dic,
        default_pcs_num=default_pcs_num,
        random_state=random_state,
    )

    return config_result
