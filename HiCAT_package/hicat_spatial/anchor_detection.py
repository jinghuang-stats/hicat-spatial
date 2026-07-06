import time
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

@dataclass
class AnchorDetectionResult:
    """
    Structured output for anchor detection.

    The object keeps only the generated, index-aligned anchor column plus a small 
    registry describing those columns, The caller remains the sole owner of the 
    complete AnnData object. 

    Column registry
    ---------------
    final_anchor_keys
        Final columns after all requested aggregation steps.
        Structure: ``{node: anchor_column}``.

    modality_anchor_keys
        Quantile-based modality-level columns after per-modality reference
        aggregation. Structure: ``{modality: {node: anchor_column}}``.

    section_anchor_keys
        NN-based reference-section-level columns after modality aggregation
        within each section. Structure: ``{ref_section: {node: anchor_column}}``.

    section_modality_anchor_keys
        NN-based lowest-level columns before aggregation.
        Structure: ``{ref_section: {modality: {node: anchor_column}}}``.

    anchor_thres_dic
        Optional quantile-based hierarchy-level anchor-count thresholds.
        Recommended structure: ``{modality: {node: threshold}}``.

    params
        Lightweight parameter metadata. Avoid storing full AnnData objects here.
    """

    anchor_df: pd.DataFrame
    method: str
    target_node: str
    nontgt_node: str

    final_anchor_keys: Dict[str, str] = field(default_factory=dict)
    modality_anchor_keys: Dict[str, Dict[str, str]] = field(default_factory=dict)
    section_anchor_keys: Dict[str, Dict[str, str]] = field(default_factory=dict)
    section_modality_anchor_keys: Dict[str, Dict[str, Dict[str, str]]] = field(
        default_factory=dict
    )
    anchor_thres_dic: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.anchor_df, pd.DataFrame):
            raise TypeError("anchor_df must be a pandas DataFrame.")
        if not self.anchor_df.index.is_unique:
            raise ValueError("anchor_df.index must contain unique obs_names.")

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_nn_single_ref(
        cls,
        adata,
        target_node: str,
        nontgt_node: str,
        ref_section: str,
        modalities: Sequence[str],
        params: Optional[Dict[str, Any]] = None,
        validate: bool = True,
    ) -> "AnchorDetectionResult":
        """Build result object for NN-based single-reference multimodal anchors."""
        modalities = list(modalities)

        section_anchor_keys = {
            ref_section: {
                target_node: f"{ref_section}_{target_node}_anchors",
                nontgt_node: f"{ref_section}_{nontgt_node}_anchors",
            }
        }

        section_modality_anchor_keys = {
            ref_section: {
                modality: {
                    target_node: f"{ref_section}_{modality}_{target_node}_anchors",
                    nontgt_node: f"{ref_section}_{modality}_{nontgt_node}_anchors",
                }
                for modality in modalities
            }
        }

        # For a single reference section, the section-level aggregate columns are
        # also the final columns.
        final_anchor_keys = dict(section_anchor_keys[ref_section])

        result = cls(
            anchor_df=pd.DataFrame(index=adata.obs_names.copy()),
            method="nn_single_ref_multimodal",
            target_node=target_node,
            nontgt_node=nontgt_node,
            final_anchor_keys=final_anchor_keys,
            section_anchor_keys=section_anchor_keys,
            section_modality_anchor_keys=section_modality_anchor_keys,
            params=params or {},
        )

        result.anchor_df = adata.obs.reindex(columns=result.anchor_columns).copy()

        if validate:
            result.validate_columns()

        return result

    @classmethod
    def from_nn_multiref(
        cls,
        adata,
        target_node: str,
        nontgt_node: str,
        ref_section_list: Sequence[str],
        modalities: Sequence[str],
        params: Optional[Dict[str, Any]] = None,
        validate: bool = True,
    ) -> "AnchorDetectionResult":
        """Build result object for NN-based multi-reference multimodal anchors."""
        ref_section_list = list(ref_section_list)
        modalities = list(modalities)

        final_anchor_keys = {
            target_node: f"{target_node}_anchors",
            nontgt_node: f"{nontgt_node}_anchors",
        }

        section_anchor_keys = {
            ref_section: {
                target_node: f"{ref_section}_{target_node}_anchors",
                nontgt_node: f"{ref_section}_{nontgt_node}_anchors",
            }
            for ref_section in ref_section_list
        }

        section_modality_anchor_keys = {
            ref_section: {
                modality: {
                    target_node: f"{ref_section}_{modality}_{target_node}_anchors",
                    nontgt_node: f"{ref_section}_{modality}_{nontgt_node}_anchors",
                }
                for modality in modalities
            }
            for ref_section in ref_section_list
        }

        result = cls(
            anchor_df=pd.DataFrame(index=adata.obs_names.copy()),
            method="nn_multiref_multimodal",
            target_node=target_node,
            nontgt_node=nontgt_node,
            final_anchor_keys=final_anchor_keys,
            section_anchor_keys=section_anchor_keys,
            section_modality_anchor_keys=section_modality_anchor_keys,
            params=params or {},
        )

        result.anchor_df = adata.obs.reindex(columns=result.anchor_columns).copy()

        if validate:
            result.validate_columns()

        return result

    @classmethod
    def from_quantile_multimodal(
        cls,
        adata,
        target_node: str,
        nontgt_node: str,
        modalities: Sequence[str],
        anchor_thres_dic: Optional[Dict[str, Dict[str, Any]]] = None,
        params: Optional[Dict[str, Any]] = None,
        validate: bool = True,
    ) -> "AnchorDetectionResult":
        """Build result object for quantile-based multimodal anchors."""
        modalities = list(modalities)

        final_anchor_keys = {
            target_node: f"{target_node}_anchors",
            nontgt_node: f"{nontgt_node}_anchors",
        }

        modality_anchor_keys = {
            modality: {
                target_node: f"{modality}_{target_node}_anchors",
                nontgt_node: f"{modality}_{nontgt_node}_anchors",
            }
            for modality in modalities
        }

        result = cls(
            anchor_df=pd.DataFrame(index=adata.obs_names.copy()),
            method="quantile_multimodal",
            target_node=target_node,
            nontgt_node=nontgt_node,
            final_anchor_keys=final_anchor_keys,
            modality_anchor_keys=modality_anchor_keys,
            anchor_thres_dic=anchor_thres_dic or {},
            params=params or {},
        )

        result.anchor_df = adata.obs.reindex(columns=result.anchor_columns).copy()

        if validate:
            result.validate_columns()

        return result

    # ------------------------------------------------------------------
    # Basic accessors
    # ------------------------------------------------------------------
    @property
    def nodes(self) -> List[str]:
        """Return the two binary nodes in target/non-target order."""
        return [self.target_node, self.nontgt_node]

    @property
    def ref_sections(self) -> List[str]:
        """Return registered NN reference sections."""
        return list(self.section_anchor_keys.keys())

    @property
    def modalities(self) -> List[str]:
        """Return registered modalities, preserving first-seen order."""
        modalities: List[str] = []

        def add_modality(modality: str) -> None:
            if modality not in modalities:
                modalities.append(modality)

        for modality in self.modality_anchor_keys.keys():
            add_modality(modality)

        for section_dic in self.section_modality_anchor_keys.values():
            for modality in section_dic.keys():
                add_modality(modality)

        return modalities

    @property
    def anchor_columns(self) -> List[str]:
        """Return all registered anchor columns, preserving output-level order."""
        columns: List[str] = []

        def add_col(col: str) -> None:
            if col not in columns:
                columns.append(col)

        # Highest-level output first.
        for col in self.final_anchor_keys.values():
            add_col(col)

        # Quantile modality-level output.
        for modality_dic in self.modality_anchor_keys.values():
            for col in modality_dic.values():
                add_col(col)

        # NN section-level output.
        for section_dic in self.section_anchor_keys.values():
            for col in section_dic.values():
                add_col(col)

        # NN section-modality-level output.
        for section_dic in self.section_modality_anchor_keys.values():
            for modality_dic in section_dic.values():
                for col in modality_dic.values():
                    add_col(col)

        return columns

    def get_anchor_key(
        self,
        node: str,
        modality: Optional[str] = None,
        ref_section: Optional[str] = None,
    ) -> str:
        """
        Retrieve an anchor column name.

        NN results
        ----------
        - ``node`` only: final cross-reference column, if present.
        - ``node + ref_section``: section-level column.
        - ``node + ref_section + modality``: section-modality-level column.

        Quantile results
        ----------------
        - ``node`` only: final cross-modality column.
        - ``node + modality``: modality-level column.
        """
        if node not in self.nodes:
            raise KeyError(
                f"node={node!r} is not one of the registered nodes: {self.nodes}."
            )

        if ref_section is not None and modality is not None:
            try:
                return self.section_modality_anchor_keys[ref_section][modality][node]
            except KeyError as exc:
                raise KeyError(
                    "No section-modality anchor key found for "
                    f"ref_section={ref_section!r}, modality={modality!r}, node={node!r}."
                ) from exc

        if ref_section is not None:
            try:
                return self.section_anchor_keys[ref_section][node]
            except KeyError as exc:
                raise KeyError(
                    "No section-level anchor key found for "
                    f"ref_section={ref_section!r}, node={node!r}."
                ) from exc

        if modality is not None:
            try:
                return self.modality_anchor_keys[modality][node]
            except KeyError as exc:
                raise KeyError(
                    "No modality-level anchor key found for "
                    f"modality={modality!r}, node={node!r}. For NN-based results, "
                    "provide both ref_section and modality."
                ) from exc

        try:
            return self.final_anchor_keys[node]
        except KeyError as exc:
            raise KeyError(f"No final anchor key found for node={node!r}.") from exc

    def _binary_series(self, column: str) -> pd.Series:
        """Return one registered anchor column as a 0/1 integer Series."""
        if column not in self.anchor_df.columns:
            raise KeyError(f"Anchor column '{column}' is not found in anchor_df.")

        return pd.to_numeric(
            self.anchor_df[column],
            errors="coerce",
        ).fillna(0).astype(int)

    def get_anchor_mask(
        self,
        node: str,
        modality: Optional[str] = None,
        ref_section: Optional[str] = None,
    ) -> pd.Series:
        """Return a boolean mask for selected anchors."""
        anchor_key = self.get_anchor_key(
            node=node,
            modality=modality,
            ref_section=ref_section,
        )
        return self._binary_series(anchor_key) == 1

    def get_anchor_obs_names(
        self,
        node: str,
        modality: Optional[str] = None,
        ref_section: Optional[str] = None,
    ) -> List[str]:
        """Return obs_names of spots/cells selected as anchors."""
        mask = self.get_anchor_mask(
            node=node,
            modality=modality,
            ref_section=ref_section,
        )
        return self.anchor_df.index[mask.to_numpy()].tolist()

    def get_anchor_adata(
        self,
        adata,
        node: str,
        modality: Optional[str] = None,
        ref_section: Optional[str] = None,
        copy: bool = True,
    ):
        """Return an AnnData subset containing only selected anchors."""
        mask = self.get_anchor_mask(
            node=node,
            modality=modality,
            ref_section=ref_section,
        )
        missing = self.anchor_df.index.difference(adata.obs_names)
        if len(missing) > 0:
            raise ValueError(
                "AnnData is missing observations present in anchor_df. "
                f"Examples: {missing[:5].tolist()}"
            )
        aligned_adata = adata[self.anchor_df.index, :]
        adata_anchor = aligned_adata[mask.to_numpy(), :]
        return adata_anchor.copy() if copy else adata_anchor

    def get_anchor_table(self, include_missing: bool = False) -> pd.DataFrame:
        """
        Return registered anchor columns.

        If ``include_missing=False``, all registered columns must exist.
        If ``include_missing=True``, missing registered columns are included as
        ``NaN`` columns, which is useful for debugging partial results.
        """
        if include_missing:
            return self.anchor_df.reindex(columns=self.anchor_columns).copy()

        self.validate_columns()
        return self.anchor_df[self.anchor_columns].copy()

    def apply_to(self, adata, copy: bool = True):
        """Attach all anchor columns to a compatible AnnData by ``obs_names``."""
        output = adata.copy() if copy or adata.is_view else adata
        missing = self.anchor_df.index.difference(output.obs_names)
        if len(missing) > 0:
            raise ValueError(
                "AnnData is missing observations present in anchor_df. "
                f"Examples: {missing[:5].tolist()}"
            )

        for column in self.anchor_df.columns:
            output.obs.loc[self.anchor_df.index, column] = self.anchor_df[column]
        return output

    # ------------------------------------------------------------------
    # Summary and validation
    # ------------------------------------------------------------------
    def validate_columns(self) -> None:
        """Check that all registered anchor columns exist in ``anchor_df``."""
        missing_cols = [
            col for col in self.anchor_columns
            if col not in self.anchor_df.columns
        ]
        if len(missing_cols) > 0:
            raise KeyError(
                "The following registered anchor columns are missing from anchor_df: "
                f"{missing_cols}"
            )

    def _get_threshold(self, node: str, modality: Optional[str]) -> Any:
        """Return a quantile hierarchy threshold when it is available."""
        if modality is None:
            return np.nan

        modality_thres = self.anchor_thres_dic.get(modality, {})
        if node in modality_thres:
            return modality_thres[node]

        # Backward-compatible support for the previous threshold names.
        if node == self.target_node:
            return modality_thres.get("target_anchor_thres", np.nan)
        if node == self.nontgt_node:
            return modality_thres.get("nontgt_anchor_thres", np.nan)

        return np.nan

    def summary(self) -> pd.DataFrame:
        """Summarize registered anchor columns and anchor counts."""
        rows = []

        def add_row(level, node, column, ref_section=None, modality=None):
            if column not in self.anchor_df.columns:
                n_anchors = np.nan
                anchor_fraction = np.nan
            else:
                values = self._binary_series(column)
                n_anchors = int(values.sum())
                anchor_fraction = float(values.mean())

            rows.append(
                {
                    "method": self.method,
                    "level": level,
                    "ref_section": ref_section,
                    "modality": modality,
                    "node": node,
                    "anchor_key": column,
                    "n_anchors": n_anchors,
                    "anchor_fraction": anchor_fraction,
                    "anchor_threshold": self._get_threshold(node, modality),
                }
            )

        for node, col in self.final_anchor_keys.items():
            add_row("final", node, col)

        for modality, node_dic in self.modality_anchor_keys.items():
            for node, col in node_dic.items():
                add_row("modality", node, col, modality=modality)

        for ref_section, node_dic in self.section_anchor_keys.items():
            for node, col in node_dic.items():
                add_row("ref_section", node, col, ref_section=ref_section)

        for ref_section, modality_dic in self.section_modality_anchor_keys.items():
            for modality, node_dic in modality_dic.items():
                for node, col in node_dic.items():
                    add_row(
                        "ref_section_modality",
                        node,
                        col,
                        ref_section=ref_section,
                        modality=modality,
                    )

        return pd.DataFrame(rows)


# ============================================================
# NN-based anchor detection
# ============================================================
def nn_detect(
    input_adata_annotated,
    input_adata_not_annotated,
    features_set,
    target_tissue_regions,
    anchor_key,
    metric="euclidean",
    label_key="label",
    knn=5,
    random_state=0,
    ef_construction=200,
    M=16,
    ef=50,
    copy=True,
):
    """
    Detect nearest-neighbor anchors in an unannotated AnnData object.

    This function uses spots/cells from selected tissue regions in an annotated
    AnnData object as the query set. It then identifies their nearest neighbors
    in an unannotated AnnData object using HNSW approximate nearest-neighbor
    search. The nearest-neighbor spots in the unannotated object are marked as
    anchors in `adata.obs[anchor_key]`.

    Parameters
    ----------
    input_adata_annotated : AnnData
        Annotated reference AnnData object. Its `obs[label_key]` should contain
        tissue-region labels.

    input_adata_not_annotated : AnnData
        Unannotated or query AnnData object in which anchors will be detected.

    features_set : list, tuple, set, or array-like
        Features used for nearest-neighbor matching. These features should be
        present in both `input_adata_annotated.var_names` and
        `input_adata_not_annotated.var_names`.

    target_tissue_regions : list, tuple, set, or array-like
        Tissue-region labels from the annotated AnnData object used as the query
        population for anchor detection.

    anchor_key : str
        Column name to be added to `input_adata_not_annotated.obs`. Spots/cells
        selected as anchors are marked as 1, and all others are marked as 0.

    metric : {"euclidean", "cosine"}, default="euclidean"
        Distance metric used by HNSW.
        - "euclidean" uses HNSW space `"l2"`.
        - "cosine" uses HNSW space `"cosine"`.

    label_key : str, default="label"
        Column in `input_adata_annotated.obs` containing tissue-region labels.

    knn : int, default=5
        Number of nearest neighbors to retrieve for each annotated query spot.
        Must be positive and cannot exceed the number of observations in
        `input_adata_not_annotated`.

    random_state : int, default=100
        Random seed used by HNSW index construction.

    ef_construction : int, default=200
        HNSW construction parameter controlling index quality. Larger values may
        improve accuracy but increase construction time.

    M : int, default=16
        HNSW graph connectivity parameter. Larger values may improve recall but
        increase memory usage.

    ef : int, default=50
        HNSW query-time search parameter. Larger values may improve recall but
        increase query time.

    copy : bool, default=True
        If True, return a copied AnnData object. If False, modify
        `input_adata_not_annotated` in place.

    Returns
    -------
    adata_ref : AnnData
        AnnData object with an added categorical column `obs[anchor_key]`,
        where 1 indicates detected anchors and 0 indicates non-anchor spots.
    """

    try:
        import hnswlib
    except ImportError as exc:
        raise ImportError(
            "Nearest-neighbor anchor detection requires hnswlib. "
            "Install the project dependencies before running this workflow."
        ) from exc

    if label_key not in input_adata_annotated.obs.columns:
        raise KeyError(
            f"label_key='{label_key}' is not found in input_adata_annotated.obs."
        )

    if metric not in ["euclidean", "cosine"]:
        raise ValueError("metric must be either 'euclidean' or 'cosine'.")

    if not isinstance(knn, int) or knn <= 0:
        raise ValueError("knn must be a positive integer.")

    # The annotated object is read-only in this routine.
    adata_qry = input_adata_annotated
    adata_ref = input_adata_not_annotated.copy() if copy else input_adata_not_annotated

    features_set = list(features_set)

    valid_features = [
        feature
        for feature in features_set
        if feature in adata_qry.var_names and feature in adata_ref.var_names
    ]

    if len(valid_features) == 0:
        raise ValueError(
            "No valid features are shared by input_adata_annotated and "
            "input_adata_not_annotated."
        )

    target_tissue_regions = list(target_tissue_regions)
    target_mask = adata_qry.obs[label_key].isin(target_tissue_regions)

    if target_mask.sum() == 0:
        raise ValueError(
            "No observations in input_adata_annotated match target_tissue_regions."
        )

    if adata_ref.n_obs == 0:
        raise ValueError("input_adata_not_annotated has no observations.")

    if knn > adata_ref.n_obs:
        raise ValueError(
            f"knn={knn} cannot be larger than the number of observations "
            f"in input_adata_not_annotated, which is {adata_ref.n_obs}."
        )

    # Query matrix: annotated spots/cells from selected tissue regions.
    ds1 = adata_qry[target_mask, valid_features].X

    # Reference/search matrix: all spots/cells in the unannotated AnnData.
    ds2 = adata_ref[:, valid_features].X

    # Convert sparse matrices to dense matrices because hnswlib expects dense input.
    if issparse(ds1):
        ds1 = ds1.toarray()

    if issparse(ds2):
        ds2 = ds2.toarray()

    # hnswlib works best with float32 NumPy arrays.
    ds1 = np.asarray(ds1, dtype=np.float32)
    ds2 = np.asarray(ds2, dtype=np.float32)

    dim = ds2.shape[1]
    num_elements = ds2.shape[0]

    if metric == "euclidean":
        space = "l2"
    else:
        space = "cosine"

    tree = hnswlib.Index(space=space, dim=dim)

    tree.init_index(
        max_elements=num_elements,
        ef_construction=ef_construction,
        M=M,
        random_seed=random_state,
    )

    tree.set_ef(ef)

    # num_threads=1 improves reproducibility.
    tree.add_items(data=ds2, num_threads=1)

    # Identify nearest neighbors.
    ind, distances = tree.knn_query(data=ds1, k=knn, num_threads=1)

    anchors = sorted(set(ind.ravel().tolist()))

    adata_ref.obs[anchor_key] = 0

    anchor_obs_names = adata_ref.obs_names[anchors]
    adata_ref.obs.loc[anchor_obs_names, anchor_key] = 1

    adata_ref.obs[anchor_key] = adata_ref.obs[anchor_key].astype("category")

    return adata_ref


def nn_based_anchor_detection(
    ref_adata_sca,
    test_adata_sca,
    combined_genes,
    target_regions,
    nontgt_regions,
    target_node,
    nontgt_node,
    label_key="label",
    knn=5,
    metric="euclidean",
    random_state=0,
    ef_construction=200,
    M=16,
    ef=50,
    print_results=True,
    copy=True,
):
    """
    Detect direction-specific nearest-neighbor anchors for one hierarchy split.

    This function detects anchors in a query/test AnnData object for both
    directions of a hierarchy split:

    1. target node anchors:
       nearest neighbors of annotated reference spots from `target_regions`

    2. non-target node anchors:
       nearest neighbors of annotated reference spots from `nontgt_regions`

    The function adds two columns to `test_adata_sca.obs`:

    - `{target_node}_anchors`
    - `{nontgt_node}_anchors`

    Each column is categorical, with 1 indicating detected anchors and 0
    indicating non-anchor spots.

    Parameters
    ----------
    ref_adata_sca : AnnData
        Scaled annotated reference AnnData object. Its `obs[label_key]` should
        contain tissue-region labels.

    test_adata_sca : AnnData
        Scaled query/test AnnData object in which anchors will be detected.

    combined_genes : list, tuple, set, or array-like
        Features used for nearest-neighbor matching. These should be shared by
        `ref_adata_sca` and `test_adata_sca`.

    target_regions : list, tuple, set, or array-like
        Tissue regions belonging to the target side of the hierarchy split.

    nontgt_regions : list, tuple, set, or array-like
        Tissue regions belonging to the non-target side of the hierarchy split.

    target_node : str
        Name of the target hierarchy node. The target anchor column will be
        named `{target_node}_anchors`.

    nontgt_node : str
        Name of the non-target hierarchy node. The non-target anchor column will
        be named `{nontgt_node}_anchors`.

    label_key : str, default="label"
        Column in `ref_adata_sca.obs` containing tissue-region labels.

    knn : int or list/tuple of two ints, default=5
        Number of nearest neighbors used for anchor detection.

        If `knn` is an integer, the same K is used for both target and non-target
        anchor detection.

        If `knn` is a list or tuple of length 2:
        - `knn[0]` is used for `target_regions`
        - `knn[1]` is used for `nontgt_regions`

    metric : {"euclidean", "cosine"}, default="euclidean"
        Distance metric used by HNSW.

    random_state : int, default=100
        Random seed used by HNSW index construction.

    ef_construction : int, default=200
        HNSW construction parameter controlling index quality.

    M : int, default=16
        HNSW graph connectivity parameter.

    ef : int, default=50
        HNSW query-time search parameter.

    print_results : bool, default=True
        Whether to print running-time information.

    copy : bool, default=True
        If True, return a copied AnnData object. If False, modify
        `test_adata_sca` in place.

    Returns
    -------
    test_adata_sca : AnnData
        Query/test AnnData object with two added categorical anchor columns:
        `{target_node}_anchors` and `{nontgt_node}_anchors`.
    """

    if label_key not in ref_adata_sca.obs.columns:
        raise KeyError(f"label_key='{label_key}' is not found in ref_adata_sca.obs.")

    target_regions = list(target_regions)
    nontgt_regions = list(nontgt_regions)

    input_adata_annotated = ref_adata_sca[
        ref_adata_sca.obs[label_key].isin(target_regions + nontgt_regions)
    ]

    if input_adata_annotated.n_obs == 0:
        raise ValueError(
            "No observations in ref_adata_sca match target_regions or nontgt_regions."
        )

    if isinstance(knn, int):
        target_knn = knn
        nontgt_knn = knn

    elif isinstance(knn, (list, tuple)):
        if len(knn) != 2:
            raise ValueError(
                "If knn is a list or tuple, it must have length 2: "
                "[target_knn, nontgt_knn]."
            )

        target_knn = knn[0]
        nontgt_knn = knn[1]

        if print_results:
            print(f"Adjusted KNN for target regions: {target_knn}")
            print(f"Adjusted KNN for nontgt regions: {nontgt_knn}")

    else:
        raise TypeError("knn must be either an integer or a list/tuple of two integers.")

    nn_start_time = time.time()

    test_adata_sca = nn_detect(
        input_adata_annotated=input_adata_annotated,
        input_adata_not_annotated=test_adata_sca,
        features_set=combined_genes,
        target_tissue_regions=target_regions,
        anchor_key=f"{target_node}_anchors",
        metric=metric,
        label_key=label_key,
        knn=target_knn,
        random_state=random_state,
        ef_construction=ef_construction,
        M=M,
        ef=ef,
        copy=copy,
    )

    test_adata_sca = nn_detect(
        input_adata_annotated=input_adata_annotated,
        input_adata_not_annotated=test_adata_sca,
        features_set=combined_genes,
        target_tissue_regions=nontgt_regions,
        anchor_key=f"{nontgt_node}_anchors",
        metric=metric,
        label_key=label_key,
        knn=nontgt_knn,
        random_state=random_state,
        ef_construction=ef_construction,
        M=M,
        ef=ef,
        copy=False,
    )

    nn_end_time = time.time()
    nn_run_time = nn_end_time - nn_start_time

    if print_results:
        print("=================================== Anchor Detection ===================================")
        print(f"The running time of nn anchor detection: {nn_run_time:.4f} seconds")
        print("\n")

    return test_adata_sca


def aggregate_anchor_columns(
    adata,
    anchor_keys,
    output_key,
    aggregate_mode="union",
    min_count=None,
):
    """
    Aggregate multiple binary anchor columns into one anchor column.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing binary anchor columns in `adata.obs`.

    anchor_keys : list
        Anchor columns to aggregate. Each column should contain 0/1 values.

    output_key : str
        Output column name in `adata.obs`.

    aggregate_mode : {"union", "shared", "min_count"}, default="union"
        Aggregation rule.

        - "union":
            A spot is an anchor if it is detected by at least one input column.

        - "shared":
            A spot is an anchor if it is detected by all input columns.

        - "min_count":
            A spot is an anchor if it is detected by at least `min_count`
            input columns.

    min_count : int or None, default=None
        Minimum number of positive anchor calls required when
        `aggregate_mode="min_count"`.

    Returns
    -------
    adata : AnnData
        AnnData object with added `adata.obs[output_key]`.
    """

    if len(anchor_keys) == 0:
        raise ValueError("anchor_keys cannot be empty.")

    missing_keys = [key for key in anchor_keys if key not in adata.obs.columns]
    if len(missing_keys) > 0:
        raise KeyError(f"The following anchor columns are missing: {missing_keys}")

    anchor_mat = pd.DataFrame(
        {
            key: pd.to_numeric(adata.obs[key], errors="coerce").fillna(0).astype(int)
            for key in anchor_keys
        },
        index=adata.obs_names,
    )

    anchor_counts = anchor_mat.sum(axis=1)

    if aggregate_mode == "union":
        threshold = 1

    elif aggregate_mode == "shared":
        threshold = len(anchor_keys)

    elif aggregate_mode == "min_count":
        if min_count is None:
            raise ValueError("min_count must be provided when aggregate_mode='min_count'.")

        if not isinstance(min_count, int) or min_count <= 0:
            raise ValueError("min_count must be a positive integer.")

        if min_count > len(anchor_keys):
            raise ValueError(
                "min_count cannot be larger than the number of anchor columns."
            )

        threshold = min_count

    else:
        raise ValueError(
            "aggregate_mode must be one of {'union', 'shared', 'min_count'}."
        )

    adata.obs[output_key] = (anchor_counts >= threshold).astype(int)
    adata.obs[output_key] = adata.obs[output_key].astype("category")

    return adata


def nn_based_anchor_detection_single_ref_multimodal(
    ref_adata_sca_dic,
    test_adata_sca_dic,
    features_dic,
    target_regions,
    nontgt_regions,
    target_node,
    nontgt_node,
    ref_section,
    modalities=("Gene", "Protein"),
    modality_aggregate_mode="union",
    label_key="label",
    knn=5,
    metric="euclidean",
    random_state=0,
    ef_construction=200,
    M=16,
    ef=50,
    print_results=True,
    copy=True,
    return_result=True,
):
    """
   Detect anchors for one reference section using multiple modalities.

    For each modality, this function independently detects target-side and
    non-target-side anchors in the modality-specific query AnnData. Then it
    copies the modality-specific anchor columns into one final AnnData object
    and aggregates anchors across modalities.

    Parameters
    ----------
    ref_adata_sca_dic : dict
        Dictionary of modality-specific scaled reference AnnData objects.

        Example:

        {
            "Gene": ref_gene_sca,
            "Protein": ref_protein_sca,
        }

    test_adata_sca_dic : dict
        Query/test AnnData object.

        {
            "Gene": test_gene_sca,
            "Protein": test_protein_sca,
        }

        All modality-specific AnnData objects must have the same `obs_names`
        in the same order.

    features_dic : dict
        Dictionary of modality-specific features.

        Example:

        {
            "Gene": gene_features,
            "Protein": protein_features,
        }

    target_regions : list
        Tissue regions belonging to the target side of the hierarchy split.

    nontgt_regions : list
        Tissue regions belonging to the non-target side of the hierarchy split.

    target_node : str
        Name of the target hierarchy node.

    nontgt_node : str
        Name of the non-target hierarchy node.

    ref_section : str
        Reference section name.

    modalities : tuple or list, default=("Gene", "Protein")
        Modalities used for anchor detection.

    modality_aggregate_mode : {"union", "shared"}, default="union"
        How to aggregate modality-specific anchors within this reference section.

    label_key : str, default="label"
        Region label column in each reference AnnData object.

    knn : int or list/tuple of two ints, default=5
        Number of nearest neighbors used by `nn_based_anchor_detection()`.

    metric : {"euclidean", "cosine"}, default="euclidean"
        Distance metric used by HNSW.

    random_state : int, default=0
        Random seed used by HNSW.

    ef_construction : int, default=200
        HNSW construction parameter.

    M : int, default=16
        HNSW graph connectivity parameter.

    ef : int, default=50
        HNSW query-time search parameter.

    print_results : bool, default=True
        Whether to print progress information.

    copy : bool, default=True
        If True, return a copied AnnData object.

    return_result : bool, default=False
        If False, return the annotated AnnData object.
        If True, return an AnchorDetectionResult object containing the annotated
        AnnData object and registered anchor-column names.

    Output columns
    --------------
    Lowest level, before modality aggregation:
        ``{ref_section}_{modality}_{target_node}_anchors``
        ``{ref_section}_{modality}_{nontgt_node}_anchors``

    Section level, after modality aggregation:
        ``{ref_section}_{target_node}_anchors``
        ``{ref_section}_{nontgt_node}_anchors``

    Returns
    -------
    AnnData or AnchorDetectionResult
        If ``return_result=False``, return the annotated query AnnData object.
        If ``return_result=True``, return an ``AnchorDetectionResult`` with
        registered final, section-level, and section-modality-level columns.
    """
    if modality_aggregate_mode not in ["union", "shared"]:
        raise ValueError("modality_aggregate_mode must be either 'union' or 'shared'.")

    modalities = list(modalities)
    if len(modalities) == 0:
        raise ValueError("modalities cannot be empty.")

    first_modality = modalities[0]
    if first_modality not in test_adata_sca_dic:
        raise KeyError(
            f"The first modality '{first_modality}' is not found in test_adata_sca_dic."
        )

    final_adata = (
        test_adata_sca_dic[first_modality].copy()
        if copy
        else test_adata_sca_dic[first_modality]
    )

    target_anchor_keys = []
    nontgt_anchor_keys = []

    for modality in modalities:
        if modality not in ref_adata_sca_dic:
            raise KeyError(f"modality='{modality}' is not found in ref_adata_sca_dic.")

        if modality not in test_adata_sca_dic:
            raise KeyError(f"modality='{modality}' is not found in test_adata_sca_dic.")

        if modality not in features_dic:
            raise KeyError(f"modality='{modality}' is not found in features_dic.")

        modality_test_adata = test_adata_sca_dic[modality]
        if not final_adata.obs_names.equals(modality_test_adata.obs_names):
            raise ValueError(
                f"obs_names of modality='{modality}' do not match the final AnnData "
                "object. Please make sure all modality-specific query AnnData objects "
                "have the same spots and the same obs_names order."
            )

        modality_target_node = f"{ref_section}_{modality}_{target_node}"
        modality_nontgt_node = f"{ref_section}_{modality}_{nontgt_node}"

        if print_results:
            print(
                f"Detecting anchors: ref_section={ref_section}, "
                f"modality={modality}"
            )

        modality_adata = nn_based_anchor_detection(
            ref_adata_sca=ref_adata_sca_dic[modality],
            test_adata_sca=modality_test_adata,
            combined_genes=features_dic[modality],
            target_regions=target_regions,
            nontgt_regions=nontgt_regions,
            target_node=modality_target_node,
            nontgt_node=modality_nontgt_node,
            label_key=label_key,
            knn=knn,
            metric=metric,
            random_state=random_state,
            ef_construction=ef_construction,
            M=M,
            ef=ef,
            print_results=print_results,
            copy=True,
        )

        modality_target_anchor_key = f"{modality_target_node}_anchors"
        modality_nontgt_anchor_key = f"{modality_nontgt_node}_anchors"

        for key in [modality_target_anchor_key, modality_nontgt_anchor_key]:
            if key not in modality_adata.obs.columns:
                raise KeyError(f"Expected anchor column '{key}' was not generated.")

            final_adata.obs[key] = modality_adata.obs[key].values
            final_adata.obs[key] = final_adata.obs[key].astype("category")

        target_anchor_keys.append(modality_target_anchor_key)
        nontgt_anchor_keys.append(modality_nontgt_anchor_key)

    section_target_key = f"{ref_section}_{target_node}_anchors"
    section_nontgt_key = f"{ref_section}_{nontgt_node}_anchors"

    final_adata = aggregate_anchor_columns(
        adata=final_adata,
        anchor_keys=target_anchor_keys,
        output_key=section_target_key,
        aggregate_mode=modality_aggregate_mode,
    )

    final_adata = aggregate_anchor_columns(
        adata=final_adata,
        anchor_keys=nontgt_anchor_keys,
        output_key=section_nontgt_key,
        aggregate_mode=modality_aggregate_mode,
    )

    if return_result:
        return AnchorDetectionResult.from_nn_single_ref(
            adata=final_adata,
            target_node=target_node,
            nontgt_node=nontgt_node,
            ref_section=ref_section,
            modalities=modalities,
            params={
                "ref_section": ref_section,
                "modalities": modalities,
                "modality_aggregate_mode": modality_aggregate_mode,
                "label_key": label_key,
                "knn": knn,
                "metric": metric,
                "random_state": random_state,
                "ef_construction": ef_construction,
                "M": M,
                "ef": ef,
            },
        )

    return final_adata


def aggregate_anchor_columns_across_ref_sections(
    adata,
    ref_section_list,
    target_node,
    nontgt_node,
    max_missing_sections=1,
    final_target_anchor_key=None,
    final_nontgt_anchor_key=None,
):
    """
    Aggregate section-level anchors across multiple reference sections.

    A spot is selected as a final anchor if it is detected as an anchor by at
    least `max(n_ref_sections - k, 1)` reference sections.

    Parameters
    ----------
    adata : AnnData
        Query/test AnnData object containing section-level anchor columns.

    ref_section_list : list
        Reference section names.

    target_node : str
        Name of the target hierarchy node.

    nontgt_node : str
        Name of the non-target hierarchy node.

    max_missing_sections : int, default=1
        Relaxation parameter for cross-section aggregation.

        The final threshold is:

        `max(n_ref_sections - max_missing_sections, 1)`

        Therefore, larger `max_missing_sections` gives a more permissive anchor definition.

    final_target_anchor_key : str or None, default=None
        Output column name for final target-side anchors.
        If None, use `{target_node}_anchors`.

    final_nontgt_anchor_key : str or None, default=None
        Output column name for final non-target-side anchors.
        If None, use `{nontgt_node}_anchors`.

    Returns
    -------
    adata : AnnData
        Query/test AnnData object with final cross-section anchor columns added.
    """

    ref_section_list = list(ref_section_list)
    n_ref_sections = len(ref_section_list)

    if n_ref_sections == 0:
        raise ValueError("ref_sections cannot be empty.")

    if not isinstance(max_missing_sections, int):
        raise TypeError("max_missing_sections must be an integer.")

    if max_missing_sections < 0:
        raise ValueError("max_missing_sections must be non-negative.")

    min_section_votes = max(n_ref_sections - max_missing_sections, 1)

    target_anchor_keys = [
        f"{ref_section}_{target_node}_anchors"
        for ref_section in ref_section_list
    ]

    nontgt_anchor_keys = [
        f"{ref_section}_{nontgt_node}_anchors"
        for ref_section in ref_section_list
    ]

    if final_target_anchor_key is None:
        final_target_anchor_key = f"{target_node}_anchors"

    if final_nontgt_anchor_key is None:
        final_nontgt_anchor_key = f"{nontgt_node}_anchors"

    adata = aggregate_anchor_columns(
        adata=adata,
        anchor_keys=target_anchor_keys,
        output_key=final_target_anchor_key,
        aggregate_mode="min_count",
        min_count=min_section_votes,
    )

    adata = aggregate_anchor_columns(
        adata=adata,
        anchor_keys=nontgt_anchor_keys,
        output_key=final_nontgt_anchor_key,
        aggregate_mode="min_count",
        min_count=min_section_votes,
    )

    return adata


def nn_based_anchor_detection_multiref_multimodal(
    ref_adata_sca_dic,
    test_adata_sca_dic,
    features_dic,
    target_regions,
    nontgt_regions,
    target_node,
    nontgt_node,
    ref_section_list=None,
    modalities=("Gene", "Protein"),
    modality_aggregate_mode="union",
    max_missing_sections=1,
    label_key="label",
    knn=5,
    metric="euclidean",
    random_state=0,
    ef_construction=200,
    M=16,
    ef=50,
    print_results=True,
    copy=True,
    return_result=True,
):
    """
    Detect anchors using multiple reference sections and multiple modalities.

    For each reference section:
        1. Detect modality-specific anchors independently.
        2. Aggregate modality-specific anchors into section-level anchors.

    Across reference sections:
        3. Aggregate section-level anchors using:

           min_section_votes = max(n_ref_sections - max_missing_sections, 1)

    Parameters
    ----------
    ref_adata_sca_dic : dict
        Nested dictionary of scaled reference AnnData objects.

        Expected structure:

        {
            "H1": {
                "Gene": H1_gene_sca,
                "Protein": H1_protein_sca,
            },
            "G2": {
                "Gene": G2_gene_sca,
                "Protein": G2_protein_sca,
            },
        }

    test_adata_sca : dict
        Query/test AnnData object.

        {
            "Gene": test_gene_sca,
            "Protein": test_protein_sca,
        }

    features_dic : dict
        Nested dictionary of modality-specific features.

        Expected structure:

        {
            "H1": {
                "Gene": H1_gene_features,
                "Protein": H1_protein_features,
            },
            "G2": {
                "Gene": G2_gene_features,
                "Protein": G2_protein_features,
            },
        }

    target_regions : list
        Tissue regions belonging to the target side of the hierarchy split.

    nontgt_regions : list
        Tissue regions belonging to the non-target side of the hierarchy split.

    target_node : str
        Name of the target hierarchy node.

    nontgt_node : str
        Name of the non-target hierarchy node.

    ref_section_list : list or None, default=None
        Reference sections to use. If None, use all keys in `ref_adata_sca_dic`.

    modalities : tuple or list, default=("Gene", "Protein")
        Modalities used for anchor detection.

    modality_aggregate_mode : {"union", "shared"}, default="union"
        How to aggregate anchors across modalities within each reference section.

    max_missing_sections : int, default=1
        Number of reference sections allowed to miss an anchor.

    label_key : str, default="label"
        Region label column in each reference AnnData object.

    knn : int or list/tuple of two ints, default=5
        Number of nearest neighbors used for anchor detection.

    metric : {"euclidean", "cosine"}, default="euclidean"
        Distance metric used by HNSW.

    random_state : int, default=0
        Random seed used by HNSW.

    ef_construction : int, default=200
        HNSW construction parameter.

    M : int, default=16
        HNSW graph connectivity parameter.

    ef : int, default=50
        HNSW query-time search parameter.

    print_results : bool, default=True
        Whether to print progress information.

    copy : bool, default=True
        If True, return a copied AnnData object.

    return_result : bool, default=False
        If False, return the annotated AnnData object as before.
        If True, return an AnchorDetectionResult object containing the annotated
        AnnData object and registered anchor-column names.

    Output columns
    --------------
    Lowest level:
        ``{ref_section}_{modality}_{node}_anchors``

    Section level:
        ``{ref_section}_{node}_anchors``

    Final level:
        ``{node}_anchors``

    The final cross-reference threshold is
    ``max(len(ref_section_list) - max_missing_sections, 1)``.

    Returns
    -------
    AnnData or AnchorDetectionResult
        If ``return_result=False``, return the annotated query AnnData object.
        If ``return_result=True``, return an ``AnchorDetectionResult`` with
        registered final, section-level, and section-modality-level columns.
    """
    modalities = list(modalities)
    if len(modalities) == 0:
        raise ValueError("modalities cannot be empty.")

    if ref_section_list is None:
        ref_section_list = list(ref_adata_sca_dic.keys())
    else:
        ref_section_list = list(ref_section_list)

    if len(ref_section_list) == 0:
        raise ValueError("ref_section_list cannot be empty.")

    final_adata = None

    for ref_section in ref_section_list:
        if ref_section not in ref_adata_sca_dic:
            raise KeyError(
                f"ref_section='{ref_section}' is not found in ref_adata_sca_dic."
            )

        if ref_section not in features_dic:
            raise KeyError(
                f"ref_section='{ref_section}' is not found in features_dic."
            )

        if print_results:
            print()
            print("================================================================================")
            print(f"Reference section: {ref_section}")
            print("================================================================================")

        section_adata = nn_based_anchor_detection_single_ref_multimodal(
            ref_adata_sca_dic=ref_adata_sca_dic[ref_section],
            test_adata_sca_dic=test_adata_sca_dic,
            features_dic=features_dic[ref_section],
            target_regions=target_regions,
            nontgt_regions=nontgt_regions,
            target_node=target_node,
            nontgt_node=nontgt_node,
            ref_section=ref_section,
            modalities=modalities,
            modality_aggregate_mode=modality_aggregate_mode,
            label_key=label_key,
            knn=knn,
            metric=metric,
            random_state=random_state,
            ef_construction=ef_construction,
            M=M,
            ef=ef,
            print_results=print_results,
            copy=True,
            return_result=False,
        )

        if final_adata is None:
            final_adata = section_adata.copy() if copy else section_adata
        else:
            if not final_adata.obs_names.equals(section_adata.obs_names):
                raise ValueError(
                    f"obs_names do not match for ref_section='{ref_section}'."
                )

            new_cols = [
                col for col in section_adata.obs.columns
                if col not in final_adata.obs.columns
            ]

            for col in new_cols:
                final_adata.obs[col] = section_adata.obs[col].values
                final_adata.obs[col] = final_adata.obs[col].astype(
                    section_adata.obs[col].dtype
                )

    final_adata = aggregate_anchor_columns_across_ref_sections(
        adata=final_adata,
        ref_section_list=ref_section_list,
        target_node=target_node,
        nontgt_node=nontgt_node,
        max_missing_sections=max_missing_sections,
        final_target_anchor_key=f"{target_node}_anchors",
        final_nontgt_anchor_key=f"{nontgt_node}_anchors",
    )

    if return_result:
        return AnchorDetectionResult.from_nn_multiref(
            adata=final_adata,
            target_node=target_node,
            nontgt_node=nontgt_node,
            ref_section_list=ref_section_list,
            modalities=modalities,
            params={
                "ref_section_list": ref_section_list,
                "modalities": modalities,
                "modality_aggregate_mode": modality_aggregate_mode,
                "max_missing_sections": max_missing_sections,
                "min_section_votes": max(len(ref_section_list) - max_missing_sections, 1),
                "label_key": label_key,
                "knn": knn,
                "metric": metric,
                "random_state": random_state,
                "ef_construction": ef_construction,
                "M": M,
                "ef": ef,
            },
        )

    return final_adata


# ============================================================
# Quantile-based anchor detection
# ============================================================
def get_expr_positive_sections(
    ref_adata_sca_dic,
    ref_section_list,
    gene,
    region,
    regions_dic,
    perct_cutoff,
    label_key,
):
    """
    Identify reference sections where a gene has positive expression percentile
    in a given tissue region.

    For one gene and one region, this function checks each reference section.
    If the region exists in that section and the selected percentile of the
    gene expression is greater than 0, that section is retained.

    Parameters
    ----------
    ref_adata_sca_dic : dict
        Dictionary of scaled reference AnnData objects.

    ref_section_list : list
        Reference section names to evaluate.

    gene : str
        Gene used for threshold evaluation.

    region : str
        Tissue region label to evaluate.

    regions_dic : dict
        Dictionary mapping each section to its valid tissue regions.

        Example
        -------
        {
            "H1": ["invasive", "immune", "CIS"],
            "G2": ["invasive", "adipose"],
        }

    perct_cutoff : float
        Percentile cutoff used to summarize expression values.

    label_key : str
        Column in `.obs` containing tissue region labels.

    Returns
    -------
    sections_kept : list
        Reference sections where the selected gene percentile in the given
        region is greater than 0.
    """
    sections_kept = []

    for section in ref_section_list:
        if region not in regions_dic[section]:
            continue

        ref_adata_sca = ref_adata_sca_dic[section]

        if label_key not in ref_adata_sca.obs.columns:
            raise KeyError(f"{section}: label_key='{label_key}' is not found in adata.obs.")

        if gene not in ref_adata_sca.var_names:
            continue

        section_exp = ref_adata_sca[
            ref_adata_sca.obs[label_key].astype(str) == str(region),
            gene,
        ].X

        if issparse(section_exp):
            section_exp = section_exp.toarray()

        section_exp = np.asarray(section_exp).reshape(-1)

        if len(section_exp) == 0:
            continue

        section_exp_perct = np.quantile(section_exp, perct_cutoff)

        if section_exp_perct > 0:
            sections_kept.append(section)

    return sections_kept


def lower_tail_thres(
    ref_adata_sca_dic,
    merged_ref_adata_sca,
    target_regions,
    opposite_regions,
    gene_list,
    perct_cf_upper=0.85,
    perct_cf_lower=0.15,
    label_key="label",
    merged_key="sample",
    exclude_regions=("nan", "unknown"),
    exclude_mode="exact",
    print_results=True,
):
    """
    Determine gene-specific expression thresholds for quantile-based anchor detection.

    For each gene, this function first tries to estimate the expression threshold
    from the opposite hierarchy regions using `perct_cf_upper`. If no valid
    opposite-region threshold can be estimated, it falls back to the target
    hierarchy regions using `perct_cf_lower`.

    Parameters
    ----------
    ref_adata_sca_dic : dict
        Dictionary of scaled reference AnnData objects.

        Example:
        {
            "H1": adata_H1,
            "G2": adata_G2,
            "E1": adata_E1,
        }

    merged_ref_adata_sca : AnnData
        Merged scaled reference AnnData object containing all reference sections.

    target_regions : list
        Tissue regions belonging to the target hierarchy node.

    opposite_regions : list
        Tissue regions belonging to the opposite hierarchy node.

    gene_list : list
        Genes used to define anchors for the target hierarchy node.

    perct_cf_upper : float, default=0.85
        Upper percentile cutoff used to estimate thresholds from opposite regions.

    perct_cf_lower : float, default=0.15
        Lower percentile cutoff used to estimate fallback thresholds from target
        regions when no valid opposite-region threshold is available.

    label_key : str, default="label"
        Column in `.obs` containing tissue region labels.

    merged_key : str, default="sample"
        Column in `merged_ref_adata_sca.obs` indicating reference section names.

    exclude_regions : tuple, default=("nan", "unknown")
        Region labels excluded from valid region collection.

    exclude_mode : {"exact", "contains"}, optional
        Whether to exclude labels by exact matching or substring matching.

    print_results : bool, default=True
        Whether to print thresholding progress.

    Returns
    -------
    hier_thres_dic : dict
        Dictionary mapping each retained gene to its expression threshold.

    Raises
    ------
    KeyError
        If `label_key` or `merged_key` is missing.
    ValueError
        If no valid gene threshold can be determined.
    """
    if label_key not in merged_ref_adata_sca.obs.columns:
        raise KeyError(f"label_key='{label_key}' is not found in merged_ref_adata_sca.obs.")

    if merged_key not in merged_ref_adata_sca.obs.columns:
        raise KeyError(f"merged_key='{merged_key}' is not found in merged_ref_adata_sca.obs.")

    if not 0 <= perct_cf_upper <= 1:
        raise ValueError("perct_cf_upper must be between 0 and 1.") 

    if not 0 <= perct_cf_lower <= 1:
        raise ValueError("perct_cf_lower must be between 0 and 1.")

    ref_section_list = list(ref_adata_sca_dic.keys())
    exclude_regions = tuple(str(x).lower() for x in exclude_regions)

    regions_dic = {}

    for section in ref_section_list:
        adata = ref_adata_sca_dic[section]

        if label_key not in adata.obs.columns:
            raise KeyError(f"{section}: label_key='{label_key}' is not found in adata.obs.")

        section_regions = adata.obs[label_key].astype(str).value_counts().index.tolist()
        #section_regions = [r for r in section_regions if r not in exclude_regions]

        if exclude_mode == "exact":
            section_regions = [
                r for r in section_regions
                if str(r).lower() not in exclude_regions
            ]

        elif exclude_mode == "contains":
            section_regions = [
                r for r in section_regions
                if not any(x in str(r).lower() for x in exclude_regions)
            ]

        else:
            raise ValueError("exclude_mode must be either 'exact' or 'contains'.")

        regions_dic[section] = section_regions

    hier_thres_dic = {}

    for gene in gene_list:
        if gene not in merged_ref_adata_sca.var_names:
            if print_results:
                print(f"{gene} skipped: not found in merged_ref_adata_sca.var_names.")
            continue

        region_to_sections = {}
        regions_kept = []
        exp_bound = []

        # ------------------------------------------------------------
        # First try thresholds from opposite regions.
        # ------------------------------------------------------------
        perct_cutoff = perct_cf_upper

        for region in opposite_regions:
            sections_kept = get_expr_positive_sections(
                ref_adata_sca_dic,
                ref_section_list,
                gene,
                region,
                regions_dic,
                perct_cutoff,
                label_key,
            )

            if len(sections_kept) > 0:
                region_to_sections[region] = sections_kept
                regions_kept.append(region)

        # ------------------------------------------------------------
        # If unavailable in opposite regions, fall back to target regions.
        # ------------------------------------------------------------
        if len(regions_kept) == 0:
            perct_cutoff = perct_cf_lower

            if print_results:
                print(
                    f"{gene} threshold = 0 in opposite regions | "
                    f"perct_cf_upper = {perct_cf_upper} -> using target regions."
                )

            for region in target_regions:
                sections_kept = get_expr_positive_sections(
                    ref_adata_sca_dic,
                    ref_section_list,
                    gene,
                    region,
                    regions_dic,
                    perct_cutoff,
                    label_key,
                )

                if len(sections_kept) > 0:
                    region_to_sections[region] = sections_kept
                    regions_kept.append(region)

        if len(regions_kept) == 0:
            if print_results:
                print(
                    f"{gene} threshold = 0 in target regions | "
                    f"perct_cf_lower = {perct_cf_lower} -> Need to increase threshold cutoffs."
                )
            continue

        if print_results:
            print(f"{gene} included tissue regions: [{', '.join(regions_kept)}]")

        for region in regions_kept:
            mask = (
                (merged_ref_adata_sca.obs[label_key].astype(str) == str(region))
                & merged_ref_adata_sca.obs[merged_key].isin(region_to_sections[region])
            )

            exp = merged_ref_adata_sca[mask, gene].X

            if issparse(exp):
                exp = exp.toarray()

            exp = np.asarray(exp).reshape(-1)

            if len(exp) == 0:
                continue

            exp_perct = np.quantile(exp, perct_cutoff)
            exp_bound.append(exp_perct)

        if len(exp_bound) == 0:
            if print_results:
                print(f"{gene} skipped: no valid expression values found.")
            continue

        hier_thres_dic[gene] = float(np.mean(exp_bound))

        if print_results:
            print()

    if len(hier_thres_dic) == 0:
        raise ValueError(
            "No valid gene thresholds were detected. "
            "Consider increasing `perct_cf_upper` or `perct_cf_lower`, "
            "or check whether genes and regions match."
        )

    return hier_thres_dic


def gene_anchor_detection(
    qry_adata_sca,
    hier_thres_dic,
    anchor_prefix="anchor_",
    copy=False,
):
    """
    Detect gene-level anchors in query data using gene-specific expression thresholds.

    For each gene, spots with expression greater than the corresponding threshold
    are marked as anchors.

    Parameters
    ----------
    qry_adata_sca : AnnData
        Scaled query AnnData object.

    hier_thres_dic : dict
        Dictionary mapping genes to expression thresholds.

    anchor_prefix : str, default="anchor_"
        Prefix for gene-level anchor columns added to `qry_adata_sca.obs`.

    copy : bool, default=False
        If True, return a modified copy. If False, modify `qry_adata_sca` in place.

    Returns
    -------
    qry_adata_sca : AnnData
        Query AnnData object with added gene-level anchor columns.

    anchor_columns : list
        Names of added anchor columns.
    """
    if copy:
        qry_adata_sca = qry_adata_sca.copy()

    anchor_columns = []

    for gene, hier_thres in hier_thres_dic.items():
        if gene not in qry_adata_sca.var_names:
            raise KeyError(f"Gene '{gene}' is not found in qry_adata_sca.var_names.")

        anchor_key = f"{anchor_prefix}{gene}"

        exp = qry_adata_sca[:, gene].X

        if issparse(exp):
            exp = exp.toarray()

        exp = np.asarray(exp).reshape(-1)

        qry_adata_sca.obs[anchor_key] = (exp > hier_thres).astype(int)
        anchor_columns.append(anchor_key)

    return qry_adata_sca, anchor_columns


def hier_anchor_detection(
    qry_adata_sca,
    hier_key,
    anchor_columns,
    max_p,
    thres_q,
    copy=False,
    print_results=True,
):
    """
    Detect hierarchy-level anchors by summarizing gene-level anchor indicators.

    Parameters
    ----------
    qry_adata_sca : AnnData
        Query AnnData object containing gene-level anchor columns in `.obs`.

    hier_key : str
        Hierarchy node name. Output columns will be named
        `{hier_key}_anchors_sum` and `{hier_key}_anchors`.

    anchor_columns : list
        Gene-level anchor columns to summarize.

    max_p : float
        Proportion of the maximum anchor count used to define the threshold.

    thres_q : float
        Quantile of anchor counts used to define the threshold.

    copy : bool, default=False
        If True, return a modified copy. If False, modify `qry_adata_sca` in place.

    print_results : bool, default=True
        Whether to print anchor count summaries and thresholds.

    Returns
    -------
    qry_adata_sca : AnnData
        Query AnnData with hierarchy-level anchor columns added.

    anchor_thres : int
        Final anchor-count threshold.
    """
    if copy:
        qry_adata_sca = qry_adata_sca.copy()

    if len(anchor_columns) == 0:
        raise ValueError("anchor_columns is empty.")

    missing_cols = [col for col in anchor_columns if col not in qry_adata_sca.obs.columns]
    if len(missing_cols) > 0:
        raise KeyError(f"Missing anchor columns in qry_adata_sca.obs: {missing_cols}")

    if not 0 <= max_p <= 1:
        raise ValueError("max_p must be between 0 and 1.")  

    if not 0 <= thres_q <= 1:
        raise ValueError("thres_q must be between 0 and 1.")

    summary_key = f"{hier_key}_anchors_sum"
    final_key = f"{hier_key}_anchors"

    qry_adata_sca.obs[summary_key] = qry_adata_sca.obs[anchor_columns].sum(axis=1)

    if print_results:
        print(qry_adata_sca.obs[summary_key].value_counts())

    anchor_max = qry_adata_sca.obs[summary_key].max()

    if anchor_max <= 0:
        anchor_thres = 1
        qry_adata_sca.obs[final_key] = 0
        qry_adata_sca.obs[final_key] = qry_adata_sca.obs[final_key].astype("category")

        if print_results:
            print(f"{hier_key}: no positive gene-level anchors detected.")

        return qry_adata_sca, anchor_thres

    anchor_thres = int(
        min(
            np.round(anchor_max * max_p),
            qry_adata_sca.obs[summary_key].quantile(thres_q),
        )
    )
    anchor_thres = int(max(1, anchor_thres))

    if print_results:
        print(f"{hier_key} hierarchy anchor counts threshold: {anchor_thres}")

    qry_adata_sca.obs[final_key] = (
        qry_adata_sca.obs[summary_key] >= anchor_thres
    ).astype(int)
    qry_adata_sca.obs[final_key] = qry_adata_sca.obs[final_key].astype("category")

    return qry_adata_sca, anchor_thres


def quantile_based_anchor_detection(
    ref_adata_sca_dic,
    merged_ref_adata_sca,
    test_adata_sca,
    target_node,
    nontgt_node,
    target_regions,
    nontgt_regions,
    target_genes,
    nontgt_genes,
    perct_cf_upper=0.85,
    perct_cf_lower=0.15,
    max_p=0.5,
    thres_q=0.8,
    label_key="label",
    merged_key="sample",
    exclude_regions=("nan", "unknown"),
    exclude_mode="exact",
    modality="Gene",
    copy=False,
    print_results=True,
    return_result=True,
):
    """
    Run quantile-based anchor detection for one binary hierarchy split.

    This function identifies anchors for both target and non-target hierarchy
    directions. It first estimates gene-level expression thresholds from
    reference data, then applies those thresholds to query/test data, and finally
    summarizes gene-level anchors into hierarchy-level anchors.

    Parameters
    ----------
    ref_adata_sca_dic : dict
        Dictionary of scaled reference AnnData objects.

    merged_ref_adata_sca : AnnData
        Merged scaled reference AnnData object.

    test_adata_sca : AnnData
        Scaled query/test AnnData object.

    target_node : str
        Name of the target hierarchy node.

    nontgt_node : str
        Name of the non-target hierarchy node.

    target_regions : list
        Region labels belonging to the target node.

    nontgt_regions : list
        Region labels belonging to the non-target node.

    target_genes : list
        Genes used to detect anchors for the target node.

    nontgt_genes : list
        Genes used to detect anchors for the non-target node.

    perct_cf_upper : float, default=0.85
        Upper percentile cutoff for estimating thresholds from opposite regions.

    perct_cf_lower : float, default=0.15
        Lower percentile cutoff for fallback threshold estimation.

    max_p : float or list, default=0.5
        Proportion of maximum anchor count used for hierarchy-level thresholding.
        If a list is provided, it should be `[target_max_p, nontgt_max_p]`.

    thres_q : float or list, default=0.8
        Quantile used for hierarchy-level thresholding.
        If a list is provided, it should be `[target_thres_q, nontgt_thres_q]`.

    label_key : str, default="label"
        Column in `.obs` containing region labels.

    merged_key : str, default="sample"
        Column in merged reference `.obs` indicating section names.

    exclude_regions : tuple, default=("nan", "unknown")
        Region labels excluded from valid region collection.

    exclude_mode : {"exact", "contains"}, optional
        Whether to exclude labels by exact matching or substring matching.

    modality : str, default="Gene"
        Name used to register modality-level anchor columns and hierarchy
        thresholds. For example, ``modality="Gene"`` produces
        ``Gene_{target_node}_anchors`` and ``Gene_{nontgt_node}_anchors``.

    copy : bool, default=False
        If True, return a modified copy. If False, modify `test_adata_sca` in place.

    print_results : bool, default=True
        Whether to print intermediate results.

    return_result : bool, default=True
        If True, return an :class:`AnchorDetectionResult` with the same schema
        used by :func:`quantile_based_anchor_detection_multimodal`. If False,
        return ``(test_adata_sca, anchor_thres_dic)``.

    Output columns
    --------------
    Modality level:
        ``{modality}_{target_node}_anchors``
        ``{modality}_{nontgt_node}_anchors``

    Final level:
        ``{target_node}_anchors``
        ``{nontgt_node}_anchors``

    Returns
    -------
    AnchorDetectionResult, or tuple
        If ``return_result=True``, return a lightweight result containing final
        and modality-level anchor columns. If ``return_result=False``, return
        ``(test_adata_sca, anchor_thres_dic)`` where ``anchor_thres_dic`` has
        structure ``{modality: {node: threshold}}``.
    """
    if not isinstance(modality, str) or len(modality.strip()) == 0:
        raise ValueError("modality must be a non-empty string.")

    if copy:
        test_adata_sca = test_adata_sca.copy()

    if isinstance(max_p, (list, tuple)):
        if len(max_p) != 2:
            raise ValueError("max_p must be a scalar or a length-2 list/tuple.")
        max_p_target, max_p_nontgt = max_p
    else:
        max_p_target = max_p
        max_p_nontgt = max_p

    if isinstance(thres_q, (list, tuple)):
        if len(thres_q) != 2:
            raise ValueError("thres_q must be a scalar or a length-2 list/tuple.")
        thres_q_target, thres_q_nontgt = thres_q
    else:
        thres_q_target = thres_q
        thres_q_nontgt = thres_q

    #----------------------------------------------
    # target regions
    #----------------------------------------------
    target_hier_thres_dic = lower_tail_thres(
        ref_adata_sca_dic=ref_adata_sca_dic,
        merged_ref_adata_sca=merged_ref_adata_sca,
        target_regions=target_regions,
        opposite_regions=nontgt_regions,
        gene_list=target_genes,
        perct_cf_upper=perct_cf_upper,
        perct_cf_lower=perct_cf_lower,
        label_key=label_key,
        merged_key=merged_key,
        exclude_regions = exclude_regions,
        exclude_mode = exclude_mode,
        print_results=print_results,
    )

    if print_results:
        print(f"=========== [{', '.join(target_regions)}] gene expression thresholds ===========")
        print(target_hier_thres_dic)

    #----------------------------------------------
    # nontarget regions
    #----------------------------------------------
    nontgt_hier_thres_dic = lower_tail_thres(
        ref_adata_sca_dic=ref_adata_sca_dic,
        merged_ref_adata_sca=merged_ref_adata_sca,
        target_regions=nontgt_regions,
        opposite_regions=target_regions,
        gene_list=nontgt_genes,
        perct_cf_upper=perct_cf_upper,
        perct_cf_lower=perct_cf_lower,
        label_key=label_key,
        merged_key=merged_key,
        exclude_regions=exclude_regions,
        exclude_mode=exclude_mode,
        print_results=print_results,
    )

    if print_results:
        print(f"=========== [{', '.join(nontgt_regions)}] gene expression thresholds ===========")
        print(nontgt_hier_thres_dic)

    #----------------------------------------------
    # target regions
    #----------------------------------------------
    test_adata_sca, target_anchor_columns = gene_anchor_detection(
        qry_adata_sca=test_adata_sca,
        hier_thres_dic=target_hier_thres_dic,
        anchor_prefix=f"{target_node}_gene_anchor_",
    )

    test_adata_sca, target_anchor_thres = hier_anchor_detection(
        qry_adata_sca=test_adata_sca,
        hier_key=target_node,
        anchor_columns=target_anchor_columns,
        max_p=max_p_target,
        thres_q=thres_q_target,
        print_results=print_results,
    )

    #----------------------------------------------
    # nontarget regions
    #----------------------------------------------
    test_adata_sca, nontgt_anchor_columns = gene_anchor_detection(
        qry_adata_sca=test_adata_sca,
        hier_thres_dic=nontgt_hier_thres_dic,
        anchor_prefix=f"{nontgt_node}_gene_anchor_",
    )

    test_adata_sca, nontgt_anchor_thres = hier_anchor_detection(
        qry_adata_sca=test_adata_sca,
        hier_key=nontgt_node,
        anchor_columns=nontgt_anchor_columns,
        max_p=max_p_nontgt,
        thres_q=thres_q_nontgt,
        print_results=print_results,
    )

    modality_target_anchor_key = f"{modality}_{target_node}_anchors"
    modality_nontgt_anchor_key = f"{modality}_{nontgt_node}_anchors"
    test_adata_sca.obs[modality_target_anchor_key] = test_adata_sca.obs[
        f"{target_node}_anchors"
    ].copy()
    test_adata_sca.obs[modality_nontgt_anchor_key] = test_adata_sca.obs[
        f"{nontgt_node}_anchors"
    ].copy()

    anchor_thres_dic = {
        modality: {
            target_node: int(target_anchor_thres),
            nontgt_node: int(nontgt_anchor_thres),
        }
    }

    if return_result:
        return AnchorDetectionResult.from_quantile_multimodal(
            adata=test_adata_sca,
            target_node=target_node,
            nontgt_node=nontgt_node,
            modalities=[modality],
            anchor_thres_dic=anchor_thres_dic,
            params={
                "modalities": [modality],
                "modality_aggregate_mode": "single",
                "perct_cf_upper": perct_cf_upper,
                "perct_cf_lower": perct_cf_lower,
                "max_p": max_p,
                "thres_q": thres_q,
                "label_key": label_key,
                "merged_key": merged_key,
                "exclude_regions": exclude_regions,
                "exclude_mode": exclude_mode,
            },
        )

    return test_adata_sca, anchor_thres_dic


def quantile_based_anchor_detection_multimodal(
    ref_adata_sca_dic,
    merged_ref_adata_sca_dic,
    test_adata_sca_dic,
    target_node,
    nontgt_node,
    target_regions,
    nontgt_regions,
    target_genes_dic,
    nontgt_genes_dic,
    modalities=("Gene", "Protein"),
    modality_aggregate_mode="union",
    perct_cf_upper=0.85,
    perct_cf_lower=0.15,
    max_p=0.5,
    thres_q=0.8,
    label_key="label",
    merged_key="sample",
    exclude_regions=("nan", "unknown"),
    exclude_mode="exact",
    copy=False,
    print_results=True,
    return_result=True,
):
    """
    Run quantile-based anchor detection using multiple molecular modalities.

    This function assumes that `quantile_based_anchor_detection()` already
    aggregates anchors across multiple reference sections within one modality.
    Therefore, this wrapper only aggregates the final modality-specific anchors
    across modalities.

    For each modality:
        1. Run quantile-based anchor detection independently.
        2. Save modality-specific target and non-target anchor columns.

    Across modalities:
        3. Aggregate modality-specific anchors by union or shared rule.

    Parameters
    ----------
    ref_adata_sca_dic : dict
        Dictionary containing modality-specific reference AnnData dictionaries.

        Expected structure:

        {
            "Gene": {
                "ref_section1": ref_gene_sca_1,
                "ref_section2": ref_gene_sca_2,
            },
            "Protein": {
                "ref_section1": ref_protein_sca_1,
                "ref_section2": ref_protein_sca_2,
            },
        }

    merged_ref_adata_sca_dic : dict
        Dictionary containing modality-specific merged reference AnnData objects.

        Expected structure:

        {
            "Gene": merged_ref_gene_sca,
            "Protein": merged_ref_protein_sca,
        }

    test_adata_sca_dic : dict
        Query/test AnnData object.

        {
            "Gene": test_gene_sca,
            "Protein": test_protein_sca,
        }

        The final returned object will be based on the first modality's query
        AnnData if a dictionary is provided.

    target_node : str
        Name of the target hierarchy node.

    nontgt_node : str
        Name of the non-target hierarchy node.

    target_regions : list
        Region labels belonging to the target node.

    nontgt_regions : list
        Region labels belonging to the non-target node.

    target_genes_dic : dict
        Dictionary of modality-specific features used to detect target anchors.

        Example:

        {
            "Gene": target_marker_genes,
            "Protein": target_marker_proteins,
        }

    nontgt_genes_dic : dict
        Dictionary of modality-specific features used to detect non-target anchors.

        Example:

        {
            "Gene": nontgt_marker_genes,
            "Protein": nontgt_marker_proteins,
        }

    modalities : tuple or list, default=("Gene", "Protein")
        Molecular modalities used for anchor detection.

    modality_aggregate_mode : {"union", "shared"}, default="union"
        How to aggregate modality-specific anchors.

        - "union":
            A spot is a final anchor if detected by at least one modality.

        - "shared":
            A spot is a final anchor if detected by all modalities.

    perct_cf_upper : float, default=0.85
        Upper percentile cutoff used by quantile-based threshold detection.

    perct_cf_lower : float, default=0.15
        Lower percentile cutoff used by quantile-based threshold detection.

    max_p : float or list, default=0.5
        Proportion of maximum anchor count used for hierarchy-level thresholding.
        Passed directly to `quantile_based_anchor_detection()`.

    thres_q : float or list, default=0.8
        Quantile used for hierarchy-level thresholding.
        Passed directly to `quantile_based_anchor_detection()`.

    label_key : str, default="label"
        Column in reference `.obs` containing region labels.

    merged_key : str, default="sample"
        Column in merged reference `.obs` indicating section names.

    exclude_regions : tuple, default=("nan", "unknown")
        Region labels excluded from valid region collection.

    exclude_mode : {"exact", "contains"}, default="exact"
        Whether to exclude labels by exact matching or substring matching.

    copy : bool, default=False
        If True, return a copied AnnData object.

    print_results : bool, default=True
        Whether to print intermediate results.

    return_result: bool, default=False
        If False, return the annotated AnnData object as before.
        If True, return an AnchorDetectionResult object containing the annotated
        AnnData object and registered anchor-column names.

    Output columns
    --------------
    Modality level:
        ``{modality}_{target_node}_anchors``
        ``{modality}_{nontgt_node}_anchors``

    Final level:
        ``{target_node}_anchors``
        ``{nontgt_node}_anchors``

    Returns
    -------
    AnnData and dict, or AnchorDetectionResult
        If ``return_result=False``, return ``(final_adata, anchor_thres_dic)``.
        ``anchor_thres_dic`` has structure ``{modality: {node: threshold}}``.

        If ``return_result=True``, return an ``AnchorDetectionResult`` with
        registered final and modality-level columns. The same threshold
        dictionary is stored in ``result.anchor_thres_dic``.
    """
    if modality_aggregate_mode not in ["union", "shared"]:
        raise ValueError("modality_aggregate_mode must be either 'union' or 'shared'.")

    modalities = list(modalities)
    if len(modalities) == 0:
        raise ValueError("modalities cannot be empty.")

    first_modality = modalities[0]
    if first_modality not in test_adata_sca_dic:
        raise KeyError(
            f"The first modality '{first_modality}' is not found in test_adata_sca_dic."
        )

    final_adata = (
        test_adata_sca_dic[first_modality].copy()
        if copy
        else test_adata_sca_dic[first_modality]
    )

    target_anchor_keys = []
    nontgt_anchor_keys = []
    anchor_thres_dic = {}

    for modality in modalities:
        if modality not in ref_adata_sca_dic:
            raise KeyError(f"modality='{modality}' is not found in ref_adata_sca_dic.")

        if modality not in merged_ref_adata_sca_dic:
            raise KeyError(
                f"modality='{modality}' is not found in merged_ref_adata_sca_dic."
            )

        if modality not in test_adata_sca_dic:
            raise KeyError(f"modality='{modality}' is not found in test_adata_sca_dic.")

        if modality not in target_genes_dic:
            raise KeyError(f"modality='{modality}' is not found in target_genes_dic.")

        if modality not in nontgt_genes_dic:
            raise KeyError(f"modality='{modality}' is not found in nontgt_genes_dic.")

        modality_test_adata = test_adata_sca_dic[modality]
        if not final_adata.obs_names.equals(modality_test_adata.obs_names):
            raise ValueError(
                f"obs_names of modality='{modality}' do not match the final AnnData "
                "object. Please make sure all modality-specific query AnnData objects "
                "have the same spots and the same obs_names order."
            )

        if print_results:
            print()
            print("================================================================================")
            print(f"Quantile-based anchor detection | Modality: {modality}")
            print("================================================================================")

        modality_adata, modality_anchor_thres_dic = quantile_based_anchor_detection(
            ref_adata_sca_dic=ref_adata_sca_dic[modality],
            merged_ref_adata_sca=merged_ref_adata_sca_dic[modality],
            test_adata_sca=modality_test_adata,
            target_node=target_node,
            nontgt_node=nontgt_node,
            target_regions=target_regions,
            nontgt_regions=nontgt_regions,
            target_genes=target_genes_dic[modality],
            nontgt_genes=nontgt_genes_dic[modality],
            perct_cf_upper=perct_cf_upper,
            perct_cf_lower=perct_cf_lower,
            max_p=max_p,
            thres_q=thres_q,
            label_key=label_key,
            merged_key=merged_key,
            exclude_regions=exclude_regions,
            exclude_mode=exclude_mode,
            modality=modality,
            copy=True,
            print_results=print_results,
            return_result=False,
        )

        modality_target_anchor_key = f"{modality}_{target_node}_anchors"
        modality_nontgt_anchor_key = f"{modality}_{nontgt_node}_anchors"

        for key in [modality_target_anchor_key, modality_nontgt_anchor_key]:
            if key not in modality_adata.obs.columns:
                raise KeyError(f"Expected anchor column '{key}' was not generated.")

            final_adata.obs[key] = modality_adata.obs[key].values
            final_adata.obs[key] = final_adata.obs[key].astype("category")

        target_anchor_keys.append(modality_target_anchor_key)
        nontgt_anchor_keys.append(modality_nontgt_anchor_key)

        anchor_thres_dic[modality] = modality_anchor_thres_dic[modality]

    final_adata = aggregate_anchor_columns(
        adata=final_adata,
        anchor_keys=target_anchor_keys,
        output_key=f"{target_node}_anchors",
        aggregate_mode=modality_aggregate_mode,
    )

    final_adata = aggregate_anchor_columns(
        adata=final_adata,
        anchor_keys=nontgt_anchor_keys,
        output_key=f"{nontgt_node}_anchors",
        aggregate_mode=modality_aggregate_mode,
    )

    if return_result:
        return AnchorDetectionResult.from_quantile_multimodal(
            adata=final_adata,
            target_node=target_node,
            nontgt_node=nontgt_node,
            modalities=modalities,
            anchor_thres_dic=anchor_thres_dic,
            params={
                "modalities": modalities,
                "modality_aggregate_mode": modality_aggregate_mode,
                "perct_cf_upper": perct_cf_upper,
                "perct_cf_lower": perct_cf_lower,
                "max_p": max_p,
                "thres_q": thres_q,
                "label_key": label_key,
                "merged_key": merged_key,
                "exclude_regions": exclude_regions,
                "exclude_mode": exclude_mode,
            },
        )

    return final_adata, anchor_thres_dic
