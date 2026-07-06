from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .utils import filter_ranked_genes, rank_genes_groups


@dataclass
class ParentSplitFeatures:
    """
    Selected features for one parent-node split.

    One parent node defines one binary split:

        parent_node -> child_node_1 vs child_node_2

    The object stores two feature summaries for this split:

    anchor_features_dic
        Direction-specific anchor features used for anchor detection.
        Its structure depends on ``anchor_scenario``.

        If ``anchor_scenario="nn_based"``, features are kept separately for
        each reference section because nearest-neighbor anchors are detected
        section by section.

        Example:
            {
                "section1": {
                    "node_1_vs_node_2": ["GeneA", "GeneB"],
                    "node_2_vs_node_1": ["GeneC", "GeneD"],
                },
                "section2": {
                    "node_1_vs_node_2": ["GeneA", "GeneE"],
                    "node_2_vs_node_1": ["GeneC", "GeneF"],
                },
            }

        If ``anchor_scenario="quantile_based"``, features are aggregated
        across reference sections and keyed directly by split direction. A
        feature is kept in a direction if it appears in at least ``count_num``
        reference-section feature lists for that direction.

        Example:
            {
                "node_1_vs_node_2": ["GeneA", "GeneB", "GeneE"],
                "node_2_vs_node_1": ["GeneC", "GeneD", "GeneF"],
            }

    clustering_features_list
        Non-directional feature list used to cluster or subtype query spots
        within this parent split.

        If ``anchor_scenario="nn_based"``, this is the union of both
        directions within each section, followed by aggregation across sections.
        A feature is kept if it appears in at least ``count_num`` section-level
        feature lists.

        Example:
            ["GeneA", "GeneC", "GeneB", "GeneD", "GeneE", "GeneF"]

        If ``anchor_scenario="quantile_based"``, this is the concatenation
        of the aggregated features from ``split_key_1`` and ``split_key_2``;
        no extra de-duplication is applied across the two directions.

        Example:
            ["GeneA", "GeneB", "GeneE", "GeneC", "GeneD", "GeneF"]
    """

    parent_node: str
    child_node_1: str
    child_node_2: str
    split_key_1: str
    split_key_2: str
    anchor_scenario: str

    anchor_features_dic: Dict[str, Any]
    clustering_features_list: List[str]

    child_1_regions: List[str] = field(default_factory=list)
    child_2_regions: List[str] = field(default_factory=list)
    ref_section_list: List[str] = field(default_factory=list)
    count_num: int = 1

    def get_anchor_features_dic(self) -> Dict[str, Any]:
        """Return anchor features for this parent split."""
        return self.anchor_features_dic

    def get_clustering_features(self) -> List[str]:
        """Return features used for clustering this parent split."""
        return self.clustering_features_list

    def get_direction_features(
        self,
        direction: str,
        section: Optional[str] = None,
    ) -> List[str]:
        """
        Retrieve one direction-specific feature list.

        For nn_based, section must be provided.
        For quantile_based, section is ignored.
        """

        if direction in {self.split_key_1, "child_1_vs_child_2"}:
            split_key = self.split_key_1
        elif direction in {self.split_key_2, "child_2_vs_child_1"}:
            split_key = self.split_key_2
        else:
            raise ValueError(
                f"direction must be one of "
                f"'{self.split_key_1}', '{self.split_key_2}', "
                "'child_1_vs_child_2', or 'child_2_vs_child_1'."
            )

        if self.anchor_scenario == "nn_based":
            if section is None:
                raise ValueError(
                    "section must be provided when anchor_scenario='nn_based'."
                )

            if section not in self.anchor_features_dic:
                raise KeyError(
                    f"section='{section}' is not available. "
                    f"Available sections: {list(self.anchor_features_dic.keys())}"
                )

            return self.anchor_features_dic[section].get(split_key, [])

        return self.anchor_features_dic.get(split_key, [])

    def get_direction_features_across_sections(
        self,
        direction: str,
        ref_section_list: Optional[Sequence[str]] = None,
        count_num: int = 1,
        strict: bool = False,
    ) -> List[str]:
        """
        Retrieve direction-specific features aggregated across reference sections.

        This is mainly useful for ``anchor_scenario="nn_based"``, where
        ``anchor_features_dic`` stores one feature list per reference section.
        A feature is kept if it appears in at least ``count_num`` reference
        sections. Setting ``count_num=1`` returns the ordered union across
        sections.

        For ``anchor_scenario="quantile_based"``, direction-specific
        features are already aggregated across reference sections, so this
        method simply returns ``get_direction_features(direction=direction)``.

        Parameters
        ----------
        direction
            Direction-specific split key, such as ``"node_1_vs_node_2"``.
            The aliases ``"child_1_vs_child_2"`` and
            ``"child_2_vs_child_1"`` are also supported.

        ref_section_list
            Reference sections to aggregate. If None, use
            ``self.ref_section_list`` when available; otherwise use all
            section keys in ``anchor_features_dic``.

        count_num
            Minimum number of reference sections in which a feature must
            appear to be kept. Use ``count_num=1`` for a union across sections.

        strict
            If True, raise an error when a requested section is missing.
            If False, skip missing sections.

        Returns
        -------
        features
            Ordered aggregated feature list for the requested direction.

        Example
        -------
        >>> target_genes = split_result.get_direction_features_across_sections(
        ...     direction="node_1_vs_node_2",
        ...     ref_section_list=["section1", "section2", "section3"],
        ...     count_num=2,
        ... )
        """

        if count_num < 1:
            raise ValueError("count_num must be at least 1.")

        if self.anchor_scenario == "quantile_based":
            return self.get_direction_features(direction=direction)

        if ref_section_list is None:
            if len(self.ref_section_list) > 0:
                ref_section_list = self.ref_section_list
            else:
                ref_section_list = list(self.anchor_features_dic.keys())
        else:
            ref_section_list = list(ref_section_list)

        feature_lists = []

        for ref_section in ref_section_list:
            if ref_section not in self.anchor_features_dic:
                if strict:
                    raise KeyError(
                        f"section='{ref_section}' is not available. "
                        f"Available sections: {list(self.anchor_features_dic.keys())}"
                    )
                continue

            features = self.get_direction_features(
                direction=direction,
                section=ref_section,
            )
            feature_lists.append(features)

        return _union_preserve_order(
            feature_lists=feature_lists,
            count_num=count_num,
        )

    def get_split_info(self) -> Dict[str, Any]:
        """Return metadata for this parent-node split."""
        return {
            "parent_node": self.parent_node,
            "child_node_1": self.child_node_1,
            "child_node_2": self.child_node_2,
            "child_1_regions": self.child_1_regions,
            "child_2_regions": self.child_2_regions,
            "included_regions": self.child_1_regions + self.child_2_regions,
            "split_key_1": self.split_key_1,
            "split_key_2": self.split_key_2,
        }


@dataclass
class HierarchicalFeatureResults:
    """
    Final organized result for hierarchical feature selection.

    This object allows easy retrieval by ``parent_node``. Each value in
    ``split_features_dic`` is a :class:`ParentSplitFeatures` object, so the
    same ``anchor_scenario``-specific structures described there apply here.

    Main structure
    --------------
    split_features_dic
        Dictionary keyed by parent node.

        Example:
            {
                "node_0": ParentSplitFeatures(...),
                "node_1": ParentSplitFeatures(...),
            }

    Example: nn_based
    -----------------
    For a parent split ``node_0 -> node_1 vs node_2``:

        split_result = feature_results.get_by_parent_node("node_0")

        split_result.anchor_features_dic
        # {
        #     "section1": {
        #         "node_1_vs_node_2": ["GeneA", "GeneB"],
        #         "node_2_vs_node_1": ["GeneC", "GeneD"],
        #     },
        #     "section2": {
        #         "node_1_vs_node_2": ["GeneA", "GeneE"],
        #         "node_2_vs_node_1": ["GeneC", "GeneF"],
        #     },
        # }

        split_result.clustering_features_list
        # ["GeneA", "GeneC", "GeneB", "GeneD", "GeneE", "GeneF"]

        split_result.get_direction_features(
            direction="node_1_vs_node_2",
            section="section1",
        )
        # ["GeneA", "GeneB"]

    Example: quantile_based
    -----------------------
    For the same parent split:

        split_result = feature_results.get_by_parent_node("node_0")

        split_result.anchor_features_dic
        # {
        #     "node_1_vs_node_2": ["GeneA", "GeneB", "GeneE"],
        #     "node_2_vs_node_1": ["GeneC", "GeneD", "GeneF"],
        # }

        split_result.clustering_features_list
        # ["GeneA", "GeneB", "GeneE", "GeneC", "GeneD", "GeneF"]

        split_result.get_direction_features(direction="node_1_vs_node_2")
        # ["GeneA", "GeneB", "GeneE"]
    """

    anchor_scenario: str
    split_features_dic: Dict[str, ParentSplitFeatures]

    root_node: Optional[str] = None
    ref_section_list: List[str] = field(default_factory=list)
    count_num: int = 1
    filtering_paras: Dict[str, Any] = field(default_factory=dict)
    raw_results_dic: Optional[Dict[str, Any]] = None

    def available_parent_nodes(self) -> List[str]:
        """Return all parent nodes with selected features."""
        return list(self.split_features_dic.keys())

    def get_by_parent_node(self, parent_node: str) -> ParentSplitFeatures:
        """Return ParentSplitFeatures for one parent node."""
        if parent_node not in self.split_features_dic:
            raise KeyError(
                f"parent_node='{parent_node}' is not available. "
                f"Available parent nodes: {self.available_parent_nodes()}"
            )

        return self.split_features_dic[parent_node]

    def get_anchor_features_dic(self, parent_node: str) -> Dict[str, Any]:
        """Return anchor_features_dic for one parent node."""
        return self.get_by_parent_node(parent_node).get_anchor_features_dic()

    def get_clustering_features(self, parent_node: str) -> List[str]:
        """Return clustering_features_list for one parent node."""
        return self.get_by_parent_node(parent_node).get_clustering_features()

    def get_direction_features(
        self,
        parent_node: str,
        direction: str,
        section: Optional[str] = None,
    ) -> List[str]:
        """
        Return one direction-specific feature list for one parent node.

        For ``anchor_scenario="nn_based"``, provide ``section``. For
        ``anchor_scenario="quantile_based"``, ``section`` is ignored because
        features are already aggregated across reference sections.
        """
        return self.get_by_parent_node(parent_node).get_direction_features(
            direction=direction,
            section=section,
        )

    def get_direction_features_across_sections(
        self,
        parent_node: str,
        direction: str,
        ref_section_list: Optional[Sequence[str]] = None,
        count_num: Optional[int] = None,
        strict: bool = False,
    ) -> List[str]:
        """
        Return direction-specific features aggregated across reference sections.

        This is the recommended access pattern when you have one gene list per
        reference section and want either the union or features shared by at
        least ``count_num`` sections.

        Parameters
        ----------
        parent_node
            Parent node defining the binary split.

        direction
            Direction-specific split key, such as ``"node_1_vs_node_2"``.

        ref_section_list
            Reference sections to aggregate. If None, use the reference section
            list stored in the corresponding :class:`ParentSplitFeatures`.

        count_num
            Minimum number of reference sections in which a feature must appear
            to be kept. If None, use ``self.count_num``. Use ``count_num=1``
            for the union across sections.

        strict
            If True, raise an error when a requested section is missing. If
            False, skip missing sections.

        Returns
        -------
        features
            Ordered aggregated feature list for the requested direction.

        Example
        -------
        >>> target_genes = gene_feature_results.get_direction_features_across_sections(
        ...     parent_node=target_parent_node,
        ...     direction=f"{target_node}_vs_{nontgt_node}",
        ...     ref_section_list=ref_section_list,
        ...     count_num=2,
        ... )
        """
        if count_num is None:
            count_num = self.count_num

        return self.get_by_parent_node(
            parent_node
        ).get_direction_features_across_sections(
            direction=direction,
            ref_section_list=ref_section_list,
            count_num=count_num,
            strict=strict,
        )

    def get_split_info(self, parent_node: str) -> Dict[str, Any]:
        """Return split metadata for one parent node."""
        return self.get_by_parent_node(parent_node).get_split_info()


@dataclass
class ParentMultimodalSplitFeatures:
    """
    Multi-modal selected features for one hierarchy parent-node split.

    One parent node defines one binary split, and each modality contributes
    its own :class:`ParentSplitFeatures` result. This class summarizes those
    modality-specific results in two complementary formats.

    modality_features_dic
        Modality-level selected features for this parent split.

        The dictionary is keyed by modality name. Each value is the
        non-directional ``clustering_features_list`` from that modality's
        :class:`ParentSplitFeatures` object for the same parent node.

        This format is useful when downstream multi-modal clustering expects
        one feature list per modality, regardless of reference section.

        Example:
            {
                "Gene": ["EPCAM", "KRT8", "COL1A1", "DCN"],
                "Image": ["hipt_12", "hipt_87", "uni_5"],
                "Protein": ["CD3", "CD20", "PanCK"],
            }

    section_features_dic
        Section-level selected features for this parent split.

        The dictionary is keyed first by reference section and then by
        modality. Each value is the selected feature list for that modality
        within that section.

        For modalities generated from ``anchor_scenario="nn_based"``, the
        section-level list is constructed from that section's two directional
        anchor feature lists using an ordered union.

        For modalities generated from ``anchor_scenario="quantile_based"``,
        features are already aggregated across reference sections. Therefore,
        the same modality-level ``clustering_features_list`` is copied to each
        requested section when section-format output is constructed.

        Example:
            {
                "section1": {
                    "Gene": ["EPCAM", "KRT8", "COL1A1"],
                    "Image": ["hipt_12", "hipt_87"],
                    "Protein": ["CD3", "PanCK"],
                },
                "section2": {
                    "Gene": ["EPCAM", "DCN", "COL1A1"],
                    "Image": ["hipt_12", "uni_5"],
                    "Protein": ["CD20", "PanCK"],
                },
            }
    """

    parent_node: str
    split_info: Dict[str, Any]

    modality_features_dic: Dict[str, List[str]] = field(default_factory=dict)
    section_features_dic: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)

    def get_features_dic(self, output_format: str = "modality") -> Dict[str, Any]:
        """
        Return features_dic in the requested format.

        Parameters
        ----------
        output_format
            Either "modality" or "section".

            If ``output_format="modality"``, return ``modality_features_dic``:
            one feature list per modality.

            If ``output_format="section"``, return ``section_features_dic``:
            one nested dictionary per reference section, with one feature list
            per modality inside each section.

        Returns
        -------
        features_dic
            If ``output_format="modality"``:
                {
                    "Gene": ["EPCAM", "KRT8", "COL1A1", "DCN"],
                    "Image": ["hipt_12", "hipt_87", "uni_5"],
                    "Protein": ["CD3", "CD20", "PanCK"],
                }

            If ``output_format="section"``:
                {
                    "section1": {
                        "Gene": ["EPCAM", "KRT8", "COL1A1"],
                        "Image": ["hipt_12", "hipt_87"],
                        "Protein": ["CD3", "PanCK"],
                    },
                    "section2": {
                        "Gene": ["EPCAM", "DCN", "COL1A1"],
                        "Image": ["hipt_12", "uni_5"],
                        "Protein": ["CD20", "PanCK"],
                    },
                }
        """

        if output_format == "modality":
            return self.modality_features_dic

        if output_format == "section":
            return self.section_features_dic

        raise ValueError("output_format must be either 'modality' or 'section'.")

    def get_modality_features(self, modality: str) -> List[str]:
        """Return selected features for one modality."""
        if modality not in self.modality_features_dic:
            raise KeyError(
                f"modality='{modality}' is not available. "
                f"Available modalities: {list(self.modality_features_dic.keys())}"
            )

        return self.modality_features_dic[modality]

    def get_section_features(self, section: str) -> Dict[str, List[str]]:
        """Return selected modality features for one reference section."""
        if section not in self.section_features_dic:
            raise KeyError(
                f"section='{section}' is not available. "
                f"Available sections: {list(self.section_features_dic.keys())}"
            )

        return self.section_features_dic[section]


@dataclass
class MultimodalHierarchicalFeatureResults:
    """
    Multi-modal hierarchical feature result.

    This object stores cross-modality feature summaries for every parent node.
    """

    split_features_dic: Dict[str, ParentMultimodalSplitFeatures]

    modality_results_dic: Dict[str, Any] = field(default_factory=dict)
    ref_section_list: List[str] = field(default_factory=list)

    def available_parent_nodes(self) -> List[str]:
        """Return all parent nodes with multi-modal features."""
        return list(self.split_features_dic.keys())

    def get_by_parent_node(self, parent_node: str) -> ParentMultimodalSplitFeatures:
        """Return multi-modal features for one parent node."""
        if parent_node not in self.split_features_dic:
            raise KeyError(
                f"parent_node='{parent_node}' is not available. "
                f"Available parent nodes: {self.available_parent_nodes()}"
            )

        return self.split_features_dic[parent_node]

    def get_features_dic(
        self,
        parent_node: str,
        output_format: str = "modality",
    ) -> Dict[str, Any]:
        """Return features_dic for one parent node."""
        return self.get_by_parent_node(parent_node).get_features_dic(
            output_format=output_format
        )

    def get_split_info(self, parent_node: str) -> Dict[str, Any]:
        """Return split metadata for one parent node."""
        return self.get_by_parent_node(parent_node).split_info

    def get_direction_features_dic(
        self,
        parent_node: str,
        direction: str,
        modalities: Optional[Sequence[str]] = None,
        ref_section_list: Optional[Sequence[str]] = None,
        count_num: Optional[int] = None,
        strict: bool = True,
    ) -> Dict[str, List[str]]:
        """Return one hierarchy direction as a modality feature dictionary.

        For NN feature results, section-specific lists are aggregated using
        ``count_num``. For quantile feature results, the already aggregated
        direction list is returned. This provides one common retrieval path
        for clustering and gene-subtyping inputs across transfer frameworks.
        """
        if modalities is None:
            modalities = list(self.modality_results_dic)
        else:
            modalities = list(modalities)

        features_dic: Dict[str, List[str]] = {}
        for modality in modalities:
            feature_results = self.modality_results_dic.get(modality)
            if feature_results is None:
                if strict:
                    raise KeyError(
                        f"No hierarchical feature result is available for "
                        f"modality={modality!r}."
                    )
                continue

            try:
                features = feature_results.get_direction_features_across_sections(
                    parent_node=parent_node,
                    direction=direction,
                    ref_section_list=ref_section_list,
                    count_num=count_num,
                    strict=strict,
                )
            except (KeyError, ValueError):
                if strict:
                    raise
                continue
            features_dic[modality] = list(features)

        return features_dic

    def get_clustering_features_dic(
        self,
        parent_node: str,
        modalities: Optional[Sequence[str]] = None,
        ref_section_list: Optional[Sequence[str]] = None,
        count_num: Optional[int] = None,
        strict: bool = True,
    ) -> Dict[str, List[str]]:
        """Build non-directional clustering features for one binary split."""
        split_info = self.get_split_info(parent_node)
        child_node_1 = split_info["child_node_1"]
        child_node_2 = split_info["child_node_2"]
        if modalities is None:
            modalities = list(self.modality_results_dic)
        else:
            modalities = list(modalities)

        direction_1 = self.get_direction_features_dic(
            parent_node=parent_node,
            direction=f"{child_node_1}_vs_{child_node_2}",
            modalities=modalities,
            ref_section_list=ref_section_list,
            count_num=count_num,
            strict=strict,
        )
        direction_2 = self.get_direction_features_dic(
            parent_node=parent_node,
            direction=f"{child_node_2}_vs_{child_node_1}",
            modalities=modalities,
            ref_section_list=ref_section_list,
            count_num=count_num,
            strict=strict,
        )

        return {
            modality: list(
                dict.fromkeys(
                    direction_1.get(modality, [])
                    + direction_2.get(modality, [])
                )
            )
            for modality in modalities
            if modality in direction_1 or modality in direction_2
        }

    def get_nn_anchor_features(
        self,
        parent_node: str,
        modalities: Optional[Sequence[str]] = None,
        ref_section_list: Optional[Sequence[str]] = None,
        strict: bool = True,
    ) -> Dict[str, Dict[str, List[str]]]:
        """Build section-by-modality feature dictionaries for NN anchors."""
        split_info = self.get_split_info(parent_node)
        direction_1 = (
            f"{split_info['child_node_1']}_vs_{split_info['child_node_2']}"
        )
        direction_2 = (
            f"{split_info['child_node_2']}_vs_{split_info['child_node_1']}"
        )
        if modalities is None:
            modalities = [
                modality
                for modality in self.modality_results_dic
                if modality in {"Gene", "Protein"}
            ]
        else:
            modalities = list(modalities)
        if ref_section_list is None:
            ref_section_list = list(self.ref_section_list)
        else:
            ref_section_list = list(ref_section_list)

        section_features_dic: Dict[str, Dict[str, List[str]]] = {
            section: {} for section in ref_section_list
        }
        for modality in modalities:
            feature_results = self.modality_results_dic.get(modality)
            if feature_results is None:
                if strict:
                    raise KeyError(
                        f"No hierarchical feature result is available for "
                        f"modality={modality!r}."
                    )
                continue
            if feature_results.anchor_scenario != "nn_based":
                if strict:
                    raise ValueError(
                        f"modality={modality!r} uses anchor_scenario="
                        f"{feature_results.anchor_scenario!r}; nn_based "
                        "features are required."
                    )
                continue

            for section in ref_section_list:
                try:
                    target_features = feature_results.get_direction_features(
                        parent_node=parent_node,
                        direction=direction_1,
                        section=section,
                    )
                    nontgt_features = feature_results.get_direction_features(
                        parent_node=parent_node,
                        direction=direction_2,
                        section=section,
                    )
                except (KeyError, ValueError):
                    if strict:
                        raise
                    continue
                section_features_dic[section][modality] = list(
                    dict.fromkeys(target_features + nontgt_features)
                )

        return section_features_dic

    def get_quantile_anchor_features(
        self,
        parent_node: str,
        modalities: Optional[Sequence[str]] = None,
        target_node: Optional[str] = None,
        nontgt_node: Optional[str] = None,
        strict: bool = True,
    ) -> tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """Build directional feature dictionaries for quantile anchors.

        The returned pair can be passed directly as ``target_genes_dic`` and
        ``nontgt_genes_dic`` to
        ``quantile_based_anchor_detection_multimodal``. By default, the first
        and second children stored for ``parent_node`` are treated as target
        and non-target, respectively. Supply both node arguments to reverse
        or explicitly control that direction.

        Parameters
        ----------
        parent_node
            Parent node defining the current binary hierarchy split.
        modalities
            Modalities to include. If None, use all available molecular
            modalities (Gene and Protein).
        target_node, nontgt_node
            The two children defining the requested direction. Either provide
            both or omit both.
        strict
            If True, raise for missing modalities, non-quantile feature
            results, invalid child nodes, or unavailable split features. If
            False, skip unavailable modalities.
        """
        split_info = self.get_split_info(parent_node)
        child_node_1 = split_info["child_node_1"]
        child_node_2 = split_info["child_node_2"]

        if (target_node is None) != (nontgt_node is None):
            raise ValueError(
                "target_node and nontgt_node must either both be provided or "
                "both be omitted."
            )
        if target_node is None:
            target_node, nontgt_node = child_node_1, child_node_2

        if {target_node, nontgt_node} != {child_node_1, child_node_2}:
            raise ValueError(
                f"target_node={target_node!r} and nontgt_node={nontgt_node!r} "
                f"must be the two children of parent_node={parent_node!r}: "
                f"{[child_node_1, child_node_2]}."
            )

        if modalities is None:
            modalities = [
                modality
                for modality in self.modality_results_dic
                if modality in {"Gene", "Protein"}
            ]
        else:
            modalities = list(modalities)

        target_features_dic: Dict[str, List[str]] = {}
        nontgt_features_dic: Dict[str, List[str]] = {}
        target_direction = f"{target_node}_vs_{nontgt_node}"
        nontgt_direction = f"{nontgt_node}_vs_{target_node}"

        for modality in modalities:
            feature_results = self.modality_results_dic.get(modality)
            if feature_results is None:
                if strict:
                    raise KeyError(
                        f"No hierarchical feature result is available for "
                        f"modality={modality!r}."
                    )
                continue

            if feature_results.anchor_scenario != "quantile_based":
                if strict:
                    raise ValueError(
                        f"modality={modality!r} uses anchor_scenario="
                        f"{feature_results.anchor_scenario!r}; quantile_based "
                        "features are required."
                    )
                continue

            try:
                target_features = feature_results.get_direction_features(
                    parent_node=parent_node,
                    direction=target_direction,
                )
                nontgt_features = feature_results.get_direction_features(
                    parent_node=parent_node,
                    direction=nontgt_direction,
                )
            except (KeyError, ValueError):
                if strict:
                    raise
                continue

            target_features_dic[modality] = list(target_features)
            nontgt_features_dic[modality] = list(nontgt_features)

        return target_features_dic, nontgt_features_dic


#=======================================================================
# Part 1. Select hierarchical features
#=======================================================================
def select_hier_genes(
    ref_adata_dic: Dict[str, Any],
    hier_tree,
    anchor_scenario: str,
    filtering_paras: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """
    Select hierarchical gene sets across reference samples using a fitted HierTree object.

    This function connects tree inference results with hierarchical gene selection.

    Parameters
    ----------
    ref_adata_dic
        Dictionary of AnnData objects.

        Example:
            {
                "ref_sample1": adata1,
                "ref_sample2": adata2,
                ...
            }

    hier_tree
        A HierTree object obtained from tree inference.

        Required methods:
            hier_tree.get_split_pairs(order="root_to_leaf")
            hier_tree.get_regions(node)
            hier_tree.root_node

    anchor_scenario
        Either "nn_based" or "quantile_based".

        This controls the default filtering parameters.
        User-provided values in filtering_paras will override the defaults.

    filtering_paras
        Dictionary of filtering parameters.

        Required keys:
            label_key
            pvals_adj
            min_in_out_group_ratio
            min_in_group_fraction
            min_fold_change
            gene_num

        Optional keys:
            two_sides
            logged
            split_order
            verbose

    Returns
    -------
    results_dic
        Dictionary containing selected hierarchical features and split metadata.    

        results_dic["anchor_scenario"]
            Anchor scenario used for feature selection, either "nn_based" or
            "quantile_based".   

        results_dic["root_node"]
            Root node of the hierarchy tree.    

        results_dic["filtering_paras"]
            Final filtering parameters used for feature selection.  

        results_dic["split_info"][parent_node]
            Metadata for each binary split, including child nodes, child regions,
            and direction-specific split keys.  

        results_dic["hier_genes_dic"][sample_name][split_key]
            Direction-specific selected genes for each reference sample and
            hierarchy split.    

            Example:
                results_dic["hier_genes_dic"]["section1"]["node_1_vs_node_2"]   

        results_dic["hier_genenum"][sample_name][parent_node]
            Number of selected genes for each direction of a parent split.  

            Example:
                results_dic["hier_genenum"]["section1"]["node_0"] = (10, 8)

    """

    if anchor_scenario not in ["nn_based", "quantile_based"]:
        raise ValueError(
            "anchor_scenario must be either 'nn_based' or 'quantile_based'."
        )

    # ------------------------------------------------------------------
    # Required parameters
    # ------------------------------------------------------------------
    label_key = filtering_paras["label_key"]

    # ------------------------------------------------------------------
    # Scenario-specific defaults
    # ------------------------------------------------------------------
    if anchor_scenario == "nn_based":
        default_paras = {
            "pvals_adj": 0.05,
            "min_in_out_group_ratio": 1.0,
            "min_in_group_fraction": 0.0,
            "min_fold_change": 1.15,
            "gene_num": 10,
        }

    else:  # quantile_based
        default_paras = {
            "pvals_adj": 0.05,
            "min_in_out_group_ratio": 1.0,
            "min_in_group_fraction": 0.0,
            "min_fold_change": 1.15,
            "gene_num": 5,
        }

    # ------------------------------------------------------------------
    # User-provided parameters override defaults
    # ------------------------------------------------------------------
    pvals_adj = filtering_paras.get(
        "pvals_adj", default_paras["pvals_adj"]
    )

    min_in_out_group_ratio = filtering_paras.get(
        "min_in_out_group_ratio",
        default_paras["min_in_out_group_ratio"],
    )

    min_in_group_fraction = filtering_paras.get(
        "min_in_group_fraction",
        default_paras["min_in_group_fraction"],
    )

    min_fold_change = filtering_paras.get(
        "min_fold_change",
        default_paras["min_fold_change"],
    )

    gene_num = filtering_paras.get(
        "gene_num",
        default_paras["gene_num"],
    )

    two_sides = filtering_paras.get("two_sides", True)
    logged = filtering_paras.get("logged", True)
    verbose = filtering_paras.get("verbose", True)

    split_order = filtering_paras.get("split_order", "root_to_leaf")

    # ------------------------------------------------------------------
    # Use HierTree object to get binary split information
    # ------------------------------------------------------------------
    split_pairs = hier_tree.get_split_pairs(order=split_order)

    results_dic = {
        "anchor_scenario": anchor_scenario,
        "root_node": hier_tree.root_node,
        "filtering_paras": {
            "pvals_adj": pvals_adj,
            "min_in_out_group_ratio": min_in_out_group_ratio,
            "min_in_group_fraction": min_in_group_fraction,
            "min_fold_change": min_fold_change,
            "gene_num": gene_num,
            "two_sides": two_sides,
            "logged": logged,
            "split_order": split_order,
        },
        "split_info": {},
        "hier_genes_dic": {},
        "hier_genenum": {},
    }

    # ==================================================================
    # Iterate over samples
    # ==================================================================
    for sample_name, gene_adata in ref_adata_dic.items():

        if verbose:
            print("\n" + "=" * 80)
            print(f"Selecting hierarchical genes for sample: {sample_name}")
            print(f"Anchor scenario: {anchor_scenario}")
            print(f"Root node: {hier_tree.root_node}")
            print("=" * 80)

        if label_key not in gene_adata.obs.columns:
            raise ValueError(
                f"label_key='{label_key}' is not found in "
                f"ref_adata_dic['{sample_name}'].obs."
            )

        hier_genes_dic = {}
        hier_genenum = {}

        # ==============================================================
        # Iterate over binary splits from inferred tree
        # ==============================================================
        for parent_node, child_node_1, child_node_2 in split_pairs:

            child_1_regions = hier_tree.get_regions(child_node_1)
            child_2_regions = hier_tree.get_regions(child_node_2)

            included_regions = child_1_regions + child_2_regions

            if verbose:
                print("\n" + "-" * 60)
                print(f"Parent node: {parent_node}")
                print(f"Binary split: {child_node_1} vs {child_node_2}")
                print(f"{child_node_1} regions: {child_1_regions}")
                print(f"{child_node_2} regions: {child_2_regions}")

            # ----------------------------------------------------------
            # Subset spots belonging to this binary split
            # ----------------------------------------------------------
            region_mask = gene_adata.obs[label_key].isin(included_regions)

            split_key_1 = f"{child_node_1}_vs_{child_node_2}"
            split_key_2 = f"{child_node_2}_vs_{child_node_1}"

            if parent_node not in results_dic["split_info"]:
                results_dic["split_info"][parent_node] ={
                    "parent_node": parent_node,
                    "child_node_1": child_node_1,
                    "child_node_2": child_node_2,
                    "child_1_regions": child_1_regions,
                    "child_2_regions": child_2_regions,
                    "split_key_1": split_key_1,
                    "split_key_2": split_key_2,
                }

            if region_mask.sum() == 0:
                if verbose:
                    print(
                        f"Skipping {parent_node}: no spots found for "
                        f"{child_node_1} or {child_node_2}."
                    )

                hier_genes_dic[f"{child_node_1}_vs_{child_node_2}"] = []
                hier_genes_dic[f"{child_node_2}_vs_{child_node_1}"] = []
                hier_genenum[parent_node] = (0, 0)

                continue

            adata_sub = gene_adata[region_mask].copy()

            # ----------------------------------------------------------
            # Define binary target label
            # 1: child_node_1 regions
            # 0: child_node_2 regions
            # ----------------------------------------------------------
            adata_sub.obs["target"] = (
                adata_sub.obs[label_key].isin(child_1_regions)
            ).astype(int)

            target_counts = adata_sub.obs["target"].value_counts().to_dict()

            if verbose:
                print(f"Target counts: {target_counts}")

            # Need both classes for binary DE
            if not ({0, 1}.issubset(set(target_counts.keys()))):
                if verbose:
                    print(
                        f"Skipping {parent_node}: only one class is present."
                    )

                hier_genes_dic[f"{child_node_1}_vs_{child_node_2}"] = []
                hier_genes_dic[f"{child_node_2}_vs_{child_node_1}"] = []
                hier_genenum[parent_node] = (0, 0)

                continue

            # ----------------------------------------------------------
            # Differential expression / gene ranking
            # ----------------------------------------------------------
            df1, df0 = rank_genes_groups(
                input_adata=adata_sub,
                target=1,
                label_key="target",
                non_target="rest",
                two_sides=two_sides,
                logged=logged,
            )

            # ----------------------------------------------------------
            # child_node_1-enriched genes
            # ----------------------------------------------------------
            child_1_genes, _ = filter_ranked_genes(
                df=df1,
                pvals_adj=pvals_adj,
                min_in_out_group_ratio=min_in_out_group_ratio,
                min_in_group_fraction=min_in_group_fraction,
                min_fold_change=min_fold_change,
                gene_num=gene_num,
            )

            # ----------------------------------------------------------
            # child_node_2-enriched genes
            # ----------------------------------------------------------
            child_2_genes, _ = filter_ranked_genes(
                df=df0,
                pvals_adj=pvals_adj,
                min_in_out_group_ratio=min_in_out_group_ratio,
                min_in_group_fraction=min_in_group_fraction,
                min_fold_change=min_fold_change,
                gene_num=gene_num,
            )

            hier_genes_dic[split_key_1] = child_1_genes
            hier_genes_dic[split_key_2] = child_2_genes

            hier_genenum[parent_node] = (
                len(child_1_genes),
                len(child_2_genes),
            )

            if verbose:
                print(f"{split_key_1}: {len(child_1_genes)} genes")
                print(f"{split_key_2}: {len(child_2_genes)} genes")

        results_dic["hier_genes_dic"][sample_name] = hier_genes_dic
        results_dic["hier_genenum"][sample_name] = hier_genenum

    return results_dic


def get_hierarchy_split_info_dic(hier_tree, order="root_to_leaf"):
    """
    Get split information for every binary split in a HierTree object.

    Each internal parent node defines one binary split:

        parent_node -> child_node_1 vs child_node_2

    Parameters
    ----------
    hier_tree : HierTree
        Fitted hierarchical tree object.

    order : {"root_to_leaf", "leaf_to_root"}, default="root_to_leaf"
        Order used to retrieve split pairs from the tree.

    Returns
    -------
    split_info_dic : dict
        Dictionary keyed by parent_node.

        Example:
            {
                "node_0": {
                    "parent_node": "node_0",
                    "child_node_1": "node_1",
                    "child_node_2": "node_2",
                    "child_1_regions": [...],
                    "child_2_regions": [...],
                    "split_key_1": "node_1_vs_node_2",
                    "split_key_2": "node_2_vs_node_1",
                },
                ...
            }
    """

    split_pairs = hier_tree.get_split_pairs(order=order)

    split_info_dic = {}

    for parent_node, child_node_1, child_node_2 in split_pairs:

        child_1_regions = hier_tree.get_regions(child_node_1)
        child_2_regions = hier_tree.get_regions(child_node_2)

        split_key_1 = f"{child_node_1}_vs_{child_node_2}"
        split_key_2 = f"{child_node_2}_vs_{child_node_1}"

        split_info_dic[parent_node] = {
            "parent_node": parent_node,
            "child_node_1": child_node_1,
            "child_node_2": child_node_2,
            "child_1_regions": child_1_regions,
            "child_2_regions": child_2_regions,
            "included_regions": child_1_regions + child_2_regions,
            "split_key_1": split_key_1,
            "split_key_2": split_key_2,
        }

    return split_info_dic


def _union_preserve_order(feature_lists, count_num=1):
    """
    Keep features that appear in at least `count_num` feature lists.

    Features are ordered by:
    1. Higher shared count first.
    2. Earlier first appearance if counts are tied.

    Parameters
    ----------
    feature_lists : list of list-like
        A list containing multiple feature lists.

    count_num : int, default=1
        Minimum number of feature lists in which a feature must appear
        to be kept. If count_num=1, this behaves like a union, but the
        returned features are ordered by shared count.

    Returns
    -------
    selected_features : list
        Features appearing in at least `count_num` lists, ordered by
        decreasing shared count.
    """

    if count_num < 1:
        raise ValueError("count_num must be at least 1.")

    feature_counts = {}
    first_seen_order = {}
    order_idx = 0

    for features in feature_lists:
        if features is None:
            continue

        # Count each feature only once within the same feature list
        seen_in_current_list = set()

        for feature in features:
            if feature in seen_in_current_list:
                continue

            seen_in_current_list.add(feature)

            if feature not in feature_counts:
                feature_counts[feature] = 0
                first_seen_order[feature] = order_idx
                order_idx += 1

            feature_counts[feature] += 1

    selected_features = [
        feature
        for feature, count in feature_counts.items()
        if count >= count_num
    ]

    selected_features = sorted(
        selected_features,
        key=lambda feature: (-feature_counts[feature], first_seen_order[feature])
    )

    return selected_features


def normalize_nn_reference_section_guide(
    reference_section_guide: Optional[Mapping[str, Sequence[str]]],
    selected_references: Sequence[str],
    available_parent_nodes: Sequence[str],
    strict: bool = True,
) -> Dict[str, List[str]]:
    """Validate and normalize a node-specific NN reference guide.

    Missing parent nodes are intentionally left absent so callers can interpret
    them as "use all selected references". Explicit empty lists are retained
    and mean that no reference is eligible for that parent split.

    Parameters
    ----------
    reference_section_guide
        User guide mapping parent nodes to reference-section sequences.
    selected_references
        Complete candidate reference list for the current query.
    available_parent_nodes
        Internal parent nodes supported by the hierarchical feature results.
    strict
        Raise for unknown nodes or references outside ``selected_references``.
        If ``False``, drop those entries.

    Returns
    -------
    normalized
        De-duplicated guide containing valid nodes and references only.

    Notes
    -----
    This function only validates and normalizes a guide chosen by the user. It
    does not infer which references are biologically informative. Passing
    ``None`` returns an empty dictionary, which the multi-reference transfer
    session interprets as "use all selected references at every node".

    Examples
    --------
    >>> query_guide = {
    ...     "node_0": ["ref_1", "ref_2", "ref_2"],
    ...     "node_1": ["ref_2"],
    ... }
    >>> normalize_nn_reference_section_guide(
    ...     reference_section_guide=query_guide,
    ...     selected_references=["ref_1", "ref_2", "ref_3"],
    ...     available_parent_nodes=["node_0", "node_1"],
    ... )
    {'node_0': ['ref_1', 'ref_2'], 'node_1': ['ref_2']}
    """
    if reference_section_guide is None:
        return {}
    if not isinstance(reference_section_guide, Mapping):
        raise TypeError(
            "reference_section_guide must map parent nodes to reference lists."
        )

    selected_references = list(dict.fromkeys(selected_references))
    selected_reference_set = set(selected_references)
    available_parent_set = set(available_parent_nodes)
    normalized: Dict[str, List[str]] = {}

    for parent_node, sections in reference_section_guide.items():
        if parent_node not in available_parent_set:
            if strict:
                raise KeyError(
                    f"Unknown parent node {parent_node!r} in "
                    "reference_section_guide. Available parent nodes: "
                    f"{list(available_parent_nodes)}"
                )
            continue
        if (
            sections is None
            or isinstance(sections, (str, bytes))
            or not isinstance(sections, Sequence)
        ):
            raise TypeError(
                f"reference_section_guide[{parent_node!r}] must be a sequence "
                "of section names; use [] for no eligible references."
            )

        unique_sections = list(dict.fromkeys(sections))
        invalid_sections = [
            section
            for section in unique_sections
            if section not in selected_reference_set
        ]
        if invalid_sections and strict:
            raise KeyError(
                f"reference_section_guide[{parent_node!r}] contains sections "
                f"outside selected_references: {invalid_sections}."
            )

        normalized[parent_node] = [
            section
            for section in unique_sections
            if section in selected_reference_set
        ]

    return normalized


def construct_hierarchical_feature_results(
    hier_feature_results_dic: Dict[str, Any],
    hier_tree=None,
    anchor_scenario: Optional[str] = None,
    ref_section_list: Optional[Sequence[str]] = None,
    count_num: int = 1,
    split_order: str = "root_to_leaf",
    feature_result_key: str = "hier_genes_dic",
    strict: bool = False,
    keep_raw_results: bool = True,
) -> HierarchicalFeatureResults:
    """
    Convert raw select_hier_genes() output into HierarchicalFeatureResults.

    Parameters
    ----------
    hier_feature_results_dic
        Raw output from select_hier_genes().

    hier_tree
        HierTree object. Only required if split_info is not already stored
        in hier_feature_results_dic.

    anchor_scenario
        Either "nn_based" or "quantile_based". If None, inferred from
        hier_feature_results_dic["anchor_scenario"].

    ref_section_list
        Reference sections to include. If None, inferred from the raw result.

    count_num
        For nn_based, keep clustering features appearing in at least this many
        reference sections.

    split_order
        Split order used when reconstructing split information from hier_tree.

    feature_result_key
        Key storing selected features. Default is "hier_genes_dic".

    strict
        If True, raise errors for missing sections or split keys.

    keep_raw_results
        If True, store the raw select_hier_genes() output in the final dataclass.

    Returns
    -------
    feature_results
        HierarchicalFeatureResults object.
    """

    if count_num < 1:
        raise ValueError("count_num must be at least 1.")

    if anchor_scenario is None:
        anchor_scenario = hier_feature_results_dic.get("anchor_scenario")

    if anchor_scenario not in {"nn_based", "quantile_based"}:
        raise ValueError(
            "anchor_scenario must be either 'nn_based' or 'quantile_based'."
        )

    if feature_result_key not in hier_feature_results_dic:
        raise KeyError(
            f"hier_feature_results_dic must contain key '{feature_result_key}'."
        )

    hier_features_dic = hier_feature_results_dic[feature_result_key]

    # ------------------------------------------------------------
    # Get split metadata.
    # ------------------------------------------------------------
    split_info_dic = hier_feature_results_dic.get("split_info", {})

    if len(split_info_dic) == 0:
        if hier_tree is None:
            raise ValueError(
                "split_info is missing from hier_feature_results_dic. "
                "Please provide hier_tree to reconstruct split information."
            )

        split_info_dic = get_hierarchy_split_info_dic(
            hier_tree=hier_tree,
            order=split_order,
        )

    # ------------------------------------------------------------
    # Infer reference sections.
    # ------------------------------------------------------------
    if ref_section_list is None:
        ref_section_list = list(hier_features_dic.keys())
    else:
        ref_section_list = list(ref_section_list)

    split_features_dic = {}

    # ============================================================
    # Build one ParentSplitFeatures object for each parent node.
    # ============================================================
    for parent_node, split_info in split_info_dic.items():

        child_node_1 = split_info["child_node_1"]
        child_node_2 = split_info["child_node_2"]

        split_key_1 = split_info["split_key_1"]
        split_key_2 = split_info["split_key_2"]

        child_1_regions = split_info.get("child_1_regions", [])
        child_2_regions = split_info.get("child_2_regions", [])

        # ========================================================
        # Case 1. nn_based
        # ========================================================
        if anchor_scenario == "nn_based":

            anchor_features_dic = {}
            section_level_feature_lists = []

            for ref_section in ref_section_list:

                if ref_section not in hier_features_dic:
                    if strict:
                        raise KeyError(
                            f"ref_section='{ref_section}' is missing from "
                            f"hier_feature_results_dic['{feature_result_key}']."
                        )
                    else:
                        continue

                section_feature_dic = hier_features_dic[ref_section]

                if split_key_1 not in section_feature_dic:
                    if strict:
                        raise KeyError(
                            f"Missing split_key='{split_key_1}' for "
                            f"ref_section='{ref_section}'."
                        )
                    features_1 = []
                else:
                    features_1 = list(section_feature_dic[split_key_1])

                if split_key_2 not in section_feature_dic:
                    if strict:
                        raise KeyError(
                            f"Missing split_key='{split_key_2}' for "
                            f"ref_section='{ref_section}'."
                        )
                    features_2 = []
                else:
                    features_2 = list(section_feature_dic[split_key_2])

                anchor_features_dic[ref_section] = {
                    split_key_1: features_1,
                    split_key_2: features_2,
                }

                section_features = _union_preserve_order(
                    feature_lists=[features_1, features_2],
                    count_num=1,
                )

                if len(section_features) > 0:
                    section_level_feature_lists.append(section_features)

            clustering_features_list = _union_preserve_order(
                feature_lists=section_level_feature_lists,
                count_num=count_num,
            )

        # ========================================================
        # Case 2. quantile_based
        # ========================================================
        else:

            direction_1_feature_lists = []
            direction_2_feature_lists = []

            for ref_section in ref_section_list:

                if ref_section not in hier_features_dic:
                    if strict:
                        raise KeyError(
                            f"ref_section='{ref_section}' is missing from "
                            f"hier_feature_results_dic['{feature_result_key}']."
                        )
                    else:
                        continue

                section_feature_dic = hier_features_dic[ref_section]

                if split_key_1 in section_feature_dic:
                    direction_1_feature_lists.append(
                        list(section_feature_dic[split_key_1])
                    )
                elif strict:
                    raise KeyError(
                        f"Missing split_key='{split_key_1}' for "
                        f"ref_section='{ref_section}'."
                    )

                if split_key_2 in section_feature_dic:
                    direction_2_feature_lists.append(
                        list(section_feature_dic[split_key_2])
                    )
                elif strict:
                    raise KeyError(
                        f"Missing split_key='{split_key_2}' for "
                        f"ref_section='{ref_section}'."
                    )

            features_1 = _union_preserve_order(
                feature_lists=direction_1_feature_lists,
                count_num=count_num,
            )

            features_2 = _union_preserve_order(
                feature_lists=direction_2_feature_lists,
                count_num=count_num,
            )

            anchor_features_dic = {
                split_key_1: features_1,
                split_key_2: features_2,
            }

            clustering_features_list = features_1 + features_2

        split_features_dic[parent_node] = ParentSplitFeatures(
            parent_node=parent_node,
            child_node_1=child_node_1,
            child_node_2=child_node_2,
            split_key_1=split_key_1,
            split_key_2=split_key_2,
            anchor_scenario=anchor_scenario,
            anchor_features_dic=anchor_features_dic,
            clustering_features_list=clustering_features_list,
            child_1_regions=child_1_regions,
            child_2_regions=child_2_regions,
            ref_section_list=ref_section_list,
            count_num=count_num,
        )

    return HierarchicalFeatureResults(
        anchor_scenario=anchor_scenario,
        split_features_dic=split_features_dic,
        root_node=hier_feature_results_dic.get("root_node"),
        ref_section_list=ref_section_list,
        count_num=count_num,
        filtering_paras=hier_feature_results_dic.get("filtering_paras", {}),
        raw_results_dic=hier_feature_results_dic if keep_raw_results else None,
    )


def select_hierarchical_genes_pipeline(
    ref_adata_dic: Dict[str, Any],
    hier_tree,
    anchor_scenario: str,
    filtering_paras: Dict[str, Any],
    ref_section_list: Optional[Sequence[str]] = None,
    count_num: int = 1,
    strict: bool = False,
    keep_raw_results: bool = True,
) -> HierarchicalFeatureResults:
    """
    Select hierarchical genes and return an organized dataclass result.

    This pipeline performs two steps:

    1. Run select_hier_genes() to select direction-specific genes for
       every parent-node split and every reference section.

    2. Convert the raw dictionary output into HierarchicalFeatureResults,
       allowing easy retrieval by parent_node.

    Parameters
    ----------
    ref_adata_dic
        Dictionary of reference AnnData objects.

        Example:
            {
                "section1": adata1,
                "section2": adata2,
            }

    hier_tree
        Fitted HierTree object.

    anchor_scenario
        Either "nn_based" or "quantile_based".

        For "nn_based":
            anchor_features_dic is section-specific.

        For "quantile_based":
            anchor_features_dic is aggregated across sections and directly
            keyed by direction-specific split keys.

    filtering_paras
        Parameters used by select_hier_genes().

        Required:
            label_key

        Common optional keys:
            pvals_adj
            min_in_out_group_ratio
            min_in_group_fraction
            min_fold_change
            gene_num
            two_sides
            logged
            split_order
            verbose

    ref_section_list
        Reference sections to include when constructing the dataclass.
        If None, all sections from ref_adata_dic are used.

    count_num
        Minimum number of reference sections in which a feature must appear
        to be included in clustering_features_list.

        Mainly used for aggregating nn_based features across sections.

    strict
        If True, raise errors for missing sections or split keys.

    keep_raw_results
        If True, store raw select_hier_genes() output inside the final result.

    Returns
    -------
    feature_results
        HierarchicalFeatureResults object.

        Main access patterns:
            feature_results.get_by_parent_node(parent_node)
            feature_results.get_anchor_features_dic(parent_node)
            feature_results.get_clustering_features(parent_node)
            feature_results.get_split_info(parent_node)
    """

    if ref_section_list is not None:
        missing_sections = [
            section
            for section in ref_section_list
            if section not in ref_adata_dic
        ]

        if len(missing_sections) > 0 and strict:
            raise KeyError(
                f"The following sections are not found in ref_adata_dic: "
                f"{missing_sections}."
            )

        valid_ref_section_list = [
            section
            for section in ref_section_list
            if section in ref_adata_dic
        ]

        if len(valid_ref_section_list) == 0:
            raise ValueError(
                "No valid sections remain after applying ref_section_list."
            )

        ref_adata_dic = {
            section: ref_adata_dic[section]
            for section in valid_ref_section_list
        }

    else:
        valid_ref_section_list = list(ref_adata_dic.keys())

    raw_results_dic = select_hier_genes(
        ref_adata_dic=ref_adata_dic,
        hier_tree=hier_tree,
        anchor_scenario=anchor_scenario,
        filtering_paras=filtering_paras,
    )

    feature_results = construct_hierarchical_feature_results(
        hier_feature_results_dic=raw_results_dic,
        hier_tree=hier_tree,
        anchor_scenario=anchor_scenario,
        ref_section_list=valid_ref_section_list,
        count_num=count_num,
        split_order=filtering_paras.get("split_order", "root_to_leaf"),
        strict=strict,
        keep_raw_results=keep_raw_results,
    )

    return feature_results


def construct_multimodal_hierarchical_feature_results(
    modality_results_dic: Dict[str, Any],
    ref_section_list: Optional[Sequence[str]] = None,
    strict: bool = False,
) -> MultimodalHierarchicalFeatureResults:
    """
    Construct multi-modal hierarchical feature results from modality-specific
    HierarchicalFeatureResults objects.

    Parameters
    ----------
    modality_results_dic
        Dictionary of modality-specific HierarchicalFeatureResults.

        Example:
            {
                "Gene": gene_feature_results,
                "Image": image_feature_results,
                "Protein": protein_feature_results,
            }

    ref_section_list
        Reference sections to include in section-level output.
        If None, inferred from nn_based modality results.

    strict
        If True, raise errors when a modality, parent node, or section is missing.
        If False, skip missing entries.

    Returns
    -------
    multimodal_results
        MultimodalHierarchicalFeatureResults object.
    """

    modality_results_dic = {
        modality: result
        for modality, result in modality_results_dic.items()
        if result is not None
    }

    if len(modality_results_dic) == 0:
        raise ValueError(
            "At least one modality-specific HierarchicalFeatureResults "
            "object must be provided."
        )

    first_result = next(iter(modality_results_dic.values()))
    parent_nodes = first_result.available_parent_nodes()

    # ------------------------------------------------------------
    # Infer section list.
    # ------------------------------------------------------------
    if ref_section_list is None:
        section_set = set()

        for feature_results in modality_results_dic.values():
            for parent_node in feature_results.available_parent_nodes():
                split_result = feature_results.get_by_parent_node(parent_node)
                section_set.update(split_result.ref_section_list)

        ref_section_list = sorted(section_set)
    else:
        ref_section_list = list(ref_section_list)

    split_features_dic = {}

    # ============================================================
    # Build multi-modal features for every parent node.
    # ============================================================
    for parent_node in parent_nodes:

        split_info = first_result.get_split_info(parent_node)

        modality_features_dic = {}
        section_features_dic = {
            section: {}
            for section in ref_section_list
        }

        for modality, feature_results in modality_results_dic.items():

            try:
                split_result = feature_results.get_by_parent_node(parent_node)
            except KeyError:
                if strict:
                    raise
                else:
                    continue

            # ----------------------------------------------------
            # Format 1: modality-level features
            # ----------------------------------------------------
            modality_features_dic[modality] = list(
                split_result.clustering_features_list
            )

            # ----------------------------------------------------
            # Format 2: section-level features
            # ----------------------------------------------------
            if split_result.anchor_scenario == "nn_based":

                for section in ref_section_list:

                    if section not in split_result.anchor_features_dic:
                        if strict:
                            raise KeyError(
                                f"section='{section}' is missing for "
                                f"modality='{modality}', parent_node='{parent_node}'."
                            )
                        else:
                            continue

                    section_anchor_dic = split_result.anchor_features_dic[section]

                    section_features = _union_preserve_order(
                        feature_lists=[
                            section_anchor_dic.get(split_result.split_key_1, []),
                            section_anchor_dic.get(split_result.split_key_2, []),
                        ],
                        count_num=1,
                    )

                    section_features_dic[section][modality] = section_features

            else:
                # quantile_based does not naturally have section-specific features.
                # Use the same aggregated features for each section if section format is requested.
                for section in ref_section_list:
                    section_features_dic[section][modality] = list(
                        split_result.clustering_features_list
                    )

        if not strict:
            section_features_dic = {
                section: modality_feature_dic
                for section, modality_feature_dic in section_features_dic.items()
                if len(modality_feature_dic) > 0
            }

        split_features_dic[parent_node] = ParentMultimodalSplitFeatures(
            parent_node=parent_node,
            split_info=split_info,
            modality_features_dic=modality_features_dic,
            section_features_dic=section_features_dic,
        )

    return MultimodalHierarchicalFeatureResults(
        split_features_dic=split_features_dic,
        modality_results_dic=modality_results_dic,
        ref_section_list=list(ref_section_list),
    )


