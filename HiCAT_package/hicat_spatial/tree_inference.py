from __future__ import annotations

import json
import pickle
import pandas as pd
import numpy as np
import scanpy as sc
from scipy import sparse
from scipy.stats import rankdata
from scipy.spatial.distance import pdist, squareform

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

# Local package imports
from .utils import get_region_genes


_COMPONENT_WEIGHT_KEYS = {
    "gene": "w_G",
    "image": "w_I",
    "spatial": "w_S",
}

# Pipeline:
# read in reference_adata_dic 
# -> select region-specific features (gene / image) within each reference sample
# -> within each sample, integrate distances from different modalities
# -> aggregate distances across samples
# -> tree inference, save inferred results
# -> tree structure to guide the subsequent hierarchical feature selection and label transfer


#=======================================================================
# Tree inference result objects
#=======================================================================
@dataclass
class HierTree:
    """
    Container for an inferred hierarchical tissue-region tree.  

    The tree is represented by node-level region membership, parent-child
    adjacency, hierarchy levels, and bigtree-compatible path strings. It is used to
    retrieve binary splits for downstream hierarchical feature selection and label
    transfer.   

    ## Attributes   

    node_dic : dict[str, list[str]]
    Mapping from each node to the tissue regions contained in that node.
    Leaf nodes contain one region, while internal nodes contain all descendant
    regions.    

    hier_dic : dict[int, list[str]]
    Mapping from hierarchy level to node names. Level 0 contains leaf nodes;
    higher levels contain progressively merged internal nodes.  

    adj_dic : dict[str, list[str]]
    Parent-child adjacency dictionary. Each key is an internal parent node, and
    each value contains its two child nodes.    

    path_dic : dict[str, list[str]]
    Mapping from each leaf node to its node path from leaf to root. 

    tree_list : list[str]
    Tree paths formatted for `bigtree.list_to_tree`, used for visualization.    

    region_names : list[str]
    Original tissue-region names used to build the hierarchy.   

    root_node : str
    Name of the root node containing all tissue regions.    

    ## Methods  

    show()
    Display the tree using `bigtree`; falls back to printing tree paths.    

    get_children(node)
    Return the two child nodes of an internal node. 

    get_regions(node)
    Return the tissue regions contained in a node.  

    is_leaf(node)
    Check whether a node is a leaf node.    

    get_leaf_nodes()
    Return all leaf nodes in the tree.  

    get_internal_nodes()
    Return all internal parent nodes in the tree.   

    get_split_pairs(order="root_to_leaf")
    Return binary split tuples as `(parent_node, child_node_1, child_node_2)`.
    Splits can be ordered from root to leaf or from leaf to root.   

    to_text()
    Generate a readable text summary of the tree structure. 

    save_txt(output_path)
    Save the readable tree summary as a text file.  

    save_png(output_path)
    Save the tree structure as a PNG figure using `networkx` and `matplotlib`.
    """
    node_dic: Dict[str, List[str]]
    hier_dic: Dict[int, List[str]]
    adj_dic: Dict[str, List[str]]
    path_dic: Dict[str, List[str]]
    tree_list: List[str]
    region_names: List[str]
    root_node: str

    def show(self) -> None:
        """Display the hierarchical tree structure using bigtree"""
        try:
            from bigtree import list_to_tree
            tree_str = list_to_tree(self.tree_list)
            tree_str.hshow()
        except ImportError:
            print("bigtree is not installed. Showing tree paths instead:")
            for path in self.tree_list:
                print(path)

    def get_children(self, node: str) -> List[str]:
        """Return the two child nodes of an internal node."""
        if node not in self.adj_dic:
            raise ValueError(f"{node} is a leaf node and has no children.")
        return self.adj_dic[node]

    def get_regions(self, node: str) -> List[str]:
        """Return tissue regions contained in a node."""
        if node not in self.node_dic:
            raise ValueError(f"{node} is not found in the tree.")
        return self.node_dic[node]

    def is_leaf(self, node: str) -> bool:
        """Check whether a node is a leaf node."""
        return node not in self.adj_dic

    def get_leaf_nodes(self) -> List[str]:
        """Return all leaf nodes."""
        return [node for node in self.node_dic if node not in self.adj_dic]

    def get_internal_nodes(self) -> List[str]:
        """Return all internal nodes."""
        return list(self.adj_dic.keys())

    def get_split_pairs(self, order: str = "root_to_leaf") -> List[Tuple[str, str, str]]:
        """
        Return binary split information.

        Returns
        -------
        split_pairs:
            List of tuples:
            (parent_node, child_node_1, child_node_2)

        This is useful for hierarchical gene selection and hierarchical label transfer.
        """
        split_pairs = [
            (parent, children[0], children[1])
            for parent, children in self.adj_dic.items()
        ]

        if order == "root_to_leaf":
            split_pairs = sorted(
                split_pairs,
                key=lambda x: len(self.node_dic[x[0]]),
                reverse=True
            )
        elif order == "leaf_to_root":
            split_pairs = sorted(
                split_pairs,
                key=lambda x: len(self.node_dic[x[0]])
            )
        else:
            raise ValueError("order must be either 'root_to_leaf' or 'leaf_to_root'.")

        return split_pairs

    def to_text(self) -> str:
        """Create a readable text representation of the tree."""
        lines = []
        lines.append(f"Root node: {self.root_node}")
        lines.append("")
        lines.append("Node dictionary:")
        for node, regions in self.node_dic.items():
            lines.append(f"{node}: {regions}")

        lines.append("")
        lines.append("Adjacency dictionary:")
        for parent, children in self.adj_dic.items():
            lines.append(f"{parent}: {children}")

        lines.append("")
        lines.append("Tree paths:")
        for path in self.tree_list:
            lines.append(path)

        return "\n".join(lines)

    def save_txt(self, output_path: str | Path) -> None:
        """Save tree structure as a .txt file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(self.to_text())

    def save_png(
        self,
        output_path: str | Path,
        *,
        fig_size: Optional[Tuple[float, float]] = None,
        dpi: int = 300,
        node_size: int = 5200,
        leaf_node_size: Optional[int] = None,
        font_size: int = 9,
        internal_color: str = "#1769B5",
        root_color: str = "#155FA8",
        leaf_colors: Optional[Mapping[str, str]] = None,
        node_alpha: float = 0.88,
        edge_color: str = "#2E2E2E",
        show_internal_regions: bool = False,
        max_regions_per_internal_label: int = 3,
        leaf_label_width: int = 20,
        title: Optional[str] = None,
    ) -> None:
        """
        Save tree structure as a .png file.

        This version uses networkx + matplotlib and draws a clean hierarchy
        with circular nodes, colored leaf regions, and adaptive spacing.

        Parameters
        ----------
        output_path : path-like
            Path to the output PNG file.
        fig_size : tuple[float, float] or None, default=None
            Matplotlib figure size. If None, size is chosen from the number of
            leaf regions and hierarchy depth.
        dpi : int, default=300
            Output resolution.
        node_size : int, default=5200
            Matplotlib scatter size for internal nodes.
        leaf_node_size : int or None, default=None
            Scatter size for leaf nodes. If None, uses ``node_size``.
        font_size : int, default=10
            Base font size for node labels.
        internal_color, root_color : str
            Colors for internal nodes and root node.
        leaf_colors : mapping or None, default=None
            Optional color mapping for leaf nodes. Keys can be region names or
            node names. Missing leaves use a default qualitative palette.
        node_alpha : float, default=0.88
            Node fill transparency. Use values closer to 1 for solid colors and
            lower values for softer, more transparent colors.
        edge_color : str, default="#2E2E2E"
            Tree edge color.
        show_internal_regions : bool, default=False
            If True, small internal branches also show their region names.
        max_regions_per_internal_label : int, default=3
            Maximum number of regions shown inside an internal node when
            ``show_internal_regions=True``.
        leaf_label_width : int, default=20
            Soft wrapping width for long leaf region labels.
        title : str or None, default=None
            Optional figure title.
        """
        import textwrap
        import matplotlib.pyplot as plt
        import networkx as nx

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if dpi < 1:
            raise ValueError("dpi must be at least 1.")
        if node_size <= 0:
            raise ValueError("node_size must be positive.")
        if leaf_node_size is not None and leaf_node_size <= 0:
            raise ValueError("leaf_node_size must be positive.")
        if font_size <= 0:
            raise ValueError("font_size must be positive.")
        if not 0 < node_alpha <= 1:
            raise ValueError("node_alpha must be in the interval (0, 1].")
        if max_regions_per_internal_label < 1:
            raise ValueError("max_regions_per_internal_label must be at least 1.")
        if leaf_label_width < 1:
            raise ValueError("leaf_label_width must be at least 1.")

        graph = nx.DiGraph()

        for parent, children in self.adj_dic.items():
            for child in children:
                graph.add_edge(parent, child)

        leaf_nodes = self.get_leaf_nodes()
        n_leaves = max(len(leaf_nodes), 1)
        depth = max(len(self.hier_dic), 1)
        if fig_size is None:
            fig_size = (
                max(9.0, 1.9 * n_leaves + 3.0),
                max(6.0, 1.35 * depth + 2.5),
            )

        pos = _hierarchy_pos(
            graph,
            self.root_node,
            width=max(2.0 * n_leaves, 2.0),
            vert_gap=1.35,
            vert_loc=0.0,
            xcenter=0.0,
        )

        default_leaf_palette = [
            "#79D3C1",
            "#9AD27F",
            "#F7D65A",
            "#F58ABD",
            "#B89BE8",
            "#F47C77",
            "#8BC7F7",
            "#F4A261",
            "#A6D854",
            "#D4A6C8",
            "#66C2A5",
            "#FC8D62",
        ]
        leaf_colors = dict(leaf_colors or {})
        default_color_by_region = {
            region: default_leaf_palette[i % len(default_leaf_palette)]
            for i, region in enumerate(self.region_names)
        }

        def _node_color(node: str) -> str:
            regions = self.node_dic[node]
            if node == self.root_node:
                return root_color
            if len(regions) == 1:
                region = regions[0]
                return leaf_colors.get(
                    node,
                    leaf_colors.get(region, default_color_by_region.get(region, "#79D3C1")),
                )
            return internal_color

        def _node_label(node: str) -> str:
            regions = self.node_dic[node]
            if len(regions) == 1:
                region = textwrap.fill(
                    str(regions[0]).replace("_", "_"),
                    width=leaf_label_width,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
                return f"{node}\n{region}"
            if show_internal_regions and len(regions) <= max_regions_per_internal_label:
                region_text = "\n".join(map(str, regions))
                return f"{node}\n{region_text}"
            return node

        fig, ax = plt.subplots(figsize=fig_size)

        for parent, child in graph.edges():
            x1, y1 = pos[parent]
            x2, y2 = pos[child]
            ax.plot(
                [x1, x2],
                [y1, y2],
                color=edge_color,
                linewidth=2.2,
                solid_capstyle="round",
                zorder=1,
            )

        nodes = list(graph.nodes())
        xs = np.array([pos[node][0] for node in nodes], dtype=float)
        ys = np.array([pos[node][1] for node in nodes], dtype=float)
        node_sizes = [
            leaf_node_size or node_size
            if node in leaf_nodes
            else node_size
            for node in nodes
        ]
        node_colors = [_node_color(node) for node in nodes]

        x_span = float(xs.max() - xs.min()) if xs.size else 1.0
        y_span = float(ys.max() - ys.min()) if ys.size else 1.0
        shadow_dx = max(x_span, 1.0) * 0.006
        shadow_dy = max(y_span, 1.0) * 0.018

        ax.scatter(
            xs + shadow_dx,
            ys - shadow_dy,
            s=[size * 1.08 for size in node_sizes],
            c="#000000",
            alpha=0.16,
            linewidths=0,
            zorder=2,
        )
        ax.scatter(
            xs,
            ys,
            s=node_sizes,
            c=node_colors,
            alpha=node_alpha,
            edgecolors="white",
            linewidths=3.0,
            zorder=3,
        )

        for node in nodes:
            x, y = pos[node]
            is_leaf = node in leaf_nodes
            color = "black" if is_leaf else "white"
            weight = "bold" if not is_leaf else "semibold"
            ax.text(
                x,
                y,
                _node_label(node),
                ha="center",
                va="center",
                fontsize=font_size if not is_leaf else max(font_size - 1, 6),
                fontweight=weight,
                color=color,
                linespacing=1.15,
                zorder=4,
            )

        if title is not None:
            ax.set_title(title, fontsize=font_size + 2, fontweight="bold", pad=16)

        x_margin = max(x_span * 0.12, 0.8)
        y_margin = max(y_span * 0.16, 0.45)
        ax.set_xlim(xs.min() - x_margin, xs.max() + x_margin)
        ax.set_ylim(ys.min() - y_margin, ys.max() + y_margin)
        ax.set_axis_off()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.25)
        plt.close(fig)


def _hierarchy_pos(
    graph,
    root,
    width: float = 1.0,
    vert_gap: float = 0.2,
    vert_loc: float = 0.0,
    xcenter: float = 0.5,
    pos: Optional[dict] = None,
):
    """
    Helper function to place tree nodes hierarchically.

    Used by HierTree.save_png().
    """
    if pos is None:
        pos = {}

    pos[root] = (xcenter, vert_loc)
    children = list(graph.successors(root))

    if len(children) != 0:
        dx = width / len(children)
        next_x = xcenter - width / 2 - dx / 2

        for child in children:
            next_x += dx
            pos = _hierarchy_pos(
                graph,
                child,
                width=dx,
                vert_gap=vert_gap,
                vert_loc=vert_loc - vert_gap,
                xcenter=next_x,
                pos=pos
            )

    return pos


def tree_str_gene_selection(
    input_adata, 
    gene_num=10, 
    min_fold_change=1.15, 
    min_in_out_group_ratio=1, 
    min_in_group_fraction=0.5, 
    pvals_adj=0.05, 
    label_key="label", 
    exclude_regions=("nan", "unknown"), 
    exclude_mode="exact",
    print_results=True):
    """
    Select region-specific genes for all tissue regions in input_adata.

    Parameters
    ----------
    input_adata : AnnData
        Input AnnData object.
    gene_num : int
        Maximum number of genes selected for each region.
    min_fold_change : float
        Minimum fold-change threshold.
    min_in_out_group_ratio : float
        Minimum in/out group ratio threshold.
    min_in_group_fraction : float
        Minimum fraction of cells/spots expressing the gene in the target group.
    pvals_adj : float
        Maximum adjusted p-value threshold.
    label_key : str
        Column name in input_adata.obs containing tissue region labels.
    exclude_regions : tuple
        Region labels to exclude.
    exclude_mode : {"contains", "exact"}
        Whether to exclude labels by substring matching or exact matching.
    print_results : bool
        Whether to print intermediate results.

    Returns
    -------
    region_genes_dic : dict
        Dictionary where keys are tissue regions and values are selected marker genes.
    gene_list : list
        Union of selected genes across all regions.
    """

    region_genes_dic={}

    region_list=input_adata.obs[label_key].value_counts().index.tolist()
    #region_list=[region for region in region_list if str(region) not in exclude_regions]

    if exclude_mode == "exact":
        region_list = [
            region for region in region_list
            if str(region).lower() not in exclude_regions
        ]

    elif exclude_mode == "contains":
        region_list = [
            region for region in region_list
            if not any(exclude_region in str(region).lower()
                    for exclude_region in exclude_regions)
        ]

    else:
        raise ValueError("exclude_mode must be either 'exact' or 'contains'.")

    print(f"Included tissue regions: {region_list}")

    # identify region-specific genes
    for region in region_list:
        region_genes, df1_filtered=get_region_genes(input_adata=input_adata, 
            region=region, 
            label_key=label_key, 
            gene_num=gene_num, 
            min_fold_change=min_fold_change, 
            min_in_out_group_ratio=min_in_out_group_ratio, 
            min_in_group_fraction=min_in_group_fraction, 
            pvals_adj=pvals_adj, 
            print_results=print_results)
        
        region_genes_dic[region]=region_genes

    # summarize the identified genes into a gene list
    gene_list = sorted(set(gene for genes in region_genes_dic.values() for gene in genes))

    return region_genes_dic, gene_list


def select_tree_inference_features(
    ref_adata_dic,
    label_key="label",
    image_available=False,
    image_feature_key="hipt",
    gene_filtering_paras=None,
    image_filtering_paras=None,
    exclude_regions=("nan", "unknown"),
    exclude_mode="exact",
    print_results=True,
):
    """
    Select region-specific gene and optional image features for tree inference.

    This function separates gene features and image features for each sample,
    performs region-specific feature selection separately for each modality,
    and returns a sample-specific feature dictionary for downstream distance
    calculation.

    Parameters
    ----------
    ref_adata_dic : dict
        Dictionary of reference-sample-specific AnnData objects 
        (including different-modal information if available).

    label_key : str
        Column in adata.obs containing tissue region labels.

    image_available : bool
        Whether image features are available and should be used.

    image_feature_key : str
        Keyword used to identify image features from adata.var.index.
        For example: "hipt", "uni", "gigapath".

    gene_filtering_paras : dict or None
        Filtering parameters for region-specific gene feature selection.

    image_filtering_paras : dict or None
        Filtering parameters for region-specific image feature selection.

    exclude_regions : tuple
        Region names or keywords to exclude.

    exclude_mode : {"exact", "contains"}
        How to exclude regions.

    print_results : bool
        Whether to print intermediate information.

    Returns
    -------
    features_dic : dict
        Sample-specific feature dictionary.

        Example:
            {
                "H1": {
                    "gene": [...],
                    "image": [...],
                },
                "G2": {
                    "gene": [...],
                    "image": [...],
                },
            }

    region_genes_dic : dict
        Region-specific selected gene features for each sample.

    region_image_dic : dict
        Region-specific selected image features for each sample.

    selected_gene_dic : dict
        Selected gene feature list for each sample.

    selected_image_dic : dict
        Selected image feature list for each sample.
    """

    # ============================================================
    # Default parameters
    # ============================================================

    if gene_filtering_paras is None:
        gene_filtering_paras = {
            "min_fold_change": 1.1,
            "min_in_out_group_ratio": 1,
            "min_in_group_fraction": 0,
            "pvals_adj": 0.05,
            "gene_num": 10,
        }

    if image_filtering_paras is None:
        image_filtering_paras = {
            "min_fold_change": 1.1,
            "min_in_out_group_ratio": 1,
            "min_in_group_fraction": 0,
            "pvals_adj": 0.05,
            "gene_num": 5,
        }

    if len(ref_adata_dic) == 0:
        raise ValueError("ref_adata_dic is empty.")

    # ============================================================
    # Initialize outputs
    # ============================================================

    features_dic = {}

    region_genes_dic = {}
    region_image_dic = {}

    selected_gene_dic = {}
    selected_image_dic = {}

    if print_results:
        print("\n============================================================")
        print("Selecting region-specific gene and image features")
        print("============================================================")

    # ============================================================
    # Select features sample by sample
    # ============================================================

    for sample_name, sample_adata in ref_adata_dic.items():

        if print_results:
            print(f"\n==================== Sample: {sample_name} ====================")

        if label_key not in sample_adata.obs.columns:
            raise ValueError(
                f"{sample_name}: label_key='{label_key}' is not found in adata.obs."
            )

        all_features = sample_adata.var.index.tolist()

        # ------------------------------------------------------------
        # Separate gene and image features
        # ------------------------------------------------------------

        if image_available:
            image_features = [
                f for f in all_features
                if image_feature_key.lower() in str(f).lower()
            ]

            gene_features = [
                f for f in all_features
                if image_feature_key.lower() not in str(f).lower()
            ]

        else:
            image_features = []
            gene_features = all_features

        if len(gene_features) == 0:
            raise ValueError(
                f"{sample_name}: No gene features found after excluding image features."
            )

        # ------------------------------------------------------------
        # Select region-specific gene features
        # ------------------------------------------------------------

        gene_adata = sample_adata[
            :,
            sample_adata.var.index.isin(gene_features)
        ].copy()

        if print_results:
            print("\n-------------------- Gene feature selection --------------------")
            print(f"Detected {len(gene_features)} gene features.")

        sample_region_genes_dic, gene_list = tree_str_gene_selection(
            input_adata=gene_adata,
            gene_num=gene_filtering_paras["gene_num"],
            min_fold_change=gene_filtering_paras["min_fold_change"],
            min_in_out_group_ratio=gene_filtering_paras["min_in_out_group_ratio"],
            min_in_group_fraction=gene_filtering_paras["min_in_group_fraction"],
            pvals_adj=gene_filtering_paras["pvals_adj"],
            label_key=label_key,
            exclude_regions=exclude_regions,
            exclude_mode=exclude_mode,
            print_results=print_results,
        )

        region_genes_dic[sample_name] = sample_region_genes_dic
        selected_gene_dic[sample_name] = gene_list

        # ------------------------------------------------------------
        # Select region-specific image features
        # ------------------------------------------------------------

        if image_available and len(image_features) > 0:

            image_adata = sample_adata[
                :,
                sample_adata.var.index.isin(image_features)
            ].copy()

            if print_results:
                print("\n-------------------- Image feature selection --------------------")
                print(
                    f"Detected {len(image_features)} image features "
                    f"using image_feature_key='{image_feature_key}'."
                )

            sample_region_image_dic, image_list = tree_str_gene_selection(
                input_adata=image_adata,
                gene_num=image_filtering_paras["gene_num"],
                min_fold_change=image_filtering_paras["min_fold_change"],
                min_in_out_group_ratio=image_filtering_paras["min_in_out_group_ratio"],
                min_in_group_fraction=image_filtering_paras["min_in_group_fraction"],
                pvals_adj=image_filtering_paras["pvals_adj"],
                label_key=label_key,
                exclude_regions=exclude_regions,
                exclude_mode=exclude_mode,
                print_results=print_results,
            )

            region_image_dic[sample_name] = sample_region_image_dic
            selected_image_dic[sample_name] = image_list

        else:
            image_list = []
            region_image_dic[sample_name] = {}
            selected_image_dic[sample_name] = []

            if image_available and print_results:
                print(
                    f"No image features detected for sample {sample_name} "
                    f"using image_feature_key='{image_feature_key}'."
                )

        # ------------------------------------------------------------
        # Construct sample-specific feature dictionary
        # ------------------------------------------------------------

        features_dic[sample_name] = {
            "gene": gene_list,
            "image": image_list,
        }

    # ============================================================
    # Feature selection summary
    # ============================================================
    if print_results:
        for sample_name in ref_adata_dic:
            print(f"\nSample: {sample_name}")
            print(f"  Selected gene features: {len(features_dic[sample_name]['gene'])}")
            print(f"  Selected image features: {len(features_dic[sample_name]['image'])}")

    return {
        "features_dic": features_dic,
        "region_genes_dic": region_genes_dic,
        "region_image_dic": region_image_dic,
        "selected_gene_dic": selected_gene_dic,
        "selected_image_dic": selected_image_dic,
    }


#========================================== Distance functions ==========================================
def spatial_distance(spatial_df, label_key="label", x_key="x", y_key="y", neighbors=10, scale=True):
    #========== Step 0. Formatting ==========#
    # Turn spatial df into spatial AnnData format
    # spa_adata.X: spatial location
    # spa_adata.obs: labels
    spa_loc = spatial_df[[x_key, y_key]].to_numpy()
    spa_adata = sc.AnnData(spa_loc, dtype=spa_loc.dtype)

    # Use numpy array to avoid index alignment issues
    spa_adata.obs[label_key] = spatial_df[label_key].to_numpy() # take only the values in the exact row order

    #========== Step 1. Pairwise Spatial Distances ==========#
    spa_dists = squareform(pdist(spa_adata.X, metric="euclidean"))

    # Get upper triangular distances, excluding the main diagonal
    upper_dists = spa_dists[np.triu_indices(len(spa_dists), k=1)] # k = 1, excluding the main diagnonal

    #========== Step 2. Infer the Radius ==========#
    spots_num = spatial_df.shape[0]

    # qtl is approximately neighbors / (spots_num - 1)
    qtl = (neighbors * spots_num) / (spots_num ** 2 - spots_num)

    # Prevent invalid quantile values
    qtl = min(qtl, 1.0)

    radius = np.quantile(upper_dists, q=qtl)

    #========== Step 3. Calculate the Spatial Neighborhood Components ==========#
    spa_mask = (spa_dists <= radius).astype(int)

    ave_nbr = np.sum(spa_mask) / len(spa_mask)
    print(f"The average number of neighboring spots: {ave_nbr.round(1)}.")

    labels = spa_adata.obs[label_key]
    tissue_regions = labels.dropna().value_counts().index.tolist()
    tissue_regions = [i for i in tissue_regions if str(i).lower() not in ["nan", "unknown"]]

    regions_num = len(tissue_regions)

    if regions_num == 0:
        raise ValueError("No valid tissue regions found after removing 'nan' and 'unknown'.")

    nbr_df = pd.DataFrame(
        np.zeros((regions_num, regions_num)),
        index=tissue_regions,
        columns=tissue_regions
    )

    for i in range(regions_num):
        name = tissue_regions[i]

        # Safer than converting AnnData obs index from string to int
        region_index = np.where(spa_adata.obs[label_key].to_numpy() == name)[0]

        tmp = spa_mask[region_index, :]
        col_indices = np.nonzero(tmp)[1]

        # Unique neighboring spot indices
        uni_indices = list(set(col_indices))

        nbr_labels = spa_adata[uni_indices].obs[label_key]
        nbr_counts = nbr_labels.dropna().value_counts()

        nbr_regions = nbr_counts.index.tolist()
        nbr_regions = [n for n in nbr_regions if str(n).lower() not in ["nan", "unknown"]]

        common_regions = [n for n in tissue_regions if n in nbr_regions]

        print(f"========== {name} ==========")
        print(nbr_counts)

        for region in common_regions:
            nbr_df.loc[name, region] = nbr_counts[region]

    print("========== Spatial Neighborhood Components ==========")
    print(nbr_df.astype(int))

    #========== Step 4. Transform Neighborhood Components into Spatial Distances ==========#
    # Update zero values
    if (nbr_df == 0).any().any():
        nbr_upd = nbr_df + 1
    else:
        nbr_upd = nbr_df.copy()

    # Calculate percentage of neighborhood components
    nbr_pct = pd.DataFrame(
        np.zeros((regions_num, regions_num)),
        index=tissue_regions,
        columns=tissue_regions
    )

    row_sums = nbr_upd.sum(axis=1)

    for i in range(regions_num):
        rowsum = row_sums.iloc[i] - nbr_upd.iloc[i, i]

        for j in range(regions_num):
            if i == j:
                nbr_pct.iloc[i, j] = 1
            else:
                nbr_pct.iloc[i, j] = nbr_upd.iloc[i, j] / rowsum

    print("========== The percentage of Neighborhood Components ==========")
    print(nbr_pct.round(2))

    # Transform into pairwise spatial distances
    dists = pd.DataFrame(
        np.zeros((regions_num, regions_num)),
        index=tissue_regions,
        columns=tissue_regions
    )

    for i in range(regions_num):
        for j in range(regions_num):
            dists.iloc[i, j] = np.log(1 / (nbr_pct.iloc[i, j] * nbr_pct.iloc[j, i]))

    print("========== Transformed Pairwise Spatial Distances ==========")
    print(dists.round(2))

    # Return unscaled distances
    if scale is False:
        return dists

    # Scale pairwise spatial distance by (x - min) / (max - min)
    scaled_dists = pd.DataFrame(
        np.zeros((regions_num, regions_num)),
        index=tissue_regions,
        columns=tissue_regions
    )

    dists_np = dists.to_numpy()
    upper_elements = dists_np[np.triu_indices(len(dists_np), k=1)]

    dists_min = upper_elements.min()
    dists_max = upper_elements.max()

    if dists_max == dists_min:
        print("All pairwise distances are identical. Scaled distance matrix is all zeros.")
        return dists, scaled_dists

    for i in range(regions_num):
        for j in range(regions_num):
            if i != j:
                scaled_dists.iloc[i, j] = (dists.iloc[i, j] - dists_min) / (dists_max - dists_min)

    print("========== Scaled Pairwise Spatial Distances ==========")
    print(scaled_dists.round(2))

    return dists, scaled_dists


def features_distance(input_adata, features_set, label_key="label", scale=True):
    # Keep feature order, remove duplicates, and keep only features present in input_adata
    features_set = list(dict.fromkeys(features_set))
    available_features = [f for f in features_set if f in input_adata.var.index]

    missing_features = [f for f in features_set if f not in input_adata.var.index]
    if len(missing_features) > 0:
        print(f"Warning: {len(missing_features)} features are not in input_adata.var.index and will be skipped:")
        print(missing_features)

    if len(available_features) == 0:
        raise ValueError("None of the features in features_set are found in input_adata.var.index.")

    adata_sub = input_adata[:, input_adata.var.index.isin(available_features)].copy()

    # Get valid tissue regions
    tissue_regions = adata_sub.obs[label_key].value_counts().index.tolist()
    tissue_regions = [
        i for i in tissue_regions
        if str(i).lower() not in ["nan", "unknown"]
    ]

    if len(tissue_regions) == 0:
        raise ValueError("No valid tissue regions found after removing 'nan' and 'unknown'.")

    # Calculate the mean of features within each tissue region
    features_mean_df = pd.DataFrame(
        np.zeros((len(tissue_regions), len(available_features))),
        index=tissue_regions,
        columns=available_features
    )

    for region in tissue_regions:
        region_mask = (adata_sub.obs[label_key].to_numpy() == region)

        for feature in available_features:
            feature_mask = (adata_sub.var.index.to_numpy() == feature)

            tmp = adata_sub[region_mask, feature_mask]

            if sparse.issparse(tmp.X):
                mean_value = tmp.X.mean(axis=0)
                mean_value = np.asarray(mean_value).ravel()[0]
            else:
                mean_value = np.asarray(tmp.X).mean(axis=0).ravel()[0]

            features_mean_df.loc[region, feature] = mean_value

    print("========== Features Mean within each Tissue Region ==========")
    print(features_mean_df.round(2))

    # Evaluate pairwise distances between tissue regions
    dists = squareform(pdist(features_mean_df, metric="euclidean"))
    dists_df = pd.DataFrame(dists, index=tissue_regions, columns=tissue_regions)

    print("========== Pairwise Distances between Tissue Regions ==========")
    print(dists_df.round(2))

    if scale is False:
        return dists_df

    # Scale pairwise distances by min-max scaling
    scaled_dists_df = pd.DataFrame(
        np.zeros(dists.shape),
        index=tissue_regions,
        columns=tissue_regions
    )

    upper_diag_elements = dists[np.triu_indices(len(dists), k=1)]

    dists_min = upper_diag_elements.min()
    dists_max = upper_diag_elements.max()

    # Handle edge case: all pairwise distances are identical
    if dists_max == dists_min:
        print("Warning: All pairwise distances are identical. Scaled non-diagonal distances are set to 0.")
    else:
        for i in range(len(tissue_regions)):
            for j in range(len(tissue_regions)):
                if i != j:
                    scaled_dists_df.iloc[i, j] = (
                        (dists_df.iloc[i, j] - dists_min) / (dists_max - dists_min)
                    )

    print("========== Scaled Pairwise Distances between Tissue Regions ==========")
    print(scaled_dists_df.round(2))

    return dists_df, scaled_dists_df


def rank_dists(scaled_dists, method="average"):
    """
    Rank pairwise distances in a square symmetric distance matrix.

    Parameters
    ----------
    scaled_dists : pandas.DataFrame
        Square distance matrix with identical row and column labels.
    method : str
        Ranking method passed to scipy.stats.rankdata.
        Common choices: "average", "min", "dense".

    Returns
    -------
    scaled_dists_ranks : pandas.DataFrame
        Symmetric matrix of pairwise distance ranks.
        Diagonal values are 0.
    """

    # Check input shape
    if scaled_dists.shape[0] != scaled_dists.shape[1]:
        raise ValueError("scaled_dists must be a square distance matrix.")

    # Check row/column labels
    if not scaled_dists.index.equals(scaled_dists.columns):
        raise ValueError("scaled_dists must have identical row and column labels.")

    # Check symmetry
    if not np.allclose(scaled_dists.values, scaled_dists.values.T, equal_nan=True):
        raise ValueError("scaled_dists must be symmetric.")

    tissue_regions = scaled_dists.index.tolist()
    n_regions = len(tissue_regions)

    scaled_dists_array = scaled_dists.to_numpy()

    # Extract upper triangle, excluding diagonal
    upper_diag_indices = np.triu_indices(n_regions, k=1)
    upper_diag_elements = scaled_dists_array[upper_diag_indices]

    # Rank distances
    ranks = rankdata(upper_diag_elements, method=method)

    # Use float if method="average"; otherwise int is usually okay
    dtype = float if method == "average" else int

    scaled_dists_ranks = pd.DataFrame(
        np.zeros(scaled_dists.shape),
        index=tissue_regions,
        columns=tissue_regions,
        dtype=dtype
    )

    # Assign ranks symmetrically
    for rank_value, i, j in zip(ranks, upper_diag_indices[0], upper_diag_indices[1]):
        scaled_dists_ranks.iloc[i, j] = rank_value
        scaled_dists_ranks.iloc[j, i] = rank_value

    print("========== The Ranking of Pairwise Distances ==========")
    print(scaled_dists_ranks)

    return scaled_dists_ranks


def multi_modal_distance(
    input_adata, 
    features_dic, 
    w_G=1, 
    w_I=0, 
    w_S=0.5, 
    neighbors=None, 
    shape="hexagon", 
    x_key="x", 
    y_key="y", 
    label_key="label", 
    scale=True,
    return_component_dists=False,
):
    weighted_dists_list = []
    component_dists = {}

    # -----------------------------
    # Gene expression distances
    # -----------------------------
    gene_list = features_dic.get("gene", [])
    compute_gene = w_G > 0 or (return_component_dists and len(gene_list) > 0)
    if compute_gene:
        if len(gene_list) == 0:
            raise ValueError("w_G > 0, but features_dic['gene'] is missing or empty.")

        print("---------------------------------------------------------------------------------")
        print("================================ Gene Expression ================================")
        print("---------------------------------------------------------------------------------")

        gene_distance_result = features_distance(
            input_adata=input_adata,
            features_set=gene_list,
            label_key=label_key,
            scale=scale
        )

        if scale:
            _, weighted_dists_gene = gene_distance_result
        else:
            weighted_dists_gene = gene_distance_result

        component_dists["gene"] = weighted_dists_gene
        if w_G != 0:
            weighted_dists_list.append((w_G, weighted_dists_gene))

    # -----------------------------
    # Image feature distances
    # -----------------------------
    image_list = features_dic.get("image", [])
    compute_image = w_I > 0 or (return_component_dists and len(image_list) > 0)
    if compute_image:

        if len(image_list) == 0:
            raise ValueError("w_I > 0, but features_dic['image'] is missing or empty.")

        print("---------------------------------------------------------------------------------")
        print("================================ Image Features =================================")
        print("---------------------------------------------------------------------------------")

        image_distance_result = features_distance(
            input_adata=input_adata,
            features_set=image_list,
            label_key=label_key,
            scale=scale
        )

        if scale:
            _, weighted_dists_img = image_distance_result
        else:
            weighted_dists_img = image_distance_result

        component_dists["image"] = weighted_dists_img
        if w_I != 0:
            weighted_dists_list.append((w_I, weighted_dists_img))

    # -----------------------------
    # Spatial neighborhood composition distances
    # -----------------------------
    has_spatial_inputs = all(
        col in input_adata.obs.columns for col in (x_key, y_key, label_key)
    )
    compute_spatial = w_S > 0 or (return_component_dists and has_spatial_inputs)
    if compute_spatial:
        if neighbors is None:
            if shape == "hexagon":
                neighbors = 6
            elif shape == "square":
                neighbors = 4
            else:
                raise ValueError(
                    "shape must be either 'hexagon' for Visium data or 'square' for ST data."
                )

        required_cols = [x_key, y_key, label_key]
        missing_cols = [col for col in required_cols if col not in input_adata.obs.columns]

        if len(missing_cols) > 0:
            raise ValueError(f"Missing columns in input_adata.obs: {missing_cols}")

        spatial_df = input_adata.obs[[x_key, y_key, label_key]].copy()

        print("---------------------------------------------------------------------------------------")
        print("================================ Spatial Neighborhoods ================================")
        print("---------------------------------------------------------------------------------------")

        spatial_distance_result = spatial_distance(
            spatial_df=spatial_df,
            label_key=label_key,
            x_key=x_key,
            y_key=y_key,
            neighbors=neighbors,
            scale=scale
        )

        if scale:
            _, weighted_dists_spa = spatial_distance_result
        else:
            weighted_dists_spa = spatial_distance_result

        component_dists["spatial"] = weighted_dists_spa
        if w_S != 0:
            weighted_dists_list.append((w_S, weighted_dists_spa))

    # -----------------------------
    # Check at least one modality is used
    # -----------------------------
    if len(weighted_dists_list) == 0:
        raise ValueError("At least one of w_G, w_I, or w_S must be greater than 0.")

    # -----------------------------
    # Check distance matrix consistency
    # -----------------------------
    first_dists = weighted_dists_list[0][1]
    base_index = first_dists.index
    base_columns = first_dists.columns

    for _, dists in weighted_dists_list[1:]:
        if not dists.index.equals(base_index) or not dists.columns.equals(base_columns):
            raise ValueError(
                "Distance matrices from different modalities have inconsistent row or column labels."
            )

    # -----------------------------
    # Integrate distances
    # -----------------------------
    scaled_dists_all = sum(
        weight * dists for weight, dists in weighted_dists_list
    )

    # -----------------------------
    # Rank the integrated distances
    # -----------------------------
    print("================================ Rank based on overall distance ================================")

    scaled_dists_ranks = rank_dists(scaled_dists_all)

    if return_component_dists:
        return scaled_dists_all, scaled_dists_ranks, component_dists

    return scaled_dists_all, scaled_dists_ranks


def integrate_distance_matrices(dists_dic, spot_counts, fill_diagonal=True, return_pair_weights=False):
    """
    Integrate region-level distance matrices across multiple samples.

    For each region pair, only samples containing both regions are used.
    The sample weights are proportional to sample spot counts and are
    re-normalized among the available samples for that region pair.

    Parameters
    ----------
    dists_dic : dict
        Dictionary of sample-specific distance matrices.

        Example:
            {
                "H1": dists_H1,
                "G2": dists_G2,
                "B1": dists_B1,
            }

        Each value should be a square pandas DataFrame with region names
        as both index and columns.

    spot_counts : dict
        Dictionary mapping sample names to total spot numbers.

        Example:
            {
                "H1": adata1.shape[0],
                "G2": adata2.shape[0],
                "B1": adata3.shape[0],
            }

    fill_diagonal : bool, default=True
        Whether to force the diagonal values to 0.

    return_pair_weights : bool, default=False
        Whether to also return the pairwise sample weights used for each
        region pair.

    Returns
    -------
    integrated_dists : pandas.DataFrame
        Integrated distance matrix across samples.

    pair_weight_dic : dict, optional
        Returned only when return_pair_weights=True.
        Keys are region pairs and values are dictionaries of normalized
        sample weights used for that pair.
    """

    if len(dists_dic) == 0:
        raise ValueError("dists_dic is empty.")

    missing_counts = [s for s in dists_dic if s not in spot_counts]
    if len(missing_counts) > 0:
        raise ValueError(f"Missing spot counts for samples: {missing_counts}")

    # Union of all regions across all samples
    all_regions = sorted(
        set(region for dists in dists_dic.values() for region in dists.index)
    )

    integrated_dists = pd.DataFrame(
        np.nan,
        index=all_regions,
        columns=all_regions,
        dtype=float,
    )

    pair_weight_dic = {}

    for region_i in all_regions:
        for region_j in all_regions:

            available_samples = []
            raw_weights = []
            dist_values = []

            for sample_name, dists in dists_dic.items():
                has_i = region_i in dists.index
                has_j = region_j in dists.columns

                if has_i and has_j:
                    dist_value = dists.loc[region_i, region_j]

                    if pd.notna(dist_value):
                        available_samples.append(sample_name)
                        raw_weights.append(spot_counts[sample_name])
                        dist_values.append(dist_value)

            if len(available_samples) == 0:
                continue

            raw_weights = np.asarray(raw_weights, dtype=float)
            norm_weights = raw_weights / raw_weights.sum()
            dist_values = np.asarray(dist_values, dtype=float)

            integrated_dists.loc[region_i, region_j] = np.sum(
                norm_weights * dist_values
            )

            if return_pair_weights:
                pair_weight_dic[(region_i, region_j)] = dict(
                    zip(available_samples, norm_weights)
                )

    # Keep symmetry stable, in case of tiny numerical differences
    integrated_dists = (integrated_dists + integrated_dists.T) / 2

    if fill_diagonal:
        np.fill_diagonal(integrated_dists.values, 0.0)

    if return_pair_weights:
        return integrated_dists, pair_weight_dic

    return integrated_dists


def multi_sample_distance(
    ref_adata_dic,
    features_dic,
    w_G=1,
    w_I=0,
    w_S=0.5,
    neighbors=None,
    shape="hexagon",
    x_key="x",
    y_key="y",
    label_key="label",
    scale=True,
    return_sample_dists=True,
    return_component_dists=False,
):
    """
    Compute and integrate region-level distances across multiple samples.

    This function first calls multi_modal_distance() for each sample, then
    integrates the resulting distance matrices using sample spot-number weights.

    Parameters
    ----------
    ref_adata_dic : dict
        Dictionary of reference AnnData objects.

        Example:
            {
                "H1": adata1,
                "G2": adata2,
                "B1": adata3,
            }

    features_dic : dict
        Either a single feature dictionary shared by all samples:

            {
                "gene": genes_features,
                "image": image_features,
            }

        or a nested dictionary with sample-specific feature dictionaries:

            {
                "H1": {"gene": genes_H1, "image": image_H1},
                "G2": {"gene": genes_G2, "image": image_G2},
            }

    w_G, w_I, w_S : float
        Modality weights for gene, image, and spatial distances.

    neighbors : int or None
        Spatial neighbor radius. If None, decided by shape.

    shape : {"hexagon", "square"}
        Spatial platform layout.

    x_key, y_key, label_key : str
        Column names in adata.obs.

    scale : bool
        Whether to scale modality-specific distance matrices.

    return_sample_dists : bool
        Whether to return sample-level distance matrices.

    Returns
    -------
    integrated_dists : pandas.DataFrame
        Integrated distance matrix across samples.

    integrated_ranks : pandas.DataFrame
        Rank matrix of the integrated distance matrix.

    sample_dists_dic : dict, optional
        Sample-specific integrated multi-modal distance matrices.
    """

    if len(ref_adata_dic) == 0:
        raise ValueError("ref_adata_dic is empty.")

    sample_dists_dic = {}
    sample_rank_dic = {}
    sample_component_dists_dic = {}
    spot_counts = {}

    for sample_name, adata in ref_adata_dic.items():
        print(f"\n==================== Sample: {sample_name} ====================")

        # Support either shared features_dic or sample-specific features_dic
        # if features_dic contains sample-specific features,
        # uses the features for the current sample.
        if sample_name in features_dic and isinstance(features_dic[sample_name], dict):
            current_features_dic = features_dic[sample_name]
        else:
            current_features_dic = features_dic

        distance_result = multi_modal_distance(
            input_adata=adata,
            features_dic=current_features_dic,
            w_G=w_G,
            w_I=w_I,
            w_S=w_S,
            neighbors=neighbors,
            shape=shape,
            x_key=x_key,
            y_key=y_key,
            label_key=label_key,
            scale=scale,
            return_component_dists=return_component_dists,
        )
        if return_component_dists:
            sample_dists, sample_ranks, component_dists = distance_result
            sample_component_dists_dic[sample_name] = component_dists
        else:
            sample_dists, sample_ranks = distance_result

        sample_dists_dic[sample_name] = sample_dists
        sample_rank_dic[sample_name] = sample_ranks
        spot_counts[sample_name] = adata.shape[0]

    print("\n==================== Integrating distances across samples ====================")

    integrated_dists = integrate_distance_matrices(
        dists_dic=sample_dists_dic,
        spot_counts=spot_counts,
        fill_diagonal=True,
    )

    integrated_ranks = rank_dists(integrated_dists)

    if return_sample_dists and return_component_dists:
        return integrated_dists, integrated_ranks, sample_dists_dic, sample_component_dists_dic

    if return_sample_dists:
        return integrated_dists, integrated_ranks, sample_dists_dic

    if return_component_dists:
        return integrated_dists, integrated_ranks, sample_component_dists_dic

    return integrated_dists, integrated_ranks


#========================================== Tree inference functions ==========================================
def identify_region_pairs(rank_matrix):
    """
    Convert a rank matrix into an ordered list of region pairs.

    Parameters
    ----------
    rank_matrix : pandas.DataFrame
        Square rank matrix.

    Returns
    -------
    region_pairs : list of tuple
        Each tuple is:
        (rank_value, region_1, region_2)
    """

    if rank_matrix.shape[0] != rank_matrix.shape[1]:
        raise ValueError("rank_matrix must be square.")

    if not rank_matrix.index.equals(rank_matrix.columns):
        raise ValueError("rank_matrix must have identical row and column labels.")

    region_names = rank_matrix.columns.tolist()
    regions_num = len(region_names)

    region_pairs = []

    for i in range(regions_num - 1):
        for j in range(i + 1, regions_num):
            rank_value = rank_matrix.iloc[i, j]
            region_pairs.append((rank_value, region_names[i], region_names[j]))

    region_pairs = sorted(region_pairs, key=lambda x: x[0])

    return region_pairs


def pairs_to_nodes(region_pairs, region_names):
    """
    Build hierarchical nodes from ordered region pairs.

    Parameters
    ----------
    region_pairs : list of tuple
        Output from identify_region_pairs().
        Each tuple is:
        (rank_value, region_1, region_2)

    region_names : list
        Original tissue region names.

    Returns
    -------
    node_dic : dict
        node -> list of regions

    hier_dic : dict
        hierarchy level -> list of nodes

    adj_dic : dict
        parent node -> two child nodes
    """

    node_dic = {}
    hier_dic = {}
    adj_dic = {}

    # Initialize leaf nodes
    for i, region in enumerate(region_names):
        node_dic[f"node{i}"] = [region]

    hier_dic[0] = list(node_dic.keys())
    node_num = len(region_names) - 1

    for rank, region0, region1 in region_pairs:
        # Find the current newest node containing region0
        node0_candidates = [
            key for key, value in node_dic.items()
            if region0 in value
        ]
        node0 = node0_candidates[-1]

        # Find the current newest node containing region1
        node1_candidates = [
            key for key, value in node_dic.items()
            if region1 in value
        ]
        node1 = node1_candidates[-1]

        # If they are already in the same node, skip
        if node0 == node1:
            continue

        node0_hier = [
            hier for hier, node_list in hier_dic.items()
            if node0 in node_list
        ][0]

        node1_hier = [
            hier for hier, node_list in hier_dic.items()
            if node1 in node_list
        ][0]

        # Merge two nodes
        node_num += 1
        new_node = f"node{node_num}"
        new_node_hier = max(node0_hier, node1_hier) + 1
        new_node_regions = node_dic[node0] + node_dic[node1]

        node_dic[new_node] = new_node_regions

        if new_node_hier not in hier_dic:
            hier_dic[new_node_hier] = []

        hier_dic[new_node_hier].append(new_node)
        adj_dic[new_node] = [node0, node1]

        # Stop once all regions are merged into one root node
        if len(new_node_regions) == len(region_names):
            break

    return node_dic, hier_dic, adj_dic


def traverse_node_path(adj_dic, start_node, end_node, path=None):
    """
    Traverse from a leaf node to the root node.

    Returns
    -------
    path : list
        Node path from start_node to end_node.
    """

    if path is None:
        path = []

    path = path + [start_node]

    if start_node == end_node:
        return path

    for root_node, leaf_nodes in adj_dic.items():
        if start_node in leaf_nodes:
            if root_node not in path:
                new_path = traverse_node_path(
                    adj_dic=adj_dic,
                    start_node=root_node,
                    end_node=end_node,
                    path=path
                )

                if new_path is not None:
                    return new_path

    return None


def node_to_tree(node_dic, adj_dic, region_names, show=True):
    """
    Convert node/adjacency dictionaries into tree paths.

    Returns
    -------
    path_dic : dict
        leaf node -> path from leaf to root

    tree_list : list
        list of tree paths compatible with bigtree.list_to_tree
    """

    path_dic = {}
    tree_list = []

    root_candidates = [
        key for key, value in node_dic.items()
        if len(value) == len(region_names)
    ]

    if len(root_candidates) != 1:
        raise ValueError(
            "Expected exactly one root node, but found "
            f"{len(root_candidates)} root candidates: {root_candidates}"
        )

    end_node = root_candidates[0]

    for region in region_names:
        start_candidates = [
            key for key, value in node_dic.items()
            if value == [region]
        ]

        if len(start_candidates) != 1:
            raise ValueError(f"Expected one leaf node for region {region}.")

        start_node = start_candidates[0]

        path = traverse_node_path(
            adj_dic=adj_dic,
            start_node=start_node,
            end_node=end_node
        )

        if path is None:
            raise ValueError(f"No path found from {start_node} to {end_node}.")

        path_dic[start_node] = path

        node_tree_path = region
        for i in range(1, len(path)):
            node_tree_path = path[i] + "/" + node_tree_path

        tree_list.append(node_tree_path)

    if show is True:
        try:
            from bigtree import list_to_tree
            tree_str = list_to_tree(tree_list)
            tree_str.hshow()
        except ImportError:
            print("bigtree is not installed. Showing tree paths instead:")
            for path in tree_list:
                print(path)

    return path_dic, tree_list


def build_hier_tree(rank_matrix, show=True):
    """
    Build a hierarchical tree object from a pairwise rank matrix.

    Parameters
    ----------
    rank_matrix : pandas.DataFrame
        Square symmetric rank matrix.
        Rows and columns should be tissue region names.

    show : bool
        Whether to print/show the tree.

    Returns
    -------
    tree : HierTree
        Tree object containing node_dic, hier_dic, adj_dic, path_dic,
        tree_list, region_names, and root_node.
    """

    region_names = rank_matrix.columns.tolist()

    region_pairs = identify_region_pairs(rank_matrix)

    node_dic, hier_dic, adj_dic = pairs_to_nodes(
        region_pairs=region_pairs,
        region_names=region_names
    )

    path_dic, tree_list = node_to_tree(
        node_dic=node_dic,
        adj_dic=adj_dic,
        region_names=region_names,
        show=show
    )

    root_candidates = [
        key for key, value in node_dic.items()
        if len(value) == len(region_names)
    ]

    if len(root_candidates) != 1:
        raise ValueError("Could not identify a unique root node.")

    root_node = root_candidates[0]

    tree = HierTree(
        node_dic=node_dic,
        hier_dic=hier_dic,
        adj_dic=adj_dic,
        path_dic=path_dic,
        tree_list=tree_list,
        region_names=region_names,
        root_node=root_node
    )

    return tree


def make_split_table(tree: HierTree) -> pd.DataFrame:
    """
    Prepare tree split information for downstream hierarchical gene selection
    and hierarchical label transfer.

    Each row represents one binary split:
        parent_node -> child_1 vs child_2

    Parameters
    ----------
    tree : HierTree
        Hierarchical tree object returned by build_hier_tree().

    Returns
    -------
    split_df : pandas.DataFrame
        DataFrame containing parent node, child nodes, and their corresponding regions.
    """

    rows = []

    for parent_node, child_1, child_2 in tree.get_split_pairs(order="root_to_leaf"):
        rows.append({
            "parent_node": parent_node,
            "child_1": child_1,
            "child_2": child_2,
            "child_1_regions": ";".join(tree.get_regions(child_1)),
            "child_2_regions": ";".join(tree.get_regions(child_2)),
            "parent_regions": ";".join(tree.get_regions(parent_node)),
        })

    split_df = pd.DataFrame(rows)

    return split_df


def save_tree_inference_results(
    config_dir: Path,
    tree,
    integrated_dists: pd.DataFrame,
    integrated_ranks: pd.DataFrame,
    sample_dists_dic: dict,
    split_df: pd.DataFrame,
    metadata: dict,
    sample_component_dists_dic: Optional[dict] = None,
) -> None:
    """
    Save all tree inference outputs for one parameter configuration.

    Output structure:
        config_dir/
        ├── integrated_dists.csv
        ├── integrated_ranks.csv
        ├── tree_structure.txt
        ├── tree_structure.png
        ├── tree_object.pkl
        ├── tree_split_table.csv
        ├── metadata.json
        └── sample_dists/
            ├── H1_multi_modal_dists.csv
            ├── G2_multi_modal_dists.csv
            └── E1_multi_modal_dists.csv
        └── sample_component_dists/
            ├── H1_gene_dists.csv
            ├── H1_image_dists.csv
            └── H1_spatial_dists.csv
    """

    config_dir.mkdir(parents=True, exist_ok=True)

    # Save integrated distance and rank matrices
    integrated_dists.to_csv(config_dir / "integrated_dists.csv")
    integrated_ranks.to_csv(config_dir / "integrated_ranks.csv")

    # Save split table for downstream analyses
    split_df.to_csv(config_dir / "tree_split_table.csv", index=False)

    # Save sample-level multi-modal distance matrices
    sample_dist_dir = config_dir / "sample_dists"
    sample_dist_dir.mkdir(parents=True, exist_ok=True)

    for sample_name, sample_dists in sample_dists_dic.items():
        sample_dists.to_csv(sample_dist_dir / f"{sample_name}_multi_modal_dists.csv")

    if sample_component_dists_dic is not None:
        component_dist_dir = config_dir / "sample_component_dists"
        component_dist_dir.mkdir(parents=True, exist_ok=True)
        for sample_name, component_dic in sample_component_dists_dic.items():
            for component_name, component_dists in component_dic.items():
                component_dists.to_csv(
                    component_dist_dir / f"{sample_name}_{component_name}_dists.csv"
                )

    # Save readable tree structure
    tree.save_txt(config_dir / "tree_structure.txt")
    tree.save_png(config_dir / "tree_structure.png")

    # Save full tree object for later reuse
    with open(config_dir / "tree_object.pkl", "wb") as f:
        pickle.dump(tree, f)

    # Save metadata for reproducibility
    with open(config_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def _validate_reweighting_weights(weights: Mapping[str, float]) -> Dict[str, float]:
    """Validate and normalize tree-inference modality weights."""
    if weights is None:
        raise ValueError("weights must be supplied when reusing precomputed distances.")

    unknown = set(weights) - set(_COMPONENT_WEIGHT_KEYS.values())
    if unknown:
        raise ValueError(
            "weights can only contain 'w_G', 'w_I', and 'w_S'. "
            f"Unsupported keys: {sorted(unknown)}."
        )

    normalized = {key: float(weights.get(key, 0.0)) for key in _COMPONENT_WEIGHT_KEYS.values()}
    negative = {key: value for key, value in normalized.items() if value < 0}
    if negative:
        raise ValueError(f"Tree-inference weights must be non-negative: {negative}.")
    if all(value == 0 for value in normalized.values()):
        raise ValueError("At least one of w_G, w_I, or w_S must be greater than 0.")
    return normalized


def _reweight_sample_component_distances(
    sample_component_dists_dic: Mapping[str, Mapping[str, pd.DataFrame]],
    weights: Mapping[str, float],
) -> Dict[str, pd.DataFrame]:
    """Build sample-level multi-modal distances from cached component distances."""
    weights = _validate_reweighting_weights(weights)
    sample_dists_dic = {}

    for sample_name, component_dic in sample_component_dists_dic.items():
        weighted_dists = []
        for component_name, weight_key in _COMPONENT_WEIGHT_KEYS.items():
            weight = weights[weight_key]
            if weight == 0:
                continue
            if component_name not in component_dic:
                raise ValueError(
                    f"Cannot use {weight_key}={weight} for sample {sample_name!r}: "
                    f"cached {component_name!r} component distances are not available. "
                    "Rerun Stage 2 once with the updated package and with the "
                    "corresponding modality available."
                )
            weighted_dists.append((component_name, weight * component_dic[component_name]))

        if not weighted_dists:
            raise ValueError(f"No nonzero weighted components are available for {sample_name!r}.")

        base = weighted_dists[0][1]
        for component_name, dists in weighted_dists[1:]:
            if not dists.index.equals(base.index) or not dists.columns.equals(base.columns):
                raise ValueError(
                    f"Cached component distance matrix {component_name!r} for sample "
                    f"{sample_name!r} has inconsistent row/column labels."
                )

        combined = weighted_dists[0][1].copy()
        for _, dists in weighted_dists[1:]:
            combined = combined + dists
        sample_dists_dic[sample_name] = combined

    return sample_dists_dic


def reweight_tree_inference_result(
    previous_result: Mapping[str, Any],
    weights: Mapping[str, float],
    output_dir=None,
    show_tree: bool = False,
    return_results: bool = True,
):
    """Infer a new tree from cached Stage-2 component distances and new weights.

    This avoids rerunning feature selection and modality-specific distance
    calculations. ``previous_result`` must be produced by
    :func:`infer_hier_tree_pipeline` or ``run_tree_inference_stage`` after
    component-distance caching was added.
    """
    if "sample_component_dists_dic" not in previous_result:
        raise KeyError(
            "previous_result does not contain 'sample_component_dists_dic'. "
            "Rerun Stage 2 once with the current HiCAT version to cache "
            "gene/image/spatial component distances, then reweight from that result."
        )

    metadata = dict(previous_result.get("metadata", {}))
    spot_counts = metadata.get("spot_counts")
    if spot_counts is None:
        raise KeyError(
            "previous_result['metadata'] does not contain 'spot_counts'. "
            "Rerun Stage 2 once with the current HiCAT version before reweighting."
        )
    spot_counts = {sample: float(count) for sample, count in spot_counts.items()}

    weights = _validate_reweighting_weights(weights)
    sample_component_dists_dic = previous_result["sample_component_dists_dic"]
    sample_dists_dic = _reweight_sample_component_distances(
        sample_component_dists_dic=sample_component_dists_dic,
        weights=weights,
    )

    integrated_dists = integrate_distance_matrices(
        dists_dic=sample_dists_dic,
        spot_counts=spot_counts,
        fill_diagonal=True,
    )
    integrated_ranks = rank_dists(integrated_dists)
    tree = build_hier_tree(rank_matrix=integrated_ranks, show=show_tree)
    split_df = make_split_table(tree)

    previous_weights = metadata.get("weights")
    metadata.update(
        {
            "weights": weights,
            "previous_weights": previous_weights,
            "reweighted_from_precomputed_components": True,
            "root_node": tree.root_node,
            "region_names": tree.region_names,
        }
    )

    if output_dir is not None:
        save_tree_inference_results(
            config_dir=Path(output_dir),
            tree=tree,
            integrated_dists=integrated_dists,
            integrated_ranks=integrated_ranks,
            sample_dists_dic=sample_dists_dic,
            sample_component_dists_dic=sample_component_dists_dic,
            split_df=split_df,
            metadata=metadata,
        )

    result = {
        "tree": tree,
        "integrated_dists": integrated_dists,
        "integrated_ranks": integrated_ranks,
        "sample_dists_dic": sample_dists_dic,
        "sample_component_dists_dic": sample_component_dists_dic,
        "split_df": split_df,
        "features_dic": previous_result.get("features_dic"),
        "region_genes_dic": previous_result.get("region_genes_dic"),
        "region_image_dic": previous_result.get("region_image_dic"),
        "selected_gene_dic": previous_result.get("selected_gene_dic"),
        "selected_image_dic": previous_result.get("selected_image_dic"),
        "metadata": metadata,
    }

    if return_results:
        return result
    return tree


def get_first_hierarchy_split_from_tree(hier_tree):
    """
    Get the first hierarchy split directly from a fitted HierTree object.

    The first hierarchy split is defined as the split from the root node
    into its two child nodes.

    Parameters
    ----------
    hier_tree : HierTree
        Fitted hierarchical tree object.

    Returns
    -------
    first_split_info : dict
        Dictionary containing the root split information.

        Keys include:
            parent_node
            child_node_1
            child_node_2
            child_1_regions
            child_2_regions
            split_key_1
            split_key_2
    """

    parent_node = hier_tree.root_node

    if hier_tree.is_leaf(parent_node):
        raise ValueError(
            f"The root_node='{parent_node}' is a leaf node. "
            "No hierarchy split is available."
        )

    children = hier_tree.get_children(parent_node)

    if len(children) != 2:
        raise ValueError(
            f"The root_node='{parent_node}' should have exactly two children, "
            f"but got {len(children)} children: {children}."
        )

    child_node_1, child_node_2 = children

    child_1_regions = hier_tree.get_regions(child_node_1)
    child_2_regions = hier_tree.get_regions(child_node_2)

    split_key_1 = f"{child_node_1}_vs_{child_node_2}"
    split_key_2 = f"{child_node_2}_vs_{child_node_1}"

    first_split_info = {
        "parent_node": parent_node,
        "child_node_1": child_node_1,
        "child_node_2": child_node_2,
        "child_1_regions": child_1_regions,
        "child_2_regions": child_2_regions,
        "split_key_1": split_key_1,
        "split_key_2": split_key_2,
    }

    return first_split_info

'''
# usage:
first_split_info = get_first_hierarchy_split_from_tree(hier_tree)

split_key_1 = first_split_info["split_key_1"]
split_key_2 = first_split_info["split_key_2"]

'''


def infer_hier_tree_pipeline(
    ref_adata_dic,
    label_key="label",
    x_key="x",
    y_key="y",
    image_available=False,
    image_feature_key="hipt",
    gene_filtering_paras=None,
    image_filtering_paras=None,
    weights=None,
    neighbors=None,
    shape="hexagon",
    scale=True,
    show_tree=True,
    output_dir=None,
    return_results=True,
    exclude_regions=("nan", "unknown"),
    exclude_mode="contains",
    print_results=True,
):
    """
    Full tree inference pipeline.

    This function integrates:

        1. Region-specific gene feature selection.
        2. Optional region-specific image feature selection.
        3. Multi-modal distance calculation within each sample.
        4. Multi-sample distance integration across samples.
        5. Hierarchical tree inference.
        6. Tree split table construction.
        7. Optional saving of all tree inference outputs.

    Parameters
    ----------
    ref_adata_dic : dict
        Dictionary of reference-sample-specific AnnData objects.

        Example:
            {
                "H1": adata_H1,
                "G2": adata_G2,
                "E1": adata_E1,
            }

        Each AnnData object may contain both gene features and image features
        in `.var.index`.

    label_key : str
        Column in adata.obs containing tissue region labels.

    x_key, y_key : str
        Columns in adata.obs containing spatial coordinates.

    image_available : bool
        Whether image features are available and should be used.

    image_feature_key : str
        Keyword used to identify image features from adata.var.index.

        Examples:
            image_feature_key="hipt"
            image_feature_key="uni"
            image_feature_key="gigapath"

        Features whose names contain this keyword are treated as image features.
        All other features are treated as gene features.

    gene_filtering_paras : dict or None
        Parameters for selecting region-specific gene features.

        Default:
            {
                "min_fold_change": 1.1,
                "min_in_out_group_ratio": 1,
                "min_in_group_fraction": 0,
                "pvals_adj": 0.05,
                "gene_num": 10,
            }

    image_filtering_paras : dict or None
        Parameters for selecting region-specific image features.

        Default:
            {
                "min_fold_change": 1.1,
                "min_in_out_group_ratio": 1,
                "min_in_group_fraction": 0,
                "pvals_adj": 0.05,
                "gene_num": 5,
            }

    weights : dict or None
        Modality weights.

        Default:
            {
                "w_G": 1,
                "w_I": 1 if image_available else 0,
                "w_S": 1,
            }

    neighbors : int or None
        Number of spatial neighbors. If None, inferred from `shape`.

    shape : {"hexagon", "square"}
        Spatial layout type.

    scale : bool
        Whether to min-max scale modality-specific distance matrices.

    show_tree : bool
        Whether to show the inferred tree.

    output_dir : str, Path, or None
        If provided, save tree inference outputs to this directory.

    return_results : bool
        If True, return a dictionary containing the tree and intermediate results.
        If False, return only the HierTree object.

    exclude_regions : tuple
        Region labels to exclude during feature selection.

    exclude_mode : {"contains", "exact"}
        Whether to exclude labels by substring matching or exact matching.

    print_results : bool
        Whether to print intermediate results.

    Returns
    -------
    results_dic : dict
        Returned when return_results=True.

    tree : HierTree
        Returned when return_results=False.
    """

    # ============================================================
    # 0. Default parameters
    # ============================================================
    if weights is None:
        weights = {
            "w_G": 1,
            "w_I": 1 if image_available else 0,
            "w_S": 1,
        }

    # If image modality is not available, force image weight to 0.
    if image_available is False:
        weights["w_I"] = 0

    if weights["w_G"] == 0 and weights["w_I"] == 0 and weights["w_S"] == 0:
        raise ValueError("At least one of w_G, w_I, or w_S must be greater than 0.")

    if len(ref_adata_dic) == 0:
        raise ValueError("ref_adata_dic is empty.")

    # ============================================================
    # 1. Select gene and image features for each sample
    # ============================================================
    if print_results:
        print("\n============================================================")
        print("Step 1. Select gene and image features for each sample")
        print("============================================================")

    feature_results = select_tree_inference_features(
        ref_adata_dic=ref_adata_dic,
        label_key=label_key,
        image_available=image_available,
        image_feature_key=image_feature_key,
        gene_filtering_paras=gene_filtering_paras,
        image_filtering_paras=image_filtering_paras,
        exclude_regions=exclude_regions,
        exclude_mode=exclude_mode,
        print_results=print_results,
    )   

    features_dic = feature_results["features_dic"]

    region_genes_dic = feature_results["region_genes_dic"]
    region_image_dic = feature_results["region_image_dic"]

    selected_gene_dic = feature_results["selected_gene_dic"]
    selected_image_dic = feature_results["selected_image_dic"]

    #----------------------------------------------------
    # Check feature selection results
    #----------------------------------------------------
    if weights["w_G"] > 0:
        empty_gene_samples = [
            sample_name
            for sample_name in features_dic
            if len(features_dic[sample_name]["gene"]) == 0
        ]

        if len(empty_gene_samples) > 0:
            raise ValueError(
                "w_G > 0, but no gene features were selected for samples: "
                f"{empty_gene_samples}"
            )

    if weights["w_I"] > 0:
        empty_image_samples = [
            sample_name
            for sample_name in features_dic
            if len(features_dic[sample_name]["image"]) == 0
        ]

        if len(empty_image_samples) > 0:
            raise ValueError(
                "w_I > 0, but no image features were selected for samples: "
                f"{empty_image_samples}. "
                "Either check image_feature_key or set w_I=0."
            )

    # ============================================================
    # 2. Multi-modal and multi-sample distance integration
    # ============================================================
    if print_results:
        print("\n============================================================")
        print("Step 2. Computing multi-modal and multi-sample distances")
        print("============================================================")

    (
        integrated_dists,
        integrated_ranks,
        sample_dists_dic,
        sample_component_dists_dic,
    ) = multi_sample_distance(
        ref_adata_dic=ref_adata_dic,
        features_dic=features_dic,
        w_G=weights["w_G"],
        w_I=weights["w_I"],
        w_S=weights["w_S"],
        neighbors=neighbors,
        shape=shape,
        x_key=x_key,
        y_key=y_key,
        label_key=label_key,
        scale=scale,
        return_sample_dists=True,
        return_component_dists=True,
    )

    # ============================================================
    # 3. Infer hierarchical tree
    # ============================================================
    if print_results:
        print("\n============================================================")
        print("Step 3. Inferring hierarchical tree")
        print("============================================================")

    tree = build_hier_tree(
        rank_matrix=integrated_ranks,
        show=show_tree,
    )

    split_df = make_split_table(tree)

    # ============================================================
    # 4. Metadata
    # ============================================================
    metadata = {
        "sample_names": list(ref_adata_dic.keys()),
        "label_key": label_key,
        "x_key": x_key,
        "y_key": y_key,
        "image_available": image_available,
        "image_feature_key": image_feature_key,
        "weights": weights,
        "neighbors": neighbors,
        "shape": shape,
        "scale": scale,
        "gene_filtering_paras": gene_filtering_paras,
        "image_filtering_paras": image_filtering_paras,
        "spot_counts": {
            sample_name: int(ref_adata_dic[sample_name].shape[0])
            for sample_name in ref_adata_dic
        },
        "n_gene_features": {
            sample_name: len(features_dic[sample_name]["gene"])
            for sample_name in features_dic
        },
        "n_image_features": {
            sample_name: len(features_dic[sample_name]["image"])
            for sample_name in features_dic
        },
        "root_node": tree.root_node,
        "region_names": tree.region_names,
    }

    # ============================================================
    # 5. Save outputs
    # ============================================================
    if output_dir is not None:
        output_dir = Path(output_dir)

        save_tree_inference_results(
            config_dir=output_dir,
            tree=tree,
            integrated_dists=integrated_dists,
            integrated_ranks=integrated_ranks,
            sample_dists_dic=sample_dists_dic,
            sample_component_dists_dic=sample_component_dists_dic,
            split_df=split_df,
            metadata=metadata,
        )

    # ============================================================
    # 6. Return
    # ============================================================
    results_dic = {
        "tree": tree,
        "integrated_dists": integrated_dists,
        "integrated_ranks": integrated_ranks,
        "sample_dists_dic": sample_dists_dic,
        "sample_component_dists_dic": sample_component_dists_dic,
        "split_df": split_df,
        "features_dic": features_dic,
        "region_genes_dic": region_genes_dic,
        "region_image_dic": region_image_dic,
        "selected_gene_dic": selected_gene_dic,
        "selected_image_dic": selected_image_dic,
        "metadata": metadata,
    }

    if return_results:
        return results_dic

    return tree
