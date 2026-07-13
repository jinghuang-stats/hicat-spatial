# HiCAT tutorial

**Authors:** Jing Huang, Xueqi Shen, Yoland Smith, Lara Harik, Linghua Wang, Jindan Yu, Michael P. Epstein, and Jian Hu

This tutorial demonstrates the seven-stage HiCAT workflow for hierarchical tissue-region annotation transfer in spatial omics data. It uses three annotated reference sections and one query section with paired gene-expression and image-feature data.

## Contents

1. [Installation](#1-installation)
2. [Input data and project layout](#2-input-data-and-project-layout)
3. [Imports and run settings](#3-imports-and-run-settings)
4. [Stage 1: Preprocessing](#4-stage-1-preprocessing)
5. [Stage 2: Tree inference](#5-stage-2-tree-inference)
6. [Stage 3: Reference selection](#6-stage-3-reference-selection)
7. [Stage 4: Hierarchical feature selection](#7-stage-4-hierarchical-feature-selection)
8. [Stage 5: Clustering configuration](#8-stage-5-clustering-configuration)
9. [Stage 6: Label transfer](#9-stage-6-label-transfer)
10. [Stage 7: Heterogeneity inference](#10-stage-7-heterogeneity-inference)

## Workflow overview

<img src="https://github.com/jinghuang-stats/HiCAT/blob/main/figures/HiCAT_stage_workflow.png" width=85% height=85%>

| Stage | Purpose | Primary result |
|---|---|---|
| 1. Preprocessing | Load and standardize reference and query data | `PreprocessPipelineResult` |
| 2. Tree inference | Infer hierarchical relationships among annotated tissue regions | `tree_result["tree"]` |
| 3. Reference selection | Select suitable reference sections for each query section | `ReferenceSelectionResult.selected_refs_dic` |
| 4. Hierarchical feature selection | Select split-specific features at each level of the hierarchy | `HierarchicalFeatureStageResult` |
| 5. Clustering configuration | Select informative modalities and determine the embedding and clustering strategy | `ClusteringConfigStageResult` |
| 6. Label transfer | Transfer hierarchical tissue-region labels to query sections | `LabelTransferStageResult` |
| 7. Heterogeneity inference | Quantify region-specific heterogeneity and identify heterogeneous subtypes | `ReferenceHeterogeneityResult` |

---

## 1. Installation

Clone the HiCAT repository, create the Conda environment, and install the package in editable mode:

```bash
git clone https://github.com/jinghuang-stats/HiCAT.git
cd HiCAT

conda env create -f HiCAT_package/environment.yml
conda activate hicat

python -m pip install --upgrade pip
python -m pip install -e "./HiCAT_package[image,notebook]"
```

Verify that the package imports correctly:

```python
from importlib.metadata import version

import hicat_spatial

print(version("hicat-spatial"))
print(hicat_spatial.HiCAT)
```

The expected package version for this tutorial is:

```text
0.1.0
```

---

## 2. Input data and project layout

To make the tutorial easy to reproduce, we provide preprocessed reference and query datasets that can be loaded directly into the HiCAT pipeline.

The tutorial datasets have already undergone the following preprocessing steps:

- Extraction of tissue-region labels from pathologist-annotated scribbles
- Extraction of image features from the corresponding H&E-stained histology images
- Optional spatial-resolution enhancement for Spatial Transcriptomics data
- Total-count normalization and log transformation of the molecular data

The preprocessed datasets are available in the [shared folder](https://drive.google.com/drive/folders/1BaqScSe3mxz7JGlixYd-4SSmzHZBOoVb?usp=sharing). They include:

- Three reference sections: `H1`, `G2`, and `E1`
- One query section: `H2`
- Two modalities: gene expression and image features

Download the tutorial datasets and place them in a `data` directory under the HiCAT project root.

### 2.1 Configure project directories and shared settings

Run the following code from the HiCAT project root:

```python
from pathlib import Path

# Project directories
project_root = Path.cwd()
data_dir = project_root / "data"
analysis_root = project_root / "tutorial_results"
preprocess_dir = analysis_root / "01_preprocessing"
checkpoint_root = project_root / "checkpoints"

# Tutorial sections and modalities
reference_sections = ["H1", "G2", "E1"]
query_sections = ["H2"]
avail_modalities = ("Gene", "Image")

# Shared AnnData keys
label_key = "label"
sample_key = "sample"
x_key = "x"
y_key = "y"

# Workflow settings
resolution_level = "enhanced"
image_feature_model = "hipt"
anchor_scenario = "nn_based"
invert_x = False
invert_y = True
random_state = 0

# Create directories used by the tutorial
data_dir.mkdir(parents=True, exist_ok=True)
analysis_root.mkdir(parents=True, exist_ok=True)
checkpoint_root.mkdir(parents=True, exist_ok=True)
```

This tutorial uses enhanced-resolution Spatial Transcriptomics data and nearest-neighbor anchors for within-study transfer. For cross-study or cross-technology transfer, set `anchor_scenario = "quantile_based"`.

### 2.2 Prepare the tutorial datasets

Place the downloaded `.h5ad` files in:

```text
HiCAT/data/
```

Expected reference files:

```text
H1_ref_enhanced_gene.h5ad
G2_ref_enhanced_gene.h5ad
E1_ref_enhanced_gene.h5ad
H1_ref_enhanced_image_features.h5ad
G2_ref_enhanced_image_features.h5ad
E1_ref_enhanced_image_features.h5ad
```

Expected query files:

```text
H2_query_enhanced_gene.h5ad
H2_query_enhanced_image_features.h5ad
```

The project should have the following general structure:

```text
HiCAT/
├── HiCAT_package/
│   ├── environment.yml
│   └── hicat_spatial/                 # HiCAT source code
├── checkpoints/                       # UNI/HIPT checkpoints, when needed
├── data/
│   ├── H1_ref_enhanced_gene.h5ad
│   ├── H1_ref_enhanced_image_features.h5ad
│   ├── G2_ref_enhanced_gene.h5ad
│   ├── G2_ref_enhanced_image_features.h5ad
│   ├── E1_ref_enhanced_gene.h5ad
│   ├── E1_ref_enhanced_image_features.h5ad
│   ├── H2_query_enhanced_gene.h5ad
│   └── H2_query_enhanced_image_features.h5ad
└── tutorial_results/
    ├── 01_preprocessing/
    ├── 02_tree_inference/
    ├── 03_reference_selection/
    ├── 04_hierarchical_features/
    ├── 05_clustering_config/
    ├── 06_label_transfer/
    └── 07_heterogeneity/
```

### 2.3 Use your own data

Although this tutorial uses preprocessed datasets, HiCAT also provides preprocessing utilities for custom reference and query sections. Users may provide:

- Their own annotated reference sections and query sections
- Their own query data together with reference datasets provided by HiCAT
- Reference annotations stored directly in `adata.obs`
- Reference annotations represented by pathologist-generated scribbles on the corresponding tissue images

The preprocessing workflow supports the following tasks.

#### 2.3.1 Reference-annotation preparation

Reference labels can be supplied directly through a column in `reference_adata.obs`:

```python
reference_adata.obs[label_key]
```

When annotations are available only as pathologist-generated scribbles, HiCAT can extract the annotated regions from the associated image and map them to spatial observations.

#### 2.3.2 Image-feature extraction

HiCAT supports image-feature extraction with pathology foundation models, including UNI and HIPT. Download the required checkpoints and place them under `checkpoint_root`.

Based on our empirical experience:

- **UNI** produces a larger feature representation and can capture detailed, intricate morphological patterns.
- **HIPT** can effectively delineate global tissue-region patterns, although its image clusters may contain boundary or background effects. HiCAT provides neighborhood-based boundary refinement to mitigate these effects.

See the checkpoint instructions for [UNI](https://github.com/jinghuang-stats/HiCAT/blob/main/scripts/download_uni_checkpoints.sh) and [HIPT](https://github.com/jinghuang-stats/HiCAT/blob/main/scripts/download_hipt_checkpoints.sh).

#### 2.3.3 Spatial-resolution enhancement (optional)

For Spatial Transcriptomics datasets with approximately 55-µm-radius spots, HiCAT provides an optional enhancement step that increases spatial resolution for downstream analysis.

---

## 3. Imports and run settings

Import the packages and HiCAT functions used throughout the tutorial:

```python
import os
import warnings
warnings.filterwarnings('ignore')
import gc
import numpy as np
import pandas as pd
import matplotlib.colors as clr

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

from hicat_spatial import (
    PreprocessConfig,
    TreeInferenceStageConfig,
    ReferenceSelectionStageConfig,
    SHARED_REFERENCE_KEY,
    HierarchicalFeatureStageConfig,
    ClusteringConfigStageConfig,
    LabelTransferStageConfig,
    HeterogeneityStageConfig,
    run_preprocessing_pipeline,
    construct_tree_reference_adata,
    run_tree_inference_stage,
    run_reference_selection_stage,
    run_hierarchical_feature_stage,
    run_clustering_config_stage,
    build_label_transfer_jobs,
    run_label_transfer_stage,
    run_heterogeneity_stage,
)
```

Define color settings used in the visualizations:

```python
cnt_color = clr.LinearSegmentedColormap.from_list(
    "pink_green",
    ["#3AB370", "#EAE7CC", "#FD1593"],
    N=256,
)

cat_color = ["#000066", "#FEB915", "#C798EE", "#59BE86", "#7495D3", "#F56867", "#15821E", "#3A84E6", "#997273", "#DB4C6C", "#AF5F3C", "#DAB370", "#268785", "#787878", "#D1D1D1",]

cat_color_map = {
    "adipose_tissue": "#74289C",       # Dark purple
    "breast_glands": "#FFCC00",        # Yellow
    "cancer_in_situ": "#FF8000",       # Orange
    "connective_tissue": "#59BE86",    # Green
    "immune_infiltrate": "#66B2FF",    # Blue
    "invasive_cancer": "#FD2B5C",      # Red
    "novel_cluster": "#C2C2C3",        # Gray
}
```

---

## 4. Stage 1: Preprocessing

- **Stage 1** loads the reference and query datasets and constructs standardized objects for the downstream workflow. 
- Because the tutorial data are already preprocessed, this stage loads the existing enhanced-resolution molecular data and image features without repeating normalization, scribble extraction, or feature extraction.

### 4.1 Configure and run preprocessing

```python
preprocess_config = PreprocessConfig(
    data_dir=data_dir,
    preprocess_dir=preprocess_dir,
    reference_sections=reference_sections,
    query_sections=query_sections,
    modalities=avail_modalities,
    # Read files directly from data_dir without copying large inputs.
    raw_file_mode="none",
    # Molecular data are already normalized, log-transformed, and enhanced.
    target_sum=None,
    log1p=False,
    uppercase_features=False,
    # Reference labels already exist in adata.obs[label_key].
    label_color_dict=None,
    # Load precomputed enhanced molecular data.
    reference_gene_template=None,
    query_gene_template=None,
    reference_enhanced_gene_template="{section}_ref_enhanced_gene.h5ad",
    query_enhanced_gene_template="{section}_query_enhanced_gene.h5ad",
    # Load pre-extracted image features at enhanced resolution.
    image_feature_mode="load",
    reference_image_feature_template=None,
    query_image_feature_template=None,
    reference_enhanced_image_feature_template=(
        "{section}_ref_enhanced_image_features.h5ad"
    ),
    query_enhanced_image_feature_template=(
        "{section}_query_enhanced_image_features.h5ad"
    ),
    label_key=label_key,
    x_key=x_key,
    y_key=y_key,
)

preprocess_result = run_preprocessing_pipeline(preprocess_config)
```

### 4.2 Inspect the loaded data

```python
reference_gene_dic = preprocess_result.reference[resolution_level]["Gene"]
query_gene_dic = preprocess_result.query[resolution_level]["Gene"]

print(reference_gene_dic.keys())
print(query_gene_dic.keys())
```

- For raw molecular counts, the default preprocessing uses `target_sum=10000` followed by `log1p=True`. When `.X` is already total-count normalized and log-transformed, use `target_sum=None` and `log1p=False`, as shown above.

- Setting `label_color_dict=None` disables scribble extraction. In this case, every reference section must already contain tissue-region labels in `adata.obs[label_key]`.

- When image features have already been extracted and saved as `.h5ad` files, set `image_feature_mode="load"` instead of rerunning UNI or HIPT extraction.

---

## 5. Stage 2: Tree inference

- **Stage 2** infers a hierarchical tree of annotated tissue regions by integrating molecular, image, and spatial-neighborhood information from the reference sections.

### 5.1 Prepare tree-inference inputs

Use all available reference sections and modalities:

```python
tree_inputs = construct_tree_reference_adata(
    preprocess_result,
    modalities=avail_modalities,
    level=resolution_level,
)
```

You may instead restrict tree inference to selected reference sections based on prior knowledge:

```python
tree_inputs_all = construct_tree_reference_adata(
    preprocess_result,
    modalities=avail_modalities,
    level=resolution_level,
)

selected_tree_sections = ["H1", "E1"]

tree_inputs = {
    section: tree_inputs_all[section]
    for section in selected_tree_sections
}
```

The `modalities` argument can also be restricted to a subset of `avail_modalities` when the tree should be inferred from selected modalities only.

### 5.2 Define feature-filtering parameters

The following settings are general starting points for selecting region-specific marker genes and image-feature dimensions:

```python
gene_filtering_paras = {
    "min_fold_change": 1.1,
    "min_in_out_group_ratio": 1.0,
    "min_in_group_fraction": 0.5,
    "pvals_adj": 0.05,
    "gene_num": 10,
}

image_filtering_paras = {
    "min_fold_change": 1.05,
    "min_in_out_group_ratio": 1.0,
    "min_in_group_fraction": 0.5,
    "pvals_adj": 0.05,
    "gene_num": 5,
}
```

| Parameter | Description | How to tune |
|---|---|---|
| `min_fold_change` | Minimum average target-versus-rest fold change required for feature selection | Increase to retain stronger markers; decrease to allow weaker markers |
| `min_in_out_group_ratio` | Minimum ratio of the target-region detection fraction to the non-target detection fraction | Increase to favor features that are more specific to the target region |
| `min_in_group_fraction` | Minimum fraction of observations in the target region with nonzero feature values | Increase to remove sparse features; decrease to retain rare but potentially meaningful markers |
| `pvals_adj` | Adjusted p-value cutoff for differential-feature testing | Decrease for stricter filtering; increase when too few features are selected |
| `gene_num` | Maximum number of selected features per tissue region; for image data, this refers to image-feature dimensions | Increase to retain more features; decrease for a smaller marker set |

### 5.3 Infer the hierarchy

```python
image_available = "Image" in avail_modalities

tree_result = run_tree_inference_stage(
    ref_adata_dic=tree_inputs,
    config=TreeInferenceStageConfig(
        output_dir=analysis_root / "02_tree_inference",
        label_key=label_key,
        x_key=x_key,
        y_key=y_key,
        image_available=image_available,
        image_feature_key=image_feature_model,
        gene_filtering_paras=gene_filtering_paras,
        image_filtering_paras=image_filtering_paras,
        weights={
            "w_G": 1.0,
            "w_I": 1.0 if image_available else 0.0,
            "w_S": 1.0,
        },
        show_tree=False,
    ),
)

hier_tree = tree_result["tree"]
hier_tree.show()
```

```text
                                          ┌─   adipose_tissue
                    ┌─       node6       ─┤
          ┌─ node9 ─┤                     └─ connective_tissue
          │         └─   breast_glands
─ node10 ─┤                               ┌─   cancer_in_situ
          │         ┌─       node7       ─┤
          └─ node8 ─┤                     └─  invasive_cancer
                    └─ immune_infiltrate

```

<img src="https://github.com/jinghuang-stats/HiCAT/blob/main/tutorial_results/02_tree_inference/tree_structure.png" width=85% height=85%>

The weights control the relative contributions of:

- `w_G`: gene-expression distance
- `w_I`: image-feature distance
- `w_S`: spatial-neighborhood composition distance

Larger values make the inferred tree rely more heavily on the corresponding information source.

### 5.4 Reweight the inferred tree without repeating feature selection

After inspecting the initial tree, you can test alternative modality weights or subset reference sections without repeating feature selection or modality-specific distance calculations:

```python
from hicat_spatial import rerun_tree_inference_with_weights

reweighted_tree_result = rerun_tree_inference_with_weights(
    tree_result=tree_result,
    weights={"w_G": 1.0, "w_I": 0.5, "w_S": 0.5},
    reference_sections=["H1", "E1"],
    output_dir=analysis_root / "02_tree_inference_reweighted",
    show_tree=False,
)

hier_tree = reweighted_tree_result["tree"]
hier_tree.show()
```

Optionally remove intermediate objects to reduce memory usage:

```python
del tree_inputs
gc.collect()
```

---

## 6. Stage 3: Reference selection

- For each query section, **Stage 3** compares its gene-expression distribution with the available reference sections using a Kolmogorov–Smirnov-based similarity score. It then selects reference sections that provide suitable supervision for label transfer.

### 6.1 Define reference-selection parameters

```python
ref_select_paras = {
    "min_fold_change": 1.1,
    "min_in_out_group_ratio": 1.0,
    "min_in_group_fraction": 0.5,
    "pvals_adj": 0.05,
    "gene_num": 10,
    "sort_by": "similarity",
    "selection_mode": "cutoff",
    "alpha": 0.85,
    "min_similarity_level": 0.75,
}
```

- Use `sort_by="weighted"` when you want the ranking to favor reference sections with more comprehensive tissue-region coverage.

- A reference is retained only when it satisfies both of the following conditions:

1. `score >= min_similarity_level`
2. The rule specified by `selection_mode`

When `selection_mode="cutoff"`, HiCAT retains references with `score >= alpha * best_score`, allowing the number of selected references to vary by query section. When `selection_mode="top_k"`, HiCAT retains the top `top_k` ranked references.

### 6.2 Run reference selection

```python
reference_result = run_reference_selection_stage(
    ref_gene_dic=preprocess_result.reference[resolution_level]["Gene"],
    query_gene_dic=preprocess_result.query[resolution_level]["Gene"],
    config=ReferenceSelectionStageConfig(
        output_dir=analysis_root / "03_reference_selection",
        label_key=label_key,
        **ref_select_paras,
    ),
)

selected_refs_dic = reference_result.selected_refs_dic

print(selected_refs_dic)
print(reference_result.to_summary_df())
```

```text
              selected_refs  n_selected_refs   ranked_refs selection_metric selection_mode  alpha  top_k  min_similarity_level
query_section                                                                                                                 
H2                     [H1]                1  [H1, E1, G2]    KS_similarity         cutoff   0.85      3                  0.75
```

- **Stage 3** also stores processed molecular objects used by **Stage 6**:

```python
scaled_reference_gene = reference_result.ref_adata_dic
scaled_query_gene = reference_result.qry_adata_dic
```

---

## 7. Stage 4: Hierarchical feature selection

- **Stage 4** selects split-specific features for each parent-node comparison in the inferred hierarchy. These features are used to guide clustering and anchor detection during hierarchical label transfer.

### 7.1 Define modality-specific filtering parameters

```python
gene_filtering_paras = {
    "label_key": label_key,
    "min_fold_change": 1.1,
    "min_in_out_group_ratio": 1.0,
    "min_in_group_fraction": 0.0,
    "pvals_adj": 0.05,
    "gene_num": 10,
    "two_sides": True,
    "logged": True,
}

image_filtering_paras = {
    "label_key": label_key,
    "min_fold_change": 1.1,
    "min_in_out_group_ratio": 1.0,
    "min_in_group_fraction": 0.0,
    "pvals_adj": 0.05,
    "gene_num": 10,
    "two_sides": True,
    "logged": True,
}

filtering_paras_by_modality = {
    "Gene": gene_filtering_paras,
}

if "Image" in avail_modalities:
    filtering_paras_by_modality["Image"] = image_filtering_paras
```

### 7.2 Run hierarchical feature selection

Estimate features using the entire reference pool:

```python
shared_result_key = SHARED_REFERENCE_KEY
shared_selected_refs_dic = {
    shared_result_key: reference_sections,
}

feature_result = run_hierarchical_feature_stage(
    ref_adata_by_modality=preprocess_result.reference[resolution_level],
    hier_tree=hier_tree,
    selected_refs_dic=shared_selected_refs_dic,
    config=HierarchicalFeatureStageConfig(
        output_dir=analysis_root / "04_hierarchical_features",
        anchor_scenario=anchor_scenario,
        filtering_paras_by_modality=filtering_paras_by_modality,
        keep_raw_results=False,
    ),
)
```

Users can also use the query-specify reference sets selected in **Stage 3** by replacing `shared_selected_refs_dic` with `selected_refs_dic`.


---

## 8. Stage 5: Clustering configuration

- **Stage 5** identifies informative modalities and determines an appropriate dimensionality-reduction and clustering strategy for each query section.

### 8.1 Define modality-selection and visualization settings

```python
clustering_config_paras = {
    "selection_criterion": "both",
    "hard_threshold": 0.25,
    "alpha": 0.85,
    "pcs_num_dic": {"Gene": 30, "Image": 10},
    "random_state": random_state,
}

visualization_config = {
    "plot_modality_clusters": True,
    "plot_dim_reduction_clusters": True,
    "x_key": x_key,
    "y_key": y_key,
    "cat_color": cat_color,
    "fig_size": 10,
    "dpi": 100,
    "invert_x": invert_x,
    "invert_y": invert_y,
}

hipt_boundary_refinement_config = {
    "enabled": True,
    "x_key": x_key,
    "y_key": y_key,
    "bd_num_nbs": 25,
    "smooth_after_reassign": True,
    "smooth_num_nbs": 15,
}
```

| `selection_criterion` | Meaning |
|---|---|
| `"hard"` | Select modalities whose ARI is at least `hard_threshold` |
| `"relative"` | Select modalities whose ARI is at least `alpha * best_ARI` |
| `"both"` | Select modalities that pass both rules; this is the recommended default |

HIPT boundary-refinement parameters:

| Parameter | Description |
|---|---|
| `bd_num_nbs` | Number of nearby non-boundary observations used to reassign observations from a detected HIPT boundary or background cluster |
| `smooth_after_reassign` | Whether to apply spatial smoothing after boundary-cluster reassignment |
| `smooth_num_nbs` | Number of spatial neighbors used during optional post-reassignment smoothing |

### 8.2 Determine informative modalities and dimensionality reduction

```python
stage5_parameters = {
    **clustering_config_paras,
    "visualization_config": visualization_config,
}

if image_feature_model.lower() == "hipt":
    stage5_parameters["hipt_boundary_refinement_config"] = (
        hipt_boundary_refinement_config
    )

embedding_result = run_clustering_config_stage(
    ref_adata_by_modality=preprocess_result.reference[resolution_level],
    feature_stage_result=feature_result,
    config=ClusteringConfigStageConfig(
        output_dir=analysis_root / "05_clustering_config",
        included_modalities=avail_modalities,
        label_key=label_key,
        parameters=stage5_parameters,
    ),
)
```

Boundary-cluster correction is recommended when HIPT image features are used to evaluate modality informativeness.

Inspect the inferred configuration for each query section:

```python
one_result = embedding_result.get_result(shared_result_key)

print("Selected modalities:", one_result.selected_modalities)
print("Dimensionality-reduction method:", one_result.dim_reduction_method)
print("Average modality ARI:", one_result.modality_avg_ari)
print("Average dimension reduction ARI:",one_result.dim_reduction_summary_df)
```

```text
Selected modalities: ['Image']
Dimensionality-reduction method: pca
```

### 8.3 Complete the clustering configuration

HiCAT supports Leiden and K-means clustering.

#### Option A: Leiden clustering

```python
shared_clustering_config = embedding_result.to_clustering_config(
    query_section=shared_result_key,
    clustering_method="leiden",
    resolution=0.2,
    n_neighbors=10,
    random_state=random_state,
    cluster_control={"enabled": False},
)

clustering_configs = {
    shared_result_key: shared_clustering_config,
}
```

Optional `cluster_control` settings can be used to reduce excessive cluster fragmentation:

```python
cluster_control = {
    "enabled": True,
    "max_clusters": "auto",
}
```

| `max_clusters` | Meaning | Suggested use |
|---|---|---|
| Integer| Use a fixed maximum number of clusters | Use when the desired maximum is known |
| `"auto"` | Use `n_parent_regions + 1` when boundary refinement is enabled; otherwise use `n_parent_regions` | Recommended for Image/HIPT-based Leiden clustering |
| `"parent_regions"` | Use exactly `n_parent_regions` | Use for a strict cap without an additional boundary cluster |
| `"legacy"` | Use `2 * n_parent_regions + 1` with boundary refinement or `2 * n_parent_regions` otherwise | Use for setting a  soft limit on the maximum number of clusters |

#### Option B: K-means clustering

K-means is recommended for Visium HD data when computational efficiency and memory usage are priorities.

```python
shared_clustering_config = embedding_result.to_clustering_config(
    query_section=shared_result_key,
    clustering_method="kmeans",
    n_clusters="auto",
    min_clusters=4,
    random_state=random_state,
    cluster_control={"enabled": False},
)

clustering_configs = {
    shared_result_key: shared_clustering_config,
}
```

- Set `n_clusters` to either a fixed integer or `"auto"`. 
- When `n_clusters="auto"`, HiCAT determines the cluster count independently in each hierarchy round. The `min_clusters` parameter is used only in automatic mode and has a default value of `2`.

---

## 9. Stage 6: Label transfer

- **Stage 6** performs hierarchical annotation transfer from the selected reference sections to each query section.

### 9.1 Build query-specific transfer jobs

Each transfer scenario requires a different reference-data structure. The helper below creates query-specific jobs from the results of the previous stages and determines the corresponding Stage 6 scenario.

| Transfer scenario | `scenario` | Reference structure | Recommended use |
|---|---|---|---|
| Single-reference nearest neighbor | `single_ref_nn` | `{modality: AnnData}` | Within-study or same-technology transfer with one selected reference |
| Multi-reference nearest neighbor | `multi_ref_nn` | `{section: {modality: AnnData}}` | Within-study or same-technology transfer with multiple selected references |
| Quantile based | `quantile` | `{modality: {section: AnnData}}`, together with merged references | Cross-study or cross-technology transfer |

- This tutorial uses nearest-neighbor anchors, which are generally appropriate for within-study or same-technology transfer. 
- When **Stage 3** selects one reference, the helper uses `single_ref_nn`; when it selects multiple references, the helper uses `multi_ref_nn`. 
- For cross-study or cross-technology transfer, use `anchor_scenario="quantile_based"`, which produces the `quantile` transfer scenario.

```python
anchor_modalities = [
    modality
    for modality in avail_modalities
    if modality in ("Gene", "Protein")
]

job_setup = build_label_transfer_jobs(
    reference_selection_result=reference_result,
    query_adata_by_modality=preprocess_result.query[resolution_level],
    feature_stage_result=feature_result,
    hier_tree=hier_tree,
    clustering_configs=clustering_configs,
    anchor_scenario=anchor_scenario,
    query_sections=query_sections,
    anchor_modalities=anchor_modalities,
    shared_result_key=shared_result_key,
)

print("Transfer scenario:", job_setup.scenario)
```

HiCAT provides two execution modes:

- `"auto"`: automatically traverses the hierarchy from the root to eligible leaf nodes.
- `"manual"`: returns a session that allows each hierarchy round to be inspected, adjusted, and committed separately. We also provide a step-by-step guide for [manual mode](https://github.com/jinghuang-stats/HiCAT/blob/main/tutorial/label_transfer_manual_mode.md).

### 9.2 Define transfer parameters

#### 9.2.1 Optional gene-subtyping parameters

Gene subtyping can add molecular information when clustering relies primarily on image features. It is useful for separating regions that appear morphologically similar but are molecularly distinct.

```python
gene_subtyping_config = {
    "enabled": True,
    "subtype_gene_num": 10,
    "subtype_min_cluster_prop": 0.1,
    "min_cluster_size": 30,
    "clustering_method": "leiden",
    "resolution": 0.05,
    "n_neighbors": 15,
    "max_subtypes": 2,
    "scale_gene_features": False,
    "smooth_subtypes": False,
    "x_key": x_key,
    "y_key": y_key,
    "subtype_num_nbs": 10,
    "random_state": random_state,
}
```

| Parameter | Description |
|---|---|
| `subtype_gene_num` | Number of target-side and non-target-side hierarchy genes used when `subtype_genes` is not supplied |
| `subtype_min_cluster_prop` | Minimum proportion of active observations required for a parent cluster to be eligible for gene subtyping |
| `min_cluster_size` | Minimum number of observations required before gene subtyping is attempted |
| `max_subtypes` | Maximum desired number of gene-based subtypes within an eligible parent cluster |
| `scale_gene_features` | Whether to standardize selected gene features before subtype clustering |
| `smooth_subtypes` | Whether to spatially smooth subtype labels after gene-based subtyping |

#### 9.2.2 Anchor-detection parameters

```python
anchor_config = {
    "modalities": anchor_modalities,
    "modality_aggregate_mode": "union",
    "knn": 5,
    "max_missing_sections": 1,
    "random_state": random_state,
}
```

| Parameter | Description |
|---|---|
| `modalities` | Molecular modalities used for anchor detection; image features are generally used for clustering rather than anchor matching |
| `modality_aggregate_mode` | Use `"union"` to retain anchors supported by any selected modality or `"shared"` to require support from all selected modalities |
| `knn` | Number of nearest neighbors used during nearest-neighbor anchor detection |
| `max_missing_sections` | Maximum number of selected reference sections that may fail to support an anchor in multi-reference transfer |


#### 9.2.3 Label-assignment parameters

```python
assignment_config = {
    "x_key": x_key,
    "y_key": y_key,
    "min_cluster_spots": 5,
    "min_anchor_pct": 0,
    "allow_novel_clusters": False,
    "prop_diff_cutoff": 5,
    "reassign_novel": True,
    "num_nbs": 25,
}
```

| Parameter | Description |
|---|---|
| `min_cluster_spots` | Minimum number of observations required before a query cluster is assigned a hierarchy label |
| `min_anchor_pct` | Minimum percentage of anchor-supported observations required for assignment to a hierarchy branch |
| `allow_novel_clusters` | Whether clusters with insufficient anchor support may remain novel or unassigned |
| `prop_diff_cutoff` | Minimum difference between competing branch-support proportions required for confident assignment; set to `None` to disable this check |
| `reassign_novel` | Whether novel or unassigned observations are spatially reassigned using nearby non-novel neighbors |
| `num_nbs` | Number of spatial neighbors used when `reassign_novel=True` |

#### 9.2.4 Postprocessing and visualization parameters

```python
postprocess_paras = {
    "x_key": x_key,
    "y_key": y_key,
    "refine": True,
    "num_nbs": 10,
    "cat_color": cat_color_map,
    "fig_size": 13,
    "dpi": 300,
    "invert_x": invert_x,
    "invert_y": invert_y,
}

intermediate_figure_paras = {
    "x_key": x_key,
    "y_key": y_key,
    "base_modality": "Gene",
    "cat_color": None,
    "clustering_cat_color": None,
    "anchor_cat_color": ["#000066", "#FEB915"],
    "assignment_cat_color": None,
    "fig_size": 13,
    "dpi": 100,
    "invert_x": invert_x,
    "invert_y": invert_y,
}
```

### 9.3 Run label transfer (`"auto"` mode)

```python
transfer_stage_result = run_label_transfer_stage(
    jobs=job_setup.jobs,
    config=LabelTransferStageConfig(
        scenario=job_setup.scenario,
        output_dir=analysis_root / "06_label_transfer",
        mode="auto",
        parameters={
            "label_key": label_key,
            "cluster_key": "query_cluster",
            "final_label_key": "hicat_label",
            "unassigned_label": "novel_cluster",
            "min_node_prop": 0.05,
            "min_node_spots": 2,
            "copy": True,  # Set to False when memory is limited.
            "boundary_refinement_config": hipt_boundary_refinement_config,
            "gene_subtyping_config": gene_subtyping_config,
            "anchor_config": anchor_config,
            "assignment_config": assignment_config,
            "print_results": True,
        },
        save_intermediate_figures=True,
        intermediate_figure_parameters=intermediate_figure_paras,
        postprocess=True,
        postprocess_parameters=postprocess_paras,
    ),
)
```

<img src="https://github.com/jinghuang-stats/HiCAT/blob/main/tutorial_results/06_label_transfer/H2/single_ref_nn/refined_predicted_regions.png" width=75% height=75%>

| Parameter | Description |
|---|---|
| `min_node_prop` | Minimum proportion of query observations required for a child node to remain eligible for further splitting |
| `min_node_spots` | Minimum number of query observations required for a child node to remain eligible for further splitting |
| `copy` | Whether to copy intermediate AnnData objects; set to `False` when memory is limited |

Inspect the automatic-mode results:

```python
for query_section in query_sections:
    query_transfer_result = transfer_stage_result.get_result(query_section)
    print(f"-------------------- Query section: {query_section} --------------------")
    print(query_transfer_result.final_labels.value_counts(dropna=False))
    print("Complete:", query_transfer_result.is_complete)
```

```text
-------------------- Query section: H2 --------------------
hicat_label
connective_tissue    8449
adipose_tissue       4798
invasive_cancer      4209
cancer_in_situ       3456
breast_glands        1249
immune_infiltrate     667
Name: count, dtype: int64
Complete: True
```

Optional steps to reduce memory usage:
```
del jobs
gc.collect()
```

---

## 10. Stage 7: Heterogeneity inference

- **Stage 7** quantifies region-specific heterogeneity across reference sections. 
- For regions classified as heterogeneous, it can identify subtypes automatically and select subtype-specific marker genes to support biological interpretation.

### 10.1 Define heterogeneity and subtype settings

```python
hetero_score_configs = {
    "region_gene_num": 10,
    "selection_method": "threshold",
    "hetero_threshold": 0.5,
    "low_exp_thres": 0.02, # set as None to skip low-expression filtering
    "n_perm": 100,
}

hetero_subtype_configs = {
    "run_subtype": False,
    "min_region_spots": 10,
    "min_cluster_fraction": 0.05,
    "pcs_num": 20,
    "section_cluster_method": "leiden_clusters",
    "leiden_res": 0.01,
    "n_neighbors": 15,
    "min_fold_change": 1.1,
    "min_in_out_group_ratio": np.nextafter(1.0, np.inf), # close to > 1
    "min_in_group_fraction": 0.5,
    "pvals_adj": 0.05,
    "overlap_cutoff": 1,
    "section_gene_num": 15,
    "set_shared_clusters_num": 2,
    "merged_gene_num": 15,
    "individual_gene_num": 35,
    "random_state": random_state,
}

hetero_fig_paras = {
    "cat_color": cat_color,
    "cnt_color": cnt_color,
    "x_key": x_key,
    "y_key": y_key,
    "fig_size": 15,
    "dpi": 150,
    "invert_x": invert_x,
    "invert_y": invert_y,
}
```

| Parameter | Description | Used in |
|---|---|---|
| `region_gene_num` | Maximum number of region-specific marker genes selected per tissue region and reference section | Heterogeneity scoring |
| `selection_method` | Use `"threshold"` to retain regions with scores at or above `hetero_threshold`, or `"top_k"` to retain the highest-ranked regions | Heterogeneous-region selection |
| `hetero_threshold` | Heterogeneity-score cutoff used when `selection_method="threshold"` | Heterogeneous-region selection |
| `low_exp_thres` | Optional low-expression filtering threshold before heterogeneity analysis. If set, genes detected in fewer than this fraction of observations in a reference section are removed. Use None to skip filtering. | Pre-filtering before heterogeneity scoring and subtype analysis
| `n_perm` | Number of permutations used to compute permutation-adjusted silhouette or stability scores | Heterogeneity scoring |
| `min_region_spots` | Minimum number of observations required for a tissue region to be evaluated within a reference section | Scoring and subtype analysis |
| `pcs_num` | Number of principal components used before subtype clustering | Subtype discovery |
| `min_cluster_fraction` | Minimum subtype-cluster proportion required for marker-gene selection | Subtype marker selection |
| `overlap_cutoff` | Minimum number of reference sections in which a section-level subtype marker must appear | Shared subtype marker selection |
| `section_gene_num` | Maximum number of subtype markers selected per cluster within each reference section | Section-level marker selection |
| `set_shared_clusters_num` | Optional fixed number of shared heterogeneous subtype clusters. If `None`, HiCAT estimates the number of shared subtype clusters automatically from the merged heterogeneous-region data. If an integer is provided, that value is used directly for shared subtype clustering. | Shared subtype discovery |
| `merged_gene_num` | Maximum number of subtype markers selected per shared subtype from merged references | Shared marker selection |
| `individual_gene_num` | Maximum number of subtype markers selected per shared subtype and reference section before overlap filtering | Per-section marker selection |


### 10.2 Run heterogeneity inference

```python
heterogeneity_result = run_heterogeneity_stage(
    ref_gene_dic=preprocess_result.reference[resolution_level]["Gene"],
    config=HeterogeneityStageConfig(
        output_dir=analysis_root / "07_heterogeneity",
        dataset_name="tutorial_dataset",
        parameters={
            "label_key": label_key,
            "sample_key": sample_key,
            **hetero_score_configs,
            **hetero_subtype_configs,
            **hetero_fig_paras,
            "print_results": True,
        },
    ),
)
```

Inspect the selected heterogeneous regions and the summary table:

```python
print(heterogeneity_result.hetero_summary["hetero_score_sca"])
print("Selected heterogeneous regions:", heterogeneity_result.selected_regions)
```

```text
cancer_in_situ       1.000000
immune_infiltrate    0.861509
invasive_cancer      0.620385
connective_tissue    0.496405
breast_glands        0.276758
adipose_tissue       0.000000
Name: hetero_score_sca, dtype: float64

Selected heterogeneous regions: ['cancer_in_situ', 'immune_infiltrate', 'invasive_cancer']
```
