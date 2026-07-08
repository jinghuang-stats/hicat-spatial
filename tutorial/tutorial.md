<h1><center>HiCAT Tutorial</center></h1>

<center>Author: Jing Huang, Xueqi Shen, Yoland Smith, Lara Harik, Linghua Wang, Jindan Yu, Michael P. Epstein* and Jian Hu*

### Outline
1. [Installation](#1-installation)
2. [Read in data](#2-input-data-and-folder-layout)
3. [Import modules](#3-imports-and-run-settings)
4. [Stage 1: Preprocessing](#stage-1-preprocessing)
5. [Stage 2: Tree inference](#stage-2-tree-inference)
6. [Stage 3: Reference selection](#stage-3-reference-selection)
7. [Stage 4: Hierarchical feature selection](#stage-4-hierarchical-feature-selection)
8. [Stage 5: Determine clustering configurations](#stage-5-determine-clustering-configurations)
9. [Stage 6: Label transfer](#stage-6-label-transfer)
10. [Stage 7: Heterogeneity inference](#stage-7-heterogeneity-inference)

### 1. Installation
To install HiCAT package you must make sure that your python version is over 3.9. If you don’t know the version of python you can check it by:


```python
import platform
platform.python_version()
```

<br>
Create an environment and install the package from the GitHub/local folder:
```python
conda create -n hicat python=3.11 -y
conda activate hicat

git clone https://github.com/jinghuang-stats/HiCAT.git
cd HiCAT
python -m pip install --upgrade pip
python -m pip install -e ".[notebook]

```

Install the image extras if extracting image features (e.g., by HIPT or UNI)
```python
python -m pip install -e ".[image,notebook]

```

Check the package imports and version
```python
from importlib.metadata import version
import hicat_spatial

print(version("hicat-spatial"))
print(hicat_spatial.HiCAT)

```

Expected version for this tutorial:
```text
0.1.0
```

### 2. Input data and folder layout
- Toydata for this tutorial are made available at the [shared folder](https://drive.google.com/drive/folders/1BaqScSe3mxz7JGlixYd-4SSmzHZBOoVb?usp=share_link)
- Create one flat raw-data folder (the tutorial uses ``data/``) to include raw reference and query gene adata as well as associated images if extracting image features is needed, e.g.,
```text
data/
	H1_ref_gene_raw.h5ad
	G2_ref_gene_raw.h5ad
	E1_ref_gene_raw.h5ad
	H2_query_gene_raw.h5ad

	H1_image.jpg
	G2_image.jpg
	E1_image.jpg
	H2_image.jpg

	H1_annotated_image.jpg
	G2_annotated_image.jpg
	E1_annotated_image.jpg

```
- We also provided precomputed reference information available in the [results folder](https://drive.google.com/drive/folders/1dvCbkgciSRbCc7SAMad0dBVa2tEQtMI-?usp=sharing). Users can directly use these files to perform label transfer for breast cancer tissue type, without having to construct their own reference datasets or generate reference information from scratch. 
  The precomputed reference information includes:
  - preprocessed reference ``gene adata`` and associated ``image adata``, with annotations included in ``adata.obs[label_key]``
  - ``hier_tree``: inferred hierarchical tree structure
  - ``multimodal_features``: identified multi-modal hierarchical features set to guide each hierarchical split
- Intermediate results are saved in the [figures folder](https://github.com/jinghuang-stats/HiCAT/blob/main/tutorial/figures).
<br>

### 3. Imports and run settings
```python
from pathlib import Path

import pandas as pd

from hicat_spatial import (
    ClusteringConfigStageConfig,
    HeterogeneityStageConfig,
    HierarchicalFeatureStageConfig,
    LabelTransferStageConfig,
    PreprocessConfig,
    ReferenceSelectionStageConfig,
    TreeInferenceStageConfig,
    construct_tree_reference_adata,
    load_stage_result,
    run_clustering_config_stage,
    run_heterogeneity_stage,
    run_hierarchical_feature_stage,
    run_label_transfer_stage,
    run_preprocessing_pipeline,
    run_reference_selection_stage,
    run_tree_inference_stage,
)
```

```python
analysis_root = Path("tutorial_results")
data_dir = Path("data")
preprocess_dir = analysis_root / "01_preprocessing"

reference_sections = ["H1", "G2", "E1"]
query_sections = ["H2"]

label_key = "label"
x_key = "pixel_x"
y_key = "pixel_y"
```

Create the raw-data folder:

```python
data_dir.mkdir(parents=True, exist_ok=True)
```

Then copy your raw `.h5ad` and image files into `data_dir` before running Stage 1. HiCAT will create the preprocessing folder tree automatically.

### Stage 1: Preprocessing
Stage 1 reads raw files, performs normalization and log-transformation, extracts scribble labels, performs optional gene expression enhancement if Spatial Transcriptomics data, and saves preprocessed objects.

For annotations, `label_color_dict=None` means Stage 1 will not extract scribbles from annotated images. In that case, each reference section should already contain labels in `adata.obs[label_key]`, for example `adata.obs["label"]`.

If image features were already extracted and saved as `.h5ad` files, use `image_feature_mode="load"` instead of re-running UNI/HIPT extraction.

```python
preprocess_config = PreprocessConfig(
    data_dir=data_dir,
    preprocess_dir=preprocess_dir,
    reference_sections=reference_sections,
    query_sections=query_sections,
    modalities=("Gene","Image"),
    raw_file_mode="copy",
    target_sum=10_000,
    label_key=label_key,
    x_key=x_key,
    y_key=y_key,
)

preprocess_result = run_preprocessing_pipeline(preprocess_config)
```

Access the processed objects:

```python
reference_gene = preprocess_result.reference["enhanced"]["Gene"]
query_gene = preprocess_result.query["enhanced"]["Gene"]

ref_h1 = preprocess_result.get_adata("reference", "enhanced", "Gene", "H1")
query_h2 = preprocess_result.get_adata("query", "enhanced", "Gene", "H2")

print(reference_gene.keys())
print(query_h2)
```

### Stage 2: Tree inference
Stage 2 builds a hierarchy of annotated reference regions.

```python
tree_inputs = construct_tree_reference_adata(
    preprocess_result,
    modalities=("Gene","Image"),
    level="enhanced",
)

tree_result = run_tree_inference_stage(
    ref_adata_dic=tree_inputs,
    config=TreeInferenceStageConfig(
        output_dir=analysis_root / "02_tree_inference",
        label_key=label_key,
        x_key=x_key,
        y_key=y_key,
        image_available=True,
        weights={"w_G": 1.0, "w_I": 1.0, "w_S": 1.0},
        show_tree=False,
    ),
)

hier_tree = tree_result["tree"]
split_table = tree_result["split_df"]

print(hier_tree.get_internal_nodes())
print(split_table)
```

### Stage 3: Reference selection
Stage 3 selects the suitable and compatible reference sections for the target query section to provide the matched supervision

```python
reference_result = run_reference_selection_stage(
    ref_gene_dic=preprocess_result.reference["enhanced"]["Gene"],
    query_gene_dic=preprocess_result.query["enhanced"]["Gene"],
    config=ReferenceSelectionStageConfig(
        output_dir=analysis_root / "03_reference_selection",
        label_key=label_key,
        selection_mode="cutoff",
        alpha=0.85,
    ),
)

selected_refs_dic = reference_result.selected_refs_dic

print(reference_result.to_summary_df())
print(reference_result.get_selected_refs("H2"))
```

Stage 3 also keeps processed molecular objects that are useful for Stage 6:

```python
scaled_reference_gene = reference_result.ref_adata_dic
scaled_query_gene = reference_result.qry_adata_dic
```

### Stage 4: Hierarchical feature selection
Stage 4 selects split-specific features for every query-specific reference set.

```python
feature_result = run_hierarchical_feature_stage(
    ref_adata_by_modality=preprocess_result.reference["enhanced"],
    hier_tree=hier_tree,
    selected_refs_dic=selected_refs_dic,
    config=HierarchicalFeatureStageConfig(
        output_dir=analysis_root / "04_hierarchical_features",
        anchor_scenario="nn_based",
        filtering_paras_by_modality={
            "Gene": {
                "label_key": label_key,
                "pvals_adj": 0.05,
                "min_in_out_group_ratio": 1.0,
                "min_in_group_fraction": 0.0,
                "min_fold_change": 1.15,
                "gene_num": 10,
                "logged": True,
            }
        },
        count_num=1,
    ),
)

gene_features_h2 = feature_result.get_modality_result("H2", "Gene")
multimodal_features_h2 = feature_result.get_multimodal_result("H2")

print(gene_features_h2.available_parent_nodes())
```

### Stage 5: Determine clustering configurations
Stage determines the informative modalities and embedding choices. It does not choose the final clustering method, please specify ``KMeans`` or ``Leiden`` afterwards.
```python
embedding_result = run_clustering_config_stage(
    ref_adata_by_modality=preprocess_result.reference["enhanced"],
    feature_stage_result=feature_result,
    config=ClusteringConfigStageConfig(
        output_dir=analysis_root / "05_clustering_config",
        included_modalities=("Gene","Image"),
        features_format="auto",
        evaluate_all_nodes=False,
        label_key=label_key,
        parameters={
            "candidate_methods": ("pca", "selected_features"),
            "selection_criterion": "both",
            "hard_threshold": 0.5,
            "alpha": 0.85,
            "pcs_num_dic": {"Gene": 30, "Image": 10},
            "default_pcs_num": 30,
            "random_state": 0,
        },
    ),
)

print(embedding_result.get_result("H2").summary())
```

For a binary hierarchy split, KMeans with two clusters is a simple starting
point:

```python
clustering_configs = {
    query_section: embedding_result.to_clustering_config(
        query_section=query_section,
        clustering_method="kmeans",
        n_clusters=2,
        random_state=0,
    )
    for query_section in query_sections
}

print(clustering_configs["H2"])
```

Leiden is another option:

```python
leiden_config = embedding_result.to_clustering_config(
    query_section="H2",
    clustering_method="leiden",
    resolution=0.5,
    n_neighbors=15,
)
```

### Stage 6: Label transfer
Stage 6 needs explicit jobs because each transfer scenario expects a different
reference dictionary layout.

| Scenario | `scenario` value | Reference dictionary structure |
|---|---|---|
| Single-reference NN | `single_ref_nn` | `{modality: AnnData}` |
| Multi-reference NN | `multi_ref_nn` | `{section: {modality: AnnData}}` |
| Quantile based | `quantile` | `{modality: {section: AnnData}}` plus merged references |

This tutorial uses multi-reference nearest-neighbor transfer.

```python
jobs = {}

for query_section in query_sections:
    selected_refs = reference_result.get_selected_refs(query_section)

    jobs[query_section] = {
        "ref_section_list": selected_refs,

        # Used for anchor detection.
        # Multi-ref NN expects: {ref_section: {modality: AnnData}}
        "ref_adata_sca_dic": {
            ref_section: {
                "Gene": reference_gene_for_anchor[ref_section],
            }
            for ref_section in selected_refs
        },

        # Used for query clustering, final labels, outputs, and optional image refinement.
        "query_adata_dic": {
            "Gene": preprocess_result.query["enhanced"]["Gene"][query_section],
            "Image": preprocess_result.query["enhanced"]["Image"][query_section],
        },

        # Used for anchor detection.
        # Usually only molecular modalities are needed here.
        "query_adata_sca_dic": {
            "Gene": query_gene_for_anchor[query_section],
        },

        "hier_tree": hier_tree,

        "gene_feature_results": feature_result.get_modality_result(
            query_section, "Gene"
        ),

        "image_feature_results": feature_result.get_modality_result(
            query_section, "Image"
        ),

        "clustering_config": clustering_configs[query_section],
    }
```

Run automatic transfer:

```python
transfer_stage_result = run_label_transfer_stage(
    jobs=jobs,
    config=LabelTransferStageConfig(
        scenario="multi_ref_nn",
        output_dir=analysis_root / "06_label_transfer",
        mode="auto",
        parameters={
            "label_key": label_key,
            "final_label_key": "hicat_label",
            "unassigned_label": "novel_cluster",
            "print_results": True,
        },
        postprocess=True,
        postprocess_parameters={
            "x_key": x_key,
            "y_key": y_key,
            "refine": True,
            "num_nbs": 25,
        },
    ),
)

h2_result = transfer_stage_result.get_result("H2")

print(h2_result.final_labels.value_counts(dropna=False))
print(h2_result.round_summary())
print(h2_result.is_complete())
```

Annotated query objects are stored in the result:

```python
annotated_query_gene = h2_result.query_adata_dic["Gene"]
print(annotated_query_gene.obs["hicat_label"].head())
```

### Stage 7: Heterogeneity inference
Stage 7 evaluates the region-specific heterogeneity levels across reference sections. And for those heterogeneous regions, it further identified heterogeneity subtypes within it.

```python
heterogeneity_result = run_heterogeneity_stage(
    ref_gene_dic=preprocess_result.reference["spot"]["Gene"],
    config=HeterogeneityStageConfig(
        output_dir=analysis_root / "07_heterogeneity",
        dataset_name="tutorial_dataset",
        parameters={
            "label_key": label_key,
            "sample_key": "sample",
            "selection_method": "threshold",
            "hetero_threshold": 0.5,
            "run_subtype": False,
            "n_perm": 200,
            "random_state": 0,
        },
    ),
)

print(heterogeneity_result.selected_regions)
print(heterogeneity_result.hetero_summary)
```


