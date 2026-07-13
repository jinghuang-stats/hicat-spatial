# HiCAT 

**From unsupervised clustering to atlas-guided annotation in cohort-scale spatial omics with HiCAT**

[![Python 3.9–3.11](https://img.shields.io/badge/python-3.9--3.11-blue.svg)](https://www.python.org/)
[![Version 0.1.0](https://img.shields.io/badge/version-0.1.0-blue.svg)](HiCAT_package/pyproject.toml)
[![Development status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](HiCAT_package/pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](HiCAT_package/pyproject.toml)
[![bioRxiv](https://img.shields.io/badge/preprint-bioRxiv-b31b1b.svg)](https://www.biorxiv.org/content/10.64898/2026.05.27.728266v1)

> **Manuscript:** *From unsupervised clustering to atlas-guided annotation in cohort-scale spatial omics with HiCAT*
>
> Jing Huang, Xueqi Shen, Yoland Smith, Lara Harik, Linghua Wang, Jindan Yu, Michael P. Epstein, and Jian Hu

HiCAT is a supervised computational framework for transferring pathologist-informed tissue-region annotations and characterizing region-level heterogeneity in multimodal spatial omics data. It combines annotated reference sections, molecular measurements, histology-derived image features, and spatial context to generate consistent and biologically informed annotations across samples.

By organizing tissue regions into a hierarchy and transferring labels from matched references to query sections, HiCAT supports cohort-scale construction of annotated spatial atlases and downstream analyses of disease-associated tissue organization and intra-region heterogeneity.

![HiCAT workflow](figures/HiCAT_workflow_most_upd.png)

## Key capabilities

HiCAT provides tools to:

- prepare pathologist-informed reference annotations, including annotations extracted from tissue-image scribbles
- integrate gene-expression, image-feature, protein, and spatial-neighborhood information
- infer hierarchical relationships among annotated tissue regions
- select query-matched reference sections for supervised annotation transfer
- identify hierarchy-specific molecular and imaging features
- determine informative modalities, dimension-reduction methods, and clustering configurations
- transfer tissue-region labels within studies or across studies and technologies
- quantify region-specific heterogeneity and identify heterogeneous subtypes for cohort-level interpretation

HiCAT is designed for spatial transcriptomics and multimodal spatial assays, including Spatial Transcriptomics, 10x Visium, 10x Visium HD, 10x Xenium, and assays with paired transcriptomic and protein measurements.

## Workflow

HiCAT is organized as a seven-stage workflow. Each stage can be run independently, or the stages can be coordinated through `HiCATWorkflowConfig` and `run_hicat_workflow`.

| Stage | Purpose | Main result |
|---|---|---|
| 1. Preprocessing | Load, standardize, and optionally preprocess reference and query data | `PreprocessPipelineResult` |
| 2. Tree inference | Infer hierarchical relationships among annotated tissue regions | Inferred hierarchical tree |
| 3. Reference selection | Select suitable reference sections for each query section | `ReferenceSelectionResult` |
| 4. Hierarchical feature selection | Select split-specific features at each level of the hierarchy | `HierarchicalFeatureStageResult` |
| 5. Clustering configuration | Select informative modalities and determine embedding and clustering settings | `ClusteringConfigStageResult` |
| 6. Label transfer | Transfer hierarchical tissue-region labels to query sections | `LabelTransferStageResult` |
| 7. Heterogeneity inference | Quantify regional heterogeneity and identify heterogeneous subtypes | `ReferenceHeterogeneityResult` |

Stage outputs are saved in numbered directories and can be reloaded independently. This makes it possible to inspect intermediate results, revise selected settings, and resume an analysis without rerunning the full workflow.

## Installation

### Requirements

- Python `3.9`, `3.10`, or `3.11`
- Conda or another environment manager is recommended
- Git

The package distribution name is `hicat-spatial`, while the Python import name is `hicat_spatial`.

### Recommended installation

Clone the repository and create the provided Conda environment:

```bash
git clone https://github.com/jinghuang-stats/HiCAT.git
cd HiCAT

conda env create -f HiCAT_package/environment.yml
conda activate hicat

python -m pip install --upgrade pip
python -m pip install -e "./HiCAT_package[image,notebook]"
```

This installs HiCAT in editable mode together with the dependencies used for image-feature extraction and Jupyter notebooks.

### Lighter installations

Install the core package only:

```bash
python -m pip install -e "./HiCAT_package"
```

Install the core package with notebook support:

```bash
python -m pip install -e "./HiCAT_package[notebook]"
```

Install the core package with image-feature extraction support:

```bash
python -m pip install -e "./HiCAT_package[image]"
```

### Verify the installation

```python
from importlib.metadata import version

import hicat_spatial

print(version("hicat-spatial"))
print(hicat_spatial.HiCAT)
```

The expected version for the current repository is `0.1.0`.

## Getting started

The recommended entry point for new users is the complete [step-by-step tutorial](tutorial/tutorial.md). It demonstrates the seven-stage workflow using three annotated reference sections and one query section with gene-expression and image-feature data.

HiCAT supports two complementary usage patterns.

### Stage-by-stage execution

Use the individual configuration classes and stage runners when you want to inspect intermediate outputs or adjust settings between stages:

```python
from hicat_spatial import (
    PreprocessConfig,
    TreeInferenceStageConfig,
    ReferenceSelectionStageConfig,
    HierarchicalFeatureStageConfig,
    ClusteringConfigStageConfig,
    LabelTransferStageConfig,
    HeterogeneityStageConfig,
    run_preprocessing_pipeline,
    run_tree_inference_stage,
    run_reference_selection_stage,
    run_hierarchical_feature_stage,
    run_clustering_config_stage,
    run_label_transfer_stage,
    run_heterogeneity_stage,
)
```

### Coordinated workflow execution

Use the high-level workflow interface when the stage configurations and label-transfer jobs have already been prepared:

```python
from hicat_spatial import HiCATWorkflowConfig, run_hicat_workflow

workflow_config = HiCATWorkflowConfig(
    preprocessing=preprocess_config,
    output_root="results",
    tree_inference=tree_config,
    reference_selection=reference_config,
    hierarchical_features=feature_config,
    clustering_config=clustering_config,
    label_transfer=label_transfer_config,
    heterogeneity=heterogeneity_config,
    tree_modalities=("Gene", "Image"),
)

workflow_result = run_hicat_workflow(
    config=workflow_config,
    label_transfer_jobs=label_transfer_jobs,
)
```

The tutorial provides complete configuration examples and shows how to inspect the result object produced by each stage.

## Input data

HiCAT uses annotated reference sections to guide annotation of one or more query sections.

### Reference sections

Reference data should be stored as `AnnData` objects. Tissue-region annotations can be supplied in either of the following forms:

- a categorical label column in `adata.obs`, such as `adata.obs["label"]`
- or pathologist-generated scribbles on the corresponding tissue image, which can be mapped to spatial observations during preprocessing.

### Query sections

Query sections do not require tissue-region labels. Depending on the analysis, they may contain:

- gene-expression measurements
- spatial coordinates stored in `adata.obs`
- precomputed image features or an associated histology image
- protein measurements (if available)

For multimodal analyses, use consistent observation identifiers and spatial-coordinate columns across modalities.

### Image-feature extraction

HiCAT supports pathology image features generated with HIPT and UNI. Install the `image` extra before extracting image features. Checkpoint helper scripts are provided in the `scripts/` directory:

```bash
bash scripts/download_hipt_checkpoints.sh
bash scripts/download_uni_checkpoints.sh
```
- For convenience, the pretrained checkpoints compatible with HiCAT framework are also available through the shared links for [HIPT](https://drive.google.com/drive/folders/1N7oYToW4c1H5AN1mGYWRl-TgkfCLgE36?usp=sharing) and [UNI](https://drive.google.com/drive/folders/1s2NFHuk5c9x1J9y1-nK1z3y4uYQZYpkm?usp=sharing).
- Depending on the model, downloading or using their checkpoints may require separate access approval, authentication, or acceptance of the original model license and terms of use. Please review and comply with the requirements specified by the respective model developers.
- Image-feature extraction can be skipped when precomputed image features are available. These features can be loaded directly and used as inputs to the HiCAT workflow. 

## Tutorial data and reference resources

- HiCAT is a supervised framework that uses annotated spatial sections as reference data. To make the framework easier to use, we provide preprocessed reference resources for breast cancer, human tonsil, and mouse brain datasets. These resources allow users to run label transfer and heterogeneity analyses without first generating their own annotated reference datasets.

- Users may also supply their own annotated spatial reference sections, which can provide more closely matched supervision for a specific tissue, disease context, or experimental platform. User-provided references can be used independently or combined with the reference resources distributed with HiCAT to support more robust and comprehensive inference.

- The tutorial uses preprocessed molecular data and precomputed image features so that users can focus on the core HiCAT workflow without repeating image-feature extraction or molecular preprocessing.

| Resource | Description | Data | Annotations | Precomputed reference information |
|---|---|---|---|---|
| Tutorial data | Preprocessed example data for reproducing the tutorial workflow | [Download](https://drive.google.com/drive/folders/1BaqScSe3mxz7JGlixYd-4SSmzHZBOoVb?usp=sharing) | - | — |
| Breast cancer | Breast cancer Spatial Transcriptomics dataset | [Download](https://zenodo.org/records/3957257) | [Download](https://drive.google.com/drive/folders/1yL9Y1f2s_N3m7O797afLyFvlZdSdeuJy?usp=sharing) | [Download](https://drive.google.com/drive/folders/1EMT0KHegTtNJVNtlEV7AJ6vw5nZHOfDn?usp=sharing) |
| Mouse brain | 10x Visium mouse-brain dataset | [Download](https://www.10xgenomics.com/datasets/multiomic-integration-neuroscience-application-note-visium-for-ffpe-plus-immunofluorescence-alzheimers-disease-mouse-model-brain-coronal-sections-from-one-hemisphere-over-a-time-course-1-standard) | [Download](https://drive.google.com/drive/folders/1h-7ZnagZgkk0nohHUByxnf2dxXZixKOS?usp=sharing) | [Download](https://drive.google.com/drive/folders/1h-7ZnagZgkk0nohHUByxnf2dxXZixKOS?usp=sharing) |


## Output structure

A complete run produces a numbered analysis directory such as:

```text
results/
├── 01_preprocessing/
├── 02_tree_inference/
├── 03_reference_selection/
├── 04_hierarchical_features/
├── 05_clustering_config/
├── 06_label_transfer/
└── 07_heterogeneity/
```

The workflow returns a `HiCATWorkflowResult` containing the in-memory result from every configured stage. Skipped stages remain `None`. Saved stage outputs can be restored with `load_stage_result`.

## Tested environment

The current package configuration has been tested with:

- **Operating system:** macOS Sonoma 14.0
- **Hardware:** Apple M1 Pro
- **Python:** 3.11.5
- **HiCAT:** 0.1.0

For the complete pinned environment, see [`HiCAT_package/environment.yml`](HiCAT_package/environment.yml). Package requirements and optional dependency groups are defined in [`HiCAT_package/pyproject.toml`](HiCAT_package/pyproject.toml).

## Citation

When using HiCAT, please cite the preprint:

> Huang J, Shen X, Smith Y, Harik L, Wang L, Yu J, Epstein MP, Hu J. **From unsupervised clustering to atlas-guided annotation in cohort-scale spatial omics with HiCAT.** bioRxiv. 2026. [doi:10.64898/2026.05.27.728266](https://doi.org/10.64898/2026.05.27.728266)

```bibtex
@article{huang2026hicat,
  title   = {From unsupervised clustering to atlas-guided annotation in cohort-scale spatial omics with HiCAT},
  author  = {Huang, Jing and Shen, Xueqi and Smith, Yoland and Harik, Lara and Wang, Linghua and Yu, Jindan and Epstein, Michael P. and Hu, Jian},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.05.27.728266}
}
```

## Contributing and support

HiCAT is actively developed, and contributions are welcome. To report a bug, request a feature, or suggest a documentation improvement, please [open a GitHub issue](https://github.com/jinghuang-stats/HiCAT/issues).

For bug reports, include the HiCAT version, Python version, operating system, a minimal reproducible example, and the complete error message when possible.

## License

HiCAT is distributed under the MIT License, as specified in the package metadata.
