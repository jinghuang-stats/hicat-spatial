# HiCAT Spatial

HiCAT is a hierarchical cohort-scale annotation-transfer workflow for
multimodal spatial omics data. The PyPI distribution is `hicat-spatial`, the
Python import is `hicat_spatial`, and the main class remains `HiCAT`.

The workflow has seven explicit stages: preprocessing, tree inference,
reference selection, hierarchical feature selection, clustering configuration,
label transfer, and heterogeneity analysis.

## Documentation

Beginner-oriented stage guides live under [`docs/`](docs/README.md). Start with
[`docs/FULL_WORKFLOW.md`](docs/FULL_WORKFLOW.md) to see how stages connect, then
open the individual stage pages for config tables, required input structures,
result objects, and saved files.

## Installation

```bash
python -m pip install hicat-spatial
```

For local development, run `python -m pip install -e ".[dev]"`. Install the
separate image stack only when extracting HIPT or UNI features:

```bash
python -m pip install "hicat-spatial[image]"
```

For reproducing the original tutorial/legacy label-transfer behavior, use the
pinned environment file before installing the local package. This is important
for HIPT/Image Leiden clustering because newer Scanpy/Leiden stacks can produce
many disconnected neighbor-graph components and therefore many more clusters.

```bash
conda env create -f environment_legacy.yml
conda activate hicat-legacy
python -m pip install -e ".[image,notebook]"
```

HIPT and UNI checkpoints are not included. Their expected locations and image
boundary behavior are described in
[`hicat_spatial/preprocessing/README.md`](hicat_spatial/preprocessing/README.md).

## Stage 1 quick start

Create one flat raw-data folder, then let HiCAT create the preprocessing
folder tree. For a Gene-only run, use filenames like:

```text
data/
  reference_1_ref_gene_raw.h5ad
  query_1_query_gene_raw.h5ad
```

Optional Protein and image files follow the same section IDs:

```text
data/
  reference_1_ref_protein_raw.h5ad
  reference_1_image.png
  reference_1_annotated_image.png
  query_1_query_protein_raw.h5ad
  query_1_image.png
```

Then run:

```python
from hicat_spatial import HiCAT, PreprocessConfig

preprocess = PreprocessConfig(
    data_dir="./data",
    preprocess_dir="./results/01_preprocessing",
    reference_sections=["reference_1"],
    query_sections=["query_1"],
    modalities=("Gene",),
    raw_file_mode="copy",
)

hicat = HiCAT({"preprocessing": preprocess})
result = hicat.run_preprocessing()
reference_gene = result.reference["spot"]["Gene"]["reference_1"]
```

The default filenames are `reference_1_ref_gene_raw.h5ad` and
`query_1_query_gene_raw.h5ad`. If images are selected, use
`reference_1_image.png`, `reference_1_annotated_image.png`, and
`query_1_image.png` or another supported image extension.

`raw_file_mode="copy"` makes a self-contained preprocessing folder. Use
`"symlink"` to save disk space or `"none"` to read directly from `data_dir`.

By default, Stage 1 normalizes molecular data to `target_sum=10_000` and then
applies `log1p=True`. To keep values on their original scale, set
`target_sum=None` and `log1p=False`.

When extracting HIPT/UNI image features, Stage 1 extracts the image embedding
grid once per section and then aggregates it to the requested coordinate level.
For enhanced analyses, this is usually enough:

```python
preprocess = PreprocessConfig(
    ...,
    modalities=("Gene", "Image"),
    gene_enhancement=True,
    enhancement_kwargs={"resolution": 50},
    image_feature_levels="enhanced",
    image_feature_kwargs={
        "model": "hipt",
        "checkpoint_path": "./checkpoints/hipt",
    },
)
```

Use `image_feature_levels=("spot", "enhanced")` plus
`image_feature_level_kwargs` if you want both spot-level and enhanced-level
Image AnnData objects with different `patch_size_spot` values.

If `label_color_dict=None`, Stage 1 does not extract scribbles from annotated
reference images. In that case, the reference AnnData should already contain
the annotation column, for example `adata.obs["label"]` or the column named by
`label_key`.

If image features were already extracted, add `"Image"` to `modalities` and set
`image_feature_mode="load"`. Stage 1 will read
`{section}_ref_image_features.h5ad` and `{section}_query_image_features.h5ad`
from `data_dir`, align them to the molecular spot IDs, and save standard
`{section}_image.h5ad` files under the preprocessing output folders.

For an ordered multi-stage run, use `HiCATWorkflowConfig` and
`run_hicat_workflow`. See [`docs/FULL_WORKFLOW.md`](docs/FULL_WORKFLOW.md).

## Development checks

```bash
python -m ruff check hicat_spatial tests
python -m pytest
python -m build
```

## License

HiCAT-authored code is distributed under the MIT License. See [`NOTICE`](NOTICE)
for bundled third-party components.
