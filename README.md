# HiCAT 

## From unsupervised clustering to atlas-guided annotation in cohort-scale spatial omics with HiCAT


#### Jing Huang, Xueqi Shen, Yoland Smith, Lara Harik, Linghua Wang, Jindan Yu, Michael P. Epstein*, Jian Hu*

HiCAT is a supervised computational framework for generating pathologist-informed region annotations and characterizing region-level heterogeneity in multimodal spatial omics data. By generating consistent, enhanced-resolution, and biologically informed region annotations across large cohorts, HiCAT constructs annotated spatial atlas that supports scalable cohort-level downstream analyses, such as identifying tumor subregions associated with clinical outcomes and brain subregions aligned with spatiotemporal disease progression. HiCAT is applicable to diverse spatial omics platforms, including spatial transcriptomics (Spatial Transcriptomics, 10x Visium, 10x Visium HD, and 10x Xenium) as well as paired transcriptomic and protein measurements (10x spatial omics and spatial CITE-seq).

![HiCAT workflow](figures/HiCAT_workflow_most_upd.png)
<br>
For a detailed description of the method and analyses, please see our preprint: [Biorxiv]()
<br>

## Usage
With [**HiCAT**](https://github.com/jinghuang-stats/HiCAT) package, you can:
- Extract pathologist-generated scribble annotations
- Infer hierarchical tissue organization by integrating multimodal spatial omics inputs
- Quantify region-specific heterogeneity across samples to pinpoint potentially disease-relevant regions for further investigation
- Select suitable reference samples from the training set to provide matched supervision for each query sample
- Transfer pathologist-informed annotations and characterize region-level heterogeneity beyond the granularity of the original annotations
- Perform cohort-level heterogeneity analyses and interpret the functional roles of identified heterogeneous subtypes

Although HiCAT is a supervised framework that requires annotated spatial sections as inputs, we provide trained reference datasets for breast cancer, human tonsil, and mouse brain, allowing users to directly perform label transfer and heterogeneity analyses without generating their own annotated reference data.

Users are also welcome and encouraged to provide their own annotated spatial reference datasets, which can offer more closely matched supervision for label transfer. These user-provided references can also be integrated with the provided HiCAT references to support more robust and comprehensive inference. 

## Tutorial
For the step-by-step tutorial, please refer to:
<br>

<br>
<br>
A Jupyter Notebook of the tutorial is accessible from:
<br>

<br>

<br>
Please install Jupter in order to open this notebook.
<br>
<br>
Toy data can be downloaded at: 
<br>

<br>
Trained reference datasets can be downloaded:



## System requirements
Python support packages: 

## Versions the software has been tested on
# Environment 1:
- System: Mac OS Sonoma 14.0 (M1 Pro)
- Python:
- Python packages:

## Contributing
Source code: [GitHub](https://github.com/jinghuang-stats/HiCAT)

We are continuing adding new features. Bug reports or feature requests are welcome.

<br>
