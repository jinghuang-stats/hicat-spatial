from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import is_color_like
import pandas as pd
import scanpy as sc


__all__ = [
    "get_cluster_palette",
    "cat_figure",
    "con_figure",
]


def get_cluster_palette(cluster_labels, cat_color):
    """
    Build a categorical color palette with one color per observed cluster label.

    This avoids indexing errors when cluster labels are strings, categorical values,
    or when cluster labels do not start from 0, such as heatmap clusters starting
    from 1.

    Parameters
    ----------
    cluster_labels : array-like
        Observed cluster labels.

    cat_color : list or dict
        List of categorical colors, or a mapping from observed label to color.

    Returns
    -------
    cluster_palette : list
        Color list with length equal to the number of observed cluster labels.
    """

    if cat_color is None or len(cat_color) == 0:
        raise ValueError("cat_color must be a non-empty palette or color mapping.")

    labels = pd.Series(cluster_labels).dropna().astype(str).unique().tolist()
    labels = sorted(labels)

    if isinstance(cat_color, Mapping):
        color_by_label = _normalize_color_mapping(cat_color)
        missing_labels = [label for label in labels if label not in color_by_label]
        if missing_labels:
            raise ValueError(
                "cat_color is missing colors for observed labels: "
                f"{missing_labels}."
            )
        return [color_by_label[label] for label in labels]

    cluster_palette = [
        cat_color[i % len(cat_color)]
        for i in range(len(labels))
    ]

    return cluster_palette


def _normalize_color_mapping(cat_color):
    """Return a string-keyed label-to-color mapping with validated colors."""
    color_by_label = {str(label): color for label, color in cat_color.items()}
    invalid = {
        label: color
        for label, color in color_by_label.items()
        if not is_color_like(color)
    }
    if invalid:
        raise ValueError(f"cat_color contains invalid matplotlib colors: {invalid}.")
    return color_by_label


def _prepare_cat_figure_palette(input_adata, color_key, cat_color):
    """Prepare categorical labels and palette for Scanpy plotting."""
    if not isinstance(cat_color, Mapping):
        input_adata.obs[color_key] = input_adata.obs[color_key].astype("category")
        return cat_color

    color_by_label = _normalize_color_mapping(cat_color)
    labels = input_adata.obs[color_key].astype("string")
    observed_labels = pd.Series(labels).dropna().unique().tolist()
    missing_labels = [
        label for label in observed_labels if label not in color_by_label
    ]
    if missing_labels:
        raise ValueError(
            "cat_color is missing colors for observed labels: "
            f"{missing_labels}."
        )

    category_order = [
        label for label in color_by_label
        if label in set(observed_labels)
    ]
    input_adata.obs[color_key] = pd.Categorical(
        labels,
        categories=category_order,
        ordered=True,
    )
    return [color_by_label[label] for label in category_order]


def cat_figure(
    input_adata, 
    x_key, 
    y_key, 
    fig_title, 
    fig_path, 
    color_key, 
    cat_color, 
    size=50, 
    dpi=100, 
    invert_x=False, 
    invert_y=True
):
    """
    Save a categorical spatial scatter plot from an AnnData object.

    This function visualizes spatial coordinates stored in `input_adata.obs`
    and colors each spot/cell by a categorical annotation, such as cluster labels,
    tissue regions, or predicted subtypes. The generated figure is saved to
    `fig_path`.

    Parameters
    ----------
    input_adata : AnnData
        Input AnnData object. Spatial coordinate columns and the categorical
        annotation column should be stored in `input_adata.obs`.

    x_key : str
        Column name in `input_adata.obs` containing the x-coordinate for plotting,
        such as `"pixel_x"` or `"array_col"`.

    y_key : str
        Column name in `input_adata.obs` containing the y-coordinate for plotting,
        such as `"pixel_y"` or `"array_row"`.

    fig_title : str
        Title displayed on the plot.

    fig_path : str
        Path where the figure will be saved. The file extension determines the
        output format, for example `.png`, `.pdf`, or `.svg`.

    color_key : str
        Column name in `input_adata.obs` used to color the scatter plot.
        This column will be converted to categorical type before plotting.

    cat_color : list, tuple, dict, or None
        Categorical color palette used for plotting. If a list or tuple is
        provided, categories are colored in categorical order. If a dictionary
        is provided, keys are category labels and values are matplotlib colors,
        for example ``{"tumor": "#FD2B5C", "stroma": "#59BE86"}``. Missing
        observed labels raise ``ValueError``. If ``None``, Scanpy uses its
        default categorical color palette.

    size : float, default=50
        Marker size used in the scatter plot.

    dpi : int, default=100
        Resolution of the saved figure.

    invert_x : bool, default=False
        Whether to invert the x-axis.

    invert_y : bool, default=True
        Whether to invert the y-axis. This is often useful for spatial transcriptomics
        or image-based coordinates where the image origin is in the upper-left corner.

    Returns
    -------
    None
        The function saves the figure to `fig_path` and does not return an object.

    Raises
    ------
    KeyError
        If `x_key`, `y_key`, or `color_key` is not present in `input_adata.obs`.
    """

    if x_key not in input_adata.obs.columns or y_key not in input_adata.obs.columns:
        raise KeyError(f"{x_key!r} and/or {y_key!r} are not present in input_adata.obs.")
    if color_key not in input_adata.obs.columns:
        raise KeyError(f"{color_key!r} is not present in input_adata.obs.")

    # create output folder if it does not exist
    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    palette = _prepare_cat_figure_palette(input_adata, color_key, cat_color)

    fig = sc.pl.scatter(
        input_adata,
        alpha=1,
        x=x_key,
        y=y_key,
        color=color_key,
        palette=palette,
        show=False,
        size=size,
    )
    fig.set_aspect("equal", "box")
    if invert_y is True:
        fig.invert_yaxis()
    if invert_x is True:
        fig.invert_xaxis()
    fig.set_title(fig_title)
    fig.figure.savefig(fig_path, dpi=dpi, bbox_inches="tight")

    input_adata.uns.pop(color_key + "_colors", None)
    plt.clf()
    plt.close()


def con_figure(
    input_adata,
    x_key,
    y_key,
    fig_title,
    fig_path,
    color_key,
    cnt_color="coolwarm",
    size=50,
    dpi=100,
    invert_x=False,
    invert_y=True
):
    """
    Save a continuous spatial scatter plot from an AnnData object.

    This function visualizes spatial coordinates stored in `input_adata.obs`
    and colors each spot/cell by a continuous variable, such as gene expression,
    module score, probability score, or heterogeneity score. The generated figure
    is saved to `fig_path`.

    Parameters
    ----------
    input_adata : AnnData
        Input AnnData object. Spatial coordinate columns and the continuous
        variable to be plotted should be stored in `input_adata.obs`.

    x_key : str
        Column name in `input_adata.obs` containing the x-coordinate for plotting,
        such as `"pixel_x"` or `"array_col"`.

    y_key : str
        Column name in `input_adata.obs` containing the y-coordinate for plotting,
        such as `"pixel_y"` or `"array_row"`.

    fig_title : str
        Title displayed on the plot.

    fig_path : str
        Path where the figure will be saved. The file extension determines the
        output format, for example `.png`, `.pdf`, or `.svg`.

    color_key : str
        Column name in `input_adata.obs` used to color the scatter plot.
        This should usually be a continuous numeric variable.

    cnt_color : str or matplotlib colormap, default="coolwarm"
        Continuous colormap used for plotting.

    size : float, default=50
        Marker size used in the scatter plot.

    dpi : int, default=200
        Resolution of the saved figure.

    invert_x : bool, default=False
        Whether to invert the x-axis.

    invert_y : bool, default=True
        Whether to invert the y-axis. This is often useful for spatial transcriptomics
        or image-based coordinates where the image origin is in the upper-left corner.

    Returns
    -------
    None
        The function saves the figure to `fig_path` and does not return an object.

    Raises
    ------
    KeyError
        If `x_key`, `y_key`, or `color_key` is not present in `input_adata.obs`.

    ValueError
        If `color_key` is not numeric.
    """

    if x_key not in input_adata.obs.columns or y_key not in input_adata.obs.columns:
        raise KeyError(
            f"{x_key!r} and/or {y_key!r} are not present in input_adata.obs."
        )

    if color_key not in input_adata.obs.columns:
        raise KeyError(
            f"{color_key!r} is not present in input_adata.obs."
        )

    if not pd.api.types.is_numeric_dtype(input_adata.obs[color_key]):
        raise ValueError(
            f"{color_key!r} should be numeric for a continuous scatter plot."
        )

    # create output folder if it does not exist
    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    fig = sc.pl.scatter(
        input_adata,
        alpha=1,
        x=x_key,
        y=y_key,
        color=color_key,
        color_map=cnt_color,
        show=False,
        size=size,
    )

    fig.set_aspect("equal", "box")

    if invert_y:
        fig.invert_yaxis()

    if invert_x:
        fig.invert_xaxis()

    fig.set_title(fig_title)
    fig.figure.savefig(fig_path, dpi=dpi, bbox_inches="tight")

    plt.clf()
    plt.close(fig.figure)
