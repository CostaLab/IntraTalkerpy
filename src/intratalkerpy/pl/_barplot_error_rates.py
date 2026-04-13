import pandas as pd
import scanpy as sc
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns
from typing import Optional

def plot_error_boxplot(
    adata: ad.AnnData,
    y: str,
    title: str = None,
    save_path: str = None,
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    Plot a boxplot of a specified error statistic grouped by number of receptors.

    Args:
        adata:      AnnData object containing regression statistics in adata.uns["regression_statistics"].
        y:          Column name to plot on the y-axis ('r2', 'mse', 'rmse', 'nrmse_mean', 'nrmse_maxmin').
        title:      Optional title for the plot.
        save_path:  Optional file path to save the figure (e.g. outpath + 'plot.png').
        figsize:    Figure size as (width, height) in inches. Default is (10, 6).
    
    Returns:
        fig:        The matplotlib Figure object.
    """
    if "regression_statistics" not in adata.uns:
        raise KeyError("adata.uns does not contain 'regression_statistics'. "
                       "Please run the regression training first.")

    df = adata.uns["regression_statistics"]

    if y not in df.columns:
        raise ValueError(f"Column '{y}' not found in regression_statistics. "
                         f"Available columns: {list(df.columns)}")

    fig, ax = plt.subplots(figsize=figsize)

    sns.boxplot(data=df, x="num_receptors", y=y, ax=ax)

    ax.set_xlabel("Number of Receptors")
    ax.set_ylabel(y.upper())

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"{y.upper()} by Number of Receptors")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=600)

    plt.show()
    
    return fig