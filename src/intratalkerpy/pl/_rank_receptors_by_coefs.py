import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import anndata as ad
from typing import Optional

def plot_receptor_coeff(
    adata: ad.AnnData,
    top_n: int = 0,
    figsize: tuple[int, int] = (10, 30),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a horizontal barplot ranking receptors by mean regression coefficient.

    Requires compute_receptor_coeff_stats() to have been called first so that
    adata.uns[stats_key] is populated.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object containing the precomputed statistics in adata.uns[stats_key].
    top_n : int
        Number of top (and bottom) receptors to display.
            - 0  : show all receptors (default).
            - N>0: show the top-N and bottom-N receptors by mean coefficient.
    figsize : tuple[int, int]
        Figure size as (width, height) in inches. Default: (10, 30).
    save_path : str or None
        If provided, the figure is saved to this path. Supports any extension
        recognised by matplotlib (e.g. ".png", ".pdf", ".svg").
        If None, the figure is not saved. Default: None.

    Returns
    -------
    plt.Figure
        The matplotlib Figure object.
    """
    coeff_stats = adata.uns["regression_coef_statistics"].copy()
    coeff_stats = coeff_stats.sort_values("Mean_Coeff", ascending=False)

    # --- Subset to top/bottom N receptors if requested ---
    if top_n > 0:
        coeff_stats = coeff_stats.iloc[np.r_[0:top_n, -top_n:0]]
        coeff_stats = coeff_stats.sort_values("Mean_Coeff", ascending=False)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.barplot(
        data=coeff_stats,
        x="Mean_Coeff",
        y="Gene",
        order=coeff_stats["Gene"],
        ax=ax,
    )
    ax.bar_label(ax.containers[0], labels=coeff_stats["count"])
    ax.set_xlabel("Mean Coefficient")
    ax.set_ylabel("Receptor")
    ax.set_title("Receptor Ranking by Mean Regression Coefficient")
    fig.tight_layout(pad=0.5)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    return fig
