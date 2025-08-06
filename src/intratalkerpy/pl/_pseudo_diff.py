import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Union, List, Any
import warnings


def plot_differential_pseudotime(
    emb: np.ndarray,
    pseudotime_diff: np.ndarray,
    ax: plt.Axes,
    receptor: str,
    colors: Optional[List[str]] = None,
    n_bins: int = 256,
    point_size: float = 15,
    title: Optional[str] = None,
    colorbar: bool = True,
    colorbar_label: str = "Pseudotime Difference",
    alpha: float = 1.0,
    edgecolors: str = 'none'
) -> plt.Axes:
    """
    Plot differential pseudotime values on a 2D embedding.
    
    Parameters
    ----------
    emb : np.ndarray
        2D embedding coordinates of shape (n_cells, 2).
    pseudotime_diff : np.ndarray
        Pseudotime difference values for each cell.
    ax : plt.Axes
        Matplotlib axes object to plot on.
    receptor : str
        Name of the receptor being analyzed.
    colors : List[str], optional
        Custom colors for the colormap. Default uses blue-white-red.
    n_bins : int, default=256
        Number of bins for the colormap.
    point_size : float, default=15
        Size of scatter plot points.
    title : str, optional
        Custom title for the plot. If None, uses default format.
    colorbar : bool, default=True
        Whether to add a colorbar.
    colorbar_label : str, default="Pseudotime Difference"
        Label for the colorbar.
    alpha : float, default=1.0
        Transparency of points (0-1).
    edgecolors : str, default='none'
        Edge colors for scatter points.
        
    Returns
    -------
    plt.Axes
        The modified axes object.
        
    Raises
    ------
    ValueError
        If input arrays have incompatible shapes or invalid values.
    """
    # Input validation
    if emb.shape[0] != len(pseudotime_diff):
        raise ValueError("Embedding and pseudotime_diff must have same number of points")
    
    if emb.shape[1] != 2:
        raise ValueError("Embedding must be 2D (shape: n_cells x 2)")
    
    if not (0 <= alpha <= 1):
        raise ValueError("Alpha must be between 0 and 1")
    
    if point_size <= 0:
        raise ValueError("Point size must be positive")
    
    # Handle empty or invalid data
    if len(pseudotime_diff) == 0:
        warnings.warn("Empty pseudotime_diff array provided")
        ax.text(0.5, 0.5, 'No data to display', ha='center', va='center', transform=ax.transAxes)
        return ax
    
    # Handle NaN values
    valid_mask = ~(np.isnan(emb).any(axis=1) | np.isnan(pseudotime_diff))
    if not valid_mask.any():
        warnings.warn("All values are NaN")
        ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
        return ax
    
    if not valid_mask.all():
        warnings.warn(f"Removing {(~valid_mask).sum()} cells with NaN values")
        emb = emb[valid_mask]
        pseudotime_diff = pseudotime_diff[valid_mask]
    
    # Set default colors
    if colors is None:
        colors = ['#3a4cc0', "#DBDADA", '#b30326']  # Blue-White-Red
    
    if len(colors) < 2:
        raise ValueError("At least 2 colors required for colormap")
    
    # Create colormap
    cmap_custom = mcolors.LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    
    # Calculate symmetric range for color normalization
    v_range = max(abs(pseudotime_diff.min()), abs(pseudotime_diff.max()))
    
    if v_range == 0:
        warnings.warn("All pseudotime differences are zero")
        v_range = 1  # Prevent division by zero
    
    # Create normalization
    norm = TwoSlopeNorm(vmin=-v_range, vcenter=0, vmax=v_range)
    
    # Create scatter plot
    sc = ax.scatter(
        emb[:, 0], emb[:, 1],
        c=pseudotime_diff, 
        cmap=cmap_custom, 
        norm=norm, 
        s=point_size,
        alpha=alpha,
        edgecolors=edgecolors,
        linewidths=0.5 if edgecolors != 'none' else 0
    )
    
    # Add colorbar if requested
    if colorbar:
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(colorbar_label, rotation=270, labelpad=20)
    
    # Set title
    if title is None:
        title = f"{receptor} Pseudotime Difference"
    ax.set_title(title)
    
    # Set axis labels
    ax.set_xlabel("Embedding 1")
    ax.set_ylabel("Embedding 2")
    
    return ax

def plot_pseudotime_distributions(
    adata: Any,
    orig_pseudotime: str,
    receptor: str,
    cell_anno: str,
    result_path: str,
    figsize: Tuple[float, float] = (6, 4),
    aspect: float = 1.5,
    height: float = 3,
    alpha: float = 0.5,
    palette: Optional[str] = None,
    save_format: str = "pdf",
    title_template: Optional[str] = None,
    show_plot: bool = True,
    dpi: int = 300
) -> sns.FacetGrid:
    """
    Plot and compare distributions of original and receptor-specific pseudotime.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object containing pseudotime information.
    orig_pseudotime : str
        Column name for original pseudotime in adata.obs.
    receptor : str
        Name of the receptor being analyzed.
    cell_anno : str
        Column name for cell annotations in adata.obs.
    result_path : str
        Path where the plot will be saved.
    figsize : Tuple[float, float], default=(6, 4)
        Figure size as (width, height) in inches.
    aspect : float, default=1.5
        Aspect ratio of each facet.
    height : float, default=3
        Height of each facet in inches.
    alpha : float, default=0.5
        Transparency of density plots (0-1).
    palette : str, optional
        Color palette for the plots.
    save_format : str, default="pdf"
        File format for saving the plot.
    title_template : str, optional
        Custom title template. If None, uses default.
    show_plot : bool, default=True
        Whether to display the plot.
    dpi : int, default=300
        Resolution for saved figure.
        
    Returns
    -------
    sns.FacetGrid
        The seaborn FacetGrid object.
        
    Raises
    ------
    ValueError
        If required columns are missing from adata.obs.
    KeyError
        If specified columns don't exist in the data.
    """
    # Input validation
    if not hasattr(adata, 'obs'):
        raise ValueError("adata must have an 'obs' attribute")
    
    required_cols = [cell_anno, orig_pseudotime]
    receptor_pseudotime_col = f'pseudotime_{receptor}'
    required_cols.append(receptor_pseudotime_col)
    
    missing_cols = [col for col in required_cols if col not in adata.obs.columns]
    if missing_cols:
        raise KeyError(f"Missing columns in adata.obs: {missing_cols}")
    
    if not (0 <= alpha <= 1):
        raise ValueError("Alpha must be between 0 and 1")
    
    # Prepare data
    try:
        data_df = adata.obs[required_cols].copy()
        
        # Check for missing values
        missing_mask = data_df.isnull().any(axis=1)
        if missing_mask.any():
            warnings.warn(f"Removing {missing_mask.sum()} cells with missing pseudotime values")
            data_df = data_df.dropna()
        
        if data_df.empty:
            raise ValueError("No valid data remaining after removing missing values")
        
        # Add dataset identifier
        data_df['Dataset'] = 'both'
        
        # Melt the DataFrame for plotting
        melted_df = data_df.melt(
            id_vars=[cell_anno, 'Dataset'],
            value_vars=[orig_pseudotime, receptor_pseudotime_col],
            var_name='Pseudotime Type',
            value_name='Pseudotime'
        )
        
        # Clean up pseudotime type names for better display
        melted_df['Pseudotime Type'] = melted_df['Pseudotime Type'].replace({
            orig_pseudotime: 'Original',
            receptor_pseudotime_col: f'{receptor} Modified'
        })
        
    except Exception as e:
        raise ValueError(f"Error preparing data: {e}")
    
    # Create the plot
    try:
        g = sns.FacetGrid(
            melted_df,
            row='Dataset',
            col='Dataset',
            hue='Pseudotime Type',
            aspect=aspect,
            height=height,
            sharex=True,
            sharey=False,
            palette=palette
        )
        
        # Map the density plots
        g.map(
            sns.kdeplot,
            'Pseudotime',
            fill=True,
            alpha=alpha,
            warn_singular=False  # Suppress warnings for identical values
        )
        
        # Customize labels and titles
        g.set_axis_labels("Pseudotime", "Density")
        
        if title_template is None:
            g.set_titles(row_template="{row_name}", col_template="{col_name} Dataset")
        else:
            g.set_titles(title_template)
        
        # Add legend
        g.add_legend(
            title="Pseudotime Type", 
            bbox_to_anchor=(0.9, 0.5), 
            loc="center left", 
            borderaxespad=0
        )
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        
        # Save the figure
        result_path = Path(result_path)
        result_path.mkdir(parents=True, exist_ok=True)
        
        save_path = result_path / f"{receptor}_pseudotime_distribution.{save_format}"
        g.savefig(save_path, bbox_inches="tight", dpi=dpi)
        
        # Show plot if requested
        if show_plot:
            plt.show()
        
        plt.close()
        
        return g
        
    except Exception as e:
        warnings.warn(f"Error creating plot: {e}")
        # Create a simple fallback plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f'Error creating plot: {e}', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"{receptor} Pseudotime Distribution")
        if show_plot:
            plt.show()
        plt.close()
        return None