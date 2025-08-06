import random
from adjustText import adjust_text
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as PathEffects
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Union, Dict, Any, Tuple
import warnings


def _subset_list(target_list, index_list):
    """
    Helper function to subset a list based on indices.
    
    Parameters
    ----------
    target_list : list
        Source list to subset.
    index_list : list of int
        Indices to extract from target_list.
        
    Returns
    -------
    list
        Subsetted list with elements at specified indices.
    """
    return list(map(target_list.__getitem__, index_list))


def _generate_random_colors(n_colors, seed=555):
    """
    Generate random hex colors.
    
    Parameters
    ----------
    n_colors : int
        Number of colors to generate.
    seed : int, default=555
        Random seed for reproducibility.
        
    Returns
    -------
    list of str
        List of hex color strings (e.g., ['#FF0000', '#00FF00']).
    """
    random.seed(seed)
    return [f"#{random.randint(0, 0xFFFFFF):06x}" for _ in range(n_colors)]


def _validate_basic_inputs(anndata, reduction_name, variable):
    """
    Basic validation for anndata inputs.
    
    Parameters
    ----------
    anndata : AnnData
        Annotated data object to validate.
    reduction_name : str
        Name of the reduction to check in anndata.obsm.
    variable : str
        Variable name to check in anndata.obs.
        
    Raises
    ------
    ValueError
        If reduction_name not found in anndata.obsm or variable not found in anndata.obs.
    """
    if reduction_name not in anndata.obsm.keys():
        raise ValueError(f"Reduction '{reduction_name}' not found in anndata.obsm")
    if variable not in anndata.obs.columns:
        raise ValueError(f"Variable '{variable}' not found in anndata.obs")


def plot_metadata_given_ax(anndata,
                        ax: matplotlib.axes,
                        reduction_name: str,
                        variable: str,
                        color_dictionary,
                        receptor: str,
                        remove_nan: Optional[bool] = True,
                        show_label: Optional[bool] = True,
                        show_legend: Optional[bool] = False,
                        cmap: Optional[Union[str, 'matplotlib.cm']] = cm.viridis,
                        dot_size: Optional[int] = 10,
                        text_size: Optional[int] = 10,
                        alpha: Optional[Union[float, int]] = 1,
                        seed: Optional[int] = 555,
                        selected_cells: Optional[List[str]] = None):
    """
    Plot metadata on a 2D embedding.
    
    Parameters
    ----------
    anndata : AnnData
        Annotated data object containing embedding and metadata.
    ax : matplotlib.axes
        Matplotlib axes object to plot on.
    reduction_name : str
        Name of the reduction/embedding in anndata.obsm (e.g., 'X_umap', 'X_pca').
    variable : str
        Variable name to plot from anndata.obs.
    color_dictionary : dict
        Dictionary containing color mappings for categorical variables.
        Expected format: {variable_name: {category: color}}.
    receptor : str
        Name of the receptor being analyzed (used in plot title).
    remove_nan : bool, default=True
        Whether to remove NaN values from categorical data.
    show_label : bool, default=True
        Whether to show category labels on the plot.
    show_legend : bool, default=False
        Whether to display legend.
    cmap : str or matplotlib.cm, default=cm.viridis
        Colormap for continuous variables.
    dot_size : int, default=10
        Size of scatter plot points.
    text_size : int, default=10
        Size of label text.
    alpha : float or int, default=1
        Transparency of points (0-1).
    seed : int, default=555
        Random seed for color generation.
    selected_cells : List[str], optional
        Subset of cell names to plot.
        
    Returns
    -------
    matplotlib.axes
        The modified axes object.
    """
    # Basic validation
    _validate_basic_inputs(anndata, reduction_name, variable)
    
    GEX_cell_names = anndata.obs_names.copy(deep=True)
    GEX_cell_names = list(GEX_cell_names)
    GEX_dr_cell = {k: pd.DataFrame(anndata.obsm[k].copy(
    ), index=GEX_cell_names) for k in anndata.obsm.keys()}

    embedding = GEX_dr_cell[reduction_name]
    data_mat = anndata.obs.copy(deep=True)

    if selected_cells is not None:
        data_mat = data_mat.loc[selected_cells]
        embedding = embedding.loc[selected_cells]

    data_mat = data_mat.loc[embedding.index.to_list()]

    var_data = data_mat.copy().loc[:, variable].dropna().to_list()
    if len(var_data) == 0:
        # Handle case where all data is NaN
        ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
        return ax
        
    if isinstance(var_data[0], str):
        if (remove_nan) & (data_mat[variable].isnull().sum() > 0):
            var_data = data_mat.copy().loc[:, variable].dropna().to_list()
            emb_nan = embedding.loc[data_mat.copy(
            ).loc[:, var_data].dropna().index.tolist()]
            label_pd = pd.concat(
                [emb_nan, data_mat.loc[:, [variable]].dropna()], axis=1, sort=False)
        else:
            var_data = data_mat.copy().astype(
                str).fillna('NA').loc[:, variable].to_list()
            label_pd = pd.concat([embedding, data_mat.astype(
                str).fillna('NA').loc[:, [variable]]], axis=1, sort=False)

        categories = set(var_data)
        try:
            color_dict = color_dictionary[variable]
        except (KeyError, TypeError):
            color = _generate_random_colors(len(categories), seed)
            color_dict = dict(zip(categories, color))

        if (remove_nan) & (data_mat[variable].isnull().sum() > 0):
            ax.scatter(emb_nan.iloc[:, 0], emb_nan.iloc[:, 1], c=data_mat.loc[:, variable].dropna(
            ).apply(lambda x: color_dict[x]), s=dot_size, alpha=alpha)
            ax.set_xlabel(emb_nan.columns[0])
            ax.set_ylabel(emb_nan.columns[1])
        else:
            ax.scatter(embedding.iloc[:, 0], embedding.iloc[:, 1], c=data_mat.astype(str).fillna(
                'NA').loc[:, variable].apply(lambda x: color_dict[x]), s=dot_size, alpha=alpha)
            ax.set_xlabel(embedding.columns[0])
            ax.set_ylabel(embedding.columns[1])

        if show_label:
            label_pos = label_pd.groupby(variable).agg(
                {label_pd.columns[0]: np.mean, label_pd.columns[1]: np.mean})
            texts = []
            for label in label_pos.index.tolist():
                texts.append(
                    ax.text(
                        label_pos.loc[label][0],
                        label_pos.loc[label][1],
                        label,
                        horizontalalignment='center',
                        verticalalignment='center',
                        size=text_size,
                        weight='bold',
                        color=color_dict[label],
                        path_effects=[
                            PathEffects.withStroke(
                                linewidth=3,
                                foreground='w')]))
            adjust_text(texts)

        ax.set_title(receptor + " Vector Field")
        patchList = []
        for key in color_dict:
            data_key = mpatches.Patch(color=color_dict[key], label=key)
            patchList.append(data_key)
        if show_legend:
            ax.legend(
                handles=patchList, bbox_to_anchor=(
                    1.04, 1), loc="upper left")
        return ax
    else:
        var_data = data_mat.copy().loc[:, variable].to_list()
        o = np.argsort(var_data)
        ax.scatter(embedding.iloc[o, 0], embedding.iloc[o, 1], c=_subset_list(
            var_data, o), cmap=cmap, s=dot_size, alpha=alpha)
        ax.set_xlabel(embedding.columns[0])
        ax.set_ylabel(embedding.columns[1])
        ax.set_title(variable)
        # setup the colorbar
        normalize = mcolors.Normalize(
            vmin=np.array(var_data).min(),
            vmax=np.array(var_data).max())
        scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=cmap)
        scalarmappaple.set_array(var_data)
        plt.colorbar(scalarmappaple, ax=ax)
        return ax

def vector_field_wrapper(
    adata: Any,
    grid: np.ndarray,
    vectors: np.ndarray,
    distances: np.ndarray,
    red_name: str,
    cell_anno: str,
    receptor: str,
    color_dict: Dict[str, str],  # Changed from nested dict to simple dict
    ax: matplotlib.axes.Axes,
    stream_density: float = 1.0,
    zorder: int = 10,
    grid_dist: int = 25,
    norm_vmin: float = 0.15,
    norm_vmax: float = 0.5,
    stream_cmap: str = 'Greys',
    linewidth: float = 1.2,
    show_label: bool = True
) -> matplotlib.axes.Axes:
    """
    Create a vector field plot with streamlines overlay.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object containing embedding and metadata.
    grid : np.ndarray
        Grid points for the vector field of shape (n_grid_points, 2).
    vectors : np.ndarray
        Vector field values at grid points of shape (n_grid_points, 2).
    distances : np.ndarray
        Distance values for coloring streamlines of shape (n_grid_points,).
    red_name : str
        Name of the reduction/embedding in adata.obsm.
    cell_anno : str
        Cell annotation variable name from adata.obs.
    receptor : str
        Receptor name for labeling (used in plot title).
    color_dict : dict
        Color dictionary mapping cell types to colors.
        Format: {cell_type: color_hex_string}.
    ax : matplotlib.axes
        Matplotlib axes object to plot on.
    stream_density : float, default=1.0
        Density of streamlines.
    zorder : int, default=10
        Z-order for streamline plotting.
    grid_dist : int, default=25
        Grid distance parameter (grid will be grid_dist x grid_dist).
    norm_vmin : float, default=0.15
        Minimum value for streamline color normalization.
    norm_vmax : float, default=0.5
        Maximum value for streamline color normalization.
    stream_cmap : str, default='Greys'
        Colormap for streamlines.
    linewidth : float, default=1.2
        Width of streamlines.
    show_label : bool, default=True
        Whether to show cell type labels.
        
    Returns
    -------
    matplotlib.axes
        The modified axes object with vector field plot.
        
    Raises
    ------
    ValueError
        If grid or vectors are empty.
    """
    # Basic validation
    if grid.size == 0 or vectors.size == 0:
        raise ValueError("Grid and vectors cannot be empty")
    
    if len(distances) != len(grid):
        raise ValueError("Distances and grid must have same length")
    
    if grid_dist <= 0:
        raise ValueError("Grid distance must be positive")
    
    expected_grid_size = grid_dist * grid_dist * 2
    if grid.size != expected_grid_size:
        warnings.warn(f"Grid size {grid.size} doesn't match expected {expected_grid_size}")
    
    try:
        # Create normalization for streamline colors
        norm = matplotlib.colors.Normalize(vmin=norm_vmin, vmax=norm_vmax, clip=True)
        
        # Scale distances for color mapping
        def scale_array(X):
            """Scale array to [0, 1] range."""
            X = np.asarray(X)
            if X.max() == X.min():
                return np.ones_like(X) * 0.5  # Return middle value if constant
            return (X - X.min()) / (X.max() - X.min())
        
        # Plot the underlying scatter plot with cell annotations
        ax = plot_metadata_given_ax(
            anndata=adata,
            reduction_name=red_name,
            receptor=receptor,
            ax=ax,
            variable=cell_anno,
            color_dictionary=color_dict,  # Pass color_dict directly, don't wrap it
            show_label=show_label,
            show_legend=False  # Don't show legend to avoid clutter
        )
        
        # Reshape arrays for streamplot
        try:
            grid_x = grid.reshape(grid_dist, grid_dist, 2)[:, :, 0]
            grid_y = grid.reshape(grid_dist, grid_dist, 2)[:, :, 1]
            vector_x = vectors.reshape(grid_dist, grid_dist, 2)[:, :, 0]
            vector_y = vectors.reshape(grid_dist, grid_dist, 2)[:, :, 1]
            scaled_distances = scale_array(distances).reshape(grid_dist, grid_dist)
        except ValueError as e:
            raise ValueError(f"Error reshaping arrays for streamplot: {e}")
        
        # Create streamplot
        ax.streamplot(
            grid_x, grid_y,
            vector_x, vector_y,
            density=stream_density,
            color=scaled_distances,
            cmap=stream_cmap,
            zorder=zorder,
            norm=norm,
            linewidth=linewidth
        )
        
        return ax
        
    except Exception as e:
        warnings.warn(f"Error in vector_field_wrapper: {e}")
        ax.text(0.5, 0.5, f'Error creating vector field: {e}', 
                ha='center', va='center', transform=ax.transAxes)
        return ax

def plot_raw_vector_field(
    emb: np.ndarray,
    delta_vec: np.ndarray,
    figsize: Tuple[float, float] = (8, 6),
    point_size: float = 5,
    point_alpha: float = 0.5,
    vector_color: str = "red",
    vector_alpha: float = 0.7,
    vector_scale: float = 1,
    title: str = "Raw Perturbation Vector Field",
    save_path: Optional[str] = None,
    show_plot: bool = True,
    dpi: int = 300
) -> plt.Figure:
    """
    Plot raw vector field showing perturbation vectors.
    
    Parameters
    ----------
    emb : np.ndarray
        2D embedding coordinates of shape (n_cells, 2).
    delta_vec : np.ndarray
        Perturbation vectors of shape (n_cells, 2).
    figsize : Tuple[float, float], default=(8, 6)
        Figure size as (width, height) in inches.
    point_size : float, default=5
        Size of cell points.
    point_alpha : float, default=0.5
        Transparency of cell points.
    vector_color : str, default="red"
        Color of vector arrows.
    vector_alpha : float, default=0.7
        Transparency of vector arrows.
    vector_scale : float, default=1
        Scale factor for vector arrows.
    title : str, default="Raw Perturbation Vector Field"
        Title for the plot.
    save_path : str, optional
        Path to save the figure.
    show_plot : bool, default=True
        Whether to display the plot.
    dpi : int, default=300
        Resolution for saved figure.
        
    Returns
    -------
    plt.Figure
        The matplotlib figure object.
        
    Raises
    ------
    ValueError
        If input arrays have incompatible shapes.
    """
    # Input validation
    if emb.shape[0] != delta_vec.shape[0]:
        raise ValueError("Embedding and delta_vec must have same number of points")
    
    if emb.shape[1] != 2 or delta_vec.shape[1] != 2:
        raise ValueError("Both arrays must be 2D (n_cells x 2)")
    
    if not (0 <= point_alpha <= 1) or not (0 <= vector_alpha <= 1):
        raise ValueError("Alpha values must be between 0 and 1")
    
    # Handle empty data
    if len(emb) == 0:
        warnings.warn("Empty embedding array provided")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No data to display', ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Handle NaN values
    valid_mask = ~(np.isnan(emb).any(axis=1) | np.isnan(delta_vec).any(axis=1))
    if not valid_mask.any():
        warnings.warn("All values are NaN")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
        return fig
    
    if not valid_mask.all():
        warnings.warn(f"Removing {(~valid_mask).sum()} cells with NaN values")
        emb = emb[valid_mask]
        delta_vec = delta_vec[valid_mask]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Plot cell points
    ax.scatter(
        emb[:, 0], emb[:, 1], 
        alpha=point_alpha, 
        s=point_size, 
        label="Cells"
    )
    
    # Plot vector field
    ax.quiver(
        emb[:, 0], emb[:, 1], 
        delta_vec[:, 0], delta_vec[:, 1], 
        angles='xy', 
        scale_units='xy', 
        scale=vector_scale, 
        color=vector_color, 
        alpha=vector_alpha,
        label="Perturbation Vectors"
    )
    
    # Formatting
    ax.set_xlabel("Embedding Dimension 1")
    ax.set_ylabel("Embedding Dimension 2")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
    
    # Show if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_smoothed_vector_field(
    emb: np.ndarray,
    delta_vec: np.ndarray,
    figsize: Tuple[float, float] = (8, 6),
    point_size: float = 5,
    point_alpha: float = 0.5,
    vector_color: str = "red",
    vector_alpha: float = 0.7,
    vector_scale: float = 1,
    title: str = "Smoothed Perturbation Vector Field",
    save_path: Optional[str] = None,
    show_plot: bool = True,
    dpi: int = 300
) -> plt.Figure:
    """
    Plot smoothed vector field showing perturbation vectors.
    
    Parameters
    ----------
    emb : np.ndarray
        2D embedding coordinates of shape (n_cells, 2).
    delta_vec : np.ndarray
        Smoothed perturbation vectors of shape (n_cells, 2).
    figsize : Tuple[float, float], default=(8, 6)
        Figure size as (width, height) in inches.
    point_size : float, default=5
        Size of cell points.
    point_alpha : float, default=0.5
        Transparency of cell points.
    vector_color : str, default="red"
        Color of vector arrows.
    vector_alpha : float, default=0.7
        Transparency of vector arrows.
    vector_scale : float, default=1
        Scale factor for vector arrows.
    title : str, default="Smoothed Perturbation Vector Field"
        Title for the plot.
    save_path : str, optional
        Path to save the figure.
    show_plot : bool, default=True
        Whether to display the plot.
    dpi : int, default=300
        Resolution for saved figure.
        
    Returns
    -------
    plt.Figure
        The matplotlib figure object.
        
    Raises
    ------
    ValueError
        If input arrays have incompatible shapes.
    """
    # This function has identical implementation to plot_raw_vector_field
    # but with different default title for clarity
    return plot_raw_vector_field(
        emb=emb,
        delta_vec=delta_vec,
        figsize=figsize,
        point_size=point_size,
        point_alpha=point_alpha,
        vector_color=vector_color,
        vector_alpha=vector_alpha,
        vector_scale=vector_scale,
        title=title,
        save_path=save_path,
        show_plot=show_plot,
        dpi=dpi
    )