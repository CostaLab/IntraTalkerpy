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


def _subset_list(target_list: List, index_list: List[int]) -> List:
    """
    Subset a list based on index list.
    
    Parameters
    ----------
    target_list : List
        Source list to subset.
    index_list : List[int]
        Indices to extract.
        
    Returns
    -------
    List
        Subsetted list.
    """
    return [target_list[i] for i in index_list]


def _generate_colors(categories: List[str], seed: int = 555) -> Dict[str, str]:
    """
    Generate random colors for categories.
    
    Parameters
    ----------
    categories : List[str]
        List of category names.
    seed : int, default=555
        Random seed for reproducibility.
        
    Returns
    -------
    Dict[str, str]
        Dictionary mapping categories to hex colors.
    """
    random.seed(seed)
    colors = [f"#{random.randint(0, 0xFFFFFF):06x}" for _ in range(len(categories))]
    return dict(zip(categories, colors))


def _validate_anndata_inputs(anndata: Any, reduction_name: str, variable: str) -> None:
    """
    Validate AnnData object and required attributes.
    
    Parameters
    ----------
    anndata : AnnData
        Annotated data object.
    reduction_name : str
        Name of the reduction in obsm.
    variable : str
        Variable name in obs.
        
    Raises
    ------
    ValueError
        If required attributes are missing.
    """
    if not hasattr(anndata, 'obs'):
        raise ValueError("anndata must have an 'obs' attribute")
    
    if not hasattr(anndata, 'obsm'):
        raise ValueError("anndata must have an 'obsm' attribute")
    
    if reduction_name not in anndata.obsm.keys():
        raise ValueError(f"Reduction '{reduction_name}' not found in anndata.obsm")
    
    if variable not in anndata.obs.columns:
        raise ValueError(f"Variable '{variable}' not found in anndata.obs")


def plot_metadata_given_ax(
    anndata: Any,
    ax: matplotlib.axes.Axes,
    reduction_name: str,
    variable: str,
    color_dictionary: Dict[str, Dict[str, str]],
    receptor: str,
    remove_nan: bool = True,
    show_label: bool = True,
    show_legend: bool = False,
    cmap: Union[str, matplotlib.colors.Colormap] = cm.viridis,
    dot_size: int = 10,
    text_size: int = 10,
    alpha: float = 1.0,
    seed: int = 555,
    selected_cells: Optional[List[str]] = None,
    title: Optional[str] = None,
    legend_bbox: Tuple[float, float] = (1.04, 1),
    legend_loc: str = "upper left"
) -> matplotlib.axes.Axes:
    """
    Plot metadata on a 2D embedding with enhanced customization and error handling.
    
    Parameters
    ----------
    anndata : AnnData
        Annotated data object containing embedding and metadata.
    ax : matplotlib.axes.Axes
        Matplotlib axes object to plot on.
    reduction_name : str
        Name of the reduction/embedding in anndata.obsm.
    variable : str
        Variable name to plot from anndata.obs.
    color_dictionary : Dict[str, Dict[str, str]]
        Nested dictionary with color mappings for variables.
    receptor : str
        Name of the receptor being analyzed.
    remove_nan : bool, default=True
        Whether to remove NaN values from categorical data.
    show_label : bool, default=True
        Whether to show category labels on the plot.
    show_legend : bool, default=False
        Whether to display legend.
    cmap : Union[str, matplotlib.colors.Colormap], default=cm.viridis
        Colormap for continuous variables.
    dot_size : int, default=10
        Size of scatter plot points.
    text_size : int, default=10
        Size of label text.
    alpha : float, default=1.0
        Transparency of points (0-1).
    seed : int, default=555
        Random seed for color generation.
    selected_cells : List[str], optional
        Subset of cells to plot.
    title : str, optional
        Custom title for the plot.
    legend_bbox : Tuple[float, float], default=(1.04, 1)
        Legend bounding box position.
    legend_loc : str, default="upper left"
        Legend location.
        
    Returns
    -------
    matplotlib.axes.Axes
        The modified axes object.
        
    Raises
    ------
    ValueError
        If required data is missing or invalid.
    """
    # Input validation
    _validate_anndata_inputs(anndata, reduction_name, variable)
    
    if not (0 <= alpha <= 1):
        raise ValueError("Alpha must be between 0 and 1")
    
    if dot_size <= 0:
        raise ValueError("Dot size must be positive")
    
    if text_size <= 0:
        raise ValueError("Text size must be positive")
    
    try:
        # Extract data
        cell_names = list(anndata.obs_names.copy())
        embeddings_dict = {
            k: pd.DataFrame(anndata.obsm[k].copy(), index=cell_names) 
            for k in anndata.obsm.keys()
        }
        
        embedding = embeddings_dict[reduction_name]
        data_mat = anndata.obs.copy()
        
        # Subset cells if specified
        if selected_cells is not None:
            missing_cells = set(selected_cells) - set(cell_names)
            if missing_cells:
                warnings.warn(f"Missing cells: {missing_cells}")
                selected_cells = [c for c in selected_cells if c in cell_names]
            
            if not selected_cells:
                raise ValueError("No valid selected cells found")
            
            data_mat = data_mat.loc[selected_cells]
            embedding = embedding.loc[selected_cells]
        
        # Align data
        common_index = embedding.index.intersection(data_mat.index)
        if len(common_index) == 0:
            raise ValueError("No common cells between embedding and metadata")
        
        data_mat = data_mat.loc[common_index]
        embedding = embedding.loc[common_index]
        
        # Process variable data
        var_data = data_mat[variable].copy()
        
        # Handle categorical vs continuous data
        if pd.api.types.is_string_dtype(var_data) or pd.api.types.is_categorical_dtype(var_data):
            return _plot_categorical_variable(
                ax, embedding, var_data, variable, color_dictionary, receptor,
                remove_nan, show_label, show_legend, dot_size, text_size, alpha,
                seed, title, legend_bbox, legend_loc
            )
        else:
            return _plot_continuous_variable(
                ax, embedding, var_data, variable, cmap, dot_size, alpha, title
            )
            
    except Exception as e:
        warnings.warn(f"Error in plot_metadata_given_ax: {e}")
        ax.text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=ax.transAxes)
        return ax


def _plot_categorical_variable(
    ax: matplotlib.axes.Axes,
    embedding: pd.DataFrame,
    var_data: pd.Series,
    variable: str,
    color_dictionary: Dict[str, Dict[str, str]],
    receptor: str,
    remove_nan: bool,
    show_label: bool,
    show_legend: bool,
    dot_size: int,
    text_size: int,
    alpha: float,
    seed: int,
    title: Optional[str],
    legend_bbox: Tuple[float, float],
    legend_loc: str
) -> matplotlib.axes.Axes:
    """Plot categorical variable on embedding."""
    
    # Handle NaN values
    if remove_nan and var_data.isnull().any():
        valid_mask = ~var_data.isnull()
        var_data = var_data[valid_mask]
        embedding = embedding.loc[valid_mask]
        
        if len(var_data) == 0:
            warnings.warn("No valid data after removing NaN values")
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
            return ax
    else:
        var_data = var_data.fillna('NA')
    
    # Get categories and colors
    categories = list(set(var_data.astype(str)))
    
    try:
        color_dict = color_dictionary if isinstance(color_dictionary, dict) else color_dictionary.get(variable, {})
        missing_colors = set(categories) - set(color_dict.keys())
        if missing_colors:
            additional_colors = _generate_colors(list(missing_colors), seed)
            color_dict.update(additional_colors)
    except Exception:
        color_dict = _generate_colors(categories, seed)
    
    # Create scatter plot
    colors = var_data.astype(str).map(color_dict)
    ax.scatter(
        embedding.iloc[:, 0], 
        embedding.iloc[:, 1], 
        c=colors, 
        s=dot_size, 
        alpha=alpha
    )
    
    # Set labels
    ax.set_xlabel(embedding.columns[0])
    ax.set_ylabel(embedding.columns[1])
    
    # Add category labels
    if show_label:
        _add_category_labels(ax, embedding, var_data, color_dict, text_size)
    
    # Set title
    if title is None:
        title = f"{receptor} Vector Field"
    ax.set_title(title)
    
    # Add legend
    if show_legend:
        _add_legend(ax, color_dict, legend_bbox, legend_loc)
    
    return ax


def _plot_continuous_variable(
    ax: matplotlib.axes.Axes,
    embedding: pd.DataFrame,
    var_data: pd.Series,
    variable: str,
    cmap: Union[str, matplotlib.colors.Colormap],
    dot_size: int,
    alpha: float,
    title: Optional[str]
) -> matplotlib.axes.Axes:
    """Plot continuous variable on embedding."""
    
    # Remove NaN values
    valid_mask = ~var_data.isnull()
    if not valid_mask.any():
        warnings.warn("All values are NaN")
        ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
        return ax
    
    if not valid_mask.all():
        warnings.warn(f"Removing {(~valid_mask).sum()} NaN values")
        var_data = var_data[valid_mask]
        embedding = embedding.loc[valid_mask]
    
    # Sort data for better visualization
    sort_order = np.argsort(var_data.values)
    
    # Create scatter plot
    scatter = ax.scatter(
        embedding.iloc[sort_order, 0], 
        embedding.iloc[sort_order, 1], 
        c=_subset_list(var_data.tolist(), sort_order), 
        cmap=cmap, 
        s=dot_size, 
        alpha=alpha
    )
    
    # Set labels
    ax.set_xlabel(embedding.columns[0])
    ax.set_ylabel(embedding.columns[1])
    
    # Set title
    if title is None:
        title = variable
    ax.set_title(title)
    
    # Add colorbar
    normalize = mcolors.Normalize(vmin=var_data.min(), vmax=var_data.max())
    scalar_mappable = cm.ScalarMappable(norm=normalize, cmap=cmap)
    scalar_mappable.set_array(var_data.values)
    plt.colorbar(scalar_mappable, ax=ax)
    
    return ax


def _add_category_labels(
    ax: matplotlib.axes.Axes,
    embedding: pd.DataFrame,
    var_data: pd.Series,
    color_dict: Dict[str, str],
    text_size: int
) -> None:
    """Add category labels to the plot."""
    try:
        # Calculate label positions (centroids)
        label_data = pd.concat([embedding, var_data], axis=1)
        label_positions = label_data.groupby(var_data.name).agg({
            embedding.columns[0]: 'mean',
            embedding.columns[1]: 'mean'
        })
        
        # Add text labels
        texts = []
        for label in label_positions.index:
            if label in color_dict:
                texts.append(
                    ax.text(
                        label_positions.loc[label, embedding.columns[0]],
                        label_positions.loc[label, embedding.columns[1]],
                        str(label),
                        horizontalalignment='center',
                        verticalalignment='center',
                        size=text_size,
                        weight='bold',
                        color=color_dict[label],
                        path_effects=[
                            PathEffects.withStroke(linewidth=3, foreground='w')
                        ]
                    )
                )
        
        # Adjust text positions to avoid overlap
        if texts:
            adjust_text(texts)
            
    except Exception as e:
        warnings.warn(f"Error adding category labels: {e}")


def _add_legend(
    ax: matplotlib.axes.Axes,
    color_dict: Dict[str, str],
    legend_bbox: Tuple[float, float],
    legend_loc: str
) -> None:
    """Add legend to the plot."""
    try:
        patches = [
            mpatches.Patch(color=color, label=label)
            for label, color in color_dict.items()
        ]
        ax.legend(
            handles=patches, 
            bbox_to_anchor=legend_bbox, 
            loc=legend_loc
        )
    except Exception as e:
        warnings.warn(f"Error adding legend: {e}")

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
        Annotated data object.
    grid : np.ndarray
        Grid points for the vector field.
    vectors : np.ndarray
        Vector field values at grid points.
    distances : np.ndarray
        Distance values for coloring streamlines.
    red_name : str
        Name of the reduction/embedding.
    cell_anno : str
        Cell annotation variable name.
    receptor : str
        Receptor name for labeling.
    color_dict : Dict[str, str]
        Color dictionary mapping cell types to colors.
        Should be a flat dictionary like {'cell_type1': '#color1', 'cell_type2': '#color2'}
    ax : matplotlib.axes.Axes
        Matplotlib axes object.
    stream_density : float, default=1.0
        Density of streamlines.
    zorder : int, default=10
        Z-order for streamline plotting.
    grid_dist : int, default=25
        Grid distance parameter.
    norm_vmin : float, default=0.15
        Minimum value for normalization.
    norm_vmax : float, default=0.5
        Maximum value for normalization.
    stream_cmap : str, default='Greys'
        Colormap for streamlines.
    linewidth : float, default=1.2
        Width of streamlines.
    show_label : bool, default=True
        Whether to show cell type labels.
        
    Returns
    -------
    matplotlib.axes.Axes
        The modified axes object.
        
    Raises
    ------
    ValueError
        If input arrays have incompatible shapes.
    """
    # Input validation
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