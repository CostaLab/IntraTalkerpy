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


def subset_list(target_list, index_list):
    X = list(map(target_list.__getitem__, index_list))
    return X


def plot_metadata_given_ax(anndata,
                        ax: matplotlib.axes,
                        reduction_name:str,
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
        except BaseException:
            random.seed(seed)
            color = list(map(
                lambda i: "#" +
                        "%06x" % random.randint(
                    0, 0xFFFFFF), range(len(categories))
            ))
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
        ax.scatter(embedding.iloc[o, 0], embedding.iloc[o, 1], c=subset_list(
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

def vector_field_wrapper(adata, grid, vectors, distances, red_name, cell_anno, receptor, color_dict, ax, stream_density=1, zorder=10, grid_dist=25):
    norm = matplotlib.colors.Normalize(vmin=0.15, vmax=0.5, clip=True)
    scale = lambda X: [(x - min(X)) / (max(X) - min(X)) for x in X]
    ax2 = ax
    ax2 = plot_metadata_given_ax(
        anndata=adata,
        reduction_name=red_name,
        receptor=receptor,
        ax=ax2,
        variable=cell_anno,
        color_dictionary=color_dict,
        show_label=True)
    ax2.streamplot(
        grid.reshape(grid_dist, grid_dist, 2)[:, :, 0],
        grid.reshape(grid_dist, grid_dist, 2)[:, :, 1],
        vectors.reshape(grid_dist, grid_dist, 2)[:, :, 0],
        vectors.reshape(grid_dist, grid_dist, 2)[:, :, 1],
        density=stream_density,
        color=np.array(scale(distances)).reshape(grid_dist, grid_dist),
        cmap='Greys',
        zorder=zorder,
        norm=norm,
        linewidth=1.2)
    return ax2

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