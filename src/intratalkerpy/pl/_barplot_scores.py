import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import warnings


def _process_receptor_data(
    receptor_scores: Dict[str, Dict[str, Any]], 
    exclude_cell_types: list, 
    cohens_d_threshold: float
) -> pd.DataFrame:
    """
    Process receptor scores data into a DataFrame for plotting.
    
    Parameters
    ----------
    receptor_scores : Dict[str, Dict[str, Any]]
        Dictionary containing receptor scores and p-values.
    exclude_cell_types : list
        List of cell types to exclude.
    cohens_d_threshold : float
        Minimum absolute Cohen's d value to include.
        
    Returns
    -------
    pd.DataFrame
        Processed DataFrame with Receptor, Cell_Type, score, and pval columns.
    """
    rows = []
    for receptor, values in receptor_scores.items():
        if "scores" not in values or "p_vals" not in values:
            warnings.warn(f"Missing scores or p_vals for receptor {receptor}")
            continue
            
        for cell_type, score in values["scores"].items():
            if cell_type in exclude_cell_types:
                continue
                
            pval = values["p_vals"].get(cell_type, 1.0)  # Default p-value if missing
            rows.append({
                "Receptor": receptor, 
                "Cell_Type": cell_type, 
                "score": score, 
                "pval": pval
            })
    
    if not rows:
        return pd.DataFrame()
    
    reshaped_df = pd.DataFrame(rows)
    reshaped_df = reshaped_df[abs(reshaped_df['score']) > cohens_d_threshold]
    
    return reshaped_df


def plot_score_barplots(
    receptor_scores: Dict[str, Dict[str, Any]], 
    result_path: str, 
    cohens_d_threshold: float = 0.5, 
    fig_width: float = 15, 
    fig_height: float = 6, 
    top_n: int = 10,
    x_limits: Optional[Tuple[float, float]] = None,
    palette: str = "Blues_r",
    exclude_cell_types: Optional[list] = None,
    save_format: str = "pdf"
) -> plt.Figure:
    """
    Plot barplots of receptor scores for different cell types.
    
    Parameters
    ----------
    receptor_scores : Dict[str, Dict[str, Any]]
        Dictionary containing receptor scores and p-values.
        Expected structure: {receptor: {"scores": {cell_type: score}, "p_vals": {cell_type: pval}}}
    result_path : str
        Path where the plot will be saved.
    cohens_d_threshold : float, default=0.5
        Minimum absolute Cohen's d value to include in the plot.
    fig_width : float, default=15
        Width of the figure in inches.
    fig_height : float, default=6
        Height of the figure in inches.
    top_n : int, default=10
        Number of top receptors to show per cell type.
    x_limits : Tuple[float, float], optional
        Custom x-axis limits. If None, uses (-3, 3).
    palette : str, default="Blues_r"
        Color palette for the barplots.
    exclude_cell_types : list, optional
        List of cell types to exclude from plotting.
    save_format : str, default="pdf"
        File format for saving the plot.
        
    Returns
    -------
    plt.Figure
        The matplotlib figure object.
        
    Raises
    ------
    ValueError
        If input data is invalid or empty.
    """
    if not receptor_scores:
        raise ValueError("receptor_scores cannot be empty")
    
    if cohens_d_threshold < 0:
        raise ValueError("cohens_d_threshold must be non-negative")
        
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    
    if exclude_cell_types is None:
        exclude_cell_types = ['all_cluster']
    
    if x_limits is None:
        x_limits = (-3, 3)
    
    reshaped_df = _process_receptor_data(receptor_scores, exclude_cell_types, cohens_d_threshold)
    
    if reshaped_df.empty:
        warnings.warn("No data remaining after filtering. Returning empty plot.")
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.text(0.5, 0.5, 'No data to display', ha='center', va='center', transform=ax.transAxes)
        return fig
    
    cell_types = reshaped_df['Cell_Type'].unique()
    n_cols = len(cell_types)
    
    if n_cols == 1:
        fig_bar_vol, axes = plt.subplots(1, 1, figsize=(fig_width, fig_height))
        axes = [axes] 
    else:
        fig_bar_vol, axes = plt.subplots(1, n_cols, figsize=(fig_width, fig_height), sharex=False, sharey=False)
    
    for i, cell_type in enumerate(cell_types):
        ax_bar = axes[i]
        df_loop = reshaped_df[reshaped_df['Cell_Type'] == cell_type].copy()
        df_loop = df_loop.sort_values('score', key=abs, ascending=False)
        df_loop = df_loop.head(top_n)
        
        if df_loop.empty:
            ax_bar.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax_bar.transAxes)
            ax_bar.set_title(f"{cell_type} - Bar Plot")
            continue
        
        sns.barplot(
            data=df_loop,
            x='score',
            y='Receptor',
            ax=ax_bar,
            palette=palette
        )
        ax_bar.set_title(f"{cell_type} - Bar Plot")
        ax_bar.set_xlabel("Cohen's d")
        ax_bar.set_ylabel("Receptor")
        ax_bar.set_xlim(x_limits)
        
        # if 'pval' in df_loop.columns:
        #     _add_significance_markers(ax_bar, df_loop)
    
    plt.tight_layout()
    
    result_path = Path(result_path)
    result_path.mkdir(parents=True, exist_ok=True)
    
    save_path = result_path / f"Scores_Barplot.{save_format}"
    fig_bar_vol.savefig(save_path, bbox_inches="tight", dpi=300)
    
    plt.show(fig_bar_vol)
    plt.close(fig_bar_vol)
    
    return fig_bar_vol


def _add_significance_markers(ax: plt.Axes, df: pd.DataFrame) -> None:
    """
    Add significance markers to barplot based on p-values.
    
    Parameters
    ----------
    ax : plt.Axes
        The axes object to add markers to.
    df : pd.DataFrame
        DataFrame with 'pval' and 'score' columns.
    """
    for i, (_, row) in enumerate(df.iterrows()):
        pval = row['pval']
        score = row['score']
        
        if pval < 0.001:
            marker = '***'
        elif pval < 0.01:
            marker = '**'
        elif pval < 0.05:
            marker = '*'
        else:
            continue
            
        x_pos = score + (0.1 if score > 0 else -0.1)
        ax.text(x_pos, i, marker, ha='center', va='center', fontweight='bold')