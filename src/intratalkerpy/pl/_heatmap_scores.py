import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import warnings


def _process_heatmap_data(
    receptor_scores: Dict[str, Dict[str, Any]], 
    exclude_cell_types: list,
    max_score_limit: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process receptor scores data for heatmap visualization.
    
    Parameters
    ----------
    receptor_scores : Dict[str, Dict[str, Any]]
        Dictionary containing receptor scores and p-values.
    exclude_cell_types : list
        List of cell types to exclude.
    max_score_limit : float
        Maximum absolute value for score clipping.
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Processed scores and p-values DataFrames.
    """
    valid_receptors = []
    for receptor, values in receptor_scores.items():
        if "scores" in values and "p_vals" in values:
            valid_receptors.append(receptor)
        else:
            warnings.warn(f"Missing scores or p_vals for receptor {receptor}")
    
    if not valid_receptors:
        return pd.DataFrame(), pd.DataFrame()
    
    try:
        scores_dict = {}
        p_vals_dict = {}
        
        for receptor in valid_receptors:
            scores = receptor_scores[receptor]["scores"]
            p_vals = receptor_scores[receptor]["p_vals"]
            
            filtered_scores = {k: v for k, v in scores.items() if k not in exclude_cell_types}
            filtered_p_vals = {k: v for k, v in p_vals.items() if k not in exclude_cell_types}
            
            scores_dict[receptor] = filtered_scores
            p_vals_dict[receptor] = filtered_p_vals
        
        scores_df = pd.DataFrame(scores_dict).T
        p_values_df = pd.DataFrame(p_vals_dict).T
        
        scores_df = scores_df.fillna(0)
        p_values_df = p_values_df.fillna(1.0)
        
        scores_df = np.clip(scores_df, -max_score_limit, max_score_limit)
        
        return scores_df, p_values_df
        
    except Exception as e:
        warnings.warn(f"Error processing heatmap data: {e}")
        return pd.DataFrame(), pd.DataFrame()


def _create_significance_mask(
    scores_df: pd.DataFrame, 
    p_values_df: pd.DataFrame,
    method: str,
    score_thresholds: Tuple[float, float, float],
    pval_thresholds: Tuple[float, float, float]
) -> pd.DataFrame:
    """
    Create significance annotation mask for heatmap.
    
    Parameters
    ----------
    scores_df : pd.DataFrame
        Scores DataFrame.
    p_values_df : pd.DataFrame
        P-values DataFrame.
    method : str
        Significance method: "score" or "pval".
    score_thresholds : Tuple[float, float, float]
        Thresholds for score-based significance.
    pval_thresholds : Tuple[float, float, float]
        Thresholds for p-value-based significance.
        
    Returns
    -------
    pd.DataFrame
        Significance mask with annotation strings.
    """
    if method == "score":
        mask = scores_df.applymap(
            lambda x: "***" if abs(x) > score_thresholds[2] 
                    else "**" if abs(x) > score_thresholds[1] 
                    else "*" if abs(x) > score_thresholds[0] 
                    else ""
        )
    else:
        mask = p_values_df.applymap(
            lambda p: "***" if p < pval_thresholds[2] 
                    else "**" if p < pval_thresholds[1] 
                    else "*" if p < pval_thresholds[0] 
                    else ""
        )
    
    return mask


def plot_score_heatmap(
    receptor_scores: Dict[str, Dict[str, Any]], 
    result_path: str,
    max_score_limit: float = 3.0,
    figsize: Tuple[float, float] = (12, 8),
    cmap: str = "coolwarm",
    significance_method: str = "score",
    score_thresholds: Tuple[float, float, float] = (0.2, 0.5, 0.8),
    pval_thresholds: Tuple[float, float, float] = (0.05, 0.01, 0.001),
    cluster_rows: bool = True,
    cluster_cols: bool = True,
    dendrogram_ratio: Tuple[float, float] = (0.2, 0.2),
    linewidths: float = 0.5,
    save_csv: bool = True,
    save_format: str = "pdf",
    exclude_cell_types: Optional[list] = None,
    title: Optional[str] = None
) -> sns.matrix.ClusterGrid:
    """
    Create a clustered heatmap of receptor scores across cell types.
    
    Parameters
    ----------
    receptor_scores : Dict[str, Dict[str, Any]]
        Dictionary containing receptor scores and p-values.
        Expected structure: {receptor: {"scores": {cell_type: score}, "p_vals": {cell_type: pval}}}
    result_path : str
        Path where the plot and CSV will be saved.
    max_score_limit : float, default=3.0
        Maximum absolute value for score clipping.
    figsize : Tuple[float, float], default=(12, 8)
        Figure size as (width, height) in inches.
    cmap : str, default="coolwarm"
        Colormap for the heatmap.
    significance_method : str, default="score"
        Method for significance annotation: "score" (based on Cohen's d) or "pval" (based on p-values).
    score_thresholds : Tuple[float, float, float], default=(0.2, 0.5, 0.8)
        Thresholds for *, **, *** significance based on absolute Cohen's d values.
    pval_thresholds : Tuple[float, float, float], default=(0.05, 0.01, 0.001)
        Thresholds for *, **, *** significance based on p-values (from high to low).
    cluster_rows : bool, default=True
        Whether to cluster rows (receptors).
    cluster_cols : bool, default=True
        Whether to cluster columns (cell types).
    dendrogram_ratio : Tuple[float, float], default=(0.2, 0.2)
        Ratio of dendrogram size to main plot.
    linewidths : float, default=0.5
        Width of lines separating heatmap cells.
    save_csv : bool, default=True
        Whether to save the data as CSV.
    save_format : str, default="pdf"
        File format for saving the plot.
    exclude_cell_types : list, optional
        List of cell types to exclude from the heatmap.
    title : str, optional
        Custom title for the plot. If None, uses default title.
        
    Returns
    -------
    sns.matrix.ClusterGrid
        The seaborn ClusterGrid object.
        
    Raises
    ------
    ValueError
        If input data is invalid or empty.
    """
    if not receptor_scores:
        raise ValueError("receptor_scores cannot be empty")
    
    if max_score_limit <= 0:
        raise ValueError("max_score_limit must be positive")
        
    if significance_method not in ["score", "pval"]:
        raise ValueError("significance_method must be 'score' or 'pval'")
    
    if len(score_thresholds) != 3 or len(pval_thresholds) != 3:
        raise ValueError("Threshold tuples must contain exactly 3 values")
    
    if exclude_cell_types is None:
        exclude_cell_types = ['all_cluster']
    
    if title is None:
        title = "Clustered Cohen's d by Receptor and Cell Type"
    
    
    scores_df, p_values_df = _process_heatmap_data(
        receptor_scores, exclude_cell_types, max_score_limit
    )
    
    if scores_df.empty:
        warnings.warn("No data remaining after processing. Cannot create heatmap.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No data to display', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return fig
    
    if save_csv:
        _save_combined_data(scores_df, p_values_df, result_path)
    
    significance_mask = _create_significance_mask(
        scores_df, p_values_df, significance_method, score_thresholds, pval_thresholds
    )
    
    try:
        fig_clustermap = sns.clustermap(
            scores_df,
            annot=significance_mask,
            fmt="",
            cmap=cmap,
            cbar_kws={"label": "Cohen's d"},
            linewidths=linewidths,
            figsize=figsize,
            row_cluster=cluster_rows,
            col_cluster=cluster_cols,
            dendrogram_ratio=dendrogram_ratio,
            xticklabels=True,
            yticklabels=True
        )
        
        fig_clustermap.fig.suptitle(title, fontsize=16, y=0.98)
        
        result_path = Path(result_path)
        result_path.mkdir(parents=True, exist_ok=True)
        
        save_path = result_path / f"clustered_heatmap.{save_format}"
        fig_clustermap.savefig(save_path, bbox_inches="tight", dpi=300)
        
        plt.show()
        
        return fig_clustermap
        
    except Exception as e:
        warnings.warn(f"Error creating clustermap: {e}")
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(scores_df, annot=significance_mask, fmt="", cmap=cmap, ax=ax)
        ax.set_title(title)
        plt.tight_layout()
        plt.show()
        return fig


def _save_combined_data(
    scores_df: pd.DataFrame, 
    p_values_df: pd.DataFrame, 
    result_path: str
) -> None:
    """
    Save combined scores and p-values data to CSV.
    
    Parameters
    ----------
    scores_df : pd.DataFrame
        Scores DataFrame.
    p_values_df : pd.DataFrame
        P-values DataFrame.
    result_path : str
        Path to save the CSV file.
    """
    try:
        scores_df_suffixed = scores_df.add_suffix("_score")
        p_values_df_suffixed = p_values_df.add_suffix("_p-val")
        
        combined_df = pd.concat([scores_df_suffixed, p_values_df_suffixed], axis=1)
        combined_df = combined_df.reindex(sorted(combined_df.columns), axis=1)
        
        result_path = Path(result_path)
        result_path.mkdir(parents=True, exist_ok=True)
        csv_path = result_path / "receptor_scores.csv"
        combined_df.to_csv(csv_path)
        
    except Exception as e:
        warnings.warn(f"Error saving CSV data: {e}")