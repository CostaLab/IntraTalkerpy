import os
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, Union

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

# Import utility functions from our package
from ..ut import (
    compute_pseudotime_gradient,
    calculate_grid_arrows,
    compute_differential_pseudotime,
    calculate_effect_size_from_differences
)

def differential_pseudotime_analysis(
    adata: AnnData,
    folder_path: Union[str, Path],
    save_path: Union[str, Path], 
    red_namem: str, 
    cell_anno: str, 
    pseudo_name: str, 
    grid_size: int = 25, 
    offset_frac: float = 0.005,
    n_neigh: int = 10,
    n_cpu: int = 1
) -> AnnData:
    """
    Perform differential pseudotime analysis for receptor-ligand interactions.
    
    This function analyzes how receptor-ligand interactions affect pseudotime
    trajectories by computing vector fields and differential pseudotime changes
    for each receptor from CSV files containing perturbation data.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object containing single-cell data.
    folder_path : str or Path
        Path to folder containing CSV files with receptor perturbation data.
        Each CSV file should be named "{receptor}_*.csv".
    save_path : str or Path
        Path where the output AnnData object will be saved.
    red_namem : str
        Key in adata.obsm containing the embedding coordinates (e.g., 'X_umap').
    cell_anno : str
        Key in adata.obs containing cell type annotations.
    pseudo_name : str
        Key in adata.obs containing pseudotime values.
    grid_size : int, default=25
        Size of the grid for vector field calculation (both rows and columns).
    offset_frac : float, default=0.005
        Fraction of embedding range to use as offset from boundaries.
    n_neigh : int, default=10
        Number of nearest neighbors for gradient computation.
    n_cpu : int, default=1
        Number of CPU cores to use for parallel processing.
        
    Returns
    -------
    AnnData
        Updated AnnData object with:
        - New columns in .obs: 'pseudotime_{receptor}' for each receptor
        - .uns['receptor_scores']: Dictionary with scores and p-values
        - .uns['vector_fields']: Dictionary with vector field data
        
    Raises
    ------
    FileNotFoundError
        If folder_path doesn't exist or contains no CSV files.
    KeyError
        If required keys are missing from adata.
    ValueError
        If input parameters are invalid.
    """
    # Input validation
    folder_path = Path(folder_path)
    save_path = Path(save_path)
    
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder path does not exist: {folder_path}")
    
    if red_namem not in adata.obsm:
        raise KeyError(f"Embedding '{red_namem}' not found in adata.obsm")
    
    if pseudo_name not in adata.obs:
        raise KeyError(f"Pseudotime '{pseudo_name}' not found in adata.obs")
    
    if cell_anno not in adata.obs:
        raise KeyError(f"Cell annotation '{cell_anno}' not found in adata.obs")
    
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    
    if not (0 < offset_frac < 0.5):
        raise ValueError("offset_frac must be between 0 and 0.5")
    
    if n_neigh <= 0:
        raise ValueError("n_neigh must be positive")
    
    try:
        # Extract data
        X_emb = adata.obsm[red_namem]
        pseudotime = adata.obs[pseudo_name].values
        
        # Compute pseudotime gradient vectors
        gradient_vectors = compute_pseudotime_gradient(X_emb, pseudotime, n_neigh=n_neigh)
        
        # Find and process CSV files
        csv_files = [f for f in folder_path.glob("*.csv")]
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {folder_path}")
        
        print(f"Found {len(csv_files)} CSV files to process")
        
        log_odds_ratios = {}
        vector_fields = {}
        
        for csv_file in csv_files:
            receptor = csv_file.stem.split("_")[0]
            print(f"Processing receptor: {receptor}")
            
            try:
                # Load receptor perturbation data
                delta_embedding = pd.read_csv(csv_file, index_col=0)
                delta_X = delta_embedding.values
                
                # Validate data dimensions
                if delta_X.shape[0] != X_emb.shape[0]:
                    warnings.warn(f"Dimension mismatch for {receptor}: {delta_X.shape[0]} vs {X_emb.shape[0]}")
                    continue
                
                # Calculate vector field
                grid_xy, uv, mask = calculate_grid_arrows(
                    embedding=X_emb,
                    delta_embedding=delta_X,
                    offset_frac=offset_frac,
                    n_grid_cols=grid_size,
                    n_grid_rows=grid_size,
                    n_neighbors=grid_size,
                    n_cpu=n_cpu
                )
                
                # Process vector field
                distances = np.sqrt((uv ** 2).sum(1))
                distances = np.nan_to_num(distances)
                
                # Apply distance-based filtering
                uv_filtered = _apply_vector_filtering(uv, mask, distances)
                
                # Store vector field data
                vector_fields[receptor] = {
                    'grid_points': grid_xy,
                    'vectors': uv_filtered,
                    'distances': distances,
                    'original_vectors': delta_X,
                    'mask': mask
                }
                
                # Compute differential pseudotime
                delta_T = compute_differential_pseudotime(X_emb, grid_xy, uv_filtered, gradient_vectors)
                
                # Store results in adata
                pseudotime_field = f'pseudotime_{receptor}'
                adata.obs[pseudotime_field] = adata.obs[pseudo_name] + delta_T
                
                # Calculate scores for all cells and by cell type
                scores, p_vals = _calculate_receptor_scores(
                    adata, receptor, pseudo_name, cell_anno
                )
                
                log_odds_ratios[receptor] = {"scores": scores, "p_vals": p_vals}
                
                print(f"Completed {receptor}: all_cluster score = {scores.get('all_cluster', 'N/A')}")
                
            except Exception as e:
                warnings.warn(f"Error processing {receptor}: {e}")
                continue
        
        # Store results
        adata.uns["receptor_scores"] = log_odds_ratios
        adata.uns["vector_fields"] = vector_fields
        
        # Save results
        save_path.mkdir(parents=True, exist_ok=True)
        output_file = save_path / "heart_data_fibroblasts12_im_dpt_pseudotime.h5ad"
        adata.write(output_file)
        print(f"Results saved to: {output_file}")
        
        return adata
        
    except Exception as e:
        raise RuntimeError(f"Error in differential pseudotime analysis: {e}")


def _apply_vector_filtering(uv: np.ndarray, mask: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """
    Apply filtering to vector field based on mask and distance scaling.
    
    Parameters
    ----------
    uv : np.ndarray
        Vector field data.
    mask : np.ndarray
        Boolean mask for valid grid points.
    distances : np.ndarray
        Vector magnitudes.
        
    Returns
    -------
    np.ndarray
        Filtered vector field.
    """
    try:
        uv_filtered = uv.copy()
        
        # Scale distances to [0, 1] range
        if distances.max() > distances.min():
            scaled_distances = (distances - distances.min()) / (distances.max() - distances.min())
        else:
            scaled_distances = np.ones_like(distances)
        
        # Apply filtering: remove vectors that are masked or have low scaled distance
        filter_condition = np.logical_or(~mask, scaled_distances < 0.15)
        uv_filtered[filter_condition] = np.nan
        
        return uv_filtered
        
    except Exception as e:
        warnings.warn(f"Error in vector filtering: {e}")
        return uv


def _calculate_receptor_scores(
    adata: AnnData, 
    receptor: str, 
    pseudo_name: str, 
    cell_anno: str
) -> tuple:
    """
    Calculate scores and p-values for receptor effects on pseudotime.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object.
    receptor : str
        Receptor name.
    pseudo_name : str
        Original pseudotime column name.
    cell_anno : str
        Cell annotation column name.
        
    Returns
    -------
    tuple
        (scores, p_vals) dictionaries.
    """
    try:
        pseudotime_field = f'pseudotime_{receptor}'
        pseudotime_diff = adata.obs[pseudotime_field] - adata.obs[pseudo_name]
        
        scores = {}
        p_vals = {}
        
        # Calculate scores for all cells
        all_score, all_pval = calculate_effect_size_from_differences(pseudotime_diff)
        scores["all_cluster"] = all_score
        p_vals["all_cluster"] = all_pval
        
        # Calculate scores by cell type
        for cell_type in adata.obs[cell_anno].unique():
            if pd.isna(cell_type):
                continue
                
            try:
                cell_mask = adata.obs[cell_anno] == cell_type
                subset_diff = pseudotime_diff[cell_mask]
                
                if len(subset_diff) > 0:
                    score, pval = calculate_effect_size_from_differences(subset_diff)
                    scores[cell_type] = score
                    p_vals[cell_type] = pval
                    
            except Exception as e:
                warnings.warn(f"Error calculating scores for {cell_type}: {e}")
                continue
        
        return scores, p_vals
        
    except Exception as e:
        warnings.warn(f"Error calculating receptor scores: {e}")
        return {}, {}