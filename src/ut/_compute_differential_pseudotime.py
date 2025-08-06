import numpy as np
import warnings
from sklearn.neighbors import NearestNeighbors
from typing import List, Optional, Union, Dict
import matplotlib.pyplot as plt

def compute_differential_pseudotime(emb, grid_xy, uv, gradient_vectors):
    """
    Compute differential pseudotime changes based on vector field analysis.
    
    This function calculates the pseudotime differential by projecting velocity
    vectors onto gradient vectors at each point in the embedding space.
    
    Parameters
    ----------
    emb : array_like
        Cell embedding coordinates with shape (n_cells, 2).
    grid_xy : array_like
        Grid point coordinates with shape (n_grid_points, 2).
    uv : array_like
        Velocity vectors at grid points with shape (n_grid_points, 2).
    gradient_vectors : array_like
        Gradient vectors at cell positions with shape (n_cells, 2).
        
    Returns
    -------
    np.ndarray
        Differential pseudotime values with shape (n_cells,).
    """
    try:
        # Input validation and conversion
        emb = np.asarray(emb)
        grid_xy = np.asarray(grid_xy)
        uv = np.asarray(uv)
        gradient_vectors = np.asarray(gradient_vectors)
        
        # Validate input shapes
        if not _validate_inputs(emb, grid_xy, uv, gradient_vectors):
            warnings.warn("Invalid input shapes, returning zeros")
            return np.zeros(len(emb))
        
        # Filter out invalid velocity vectors
        mask = np.isfinite(uv).all(axis=-1)
        if not mask.any():
            warnings.warn("No valid velocity vectors found")
            return np.zeros(len(emb))
        
        filtered_grid = grid_xy[mask]
        filtered_vectors = uv[mask]
        
        # Prepare embedding array (handle different input formats)
        embedding_array = _prepare_embedding_array(emb)
        
        # Calculate adaptive threshold based on grid spacing
        threshold = _calculate_assignment_threshold(filtered_grid)
        
        # Find nearest grid points and assign vectors
        assigned_vectors = _assign_vectors_to_points(
            embedding_array, filtered_grid, filtered_vectors, threshold
        )
        
        # Compute differential pseudotime using dot product
        delta_T = _compute_pseudotime_differential(assigned_vectors, gradient_vectors)
        
        return delta_T
        
    except Exception as e:
        warnings.warn(f"Error in differential pseudotime computation: {e}")
        return np.zeros(len(emb) if hasattr(emb, '__len__') else 1)


def _validate_inputs(emb, grid_xy, uv, gradient_vectors):
    """
    Validate input arrays for compatibility.
    
    Parameters
    ----------
    emb : np.ndarray
        Cell embedding coordinates.
    grid_xy : np.ndarray
        Grid point coordinates.
    uv : np.ndarray
        Velocity vectors at grid points.
    gradient_vectors : np.ndarray
        Gradient vectors at cell positions.
        
    Returns
    -------
    bool
        True if inputs are valid, False otherwise.
    """
    try:
        # Check for empty arrays
        if (len(emb) == 0 or len(grid_xy) == 0 or 
            len(uv) == 0 or len(gradient_vectors) == 0):
            return False
        
        # Check dimensional consistency
        if len(grid_xy) != len(uv):
            return False
        
        if len(emb) != len(gradient_vectors):
            return False
        
        # Check that arrays are 2D with proper second dimension
        if (emb.ndim != 2 or grid_xy.ndim != 2 or 
            uv.ndim != 2 or gradient_vectors.ndim != 2):
            return False
        
        if (emb.shape[1] != 2 or grid_xy.shape[1] != 2 or 
            uv.shape[1] != 2 or gradient_vectors.shape[1] != 2):
            return False
        
        return True
        
    except Exception:
        return False


def _prepare_embedding_array(emb):
    """
    Prepare embedding array, handling different input formats.
    
    Parameters
    ----------
    emb : np.ndarray
        Raw embedding coordinates.
        
    Returns
    -------
    np.ndarray
        Processed embedding array with shape (n_cells, 2).
    """
    try:
        # Handle standard case
        if emb.shape[1] == 2:
            return emb.copy()
        
        # Handle case where columns need to be extracted and stacked
        return np.array([np.hstack(emb[:, 0]), np.hstack(emb[:, 1])]).T
        
    except Exception:
        # Fallback to original approach
        try:
            return np.array([np.hstack(emb[:, 0]), np.hstack(emb[:, 1])]).T
        except Exception:
            return emb.copy()


def _calculate_assignment_threshold(filtered_grid):
    """
    Calculate threshold for assigning vectors to embedding points.
    
    Parameters
    ----------
    filtered_grid : np.ndarray
        Grid coordinates with valid velocity vectors.
        
    Returns
    -------
    float
        Threshold distance for vector assignment.
    """
    try:
        n_points = len(filtered_grid)
        
        # Ensure we have enough points for nearest neighbor calculation
        if n_points < 2:
            warnings.warn("Insufficient grid points for threshold calculation")
            return 0.1  # Default threshold
        
        # Use minimum of available points and desired neighbors
        n_neighbors = min(2, n_points)
        
        # Calculate nearest neighbor distances
        nn_threshold = NearestNeighbors(n_neighbors=n_neighbors)
        nn_threshold.fit(filtered_grid)
        distances_grid, _ = nn_threshold.kneighbors(filtered_grid)
        
        # Use second nearest neighbor distance (or first if only one neighbor)
        if distances_grid.shape[1] > 1:
            threshold_distances = distances_grid[:, 1]
        else:
            threshold_distances = distances_grid[:, 0]
        
        # Calculate mean distance and apply factor
        d_mean = np.mean(threshold_distances)
        threshold = 0.5 * d_mean
        
        # Ensure reasonable threshold bounds
        if threshold <= 0 or not np.isfinite(threshold):
            threshold = 0.1
        
        # Prevent extremely large thresholds
        if threshold > 10.0:
            threshold = 1.0
        
        return threshold
        
    except Exception as e:
        warnings.warn(f"Error calculating threshold: {e}")
        return 0.1


def _assign_vectors_to_points(embedding_array, filtered_grid, filtered_vectors, threshold):
    """
    Assign velocity vectors to embedding points based on proximity.
    
    Parameters
    ----------
    embedding_array : np.ndarray
        Embedding coordinates.
    filtered_grid : np.ndarray
        Grid coordinates with valid vectors.
    filtered_vectors : np.ndarray
        Valid velocity vectors.
    threshold : float
        Distance threshold for assignment.
        
    Returns
    -------
    np.ndarray
        Assigned velocity vectors for each embedding point.
    """
    try:
        # Find nearest grid points
        nn_grid = NearestNeighbors(n_neighbors=1)
        nn_grid.fit(filtered_grid)
        distances_to_grid, indices = nn_grid.kneighbors(embedding_array)
        
        # Initialize with NaN values
        assigned_vectors = np.full_like(embedding_array, np.nan)
        
        # Assign vectors only for points within threshold distance
        close_enough = distances_to_grid[:, 0] < threshold
        
        if close_enough.any():
            assigned_indices = indices[close_enough, 0]
            assigned_vectors[close_enough] = filtered_vectors[assigned_indices]
        
        return assigned_vectors
        
    except Exception as e:
        warnings.warn(f"Error in vector assignment: {e}")
        return np.full_like(embedding_array, np.nan)


def _compute_pseudotime_differential(assigned_vectors, gradient_vectors):
    """
    Compute pseudotime differential from assigned vectors and gradients.
    
    Parameters
    ----------
    assigned_vectors : np.ndarray
        Velocity vectors assigned to each embedding point.
    gradient_vectors : np.ndarray
        Gradient vectors at each embedding point.
        
    Returns
    -------
    np.ndarray
        Differential pseudotime values.
    """
    try:
        # Find points with valid assigned vectors
        valid_mask = np.isfinite(assigned_vectors).all(axis=1)
        
        # Initialize result array
        delta_T = np.zeros(assigned_vectors.shape[0])
        
        if valid_mask.any():
            # Compute dot product between velocity and gradient vectors
            delta_T[valid_mask] = np.einsum(
                'ij,ij->i', 
                assigned_vectors[valid_mask], 
                gradient_vectors[valid_mask]
            )
        
        # Handle any NaN or infinite values
        delta_T = np.nan_to_num(delta_T, nan=0.0, posinf=0.0, neginf=0.0)
        
        return delta_T
        
    except Exception as e:
        warnings.warn(f"Error computing pseudotime differential: {e}")
        return np.zeros(len(assigned_vectors))