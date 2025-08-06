import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm as normal
from typing import Tuple
import warnings


def calculate_grid_arrows(
    embedding: np.ndarray,
    delta_embedding: np.ndarray,
    offset_frac: float = 0.1,
    n_grid_cols: int = 25,
    n_grid_rows: int = 25,
    n_neighbors: int = 30,
    n_cpu: int = 1,
    gaussian_scale: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate smoothed vector field on a regular grid from perturbation data.
    
    This function creates a regular grid over the embedding space and calculates
    smoothed vector field arrows by averaging nearby perturbation vectors using
    a Gaussian kernel weighting scheme.
    
    Parameters
    ----------
    embedding : np.ndarray
        2D embedding coordinates of shape (n_cells, 2).
    delta_embedding : np.ndarray
        Perturbation vectors of shape (n_cells, 2).
    offset_frac : float, default=0.1
        Fraction of the embedding range to use as offset from boundaries.
        Must be between 0 and 0.5.
    n_grid_cols : int, default=25
        Number of grid columns (x-direction).
    n_grid_rows : int, default=25
        Number of grid rows (y-direction).
    n_neighbors : int, default=30
        Number of nearest neighbors to consider for each grid point.
    n_cpu : int, default=1
        Number of CPU cores to use for nearest neighbor search.
    gaussian_scale : float, default=0.5
        Scale parameter for the Gaussian kernel weighting.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        - grid_xy: Grid point coordinates of shape (n_grid_points, 2)
        - uv: Smoothed vector field of shape (n_grid_points, 2)  
        - mask: Boolean mask indicating valid grid points
        
    Raises
    ------
    ValueError
        If input arrays have incompatible shapes or invalid parameters.
        
    Notes
    -----
    The function uses a Gaussian kernel to weight the contribution of nearby
    cells to each grid point. The mask identifies grid points that are too
    far from any actual data points and should be excluded from visualization.
    
    Examples
    --------
    >>> grid_xy, uv, mask = calculate_grid_arrows(embedding, delta_vectors)
    >>> # Use only valid grid points
    >>> valid_grid = grid_xy[mask]
    >>> valid_vectors = uv[mask]
    """
    # Input validation
    if embedding.shape[0] != delta_embedding.shape[0]:
        raise ValueError("embedding and delta_embedding must have same number of points")
    
    if embedding.shape[1] != 2 or delta_embedding.shape[1] != 2:
        raise ValueError("Both arrays must be 2D (n_cells x 2)")
    
    if not (0 < offset_frac < 0.5):
        raise ValueError("offset_frac must be between 0 and 0.5")
    
    if n_grid_cols <= 0 or n_grid_rows <= 0:
        raise ValueError("Grid dimensions must be positive")
    
    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be positive")
    
    if gaussian_scale <= 0:
        raise ValueError("gaussian_scale must be positive")
    
    # Handle empty data
    if len(embedding) == 0:
        warnings.warn("Empty embedding array provided")
        return np.array([]), np.array([]), np.array([])
    
    # Handle NaN values
    valid_mask = ~(np.isnan(embedding).any(axis=1) | np.isnan(delta_embedding).any(axis=1))
    if not valid_mask.any():
        warnings.warn("All values contain NaN")
        return np.array([]), np.array([]), np.array([])
    
    if not valid_mask.all():
        n_invalid = (~valid_mask).sum()
        warnings.warn(f"Removing {n_invalid} cells with NaN values")
        embedding = embedding[valid_mask]
        delta_embedding = delta_embedding[valid_mask]
    
    try:
        # Calculate embedding bounds
        min_x, max_x = np.min(embedding[:, 0]), np.max(embedding[:, 0])
        min_y, max_y = np.min(embedding[:, 1]), np.max(embedding[:, 1])
        
        # Check for degenerate cases
        if min_x == max_x or min_y == max_y:
            warnings.warn("Embedding has zero range in one dimension")
            # Add small epsilon to prevent division by zero
            if min_x == max_x:
                max_x += 1e-6
            if min_y == max_y:
                max_y += 1e-6
        
        # Calculate offsets
        offset_x = (max_x - min_x) * offset_frac
        offset_y = (max_y - min_y) * offset_frac
        
        # Calculate grid spacing
        x_dist_between_points = (max_x - min_x) / n_grid_cols
        y_dist_between_points = (max_y - min_y) / n_grid_rows
        minimal_distance = np.mean([x_dist_between_points, y_dist_between_points])
        
        # Create regular grid
        grid_x, grid_y = np.meshgrid(
            np.linspace(min_x + offset_x, max_x - offset_x, n_grid_cols),
            np.linspace(min_y + offset_y, max_y - offset_y, n_grid_rows)
        )
        grid_xy = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        
        # Find nearest neighbors for each grid point
        nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(embedding)), n_jobs=n_cpu)
        nn.fit(embedding)
        dists, neighs = nn.kneighbors(grid_xy)
        
        # Calculate standard deviation for Gaussian kernel
        # Use a more robust estimate based on grid spacing
        std = np.mean([x_dist_between_points, y_dist_between_points]) * gaussian_scale
        
        # Apply Gaussian kernel weighting
        gaussian_weights = normal.pdf(dists, loc=0, scale=std)
        total_p_mass = gaussian_weights.sum(axis=1)
        
        # Prevent division by zero
        total_p_mass = np.maximum(total_p_mass, 1e-10)
        
        # Calculate weighted average of perturbation vectors
        weighted_deltas = delta_embedding[neighs] * gaussian_weights[:, :, np.newaxis]
        uv = weighted_deltas.sum(axis=1) / total_p_mass[:, np.newaxis]
        
        # Create mask for valid grid points
        # Points are considered valid if they have nearby data points
        mask = dists.min(axis=1) < minimal_distance
        
        return grid_xy, uv, mask
        
    except Exception as e:
        raise RuntimeError(f"Error calculating grid arrows: {e}")


def _calculate_grid_arrows(embedding, delta_embedding, offset_frac, n_grid_cols, n_grid_rows, n_neighbors, n_cpu):
    """
    Backward-compatible wrapper for calculate_grid_arrows.
    
    This function maintains compatibility with existing code while providing
    access to the improved implementation.
    
    Parameters
    ----------
    embedding : np.ndarray
        2D embedding coordinates.
    delta_embedding : np.ndarray
        Perturbation vectors.
    offset_frac : float
        Fraction of embedding range to use as offset.
    n_grid_cols : int
        Number of grid columns.
    n_grid_rows : int
        Number of grid rows.
    n_neighbors : int
        Number of nearest neighbors.
    n_cpu : int
        Number of CPU cores.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Grid coordinates, vectors, and mask.
    """
    warnings.warn(
        "_calculate_grid_arrows is deprecated. Use calculate_grid_arrows instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    return calculate_grid_arrows(
        embedding=embedding,
        delta_embedding=delta_embedding,
        offset_frac=offset_frac,
        n_grid_cols=n_grid_cols,
        n_grid_rows=n_grid_rows,
        n_neighbors=n_neighbors,
        n_cpu=n_cpu
    )


def validate_grid_parameters(
    n_grid_cols: int,
    n_grid_rows: int,
    offset_frac: float,
    n_neighbors: int
) -> None:
    """
    Validate parameters for grid calculation.
    
    Parameters
    ----------
    n_grid_cols : int
        Number of grid columns.
    n_grid_rows : int
        Number of grid rows.
    offset_frac : float
        Offset fraction.
    n_neighbors : int
        Number of neighbors.
        
    Raises
    ------
    ValueError
        If any parameter is invalid.
    """
    if n_grid_cols <= 0 or n_grid_rows <= 0:
        raise ValueError("Grid dimensions must be positive integers")
    
    if not (0 < offset_frac < 0.5):
        raise ValueError("offset_frac must be between 0 and 0.5")
    
    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be positive")
    
    # Warn about potentially problematic values
    if n_grid_cols * n_grid_rows > 10000:
        warnings.warn("Large grid size may be slow to compute")
    
    if offset_frac < 0.05:
        warnings.warn("Very small offset_frac may cause boundary effects")


def estimate_optimal_grid_size(embedding: np.ndarray) -> Tuple[int, int]:
    """
    Estimate optimal grid size based on embedding density.
    
    Parameters
    ----------
    embedding : np.ndarray
        2D embedding coordinates.
        
    Returns
    -------
    Tuple[int, int]
        Recommended (n_grid_cols, n_grid_rows).
    """
    n_points = len(embedding)
    
    # Calculate embedding aspect ratio
    x_range = np.ptp(embedding[:, 0])  # peak-to-peak (max - min)
    y_range = np.ptp(embedding[:, 1])
    
    if y_range == 0:
        aspect_ratio = 1
    else:
        aspect_ratio = x_range / y_range
    
    # Base grid size on number of points
    if n_points < 1000:
        base_size = 15
    elif n_points < 5000:
        base_size = 20
    elif n_points < 20000:
        base_size = 25
    else:
        base_size = 30
    
    # Adjust for aspect ratio
    if aspect_ratio > 1:
        n_grid_cols = int(base_size * np.sqrt(aspect_ratio))
        n_grid_rows = int(base_size / np.sqrt(aspect_ratio))
    else:
        n_grid_cols = int(base_size / np.sqrt(1/aspect_ratio))
        n_grid_rows = int(base_size * np.sqrt(1/aspect_ratio))
    
    # Ensure minimum size
    n_grid_cols = max(10, n_grid_cols)
    n_grid_rows = max(10, n_grid_rows)
    
    return n_grid_cols, n_grid_rows
