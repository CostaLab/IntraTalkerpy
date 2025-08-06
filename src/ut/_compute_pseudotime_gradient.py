import numpy as np
import warnings
from sklearn.neighbors import NearestNeighbors
from typing import Optional

def compute_pseudotime_gradient(emb, pseudotime, n_neigh=10):
    """
    Compute pseudotime gradient vectors using nearest neighbor analysis.
    
    This function calculates gradient vectors that represent the direction of
    pseudotime increase in the embedding space. For each point, it uses
    least squares fitting on the nearest neighbors to estimate the local
    gradient direction.
    
    Parameters
    ----------
    emb : array_like
        Cell embedding coordinates with shape (n_cells, n_dims).
        Typically 2D embeddings with shape (n_cells, 2).
    pseudotime : array_like
        Pseudotime values for each cell with shape (n_cells,).
    n_neigh : int, default=10
        Number of nearest neighbors to use for gradient estimation.
        
    Returns
    -------
    np.ndarray
        Normalized gradient vectors with shape (n_cells, n_dims).
        Each vector points in the direction of steepest pseudotime increase.
    """
    try:
        # Input validation and conversion
        emb = np.asarray(emb)
        pseudotime = np.asarray(pseudotime)
        
        # Validate inputs
        if not _validate_gradient_inputs(emb, pseudotime, n_neigh):
            warnings.warn("Invalid inputs, returning zero gradients")
            return np.zeros_like(emb)
        
        # Adjust n_neigh if necessary
        n_cells = len(emb)
        n_neigh = min(n_neigh, n_cells)
        
        if n_neigh < 2:
            warnings.warn("Insufficient neighbors for gradient computation")
            return np.zeros_like(emb)
        
        # Build nearest neighbor model
        nn_gradient = NearestNeighbors(n_neighbors=n_neigh)
        nn_gradient.fit(emb)
        distances, indices = nn_gradient.kneighbors(emb)
        
        # Compute gradients for each point
        gradient_vectors = _compute_local_gradients(emb, pseudotime, indices)
        
        # Normalize gradient vectors
        gradient_vectors = _normalize_gradients(gradient_vectors)
        
        return gradient_vectors
        
    except Exception as e:
        warnings.warn(f"Error in pseudotime gradient computation: {e}")
        return np.zeros_like(emb) if hasattr(emb, 'shape') else np.zeros((len(emb), 2))


def _validate_gradient_inputs(emb, pseudotime, n_neigh):
    """
    Validate inputs for gradient computation.
    
    Parameters
    ----------
    emb : np.ndarray
        Embedding coordinates.
    pseudotime : np.ndarray
        Pseudotime values.
    n_neigh : int
        Number of neighbors.
        
    Returns
    -------
    bool
        True if inputs are valid, False otherwise.
    """
    try:
        # Check for empty arrays
        if len(emb) == 0 or len(pseudotime) == 0:
            return False
        
        # Check length consistency
        if len(emb) != len(pseudotime):
            return False
        
        # Check array dimensions
        if emb.ndim != 2:
            return False
        
        if pseudotime.ndim != 1:
            return False
        
        # Check for valid n_neigh
        if n_neigh <= 0:
            return False
        
        # Check for finite values
        if not (np.isfinite(emb).all() and np.isfinite(pseudotime).all()):
            warnings.warn("Non-finite values detected in inputs")
            # Don't return False here, handle in computation
        
        return True
        
    except Exception:
        return False


def _compute_local_gradients(emb, pseudotime, indices):
    """
    Compute local gradient vectors using least squares fitting.
    
    Parameters
    ----------
    emb : np.ndarray
        Embedding coordinates.
    pseudotime : np.ndarray
        Pseudotime values.
    indices : np.ndarray
        Nearest neighbor indices for each point.
        
    Returns
    -------
    np.ndarray
        Gradient vectors for each point.
    """
    try:
        n_cells, n_dims = emb.shape
        gradient_vectors = np.zeros_like(emb)
        
        for i, neighbors in enumerate(indices):
            try:
                # Get neighbor coordinates and pseudotime differences
                neighbor_coords = emb[neighbors]
                neighbor_pseudotime = pseudotime[neighbors]
                
                # Calculate differences from current point
                dT = neighbor_pseudotime - pseudotime[i]
                dX = neighbor_coords - emb[i]
                
                # Handle edge cases
                if np.all(np.abs(dT) < 1e-12):
                    # No pseudotime variation among neighbors
                    gradient_vectors[i] = np.zeros(n_dims)
                    continue
                
                if np.all(np.linalg.norm(dX, axis=1) < 1e-12):
                    # All neighbors at same location
                    gradient_vectors[i] = np.zeros(n_dims)
                    continue
                
                # Use least squares to fit gradient
                try:
                    pseudo_grad, residuals, rank, s = np.linalg.lstsq(dX, dT, rcond=None)
                    
                    # Check for numerical issues
                    if np.isfinite(pseudo_grad).all():
                        gradient_vectors[i] = pseudo_grad
                    else:
                        # Fallback: use simple average direction
                        gradient_vectors[i] = _compute_simple_gradient(dX, dT)
                        
                except np.linalg.LinAlgError:
                    # Matrix is singular, use fallback method
                    gradient_vectors[i] = _compute_simple_gradient(dX, dT)
                    
            except Exception as e:
                warnings.warn(f"Error computing gradient for point {i}: {e}")
                gradient_vectors[i] = np.zeros(n_dims)
        
        return gradient_vectors
        
    except Exception as e:
        warnings.warn(f"Error in local gradient computation: {e}")
        return np.zeros_like(emb)


def _compute_simple_gradient(dX, dT):
    """
    Compute simple gradient estimate when least squares fails.
    
    Parameters
    ----------
    dX : np.ndarray
        Coordinate differences.
    dT : np.ndarray
        Pseudotime differences.
        
    Returns
    -------
    np.ndarray
        Simple gradient estimate.
    """
    try:
        # Weight by pseudotime difference and distance
        weights = np.abs(dT) / (np.linalg.norm(dX, axis=1) + 1e-12)
        weights = weights / (np.sum(weights) + 1e-12)
        
        # Weighted average of directions
        gradient = np.sum(dX * weights[:, np.newaxis], axis=0)
        
        return gradient
        
    except Exception:
        return np.zeros(dX.shape[1])


def _normalize_gradients(gradient_vectors):
    """
    Normalize gradient vectors to unit length.
    
    Parameters
    ----------
    gradient_vectors : np.ndarray
        Raw gradient vectors.
        
    Returns
    -------
    np.ndarray
        Normalized gradient vectors.
    """
    try:
        # Calculate norms
        norms = np.linalg.norm(gradient_vectors, axis=1, keepdims=True)
        
        # Handle zero vectors
        zero_mask = norms.flatten() < 1e-12
        norms[zero_mask] = 1.0  # Prevent division by zero
        
        # Normalize
        normalized_vectors = gradient_vectors / norms
        
        # Set zero gradient vectors to zero (not random unit vectors)
        normalized_vectors[zero_mask] = 0.0
        
        # Handle any remaining NaN or infinite values
        finite_mask = np.isfinite(normalized_vectors).all(axis=1)
        normalized_vectors[~finite_mask] = 0.0
        
        return normalized_vectors
        
    except Exception as e:
        warnings.warn(f"Error normalizing gradients: {e}")
        return np.zeros_like(gradient_vectors)


def estimate_optimal_neighbors(emb, pseudotime, max_neighbors=50):
    """
    Estimate optimal number of neighbors for gradient computation.
    
    Parameters
    ----------
    emb : np.ndarray
        Embedding coordinates.
    pseudotime : np.ndarray
        Pseudotime values.
    max_neighbors : int, default=50
        Maximum number of neighbors to consider.
        
    Returns
    -------
    int
        Estimated optimal number of neighbors.
    """
    try:
        n_cells = len(emb)
        
        # Start with reasonable bounds
        min_neighbors = max(3, int(np.sqrt(n_cells) / 2))
        max_neighbors = min(max_neighbors, n_cells // 2, 100)
        
        # Use heuristic based on data size and dimensionality
        if n_cells < 100:
            optimal = min(10, max_neighbors)
        elif n_cells < 1000:
            optimal = min(15, max_neighbors)
        else:
            optimal = min(20, max_neighbors)
        
        # Ensure within bounds
        optimal = max(min_neighbors, min(optimal, max_neighbors))
        
        return optimal
        
    except Exception:
        return 10  # Default fallback