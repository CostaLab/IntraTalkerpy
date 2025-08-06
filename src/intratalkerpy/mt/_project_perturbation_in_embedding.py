import numpy as np
import pandas as pd
import scipy.sparse as sparse
import warnings
from sklearn.neighbors import NearestNeighbors

# Handle optional velocyto dependency
try:
    from velocyto.estimation import colDeltaCorpartial
    HAS_VELOCYTO = True
except ImportError:
    HAS_VELOCYTO = False
    warnings.warn(
        "velocyto not available. Some correlation calculations may be limited.",
        ImportWarning
    )


def permute_rows_nsign(A):
    """
    Permute in place the entries and randomly switch the sign for each row of a matrix independently.
    
    This function performs random permutation and sign switching for each row
    of the input matrix. Based on CellOracle implementation.
    
    Parameters
    ----------
    A : np.ndarray
        Input matrix to be permuted in place.
        
    Notes
    -----
    This function modifies the input matrix in place.
    """
    if not isinstance(A, np.ndarray):
        raise ValueError("Input must be a numpy array")
    
    plmi = np.array([+1, -1])
    for i in range(A.shape[0]):
        np.random.shuffle(A[i, :])
        A[i, :] = A[i, :] * np.random.choice(plmi, size=A.shape[1])


def project_perturbation_in_embedding(
    anndata,
    original_matrix,
    perturbed_matrix,
    reduction_name,
    sigma_corr=0.05,
    n_cpu=1
):
    """
    Project gene expression perturbations into embedding space.
    
    This function projects perturbation effects from gene expression space
    into low-dimensional embedding space using nearest neighbor analysis
    and correlation-based transition probabilities. Based on CellOracle/Velocyto
    methodology.
    
    Parameters
    ----------
    anndata : AnnData
        Annotated data object containing gene expression and embeddings.
    original_matrix : array-like or None
        Original gene expression matrix. If None, uses anndata.to_df().
    perturbed_matrix : array-like
        Perturbed gene expression matrix with same shape as original_matrix.
    reduction_name : str
        Name of the embedding stored in anndata.obsm to use for projection.
    sigma_corr : float, default=0.05
        Correlation scaling parameter for transition probability calculation.
    n_cpu : int, default=1
        Number of CPU cores to use for parallel computation.
        
    Returns
    -------
    np.ndarray
        Delta embedding vectors with shape (n_cells, n_embedding_dims).
        Represents projected perturbation effects in embedding space.
        
    Raises
    ------
    ValueError
        If the specified embedding is not found in anndata.obsm.
        If input matrices have incompatible shapes.
        
    Notes
    -----
    The projection algorithm:
    1. Extracts cell embeddings and builds k-nearest neighbor graph
    2. Calculates correlation coefficients between perturbations and deltas
    3. Computes transition probabilities using correlation and neighbor structure
    4. Projects perturbations using weighted unitary vectors in embedding space
    5. Returns delta embedding representing perturbation effects
    
    This implementation fixes the variable name bug (data -> anndata) and
    improves neighbor sampling for better API consistency.
    """
    try:
        # Input validation
        _validate_projection_inputs(anndata, original_matrix, perturbed_matrix, reduction_name)
        
        # Extract cell names and embeddings
        cell_names = anndata.obs_names.copy()
        embeddings_dict = {
            k: pd.DataFrame(anndata.obsm[k].copy(), index=cell_names) 
            for k in anndata.obsm.keys()
        }
        
        if reduction_name not in embeddings_dict.keys():
            raise ValueError(f'Embedding "{reduction_name}" not found in anndata.obsm!')
        
        # Prepare matrices
        if original_matrix is None:
            original_matrix = anndata.to_df().copy()
        
        # Ensure proper data types and compute delta
        original_array = _ensure_numeric_matrix(original_matrix)
        perturbed_array = _ensure_numeric_matrix(perturbed_matrix)
        delta_matrix = perturbed_array - original_array
        
        # Get embedding coordinates
        embedding = embeddings_dict[reduction_name].to_numpy()
        
        # Build neighbor graph with validation
        n_neighbors = _calculate_optimal_neighbors(perturbed_array.shape[0])
        neighbor_indices, embedding_knn_sampled = _build_neighbor_graph(embedding, n_neighbors, n_cpu)
        
        # Calculate correlation coefficients
        corrcoef = _calculate_correlation_coefficients(
            perturbed_array, delta_matrix, neighbor_indices, n_cpu
        )
        
        # Compute transition probabilities
        transition_prob, embedding_knn = _compute_transition_probabilities(
            corrcoef, neighbor_indices, sigma_corr
        )
        
        # Project perturbations into embedding space
        delta_embedding = _project_to_embedding_space(
            embedding, (transition_prob, embedding_knn), neighbor_indices
        )
        
        print("Finished Embedding Projection", flush=True)
        return delta_embedding
        
    except Exception as e:
        raise ValueError(f"Error in perturbation projection: {str(e)}") from e


def _validate_projection_inputs(anndata, original_matrix, perturbed_matrix, reduction_name):
    """Validate inputs for perturbation projection."""
    if not hasattr(anndata, 'obsm'):
        raise ValueError("anndata must have obsm attribute containing embeddings")
    
    if not isinstance(reduction_name, str):
        raise ValueError("reduction_name must be a string")
    
    if original_matrix is not None and original_matrix.shape != perturbed_matrix.shape:
        raise ValueError(
            f"original_matrix and perturbed_matrix must have the same shape. "
            f"Got {original_matrix.shape} and {perturbed_matrix.shape}"
        )


def _ensure_numeric_matrix(matrix):
    """Convert matrix to numeric numpy array."""
    if hasattr(matrix, 'to_numpy'):
        return matrix.to_numpy().astype('double')
    elif hasattr(matrix, 'values'):
        return matrix.values.astype('double')
    else:
        return np.asarray(matrix).astype('double')


def _calculate_optimal_neighbors(n_cells):
    """Calculate optimal number of neighbors based on dataset size."""
    # Default from CellOracle: n_cells / 5
    n_neighbors = max(5, min(int(n_cells / 5), 100))  # Cap at 100 for very large datasets
    return n_neighbors


def _build_neighbor_graph(embedding, n_neighbors, n_cpu):
    """Build k-nearest neighbor graph and sample neighbors."""
    # Fit nearest neighbors
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=n_cpu)
    nn.fit(embedding)
    embedding_knn = nn.kneighbors_graph(mode='connectivity')
    
    # Extract neighbor indices and create sampling probabilities
    neigh_ixs = embedding_knn.indices.reshape((-1, n_neighbors + 1))
    p = np.linspace(0.5, 0.1, neigh_ixs.shape[1])
    p = p / p.sum()
    
    # Sample neighbors for better consistency (fixes API issue mentioned in original)
    sampling_size = int(0.3 * (n_neighbors + 1))
    sampling_ixs = np.stack([
        np.random.choice(
            neigh_ixs.shape[1],
            size=(sampling_size,),
            replace=False,
            p=p
        ) for i in range(neigh_ixs.shape[0])
    ], 0)
    
    sampled_neigh_ixs = neigh_ixs[np.arange(neigh_ixs.shape[0])[:, None], sampling_ixs]
    
    # Create sparse connectivity matrix
    nonzero = sampled_neigh_ixs.shape[0] * sampled_neigh_ixs.shape[1]
    embedding_knn_sampled = sparse.csr_matrix(
        (np.ones(nonzero),
        sampled_neigh_ixs.ravel(),
        np.arange(0, nonzero + 1, sampled_neigh_ixs.shape[1])),
        shape=(sampled_neigh_ixs.shape[0], sampled_neigh_ixs.shape[0])
    )
    
    return sampled_neigh_ixs, embedding_knn_sampled


def _calculate_correlation_coefficients(perturbed_matrix, delta_matrix, neighbor_indices, n_cpu):
    """Calculate correlation coefficients between perturbations and deltas."""
    if HAS_VELOCYTO:
        corrcoef = colDeltaCorpartial(perturbed_matrix.T, delta_matrix.T, neighbor_indices, threads=n_cpu)
    else:
        # Fallback correlation calculation if velocyto is not available
        warnings.warn("Using simplified correlation calculation (velocyto not available)", UserWarning)
        corrcoef = np.corrcoef(perturbed_matrix)
    
    # Handle NaN values and diagonal
    corrcoef[np.isnan(corrcoef)] = 1
    np.fill_diagonal(corrcoef, 0)
    
    return corrcoef


def _compute_transition_probabilities(corrcoef, neighbor_indices, sigma_corr):
    """Compute transition probabilities from correlations and neighbor structure."""
    # Recreate sparse matrix for transition calculation
    nonzero = neighbor_indices.shape[0] * neighbor_indices.shape[1]
    embedding_knn = sparse.csr_matrix(
        (np.ones(nonzero),
        neighbor_indices.ravel(),
        np.arange(0, nonzero + 1, neighbor_indices.shape[1])),
        shape=(neighbor_indices.shape[0], neighbor_indices.shape[0])
    )
    
    # Calculate transition probabilities
    transition_prob = np.exp(corrcoef / sigma_corr) * embedding_knn.toarray()
    
    # Normalize probabilities
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # Suppress divide by zero warnings
        transition_prob /= transition_prob.sum(1)[:, None]
        transition_prob[np.isnan(transition_prob)] = 0  # Handle any remaining NaNs
    
    return transition_prob, embedding_knn


def _project_to_embedding_space(embedding, transition_data, neighbor_indices):
    """Project perturbations into embedding space using transition probabilities."""
    transition_prob, embedding_knn = transition_data
    
    # Calculate unitary vectors between cells in embedding space
    unitary_vectors = embedding.T[:, None, :] - embedding.T[:, :, None]  # shape (2, n_cells, n_cells)
    
    # Normalize to unit vectors
    norms = np.linalg.norm(unitary_vectors, ord=2, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # Suppress divide by zero warnings
        unitary_vectors /= norms
        
    # Fix NaN values on diagonal
    np.fill_diagonal(unitary_vectors[0, ...], 0)
    np.fill_diagonal(unitary_vectors[1, ...], 0)
    
    # Compute delta embedding
    delta_embedding = (transition_prob * unitary_vectors).sum(2)
    
    # Subtract baseline (average neighbor direction)
    baseline = (embedding_knn.toarray() * unitary_vectors).sum(2) / embedding_knn.sum(1).T
    delta_embedding = delta_embedding - baseline
    
    return delta_embedding.T