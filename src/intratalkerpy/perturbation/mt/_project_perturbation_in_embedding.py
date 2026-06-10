import numpy as np
import pandas as pd
import scipy.sparse as sparse
 
from anndata import AnnData
from sklearn.neighbors import NearestNeighbors
from typing import Optional

def permute_rows_nsign(A: np.ndarray) -> None:
    """Permute entries in place and randomly flip the sign for each row
    of a matrix independently.
 
    Adapted from CellOracle.
 
    Parameters
    ----------
    A:
        2-D array whose rows are shuffled in place.
    """
    plmi = np.array([+1, -1])
    for i in range(A.shape[0]):
        np.random.shuffle(A[i, :])
        A[i, :] = A[i, :] * np.random.choice(plmi, size=A.shape[1])
 
 
def project_perturbation_in_embedding(
    anndata: AnnData,
    perturbed_matrix: pd.DataFrame,
    reduction_name: str,
    original_matrix: Optional[pd.DataFrame] = None,
    sigma_corr: float = 0.05,
    n_cpu: int = 1,
) -> np.ndarray:
    """Project a perturbation simulation result onto a low-dimensional embedding.
 
    Based on CellOracle / Velocyto code.
 
    Parameters
    ----------
    anndata:
        AnnData object containing the embedding in ``obsm``.
    perturbed_matrix:
        Gene expression DataFrame of the perturbed state (cells x genes).
    reduction_name:
        Key in ``anndata.obsm`` to use as the embedding (e.g. ``"X_umap"``).
    original_matrix:
        Gene expression DataFrame of the unperturbed state (cells x genes).
        Defaults to ``anndata.to_df()`` when ``None``.
    sigma_corr:
        Kernel width for the transition probability calculation. Default ``0.05``.
    n_cpu:
        Number of threads / CPUs to use. Default ``1``.
 
    Returns
    -------
    np.ndarray
        Delta embedding vectors for each cell (cells x embedding dimensions).
 
    Raises
    ------
    ValueError
        If ``reduction_name`` is not found in ``anndata.obsm``.
    """
    try:
        from deltacorrpy import colDeltaCorpartial
    except ImportError as exc:
        raise ImportError(
            "project_perturbation_in_embedding requires deltacorrpy. "
            "Install it with: pip install git+https://github.com/vckraemer/deltacorrpy.git"
        ) from exc

    cell_names = list(anndata.obs_names)
    embeddings = {
        k: pd.DataFrame(anndata.obsm[k].copy(), index=cell_names)
        for k in anndata.obsm.keys()
    }
 
    if reduction_name not in embeddings:
        raise ValueError(f'Embedding "{reduction_name}" not found in anndata.obsm.')
 
    if original_matrix is None:
        original_matrix = anndata.to_df()
 
    delta_matrix = (
        perturbed_matrix.to_numpy().astype("double")
        - original_matrix.to_numpy().astype("double")
    )
    delta_matrix_random = delta_matrix.copy()
    permute_rows_nsign(delta_matrix_random)
 
    embedding = embeddings[reduction_name].to_numpy()
    n_neighbors = int(perturbed_matrix.shape[0] / 5)  # default from CellOracle
 
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=n_cpu)
    nn.fit(embedding)
    embedding_knn = nn.kneighbors_graph(mode="connectivity")
 
    # Pick a random subset of neighbours and prune the rest.
    neigh_ixs = embedding_knn.indices.reshape((-1, n_neighbors + 1))
    p = np.linspace(0.5, 0.1, neigh_ixs.shape[1])
    p = p / p.sum()
    sampling_ixs = np.stack(
        [
            np.random.choice(
                neigh_ixs.shape[1],
                size=(int(0.3 * (n_neighbors + 1)),),
                replace=False,
                p=p,
            )
            for _ in range(neigh_ixs.shape[0])
        ],
        axis=0,
    )
    neigh_ixs = neigh_ixs[np.arange(neigh_ixs.shape[0])[:, None], sampling_ixs]
 
    nonzero = neigh_ixs.shape[0] * neigh_ixs.shape[1]
    embedding_knn = sparse.csr_matrix(
        (
            np.ones(nonzero),
            neigh_ixs.ravel(),
            np.arange(0, nonzero + 1, neigh_ixs.shape[1]),
        ),
        shape=(neigh_ixs.shape[0], neigh_ixs.shape[0]),
    )
 
    corrcoef = colDeltaCorpartial(
        perturbed_matrix.T, delta_matrix.T, neigh_ixs, threads=n_cpu
    )
    corrcoef[np.isnan(corrcoef)] = 1
    np.fill_diagonal(corrcoef, 0)
 
    transition_prob = np.exp(corrcoef / sigma_corr) * embedding_knn.toarray()
    transition_prob /= transition_prob.sum(1)[:, None]
 
    # Compute unit vectors between all pairs of embedded cells.
    unitary_vectors = (
        embedding.T[:, None, :] - embedding.T[:, :, None]
    )  # shape: (n_dims, n_cells, n_cells)
    unitary_vectors /= np.linalg.norm(unitary_vectors, ord=2, axis=0)
    np.fill_diagonal(unitary_vectors[0, ...], 0)  # remove NaNs on diagonal
    np.fill_diagonal(unitary_vectors[1, ...], 0)
 
    knn_dense = embedding_knn.toarray()
    delta_embedding = (transition_prob * unitary_vectors).sum(2)
    delta_embedding -= (knn_dense * unitary_vectors).sum(2) / embedding_knn.sum(1).T
    delta_embedding = delta_embedding.T
 
    print("Finished embedding projection.", flush=True)
    return delta_embedding
