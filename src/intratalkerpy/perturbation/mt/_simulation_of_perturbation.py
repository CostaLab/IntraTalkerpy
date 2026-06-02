import numpy as np
import pandas as pd
import scipy.sparse as sparse
import warnings
from typing import Union


def simulation_of_perturbation(gem, simulation_input, coef_matrix, n_propagation):
    """
    Simulate gene expression perturbations through iterative propagation.
    
    This function simulates the effect of gene expression perturbations by
    iteratively propagating changes through a coefficient matrix representing
    gene-gene interactions. The simulation ensures non-negative gene expression
    values throughout the process.
    
    Parameters
    ----------
    gem : array-like
        Original gene expression matrix with shape (n_cells, n_genes).
    simulation_input : array-like
        Target gene expression matrix after perturbation with shape (n_cells, n_genes).
        Must have the same shape as `gem`.
    coef_matrix : array-like
        Coefficient matrix representing gene-gene interactions with shape (n_genes, n_genes).
        Used for propagating perturbation effects across genes.
    n_propagation : int
        Number of propagation iterations to perform. Must be non-negative.
        
    Returns
    -------
    array-like
        Simulated gene expression matrix after perturbation propagation.
        Same type and shape as input `gem`.
        
    Raises
    ------
    ValueError
        If input matrices have incompatible shapes.
        If n_propagation is negative.   
    """
    _validate_inputs(gem, simulation_input, coef_matrix, n_propagation)
    
    delta_input = simulation_input - gem
    delta_simulated = delta_input.copy()
    
    for i in range(n_propagation):
        delta_simulated = delta_simulated.dot(coef_matrix)
        
        delta_simulated[delta_input != 0] = delta_input
        
        gem_tmp = gem + delta_simulated
        gem_tmp[gem_tmp < 0] = 0
        delta_simulated = gem_tmp - gem
    
    gem_simulated = gem + delta_simulated
    
    return gem_simulated


def _validate_inputs(gem, simulation_input, coef_matrix, n_propagation):
    """Validate all input parameters."""
    
    if not isinstance(n_propagation, int) or n_propagation < 0:
        raise ValueError(f"n_propagation must be a non-negative integer, got {n_propagation}")
    
    if gem.shape != simulation_input.shape:
        raise ValueError(
            f"gem and simulation_input must have the same shape. "
            f"Got {gem.shape} and {simulation_input.shape}"
        )
    
    n_genes = gem.shape[1]
    if coef_matrix.shape != (n_genes, n_genes):
        raise ValueError(
            f"coef_matrix must be square with dimensions ({n_genes}, {n_genes}). "
            f"Got {coef_matrix.shape}"
        )