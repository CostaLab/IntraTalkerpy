"""
IntraTalkerPy Methods Module (mt)

This module provides analysis methods for receptor-ligand interaction studies
in single-cell data, including differential pseudotime analysis, perturbation
simulation, and embedding projection methods.

Functions:
    Analysis Methods:
        - differential_pseudotime_analysis: Analyze receptor effects on pseudotime trajectories
    
    Simulation Methods:
        - simulation_of_perturbation: Simulate gene expression perturbations through propagation
        - project_perturbation_in_embedding: Project perturbations into embedding space
        - permute_rows_nsign: Helper function for random matrix permutation
"""

# Import analysis methods
from ._differential_pseudotime_analysis import differential_pseudotime_analysis

# Import simulation methods
from ._simulation_of_perturbation import simulation_of_perturbation

# Import projection methods
from ._project_perturbation_in_embedding import (
    project_perturbation_in_embedding,
    permute_rows_nsign
)

# Define what gets imported with "from intratalkerpy.perturbation.mt import *"
__all__ = [
    # Analysis methods
    "differential_pseudotime_analysis",
    
    # Simulation methods
    "simulation_of_perturbation",
    
    # Projection methods
    "project_perturbation_in_embedding",
    "permute_rows_nsign",
]

# Module metadata
__version__ = "0.1.0"
__author__ = "Vanessa Kloeker"
