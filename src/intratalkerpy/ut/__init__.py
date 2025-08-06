"""
IntraTalkerPy Utilities Module (ut)

This module provides utility functions for data processing and analysis,
including grid-based vector field calculations, statistical ranking functions,
pseudotime gradient computation, and differential pseudotime analysis.

Functions:
    Grid Calculations:
        - calculate_grid_arrows: Calculate smoothed vector field on regular grid
        - validate_grid_parameters: Validate parameters for grid calculation
        - estimate_optimal_grid_size: Estimate optimal grid size for data

    Statistical Rankings:
        - calculate_pseudotime_comparison: Comprehensive pseudotime comparison (returns dict)
        - calculate_effect_size_from_differences: Effect size analysis (returns dict)

    Pseudotime Analysis:
        - compute_pseudotime_gradient: Compute gradient vectors from pseudotime
        - compute_differential_pseudotime: Compute differential pseudotime changes
        - estimate_optimal_neighbors: Estimate optimal neighbor count for gradients
"""

# Grid-based vector field calculations
from ._calculate_grid_arrows import (
    calculate_grid_arrows,
    validate_grid_parameters, 
    estimate_optimal_grid_size
)

# Statistical analysis and rankings
from ._calculate_rankings import (
    calculate_pseudotime_comparison,
    calculate_effect_size_from_differences
)

# Pseudotime gradient computation
from ._compute_pseudotime_gradient import (
    compute_pseudotime_gradient,
    estimate_optimal_neighbors
)

# Differential pseudotime analysis
from ._compute_differential_pseudotime import (
    compute_differential_pseudotime
)

# Define what gets imported with "from intratalkerpy.ut import *"
__all__ = [
    # Grid calculation functions
    "calculate_grid_arrows",
    "validate_grid_parameters",
    "estimate_optimal_grid_size",
    
    # Statistical ranking functions
    "calculate_pseudotime_comparison",
    "calculate_effect_size_from_differences",
    
    # Pseudotime analysis functions
    "compute_pseudotime_gradient",
    "compute_differential_pseudotime",
    "estimate_optimal_neighbors",
]

# Module metadata
__version__ = "0.1.0"
__author__ = "Vanessa Kloeker"