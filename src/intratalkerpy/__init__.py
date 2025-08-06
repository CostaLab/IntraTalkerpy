"""
IntraTalkerPy: A Python package for receptor analysis and visualization.

This package provides comprehensive tools for analyzing receptor-related
perturbations in single-cell data, including plotting utilities, computational
tools, and analysis methods for studying receptor-ligand interactions.

Modules:
    pl: Plotting functions for visualization (barplots, heatmaps, pseudotime, vector fields)
    ut: Utility functions for data processing (grid calculations, rankings, pseudotime)
    mt: Methods for analysis (differential analysis, simulations, projections)
"""

from importlib.metadata import version

__version__ = version("intratalkerpy")

# Import submodules
from . import pl  # Plotting module
from . import ut  # Utilities module
from . import mt  # Methods module

# Make key functions easily accessible at package level
from .mt import differential_pseudotime_analysis

__all__ = [
    # Submodules
    "pl", 
    "ut", 
    "mt",
    
    # Key functions for easy access
    "differential_pseudotime_analysis",
    
    # Package metadata
    "__version__"
]

# Package metadata
__author__ = "Vanessa Kloeker"
__email__ = "vanessa.kloeker@gmail.com"  # Update with actual email
__description__ = "A Python package for receptor analysis and visualization in single-cell data"