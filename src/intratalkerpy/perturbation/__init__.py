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

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover - Python < 3.8 fallback
    from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version("intratalkerpy")
except PackageNotFoundError:  # pragma: no cover - local source tree import
    __version__ = "0.3.0"

# Import submodules
from . import pl  # Plotting module
from . import ut  # Utilities module
from . import mt  # Methods module

__all__ = [
    # Submodules
    "pl", 
    "ut", 
    "mt",
    
    # Package metadata
    "__version__"
]

# Package metadata
__author__ = "Vanessa Kloeker"
__email__ = "vanessa.kloeker@gmail.com"
__description__ = "A Python package for receptor analysis and visualization in single-cell data"
