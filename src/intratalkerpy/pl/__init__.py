"""
IntraTalkerPy Plotting Module (pl)

This module provides comprehensive plotting functions for receptor analysis,
including barplots, heatmaps, pseudotime analysis, and vector field visualization.

The module contains the following main plotting functions:

Score Visualization:
    - plot_score_barplots: Create barplots of receptor scores across cell types
    - plot_score_heatmap: Generate clustered heatmaps of receptor scores

Pseudotime Analysis:
    - plot_differential_pseudotime: Plot pseudotime differences on embeddings
    - plot_pseudotime_distributions: Compare pseudotime distributions

Vector Field Analysis:
    - plot_metadata_given_ax: Plot metadata on 2D embeddings
    - vector_field_wrapper: Create streamline vector field plots
    - plot_raw_vector_field: Plot raw perturbation vector fields
    - plot_smoothed_vector_field: Plot smoothed perturbation vector fields
"""

# Import public plotting functions
from ._barplot_scores import plot_score_barplots
from ._heatmap_scores import plot_score_heatmap
from ._pseudo_diff import (
    plot_differential_pseudotime,
    plot_pseudotime_distributions
)
from ._vectorfield import (
    plot_metadata_given_ax,
    vector_field_wrapper,
    plot_raw_vector_field,
    plot_smoothed_vector_field
)

# Define what gets imported with "from intratalkerpy.pl import *"
__all__ = [
    # Score visualization functions
    "plot_score_barplots",
    "plot_score_heatmap",
    
    # Pseudotime analysis functions
    "plot_differential_pseudotime",
    "plot_pseudotime_distributions",
    
    # Vector field and metadata plotting functions
    "plot_metadata_given_ax",
    "vector_field_wrapper",
    "plot_raw_vector_field",
    "plot_smoothed_vector_field",
]

# Module metadata
__version__ = "0.1.0"
__author__ = "Vanessa Kloeker"