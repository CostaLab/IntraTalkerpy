.. lr2tf-py documentation master file, created by
   sphinx-quickstart on Wed Apr 16 13:23:35 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

IntraTalkerPy documentation
===========================

IntraTalkerPy combines transcription factor analysis and gene expression perturbation analysis for single-cell data.

The transcription factor module estimates transcription factor activities using decoupleR. Using the DoRothEA regulon version from decoupleR and post-translational interactions from the Omnipath database[2,3,4], connections are made between transcription factors and ligands and receptors.

The results can be combined with ligand-receptor interactions and then analyzed using CrossTalkeR (https://github.com/CostaLab/CrossTalkeR/) [5].

The original R version can be found here: (https://github.com/CostaLab/LR2TF/)

The perturbation module provides methods, utilities, and plotting functions for receptor-related gene expression perturbation workflows.

.. toctree::
   :maxdepth: 3
   :caption: Contents:
   
   install
   support
   api
