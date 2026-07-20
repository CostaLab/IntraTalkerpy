
<img src="docs/_static/IntraTalker_Logo.svg" align="right" width="200" alt="Logo">

IntraTalkerPy
=============

IntraTalkerPy combines two single-cell analysis workflows in one package:

- `intratalkerpy.tf` for transcription factor activity analysis and LR-to-TF workflows.
- `intratalkerpy.perturbation` for gene expression perturbation methods, utilities, and plotting.

## Abstract

Single-cell sequencing has advanced the study of cell-cell communication, yet most methods focus on intercellular ligand- receptor interactions while neglecting downstream intracellular signalling cascades and the possibility that downstream target genes themselves encode ligands, thereby propagating communication across multiple cells. We present In- traTalker+CrossTalkeR that combines intracellular (IntraTalker) and intercellular (CrossTalkeR) signalling from multimodal single-cell data. IntraTalker infers cell-type-specific transcription factor activities and constructs receptomes that link receptors to downstream target genes, which are then integrated with ligand-receptor predictions in CrossTalkeR. To prioritize signalling receptors, the framework performs in silico receptor perturbation.

## Install

Install the package from this repository root:

```bash
pip install .
```

Use the two analysis areas as separate modules:

```python
from intratalkerpy import tf
from intratalkerpy import perturbation

result = tf.IntraTalker_analysis(...)
perturbation.mt.simulation_of_perturbation(...)
```

