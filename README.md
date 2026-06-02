
IntraTalkerPy
=============

IntraTalkerPy combines two single-cell analysis workflows in one package:

- `intratalkerpy.tf` for transcription factor activity analysis and LR-to-TF workflows.
- `intratalkerpy.perturbation` for gene expression perturbation methods, utilities, and plotting.

Install the merged package from this repository root:

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

The existing Sphinx documentation is kept in `docs/` and now includes separate API pages for transcription factor and perturbation functionality.
