import csv
from importlib import resources

import anndata as ad
import pandas as pd


def test_public_modules_import_and_expose_expected_api():
    import intratalkerpy
    from intratalkerpy import perturbation, tf

    assert intratalkerpy.__version__
    assert callable(tf.IntraTalker_analysis)
    assert callable(tf.get_significant_tfs)
    assert hasattr(perturbation, "mt")
    assert hasattr(perturbation, "ut")
    assert hasattr(perturbation, "pl")


def test_external_dependency_functions_are_callable():
    import decoupler as dc
    from deltacorrpy import colDeltaCorpartial

    assert callable(dc.tl.rankby_group)
    assert callable(colDeltaCorpartial)


def test_score_ulm_is_used_when_tf_activities_is_missing():
    from intratalkerpy.tf.utils import load_tf_activities

    anndataobject = ad.AnnData(
        X=pd.DataFrame([[1, 2], [3, 4]], index=["cell1", "cell2"], columns=["gene1", "gene2"])
    )
    anndataobject.obs["celltype"] = ["a", "b"]
    anndataobject.obs["condition"] = ["x", "y"]
    anndataobject.obsm["score_ulm"] = pd.DataFrame(
        [[0.1, 0.2], [0.3, 0.4]], index=anndataobject.obs_names, columns=["tf1", "tf2"]
    )

    tf_activities = load_tf_activities(
        anndataobject,
        None,
        {"decoupler_matrix_format": None},
    )

    assert isinstance(tf_activities, ad.AnnData)
    assert list(tf_activities.obs_names) == ["cell1", "cell2"]
    assert list(tf_activities.var_names) == ["tf1", "tf2"]


def test_transcription_factor_reference_data_is_packaged():
    data_files = [
        "rtf_db_human.csv",
        "rtf_db_mouse.csv",
        "ligands_human.csv",
        "ligands_mouse.csv",
    ]

    for filename in data_files:
        resource = resources.files("intratalkerpy.tf.data").joinpath(filename)
        assert resource.is_file(), f"{filename} is missing from package data"

        with resource.open(newline="") as csv_file:
            rows = list(csv.DictReader(csv_file))

        assert rows, f"{filename} should contain reference rows"
