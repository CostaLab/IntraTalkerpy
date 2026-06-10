import csv
from importlib import resources


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
