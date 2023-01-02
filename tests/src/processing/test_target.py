import pandas as pd

from src.processing import target


def test_encode_target():
    targets = pd.Series(["setosa", "versicolor", "virginica"])
    targets = target.encode_target(targets)
    assert list(targets)[1] == 1


def test_decode_target():
    targets = pd.Series([0, 1, 2])
    targets = target.decode_target(targets)
    assert list(targets)[1] == "versicolor"
