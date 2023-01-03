import pandas as pd


def encode_target(
    target: pd.Series,
    target_list: list = ["setosa", "versicolor", "virginica"],
) -> pd.Series:
    assert all(target.isin(target_list)), "Found unknown target value"
    return target.map({target_list[i]: i for i in range(len(target_list))})


def decode_target(
    target: pd.Series,
    target_list: list = ["setosa", "versicolor", "virginica"],
) -> pd.Series:
    assert all(target.isin(list(range(len(target_list))))), "Found unknown target value"
    return target.map({i: target_list[i] for i in range(len(target_list))})
