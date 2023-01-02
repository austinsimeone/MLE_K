import pandas as pd


def multiple_substring_replace(string: str, replace_dict: dict) -> str:
    for key, val in replace_dict.items():
        string = string.replace(key, val)
    return string


def clean_column_names(input: pd.DataFrame, replace_dict: dict = {" ": "_", "(": "", ")": ""}) -> pd.DataFrame:
    input.columns = [multiple_substring_replace(x, replace_dict) for x in input.columns]
    return input
