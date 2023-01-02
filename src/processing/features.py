import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# common column transformations just in case things change
def multiple_substring_replace(string: str, replace_dict: dict) -> str:
    for key, val in replace_dict.items():
        string = string.replace(key, val)
    return string


def clean_column_names(input: pd.DataFrame, replace_dict: dict = {" ": "_", "(": "", ")": ""}) -> pd.DataFrame:
    input.columns = [multiple_substring_replace(x, replace_dict) for x in input.columns]
    return input


# seperate these functions for easier editing
def numeric_transformer(data: pd.DataFrame):
    transform_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    num_data = transform_pipeline.fit_transform(data)
    return num_data


def categorical_transformer(features: list, data: pd.DataFrame):
    cat_data = pd.get_dummies(data, columns=features)
    return cat_data
