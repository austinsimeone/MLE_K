import pandas as pd

from src.processing import features


def test_clean_column_names():
    df_cols = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]
    test_df_cols = [x.replace(" ", "_").replace("(", "").replace(")", "") for x in df_cols]
    df = pd.DataFrame([{x: "" for x in df_cols}])
    test_df = pd.DataFrame([{x: "" for x in test_df_cols}])
    df = features.clean_column_names(df)
    assert test_df.equals(df)
