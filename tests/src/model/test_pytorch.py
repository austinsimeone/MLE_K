import pandas as pd

from src.model.pytorch import MultiClassClassifier


def test_df_to_loader():
    df = pd.DataFrame([{"col1": 0, "col2": 0, "col3": 0, "col4": 0}])
    model = MultiClassClassifier(seed=7)
    tensor = model.df_to_tensor(df)
    loader = model.tensor_to_loader(tensor, batch_size=32, num_workers=0)


def test_predict():
    df = pd.DataFrame([{"col1": 0, "col2": 0, "col3": 0, "col4": 0}])
    model = MultiClassClassifier(seed=7)
    prediction = model.predict(df)
    assert len(prediction) == 1
