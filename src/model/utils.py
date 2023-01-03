from pathlib import Path
from pydoc import locate
import mlflow
import pandas as pd
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec


def fetch_logged_data(run_id: str):
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(run_id).data
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
    return data.params, data.metrics, tags, artifacts


def locate_model(model_type: str):
    model = locate(model_type)
    return model


def create_empty_ohe(CONFIG):
    load_dir = Path("/".join([CONFIG["data"]["local_data_path"], "processed_split"]))
    file_name = CONFIG["data"]["file_name"]
    extension = CONFIG["data"]["extension"]
    column_names_df = pd.read_csv(str(load_dir / file_name) + "_train" + extension, header=0, nrows=0)
    return column_names_df
