import logging
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import mlflow

from src.processing.features import numeric_transformer, categorical_transformer
from src.model.utils import create_empty_ohe

app = FastAPI(
    title="Model Server",
    description="Inference server for machine learning models",
    version="0.0.1",
)
# Setup config and logging
with open("config.yaml") as f:
    CONFIG = yaml.load(f, Loader=yaml.FullLoader)

logging.basicConfig(
    level=CONFIG["log_level"],
    format=CONFIG["log_format"],
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
logging.info("---Starting model server---")
with open("mlruns/model.yml") as f:
    MODEL_CONFIG = yaml.load(f, Loader=yaml.FullLoader)

logging.info("Loading model")
model_type = MODEL_CONFIG["model_type"].replace("src.model.", "").split(".")[0]
if model_type == "sklearn":
    model_path = "/".join(["mlruns", MODEL_CONFIG["experiment_id"], MODEL_CONFIG["run_id"], "artifacts", "sklearn_model"])
    model = mlflow.pyfunc.load_model(model_path)
else:
    logging.error(f"Unknown model type {model_type}")
logging.info(f"Loaded {model_type} model from {model_path}")


@app.get("/")
async def read_root():
    return {"Model Server"}


class ModelRequestRow(BaseModel):
    x1: float
    x2: float
    x3: str
    x4: float
    x5: float
    x6: str
    x7: str


class ModelRequest(BaseModel):
    __root__: List[ModelRequestRow]


class ModelResponse(BaseModel):
    __root__: List[str]


@app.post("/predict", response_model=ModelResponse)
async def get_prediction(request_data: ModelRequest):
    data = [dict(x) for x in request_data.__root__]
    df = pd.DataFrame.from_records(data)

    if not set(MODEL_CONFIG["feature_cols"]).issubset(df.columns):
        raise HTTPException(status_code=422, detail="Missing required features")
    # Process numeric data
    num_features = numeric_transformer(df[CONFIG["model"]["num_cols"]])
    df[CONFIG["model"]["num_cols"]] = num_features

    # Process cat data
    df = categorical_transformer(CONFIG["model"]["cat_cols"], df)
    # put transformed columns back in df

    df = pd.concat([df, create_empty_ohe(CONFIG)])

    df = df.fillna(0)

    predictions = model.predict(df).tolist()

    return predictions
