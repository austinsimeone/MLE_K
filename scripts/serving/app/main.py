import logging
import yaml
import os
from logging import config
from yaml.loader import FullLoader
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import mlflow

from src.model.utils import locate_model
from src.processing.target import decode_target

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
elif model_type == "pytorch":
    model_path = "/".join(["mlruns", MODEL_CONFIG["experiment_id"], MODEL_CONFIG["run_id"], "checkpoints"])
    checkpoint_path = "/".join([model_path, os.listdir(model_path)[0]])
    model = locate_model(MODEL_CONFIG["model_type"]).load_from_checkpoint(checkpoint_path=checkpoint_path)
else:
    logging.error(f"Unknown model type {model_type}")
logging.info(f"Loaded {model_type} model from {model_path}")


@app.get("/")
async def read_root():
    return {"Model Server"}


class ModelRequestRow(BaseModel):
    sepal_length_cm: float
    sepal_width_cm: float
    petal_length_cm: float
    petal_width_cm: float


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
    predictions = model.predict(df[MODEL_CONFIG["feature_cols"]])
    return list(decode_target(pd.Series(predictions)))
