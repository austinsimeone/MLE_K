from datetime import datetime
import logging
import yaml
import argparse

from prefect import flow, task, Flow
from typing import Dict

from scripts.training.preprocess import preprocess
from scripts.training.split import split
from scripts.training.train_val_test import train_val_test
from scripts.training.register import register


@task()
def config_task():
    with open("config.yaml") as f:
        CONFIG = yaml.load(f, Loader=yaml.FullLoader)
    CONFIG["experiment"] = datetime.today().strftime("%m/%d/%Y-%H:%M")
    return CONFIG


@task()
def preprocess_task(CONFIG: Dict):
    preprocess(CONFIG)


@task()
def split_task(CONFIG: Dict):
    split(CONFIG)


@task()
def train_val_test_task(CONFIG: Dict, experiment: str, model: str):
    train_val_test(CONFIG, experiment, model)


@task()
def register_task(CONFIG: Dict, experiment: str):
    register(CONFIG, experiment)


@flow
def train_flow():
    # Grab pipeline configuration
    CONFIG = config_task()
    # Prepare data for training
    preprocessed = preprocess_task(CONFIG=CONFIG)
    splits = split_task(CONFIG=CONFIG)
    # Run experiments on train, val, and test data splits
    exp0 = train_val_test_task(CONFIG, CONFIG["experiment"], CONFIG["model"]["model_type"])
    registered = register_task(CONFIG, CONFIG["experiment"])


if __name__ == "__main__":
    # Grab command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--register", help="whether to register the flow", default="false")
    args = parser.parse_args()

    train_flow()
