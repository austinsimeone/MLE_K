from datetime import datetime
import logging
import yaml
import argparse
from prefect import task, Flow, Parameter, Client
from typing import Dict

from scripts.training.retrieve import retrieve
from scripts.training.preprocess import preprocess
from scripts.training.split import split
from scripts.training.train_val_test import train_val_test
from scripts.training.register import register


@task(log_stdout=True)
def config_task():
    with open("config.yaml") as f:
        CONFIG = yaml.load(f, Loader=yaml.FullLoader)
    CONFIG["experiment"] = datetime.today().strftime("%m/%d/%Y-%H:%M")
    return CONFIG


@task(log_stdout=True)
def retrieve_task(CONFIG: Dict):
    retrieve(CONFIG)


@task(log_stdout=True)
def preprocess_task(CONFIG: Dict):
    preprocess(CONFIG)


@task(log_stdout=True)
def split_task(CONFIG: Dict):
    split(CONFIG)


@task(log_stdout=True)
def train_val_test_task(CONFIG: Dict, experiment: str, model: str):
    train_val_test(CONFIG, experiment, model)


@task(log_stdout=True)
def register_task(CONFIG: Dict, experiment: str):
    register(CONFIG, experiment)


with Flow("Pipeline") as flow:
    # Grab pipeline configuration
    CONFIG = config_task()
    # Prepare data for training
    retrieved = retrieve_task(CONFIG=CONFIG)
    preprocessed = preprocess_task(CONFIG=CONFIG, upstream_tasks=[retrieved])
    splits = split_task(CONFIG=CONFIG, upstream_tasks=[preprocessed])
    # Run experiments on train, val, and test data splits
    exp1 = train_val_test_task(CONFIG, CONFIG["experiment"], "sklearn.tree.DecisionTreeClassifier", upstream_tasks=[splits])
    exp2 = train_val_test_task(CONFIG, CONFIG["experiment"], "sklearn.ensemble.RandomForestClassifier", upstream_tasks=[splits])
    exp3 = train_val_test_task(CONFIG, CONFIG["experiment"], "sklearn.svm.SVC", upstream_tasks=[splits])
    exp4 = train_val_test_task(CONFIG, CONFIG["experiment"], "src.model.pytorch.MultiClassClassifier", upstream_tasks=[splits])
    registered = register_task(CONFIG, CONFIG["experiment"], upstream_tasks=[exp1, exp2, exp3, exp4])

# Grab command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--register", help="whether to register the flow", default="false")
args = parser.parse_args()

if args.register.lower() == "true":
    prefect_project = "ml template"
    client = Client()
    client.create_project(project_name=prefect_project)
    flow.register(project_name=prefect_project)
else:
    flow.run()
