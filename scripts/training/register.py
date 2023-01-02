import logging
import argparse
import yaml
import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

from src.model.utils import fetch_logged_data


def register(CONFIG, experiment):
    client = MlflowClient()
    experiment_id = client.get_experiment_by_name(experiment).experiment_id
    best_run = mlflow.search_runs(
        experiment_id,
        max_results=1,
        order_by=["metric." + CONFIG["model"]["decision_metric"] + " DESC"])[0]
    print(best_run)
    params, metrics, tags, artifacts = fetch_logged_data(best_run)
    logging.info(f"Best model: run_id {best_run.run_id}, model_type {tags['model_type']}, metric_value {metrics[CONFIG['model']['decision_metric']]}")

    model_dict = {
        "experiment": experiment,
        "experiment_id": experiment_id,
        "run_id": best_run.run_id,
        "model_type": tags["model_type"],
        "decision_metric": CONFIG["model"]["decision_metric"],
        "metric_value": metrics[CONFIG["model"]["decision_metric"]],
        "feature_cols": CONFIG["model"]["num_cols"] + CONFIG["model"]["cat_cols"],
        "params": params,
    }
    with open("./mlruns/model.yml", "w") as outfile:
        yaml.dump(model_dict, outfile)
    logging.info("Registered best model to ./mlruns/model.yml")


if __name__ == "__main__":
    # Grab command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", help="experiment id", default="None")
    args = parser.parse_args()

    # Setup config and logging
    with open("config.yaml") as f:
        CONFIG = yaml.load(f, Loader=yaml.FullLoader)

    logging.basicConfig(
        level=CONFIG["log_level"],
        format=CONFIG["log_format"],
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    logging.info("---Registering best model---")
    register(CONFIG, args.experiment)
