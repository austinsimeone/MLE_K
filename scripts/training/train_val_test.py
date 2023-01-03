import logging
import argparse
import yaml
from pathlib import Path
import pandas as pd
import mlflow

from src.model.utils import fetch_logged_data
from src.evaluation.metrics import class_metrics
from src.model.utils import locate_model


def train_val_test(CONFIG, experiment, model):
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=model) as run:
        # Log model type
        mlflow.set_tag("model_type", model)

        # Load data splits
        load_dir = Path("/".join([CONFIG["data"]["local_data_path"], "processed_split"]))
        file_name = CONFIG["data"]["file_name"]
        extension = CONFIG["data"]["extension"]
        train_df = pd.read_csv(str(load_dir / file_name) + "_train" + extension)
        val_df = pd.read_csv(str(load_dir / file_name) + "_val" + extension)
        test_df = pd.read_csv(str(load_dir / file_name) + "_test" + extension)
        logging.info(f"Loaded training datasets from {load_dir}")

        # Train model
        model_type = model.replace("src.model.", "").split(".")[0]
        if model_type == "sklearn":
            model = locate_model(model)(random_state=CONFIG["seed"], max_iter=CONFIG["model"]["max_epochs"])
            mlflow.log_param("seed", CONFIG["seed"])
            model.fit(
                train_df.filter(regex=f'^{CONFIG["model"]["feature_col"]}', axis=1),
                train_df[CONFIG["model"]["target_col"]],
            )
            mlflow.sklearn.log_model(
                model,
                "sklearn_model",
                signature=mlflow.models.signature.infer_signature(
                    train_df.filter(regex=f'^{CONFIG["model"]["feature_col"]}', axis=1),
                    train_df[CONFIG["model"]["target_col"]],
                ),
            )
        else:
            logging.error(f"Unknown model type {model_type}")

        logging.info(f"Trained and stored model in run {run.info.run_id}")

        # Calculate and log metrics
        mlflow.log_metrics(
            class_metrics(
                train_df[CONFIG["model"]["target_col"]], model.predict(train_df.filter(regex=f'^{CONFIG["model"]["feature_col"]}', axis=1)), "train"
            )
        )
        mlflow.log_metrics(
            class_metrics(val_df[CONFIG["model"]["target_col"]], model.predict(val_df.filter(regex=f'^{CONFIG["model"]["feature_col"]}', axis=1)), "val")
        )
        mlflow.log_metrics(
            class_metrics(test_df[CONFIG["model"]["target_col"]], model.predict(test_df.filter(regex=f'^{CONFIG["model"]["feature_col"]}', axis=1)), "test")
        )

    # fetch logged data
    params, metrics, tags, artifacts = fetch_logged_data(run.info.run_id)
    for key, val in metrics.items():
        logging.info(f"{key}:{round(val, 3)}")


if __name__ == "__main__":
    # Grab command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="model type", default="sklearn.ensemble.RandomForestClassifier")
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
    logging.info("---Training and validating model---")
    train_val_test(CONFIG, args.experiment, args.model)
