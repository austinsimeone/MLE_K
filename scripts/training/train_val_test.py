import logging
import argparse
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import mlflow

from src.model.utils import fetch_logged_data
from src.evaluation.metrics import multi_class_metrics
from src.model.utils import locate_model


def train_val_test(CONFIG, experiment, model):
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=model) as run:
        # Log model type
        mlflow.set_tag("model_type", model)

        # Load data splits
        load_dir = Path("/".join([CONFIG["data"]["local_data_path"], "processed"]))
        file_name = CONFIG["data"]["file_name"]
        extension = CONFIG["data"]["extension"]
        train_df = pd.read_parquet(str(load_dir / file_name) + "_train" + extension)
        val_df = pd.read_parquet(str(load_dir / file_name) + "_val" + extension)
        test_df = pd.read_parquet(str(load_dir / file_name) + "_test" + extension)
        logging.info(f"Loaded training datasets from {load_dir}")

        # Train model
        model_type = model.replace("src.model.", "").split(".")[0]
        if model_type == "sklearn":
            model = locate_model(model)(random_state=CONFIG["seed"])
            mlflow.log_param("seed", CONFIG["seed"])
            model.fit(
                train_df[CONFIG["model"]["feature_cols"]],
                train_df[CONFIG["model"]["target_col"]],
            )
            mlflow.sklearn.log_model(
                model,
                "sklearn_model",
                signature=mlflow.models.signature.infer_signature(
                    train_df[CONFIG["model"]["feature_cols"]],
                    train_df[CONFIG["model"]["target_col"]],
                ),
            )
        elif model_type == "pytorch":
            model = locate_model(model)(
                seed=CONFIG["seed"],
                hidden_dim=CONFIG["model"]["hidden_dim"],
                learning_rate=CONFIG["model"]["learning_rate"],
                input_dim=len(CONFIG["model"]["feature_cols"]),
                total_steps=CONFIG["model"]["batch_size"] * train_df[CONFIG["model"]["feature_cols"]].shape[0] // CONFIG["model"]["batch_size"],
            )

            # Create data loaders
            train_tensor = model.df_to_tensor(
                train_df[CONFIG["model"]["feature_cols"] + [CONFIG["model"]["target_col"]]], target_col=CONFIG["model"]["target_col"]
            )
            val_tensor = model.df_to_tensor(
                val_df[CONFIG["model"]["feature_cols"] + [CONFIG["model"]["target_col"]]], target_col=CONFIG["model"]["target_col"]
            )
            test_tensor = model.df_to_tensor(
                test_df[CONFIG["model"]["feature_cols"] + [CONFIG["model"]["target_col"]]], target_col=CONFIG["model"]["target_col"]
            )
            train_loader = model.tensor_to_loader(train_tensor, CONFIG["model"]["batch_size"], CONFIG["model"]["num_workers"], shuffle=True)
            val_loader = model.tensor_to_loader(val_tensor, CONFIG["model"]["batch_size"], CONFIG["model"]["num_workers"], shuffle=False)
            test_loader = model.tensor_to_loader(test_tensor, CONFIG["model"]["batch_size"], CONFIG["model"]["num_workers"], shuffle=False)

            # Setup and run trainer
            trainer = model.setup_trainer(experiment, run.info.run_id, CONFIG["model"]["max_epochs"])
            trainer.fit(model, train_loader, val_loader)
            trainer.test(test_dataloaders=test_loader)
            mlflow.pytorch.log_model(
                model,
                "pytorch_model",
                signature=mlflow.models.signature.infer_signature(
                    train_df[CONFIG["model"]["feature_cols"]],
                    train_df[CONFIG["model"]["target_col"]],
                ),
            )
        else:
            logging.error(f"Unknown model type {model_type}")
        logging.info(f"Trained and stored model in run {run.info.run_id}")

        # Calculate and log metrics
        mlflow.log_metrics(multi_class_metrics(train_df[CONFIG["model"]["target_col"]], model.predict(train_df[CONFIG["model"]["feature_cols"]]), "train"))
        mlflow.log_metrics(multi_class_metrics(val_df[CONFIG["model"]["target_col"]], model.predict(val_df[CONFIG["model"]["feature_cols"]]), "val"))
        mlflow.log_metrics(multi_class_metrics(test_df[CONFIG["model"]["target_col"]], model.predict(test_df[CONFIG["model"]["feature_cols"]]), "test"))

    # fetch logged data
    params, metrics, tags, artifacts = fetch_logged_data(run.info.run_id)
    for key, val in metrics.items():
        logging.info(f"{key}:{round(val,3)}")


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
