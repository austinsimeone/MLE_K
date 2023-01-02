import logging
import yaml
from pathlib import Path
import pandas as pd

from src.processing.features import numeric_transformer, categorical_transformer


def preprocess(CONFIG):
    # Load raw data
    load_dir = Path("/".join([CONFIG["data"]["local_data_path"], "raw"]))
    file_name = CONFIG["data"]["file_name"] + CONFIG["data"]["extension"]
    df = pd.read_csv(load_dir / file_name)
    logging.info(f"Loaded {df.shape[0]} rows of data")

    # Process numeric data
    num_features = numeric_transformer(df[CONFIG["model"]["num_cols"]])
    df[CONFIG["model"]["num_cols"]] = num_features

    # Process cat data
    df = categorical_transformer(CONFIG["model"]["cat_cols"], df)
    # put transformed columns back in df

    # if we were to need encoding it would go here
    # df[CONFIG["model"]["target_col"]] = encode_target(df[CONFIG["model"]["target_col"]])
    # logging.info("Encoded targets")

    # Save processed data
    save_dir = Path("/".join([CONFIG["data"]["local_data_path"], "processed"]))
    save_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_dir / file_name, index=False)
    logging.info(f"Saved processed data to {save_dir / file_name}")


if __name__ == "__main__":
    # Setup config and logging
    with open("config.yaml") as f:
        CONFIG = yaml.load(f, Loader=yaml.FullLoader)

    logging.basicConfig(
        level=CONFIG["log_level"],
        format=CONFIG["log_format"],
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    logging.info("---Preprocessing data---")
    preprocess(CONFIG)
