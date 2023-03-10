# import logging
# import yaml
# from pathlib import Path
# import pandas as pd
# from sklearn import datasets
# from src.processing.features import clean_column_names
#
#
# def retrieve(CONFIG):
#     # this is where we would add the retrieve portion.
#     # add code below and above
#     ##################################################
#
#     ##################################################
#     # Save dataset to data folder
#     save_dir = Path("/".join([CONFIG["data"]["local_data_path"], "raw"]))
#     save_dir.mkdir(parents=True, exist_ok=True)
#     file_name = CONFIG["data"]["file_name"] + CONFIG["data"]["extension"]
#     #df.frame.to_parquet(save_dir / file_name, index=False)
#     logging.info(f"Saved data to {save_dir / file_name}")
#
#
# if __name__ == "__main__":
#     # Setup config and logging
#     with open("config.yaml") as f:
#         CONFIG = yaml.load(f, Loader=yaml.FullLoader)
#
#     logging.basicConfig(
#         level=CONFIG["log_level"],
#         format=CONFIG["log_format"],
#         datefmt="%Y-%m-%d %H:%M:%S",
#         force=True,
#     )
#     logging.info("---Retrieving raw data---")
#     retrieve(CONFIG)
