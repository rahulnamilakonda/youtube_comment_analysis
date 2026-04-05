from pathlib import Path

import kagglehub
from loguru import logger
from dotenv import load_dotenv
import pandas as pd
import os
import tempfile

from youtube_comment_analysis.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

def get_raw_data(download_path: Path):
    # ---- DOWNLOAD DATA FROM KAGGLE  ----
    logger.info("Processing dataset...")    
    path = None

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:   
            dataset_path = kagglehub.dataset_download("gajalshankar/reddit-comments-from-subreddits-humour-and-news", output_dir=tmp_dir)
            logger.info("Dataset downloaded") 

            files  = os.listdir(dataset_path)
            pkl_file = [f for f in files if f.endswith(".pkl")]

            if not pkl_file:
                logger.error("No .pkl file found in dataset") 
                raise FileNotFoundError("No .pkl file found in dataset")

            file_path = os.path.join(dataset_path, pkl_file[0])

            
            data: pd.DataFrame = pd.read_pickle(file_path)
            data.to_csv(download_path, index=False)
            logger.info("Dataset Saved!") 
        
        except Exception as e:
            logger.error(f"Failed to download the dataset {e}")
            raise e

    logger.info("Path to dataset files")
    logger.success("Processing dataset complete.")
    # -----------------------------------------


def main(
    # ---- DEFAULT DATA STORAGE PATHS ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
   get_raw_data(download_path=input_path)


if __name__ == "__main__":
    # --- Load env variables --- 
    load_dotenv()
    main()
