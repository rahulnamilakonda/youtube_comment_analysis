import gc
import multiprocessing
from pathlib import Path
import re
import string

from huggingface_hub import snapshot_download
from joblib import Parallel, delayed
import kagglehub
from loguru import logger
from dotenv import load_dotenv
import pandas as pd
import os
import tempfile
from langdetect import DetectorFactory, detect, LangDetectException
import emoji
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers.pipelines import pipeline
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk

from youtube_comment_analysis.config import INTERIM_DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

DATASET_URL = 'vijayj0shi/reddit-dataset-with-sentiment-analysis'

# --- Download raw data ---

def get_raw_data(download_path: Path):
    # ---- DOWNLOAD DATA FROM KAGGLE  ----
    logger.info("Processing dataset...")    
    path = None

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:   
            dataset_path = kagglehub.dataset_download(DATASET_URL, output_dir=tmp_dir)
            logger.info("Dataset downloaded") 

            files  = os.listdir(dataset_path)
            csv_file = [f for f in files if f.endswith(".csv")]

            if not csv_file:
                logger.error("No .pkl file found in dataset") 
                raise FileNotFoundError("No .pkl file found in dataset")

            file_path = os.path.join(dataset_path, csv_file[0])

            
            data: pd.DataFrame = pd.read_csv(file_path)
            data.to_csv(download_path, index=False)
            logger.info("Dataset Saved!") 
        
        except Exception as e:
            logger.error(f"Failed to download the dataset {e}")
            raise e

    logger.info("Path to dataset files")
    logger.success("Processing dataset complete.")
    # -----------------------------------------


# --- Inital preprocessing stage ---

def clean_text(text: str) -> str | None:
    """
    Stage 1 cleaning (pre-EDA):
      - Removes URLs, Reddit markdown, mentions, hashtags
      - Removes punctuation
      - Lowercases & collapses whitespace
      - Drops texts shorter than 10 chars (no sentiment signal)
    """
    if not isinstance(text, str):
        return None

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove Reddit-specific formatting
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # [text](url) markdown links
    text = re.sub(r'&gt;.*?\n', '', text)        # quoted text (>)
    text = re.sub(r'r/\w+|u/\w+', '', text)     # subreddit/user mentions

    # Remove mentions and hashtags
    text = re.sub(r'[@#]\w+', '', text)

    # Remove punctuation (keeps alphanumerics + spaces)
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Collapse whitespace + lowercase
    text = re.sub(r'\s+', ' ', text).strip().lower()

    return text if len(text) >= 10 else None

def deemojize(text:str):
    return emoji.demojize(text)


def stem_and_remove_stopwords(df: pd.DataFrame, text_col: str) -> str | None:
    nltk.download('stopwords')

    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    def apply(text: str) -> str | None:
        if not isinstance(text, str):
            return None
        
        tokens = text.split()
        tokens = [ps.stem(w) for w in tokens if w not in stop_words]
        
        return " ".join(tokens) if tokens else None

    df['text_processed'] = df[text_col].apply(apply)



def apply_cleaning_col(df: pd.DataFrame, text_col: str, batch_size: int = 100000,
                           cpu_fraction: float = 0.5) -> pd.DataFrame:
    logger.info("Applying text cleaning...")
    original_count = len(df)

    # use only 50% of available cores
    n_workers = max(1, int(multiprocessing.cpu_count() * cpu_fraction))
    logger.info(f"Using {n_workers}/{multiprocessing.cpu_count()} cores")
    logger.info(f"Detecting language for {len(df):,} rows in batches of {batch_size:,}...")
    
    # -------------------- Stage 1: clean body ---------------------
    texts = df[text_col].tolist()
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    logger.info(f"Total batches: {len(batches)}")

    def clean(batch: list, batch_idx: int) -> list:
        return [
            clean_text(text)
            for text in tqdm(
                batch,
                desc=f"Batch {batch_idx + 1}/{len(batches)}",
                position=batch_idx,   
                leave=True          
            )
        ]

    batch_results = Parallel(n_jobs=n_workers, backend="threading")(
        delayed(clean)(batch, idx)
        for idx, batch in enumerate(batches)
    )

    # flatten results
    results = [lang for batch in batch_results for lang in batch]  # type: ignore


    df['text_clean'] = results
    
    # Drop rows that became None after cleaning
    df = df.dropna(subset=['text_clean']).reset_index(drop=True)
    
    # --------------- Stage 2: Convert emoji's to text ------------------
    logger.info("Applying deemojizing...")
    def conv_emoji(batch: list, batch_idx: int) -> list:
        return [
            deemojize(text)
            for text in tqdm(
                batch,
                desc=f"Batch {batch_idx + 1}/{len(batches)}",
                position=batch_idx,   
                leave=True          
            )
        ]

    texts = df['text_clean'].tolist()
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    logger.info(f"Total batches: {len(batches)}")

    batch_results = Parallel(n_jobs=n_workers, backend="threading")(
        delayed(conv_emoji)(batch, idx)
        for idx, batch in enumerate(batches)
    )

    # flatten results
    results = [lang for batch in batch_results for lang in batch]  # type: ignore
    df['text_clean'] = results
    
    dropped = original_count - len(df)
    logger.info(f"Dropped {dropped:,} rows after cleaning | "
                f"Remaining: {len(df):,}")
    return df


# main
def process_data(input_path: Path, output_path: Path, text_col: str):
    
    logger.info("Started Process")

    # 1. Read CSV.
    df = pd.read_csv(input_path)

    logger.info("Dataset loaded")

    # 2. Apply cleaning to body
    df = apply_cleaning_col(df=df, text_col=text_col)

    logger.success("Cleaning completed")

    logger.success("Language detected for all the comments (body)")

    # 3. save processed dataset
    df.to_csv(output_path, index=False)

    logger.success("Dataset saved")
    
    return    

def main(
    # ---- DEFAULT DATA STORAGE PATHS ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = INTERIM_DATA_DIR / "processed.csv",
    # ----------------------------------------------
):
    # download the data
    get_raw_data(download_path=input_path)

    # preprocess data
    process_data(input_path=input_path, output_path=output_path, text_col='Body') 


if __name__ == "__main__":
    # --- Load env variables --- 
    load_dotenv()
    main()
