import gc
import multiprocessing
from pathlib import Path
import re

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

from youtube_comment_analysis.config import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

DATASET_URL = 'smagnan/1-million-reddit-comments-from-40-subreddits'
NMT_MODEL_NAME = 'Helsinki-NLP/opus-mt-ru-en'

DetectorFactory.seed = 42

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
    if not isinstance(text, str):
        return None
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove Reddit-specific formatting
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # [text](url) markdown links
    text = re.sub(r'&gt;.*?\n', '', text)        # quoted text (>)
    text = re.sub(r'r/\w+|u/\w+', '', text)     # subreddit/user mentions
    
    # Remove mentions, hashtags
    text = re.sub(r'[@#]\w+', '', text)
    
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Convert to lower case
    text = text.lower()

    # Discard if too short after cleaning
    # WHY 10? Less than 10 chars = no sentiment signal
    return text if len(text) >= 10 else None

def deemojize(text:str):
    return emoji.demojize(text)



def apply_cleaning_col(df: pd.DataFrame, text_col: str, batch_size: int = 100000,
                           cpu_fraction: float = 0.5) -> pd.DataFrame:
    logger.info("Applying text cleaning...")
    original_count = len(df)

    # use only 50% of available cores
    n_workers = max(1, int(multiprocessing.cpu_count() * cpu_fraction))
    logger.info(f"Using {n_workers}/{multiprocessing.cpu_count()} cores")
    logger.info(f"Detecting language for {len(df):,} rows in batches of {batch_size:,}...")
    
    # Stage 1: clean body
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
    
    # Stage 2: Convert emoji's to text
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


# --- Translating Russian to English ---
class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, i):
        return self.texts[i]
    
def detect_language_single(text: str) -> str:
    
    if not isinstance(text, str) or len(text.strip()) < 10:
        return 'unknown'
    
    try:
        return detect(text)
    except LangDetectException:
        return 'unknown'

def detect_language_batch(df: pd.DataFrame,
                           text_col: str,
                           batch_size: int = 100000,
                           cpu_fraction: float = 0.5) -> pd.DataFrame:
    
    # use only 50% of available cores
    n_workers = max(1, int(multiprocessing.cpu_count() * cpu_fraction))
    logger.info(f"Using {n_workers}/{multiprocessing.cpu_count()} cores")
    logger.info(f"Detecting language for {len(df):,} rows in batches of {batch_size:,}...")

    texts = df[text_col].tolist()

    # split texts into batches of 100000
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    logger.info(f"Total batches: {len(batches)}")

    def process_batch(batch: list, batch_idx: int) -> list:
        # ✅ tqdm inside each batch to track row level progress
        return [
            detect_language_single(text)
            for text in tqdm(
                batch,
                desc=f"Batch {batch_idx + 1}/{len(batches)}",
                position=batch_idx,
                leave=True
            )
        ]
    # run batches in parallel
    batch_results = Parallel(n_jobs=n_workers, backend="threading")(
        delayed(process_batch)(batch, idx)
        for idx, batch in enumerate(batches)
    )

    # flatten batch results back to single list
    results = [lang for batch in batch_results for lang in batch]  # type: ignore

    df['language'] = results

    # drop unknown language rows
    before = len(df)
    df = df[df['language'] != 'unknown'].reset_index(drop=True)
    after = len(df)

    logger.info(f"Dropped {before - after:,} unknown language rows")
    logger.info(f"\nFull dataset language counts:\n"
                f"{df['language'].value_counts().to_string()}")

    return df

def translate_group(texts: list, model_name: str, model_path: Path = MODELS_DIR) -> list:
    
    if not texts:
        return []  # handle empty list early


    device = 0 if torch.cuda.is_available() else -1
    
    # check model is downloaded or not
    source = None

    local_model_path = model_path / model_name.replace("/", "_")

    if local_model_path.exists() and (local_model_path / "config.json").exists():
        print(f"Loading model from local path: {local_model_path}")
        source = str(local_model_path)
    else:
        logger.info(f"Downloading model: {model_name} to {local_model_path}")
        local_model_path.mkdir(parents=True, exist_ok=True)
        source = snapshot_download(
            repo_id=model_name,
            local_dir=str(local_model_path)
        )

    translator = pipeline(
        "translation",
        model=source,
        device=device,
        max_length=512,
        model_kwargs={"cache_dir": str(local_model_path)} # save to custom path on first download 
    ) # type: ignore
    
    dataset  = TextDataset(texts)
    results  = []
    
    for out in tqdm(
        translator(dataset, batch_size=64),
        total=len(texts),
        desc=f"Translating"
    ):
        if isinstance(out, list):
            results.append(out[0].get('translation_text', ''))
        else:
            results.append(out.get('translation_text', ''))
    
    # Free GPU memory before loading next model
    del translator
    gc.collect()
    torch.cuda.empty_cache()
    
    return results


# Step 4: Translate russian texts to english
def translate(df: pd.DataFrame, text_col: str, model_name: str) -> pd.DataFrame:
    ru_mask = df['language'] == 'ru'
    logger.info("Russian comments are selected and ready to translate")
    translated_texts = translate_group(df[ru_mask][text_col].values.tolist(), model_name)

    logger.info("Rows are translated to russian, Now adding them to dataframe")

    # Initialize with original text for non-Russian rows
    df['translated_body'] = df[text_col]

    # Overwrite only Russian rows with translated text
    df.loc[ru_mask, 'translated_body'] = translated_texts

    return df
 

# main
def process_data(input_path: Path, output_path: Path):
    
    logger.info("Started Process")

    df = pd.read_csv(input_path)

    logger.info("Dataset loaded")


    # Rename Columns
    df.rename(columns={'score': 'upvotes'}, inplace=True)

    logger.info("Columns renamed")

    # Drop columns
    df.drop(columns=['subreddit'], inplace=True)

    logger.info("Unneccesary columns dropped")

    # apply cleaning to body
    df = apply_cleaning_col(df=df, text_col='body')

    logger.success("Cleaning completed")

    # detect language
    df = detect_language_batch(df, 'text_clean')

    logger.success("Language detected for all the comments (body)")

    # translate ru to english
    # df = translate(df, 'text_clean', NMT_MODEL_NAME)

    # logger.success("Translated from russian to english")

    # save processed dataset
    df.to_csv(output_path, index=False)

    logger.success("Dataset saved")
    
    return    

def main(
    # ---- DEFAULT DATA STORAGE PATHS ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "processed.csv",
    # ----------------------------------------------
):
#    get_raw_data(download_path=input_path)

    # preprocess data
    process_data(input_path=input_path, output_path=output_path) 


if __name__ == "__main__":
    # --- Load env variables --- 
    load_dotenv()
    main()
