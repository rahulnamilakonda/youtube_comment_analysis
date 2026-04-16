from pathlib import Path
import re
import string
import pandas as pd
from loguru import logger
import yaml
from sklearn.model_selection import train_test_split
from youtube_comment_analysis.config import RAW_DATA_DIR, TRAIN_DATA_PATH, TEST_DATA_PATH, INTERIM_DATA_DIR

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def clean_text(text: str) -> str | None:
    if not isinstance(text, str):
        return None
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove Reddit-specific formatting
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'&gt;.*?\n', '', text)
    text = re.sub(r'r/\w+|u/\w+', '', text)
    # Remove mentions and hashtags
    text = re.sub(r'[@#]\w+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Collapse whitespace + lowercase
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text if len(text) >= 10 else None

import emoji

def deemojize(text: str):
    if isinstance(text, str) and len(text) > 0:
        return emoji.demojize(text)
    return None

def preprocess(raw_dir: Path, train_path: Path, test_path: Path):
    params = load_params()
    logger.info("Loading and merging datasets...")
    
    # 1. Primary Dataset (Reddit)
    primary_path = raw_dir / "primary" / "comments.csv"
    logger.info("Primary path: " + str(primary_path))
    df_primary = pd.read_csv(primary_path)

    df_primary = df_primary.rename(columns=params['preprocessing']['rename_columns'])
    # Keep Negative (-1)
    df_primary['Sentiment'] = df_primary['Sentiment'].map(lambda x: -1 if str(x).lower() == 'negative' else 1)
    df_primary.rename(columns={'Sentiment': 'sentiment'}, inplace=True)
    df_neg = df_primary[df_primary['sentiment'] == -1].copy()


    df_neg = df_neg[['sentiment', 'Body', 'upvotes']]
    df_neg.columns = ['sentiment', 'text', 'upvotes']

    logger.info("Primary Dataset (Reddit) Dataset head: ")
    logger.info("Dataset columns: " + str(df_neg.columns))
    logger.info(df_neg.head())
    
    # 2. Twitter-Reddit Combined Dataset
    tr_path = raw_dir / "twitter_reddit_combined" / "Reddit_Data.csv"

    logger.info("Twitter-Reddit Combined Dataset path: " + str(tr_path))
    df_tr = pd.read_csv(tr_path)

    # Sample Neutral (category 0)
    df_neu = df_tr[df_tr['category'] == 0].sample(n=2500, random_state=42).copy()
    df_neu = df_neu.rename(columns={'category': 'sentiment', 'clean_comment': 'text'})
    df_neu['upvotes'] = 0
    df_neu = df_neu[['sentiment', 'text', 'upvotes']]

    logger.info("Twitter-Reddit Combined Dataset head: ")
    logger.info("Dataset columns: " + str(df_neu.columns))
    logger.info(df_neu.head())
    
    # 3. Twitter Entity Sentiment
    ts_path = raw_dir / "twitter_sentiment" / "twitter_training.csv"

    logger.info("Twitter-Reddit Combined Dataset path: " + str(ts_path))
    df_ts = pd.read_csv(ts_path, header=None, names=['id', 'entity', 'label', 'content'], on_bad_lines='skip')

    # Map Positive to 1
    df_pos = df_ts[df_ts['label'].str.lower() == 'positive'].sample(n=2800, random_state=42).copy()
    df_pos.rename({'label': 'sentiment'}, inplace=True)
    df_pos['sentiment'] = 1
    df_pos = df_pos.rename(columns={'content': 'text'})
    df_pos['upvotes'] = 0
    df_pos = df_pos[['sentiment', 'text', 'upvotes']]

    logger.info("Twitter Entity Sentiment head: ")
    logger.info("Dataset columns: " + str(df_pos.columns))
    logger.info(df_pos.head())

    
    # Merge
    df = pd.concat([df_neg, df_neu, df_pos], ignore_index=True)
    logger.info(f"Merged dataset shape: {df.shape}")
    
    # 3. Clean text
    logger.info("Cleaning and deemojizing text...")
    processed_col = params['columns']['text_processed']
    target_col = params['columns']['target']
    
    df[processed_col] = df['text'].apply(clean_text)
    df[processed_col] = df[processed_col].apply(deemojize)
    
    # 4. Handle Missing Values & Duplicates
    df = df.dropna(subset=[processed_col, target_col])
    df = df.drop_duplicates()
    
    # Save interim merged dataset for EDA
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(INTERIM_DATA_DIR / "dataset.csv", index=False)
    
    logger.info(f"Final data shape after cleaning: {df.shape}")
    
    # 5. Train-Test Split
    test_size = params['preprocessing']['train_test_split']
    random_state = params['base']['random_state']
    
    logger.info("Final Dataset Columns: " + str(df.columns))
    logger.info("Final Dataset Head")
    logger.info(df.head())
    logger.info("Dataset size: " + str(df.shape[0]))
    logger.info("Dataset target distribution: " + str(df.sentiment.value_counts()))
    

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[target_col]
    )
    
    # Save datasets
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.success(f"Train/Test datasets saved to {train_path.parent}")

if __name__ == "__main__":
    preprocess(
        raw_dir=RAW_DATA_DIR,
        train_path=TRAIN_DATA_PATH,
        test_path=TEST_DATA_PATH
    )
