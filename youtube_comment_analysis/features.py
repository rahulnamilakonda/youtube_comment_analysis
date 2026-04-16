from pathlib import Path
import pandas as pd
from loguru import logger
import yaml
from sklearn.preprocessing import RobustScaler
import joblib
from youtube_comment_analysis.config import TRAIN_DATA_PATH, TEST_DATA_PATH, PROCESSED_DATA_DIR, MODELS_DIR

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def extract_numeric_features(df: pd.DataFrame, processed_col: str) -> pd.DataFrame:
    df = df.copy()
    df['char_count'] = df[processed_col].apply(len)
    df['word_count'] = df[processed_col].apply(lambda x: len(x.split()))
    df['avg_word_len'] = df['char_count'] / (df['word_count'] + 1e-6)
    return df

def generate_features(train_input: Path, test_input: Path, output_dir: Path):
    params = load_params()
    logger.info("Extracting numeric features...")
    
    train_df = pd.read_csv(train_input)
    test_df = pd.read_csv(test_input)
    
    processed_col = params['columns']['text_processed']
    train_df = extract_numeric_features(train_df, processed_col)
    test_df = extract_numeric_features(test_df, processed_col)
    
    numeric_cols = params['columns']['numeric_features']
    
    if params['features']['scaling'] == 'robust':
        logger.info("Applying RobustScaler...")
        scaler = RobustScaler()
        # Ensure numeric_cols exist (some like upvotes come from preprocessing)
        existing_cols = [c for c in numeric_cols if c in train_df.columns]
        train_df[existing_cols] = scaler.fit_transform(train_df[existing_cols])
        test_df[existing_cols] = scaler.transform(test_df[existing_cols])
        
        # Save scaler
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    
    # Save datasets
    train_df.to_csv(output_dir / "train_features.csv", index=False)
    test_df.to_csv(output_dir / "test_features.csv", index=False)
    
    logger.success(f"Features saved to {output_dir}")

if __name__ == "__main__":
    generate_features(
        train_input=TRAIN_DATA_PATH,
        test_input=TEST_DATA_PATH,
        output_dir=PROCESSED_DATA_DIR
    )
