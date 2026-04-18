import pytest
import mlflow
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from mlflow import tracking, models
from youtube_comment_analysis.config import MODELS_DIR, PROCESSED_DATA_DIR

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

@pytest.fixture
def model_info():
    params = load_params()
    model_name = params['base']['project']
    client = tracking.MlflowClient()
    
    # Industrial Practice: Retry logic for cloud registry propagation.
    # There is often a slight delay between a stage transition and the model 
    # being visible in the 'Staging' list on remote servers like Dagshub.
    import time
    max_retries = 5
    for i in range(max_retries):
        versions = client.get_latest_versions(model_name, stages=["Staging"])
        if versions:
            return client, model_name, versions[0]
        
        logger.info(f"Waiting for model '{model_name}' to appear in Staging (Attempt {i+1}/{max_retries})...")
        time.sleep(5) # Wait 5 seconds before retrying
    
    logger.warning(f"No version of model '{model_name}' found in Staging after {max_retries} attempts.")
    pytest.skip(f"No version of model '{model_name}' found in Staging.")

@pytest.mark.parametrize("sample_size", [1, 5, 10])
def test_model_signature_verification(model_info, sample_size):
    """
    Industrial Test: Signature Validation with Real Data.
    We use @pytest.mark.parametrize to ensure the model signature is 
    robust across different batch sizes (1, 5, and 10 rows).
    """
    client, model_name, version = model_info
    params = load_params()
    
    # 1. Load the model as a PyFunc
    model_uri = f"models:/{model_name}/{version.version}"
    logger.info(f"Loading model for signature verification: {model_uri}")
    loaded_model = mlflow.pyfunc.load_model(model_uri) 
    
    # 2. Get a real sample of processed data based on sample_size
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test_features.csv").iloc[:sample_size]
    
    # 3. Prepare the input
    import joblib
    from scipy.sparse import hstack
    
    tfidf = joblib.load(MODELS_DIR / "tfidf.pkl")
    processed_col = params['columns']['text_processed']
    numeric_cols = params['columns']['numeric_features']
    
    X_tfidf = tfidf.transform(test_df[processed_col])
    X_numeric = test_df[numeric_cols].values
    X_combined = hstack([X_tfidf, X_numeric]).toarray()

    # Industrial Practice: Use Pandas DataFrame with string column names.
    # This must match the training signature exactly.
    X_com_df = pd.DataFrame(X_combined)
    X_com_df.columns = [str(i) for i in range(X_com_df.shape[1])]
    
    logger.info(f"Input shape for batch size {sample_size}: {X_com_df.shape}")
    
    # 4. Verify prediction
    try:
        preds = loaded_model.predict(X_com_df)
        assert len(preds) == sample_size
        logger.success(f"Signature verified for batch size {sample_size}.")
    except Exception as e:
        logger.error(f"Signature mismatch for batch size {sample_size}, Shape of the data is {X_com_df.shape}")
        pytest.fail(f"Signature validation failed: {e}, Shape of the data is {X_com_df.shape}")
