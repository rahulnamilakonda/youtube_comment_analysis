from loguru import logger
import pytest
import mlflow
import yaml
import joblib
import numpy as np
from youtube_comment_analysis.config import MODELS_DIR
from mlflow import tracking

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

@pytest.fixture
def model_info():
    params = load_params()
    model_name = params['base']['project']
    client = tracking.MlflowClient()
    
    # Industrial Practice: Polling for Alias propagation.
    import time
    max_retries = 5
    for i in range(max_retries):
        try:
            version = client.get_model_version_by_alias(model_name, "challenger")
            return client, model_name, version
        except Exception:
            logger.info(f"Waiting for @challenger alias on '{model_name}' (Attempt {i+1}/{max_retries})...")
            time.sleep(5)
    
    logger.warning(f"No version found with @challenger alias for model '{model_name}'.")
    pytest.skip(f"No @challenger model found.")

def test_model_performance_threshold(model_info):
    """
    Industrial Test: Performance Gatekeeping.
    Checks if accuracy > 85% and promotes to @champion if passed.
    """
    client, model_name, version = model_info
    
    import json
    metrics_path = "reports/metrics.json"
    
    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    except FileNotFoundError:
        logger.error(f"Metrics file {metrics_path} not found.")
        pytest.fail(f"Metrics file {metrics_path} not found.")

    accuracy = metrics.get("accuracy")
    assert accuracy is not None, f"Accuracy metric not found in {metrics_path}"
    
    THRESHOLD = 0.85
    if accuracy >= THRESHOLD:
        # Promote to 'champion' alias
        client.set_registered_model_alias(
            name=model_name,
            alias="champion",
            version=version.version
        )
        logger.success(f"Performance PASSED ({accuracy:.4f} >= {THRESHOLD}). Promoted to @champion.")
    else:
        logger.error(f"Performance FAILED ({accuracy:.4f} < {THRESHOLD}). Model remains @challenger.")
        pytest.fail(f"Performance failed.")

@pytest.mark.parametrize("sample_text", [
    "This is an amazing video, I loved the content!",  # Positive
    "I really hate this, it was a waste of time.",      # Negative
    "The video was okay, nothing special."             # Neutral
])
def test_model_smoke_prediction(sample_text):
    """
    Industrial Test: Smoke Test with Parametrized Inputs.
    Verifies that the local saved model can generate predictions for 
    different sentiment types (Positive, Negative, Neutral).
    """
    params = load_params()
    clf = joblib.load(MODELS_DIR / "model.pkl")
    tfidf = joblib.load(MODELS_DIR / "tfidf.pkl")
    
    # Preprocess
    X_tfidf = tfidf.transform([sample_text])
    
    # Extract numeric features matching training logic
    char_count = len(sample_text)
    word_count = len(sample_text.split())
    avg_word_len = char_count / (word_count + 1e-6)
    X_numeric = np.array([[char_count, word_count, avg_word_len]])
    
    from scipy.sparse import hstack
    X_combined = hstack([X_tfidf, X_numeric])
    
    prediction = clf.predict(X_combined)
    
    assert len(prediction) == 1
    assert prediction[0] in [-1, 0, 1]
    logger.success(f"Smoke test passed for input: '{sample_text[:30]}...' -> Prediction: {prediction[0]}")
