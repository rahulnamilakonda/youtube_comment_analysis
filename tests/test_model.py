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
    
    # Industrial Practice: Retry logic for cloud registry propagation.
    import time
    max_retries = 5
    for i in range(max_retries):
        versions = client.get_latest_versions(model_name, stages=["Staging"])
        if versions:
            return client, model_name, versions[0]
        
        logger.info(f"Waiting for model '{model_name}' to appear in Staging (Attempt {i+1}/{max_retries})...")
        time.sleep(5)
    
    logger.warning(f"No version of model '{model_name}' found in Staging after {max_retries} attempts.")
    pytest.skip(f"No version of model '{model_name}' found in Staging.")

def test_model_performance_threshold(model_info):
    """
    Industrial Test: Performance Gatekeeping.
    Checks if the model accuracy > 85% and promotes to Production if passed.
    Uses the local metrics.json generated during the evaluate stage.
    """
    client, model_name, version = model_info
    
    # Industrial Practice: Load metrics from local artifact for robustness
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
        # Promote to Production
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Production",
            archive_existing_versions=True
        )
        logger.success(f"Performance PASSED ({accuracy:.4f} >= {THRESHOLD}). Promoted to PRODUCTION.")
    else:
        logger.error(f"Performance FAILED ({accuracy:.4f} < {THRESHOLD}). Model remains in Staging.")
        pytest.fail(f"Performance FAILED ({accuracy:.4f} < {THRESHOLD}). Model remains in Staging.")

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
