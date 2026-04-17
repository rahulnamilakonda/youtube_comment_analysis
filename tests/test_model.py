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
    
    # Get the latest version in Staging
    versions = client.get_latest_versions(model_name, stages=["Staging"])
    if not versions:
        logger.warning(f"No version of model '{model_name}' found in Staging.")
        pytest.skip(f"No version of model '{model_name}' found in Staging.")
    
    return client, model_name, versions[0] # latest version is selected.

def test_model_performance_threshold(model_info):
    """
    Industrial Test: Performance Gatekeeping.
    Checks if the model accuracy > 85% and promotes to Production if passed.
    """
    client, model_name, version = model_info
    
    # Fetch metrics from the run that produced this model version
    run_id = version.run_id
    run = client.get_run(run_id)
    
    # We look for the 'accuracy' metric logged during evaluate stage
    accuracy = run.data.metrics.get("accuracy")
    
    assert accuracy is not None, f"Accuracy metric not found for run {run_id}"
    
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
