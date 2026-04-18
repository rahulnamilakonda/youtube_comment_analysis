import joblib
from loguru import logger
import pandas as pd
import numpy as np
import mlflow
from pathlib import Path
from dotenv import load_dotenv
from mlflow import sklearn
import os
from youtube_comment_analysis.config import MODELS_DIR
from mlflow import tracking

# Load environment variables for MLflow/Dagshub
load_dotenv()

class SentimentPredictor:
    """
    Production-ready inference class.
    Encapsulates the model (from MLflow), vectorizer, and scaler.
    """
    def __init__(self, model_dir: Path = MODELS_DIR, use_registry: bool = True):
        try:
            # 1. Load Local Preprocessing Artifacts
            self.tfidf = joblib.load(model_dir / "tfidf.pkl") # load vectorizer
            self.scaler = joblib.load(model_dir / "scaler.pkl") # load scalar
            
            # 2. Load Model from Dagshub Registry or Local
            if use_registry:
                model_name = "youtube_comment_analysis"
                # We use the new @champion alias for production models
                model_uri = f"models:/{model_name}@champion"
                
                # Industrial Practice: Log specific version for traceability
                client = tracking.MlflowClient()
                try:
                    version_info = client.get_model_version_by_alias(model_name, "champion")
                    logger.info(f"Loading @champion model: {model_name} (v{version_info.version})")
                except Exception:
                    logger.warning("Could not find a model with @champion alias. Check Registry.")

                self.model = sklearn.load_model(model_uri)
            else:
                self.model = joblib.load(model_dir / "model.pkl")
                
            self.label_map = {-1: "Negative", 0: "Neutral", 1: "Positive"}
            logger.info("Predictor initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize SentimentPredictor: {e}")
            raise e
        
    def _extract_features(self, text: str):
        # Numeric features: char_count, word_count, avg_word_len
        char_count = len(text)
        word_count = len(text.split())
        avg_word_len = char_count / (word_count + 1e-6)
        
        numeric_feats = np.array([[char_count, word_count, avg_word_len]])
        scaled_numeric = self.scaler.transform(numeric_feats)

        # Text features: TF-IDF
        text_tfidf = self.tfidf.transform([text])
        
        from scipy.sparse import hstack
        X_combined = hstack([text_tfidf, scaled_numeric]).toarray()
        
        # Convert to DataFrame with string column names to match model signature
        X_df = pd.DataFrame(X_combined)
        X_df.columns = [str(i) for i in range(X_df.shape[1])]
        return X_df

    def predict(self, text: str):
        """Predict sentiment, label, and probabilities for a single input."""
        if not text:
            return None
        
        features = self._extract_features(text)
        
        # 1. Class Prediction
        prediction = self.model.predict(features)[0] # type: ignore
        
        # 2. Probability Prediction (Confidence)
        # LightGBM/Sklearn models return probabilities for each class [-1, 0, 1]
        probs = self.model.predict_proba(features)[0] # type: ignore
        classes = self.model.classes_ # type: ignore

        logger.info("classes: " + str(classes))
        logger.info("probabilities: " + str(probs))
        logger.info("prediction " + str(prediction))

        prob_dict = {self.label_map[int(c)]: float(p) for c, p in zip(classes, probs)}
        
        # Find index of the predicted class to get its specific confidence
        pred_idx = np.where(classes == prediction)[0][0]
        confidence = float(probs[pred_idx])

        return {
            "text": text,
            "sentiment": int(prediction),
            "label": self.label_map.get(int(prediction), "Unknown"),
            "confidence": confidence,
            "probabilities": prob_dict
        }

if __name__ == "__main__":
    # Example usage for production inference
    predictor = SentimentPredictor()
    sample_text = "This is an amazing video, I loved the content!"
    result = predictor.predict(sample_text)
    print(result)
