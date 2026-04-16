import joblib
from loguru import logger
import pandas as pd
import numpy as np
from pathlib import Path
from youtube_comment_analysis.config import MODELS_DIR

class SentimentPredictor:
    """
    Production-ready inference class.
    Encapsulates the model, vectorizer, and scaler.
    """
    def __init__(self, model_dir: Path = MODELS_DIR):
        try:
            self.model = joblib.load(model_dir / "model.pkl") # load model
            self.tfidf = joblib.load(model_dir / "tfidf.pkl") # load vectorizer
            self.scaler = joblib.load(model_dir / "scaler.pkl") # load scalar
            logger.info("Predictor initialized successfully.")
        except FileNotFoundError as e:
            logger.error(f"Required model artifacts not found in {model_dir}: {e}")
            raise e
        
    def _extract_features(self, text: str):
        # Extracting numeric features
        # Numeric features: char_count, word_count, avg_word_len
        char_count = len(text)
        word_count = len(text.split())
        avg_word_len = char_count / (word_count + 1e-6)
        
        # numeric features: char_count, word_count, avg_word_len
        numeric_feats = np.array([[char_count, word_count, avg_word_len]])
        scaled_numeric = self.scaler.transform(numeric_feats)


        # Extract textual features using vectorizer.
        # Text features: TF-IDF
        text_tfidf = self.tfidf.transform([text])
        
        from scipy.sparse import hstack
        return hstack([text_tfidf, scaled_numeric])

    def predict(self, text: str):
        """Predict sentiment for a single input."""
        if not text:
            return None
        features = self._extract_features(text)
        prediction = self.model.predict(features)
        return prediction[0]

if __name__ == "__main__":
    # Example usage for production inference
    predictor = SentimentPredictor()
    sample_text = "This is an amazing video, I loved the content!"
    result = predictor.predict(sample_text)
    print(f"Text: {sample_text}")
    print(f"Predicted Sentiment: {result}")
