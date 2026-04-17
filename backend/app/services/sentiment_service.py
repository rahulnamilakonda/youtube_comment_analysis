from youtube_comment_analysis.modeling.predict import SentimentPredictor
from backend.app.core.config import settings
from loguru import logger

class SentimentService:
    def __init__(self):
        try:
            self.predictor = SentimentPredictor(
                model_dir=settings.MODELS_DIR, 
                use_registry=True
            )
        except Exception as e:
            logger.error(f"Failed to initialize SentimentPredictor from Registry: {e}")
            logger.info("Falling back to local model artifacts...")
            self.predictor = SentimentPredictor(
                model_dir=settings.MODELS_DIR, 
                use_registry=False
            )
        
        # Extension contract requires lowercase strings
        self.label_map = {
            -1: "negative",
            0: "neutral",
            1: "positive"
        }

    def analyze_text(self, text: str):
        """
        Predicts sentiment and returns data matching the extension contract.
        """
        res = self.predictor.predict(text)
        if not res:
            return None
            
        return {
            "sentiment": self.label_map.get(res["sentiment"], "neutral"),
            "confidence": round(res["confidence"], 4)
        }

    def analyze_batch(self, comments: list):
        """
        Process a batch of comments. Matches the order of input.
        """
        results = []
        for item in comments:
            prediction = self.analyze_text(item.comment_text)
            if prediction:
                results.append({
                    "comment_id": item.comment_id,
                    "sentiment": prediction["sentiment"],
                    "confidence": prediction["confidence"]
                })
        return results

sentiment_service = SentimentService()
