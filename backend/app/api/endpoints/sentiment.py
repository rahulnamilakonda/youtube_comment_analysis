from fastapi import APIRouter, HTTPException
from backend.app.schemas.sentiment import CommentItem, SentimentResult
from backend.app.services.sentiment_service import sentiment_service
from loguru import logger

router = APIRouter()

@router.post("/predict_batch", response_model=SentimentResult)
async def predict_batch(request: CommentItem):
    """
    Predict sentiment for a single comment. 
    Note: The frontend calls this 'predict_batch' but sends comments individually.
    """
    try:
        prediction = sentiment_service.analyze_text(request.comment_text)
        if not prediction:
            raise HTTPException(status_code=404, detail="Sentiment could not be determined")
            
        return {
            "comment_id": request.comment_id,
            "sentiment": prediction["sentiment"],
            "confidence": prediction["confidence"]
        }
    except Exception as e:
        logger.error(f"Prediction error for comment {request.comment_id}: {e}")
        raise HTTPException(status_code=500, detail="Error processing sentiment prediction")
