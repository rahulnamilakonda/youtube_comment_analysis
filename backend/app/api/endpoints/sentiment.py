from fastapi import APIRouter, HTTPException
from backend.app.schemas.sentiment import BatchSentimentRequest, BatchSentimentResponse
from backend.app.services.sentiment_service import sentiment_service
from loguru import logger

router = APIRouter()

@router.post("/predict_batch", response_model=BatchSentimentResponse)
async def predict_batch(request: BatchSentimentRequest):
    """
    Predict sentiment for a batch of comments.
    Order is preserved as per input array in the response.
    """
    logger.info(f"Received batch prediction request with {len(request.comments)} comments")
    try:
        results = sentiment_service.analyze_batch(request.comments)
        if not results:
            logger.warning("Batch processing returned no results.")
        
        logger.info(f"Batch prediction complete. Successfully processed {len(results)} comments")
        return {"results": results}
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail="Error processing batch sentiment prediction")
