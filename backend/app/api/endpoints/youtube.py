from fastapi import APIRouter, HTTPException
from backend.app.schemas.youtube import VideoAnalysisRequest, VideoAnalysisResponse
from backend.app.services.youtube_service import youtube_service
from backend.app.services.sentiment_service import sentiment_service
from backend.app.services.visual_service import generate_wordcloud_base64
from loguru import logger
# import numpy as np

router = APIRouter()

@router.post("/analyze_video", response_model=VideoAnalysisResponse)
async def analyze_video(request: VideoAnalysisRequest):
    """
    Full video-level analysis: fetch comments, predict sentiment, 
    compute engagement, and generate a word cloud.
    """
    logger.info(f"Starting video analysis for video_id: {request.video_id}")
    try:
        # 1. Fetch comments from YouTube Data API
        comments_data = youtube_service.fetch_comments_with_metadata(
            request.video_id,
        )
        
        if not comments_data:
            logger.warning(f"No comments found for video_id: {request.video_id}")
            raise HTTPException(status_code=404, detail="No comments found for this video")

        logger.info(f"Fetched {len(comments_data)} comments for video_id: {request.video_id}")

        # 2. Analyze Sentiment for all fetched comments
        total_likes = 0
        total_replies = 0
        confidences = []
        labels = []
        enriched_comments = []
        
        logger.info(f"Starting sentiment analysis for {len(comments_data)} comments")
        for c in comments_data:
            sentiment_data = sentiment_service.analyze_text(c["text"])
            if sentiment_data:
                c.update(sentiment_data)
                enriched_comments.append(c)
                labels.append(sentiment_data["sentiment"])
                confidences.append(sentiment_data["confidence"])
                total_likes += c["likes"]
                total_replies += c["reply_count"]
        
        logger.info(f"Sentiment analysis complete. Successfully enriched {len(enriched_comments)} comments")

        # 3. Calculate Summary Stats
        total = len(enriched_comments)
        pos_count = labels.count("positive")
        neg_count = labels.count("negative")
        neu_count = labels.count("neutral")
        
        summary = {
            "total_comments_fetched": total,
            "positive_pct": round((pos_count / total) * 100, 1) if total > 0 else 0,
            "negative_pct": round((neg_count / total) * 100, 1) if total > 0 else 0,
            "neutral_pct": round((neu_count / total) * 100, 1) if total > 0 else 0,
             "avg_confidence": round(float(sum(confidences)/len(confidences)), 4) if confidences else 0
            # "avg_confidence": round(float(np.mean(confidences)), 4) if confidences else 0
        }

        # 4. Calculate Engagement
        engagement = {
            "total_likes": total_likes,
            "avg_likes_per_comment": round(total_likes / total, 1) if total > 0 else 0,
            "top_comment_likes": max([c["likes"] for c in enriched_comments]) if enriched_comments else 0,
            "total_replies": total_replies
        }

        # 5. Build Trend and Top Comments
        # Sort by likes descending and take top max_comments
        sorted_comments = sorted(enriched_comments, key=lambda x: x["likes"], reverse=True)
        top_comments = sorted_comments[:request.max_comments]
        
        # Trend is sentiment labels for top comments in order
        trend = [c["sentiment"] for c in top_comments]

        # 6. Generate Word Cloud
        logger.info(f"Generating wordcloud for video_id: {request.video_id}")
        wordcloud = generate_wordcloud_base64([c["text"] for c in enriched_comments])
        logger.info(f"Wordcloud generated for video_id: {request.video_id}")

        logger.info(f"Successfully completed video analysis for video_id: {request.video_id}")
        return {
            "video_id": request.video_id,
            "summary": summary,
            "engagement": engagement,
            "trend": trend,
            "wordcloud_image": wordcloud,
            "top_comments": top_comments
        }

    except Exception as e:
        logger.error(f"Video analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
