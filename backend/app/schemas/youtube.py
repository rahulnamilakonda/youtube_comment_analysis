from pydantic import BaseModel, Field
from typing import List, Dict, Optional

# ----- Sentiment Result (Video Analysis) -----------------------------

class VideoAnalysisRequest(BaseModel):
    video_id: str = Field(..., description="YouTube video ID")
    max_comments: int = Field(25, ge=1, le=100)

class SentimentSummary(BaseModel):
    total_comments_fetched: int
    positive_pct: float
    negative_pct: float
    neutral_pct: float
    avg_confidence: float

class EngagementMetrics(BaseModel):
    total_likes: int
    avg_likes_per_comment: float
    top_comment_likes: int
    total_replies: int

class TopComment(BaseModel):
    comment_id: str
    author: str
    text: str
    likes: int
    reply_count: int
    sentiment: str
    confidence: float

class VideoAnalysisResponse(BaseModel):
    video_id: str
    summary: SentimentSummary
    engagement: EngagementMetrics
    trend: List[str]
    wordcloud_image: Optional[str] = None
    top_comments: List[TopComment]
