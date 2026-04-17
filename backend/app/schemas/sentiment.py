from pydantic import BaseModel, Field
from typing import List, Optional

# ----- Batch Sentiment Request (Comment by Comment Analysis) -----------------------

class CommentItem(BaseModel):
    comment_id: str
    comment_text: str

class BatchSentimentRequest(BaseModel):
    comments: List[CommentItem]

class SentimentResult(BaseModel):
    comment_id: str
    sentiment: str = Field(..., description="one of 'positive', 'negative', 'neutral'")
    confidence: Optional[float] = None

class BatchSentimentResponse(BaseModel):
    results: List[SentimentResult]


