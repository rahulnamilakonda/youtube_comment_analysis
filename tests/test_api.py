import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
import os

# Ensure the backend and root packages are in the python path
sys.path.append(os.getcwd())

# Mock the services to avoid loading models and calling YouTube API during tests
with patch("backend.app.services.sentiment_service.SentimentService"), \
     patch("backend.app.services.youtube_service.YouTubeService"), \
     patch("backend.app.services.visual_service.generate_wordcloud_base64"):
    from backend.app.main import app

client = TestClient(app)

def test_health_check():
    """Industrial Test: Basic API availability."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

@patch("backend.app.api.endpoints.sentiment.sentiment_service")
def test_predict_batch_endpoint(mock_sentiment_service):
    """
    Industrial Test: Batch Prediction API Contract.
    Verifies that the endpoint correctly handles the list of comments 
    and returns results in the expected format.
    """
    # 1. Setup Mock Response
    mock_sentiment_service.analyze_batch.return_value = [
        {"comment_id": "st-1", "sentiment": "positive", "confidence": 0.95},
        {"comment_id": "st-2", "sentiment": "negative", "confidence": 0.88}
    ]

    # 2. Define Request Payload
    payload = {
        "comments": [
            {"comment_id": "st-1", "comment_text": "This is great!"},
            {"comment_id": "st-2", "comment_text": "I hate this."}
        ]
    }

    # 3. Call Endpoint
    response = client.post("/predict_batch", json=payload)

    # 4. Assertions
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 2
    assert data["results"][0]["sentiment"] == "positive"
    assert data["results"][1]["comment_id"] == "st-2"

@patch("backend.app.api.endpoints.youtube.youtube_service")
@patch("backend.app.api.endpoints.youtube.sentiment_service")
@patch("backend.app.api.endpoints.youtube.generate_wordcloud_base64")
def test_analyze_video_endpoint(mock_wordcloud, mock_sentiment_service, mock_youtube_service):
    """
    Industrial Test: Video Analysis API Contract.
    Verifies full integration of fetching, analyzing, and visual generation.
    """
    # 1. Setup Mock YouTube Service
    mock_youtube_service.fetch_comments_with_metadata.return_value = [
        {"comment_id": "y-1", "author": "@user1", "text": "Cool!", "likes": 10, "reply_count": 1}
    ]
    
    # 2. Setup Mock Sentiment Service
    mock_sentiment_service.analyze_text.return_value = {
        "sentiment": "positive",
        "confidence": 0.99
    }
    
    # 3. Setup Mock Wordcloud
    mock_wordcloud.return_value = "data:image/png;base64,mock_data"

    # 4. Define Request
    payload = {
        "video_id": "mock_vid_123",
        "max_comments": 5
    }

    # 5. Call Endpoint
    response = client.post("/analyze_video", json=payload)

    # 6. Assertions
    assert response.status_code == 200
    data = response.json()
    assert data["video_id"] == "mock_vid_123"
    assert "summary" in data
    assert data["summary"]["total_comments_fetched"] == 1
    assert data["summary"]["positive_pct"] == 100.0
    assert data["wordcloud_image"] == "data:image/png;base64,mock_data"
    assert len(data["top_comments"]) == 1
    assert data["top_comments"][0]["author"] == "@user1"

def test_predict_batch_invalid_payload():
    """Industrial Test: Error Handling for bad requests."""
    # Missing 'comments' key
    payload = {"wrong_key": []}
    response = client.post("/predict_batch", json=payload)
    assert response.status_code == 422 # Unprocessable Entity
