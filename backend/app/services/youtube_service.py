import re
from googleapiclient.discovery import build
from backend.app.core.config import settings
from loguru import logger

class YouTubeService:
    def __init__(self):
        self.api_key = settings.YOUTUBE_API_KEY
        if not self.api_key:
            logger.warning("YouTube API Key not found in environment settings.")
        self.youtube = build("youtube", "v3", developerKey=self.api_key)

    def extract_video_id(self, url: str) -> str:
        """Extracts the video ID from a YouTube URL."""
        pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
        match = re.search(pattern, url)
        if match:
            return match.group(1)
        raise ValueError("Invalid YouTube URL")

    def fetch_comments_with_metadata(self, video_id: str):
        """Fetches comments with authors, likes, and reply counts."""
        try:
            request = self.youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                order="relevance",
                textFormat="plainText"
            )
            response = request.execute()
            
            comments_data = []
            for item in response.get("items", []):
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                comments_data.append({
                    "comment_id": item["id"],
                    "author": snippet.get("authorDisplayName", "@unknown"),
                    "text": snippet["textDisplay"],
                    "likes": int(snippet.get("likeCount", 0)),
                    "reply_count": int(item["snippet"].get("totalReplyCount", 0))
                })
            return comments_data
        except Exception as e:
            logger.error(f"Error fetching detailed comments for video {video_id}: {e}")
            raise e

youtube_service = YouTubeService()
