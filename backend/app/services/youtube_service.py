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

    def fetch_comments_with_metadata(
        self,
        video_id: str,
        max_comments: int = 500,       # hard cap — None means fetch everything
        order: str = "relevance",      # "relevance" | "time"
    ) -> list[dict]:
        """
        Fetches ALL comments for a video by paginating through every
        nextPageToken until exhausted or max_comments is reached.

        Why the old code only returned 20-25:
        - A single commentThreads().list() call returns at most 100 items
          (default is 20 when maxResults isn't set).
        - The response contains a nextPageToken when more pages exist.
        - Without looping on that token, you only ever see page 1.

        YouTube Data API v3 quota cost:
        - commentThreads.list costs 1 unit per call.
        - With maxResults=100, a video with 1 000 comments = 10 API calls = 10 units.
        - Default daily quota is 10 000 units, so you can fetch ~1 000 videos
          of that size per day before hitting limits.

        Args:
            video_id:     YouTube video ID (11-char string).
            max_comments: Stop fetching after collecting this many comments.
                          Set to None to fetch every available comment.
                          YouTube itself caps public comment access at ~10 000
                          regardless of what you request.
            order:        "relevance" (YouTube's ranking) or "time" (newest first).

        Returns:
            List of comment dicts, each with keys:
                comment_id, author, text, likes, reply_count
        """
        comments_data: list[dict] = []
        next_page_token: str | None = None
        page_num = 0

        logger.info(f"Fetching comments for video_id={video_id}, max={max_comments}, order={order}")

        while True:
            # How many to request on this page:
            # - API max per page is 100.
            # - If max_comments is set, don't request more than we still need.
            remaining = (max_comments - len(comments_data)) if max_comments else None
            page_size = min(100, remaining) if remaining is not None else 100

            try:
                request = self.youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    order=order,
                    textFormat="plainText",
                    maxResults=page_size,
                    pageToken=next_page_token,   # None on first call, token on subsequent
                )
                response = request.execute()
            except Exception as e:
                logger.error(
                    f"YouTube API error on page {page_num + 1} "
                    f"for video {video_id}: {e}"
                )
                raise

            page_num += 1
            items = response.get("items", [])
            logger.debug(f"Page {page_num}: received {len(items)} comments")

            for item in items:
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                comments_data.append({
                    "comment_id":  item["id"],
                    "author":      snippet.get("authorDisplayName", "@unknown"),
                    "text":        snippet["textDisplay"],
                    "likes":       int(snippet.get("likeCount", 0)),
                    "reply_count": int(item["snippet"].get("totalReplyCount", 0)),
                })

            # Check hard cap
            if max_comments and len(comments_data) >= max_comments:
                logger.info(
                    f"Reached max_comments cap ({max_comments}). "
                    f"Stopping after {page_num} pages."
                )
                break

            # Check if another page exists
            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                logger.info(
                    f"All comments fetched. "
                    f"Total: {len(comments_data)} across {page_num} page(s)."
                )
                break

        return comments_data


youtube_service = YouTubeService()