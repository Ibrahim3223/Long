# FILE: autoshorts/video/pexels_client.py
# -*- coding: utf-8 -*-
"""
Pexels API client for fetching stock videos and photos.
Optimized: enforce landscape and prefer smaller files when desired.
"""
import requests
import logging
import time
from typing import List, Dict, Optional
from autoshorts.config import settings

logger = logging.getLogger(__name__)


class PexelsClient:
    """Client for interacting with Pexels API."""

    BASE_URL = "https://api.pexels.com/v1"
    VIDEO_URL = "https://api.pexels.com/videos"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.PEXELS_API_KEY
        if not self.api_key:
            raise ValueError("PEXELS_API_KEY is required")

        self.headers = {"Authorization": self.api_key}
        self.session = requests.Session()
        self.session.headers.update(self.headers)

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.18  # slight throttle

    def _rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def search_videos(
        self,
        query: str,
        per_page: int = 15,
        page: int = 1,
        orientation: str = "landscape",
        size: str = "medium"
    ) -> List[Dict]:
        """Search for videos on Pexels."""
        self._rate_limit()

        # Force landscape for long-form assets
        if orientation != "landscape":
            logger.debug(f"Forcing landscape orientation (was: {orientation})")
            orientation = "landscape"

        params = {
            "query": query,
            "per_page": min(per_page, 80),
            "page": page,
            "orientation": orientation,
            "size": size
        }

        try:
            logger.info(f"🔍 Pexels search: '{query}' (landscape only)")
            response = self.session.get(
                f"{self.VIDEO_URL}/search",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            videos = data.get("videos", []) or []

            # Keep only landscape videos
            landscape_videos: List[Dict] = []
            for video in videos:
                width = video.get("width", 0) or 0
                height = video.get("height", 0) or 0
                if width > height:
                    landscape_videos.append(video)

            logger.info(f"✅ Found {len(landscape_videos)} landscape videos (from {len(videos)})")
            return landscape_videos

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Pexels API error: {e}")
            return []

    def search_photos(
        self,
        query: str,
        per_page: int = 15,
        page: int = 1,
        orientation: str = "landscape"
    ) -> List[Dict]:
        """Search for photos on Pexels."""
        self._rate_limit()
        params = {
            "query": query,
            "per_page": min(per_page, 80),
            "page": page,
            "orientation": orientation
        }
        try:
            response = self.session.get(
                f"{self.BASE_URL}/search",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            return data.get("photos", []) or []
        except requests.exceptions.RequestException as e:
            logger.error(f"Pexels photo search error: {e}")
            return []

    def get_video_file_url(
        self,
        video_data: Dict,
        quality: Optional[str] = None
    ) -> Optional[str]:
        """
        Extract video file URL from video data.

        Args:
            video_data: Video data from Pexels API
            quality: Preferred quality (hd, sd). If None, use settings.

        Returns:
            Video file URL or None
        """
        preferred = (quality or settings.PEXELS_PREFERRED_QUALITY or "sd").lower()
        video_files = video_data.get("video_files", []) or []
        if not video_files:
            return None

        # Prefer smaller files in FAST_MODE / 'sd'
        def _pick(files, q):
            for vf in files:
                if (vf.get("quality") or "").lower() == q and (vf.get("width", 0) or 0) > (vf.get("height", 0) or 0):
                    return vf.get("link")
            return None

        link = _pick(video_files, preferred)
        if link:
            return link

        # Fallback to any landscape file (smallest first)
        landscape = [vf for vf in video_files if (vf.get("width", 0) or 0) > (vf.get("height", 0) or 0)]
        if not landscape:
            return video_files[0].get("link")

        # Sort by width ascending to pick a smaller file (faster download)
        landscape.sort(key=lambda f: f.get("width", 0) or 0)
        return landscape[0].get("link")
