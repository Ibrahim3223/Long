# -*- coding: utf-8 -*-
"""
Pexels API client for fetching stock videos and photos.
FIXED: Added landscape-only video filtering (orientation="landscape")
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
        """Initialize Pexels client."""
        self.api_key = api_key or settings.PEXELS_API_KEY
        if not self.api_key:
            raise ValueError("PEXELS_API_KEY is required")
        
        self.headers = {"Authorization": self.api_key}
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.2  # 200ms between requests
    
    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def search_videos(
        self,
        query: str,
        per_page: int = 15,
        page: int = 1,
        orientation: str = "landscape",  # ✅ FIXED: Default to landscape
        size: str = "medium"
    ) -> List[Dict]:
        """
        Search for videos on Pexels.
        
        Args:
            query: Search query
            per_page: Results per page (max 80)
            page: Page number
            orientation: Video orientation - ALWAYS use "landscape" for 16:9 videos
            size: Video quality (large, medium, small)
        
        Returns:
            List of video data dictionaries
        """
        self._rate_limit()
        
        # ✅ ENFORCE landscape orientation for long-form videos
        if orientation != "landscape":
            logger.warning(f"⚠️ Forcing landscape orientation (was: {orientation})")
            orientation = "landscape"
        
        params = {
            "query": query,
            "per_page": min(per_page, 80),
            "page": page,
            "orientation": orientation,  # ✅ CRITICAL: Only landscape videos
            "size": size
        }
        
        try:
            logger.info(f"   🔍 Pexels search: '{query}' (landscape only)")
            response = self.session.get(
                f"{self.VIDEO_URL}/search",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            videos = data.get("videos", [])
            
            # ✅ DOUBLE-CHECK: Filter out any non-landscape videos
            landscape_videos = []
            for video in videos:
                width = video.get("width", 0)
                height = video.get("height", 0)
                
                # Ensure aspect ratio is horizontal (width > height)
                if width > height:
                    landscape_videos.append(video)
                else:
                    logger.debug(f"      ⚠️ Filtered out non-landscape video: {width}x{height}")
            
            logger.info(f"      ✅ Found {len(landscape_videos)} landscape videos (filtered from {len(videos)})")
            return landscape_videos
            
        except requests.exceptions.RequestException as e:
            logger.error(f"      ❌ Pexels API error: {e}")
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
            return data.get("photos", [])
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Pexels photo search error: {e}")
            return []
    
    def get_video_file_url(
        self,
        video_data: Dict,
        quality: str = "hd"
    ) -> Optional[str]:
        """
        Extract video file URL from video data.
        
        Args:
            video_data: Video data from Pexels API
            quality: Preferred quality (hd, sd, etc.)
        
        Returns:
            Video file URL or None
        """
        video_files = video_data.get("video_files", [])
        
        if not video_files:
            return None
        
        # ✅ Prefer landscape HD videos
        for video_file in video_files:
            if video_file.get("quality") == quality:
                width = video_file.get("width", 0)
                height = video_file.get("height", 0)
                
                # Ensure it's landscape
                if width > height:
                    return video_file.get("link")
        
        # Fallback: any landscape video
        for video_file in video_files:
            width = video_file.get("width", 0)
            height = video_file.get("height", 0)
            
            if width > height:
                return video_file.get("link")
        
        # Last resort: first available (shouldn't happen with our filtering)
        return video_files[0].get("link") if video_files else None
    
    def get_photo_url(
        self,
        photo_data: Dict,
        size: str = "large"
    ) -> Optional[str]:
        """Extract photo URL from photo data."""
        src = photo_data.get("src", {})
        return src.get(size) or src.get("large2x") or src.get("original")
