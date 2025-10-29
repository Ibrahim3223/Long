# -*- coding: utf-8 -*-
"""
Pixabay API client for video search - BACKUP for Pexels
"""

import logging
import time
import requests
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class PixabayClient:
    """Pixabay API client for video search."""
    
    BASE_URL = "https://pixabay.com/api/videos/"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Pixabay client.
        
        Args:
            api_key: Pixabay API key
        """
        if not api_key:
            from autoshorts.config import settings
            api_key = getattr(settings, 'PIXABAY_API_KEY', None)
        
        if not api_key:
            logger.warning("⚠️ PIXABAY_API_KEY not found, Pixabay search disabled")
            self.enabled = False
            return
        
        self.api_key = api_key
        self.enabled = True
        self.session = requests.Session()
        
        logger.info("🎬 PixabayClient initialized")
    
    def search_videos(
        self,
        query: str,
        per_page: int = 20,
        page: int = 1,
        min_width: int = 1920,
        min_height: int = 1080
    ) -> Dict:
        """
        Search for videos on Pixabay.
        
        Args:
            query: Search query
            per_page: Results per page (3-200)
            page: Page number
            min_width: Minimum video width
            min_height: Minimum video height
        
        Returns:
            Dictionary containing search results
        """
        if not self.enabled:
            return {"hits": [], "total": 0}
        
        params = {
            "key": self.api_key,
            "q": query,
            "per_page": min(per_page, 200),
            "page": page,
            "video_type": "all",
            "min_width": min_width,
            "min_height": min_height,
        }
        
        try:
            logger.info(f"🔍 Pixabay search: '{query}' (page {page})")
            
            response = self.session.get(
                self.BASE_URL,
                params=params,
                timeout=15
            )
            
            response.raise_for_status()
            data = response.json()
            
            hits = data.get("hits", [])
            logger.info(f"✅ Found {len(hits)} videos on Pixabay")
            
            return data
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"❌ Pixabay API error: {e.response.status_code}")
            return {"hits": [], "total": 0}
            
        except Exception as e:
            logger.error(f"❌ Pixabay request failed: {e}")
            return {"hits": [], "total": 0}
    
    def get_video_url(self, video: Dict, quality: str = "large") -> Optional[str]:
        """
        Get video URL from Pixabay video object.
        
        Args:
            video: Pixabay video dict
            quality: Video quality (large, medium, small)
        
        Returns:
            Video URL or None
        """
        videos = video.get("videos", {})
        
        # Try requested quality first
        if quality in videos:
            return videos[quality].get("url")
        
        # Fallback to other qualities
        for q in ["large", "medium", "small"]:
            if q in videos:
                return videos[q].get("url")
        
        return None
