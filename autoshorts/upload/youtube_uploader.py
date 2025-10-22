"""
YouTube Uploader - LONG-FORM with CHAPTER SUPPORT
Uploads normal videos (not shorts) with automatic chapter timestamps
"""

import logging
import re
from typing import Dict, Any, List, Optional
from autoshorts.config import settings

logger = logging.getLogger(__name__)


class YouTubeUploader:
    """Long-form YouTube uploader with chapters"""
    
    CATEGORIES = {
        "education": "27",
        "people_blogs": "22",
        "entertainment": "24",
        "howto_style": "26",
        "science_tech": "28",
        "news_politics": "25",
        "comedy": "23",
        "sports": "17",
        "gaming": "20",
        "travel": "19",
        "pets_animals": "15"
    }
    
    def __init__(self):
        """Initialize YouTube uploader"""
        if not all([settings.YT_CLIENT_ID, settings.YT_CLIENT_SECRET, settings.YT_REFRESH_TOKEN]):
            raise ValueError("YouTube credentials missing")
        
        self.client_id = settings.YT_CLIENT_ID
        self.client_secret = settings.YT_CLIENT_SECRET
        self.refresh_token = settings.YT_REFRESH_TOKEN
        
        logger.info("[YouTube] Long-form uploader initialized")
    
    def upload(
        self,
        video_path: str,
        title: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        category_id: str = "27",  # Default: Education
        privacy_status: str = "public",
        topic: Optional[str] = None,
        chapters: Optional[List[Dict[str, Any]]] = None,  # NEW: Chapter data
        audio_durations: Optional[List[float]] = None  # NEW: For timestamp calculation
    ) -> str:
        """Upload long-form video with chapter timestamps"""
        
        try:
            from google.oauth2.credentials import Credentials
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaFileUpload
            
            logger.info(f"[YouTube] Uploading: {title[:50]}...")
            
            # Optimize metadata
            optimized_title = self._optimize_title(title)
            optimized_description = self._build_description_with_chapters(
                description, chapters, audio_durations
            )
            optimized_tags = self._optimize_tags(tags)
            smart_category = self._detect_category(topic, title, description) if topic else category_id
            
            logger.info(f"[YouTube] Title: {optimized_title}")
            logger.info(f"[YouTube] Category: {smart_category}")
            logger.info(f"[YouTube] Chapters: {len(chapters) if chapters else 0}")
            logger.info(f"[YouTube] Tags: {len(optimized_tags) if optimized_tags else 0}")
            
            # Validate
            if not optimized_title:
                raise ValueError("Title is empty")
            if len(optimized_description) > 5000:
                optimized_description = optimized_description[:5000]
            
            # Credentials
            creds = Credentials(
                token=None,
                refresh_token=self.refresh_token,
                token_uri="https://oauth2.googleapis.com/token",
                client_id=self.client_id,
                client_secret=self.client_secret,
                scopes=["https://www.googleapis.com/auth/youtube.upload"]
            )
            creds.refresh(Request())
            
            # Build service
            youtube = build("youtube", "v3", credentials=creds, cache_discovery=False)
            
            # ✅ FIXED: Build body with proper validation
            body = {
                "snippet": {
                    "title": optimized_title,
                    "description": optimized_description,
                    "categoryId": smart_category
                },
                "status": {
                    "privacyStatus": privacy_status,
                    "selfDeclaredMadeForKids": False
                }
            }
            
            # Add tags only if they exist and are not empty
            if optimized_tags and len(optimized_tags) > 0:
                body["snippet"]["tags"] = optimized_tags
                logger.info(f"[YouTube] Adding {len(optimized_tags)} tags")
            
            # Add language only if valid (2-letter ISO code)
            if hasattr(settings, 'LANG') and settings.LANG:
                lang_code = str(settings.LANG)[:2].lower()  # Get first 2 chars
                # Valid YouTube language codes
                valid_langs = ['en', 'tr', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh', 'ar', 'hi', 'nl', 'pl', 'sv']
                if lang_code in valid_langs:
                    body["snippet"]["defaultLanguage"] = lang_code
                    body["snippet"]["defaultAudioLanguage"] = lang_code
                    logger.info(f"[YouTube] Language: {lang_code}")
            
            logger.info("[YouTube] Uploading video...")
            
            # Upload
            media = MediaFileUpload(video_path, chunksize=-1, resumable=True)
            request = youtube.videos().insert(
                part="snippet,status",
                body=body,
                media_body=media
            )
            
            response = request.execute()
            video_id = response.get("id", "")
            
            if not video_id:
                raise ValueError("No video ID returned from YouTube")
            
            video_url = f"https://youtube.com/watch?v={video_id}"
            
            logger.info(f"[YouTube] ✅ Uploaded: {video_url}")
            return video_id
            
        except Exception as e:
            logger.error(f"[YouTube] ❌ Upload failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            raise
    
    def _build_description_with_chapters(
        self,
        description: str,
        chapters: Optional[List[Dict[str, Any]]],
        audio_durations: Optional[List[float]]
    ) -> str:
        """Add chapter timestamps to description"""
        
        # Start with original description
        full_description = description if description else ""
        
        # Add chapters if available
        if chapters and audio_durations:
            chapter_text = "\n\n📑 CHAPTERS:\n"
            
            # Calculate timestamps
            current_time = 0.0
            for chapter in chapters:
                timestamp = self._format_timestamp(current_time)
                chapter_title = chapter.get('title', 'Chapter')
                chapter_text += f"{timestamp} {chapter_title}\n"
                
                # Add duration of sentences in this chapter
                start_idx = chapter.get('start_sentence', 0)
                end_idx = chapter.get('end_sentence', 0)
                
                for i in range(start_idx, min(end_idx + 1, len(audio_durations))):
                    if i < len(audio_durations):
                        current_time += audio_durations[i]
            
            full_description += chapter_text
        
        # Add call to action
        full_description += "\n\n🔔 Subscribe for more in-depth educational content!"
        full_description += "\n💬 Drop your thoughts in the comments below."
        
        return full_description
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds to MM:SS or HH:MM:SS"""
        total_seconds = int(seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes}:{secs:02d}"
    
    def _optimize_title(self, title: str) -> str:
        """Optimize title for YouTube"""
        if not title:
            return "Untitled Video"
        
        title = title.strip()
        if len(title) > 100:
            title = title[:97] + "..."
        return title
    
    def _optimize_tags(self, tags: Optional[List[str]]) -> List[str]:
        """Optimize tags (max 500 chars total)"""
        if not tags:
            return []
        
        optimized = []
        total_length = 0
        
        for tag in tags[:30]:  # Max 30 tags
            if not tag:
                continue
            tag = str(tag).strip()
            if len(tag) > 0 and len(tag) + total_length < 500:
                optimized.append(tag)
                total_length += len(tag) + 1  # +1 for comma
        
        return optimized
    
    def _detect_category(self, topic: str, title: str, description: str) -> str:
        """Smart category detection"""
        text = f"{topic or ''} {title or ''} {description or ''}".lower()
        
        patterns = {
            "27": ["fact", "learn", "explain", "teach", "science", "history", "educational", "education"],
            "28": ["tech", "ai", "robot", "future", "innovation", "computer", "digital", "technology"],
            "24": ["story", "tale", "movie", "entertainment", "fun"],
            "26": ["how to", "tutorial", "guide", "tips", "diy", "howto"],
            "19": ["travel", "country", "city", "geography", "world", "place"],
            "22": ["life", "daily", "personal", "vlog", "lifestyle"]
        }
        
        # Count matches
        scores = {}
        for cat_id, keywords in patterns.items():
            score = sum(1 for kw in keywords if kw in text)
            scores[cat_id] = score
        
        # Return best match
        best_category = max(scores, key=scores.get) if scores else "27"
        return best_category if scores.get(best_category, 0) > 0 else "27"
