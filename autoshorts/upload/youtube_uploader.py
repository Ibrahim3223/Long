# -*- coding: utf-8 -*-
"""
YouTube Uploader - LONG-FORM with CHAPTER SUPPORT
Uploads normal videos (not shorts) with automatic chapter timestamps
"""

import logging
import re
import unicodedata
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
        "pets_animals": "15",
    }

    # Yalnızca harf/rakam/boşluk/altçizgi/tire izin ver
    _SAFE_TAG_RE = re.compile(r"[^A-Za-z0-9 _\-]")

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
        chapters: Optional[List[Dict[str, Any]]] = None,  # Chapter data
        audio_durations: Optional[List[float]] = None,  # For timestamp calculation
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
                scopes=["https://www.googleapis.com/auth/youtube.upload"],
            )
            creds.refresh(Request())

            # Build service
            youtube = build("youtube", "v3", credentials=creds, cache_discovery=False)

            # Build request body
            body: Dict[str, Any] = {
                "snippet": {
                    "title": optimized_title,
                    "description": optimized_description,
                    "categoryId": str(smart_category),
                },
                "status": {
                    "privacyStatus": privacy_status,
                    "selfDeclaredMadeForKids": False,
                },
            }

            # Add tags only if they exist and are not empty
            if optimized_tags:
                body["snippet"]["tags"] = optimized_tags
                logger.info(f"[YouTube] Adding {len(optimized_tags)} tags")
                logger.debug(f"[YouTube] Tags preview: {optimized_tags}")

            # Add language only if valid (2-letter ISO code)
            if hasattr(settings, "LANG") and settings.LANG:
                lang_code = str(settings.LANG)[:2].lower()
                valid_langs = [
                    "en",
                    "tr",
                    "es",
                    "fr",
                    "de",
                    "it",
                    "pt",
                    "ru",
                    "ja",
                    "ko",
                    "zh",
                    "ar",
                    "hi",
                    "nl",
                    "pl",
                    "sv",
                ]
                if lang_code in valid_langs:
                    body["snippet"]["defaultLanguage"] = lang_code
                    body["snippet"]["defaultAudioLanguage"] = lang_code
                    logger.info(f"[YouTube] Language: {lang_code}")

            logger.info("[YouTube] Uploading video...")

            # Upload helper (resumable=False -> invalidTags retry'ı kolay)
            def _insert_video(y, bdy, path):
                media = MediaFileUpload(path, chunksize=-1, resumable=False)
                request = y.videos().insert(part="snippet,status", body=bdy, media_body=media)
                return request.execute()

            # First attempt
            try:
                response = _insert_video(youtube, body, video_path)
            except Exception as e:
                msg = str(e)
                # If tags are invalid, drop tags and retry once
                if "invalidTags" in msg or "video keywords" in msg:
                    logger.warning("[YouTube] invalidTags received, removing tags and retrying…")
                    body["snippet"].pop("tags", None)
                    response = _insert_video(youtube, body, video_path)
                else:
                    raise

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
        audio_durations: Optional[List[float]],
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
                chapter_title = chapter.get("title", "Chapter")
                chapter_text += f"{timestamp} {chapter_title}\n"

                # Add duration of sentences in this chapter
                start_idx = chapter.get("start_sentence", 0)
                end_idx = chapter.get("end_sentence", 0)

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

    # --- Tag helpers ---
    def _sanitize_tag(self, tag: str) -> str:
        """Reduce a single tag to a safe form accepted by YouTube."""
        if not tag:
            return ""
        t = unicodedata.normalize("NFKD", str(tag))
        # Replace characters that often cause issues
        t = t.replace(",", " ").replace("#", " ")
        # Drop anything not in the safe set
        t = self._SAFE_TAG_RE.sub("", t)
        # Collapse whitespace
        t = re.sub(r"\s+", " ", t).strip()
        # Keep individual tags reasonably short (defensive)
        return t[:30]

    def _optimize_tags(self, tags: Optional[List[str]]) -> List[str]:
        """Clean tags, ensure uniqueness, keep ≤30 items and ~≤490 total chars."""
        if not tags:
            return []

        seen = set()
        out: List[str] = []
        total_len = 0

        for raw in tags:
            t = self._sanitize_tag(raw)
            if not t:
                continue
            key = t.lower()
            if key in seen:
                continue
            # Leave headroom under 500 char total limit
            if total_len + len(t) + (1 if out else 0) > 490:
                break
            out.append(t)
            seen.add(key)
            total_len += len(t) + 1

            if len(out) >= 30:
                break

        return out

    def _detect_category(self, topic: str, title: str, description: str) -> str:
        """Smart category detection"""
        text = f"{topic or ''} {title or ''} {description or ''}".lower()

        patterns = {
            "27": ["fact", "learn", "explain", "teach", "science", "history", "educational", "education"],
            "28": ["tech", "ai", "robot", "future", "innovation", "computer", "digital", "technology"],
            "24": ["story", "tale", "movie", "entertainment", "fun"],
            "26": ["how to", "tutorial", "guide", "tips", "diy", "howto"],
            "19": ["travel", "country", "city", "geography", "world", "place"],
            "22": ["life", "daily", "personal", "vlog", "lifestyle"],
        }

        # Count matches
        scores: Dict[str, int] = {}
        for cat_id, keywords in patterns.items():
            score = sum(1 for kw in keywords if kw in text)
            scores[cat_id] = score

        # Return best match
        best_category = max(scores, key=scores.get) if scores else "27"
        return best_category if scores.get(best_category, 0) > 0 else "27"
