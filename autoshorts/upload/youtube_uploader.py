# FILE: autoshorts/upload/youtube_uploader.py
# -*- coding: utf-8 -*-
"""
YouTube Uploader - LONG-FORM with CHAPTER SUPPORT
Uploads normal videos (not shorts) with automatic chapter timestamps
+ SEO güçlendirme: açıklama lead, chapter outline, hashtag (5000 char korumalı)
"""

import logging
import re
import unicodedata
from typing import Dict, Any, List, Optional
from autoshorts.config import settings

# Güvenli import: text_utils yoksa minimal fallback kullan
try:
    from autoshorts.utils.text_utils import hashtags_from_tags as _hashtags_from_tags
except Exception:
    def _hashtags_from_tags(tags, title, limit=5):
        out: List[str] = []
        seen = set()
        pool = list(tags or [])
        if title:
            # başlıktan basit token üret
            t = re.sub(r"[^a-z0-9 ]+", " ", title.lower())
            pool.extend([w for w in t.split() if len(w) >= 3])
        for tok in pool:
            key = re.sub(r"[^a-z0-9]+", "", str(tok).lower())
            if not key or key in seen:
                continue
            seen.add(key)
            out.append("#" + key[:28])
            if len(out) >= limit:
                break
        return out

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
        audio_durations: Optional[List[float]] = None,    # For timestamp calculation
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
            optimized_tags = self._optimize_tags(tags)
            smart_category = self._detect_category(topic, title, description) if topic else category_id

            optimized_description = self._build_description_with_chapters(
                description=description,
                chapters=chapters,
                audio_durations=audio_durations,
                title=optimized_title,
                tags=optimized_tags,
                topic=topic,
            )

            logger.info(f"[YouTube] Title: {optimized_title}")
            logger.info(f"[YouTube] Category: {smart_category}")
            logger.info(f"[YouTube] Chapters: {len(chapters) if chapters else 0}")
            logger.info(f"[YouTube] Tags: {len(optimized_tags) if optimized_tags else 0}")
            logger.debug(f"[YouTube] Description length: {len(optimized_description)} chars")

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
                    "description": optimized_description[:5000],  # güvenlik için tekrar kes
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

            # Add language only if valid (2-letter ISO code)
            if getattr(settings, "LANG", None):
                lang_code = str(settings.LANG)[:2].lower()
                if lang_code in {
                    "en", "tr", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", "ar", "hi", "nl", "pl", "sv"
                }:
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

    # --------------------------- SEO description --------------------------- #
    def _build_description_with_chapters(
        self,
        description: str,
        chapters: Optional[List[Dict[str, Any]]],
        audio_durations: Optional[List[float]],
        *,
        title: str,
        tags: Optional[List[str]],
        topic: Optional[str],
    ) -> str:
        """
        Add an SEO-friendly lead, chapter timestamps (varsa), ve hashtags.
        - İlk satır: konu/başlık içeren kısa lead (keyword içerir)
        - Chapters: zaman damgalı; süre yoksa başlık listesi
        - Hashtags: 3–5 adet, en sonda
        - 5000 karakter limiti güvenli şekilde sağlanır
        """
        base = (description or "").strip()
        primary_kw = (topic or title or "").strip()

        # 1) SEO lead (ilk satıra anahtar kelime dokundur)
        lead_prefix = ""
        if primary_kw and primary_kw.lower() not in base.lower()[:200]:
            lead_prefix = f"{primary_kw}: "
        lead_line = re.sub(r"\s+", " ", (lead_prefix + base).strip())

        core = lead_line

        # 2) Chapters
        chapter_block = ""
        if chapters:
            if audio_durations:
                chapter_block += "\n\n📑 CHAPTERS:\n"
                current_time = 0.0
                safe_durations = [max(0.0, float(d or 0.0)) for d in (audio_durations or [])]
                for chapter in chapters:
                    timestamp = self._format_timestamp(current_time)
                    chapter_title = (chapter.get("title") or "Chapter").strip()
                    chapter_block += f"{timestamp} {chapter_title}\n"

                    # Bu bölümün süre toplamı
                    start_idx = int(chapter.get("start_sentence", 0) or 0)
                    end_idx = int(chapter.get("end_sentence", 0) or 0)
                    for i in range(start_idx, min(end_idx + 1, len(safe_durations))):
                        current_time += safe_durations[i]
            else:
                # Süreler yoksa zaman damgasız başlıklar
                names = [c.get("title", "").strip() for c in chapters if c.get("title")]
                if names:
                    chapter_block += "\n\n📑 CHAPTERS:\n" + "\n".join(f"- {n}" for n in names)

        core += chapter_block

        # 3) Kısa “what you'll learn” outline (chapter başlıklarından)
        if chapters:
            names = [c.get("title", "").strip() for c in chapters if c.get("title")]
            if names:
                core += "\n\nWhat you’ll learn:\n" + "\n".join(f"• {n}" for n in names[:8])

        # 4) CTA
        core += "\n\n🔔 Subscribe for more in-depth educational content!"
        core += "\n💬 Share your thoughts below."

        # 5) Hashtags (en sonda; 3–5 adet) — önce hesapla, sonra uzunluk kontrolünü yap
        safe_tags = tags or []
        hashtag_line = " ".join(_hashtags_from_tags(safe_tags, title, limit=5))
        hashtag_suffix = ("\n\n" + hashtag_line) if hashtag_line else ""

        # 5000 karakter limitini koru: önce çekirdek metni kısalt, sonra hashtag'i ekle
        MAX_LEN = 5000
        reserve = len(hashtag_suffix)
        if len(core) + reserve > MAX_LEN:
            core = core[: MAX_LEN - reserve].rstrip()

        full_description = (core + hashtag_suffix).strip()
        return full_description

    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds to MM:SS or HH:MM:SS"""
        total_seconds = int(max(0, seconds))
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60

        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes}:{secs:02d}"

    def _optimize_title(self, title: str) -> str:
        """Optimize title for YouTube (60–70 chars hedeflenir)."""
        if not title:
            return "Untitled Video"
        t = title.strip()
        if len(t) > 100:
            t = t[:97] + "..."
        # 60–70 bandına yumuşak yaklaşım (zorunlu değil; kısa ise ufak ek)
        if len(t) < 55:
            t = (t + " | complete guide")[:70]
        return t

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
