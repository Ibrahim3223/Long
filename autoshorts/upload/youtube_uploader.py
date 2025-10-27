# -*- coding: utf-8 -*-
"""
YouTube Uploader - LONG-FORM with CHAPTER & THUMBNAIL SUPPORT - FIXED VERSION
✅ Timestamp hesaplama düzeltildi
✅ Başlık kesme limiti düzeltildi (60-70 karakter)
✅ Indent hataları düzeltildi
✅ THUMBNAIL UPLOAD SUPPORT ADDED
"""

import logging
import re
import os
import unicodedata
from typing import Dict, Any, List, Optional, Tuple
from autoshorts.config import settings
from autoshorts.utils.text_utils import hashtags_from_tags

logger = logging.getLogger(__name__)


class YouTubeUploader:
    CATEGORIES = {
        "education": "27", "people_blogs": "22", "entertainment": "24",
        "howto_style": "26", "science_tech": "28", "news_politics": "25",
        "comedy": "23", "sports": "17", "gaming": "20", "travel": "19", "pets_animals": "15",
    }
    _SAFE_TAG_RE = re.compile(r"[^A-Za-z0-9 _\-]")

    def __init__(self):
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
        category_id: str = "27",
        privacy_status: str = "public",
        topic: Optional[str] = None,
        chapters: Optional[List[Dict[str, Any]]] = None,
        audio_durations: Optional[List[float]] = None,
        thumbnail_path: Optional[str] = None,  # ✅ YENİ PARAMETRE
    ) -> str:
        try:
            from google.oauth2.credentials import Credentials
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaFileUpload

            logger.info(f"[YouTube] Uploading: {title[:50]}...")

            optimized_title = self._optimize_title(title)
            chapter_block = self._chapter_block_at_top(chapters, audio_durations)
            tail_description = self._seo_tail(description, optimized_title, tags, topic)
            optimized_description = (chapter_block + "\n\n" + tail_description).strip()

            optimized_tags = self._optimize_tags(tags)
            smart_category = self._detect_category(topic, title, description) if topic else category_id

            if not optimized_title:
                raise ValueError("Title is empty")
            if len(optimized_description) > 5000:
                optimized_description = optimized_description[:5000]

            creds = Credentials(
                token=None, refresh_token=self.refresh_token,
                token_uri="https://oauth2.googleapis.com/token",
                client_id=self.client_id, client_secret=self.client_secret,
                scopes=["https://www.googleapis.com/auth/youtube.upload"],
            )
            creds.refresh(Request())
            youtube = build("youtube", "v3", credentials=creds, cache_discovery=False)

            body: Dict[str, Any] = {
                "snippet": {"title": optimized_title, "description": optimized_description, "categoryId": str(smart_category)},
                "status": {"privacyStatus": privacy_status, "selfDeclaredMadeForKids": False},
            }
            if optimized_tags:
                body["snippet"]["tags"] = optimized_tags

            if getattr(settings, "LANG", None):
                lang_code = str(settings.LANG)[:2].lower()
                if lang_code in {"en","tr","es","fr","de","it","pt","ru","ja","ko","zh","ar","hi","nl","pl","sv"}:
                    body["snippet"]["defaultLanguage"] = lang_code
                    body["snippet"]["defaultAudioLanguage"] = lang_code

            def _insert_video(y, bdy, path):
                media = MediaFileUpload(path, chunksize=-1, resumable=False)
                return y.videos().insert(part="snippet,status", body=bdy, media_body=media).execute()

            try:
                response = _insert_video(youtube, body, video_path)
            except Exception as e:
                msg = str(e)
                if "invalidTags" in msg or "video keywords" in msg:
                    logger.warning("[YouTube] invalidTags; retrying without tags…")
                    body["snippet"].pop("tags", None)
                    response = _insert_video(youtube, body, video_path)
                else:
                    raise

            vid = response.get("id", "")
            if not vid:
                raise ValueError("No video ID returned from YouTube")
            
            # ✅ Upload thumbnail if provided
            if thumbnail_path and os.path.exists(thumbnail_path):
                try:
                    logger.info(f"[YouTube] Uploading thumbnail...")
                    
                    thumbnail_media = MediaFileUpload(
                        thumbnail_path,
                        mimetype="image/jpeg",
                        resumable=False
                    )
                    
                    youtube.thumbnails().set(
                        videoId=vid,
                        media_body=thumbnail_media
                    ).execute()
                    
                    logger.info(f"[YouTube] ✅ Thumbnail uploaded successfully")
                except Exception as thumb_err:
                    logger.warning(f"[YouTube] ⚠️ Thumbnail upload failed: {thumb_err}")
                    # Don't fail the whole upload if thumbnail fails
            
            logger.info(f"[YouTube] ✅ Uploaded: https://youtube.com/watch?v={vid}")
            return vid

        except Exception as e:
            logger.error(f"[YouTube] ❌ Upload failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            raise

    def _chapter_block_at_top(
        self,
        chapters: Optional[List[Dict[str, Any]]],
        audio_durations: Optional[List[float]],
    ) -> str:
        """
        ✅ FIXED: YouTube manual chapters
        - Chapter başlıkları KISA (max 50 karakter)
        - Timestamp hesaplama düzeltildi
        """
        lines: List[str] = []

        # Eğer veri yoksa boş dön
        if not chapters or not audio_durations:
            logger.warning("[Chapters] No data - skipping chapters")
            return ""

        # ✅ Bölüm başlangıç zamanlarını hesapla - CUMULATIVE SUM
        starts: List[Tuple[float, str]] = []
        
        for ch in chapters:
            # ✅ Sadece TITLE kullan, description KULLANMA
            title_raw = ch.get("title", "Chapter").strip()
            
            # ✅ Başlık çok uzunsa kısalt (max 50 karakter)
            if len(title_raw) > 50:
                # Tire veya iki nokta varsa orada kes
                if ":" in title_raw:
                    title = title_raw.split(":")[0].strip()
                elif "–" in title_raw or "—" in title_raw:
                    title = title_raw.split("–")[0].split("—")[0].strip()
                else:
                    # Son kelimede kes
                    title = title_raw[:47] + "..."
            else:
                title = title_raw
        
            # Chapter indeksleri
            s_idx = int(ch.get("start_sentence", 0))
            
            # ✅ DOĞRU MANTIK: Bu chapter'ın başlangıç zamanı = 
            #    başlangıcından ÖNCEKİ tüm sentence'ların toplam süresi
            #    (kendi sentence'larını DAHIL ETME!)
            chapter_start_time = 0.0
            if s_idx > 0 and s_idx <= len(audio_durations):
                # İlk s_idx sentence'ın toplam süresi
                chapter_start_time = sum(float(audio_durations[i]) for i in range(min(s_idx, len(audio_durations))))
            
            starts.append((chapter_start_time, title))
        
        # Toplam video süresini hesapla (validation için)
        total_duration = sum(float(d) for d in audio_durations) if audio_durations else 0.0
        logger.info(f"[Chapters] Total video duration: {total_duration:.1f}s ({self._fmt_ts(total_duration)})")
        
        # ✅ CRITICAL: Timestamp'leri video süresinden uzun olanları temizle
        starts = [(t, title) for t, title in starts if t < total_duration]
        logger.info(f"[Chapters] Generated {len(starts)} chapter timestamps")

        # ✅ Fallback: En az 3 bölüm kontrolü
        if len(starts) < 3:
            logger.warning(f"[Chapters] Only {len(starts)} chapters, need 3+ for YouTube")
            
            # Gerçek video süresini kullan
            if not total_duration or total_duration < 30:
                logger.warning(f"[Chapters] Video too short ({total_duration}s), skipping chapters")
                return ""
        
            # ✅ AKILLI fallback: Giriş/Orta/Sonuç
            starts = [
                (0.0, "Introduction"),
                (total_duration * 0.4, "Main Content"),
                (total_duration * 0.8, "Conclusion")
            ]

        # ✅ Süre ≥10s şartını kontrol et
        merged: List[Tuple[float, str]] = []
        for i, (t, title) in enumerate(starts):
            # Sonraki chapter'ın başlangıcı veya video sonu
            end_t = starts[i + 1][0] if i + 1 < len(starts) else total_duration
            dur = max(0.0, end_t - t)
        
            # İlk chapter hariç, <10s olan chapter'ları önceki ile birleştir
            if i > 0 and dur < 10.0 and merged:
                # Önceki chapter'a dahil et (başlığı güncelleme)
                continue
        
            merged.append((t, title))

        # İlk satır 00:00 garanti
        if not merged or merged[0][0] != 0.0:
            if merged:
                merged[0] = (0.0, merged[0][1])
            else:
                merged = [(0.0, "Introduction")]

        # Minimum 3 satır kontrolü (son çare)
        while len(merged) < 3 and total_duration > 30:
            step = total_duration / (len(merged) + 1)
            merged.append((step * len(merged), f"Part {len(merged)}"))
            merged.sort(key=lambda x: x[0])

        # Final kontrol: Hala 3'ten az ve çok kısa video
        if len(merged) < 3:
            logger.warning(f"[Chapters] Could not create 3 chapters, skipping")
            return ""
        
        # ✅ FINAL VALIDATION: Video süresini aşan timestamp'leri temizle
        merged = [(t, title) for t, title in merged if t < total_duration]
        
        # Son chapter'ı video sonuna çok yakınsa çıkar (son 5 saniye)
        if merged and len(merged) > 3:
            last_time = merged[-1][0]
            if total_duration - last_time < 5.0:
                merged = merged[:-1]
                logger.info(f"[Chapters] Removed last chapter (too close to end)")

        # YouTube formatı: 00:00 Title
        for t, title in merged:
            lines.append(f"{self._fmt_ts(t)} {title}")

        logger.info(f"[Chapters] Created {len(lines)} chapters")
        for line in lines:
            logger.info(f"[Chapters]   {line}")

        return "\n".join(lines)

    def _fmt_ts(self, seconds: float) -> str:
        """
        YouTube timestamp format:
        - Always MM:SS minimum
        - If hours: H:MM:SS (no leading zero on hour)
        """
        total = int(max(0, round(seconds)))
        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60

        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        else:
            return f"{m:02d}:{s:02d}"

    def _seo_tail(self, description: str, title: str, tags: Optional[List[str]], topic: Optional[str]) -> str:
        """SEO tail (chapters sonrası)"""
        base = (description or "").strip()
        primary_kw = (topic or title or "").strip()

        # kısa lead (chapters'tan sonra)
        lead = base
        if primary_kw and primary_kw.lower() not in lead.lower()[:200]:
            lead = f"{primary_kw}: {lead}".strip()

        # küçük outline yoksa ekleme (isteğe bağlı)
        tail = lead

        # CTA
        tail += "\n\n🔔 Subscribe for more in-depth educational content!\n💬 Share your thoughts below."
        
        # Hashtags (3–5)
        safe_tags = self._optimize_tags(tags) if tags else []
        hashtags = hashtags_from_tags(safe_tags, title, limit=5)
        if hashtags:
            tail += "\n\n" + " ".join(hashtags)
        return tail.strip()

    def _optimize_title(self, title: str) -> str:
        """
        ✅ FIXED: Optimize title for YouTube
        - Gemini prompt: "50-60 chars - MOBİL UYUMLU! MAX 60 characters"
        - Eski limit: 100 karakter (çok uzun!)
        - Yeni limit: 70 karakter (YouTube recommended)
        """
        title = (title or "").strip()
        if not title:
            return "Untitled Video"
        
        # ✅ Normalize Unicode
        title = unicodedata.normalize('NFKC', title)
        
        # ✅ 70 karakterden uzunsa akıllıca kısalt
        if len(title) > 70:
            # Tire veya iki nokta varsa, ilk kısmı al
            if ':' in title and title.index(':') < 50:
                title = title.split(':')[0].strip()
            elif '–' in title or '—' in title:
                parts = title.replace('—', '–').split('–')
                title = parts[0].strip()
            
            # Hala uzunsa, son kelimeyi yarıda kesmeden kes
            if len(title) > 70:
                cutoff = title[:67].rfind(' ')
                if cutoff > 50:  # Makul bir nokta
                    title = title[:cutoff] + "..."
                else:
                    title = title[:67] + "..."
        
        return title

    def _optimize_tags(self, tags: Optional[List[str]]) -> List[str]:
        """Optimize and sanitize tags"""
        if not tags:
            return []
        
        clean = []
        for t in tags:
            if not t or not isinstance(t, str):
                continue
            
            t = t.strip().lower()
            t = self._SAFE_TAG_RE.sub("", t)
            t = " ".join(t.split())
            
            if not t or len(t) < 2 or len(t) > 30:
                continue
            if t in {"video", "youtube", "watch", "subscribe", "like"}:
                continue
            
            if t not in clean:
                clean.append(t)
        
        return clean[:30]

    def _detect_category(self, topic: str, title: str, description: str) -> str:
        """Detect YouTube category from content"""
        text = f"{topic} {title} {description}".lower()
        
        # Education indicators
        if any(w in text for w in ["history", "science", "explain", "learn", "tutorial", "guide", "educational"]):
            return self.CATEGORIES["education"]
        
        # Entertainment indicators
        if any(w in text for w in ["fun", "funny", "entertainment", "story", "stories"]):
            return self.CATEGORIES["entertainment"]
        
        # How-to indicators
        if any(w in text for w in ["how to", "diy", "make", "create", "build"]):
            return self.CATEGORIES["howto_style"]
        
        # Science & Tech indicators
        if any(w in text for w in ["technology", "tech", "innovation", "invention", "scientific"]):
            return self.CATEGORIES["science_tech"]
        
        # Default to Education
        return self.CATEGORIES["education"]
