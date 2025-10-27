# -*- coding: utf-8 -*-
"""
YouTube Uploader - LONG-FORM with CHAPTER SUPPORT - FIXED VERSION
✅ Timestamp hesaplama düzeltildi
✅ Başlık kesme limiti düzeltildi (60-70 karakter)
"""

import logging
import re
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
            logger.info(f"[YouTube] ✅ Uploaded: https://youtube.com/watch?v={vid}")
            return vid

        except Exception as e:
            logger.error(f"[YouTube] ❌ Upload failed: {e}")
            import traceback; logger.debug(traceback.format_exc())
            raise

    # -------------------- Chapters builder (top of description) -------------------- #
# -*- coding: utf-8 -*-
"""
YouTube Uploader - FIXED VERSION V2
✅ Chapter başlıkları KISA (sadece title, description YOK)
✅ Timestamp hesaplama DÜZELTİLDİ
✅ Fallback sistem daha akıllı
"""

import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


    def _chapter_block_at_top_fixed(
        chapters: Optional[List[Dict[str, Any]]],
        audio_durations: Optional[List[float]],
    ) -> str:
        """
        ✅ FIXED: YouTube manual chapters
        - Chapter başlıkları KISA (max 50 karakter)
        - Timestamp hesaplama düzeltildi
        - Fallback daha akıllı
        """
        lines: List[str] = []
    
        # Eğer veri yoksa boş dön
        if not chapters or not audio_durations:
            logger.warning("[Chapters] No data - skipping chapters")
            return ""
    
        # ✅ Bölüm başlangıç zamanlarını hesapla
        starts: List[Tuple[float, str]] = []
        cur_time = 0.0
        
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
            e_idx = int(ch.get("end_sentence", s_idx))
            
            # ✅ FIX: Chapter başlangıcı MEVCUT zamanda kaydet
            starts.append((cur_time, title))
        
            # ✅ Şimdi bu bölümün toplam süresini ekle (SONRAKİ chapter için)
            for i in range(s_idx, min(e_idx + 1, len(audio_durations))):
                if i < len(audio_durations):
                    cur_time += float(audio_durations[i])
    
        # ✅ Fallback: En az 3 bölüm kontrolü
        if len(starts) < 3:
            logger.warning(f"[Chapters] Only {len(starts)} chapters, need 3+ for YouTube")
            
            # ✅ Gerçek video süresini kullan (fallback değil!)
            if audio_durations:
                total = sum(audio_durations)
            else:
                total = cur_time if cur_time > 0 else 180.0  # Son çare fallback
        
            if total < 30:
                logger.warning(f"[Chapters] Video too short ({total}s), skipping chapters")
                return ""
        
            # ✅ AKILLI fallback: Giriş/Orta/Sonuç
            starts = [
                (0.0, "Introduction"),
                (total * 0.4, "Main Content"),
                (total * 0.8, "Conclusion")
            ]
    
        # ✅ Süre ≥10s şartını kontrol et
        merged: List[Tuple[float, str]] = []
        for i, (t, title) in enumerate(starts):
            end_t = starts[i + 1][0] if i + 1 < len(starts) else cur_time
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
        while len(merged) < 3 and cur_time > 30:
            total = cur_time if cur_time > 0 else 180.0
            step = total / (len(merged) + 1)
            merged.append((step * len(merged), f"Part {len(merged)}"))
            merged.sort(key=lambda x: x[0])
    
    # Final kontrol: Hala 3'ten az ve çok kısa video
        if len(merged) < 3:
            logger.warning(f"[Chapters] Could not create 3 chapters, skipping")
            return ""
    
    # YouTube formatı: 00:00 Title
        for t, title in merged:
            lines.append(f"{_fmt_ts(t)} {title}")
    
        logger.info(f"[Chapters] Created {len(lines)} chapters")
        for line in lines:
            logger.info(f"[Chapters]   {line}")
    
        return "\n".join(lines)


    def _fmt_ts(seconds: float) -> str:
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
    
    # -------------------- SEO tail (chapters sonrası) -------------------- #
    def _seo_tail(self, description: str, title: str, tags: Optional[List[str]], topic: Optional[str]) -> str:
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

    # -------------------- helpers -------------------- #
    def _optimize_title(self, title: str) -> str:
        """
        ✅ FIXED: Optimize title for YouTube
        - Gemini prompt: "50-60 chars - MOBİL UYUMLU! MAX 60 characters"
        - Eski limit: 100 karakter (çok uzun!)
        - Yeni limit: 70 karakter (Gemini'nin 60-70 önerisine uygun)
        """
        if not title:
            return "Untitled Video"
        
        t = title.strip()
        
        # ✅ Önce çok uzun başlıkları kes (70 karaktere kadar)
        if len(t) > 70:
            # Son kelimeyi yarım bırakma, tam kelimede kes
            cutoff = t[:67].rfind(' ')
            if cutoff > 50:  # En az 50 karakter kalsın
                t = t[:cutoff] + "..."
            else:
                t = t[:67] + "..."
        
        # Çok kısa başlıkları uzat (opsiyonel)
        elif len(t) < 50:
            t = (t + " | Complete Guide")[:70]
        
        return t

    def _sanitize_tag(self, tag: str) -> str:
        if not tag:
            return ""
        t = unicodedata.normalize("NFKD", str(tag))
        t = t.replace(",", " ").replace("#", " ")
        t = self._SAFE_TAG_RE.sub("", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t[:30]

    def _optimize_tags(self, tags: Optional[List[str]]) -> List[str]:
        if not tags:
            return []
        seen, out, total_len = set(), [], 0
        for raw in tags:
            t = self._sanitize_tag(raw)
            if not t:
                continue
            k = t.lower()
            if k in seen:
                continue
            if total_len + len(t) + (1 if out else 0) > 490:
                break
            out.append(t)
            seen.add(k)
            total_len += len(t) + 1
            if len(out) >= 30:
                break
        return out

    def _detect_category(self, topic: str, title: str, description: str) -> str:
        text = f"{topic or ''} {title or ''} {description or ''}".lower()
        patterns = {
            "27": ["fact","learn","explain","teach","science","history","educational","education"],
            "28": ["tech","ai","robot","future","innovation","computer","digital","technology"],
            "24": ["story","tale","movie","entertainment","fun"],
            "26": ["how to","tutorial","guide","tips","diy","howto"],
            "19": ["travel","country","city","geography","world","place"],
            "22": ["life","daily","personal","vlog","lifestyle"],
        }
        scores: Dict[str,int] = {cid: sum(1 for kw in kws if kw in text) for cid, kws in patterns.items()}
        best = max(scores, key=scores.get) if scores else "27"
        return best if scores.get(best, 0) > 0 else "27"
