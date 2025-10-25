# FILE: autoshorts/upload/youtube_uploader.py
# -*- coding: utf-8 -*-
"""
YouTube Uploader - LONG-FORM with CHAPTER SUPPORT
- SEO güçlendirme
- Gerçek zaman damgalı CHAPTERS (audio_durations yoksa otomatik hesap)
"""
import logging
import re
import unicodedata
from typing import Dict, Any, List, Optional, Tuple

from autoshorts.config import settings
from autoshorts.utils.ffmpeg_utils import ffprobe_duration

logger = logging.getLogger(__name__)

class YouTubeUploader:
    """Long-form YouTube uploader with chapters"""

    CATEGORIES = {
        "education": "27", "people_blogs": "22", "entertainment": "24",
        "howto_style": "26", "science_tech": "28", "news_politics": "25",
        "comedy": "23", "sports": "17", "gaming": "20",
        "travel": "19", "pets_animals": "15",
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
        """Upload long-form video with chapter timestamps"""
        try:
            from google.oauth2.credentials import Credentials
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaFileUpload

            logger.info(f"[YouTube] Uploading: {title[:50]}...")

            # Toplam süreyi ölç (fallback için)
            total_seconds = 0.0
            try:
                total_seconds = float(ffprobe_duration(video_path) or 0.0)
            except Exception:
                total_seconds = 0.0

            optimized_title = self._optimize_title(title)
            optimized_description = self._build_description_with_chapters(
                description, chapters, audio_durations,
                title=optimized_title, tags=tags, topic=topic,
                video_seconds=total_seconds,
            )
            optimized_tags = self._optimize_tags(tags)
            smart_category = self._detect_category(topic, title, description) if topic else category_id

            logger.info(f"[YouTube] Title: {optimized_title}")
            logger.info(f"[YouTube] Category: {smart_category}")
            logger.info(f"[YouTube] Chapters: {len(chapters) if chapters else 0}")
            logger.info(f"[YouTube] Tags: {len(optimized_tags) if optimized_tags else 0}")

            if not optimized_title:
                raise ValueError("Title is empty")
            if len(optimized_description) > 5000:
                optimized_description = optimized_description[:5000]

            creds = Credentials(
                token=None,
                refresh_token=self.refresh_token,
                token_uri="https://oauth2.googleapis.com/token",
                client_id=self.client_id,
                client_secret=self.client_secret,
                scopes=["https://www.googleapis.com/auth/youtube.upload"],
            )
            creds.refresh(Request())
            youtube = build("youtube", "v3", credentials=creds, cache_discovery=False)

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

            if optimized_tags:
                body["snippet"]["tags"] = optimized_tags
                logger.info(f"[YouTube] Adding {len(optimized_tags)} tags")

            if hasattr(settings, "LANG") and settings.LANG:
                lang_code = str(settings.LANG)[:2].lower()
                valid_langs = ["en","tr","es","fr","de","it","pt","ru","ja","ko","zh","ar","hi","nl","pl","sv"]
                if lang_code in valid_langs:
                    body["snippet"]["defaultLanguage"] = lang_code
                    body["snippet"]["defaultAudioLanguage"] = lang_code
                    logger.info(f"[YouTube] Language: {lang_code}")

            logger.info("[YouTube] Uploading video...")

            def _insert_video(y, bdy, path):
                media = MediaFileUpload(path, chunksize=-1, resumable=False)
                request = y.videos().insert(part="snippet,status", body=bdy, media_body=media)
                return request.execute()

            try:
                response = _insert_video(youtube, body, video_path)
            except Exception as e:
                msg = str(e)
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

    # --------------------------- SEO + CHAPTERS --------------------------- #
    def _build_description_with_chapters(
        self,
        description: str,
        chapters: Optional[List[Dict[str, Any]]],
        audio_durations: Optional[List[float]],
        *,
        title: str,
        tags: Optional[List[str]],
        topic: Optional[str],
        video_seconds: float = 0.0,
    ) -> str:
        """
        Açıklamayı SEO giriş + gerçek zaman damgalı chapters + CTA + hashtag şeklinde kurar.
        - audio_durations yoksa: toplam video süresini cümle/chapters dağılımıyla tahminler.
        """
        base = (description or "").strip()
        primary_kw = (topic or title or "").strip()

        # 1) SEO lead (ilk satır)
        lead = f"{primary_kw}: " if primary_kw and primary_kw.lower() not in base.lower()[:200] else ""
        full_description = re.sub(r"\s+", " ", (lead + base)).strip()

        # 2) CHAPTERS (timestamp zorunlu)
        lines = []
        ts_list: List[Tuple[int, str]] = []  # (start_seconds, title)
        if chapters:
            ts_list = self._compute_chapter_timestamps(chapters, audio_durations, video_seconds)
        if ts_list:
            lines.append("")
            lines.append("Chapters:")
            for sec, name in ts_list:
                lines.append(f"{self._format_timestamp(sec)} {name}")
            full_description += "\n" + "\n".join(lines)

        # 3) What you’ll learn (chapter başlıkları)
        if chapters:
            names = [c.get("title", "").strip() for c in chapters if c.get("title")]
            if names:
                full_description += "\n\nWhat you’ll learn:\n" + "\n".join(f"• {n}" for n in names[:8])

        # 4) CTA
        full_description += "\n\n🔔 Subscribe for more in-depth educational content!"
        full_description += "\n💬 Share your thoughts below."

        # 5) Hashtags
        safe_tags = self._optimize_tags(tags) if tags else []
        hashtags = self._hashtags_from_tags(safe_tags, title, limit=5)
        if hashtags:
            full_description += "\n\n" + " ".join(hashtags)

        return full_description.strip()

    def _compute_chapter_timestamps(
        self,
        chapters: List[Dict[str, Any]],
        audio_durations: Optional[List[float]],
        video_seconds: float
    ) -> List[Tuple[int, str]]:
        """
        YouTube'un chapter algılaması için satır satır 'MM:SS Başlık' üretir.
        Öncelik: audio_durations -> cümle indeksleri -> eşit paylaşım.
        Her bölüm en az 10 sn olacak şekilde korunur.
        """
        if not chapters:
            return []

        # 1) Cümle bazlı süre varsa (en doğrusu)
        if audio_durations and len(audio_durations) > 0:
            cur = 0.0
            out: List[Tuple[int, str]] = []
            for ch in chapters:
                title = ch.get("title", "Chapter").strip() or "Chapter"
                out.append((int(cur), title))
                s = int(ch.get("start_sentence", 0))
                e = int(ch.get("end_sentence", -1))
                if e < s:
                    e = s
                for i in range(s, min(e + 1, len(audio_durations))):
                    cur += max(0.0, float(audio_durations[i]))
            # En az 3 chapter ve 0:00 ile başlamalı
            out = self._enforce_chapter_rules(out, total_seconds=int(cur or video_seconds))
            return out

        # 2) Video süresi biliniyorsa, cümle indekslerine göre oranla
        total = int(video_seconds or 0)
        if total <= 0:
            # 3) Son çare: sabit 60 sn kabul edip eşit böl
            total = 60 * max(3, len(chapters))
        # Cümle sayısı tahmini
        max_end = max((int(c.get("end_sentence", -1)) for c in chapters), default=-1)
        total_sentences = max_end + 1 if max_end >= 0 else None

        starts: List[int] = []
        if total_sentences and total_sentences > 0:
            per_sentence = max(1.0, total / total_sentences)
            cur = 0.0
            for ch in chapters:
                starts.append(int(cur))
                s = int(ch.get("start_sentence", 0))
                e = int(ch.get("end_sentence", s))
                count = max(1, (e - s + 1))
                cur += per_sentence * count
        else:
            # Eşit paylaş
            per = total / max(1, len(chapters))
            starts = [int(i * per) for i in range(len(chapters))]

        out = [(starts[i], (chapters[i].get("title") or "Chapter").strip() or "Chapter")
               for i in range(len(chapters))]
        out = self._enforce_chapter_rules(out, total_seconds=total)
        return out

    def _enforce_chapter_rules(self, stamps: List[Tuple[int, str]], *, total_seconds: int) -> List[Tuple[int, str]]:
        """
        - İlk satır 0:00 olmalı
        - En az 3 chapter olmalı
        - Her chapter >= 10 sn
        """
        if not stamps:
            return []
        # 0:00 ile başlat
        if stamps[0][0] != 0:
            stamps[0] = (0, stamps[0][1])

        # En az 3 satır
        if len(stamps) < 3 and total_seconds > 0:
            # Eşit bölüp 3'e tamamla
            thirds = [0, int(total_seconds / 3), int(2 * total_seconds / 3)]
            while len(thirds) > len(stamps):
                stamps.append((thirds[len(stamps)], f"Part {len(stamps)+1}"))
        # Süre kontrolü (min 10 sn). Çok kısa kalanları birleştir.
        MIN_SEC = 10
        fixed: List[Tuple[int, str]] = []
        for i, (ts, name) in enumerate(stamps):
            fixed.append((max(0, int(ts)), name))
        fixed.sort(key=lambda x: x[0])
        pruned: List[Tuple[int, str]] = []
        for i, (ts, name) in enumerate(fixed):
            if i == 0:
                pruned.append((ts, name))
                continue
            prev_ts = pruned[-1][0]
            if ts - prev_ts < MIN_SEC:
                # Öncekiyle birleştir
                continue
            pruned.append((ts, name))
        # Total'e göre son kontrol
        if len(pruned) >= 2 and total_seconds > 0:
            if total_seconds - pruned[-1][0] < MIN_SEC:
                pruned.pop()  # son chapter çok kısaysa at
        if len(pruned) < 3 and total_seconds > 0:
            # tekrar eşit paylaştır
            per = max(MIN_SEC, int(total_seconds / 3))
            pruned = [(0, "Part 1"), (per, "Part 2"), (min(total_seconds - MIN_SEC, 2*per), "Part 3")]
        return pruned

    def _format_timestamp(self, seconds: int) -> str:
        seconds = max(0, int(seconds))
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m}:{s:02d}"

    def _optimize_title(self, title: str) -> str:
        if not title:
            return "Untitled Video"
        t = title.strip()
        if len(t) > 100:
            t = t[:97] + "..."
        if len(t) < 55:
            t = (t + " | complete guide")[:70]
        return t

    # --- Tags / Hashtags ---
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
            if total_len + len(t) + (1 if out else 0) > 490:
                break
            out.append(t)
            seen.add(key)
            total_len += len(t) + 1
            if len(out) >= 30:
                break
        return out

    def _hashtags_from_tags(self, tags: List[str], title: str, limit: int = 5) -> List[str]:
        """
        Basit ve güvenli hashtag üretimi: tag'lerden ve başlıktan.
        """
        hs = []
        for t in tags:
            core = re.sub(r"[^A-Za-z0-9]", "", t)
            if core and len(hs) < limit:
                hs.append("#" + core[:28])
        if len(hs) < limit:
            # Başlıktan 1-2 anahtar
            words = [w for w in re.split(r"[^A-Za-z0-9]+", title) if len(w) >= 3]
            for w in words:
                if len(hs) >= limit:
                    break
                core = re.sub(r"[^A-Za-z0-9]", "", w)
                if core and ("#" + core) not in hs:
                    hs.append("#" + core[:28])
        return hs

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
        scores: Dict[str, int] = {}
        for cat_id, kws in patterns.items():
            scores[cat_id] = sum(1 for kw in kws if kw in text)
        best = max(scores, key=scores.get) if scores else "27"
        return best if scores.get(best, 0) > 0 else "27"
