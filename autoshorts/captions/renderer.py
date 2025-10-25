# -*- coding: utf-8 -*-
"""
Caption rendering - LONG-FORM (16:9)
- Varsayılan: vurgu KAPALI, hareket/animasyon KAPALI (yukarı kayma yok)
- Altyazılar sabit konumda, okunaklı ve sade
"""

import os
import re
import pathlib
import logging
from typing import List, Tuple, Optional

from autoshorts.config import settings
from autoshorts.utils.ffmpeg_utils import run, ffprobe_duration
from autoshorts.captions.karaoke_ass import build_karaoke_ass, get_random_style

logger = logging.getLogger(__name__)


class CaptionRenderer:
    """Render captions using colorful karaoke ASS (motion/emphasis off by default)."""

    WORDS_PER_CHUNK = 3
    MIN_WORD_DURATION = 0.08
    TIMING_PRECISION = 0.001
    FADE_DURATION = 0.0  # görsel fade kullanmıyoruz

    def __init__(self, caption_offset: Optional[float] = None):
        self.language = getattr(settings, "LANG", "en").lower()
        self.caption_offset = caption_offset or 0.0
        logger.info(f"      🎯 Caption renderer initialized ({self.language.upper()})")

    def render(
        self,
        video_path: str,
        text: str,
        words: List[Tuple[str, float]],
        duration: float,
        is_hook: bool = False,
        sentence_type: str = "buildup",
        temp_dir: Optional[str] = None,
    ) -> str:
        """
        Verilen video segmenti üzerine ASS altyazı bind eder.
        - Varsayılan ayarlar: vurgu (emphasis) = False, hareket (motion) = False
        """
        ass_path = video_path.replace(".mp4", ".ass")
        try:
            if duration <= 0:
                duration = ffprobe_duration(video_path) or 0.0

            if duration <= 0:
                logger.warning("         Unknown duration; skipping captions.")
                return video_path

            # Kelime zamanlamaları yoksa, eşit dağıt
            if not words:
                logger.info("         No word timings, generating fallback…")
                words = self._generate_fallback_timings(text, duration)

            # Sağlamlaştırma ve toplam süreye normalize
            words = self._aggressive_validate(words, duration)
            if not words:
                logger.warning("         No valid words after validation; skipping captions.")
                return video_path

            # ASS içeriği üret
            style_name = get_random_style()
            logger.info(f"         🎨 Using caption style: {style_name}")

            ass_content = build_karaoke_ass(
                text=text or "",
                seg_dur=duration,
                words=words,
                is_hook=(sentence_type == "hook"),
                style_name=style_name,
                # Varsayılanlar kapalı; settings ile açılabilir
                emphasize=getattr(settings, "KARAOKE_EMPHASIS", False),
                fancy_motion=getattr(settings, "KARAOKE_MOTION", False),
            )

            with open(ass_path, "w", encoding="utf-8") as f:
                f.write(ass_content)

            if not os.path.exists(ass_path):
                logger.error("         ❌ ASS file not created")
                return video_path

            logger.info(f"         ✅ ASS file created: {os.path.getsize(ass_path)} bytes")

            # FFmpeg ile bind et
            frames = max(1, int(round(duration * settings.TARGET_FPS)))
            ass_arg = pathlib.Path(ass_path).as_posix().replace("'", r"\'")

            subtitle_filter = (
                f"subtitles='{ass_arg}':force_style='Kerning=1',"
                f"setsar=1,fps={settings.TARGET_FPS},"
                f"trim=start_frame=0:end_frame={frames},setpts=PTS-STARTPTS"
            )

            output = video_path.replace(".mp4", "_caption.mp4")

            run(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    video_path,
                    "-vf",
                    subtitle_filter,
                    "-r",
                    str(settings.TARGET_FPS),
                    "-vsync",
                    "cfr",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    str(settings.CRF_VISUAL),
                    "-pix_fmt",
                    "yuv420p",
                    # Ses akışı pipeline’ına göre:
                    # Eğer bu aşamada sesi preserve etmek istiyorsanız aşağıdaki 3 satırı açıp `-an`'ı kaldırın.
                    # "-map", "0:v:0", "-map", "0:a?", "-c:a", "copy",
                    "-an",
                    output,
                ]
            )

            pathlib.Path(ass_path).unlink(missing_ok=True)

            if os.path.exists(output):
                logger.info("         ✅ Captions rendered successfully (no motion/emphasis)!")
                return output

            # Başarısızsa temizle ve orijinal segmenti döndür
            pathlib.Path(output).unlink(missing_ok=True)
            return video_path

        except Exception as e:
            logger.error(f"         ❌ Caption error: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            pathlib.Path(ass_path).unlink(missing_ok=True)
            return video_path

    # --------------------- helpers --------------------- #
    def _generate_fallback_timings(self, text: str, duration: float) -> List[Tuple[str, float]]:
        words = [w for w in re.split(r"\s+", (text or "").strip()) if w]
        if not words:
            return []
        per_word = max(self.MIN_WORD_DURATION, duration / len(words))
        timings = [(w, per_word) for w in words]
        if timings:
            current_sum = sum(d for _, d in timings)
            diff = duration - current_sum
            if abs(diff) > 0.001:
                last_word, last_dur = timings[-1]
                timings[-1] = (last_word, max(self.MIN_WORD_DURATION, last_dur + diff))
        return timings

    def _aggressive_validate(
        self, word_timings: List[Tuple[str, float]], total_duration: float
    ) -> List[Tuple[str, float]]:
        if not word_timings:
            return []
        word_timings = [(w.strip(), d) for w, d in word_timings if w and w.strip()]
        if not word_timings:
            return []

        fixed: List[Tuple[str, float]] = []
        for word, dur in word_timings:
            dur = max(self.MIN_WORD_DURATION, min(float(dur), 5.0))
            fixed.append((word, dur))

        current_total = sum(d for _, d in fixed)
        if abs(current_total - total_duration) > 0.001:
            scale = (total_duration / current_total) if current_total > 0 else 1.0
            fixed = [(w, max(self.MIN_WORD_DURATION, round(d * scale, 3))) for w, d in fixed]
            new_total = sum(d for _, d in fixed)
            diff = total_duration - new_total
            if abs(diff) > 0.001 and fixed:
                lw, ld = fixed[-1]
                fixed[-1] = (lw, max(self.MIN_WORD_DURATION, ld + diff))
        return fixed


# Geriye dönük uyumluluk: Bazı projelerde Renderer sınıf adı bekleniyor olabilir.
Renderer = CaptionRenderer
