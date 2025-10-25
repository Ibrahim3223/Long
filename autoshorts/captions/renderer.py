# -*- coding: utf-8 -*-
"""
Caption rendering - LONG-FORM (16:9)
Sade altyazı: vurgu YOK, animasyon YOK (yukarı kayma yok)
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
    WORDS_PER_CHUNK = 3
    MIN_WORD_DURATION = 0.08
    TIMING_PRECISION = 0.001
    FADE_DURATION = 0.0

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
        ass_path = video_path.replace(".mp4", ".ass")
        try:
            if duration <= 0:
                duration = ffprobe_duration(video_path) or 0.0
            if duration <= 0:
                logger.warning("         Unknown duration; skipping captions.")
                return video_path

            if not words:
                words = self._fallback_timings(text, duration)
            words = self._normalize_timings(words, duration)
            if not words:
                return video_path

            style_name = get_random_style()
            ass_content = build_karaoke_ass(
                text=text or "",
                seg_dur=duration,
                words=words,
                is_hook=(sentence_type == "hook"),
                style_name=style_name,
                emphasize=False,      # 🔒 kapalı
                fancy_motion=False,   # 🔒 kapalı
            )
            with open(ass_path, "w", encoding="utf-8") as f:
                f.write(ass_content)

            frames = max(1, int(round(duration * settings.TARGET_FPS)))
            ass_arg = pathlib.Path(ass_path).as_posix().replace("'", r"\'")
            vf = (
                f"subtitles='{ass_arg}':force_style='Kerning=1',"
                f"setsar=1,fps={settings.TARGET_FPS},"
                f"trim=start_frame=0:end_frame={frames},setpts=PTS-STARTPTS"
            )
            output = video_path.replace(".mp4", "_caption.mp4")

            run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", video_path,
                "-vf", vf,
                "-r", str(settings.TARGET_FPS), "-vsync", "cfr",
                "-c:v", "libx264", "-preset", "medium",
                "-crf", str(settings.CRF_VISUAL),
                "-pix_fmt", "yuv420p",
                "-an",
                output
            ])

            pathlib.Path(ass_path).unlink(missing_ok=True)
            return output if os.path.exists(output) else video_path

        except Exception as e:
            logger.error(f"         ❌ Caption error: {e}")
            import traceback; logger.debug(traceback.format_exc())
            pathlib.Path(ass_path).unlink(missing_ok=True)
            return video_path

    # ---- helpers ----
    def _fallback_timings(self, text: str, duration: float) -> List[Tuple[str, float]]:
        ws = [w for w in re.split(r"\s+", (text or "").strip()) if w]
        if not ws:
            return []
        per = max(self.MIN_WORD_DURATION, duration / len(ws))
        times = [(w, per) for w in ws]
        total = sum(d for _, d in times)
        diff = duration - total
        if abs(diff) > 1e-3:
            w, d = times[-1]
            times[-1] = (w, max(self.MIN_WORD_DURATION, d + diff))
        return times

    def _normalize_timings(self, word_timings: List[Tuple[str, float]], total: float) -> List[Tuple[str, float]]:
        if not word_timings:
            return []
        fixed = []
        for w, d in word_timings:
            if not w or not str(w).strip():
                continue
            d = max(self.MIN_WORD_DURATION, min(float(d), 5.0))
            fixed.append((str(w).strip(), d))
        if not fixed:
            return []
        s = sum(d for _, d in fixed)
        if abs(s - total) > 1e-3 and s > 0:
            k = total / s
            fixed = [(w, round(d * k, 3)) for w, d in fixed]
            s2 = sum(d for _, d in fixed)
            diff = total - s2
            if abs(diff) > 1e-3:
                w, d = fixed[-1]
                fixed[-1] = (w, max(self.MIN_WORD_DURATION, d + diff))
        return fixed

# Geriye dönük alias
Renderer = CaptionRenderer
