# FILE: autoshorts/captions/renderer.py
# -*- coding: utf-8 -*-
"""
Caption rendering – optimized + robust
- Colorful karaoke ASS styles (via karaoke_ass)
- Precise duration matching
- Safer path escaping for the subtitles filter
- Respects FAST_MODE for ffmpeg preset
- Optional audio preservation (settings.CAPTIONS_KEEP_AUDIO)
"""
import os
import pathlib
import logging
import re
from typing import List, Tuple, Optional

from autoshorts.config import settings
from autoshorts.utils.ffmpeg_utils import run, has_subtitles, ffprobe_duration
from autoshorts.captions.karaoke_ass import build_karaoke_ass, get_random_style

logger = logging.getLogger(__name__)


class CaptionRenderer:
    """Render captions with resilient timing and colorful styles."""

    WORDS_PER_CHUNK = 3
    MIN_WORD_DURATION = 0.08
    TIMING_PRECISION = 0.001

    def __init__(self, caption_offset: Optional[float] = None) -> None:
        self.language = getattr(settings, "LANG", "en").lower()
        self.caption_offset = caption_offset or 0.0
        self._preset = os.getenv(
            "FFMPEG_PRESET", "fast" if getattr(settings, "FAST_MODE", False) else "medium"
        )
        # Keep source audio only if explicitly requested (orchestrator replaces audio later)
        self._keep_audio = bool(getattr(settings, "CAPTIONS_KEEP_AUDIO", False))
        logger.info(f"      🎯 Caption renderer initialized ({self.language.upper()})")

    # ----------------------------- Public API ----------------------------- #

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
        """Burn colorful karaoke captions into the video and return new path."""
        try:
            if duration <= 0:
                duration = ffprobe_duration(video_path)

            # Generate word timings if missing
            if not words:
                logger.info("         No word timings provided; generating equal split fallback…")
                words = self._generate_fallback_timings(text, duration)

            # Validate/normalize timings to match duration exactly
            words = self._validate_timings(words, duration)
            if not words:
                logger.warning("         No valid word timings; returning original video")
                return video_path

            # Respect global toggle
            if not getattr(settings, "KARAOKE_CAPTIONS", True):
                logger.info("         Captions disabled in settings; returning original video")
                return video_path

            # Ensure subtitles filter is available
            if not has_subtitles():
                logger.warning("         'subtitles' filter is unavailable in ffmpeg build; skipping burn-in")
                return video_path

            # Build colorful ASS
            ass_path = pathlib.Path(video_path).with_suffix(".ass")
            style_name = get_random_style()
            logger.info(f"         🎨 Using caption style: {style_name}")

            ass_content = build_karaoke_ass(
                text=text,
                seg_dur=duration,
                words=words,
                is_hook=(sentence_type == "hook"),
                style_name=style_name,
            )
            ass_path.write_text(ass_content, encoding="utf-8")

            # Compose filter chain (trim by frames for exact length)
            frames = max(1, int(round(duration * settings.TARGET_FPS)))
            sub_arg = self._escape_for_subtitles_filter(str(ass_path))
            subtitle_filter = (
                f"subtitles='{sub_arg}':force_style='Kerning=1',"
                f"setsar=1,fps={settings.TARGET_FPS},"
                f"trim=start_frame=0:end_frame={frames},setpts=PTS-STARTPTS"
            )

            out_path = str(pathlib.Path(video_path).with_name(
                pathlib.Path(video_path).stem + "_caption.mp4"
            ))

            # Build ffmpeg args
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", video_path,
                "-vf", subtitle_filter,
                "-r", str(settings.TARGET_FPS), "-vsync", "cfr",
                "-c:v", "libx264", "-preset", self._preset,
                "-crf", str(settings.CRF_VISUAL),
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
            ]
            if not self._keep_audio:
                cmd += ["-an"]
            else:
                # Keep any source audio (useful outside orchestrator)
                cmd += ["-c:a", "copy"]
            cmd += [out_path]

            run(cmd)

            # Cleanup ASS and return result if created
            exists = os.path.exists(out_path)
            try:
                ass_path.unlink(missing_ok=True)
            finally:
                if exists:
                    logger.info("         ✅ Captions rendered successfully")
                    return out_path
                logger.warning("         ⚠️ Captioned file missing after render; returning original")
                return video_path

        except Exception as e:
            logger.error(f"         ❌ Caption error: {e}")
            try:
                pathlib.Path(video_path).with_suffix(".ass").unlink(missing_ok=True)
            finally:
                return video_path

    # --------------------------- Internal utils --------------------------- #

    def _generate_fallback_timings(self, text: str, duration: float) -> List[Tuple[str, float]]:
        words = [w for w in re.split(r"\s+", (text or "").strip()) if w]
        if not words:
            return []
        per = max(self.MIN_WORD_DURATION, duration / len(words))
        timings = [(w, per) for w in words]
        # Adjust last to match exactly
        cur = sum(d for _, d in timings)
        diff = duration - cur
        if abs(diff) > self.TIMING_PRECISION and timings:
            w, d = timings[-1]
            timings[-1] = (w, max(self.MIN_WORD_DURATION, d + diff))
        return timings

    def _validate_timings(self, word_timings: List[Tuple[str, float]], total: float) -> List[Tuple[str, float]]:
        if not word_timings:
            return []
        cleaned = [(w.strip(), max(self.MIN_WORD_DURATION, float(d))) for w, d in word_timings if w and w.strip()]
        if not cleaned:
            return []
        # Proportional scaling to match exact total
        cur = sum(d for _, d in cleaned)
        if cur <= 0:
            return []
        scale = total / cur
        scaled = [(w, d * scale) for w, d in cleaned]
        # Final precision fix on last word
        cur2 = sum(d for _, d in scaled)
        diff = total - cur2
        if abs(diff) > self.TIMING_PRECISION and scaled:
            w, d = scaled[-1]
            scaled[-1] = (w, max(self.MIN_WORD_DURATION, d + diff))
        # Round gently to milliseconds
        scaled = [(w, round(d, 3)) for w, d in scaled]
        return scaled

    def _escape_for_subtitles_filter(self, path: str) -> str:
        """
        Escape a filesystem path for use inside subtitles='…' filter:
        - Use POSIX slashes
        - Escape ':', ',', "'", and backslashes
        """
        p = pathlib.Path(path).as_posix()
        # Escape single quotes first for filter arg in single quotes
        p = p.replace("'", r"\'")
        # Escape characters that libass path parser is picky about
        for ch, rep in [(":", r"\:"), (",", r"\,"), ("[", r"\["), ("]", r"\]")]:
            p = p.replace(ch, rep)
        return p
