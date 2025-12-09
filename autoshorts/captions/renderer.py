# -*- coding: utf-8 -*-
"""
Caption rendering - LONG-FORM (16:9)
✅ BÜYÜK HARF altyazılar
✅ İYİLEŞTİRİLMİŞ senkronizasyon
✅ CHUNK CHUNK altyazı (cümle cümle görünüm)
✅ Keyword highlighting
"""

import os
import re
import pathlib
import logging
from typing import List, Tuple, Optional, Dict, Any

from autoshorts.config import settings
from autoshorts.utils.ffmpeg_utils import run, ffprobe_duration
from autoshorts.captions.karaoke_ass import CAPTION_STYLES, get_random_style

# Keyword highlighter (engagement boost)
try:
    from autoshorts.captions.keyword_highlighter import ShortsKeywordHighlighter
    HIGHLIGHTER_AVAILABLE = True
except ImportError:
    HIGHLIGHTER_AVAILABLE = False
    ShortsKeywordHighlighter = None

logger = logging.getLogger(__name__)


class CaptionRenderer:
    WORDS_PER_CHUNK = 3  # 2-3 kelimelik chunks (cümle cümle görünüm)
    MIN_WORD_DURATION = 0.06  # ✅ 60ms minimum (sync için)
    MAX_WORD_DURATION = 4.0   # ✅ Maksimum sınır
    TIMING_PRECISION = 0.001  # ✅ Milisaniye hassasiyeti
    FADE_DURATION = 0.0       # ❌ Fade YOK (sync için critical)

    def __init__(self, caption_offset: Optional[float] = None):
        self.language = getattr(settings, "LANG", "en").lower()
        self.caption_offset = caption_offset or 0.0

        # Keyword highlighter (engagement boost)
        self.highlighter = None
        if HIGHLIGHTER_AVAILABLE and ShortsKeywordHighlighter:
            try:
                self.highlighter = ShortsKeywordHighlighter()
            except Exception:
                pass

        logger.info(f"      🎯 Caption renderer initialized ({self.language.upper()}) - CHUNK-BASED sync")

    def render(
        self,
        video_path: str,
        text: str,
        words: List[Tuple[str, float]],
        duration: float,
        is_hook: bool = False,
        sentence_type: str = "buildup",
        temp_dir: Optional[str] = None,
        caption_offset: float = 0.0,
    ) -> str:
        """
        Render captions with CHUNK-BASED timing (cümle cümle görünüm).

        ✅ 2-3 kelimelik chunks
        ✅ Her chunk kendi timing'i ile
        ✅ Aggressive validation (drift prevention)
        """
        # Use instance offset if not explicitly provided
        if caption_offset == 0.0:
            caption_offset = self.caption_offset

        ass_path = video_path.replace(".mp4", ".ass")
        try:
            # ✅ Duration doğrulama
            if duration <= 0:
                duration = ffprobe_duration(video_path) or 0.0
            if duration <= 0:
                logger.warning("         Unknown duration; skipping captions.")
                return video_path

            # ✅ METNİ BÜYÜK HARFE ÇEVİR
            text = (text or "").strip().upper()

            # ✅ Word timing normalleştir
            if not words:
                words = self._fallback_timings(text, duration)
            words = self._normalize_timings(words, duration)
            if not words:
                logger.warning("         No valid word timings; skipping captions.")
                return video_path

            # ✅ AGGRESSIVE validation (drift prevention)
            words = self._aggressive_validate(words, duration)

            # ✅ ASS dosyası oluştur - CHUNK-BASED (cümle cümle görünüm)
            try:
                self._write_exact_ass(
                    words=words,
                    total_duration=duration,
                    sentence_type=sentence_type,
                    output_path=ass_path,
                    time_offset=caption_offset
                )
            except Exception as e:
                logger.error(f"         ❌ ASS generation failed: {e}")
                return video_path

            if not os.path.exists(ass_path):
                logger.error(f"         ❌ ASS file not created")
                return video_path

            # ✅ FFmpeg render (frame-accurate)
            frames = max(2, int(round(duration * settings.TARGET_FPS)))
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
                "-r", str(settings.TARGET_FPS),
                "-vsync", "cfr",  # ✅ Constant frame rate
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", str(settings.CRF_VISUAL),
                "-pix_fmt", "yuv420p",
                "-an",  # Audio yok (sonra eklenecek)
                output
            ])

            pathlib.Path(ass_path).unlink(missing_ok=True)
            return output if os.path.exists(output) else video_path

        except Exception as e:
            logger.error(f"         ❌ Caption error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            pathlib.Path(ass_path).unlink(missing_ok=True)
            return video_path

    # ✅ İYİLEŞTİRİLMİŞ HELPERS
    
    def _fallback_timings(self, text: str, duration: float) -> List[Tuple[str, float]]:
        """Fallback timing with character-weighted distribution."""
        # ✅ Büyük harf + temizlik
        text = (text or "").strip().upper()
        ws = [w for w in re.split(r"\s+", text) if w]
        if not ws:
            return []
        
        # ✅ Karakter ağırlıklı dağılım (daha adil)
        total_chars = sum(len(w) for w in ws)
        if total_chars == 0:
            per = max(self.MIN_WORD_DURATION, duration / len(ws))
            return [(w, per) for w in ws]
        
        times = []
        for w in ws:
            char_ratio = len(w) / total_chars
            word_dur = max(self.MIN_WORD_DURATION, duration * char_ratio)
            times.append((w, word_dur))
        
        # ✅ Hassas düzeltme
        total = sum(d for _, d in times)
        diff = duration - total
        if abs(diff) > self.TIMING_PRECISION and times:
            w, d = times[-1]
            times[-1] = (w, max(self.MIN_WORD_DURATION, d + diff))
        
        return times

    def _normalize_timings(
        self, 
        word_timings: List[Tuple[str, float]], 
        total: float
    ) -> List[Tuple[str, float]]:
        """
        ✅ İYİLEŞTİRİLMİŞ timing normalleştirme:
        - Büyük harf dönüşümü
        - Sıkı min/max kontrolü
        - Hassas ölçekleme
        - Son kelime düzeltmesi
        """
        if not word_timings:
            return []
        
        # ✅ Temizlik + büyük harf + sınırlar
        fixed = []
        for w, d in word_timings:
            if not w or not str(w).strip():
                continue
            clean_word = str(w).strip().upper()
            # ✅ Min/max sınırları
            clamped_dur = max(
                self.MIN_WORD_DURATION, 
                min(float(d), self.MAX_WORD_DURATION)
            )
            fixed.append((clean_word, clamped_dur))
        
        if not fixed:
            return []
        
        # ✅ Ölçekleme gerekli mi?
        current_sum = sum(d for _, d in fixed)
        if abs(current_sum - total) > self.TIMING_PRECISION and current_sum > 0:
            scale_factor = total / current_sum
            fixed = [
                (w, max(self.MIN_WORD_DURATION, round(d * scale_factor, 3))) 
                for w, d in fixed
            ]
            
            # ✅ Final hassas düzeltme (rounding hatalarını gider)
            final_sum = sum(d for _, d in fixed)
            diff = total - final_sum
            if abs(diff) > self.TIMING_PRECISION and fixed:
                w, d = fixed[-1]
                new_d = max(self.MIN_WORD_DURATION, d + diff)
                fixed[-1] = (w, round(new_d, 3))
        
        return fixed

    # ========================================================================
    # AGGRESSIVE VALIDATION - Prevents ALL drift
    # ========================================================================

    def _aggressive_validate(
        self,
        word_timings: List[Tuple[str, float]],
        total_duration: float
    ) -> List[Tuple[str, float]]:
        """
        AGGRESSIVE validation to prevent segment-internal drift.

        CRITICAL: This ensures exact timing within each segment.
        """
        if not word_timings:
            return []

        # Clean
        word_timings = [(w.strip(), d) for w, d in word_timings if w.strip()]
        if not word_timings:
            return []

        # Enforce min/max bounds
        fixed = []
        for word, dur in word_timings:
            dur = max(self.MIN_WORD_DURATION, min(dur, 3.0))
            fixed.append((word, round(dur, 3)))

        # CRITICAL: Check total sum
        current_total = sum(d for _, d in fixed)

        # If total exceeds target by >0.5%, scale DOWN
        if current_total > total_duration * 1.005:
            scale = total_duration / current_total
            logger.debug(f"      📏 Scaling DOWN: {scale:.3f}x (prevent overflow)")
            fixed = [
                (w, max(self.MIN_WORD_DURATION, round(d * scale, 3)))
                for w, d in fixed
            ]

        # EXACT match (±1ms tolerance)
        current_total = sum(d for _, d in fixed)
        diff = total_duration - current_total

        if abs(diff) > 0.001:
            # Distribute difference across ALL words proportionally
            for i in range(len(fixed)):
                word, dur = fixed[i]
                weight = dur / current_total if current_total > 0 else 1.0 / len(fixed)
                adjustment = diff * weight
                new_dur = max(self.MIN_WORD_DURATION, round(dur + adjustment, 3))
                fixed[i] = (word, new_dur)

        # Final check
        final_total = sum(d for _, d in fixed)
        diff_ms = abs(final_total - total_duration) * 1000

        if diff_ms > 1.0:
            # Last resort: adjust last word
            last_word, last_dur = fixed[-1]
            final_diff = total_duration - final_total
            fixed[-1] = (last_word, max(self.MIN_WORD_DURATION, round(last_dur + final_diff, 3)))

        # Validation log
        validated_total = sum(d for _, d in fixed)
        diff_ms = abs(validated_total - total_duration) * 1000
        logger.debug(f"      ✅ Validated: {validated_total:.3f}s (target: {total_duration:.3f}s, diff: {diff_ms:.1f}ms)")

        return fixed

    # ========================================================================
    # EXACT ASS WRITER - CHUNK-BASED (cümle cümle görünüm)
    # ========================================================================

    def _write_exact_ass(
        self,
        words: List[Tuple[str, float]],
        total_duration: float,
        sentence_type: str,
        output_path: str,
        time_offset: float = 0.0
    ):
        """
        Write ASS with CHUNK-BASED timing (cümle cümle görünüm).

        ✅ 2-3 kelimelik chunks
        ✅ Her chunk kendi timing'i ile
        ✅ No animation drift
        """
        if not words:
            logger.warning("      ⚠️ No words to write to ASS")
            return

        try:
            style = CAPTION_STYLES[get_random_style(sentence_type=sentence_type)]
        except Exception:
            style = CAPTION_STYLES.get("capcut_impact", {
                "fontname": "Impact",
                "fontsize_normal": 52,
                "fontsize_hook": 58,
                "bold": -1,
                "outline": 7,
                "shadow": "4",
                "color_outline": "&H00000000",
                "color_shadow": "&H80000000",
                "margin_v": 80
            })

        is_hook = (sentence_type == "hook")
        fontname = style.get("fontname", "Impact")
        fontsize = style.get("fontsize_hook" if is_hook else "fontsize_normal", 52)
        outline = style.get("outline", 7)
        shadow = style.get("shadow", "4")
        margin_v = style.get("margin_v", 80)

        # Colors (16:9 için beyaz/sarı)
        primary_color = "&H0000FFFF"  # Sarı
        secondary_color = "&H00FFFFFF"  # Beyaz
        outline_color = style.get("color_outline", "&H00000000")
        back_color = style.get("color_shadow", "&H80000000")

        # ASS Header - 16:9 aspect ratio
        ass = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{fontname},{fontsize},{primary_color},{secondary_color},{outline_color},{back_color},{style.get('bold', -1)},0,0,0,100,100,1.5,0,1,{outline},{shadow},2,50,50,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

        # ✅ Create chunks (2-3 words per chunk)
        max_words = 2 if is_hook else self.WORDS_PER_CHUNK
        chunks = self._create_chunks(words, max_words)

        # ✅ Validate chunks
        chunks = self._validate_chunks(chunks, total_duration)

        # ✅ Write dialogues with exact cumulative timing
        cumulative_time = 0.0

        for chunk_idx, chunk in enumerate(chunks):
            chunk_text = " ".join(w.upper() for w, _ in chunk)

            # DEBUG: Log words with numbers
            chunk_words = [w for w, _ in chunk]
            has_numbers = any(any(c.isdigit() for c in w) for w in chunk_words)
            if has_numbers:
                logger.info(f"      🔢 CAPTION NUMBERS: {chunk_words}")

            # Apply keyword highlighting if available
            if self.highlighter:
                try:
                    chunk_text = self.highlighter.highlight(chunk_text)
                except Exception:
                    pass

            # Calculate exact chunk duration from word timings
            chunk_duration = sum(d for _, d in chunk)

            # EXACT start/end with time offset applied
            start = cumulative_time - time_offset
            end = start + chunk_duration

            # Clamp to valid range
            start = max(0, start)
            end = min(end, total_duration)

            # Don't create zero-duration or negative chunks
            if end <= start:
                break

            # Convert to ASS time strings
            start_str = self._ass_time(start)
            end_str = self._ass_time(end)

            # NO EFFECTS for perfect sync!
            ass += f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{chunk_text}\n"

            # Update cumulative time
            cumulative_time += chunk_duration

        # Validation
        diff_ms = abs(cumulative_time - total_duration) * 1000
        if diff_ms > 10:
            logger.warning(f"      ⚠️ Caption timing drift: {diff_ms:.1f}ms")
        else:
            logger.debug(f"      ✅ Caption sync: {cumulative_time:.3f}s (drift: {diff_ms:.1f}ms)")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(ass)

    def _create_chunks(
        self,
        words: List[Tuple[str, float]],
        max_words: int
    ) -> List[List[Tuple[str, float]]]:
        """Create natural chunks for sentence-by-sentence display."""
        chunks = []
        current = []

        for word, dur in words:
            current.append((word, dur))

            should_finalize = False

            # Natural break points
            if word.rstrip().endswith(('.', '!', '?', '…', ':')):
                should_finalize = True
            elif len(current) >= max_words:
                should_finalize = True
            elif ',' in word and len(current) >= 2:
                should_finalize = True

            if should_finalize and current:
                chunks.append(current)
                current = []

        if current:
            chunks.append(current)

        return chunks

    def _validate_chunks(
        self,
        chunks: List[List[Tuple[str, float]]],
        total_duration: float
    ) -> List[List[Tuple[str, float]]]:
        """
        CRITICAL: Validate each chunk to prevent intra-segment drift.

        This ensures exact timing within each segment.
        """
        if not chunks:
            return chunks

        # Calculate current total
        current_total = sum(sum(d for _, d in chunk) for chunk in chunks)

        if current_total <= 0:
            return chunks

        # First pass: Scale all chunks proportionally
        scale = total_duration / current_total
        validated_chunks = []

        for chunk in chunks:
            scaled_chunk = []
            for word, dur in chunk:
                scaled_dur = max(self.MIN_WORD_DURATION, round(dur * scale, 3))
                scaled_chunk.append((word, scaled_dur))
            validated_chunks.append(scaled_chunk)

        # Second pass: Fix any remaining difference by adjusting last chunk
        actual_total = sum(sum(d for _, d in chunk) for chunk in validated_chunks)
        diff = total_duration - actual_total

        if abs(diff) > 0.001 and validated_chunks:
            # Adjust last word of last chunk
            last_chunk = validated_chunks[-1]
            if last_chunk:
                last_word, last_dur = last_chunk[-1]
                new_dur = max(self.MIN_WORD_DURATION, round(last_dur + diff, 3))
                validated_chunks[-1][-1] = (last_word, new_dur)

        # Log validation results
        final_total = sum(sum(d for _, d in chunk) for chunk in validated_chunks)
        diff_ms = abs(final_total - total_duration) * 1000

        if diff_ms > 1.0:
            logger.warning(f"      ⚠️ Chunk validation drift: {diff_ms:.1f}ms")
        else:
            logger.debug(f"      ✅ Chunks validated: {final_total:.3f}s (target: {total_duration:.3f}s)")

        return validated_chunks

    def _ass_time(self, seconds: float) -> str:
        """
        Format seconds to ASS time with MAXIMUM precision.

        Uses millisecond-level calculations to minimize rounding errors.
        """
        # Work in milliseconds for precision
        total_ms = int(round(seconds * 1000))

        # Convert to centiseconds (ASS format requirement)
        cs = total_ms // 10

        h = cs // 360000
        cs -= h * 360000
        m = cs // 6000
        cs -= m * 6000
        s = cs // 100
        cs %= 100

        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


# Geriye dönük uyumluluk
Renderer = CaptionRenderer
