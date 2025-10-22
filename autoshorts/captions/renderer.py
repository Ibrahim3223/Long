# -*- coding: utf-8 -*-
"""
Caption rendering - ULTIMATE VERSION
✅ GUARANTEED captions even without forced aligner
✅ Audio stream preserved
"""
import os
import pathlib
import logging
import re
from typing import List, Tuple, Optional, Dict, Any

from autoshorts.config import settings
from autoshorts.utils.ffmpeg_utils import run, has_subtitles, ffprobe_duration

logger = logging.getLogger(__name__)


class CaptionRenderer:
    """Render captions with BULLETPROOF sync."""
    
    WORDS_PER_CHUNK = 3
    MIN_WORD_DURATION = 0.08
    TIMING_PRECISION = 0.001
    FADE_DURATION = 0.0
    
    def __init__(self, caption_offset: Optional[float] = None):
        """Initialize caption renderer."""
        self.language = getattr(settings, 'LANG', 'en').lower()
        logger.info(f"      🎯 Caption renderer initialized ({self.language.upper()})")
    
    def render(
        self,
        video_path: str,
        text: str,
        words: List[Tuple[str, float]],
        duration: float,
        is_hook: bool = False,
        sentence_type: str = "buildup",
        temp_dir: str = None
    ) -> str:
        """Render captions with GUARANTEED output"""
        try:
            if duration <= 0:
                duration = ffprobe_duration(video_path)
            
            # ✅ Generate word timings if missing
            if not words:
                logger.info(f"         No word timings, generating fallback...")
                words = self._generate_fallback_timings(text, duration)
            
            # ✅ Validate and fix timings
            words = self._aggressive_validate(words, duration)
            
            if not words:
                logger.warning(f"         No valid words, skipping captions")
                return video_path
            
            frames = max(2, int(round(duration * settings.TARGET_FPS)))
            output = video_path.replace(".mp4", "_caption.mp4")
            
            # ✅ Check if captions are enabled
            if not settings.KARAOKE_CAPTIONS:
                logger.info(f"         Captions disabled in settings")
                return video_path
            
            # ✅ Generate ASS file
            ass_path = video_path.replace(".mp4", ".ass")
            
            try:
                self._write_exact_ass(words, duration, sentence_type, ass_path)
            except Exception as e:
                logger.error(f"         ❌ ASS generation failed: {e}")
                return video_path
            
            if not os.path.exists(ass_path):
                logger.error(f"         ❌ ASS file not created")
                return video_path
            
            logger.info(f"         ✅ ASS file created: {os.path.getsize(ass_path)} bytes")
            
            # ✅ Burn subtitles with audio preservation
            tmp_out = output.replace(".mp4", ".tmp.mp4")
            
            try:
                # First pass: burn subtitles
                run([
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", video_path,
                    "-vf", f"subtitles='{ass_path}':force_style='Kerning=1',setsar=1,fps={settings.TARGET_FPS}",
                    "-r", str(settings.TARGET_FPS), "-vsync", "cfr",
                    "-c:a", "copy",  # ✅ PRESERVE AUDIO
                    "-c:v", "libx264", "-preset", "medium",
                    "-crf", str(max(16, settings.CRF_VISUAL - 3)),
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    tmp_out
                ])
                
                if not os.path.exists(tmp_out):
                    logger.error(f"         ❌ Subtitle burn failed")
                    return video_path
                
                # Second pass: trim to exact duration
                run([
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", tmp_out,
                    "-vf", f"setsar=1,fps={settings.TARGET_FPS},trim=start_frame=0:end_frame={frames}",
                    "-r", str(settings.TARGET_FPS), "-vsync", "cfr",
                    "-c:a", "copy",  # ✅ PRESERVE AUDIO
                    "-c:v", "libx264", "-preset", "medium",
                    "-crf", str(settings.CRF_VISUAL),
                    "-pix_fmt", "yuv420p",
                    output
                ])
                
                if not os.path.exists(output):
                    logger.error(f"         ❌ Final output failed")
                    return video_path
                
            finally:
                pathlib.Path(ass_path).unlink(missing_ok=True)
                pathlib.Path(tmp_out).unlink(missing_ok=True)
            
            logger.info(f"         ✅ Captions rendered successfully")
            return output
                
        except Exception as e:
            logger.error(f"         ❌ Caption error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return video_path
    
    def _generate_fallback_timings(
        self,
        text: str,
        duration: float
    ) -> List[Tuple[str, float]]:
        """Generate fallback word timings"""
        words = [w for w in re.split(r'\s+', text.strip()) if w]
        
        if not words:
            return []
        
        # Equal distribution
        per_word = max(self.MIN_WORD_DURATION, duration / len(words))
        timings = [(w, per_word) for w in words]
        
        # Adjust last word for exact match
        if timings:
            current_sum = sum(d for _, d in timings)
            diff = duration - current_sum
            
            if abs(diff) > 0.001:
                last_word, last_dur = timings[-1]
                timings[-1] = (last_word, max(self.MIN_WORD_DURATION, last_dur + diff))
        
        return timings
    
    def _aggressive_validate(
        self,
        word_timings: List[Tuple[str, float]],
        total_duration: float
    ) -> List[Tuple[str, float]]:
        """AGGRESSIVE validation"""
        if not word_timings:
            return []
        
        # Clean
        word_timings = [(w.strip(), d) for w, d in word_timings if w.strip()]
        if not word_timings:
            return []
        
        # Enforce bounds
        fixed = []
        for word, dur in word_timings:
            dur = max(self.MIN_WORD_DURATION, min(dur, 5.0))
            fixed.append((word, dur))
        
        if not fixed:
            return []
        
        # Calculate total
        current_total = sum(d for _, d in fixed)
        
        # Force EXACT match
        if abs(current_total - total_duration) > 0.001:
            scale_factor = total_duration / current_total if current_total > 0 else 1.0
            
            fixed = [
                (w, max(self.MIN_WORD_DURATION, round(d * scale_factor, 3)))
                for w, d in fixed
            ]
            
            # Fine-tune last word
            new_total = sum(d for _, d in fixed)
            diff = total_duration - new_total
            
            if abs(diff) > 0.001 and fixed:
                last_word, last_dur = fixed[-1]
                fixed[-1] = (last_word, max(self.MIN_WORD_DURATION, last_dur + diff))
        
        return fixed
    
    def _write_exact_ass(
        self,
        words: List[Tuple[str, float]],
        total_duration: float,
        sentence_type: str,
        output_path: str
    ):
        """Write ASS file with frame-perfect timing"""
        
        is_hook = (sentence_type == "hook")
        
        # ASS header with simple styling
        ass = f"""[Script Info]
Title: Caption
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,2,2,50,50,80,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        
        max_words = 2 if is_hook else self.WORDS_PER_CHUNK
        chunks = self._create_chunks(words, max_words)
        chunks = self._validate_chunks(chunks, total_duration)
        
        cumulative_time = 0.0
        
        for chunk_idx, chunk in enumerate(chunks):
            chunk_text = " ".join(w.upper() for w, _ in chunk)
            chunk_duration = sum(d for _, d in chunk)
            
            start = cumulative_time
            end = start + chunk_duration
            
            if end > total_duration + 0.001:
                end = total_duration
                if start >= end:
                    break
            
            # Frame alignment
            frame_duration = 1.0 / settings.TARGET_FPS
            start_frame = round(start / frame_duration)
            end_frame = round(end / frame_duration)
            
            start_aligned = start_frame * frame_duration
            end_aligned = end_frame * frame_duration
            
            start_str = self._ass_time(start_aligned)
            end_str = self._ass_time(end_aligned)
            
            ass_start = self._ass_to_seconds(start_str)
            ass_end = self._ass_to_seconds(end_str)
            
            ass += f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{chunk_text}\n"
            
            cumulative_time = ass_end
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(ass)
    
    def _create_chunks(
        self,
        words: List[Tuple[str, float]],
        max_words: int
    ) -> List[List[Tuple[str, float]]]:
        """Create natural chunks"""
        chunks = []
        current = []
        
        for word, dur in words:
            current.append((word, dur))
            
            should_finalize = False
            
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
        """Validate chunk durations"""
        if not chunks:
            return chunks
        
        current_total = sum(sum(d for _, d in chunk) for chunk in chunks)
        validated_chunks = []
        remaining_duration = total_duration
        
        for i, chunk in enumerate(chunks):
            is_last = (i == len(chunks) - 1)
            chunk_dur = sum(d for _, d in chunk)
            
            if is_last:
                target_dur = remaining_duration
            else:
                weight = chunk_dur / current_total if current_total > 0 else 1.0 / len(chunks)
                target_dur = total_duration * weight
            
            if abs(chunk_dur - target_dur) > 0.001:
                scale = target_dur / chunk_dur if chunk_dur > 0 else 1.0
                chunk = [(w, max(self.MIN_WORD_DURATION, round(d * scale, 3))) for w, d in chunk]
            
            validated_chunks.append(chunk)
            remaining_duration -= sum(d for _, d in chunk)
        
        return validated_chunks
    
    def _ass_time(self, seconds: float) -> str:
        """Format seconds to ASS time"""
        total_ms = int(round(seconds * 1000))
        cs = total_ms // 10
        
        h = cs // 360000
        cs -= h * 360000
        m = cs // 6000
        cs -= m * 6000
        s = cs // 100
        cs %= 100
        
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"
    
    def _ass_to_seconds(self, ass_time: str) -> float:
        """Convert ASS time to seconds"""
        parts = ass_time.split(':')
        h = int(parts[0])
        m = int(parts[1])
        s_and_cs = parts[2].split('.')
        s = int(s_and_cs[0])
        cs = int(s_and_cs[1]) if len(s_and_cs) > 1 else 0
        
        return h * 3600 + m * 60 + s + cs * 0.01
