# -*- coding: utf-8 -*-
"""
Caption rendering - LONG-FORM (16:9)
✅ BÜYÜK HARF altyazılar
✅ İYİLEŞTİRİLMİŞ senkronizasyon
✅ Vurgu YOK, animasyon YOK
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
    MIN_WORD_DURATION = 0.05  # ✅ Daha düşük minimum (senkronizasyon için)
    MAX_WORD_DURATION = 4.0   # ✅ Maksimum sınır
    TIMING_PRECISION = 0.001  # ✅ Milisaniye hassasiyeti

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
        caption_offset: float = 0.0,
    ) -> str:
        """Render captions with improved synchronization."""
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

            # ✅ Offset varsa word timings'i kaydır
            if caption_offset > 0:
                words = [(w, d) for w, d in words]  # Copy list
                logger.debug(f"Applying caption offset: +{caption_offset:.2f}s")
            
          # ✅ ASS dosyası oluştur - sentence type'a göre stil seç
            style_name = get_random_style(sentence_type=sentence_type)
            ass_content = build_karaoke_ass(
                text=text,
                seg_dur=duration,
                words=words,
                is_hook=(sentence_type == "hook"),
                style_name=style_name,
                time_offset=caption_offset,
            )
            
            with open(ass_path, "w", encoding="utf-8") as f:
                f.write(ass_content)

            # ✅ FFmpeg render (frame-accurate)
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
                "-r", str(settings.TARGET_FPS), 
                "-vsync", "cfr",  # ✅ Constant frame rate
                "-c:v", "libx264", 
                "-preset", "medium",  # ✅ Kalite/hız dengesi
                "-crf", str(settings.CRF_VISUAL),
                "-pix_fmt", "yuv420p",
                "-an",  # ✅ Audio yok (sonra eklenecek)
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


# Geriye dönük uyumluluk
Renderer = CaptionRenderer
