# FILE: autoshorts/tts/edge_handler.py
# -*- coding: utf-8 -*-
"""
Edge-TTS handler - BULLETPROOF with Fast Fallback
Quick fail (2 retries) → Immediate Google TTS fallback
Prioritizes reliability over retrying Edge-TTS
(Kept API intact; minor speed optimizations)
"""
import re
import asyncio
import logging
import time
import tempfile
import os
from typing import List, Tuple, Dict, Any

try:
    import edge_tts
    import nest_asyncio
except ImportError:
    raise ImportError("edge-tts and nest_asyncio required: pip install edge-tts nest_asyncio")

try:
    import requests
except ImportError:
    raise ImportError("requests required: pip install requests")

from autoshorts.config import settings
from autoshorts.utils.ffmpeg_utils import run, ffprobe_duration

logger = logging.getLogger(__name__)


class TTSHandler:
    """Handle text-to-speech generation with bulletproof Edge-TTS."""

    MAX_RETRIES = 2
    INITIAL_RETRY_DELAY = 0.25
    MAX_RETRY_DELAY = 1.5

    REQUEST_DELAY = 0.15  # slightly faster
    LAST_REQUEST_TIME = 0

    def __init__(self):
        self.voice = settings.VOICE
        self.rate = settings.TTS_RATE
        self.lang = settings.LANG
        nest_asyncio.apply()
        logger.info(f"🎤 TTS initialized: voice={self.voice}, rate={self.rate}")

    def _rate_limit_wait(self):
        current_time = time.time()
        time_since_last = current_time - TTSHandler.LAST_REQUEST_TIME
        if time_since_last < self.REQUEST_DELAY:
            time.sleep(self.REQUEST_DELAY - time_since_last)
        TTSHandler.LAST_REQUEST_TIME = time.time()

    def generate(self, text: str) -> Dict[str, Any]:
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            wav_path = tmp.name
        try:
            duration, word_timings = self.synthesize(text.strip(), wav_path)
            with open(wav_path, 'rb') as f:
                audio_data = f.read()
            return {
                'audio': audio_data,
                'duration': duration,
                'word_timings': word_timings
            }
        finally:
            if os.path.exists(wav_path):
                try:
                    os.unlink(wav_path)
                except Exception:
                    pass

    def synthesize(self, text: str, wav_out: str) -> Tuple[float, List[Tuple[str, float]]]:
        text = (text or "").strip()
        if not text:
            self._generate_silence(wav_out, 1.0)
            return 1.0, []

        atempo = self._rate_to_atempo(self.rate)
        retry_delay = self.INITIAL_RETRY_DELAY

        # Layer 1: Edge with word boundaries
        for attempt in range(self.MAX_RETRIES):
            try:
                self._rate_limit_wait()
                marks = self._edge_stream_tts(text, wav_out)
                duration = self._apply_atempo(wav_out, atempo)
                words = self._merge_marks_to_words(text, marks, duration, atempo)
                logger.info(f"✅ Edge-TTS success: {len(words)} words | {duration:.2f}s")
                return duration, words
            except Exception as e:
                if attempt < self.MAX_RETRIES - 1:
                    logger.warning(f"⚠️ Edge-TTS attempt {attempt+1} failed: {str(e)[:60]}")
                    time.sleep(retry_delay); retry_delay = min(retry_delay * 1.5, self.MAX_RETRY_DELAY)
                else:
                    logger.warning("⚠️ Edge-TTS with marks failed, trying simple mode...")

        # Layer 2: Edge simple
        retry_delay = self.INITIAL_RETRY_DELAY
        for attempt in range(2):
            try:
                self._rate_limit_wait()
                self._edge_simple(text, wav_out)
                duration = self._apply_atempo(wav_out, atempo)
                logger.info(f"✅ Edge-TTS simple: audio generated ({duration:.2f}s)")
                return duration, []  # force align later
            except Exception as e2:
                if attempt < 1:
                    logger.warning(f"⚠️ Edge-TTS simple attempt failed: {str(e2)[:60]}")
                    time.sleep(retry_delay)
                else:
                    logger.warning("⚠️ Edge-TTS simple failed, falling back to Google...")

        # Layer 3: Google fallback
        try:
            logger.info("🔄 Using Google TTS fallback...")
            self._google_tts(text, wav_out)
            duration = self._apply_atempo(wav_out, atempo)
            logger.info(f"✅ Google TTS: audio generated ({duration:.2f}s)")
            return duration, []
        except Exception as e3:
            logger.error(f"❌ Google TTS failed: {str(e3)[:100]}")
            self._generate_silence(wav_out, 4.0)
            return 4.0, []

    def _edge_stream_tts(self, text: str, wav_out: str) -> List[Dict[str, Any]]:
        mp3_path = wav_out.replace(".wav", ".mp3")
        marks: List[Dict[str, Any]] = []

        async def _run():
            audio = bytearray()
            comm = edge_tts.Communicate(text=text, voice=self.voice, rate=self.rate)
            try:
                async for chunk in comm.stream():
                    ctype = chunk.get("type")
                    if ctype == "audio":
                        audio.extend(chunk.get("data", b""))
                    elif ctype == "WordBoundary":
                        offset = float(chunk.get("offset", 0)) / 10_000_000.0
                        duration = float(chunk.get("duration", 0)) / 10_000_000.0
                        marks.append({"t0": offset, "t1": offset + duration, "text": str(chunk.get("text", ""))})
                if not audio:
                    raise RuntimeError("No audio data received from Edge-TTS")
                with open(mp3_path, "wb") as f:
                    f.write(bytes(audio))
            except Exception as e:
                pathlib = __import__("pathlib")
                pathlib.Path(mp3_path).unlink(missing_ok=True)
                raise e

        try:
            asyncio.run(asyncio.wait_for(_run(), timeout=30.0))
        except asyncio.TimeoutError:
            raise RuntimeError("Edge-TTS timeout after 30 seconds")
        return marks

    def _edge_simple(self, text: str, wav_out: str):
        mp3_path = wav_out.replace(".wav", ".mp3")
        async def _run():
            comm = edge_tts.Communicate(text, voice=self.voice, rate=self.rate)
            await comm.save(mp3_path)
        try:
            asyncio.run(asyncio.wait_for(_run(), timeout=30.0))
        except asyncio.TimeoutError:
            raise RuntimeError("Edge-TTS simple timeout after 30 seconds")

    def _google_tts(self, text: str, wav_out: str):
        mp3_path = wav_out.replace(".wav", ".mp3")
        text = text.strip()
        if not text:
            raise ValueError("Empty text")
        if len(text) > 200:
            text = text[:197] + "..."
        q = requests.utils.quote(text.replace('"', '').replace("'", ""))
        lang_code = self.lang or "en"
        url = (
            f"https://translate.google.com/translate_tts?"
            f"ie=UTF-8&q={q}&tl={lang_code}&client=tw-ob&ttsspeed=1.0"
        )
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        for attempt in range(2):
            try:
                r = requests.get(url, headers=headers, timeout=12)
                r.raise_for_status()
                if len(r.content) < 100:
                    raise ValueError("Response too short")
                with open(mp3_path, "wb") as f:
                    f.write(r.content)
                return
            except Exception:
                if attempt == 1:
                    raise
                time.sleep(0.4)

    def _apply_atempo(self, wav_out: str, atempo: float) -> float:
        mp3_path = wav_out.replace(".wav", ".mp3")
        if not os.path.exists(mp3_path):
            raise FileNotFoundError(f"MP3 not found: {mp3_path}")
        try:
            run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", mp3_path,
                "-ar", "48000", "-ac", "1", "-acodec", "pcm_s16le",
                "-af", f"dynaudnorm=g=7:f=250,atempo={atempo:.3f}",
                wav_out
            ])
        finally:
            try:
                os.unlink(mp3_path)
            except Exception:
                pass
        return ffprobe_duration(wav_out)

    def _rate_to_atempo(self, rate_str: str, default: float = 1.10) -> float:
        try:
            if not rate_str:
                return default
            rate_str = rate_str.strip()
            if rate_str.endswith("%"):
                val = float(rate_str.replace("%", ""))
                return max(0.5, min(2.0, 1.0 + val / 100.0))
            if rate_str.endswith(("x", "X")):
                return max(0.5, min(2.0, float(rate_str[:-1])))
            return max(0.5, min(2.0, float(rate_str)))
        except Exception:
            return default

    def _merge_marks_to_words(
        self,
        text: str,
        marks: List[Dict[str, Any]],
        total_duration: float,
        atempo: float = 1.0
    ) -> List[Tuple[str, float]]:
        words = [w for w in re.split(r"\s+", text.strip()) if w]
        if not words:
            return []
        out: List[Tuple[str, float]] = []
        if marks and len(marks) >= len(words) * 0.7:
            N = min(len(words), len(marks))
            raw_durs = []
            for i in range(N):
                t0 = float(marks[i]["t0"]); t1 = float(marks[i]["t1"])
                raw_durs.append(max(0.05, t1 - t0))
            scaled = [d / atempo for d in raw_durs]
            ssum = sum(scaled) or 1.0
            corr = total_duration / ssum
            for i in range(N):
                out.append((words[i], max(0.05, scaled[i] * corr)))
            if len(words) > N:
                used = sum(d for _, d in out)
                remain = max(0.0, total_duration - used)
                each = remain / (len(words) - N) if (len(words) - N) > 0 else 0.1
                for i in range(N, len(words)):
                    out.append((words[i], max(0.05, each)))
            cur = sum(d for _, d in out)
            diff = total_duration - cur
            if abs(diff) > 0.01 and out:
                w, d = out[-1]; out[-1] = (w, max(0.05, d + diff))
        else:
            each = max(0.05, total_duration / len(words))
            out = [(w, each) for w in words]
            cur = sum(d for _, d in out); diff = total_duration - cur
            if abs(diff) > 0.01 and out:
                w, d = out[-1]; out[-1] = (w, max(0.05, d + diff))
        return out

    def _generate_silence(self, wav_out: str, duration: float):
        run([
            "ffmpeg", "-y", "-f", "lavfi",
            "-t", f"{duration:.3f}",
            "-i", "anullsrc=r=48000:cl=mono",
            wav_out
        ])
