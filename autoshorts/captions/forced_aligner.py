# FILE: autoshorts/captions/forced_alignment.py
# -*- coding: utf-8 -*-
"""
Forced Alignment – resilient, with known-text optimization.
- Uses stable-ts (stable_whisper) when available
- Falls back to TTS timings or character-based estimation
- Exact-duration validation and proportional scaling
"""
import logging
import warnings
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

# Lazy import / availability flag
_STABLE_TS_AVAILABLE = False
_stable_models = {}

try:
    import stable_whisper  # provided by 'stable-ts' package
    _STABLE_TS_AVAILABLE = True
    logger.info("✅ stable-ts available – forced alignment enabled")
except ImportError:
    logger.warning("⚠️ stable-ts not installed. Install: pip install stable-ts")


class ForcedAligner:
    MIN_WORD_DURATION = 0.08
    MAX_WORD_DURATION = 3.0
    WHISPER_MODEL = "base"  # good balance for CPU

    def __init__(self, language: str = "en") -> None:
        self.language = (language or "en").lower()
        logger.info(f"      🎯 Forced aligner ready (lang={self.language.upper()})")

    # --------------------------- Public API --------------------------- #

    def align(
        self,
        text: str,
        audio_path: str,
        tts_word_timings: Optional[List[Tuple[str, float]]] = None,
        total_duration: Optional[float] = None,
        language: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        lang = (language or self.language).lower()

        # 1) If TTS already provided timings, validate and return
        if tts_word_timings:
            logger.debug(f"      Using TTS word timings ({len(tts_word_timings)} words)")
            return self._validate_timings(tts_word_timings, total_duration)

        # 2) Forced alignment via stable-ts if available
        if _STABLE_TS_AVAILABLE:
            try:
                words = self._stable_ts_align(text, audio_path, total_duration, lang)
                if words:
                    return words
            except Exception as e:
                logger.warning(f"      stable-ts alignment failed: {e}")

        # 3) Character-based estimation fallback
        if total_duration is not None:
            return self._smart_estimation(text, total_duration)

        # 4) Last resort: uniform split (0.25s per word)
        words = [w for w in (text or "").split() if w.strip()]
        return [(w, 0.25) for w in words]

    # -------------------------- Core logic --------------------------- #

    def _get_model(self, language: str):
        if not _STABLE_TS_AVAILABLE:
            return None
        if language in _stable_models:
            return _stable_models[language]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = stable_whisper.load_model(self.WHISPER_MODEL, device="cpu")
            _stable_models[language] = model
            logger.info(f"      ✅ Loaded stable-ts model '{self.WHISPER_MODEL}' on CPU")
            return model
        except Exception as e:
            logger.error(f"      ❌ Failed to load stable-ts model: {e}")
            return None

    def _stable_ts_align(
        self,
        known_text: str,
        audio_path: str,
        total_duration: Optional[float],
        language: str,
    ) -> Optional[List[Tuple[str, float]]]:
        model = self._get_model(language)
        if model is None:
            return None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.transcribe(
                audio_path,
                language=language,
                word_timestamps=True,
                regroup=False,
                verbose=False,
                initial_prompt=(known_text[:200] if len(known_text) > 200 else known_text),
                condition_on_previous_text=False,
                temperature=0.0,
            )

        # Collect transcribed word durations (start/end) → durations
        trans_durs: List[Tuple[str, float]] = []
        for seg in getattr(result, "segments", []):
            for w in getattr(seg, "words", []) or []:
                token = (getattr(w, "word", "") or "").strip()
                if not token:
                    continue
                start = float(getattr(w, "start", 0.0) or 0.0)
                end = float(getattr(w, "end", 0.0) or 0.0)
                dur = max(self.MIN_WORD_DURATION, end - start)
                trans_durs.append((token, dur))

        if not trans_durs:
            logger.warning("      stable-ts returned no word timings")
            return None

        # Map to known text tokens if lengths differ (proportional to chars)
        known_tokens = [t.strip() for t in known_text.split() if t.strip()]
        if len(known_tokens) == len(trans_durs):
            mapped = [(known_tokens[i], trans_durs[i][1]) for i in range(len(known_tokens))]
        else:
            sum_trans = sum(d for _, d in trans_durs) or (total_duration or 0.0)
            target_total = total_duration if total_duration is not None else sum_trans
            total_chars = sum(len(t) for t in known_tokens) or 1
            mapped = [(tok, max(self.MIN_WORD_DURATION, target_total * (len(tok) / total_chars)))
                      for tok in known_tokens]

        return self._validate_timings(mapped, total_duration)

    # --------------------------- Utilities --------------------------- #

    def _validate_timings(
        self, word_timings: List[Tuple[str, float]], total_duration: Optional[float]
    ) -> List[Tuple[str, float]]:
        words = [(w.strip(), float(d)) for w, d in word_timings if w and w.strip()]
        if not words:
            return []
        # Clamp per-word durations
        words = [(w, max(self.MIN_WORD_DURATION, min(self.MAX_WORD_DURATION, d))) for w, d in words]

        if total_duration is not None:
            s = sum(d for _, d in words)
            if s <= 0:
                return []
            scale = total_duration / s
            words = [(w, d * scale) for w, d in words]
            # Final precision fix on last token
            cur = sum(d for _, d in words)
            diff = total_duration - cur
            if abs(diff) > 1e-3 and words:
                w, d = words[-1]
                words[-1] = (w, max(self.MIN_WORD_DURATION, d + diff))
            words = [(w, round(d, 3)) for w, d in words]

        return words

    def _smart_estimation(self, text: str, total_duration: float) -> List[Tuple[str, float]]:
        tokens = [t for t in (text or "").split() if t.strip()]
        if not tokens:
            return []
        total_chars = sum(len(t) for t in tokens) or 1
        words = [(t, max(self.MIN_WORD_DURATION, total_duration * (len(t) / total_chars))) for t in tokens]
        return self._validate_timings(words, total_duration)
