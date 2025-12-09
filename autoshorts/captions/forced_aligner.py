# -*- coding: utf-8 -*-
"""
Forced Alignment - ✅ İYİLEŞTİRİLMİŞ
- Daha hassas timing
- Büyük harf desteği
- Gelişmiş validasyon
- ✅ SAYI KORUMA: "2" -> "two" sorunu çözüldü
"""
import os
import logging
import warnings
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

# ============================================================
# NUMBER WORD MAPPINGS - Fix "2" -> "two" issue in captions
# ============================================================
NUMBER_WORDS = {
    "0": ["zero", "o", "oh"],
    "1": ["one", "won"],
    "2": ["two", "to", "too"],
    "3": ["three"],
    "4": ["four", "for", "fore"],
    "5": ["five"],
    "6": ["six"],
    "7": ["seven"],
    "8": ["eight", "ate"],
    "9": ["nine"],
    "10": ["ten"],
    "11": ["eleven"],
    "12": ["twelve"],
    "13": ["thirteen"],
    "14": ["fourteen"],
    "15": ["fifteen"],
    "16": ["sixteen"],
    "17": ["seventeen"],
    "18": ["eighteen"],
    "19": ["nineteen"],
    "20": ["twenty"],
    "30": ["thirty"],
    "40": ["forty"],
    "50": ["fifty"],
    "60": ["sixty"],
    "70": ["seventy"],
    "80": ["eighty"],
    "90": ["ninety"],
    "100": ["hundred"],
    "1000": ["thousand"],
}

# Reverse mapping: word -> digit
WORD_TO_NUMBER = {}
for digit, words in NUMBER_WORDS.items():
    for word in words:
        WORD_TO_NUMBER[word] = digit


def _numbers_match(known: str, trans: str) -> bool:
    """
    Check if a number digit matches a transcribed word.

    Examples:
        _numbers_match("2", "two") -> True
        _numbers_match("4", "for") -> True
        _numbers_match("10", "ten") -> True
    """
    # Direct match
    if known == trans:
        return True

    # Check if known is a digit and trans is its word form
    if known in NUMBER_WORDS:
        if trans in NUMBER_WORDS[known]:
            return True

    # Check if trans is a digit and known is its word form
    if trans in NUMBER_WORDS:
        if known in NUMBER_WORDS[trans]:
            return True

    # Check reverse mapping
    if known in WORD_TO_NUMBER and WORD_TO_NUMBER[known] == trans:
        return True
    if trans in WORD_TO_NUMBER and WORD_TO_NUMBER[trans] == known:
        return True

    return False


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
    # ✅ İyileştirilmiş parametreler
    MIN_WORD_DURATION = 0.05  # Daha düşük minimum (hızlı konuşma için)
    MAX_WORD_DURATION = 4.0    # Makul maksimum
    TIMING_PRECISION = 0.001   # Milisaniye hassasiyeti
    WHISPER_MODEL = "base"     # CPU için iyi denge

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
        """Main alignment entry point with fallback chain."""
        lang = (language or self.language).lower()

        # ✅ Metni büyük harfe çevir
        text = (text or "").strip().upper()

        # ✅ CRITICAL: Get KNOWN words from original text (preserves "2", "10", etc.)
        known_words = [w.strip() for w in text.split() if w.strip()]

        # 1) TTS timings varsa - MAP TO KNOWN WORDS (sayı koruma!)
        if tts_word_timings and len(tts_word_timings) > 0:
            logger.debug(f"      📝 TTS provided {len(tts_word_timings)} word timings")

            # ✅ CRITICAL FIX: Map durations to KNOWN WORDS, not transcribed words
            # This fixes numbers being replaced (e.g., "2" -> "two")
            mapped_timings = self._map_tts_timings_to_known_words(
                known_words=known_words,
                tts_word_timings=tts_word_timings,
                total_duration=total_duration
            )

            if mapped_timings:
                logger.debug(f"      ✅ Mapped TTS timings to {len(mapped_timings)} known words")
                return self._validate_timings(mapped_timings, total_duration)

        # 2) Forced alignment via stable-ts
        if _STABLE_TS_AVAILABLE:
            try:
                words = self._stable_ts_align(text, audio_path, total_duration, lang)
                if words:
                    return words
            except Exception as e:
                logger.warning(f"      stable-ts alignment failed: {e}")

        # 3) Character-based smart estimation
        if total_duration is not None:
            return self._smart_estimation(text, total_duration)

        # 4) Last resort: uniform split (0.20s per word)
        words = [w for w in text.split() if w.strip()]
        return [(w, 0.20) for w in words]

    # -------------------------- Core logic --------------------------- #

    def _get_model(self, language: str):
        """Load or retrieve cached Whisper model."""
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
        """Stable-ts alignment with improved mapping."""
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

        # ✅ Transcribed word durations topla
        trans_durs: List[Tuple[str, float]] = []
        for seg in getattr(result, "segments", []):
            for w in getattr(seg, "words", []) or []:
                token = (getattr(w, "word", "") or "").strip().upper()
                if not token:
                    continue
                start = float(getattr(w, "start", 0.0) or 0.0)
                end = float(getattr(w, "end", 0.0) or 0.0)
                dur = max(self.MIN_WORD_DURATION, end - start)
                trans_durs.append((token, dur))

        if not trans_durs:
            logger.warning("      stable-ts returned no word timings")
            return None

        # ✅ Bilinen metne map et
        known_tokens = [t.strip() for t in known_text.split() if t.strip()]
        
        if len(known_tokens) == len(trans_durs):
            # ✅ Aynı uzunluk: direkt eşle
            mapped = [(known_tokens[i], trans_durs[i][1]) for i in range(len(known_tokens))]
        else:
            # ✅ Farklı uzunluk: karakter ağırlıklı dağıt
            sum_trans = sum(d for _, d in trans_durs) or (total_duration or 0.0)
            target_total = total_duration if total_duration is not None else sum_trans
            total_chars = sum(len(t) for t in known_tokens) or 1
            
            mapped = [
                (tok, max(self.MIN_WORD_DURATION, target_total * (len(tok) / total_chars)))
                for tok in known_tokens
            ]

        return self._validate_timings(mapped, total_duration)

    def _map_tts_timings_to_known_words(
        self,
        known_words: List[str],
        tts_word_timings: List[Tuple[str, float]],
        total_duration: Optional[float]
    ) -> Optional[List[Tuple[str, float]]]:
        """
        ✅ CRITICAL: Map TTS word timings to KNOWN words from original text.

        This preserves numbers in captions:
        - Known text: "PART 2 IS COMING"
        - TTS output: "PART TWO IS COMING"
        - Result: ["PART", "2", "IS", "COMING"] with correct durations

        Uses _numbers_match() to handle "2" <-> "two" matching.
        """
        if not known_words or not tts_word_timings:
            return None

        # Normalize TTS words to uppercase
        tts_words = [(w.strip().upper(), d) for w, d in tts_word_timings if w.strip()]

        if not tts_words:
            return None

        # ✅ Case 1: Same word count - direct mapping with number preservation
        if len(known_words) == len(tts_words):
            result = []
            for i, known in enumerate(known_words):
                tts_word, tts_dur = tts_words[i]

                # Use known word (preserves "2") with TTS duration
                result.append((known, tts_dur))

                # Log if we preserved a number
                if known != tts_word and _numbers_match(known.lower(), tts_word.lower()):
                    logger.debug(f"      🔢 Number preserved: '{tts_word}' -> '{known}'")

            return result

        # ✅ Case 2: Different word counts - try smart alignment
        logger.debug(f"      ⚠️ Word count mismatch: known={len(known_words)}, tts={len(tts_words)}")

        # Try to align words using fuzzy matching
        result = []
        tts_idx = 0

        for known in known_words:
            known_lower = known.lower()

            # Find matching TTS word
            matched = False
            search_range = min(3, len(tts_words) - tts_idx)  # Look ahead up to 3 words

            for offset in range(search_range):
                if tts_idx + offset >= len(tts_words):
                    break

                tts_word, tts_dur = tts_words[tts_idx + offset]
                tts_lower = tts_word.lower()

                # Check for exact match or number match
                if known_lower == tts_lower or _numbers_match(known_lower, tts_lower):
                    result.append((known, tts_dur))
                    tts_idx = tts_idx + offset + 1
                    matched = True

                    if known_lower != tts_lower:
                        logger.debug(f"      🔢 Number preserved: '{tts_word}' -> '{known}'")
                    break

            # If no match found, use average duration
            if not matched:
                avg_dur = sum(d for _, d in tts_words) / len(tts_words)
                result.append((known, avg_dur))
                logger.debug(f"      ⚠️ No TTS match for '{known}', using avg duration {avg_dur:.3f}s")

        # ✅ Scale to total duration if provided
        if result and total_duration is not None:
            current_sum = sum(d for _, d in result)
            if current_sum > 0:
                scale = total_duration / current_sum
                result = [(w, d * scale) for w, d in result]

        return result if result else None

    # --------------------------- Utilities --------------------------- #

    def _validate_timings(
        self, 
        word_timings: List[Tuple[str, float]], 
        total_duration: Optional[float]
    ) -> List[Tuple[str, float]]:
        """
        ✅ İYİLEŞTİRİLMİŞ validation:
        - Boş kelimeleri filtrele
        - Min/max clamp
        - Hassas ölçekleme
        - Son kelime düzeltmesi
        """
        # ✅ Temizlik + büyük harf
        words = [
            (w.strip().upper(), float(d)) 
            for w, d in word_timings 
            if w and w.strip()
        ]
        if not words:
            return []
        
        # ✅ Clamp durations
        words = [
            (w, max(self.MIN_WORD_DURATION, min(self.MAX_WORD_DURATION, d))) 
            for w, d in words
        ]

        # ✅ Total duration varsa ölçekle
        if total_duration is not None and total_duration > 0:
            current_sum = sum(d for _, d in words)
            if current_sum <= 0:
                return []
            
            # ✅ Ölçekleme faktörü
            scale = total_duration / current_sum
            words = [(w, d * scale) for w, d in words]
            
            # ✅ Final hassas düzeltme (rounding errors)
            current = sum(d for _, d in words)
            diff = total_duration - current
            if abs(diff) > self.TIMING_PRECISION and words:
                w, d = words[-1]
                new_d = max(self.MIN_WORD_DURATION, d + diff)
                words[-1] = (w, round(new_d, 3))
            
            # ✅ Tüm durations'ı round et
            words = [(w, round(d, 3)) for w, d in words]

        return words

    def _smart_estimation(self, text: str, total_duration: float) -> List[Tuple[str, float]]:
        """
        ✅ Karakter ağırlıklı akıllı tahmin:
        - Uzun kelimeler daha fazla süre alır
        - Hassas dağılım
        """
        tokens = [t for t in text.split() if t.strip()]
        if not tokens:
            return []
        
        # ✅ Karakter bazlı ağırlık
        total_chars = sum(len(t) for t in tokens) or 1
        words = [
            (t, max(self.MIN_WORD_DURATION, total_duration * (len(t) / total_chars))) 
            for t in tokens
        ]
        
        return self._validate_timings(words, total_duration)
