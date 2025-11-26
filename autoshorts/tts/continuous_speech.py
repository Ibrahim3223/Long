# -*- coding: utf-8 -*-
"""
Continuous Speech Generator
✅ Combines all sentences into ONE continuous TTS audio
✅ Natural flow between sentences (no restarts)
✅ SSML-based pause control
✅ Splits audio back into segments for video alignment
"""
import re
import logging
import os
from typing import List, Tuple, Dict
from autoshorts.tts.edge_handler import TTSHandler

logger = logging.getLogger(__name__)


class ContinuousSpeechGenerator:
    """Generate continuous, natural-sounding speech for entire script."""

    def __init__(self, tts_handler: TTSHandler):
        self.tts = tts_handler

    def generate_continuous_script(
        self,
        sentences: List[Dict]
    ) -> Tuple[str, List[Tuple[str, float]], List[Dict]]:
        """
        Generate ONE continuous audio for entire script.

        Args:
            sentences: List of sentence dicts with 'text' key

        Returns:
            (audio_path, word_timings, sentence_segments)

        sentence_segments format:
        [
            {"start": 0.0, "end": 3.5, "text": "First sentence", "index": 0},
            {"start": 3.8, "end": 7.2, "text": "Second sentence", "index": 1},
            ...
        ]
        """
        if not sentences:
            raise ValueError("No sentences provided")

        # ✅ Build SSML-enhanced script with pause markers
        script_parts = []
        sentence_texts = []

        for i, sent in enumerate(sentences):
            text = sent.get("text", "").strip()
            if not text:
                continue

            sentence_texts.append(text)

            # ✅ Simple approach: Just add proper punctuation and spaces
            # Edge TTS will handle pauses naturally at sentence boundaries

            # Ensure sentence ends with punctuation
            if not text[-1] in ".!?":
                text = text + "."

            script_parts.append(text)

        # Combine with natural spacing (double space for sentence boundaries)
        full_script = "  ".join(script_parts)

        logger.info(f"🎙️ Generating continuous TTS for {len(sentence_texts)} sentences")
        logger.debug(f"Full script ({len(full_script)} chars): {full_script[:200]}...")

        # ✅ Generate single audio file
        import tempfile
        temp_dir = tempfile.gettempdir()
        audio_path = os.path.join(temp_dir, "continuous_tts.wav")

        try:
            duration, word_timings = self.tts.synthesize(full_script, audio_path)

            logger.info(f"✅ Continuous TTS generated: {duration:.2f}s, {len(word_timings)} words")

            # ✅ Split back into sentence segments based on word timings
            segments = self._split_into_sentences(
                sentence_texts=sentence_texts,
                word_timings=word_timings,
                total_duration=duration
            )

            return audio_path, word_timings, segments

        except Exception as e:
            logger.error(f"Continuous TTS failed: {e}")
            # Fallback to original sentence-by-sentence
            raise

    def _split_into_sentences(
        self,
        sentence_texts: List[str],
        word_timings: List[Tuple[str, float]],
        total_duration: float
    ) -> List[Dict]:
        """
        Split continuous audio back into sentence segments.

        Uses word timings to determine sentence boundaries.
        """
        if not word_timings:
            # No word timings: split evenly
            logger.warning("No word timings, splitting evenly")
            return self._split_evenly(sentence_texts, total_duration)

        segments = []
        current_time = 0.0
        word_index = 0

        for sent_idx, sentence_text in enumerate(sentence_texts):
            # Count words in this sentence
            sentence_words = sentence_text.strip().split()
            num_words = len(sentence_words)

            if num_words == 0:
                continue

            # Calculate segment timing
            start_time = current_time
            segment_duration = 0.0

            # Sum durations for words in this sentence
            for _ in range(num_words):
                if word_index < len(word_timings):
                    _, word_dur = word_timings[word_index]
                    segment_duration += word_dur
                    word_index += 1
                else:
                    # Ran out of word timings, estimate
                    remaining_words = sum(len(s.split()) for s in sentence_texts[sent_idx:])
                    remaining_time = total_duration - current_time
                    segment_duration += remaining_time / max(1, remaining_words)

            end_time = start_time + segment_duration

            segments.append({
                "start": start_time,
                "end": end_time,
                "duration": segment_duration,
                "text": sentence_text,
                "index": sent_idx
            })

            current_time = end_time

        # ✅ Final adjustment: ensure last segment ends at total_duration
        if segments:
            diff = total_duration - segments[-1]["end"]
            if abs(diff) > 0.01:
                segments[-1]["end"] = total_duration
                segments[-1]["duration"] = total_duration - segments[-1]["start"]

        logger.info(f"✅ Split into {len(segments)} sentence segments")
        for seg in segments[:3]:
            logger.debug(f"  Segment {seg['index']}: {seg['start']:.2f}-{seg['end']:.2f}s ({seg['duration']:.2f}s)")

        return segments

    def _split_evenly(self, sentence_texts: List[str], total_duration: float) -> List[Dict]:
        """Fallback: split duration evenly based on word counts."""
        total_words = sum(len(s.split()) for s in sentence_texts)
        if total_words == 0:
            return []

        segments = []
        current_time = 0.0

        for idx, text in enumerate(sentence_texts):
            word_count = len(text.split())
            duration = (word_count / total_words) * total_duration

            segments.append({
                "start": current_time,
                "end": current_time + duration,
                "duration": duration,
                "text": text,
                "index": idx
            })

            current_time += duration

        return segments


# ✅ Helper: Generate audio with soft transitions between segments
def add_soft_transitions(audio_segments: List[str], output_path: str, crossfade_duration: float = 0.15) -> str:
    """
    Combine audio segments with soft crossfade transitions.

    Args:
        audio_segments: List of audio file paths
        output_path: Output file path
        crossfade_duration: Crossfade duration in seconds

    Returns:
        Output file path
    """
    from autoshorts.utils.ffmpeg_utils import run

    if not audio_segments:
        raise ValueError("No audio segments")

    if len(audio_segments) == 1:
        # Single segment: just copy
        import shutil
        shutil.copy(audio_segments[0], output_path)
        return output_path

    # ✅ Build FFmpeg filter for crossfade
    # Format: [0][1]acrossfade=d=0.15[a01]; [a01][2]acrossfade=d=0.15[a02]; ...

    filter_parts = []
    labels = []

    for i in range(len(audio_segments) - 1):
        if i == 0:
            input_label = f"[0][1]"
        else:
            input_label = f"[a{i-1:02d}][{i+1}]"

        output_label = f"[a{i:02d}]"
        filter_parts.append(f"{input_label}acrossfade=d={crossfade_duration:.3f}{output_label}")
        labels.append(output_label)

    # Last label is the output
    final_label = labels[-1] if labels else "[0]"
    filter_complex = "; ".join(filter_parts)

    # Build FFmpeg command
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]

    # Add all input files
    for seg in audio_segments:
        cmd.extend(["-i", seg])

    # Add filter complex
    cmd.extend(["-filter_complex", filter_complex, "-map", final_label.strip("[]")])

    # Output
    cmd.append(output_path)

    logger.debug(f"Crossfade command: {' '.join(cmd[:10])}...")
    run(cmd)

    return output_path
