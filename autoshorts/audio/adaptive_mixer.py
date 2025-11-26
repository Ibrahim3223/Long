# -*- coding: utf-8 -*-
"""
Adaptive Audio Mixer - Context-Aware Audio Processing
✅ Adaptive ducking based on content importance
✅ Dynamic BGM volume based on sentence type
✅ Scene normalization for consistent levels
✅ Smooth fade transitions
"""
import os
import logging
from typing import Dict, Optional
from pathlib import Path

from autoshorts.utils.ffmpeg_utils import run
from autoshorts.config import settings

logger = logging.getLogger(__name__)


# Audio processing profiles for different content types
AUDIO_PROFILES = {
    "hook": {
        "bgm_gain_db": -18,  # Louder BGM for excitement
        "duck_threshold_db": -20,  # More aggressive ducking
        "duck_ratio": 5.0,
        "voice_boost_db": 2,  # Boost voice for clarity
        "description": "Energetic mix for hooks",
    },
    "content": {
        "bgm_gain_db": -22,  # Moderate BGM
        "duck_threshold_db": -25,  # Standard ducking
        "duck_ratio": 4.0,
        "voice_boost_db": 0,  # Neutral voice
        "description": "Balanced mix for content",
    },
    "cta": {
        "bgm_gain_db": -26,  # Quiet BGM for clarity
        "duck_threshold_db": -30,  # Gentle ducking
        "duck_ratio": 3.0,
        "voice_boost_db": 3,  # Boost voice for emphasis
        "description": "Voice-focused mix for CTA",
    },
    "important": {
        "bgm_gain_db": -28,  # Very quiet BGM
        "duck_threshold_db": -35,  # Minimal ducking
        "duck_ratio": 2.0,
        "voice_boost_db": 4,  # Strong voice boost
        "description": "Voice-dominant mix for key points",
    },
}


class AdaptiveAudioMixer:
    """Enhanced audio mixer with adaptive processing."""

    def __init__(self):
        """Initialize adaptive mixer."""
        self.scene_levels: list = []  # Track audio levels for normalization

    def mix_scene_audio(
        self,
        voice_path: str,
        bgm_path: str,
        output_path: str,
        sentence_type: str = "content",
        duration: float = 5.0,
        is_important: bool = False,
    ) -> bool:
        """
        Mix voice and BGM with adaptive processing.

        Args:
            voice_path: Path to voice audio
            bgm_path: Path to background music
            output_path: Output path for mixed audio
            sentence_type: Type of sentence (hook, content, cta)
            duration: Audio duration in seconds
            is_important: Whether this is a key point (extra voice emphasis)

        Returns:
            True if successful
        """
        try:
            # Select audio profile
            if is_important:
                profile = AUDIO_PROFILES["important"]
            else:
                profile = AUDIO_PROFILES.get(sentence_type, AUDIO_PROFILES["content"])

            logger.debug(f"Using audio profile: {profile['description']}")

            # Extract profile settings
            bgm_gain_db = profile["bgm_gain_db"]
            duck_threshold = profile["duck_threshold_db"]
            duck_ratio = profile["duck_ratio"]
            voice_boost = profile["voice_boost_db"]

            # Build filter complex for adaptive mixing
            filter_complex = self._build_adaptive_filter(
                bgm_gain_db=bgm_gain_db,
                duck_threshold=duck_threshold,
                duck_ratio=duck_ratio,
                voice_boost=voice_boost,
                duration=duration,
            )

            # Run FFmpeg with adaptive filter
            run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", voice_path,
                "-i", bgm_path,
                "-filter_complex", filter_complex,
                "-map", "[out]",
                "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le",
                output_path
            ])

            return os.path.exists(output_path)

        except Exception as exc:
            logger.error(f"Adaptive mixing failed: {exc}")
            logger.debug("", exc_info=True)
            return False

    def _build_adaptive_filter(
        self,
        bgm_gain_db: float,
        duck_threshold: float,
        duck_ratio: float,
        voice_boost: float,
        duration: float,
    ) -> str:
        """
        Build FFmpeg filter complex for adaptive audio mixing.

        Args:
            bgm_gain_db: BGM volume adjustment
            duck_threshold: Ducking threshold
            duck_ratio: Ducking ratio
            voice_boost: Voice volume boost
            duration: Audio duration

        Returns:
            FFmpeg filter_complex string
        """
        fade_duration = min(0.5, duration * 0.1)  # 10% fade, max 0.5s

        # Voice processing
        voice_filter = (
            # EQ for clarity
            f"highpass=f=80,"
            f"equalizer=f=200:width_type=o:width=1:g=-2,"
            f"equalizer=f=3000:width_type=o:width=1.5:g=3,"
            # Compression for consistency
            f"acompressor=threshold=-18dB:ratio=3:attack=5:release=50:makeup=2,"
            # Volume boost
            f"volume={voice_boost}dB,"
            # Normalization
            f"loudnorm=I=-16:TP=-1.5:LRA=11"
        )

        # BGM processing
        bgm_filter = (
            # EQ to make space for voice
            f"equalizer=f=500:width_type=o:width=2.5:g=-6,"  # Cut mids
            f"equalizer=f=60:width_type=o:width=1:g=2,"      # Boost lows
            # Volume adjustment
            f"volume={bgm_gain_db}dB,"
            # Fades
            f"afade=t=in:st=0:d={fade_duration:.2f},"
            f"afade=t=out:st={duration - fade_duration:.2f}:d={fade_duration:.2f}"
        )

        # Sidechain ducking settings
        duck_attack = 10  # ms - fast attack for quick ducking
        duck_release = 250  # ms - slower release for smooth recovery
        duck_knee = 3.0  # dB - soft knee for natural sound

        # Build complete filter
        filter_complex = (
            # Process voice
            f"[0:a]{voice_filter}[voice];"
            # Process BGM
            f"[1:a]{bgm_filter}[bgm];"
            # Sidechain compression (ducking)
            f"[bgm][voice]sidechaincompress="
            f"threshold={duck_threshold}dB:"
            f"ratio={duck_ratio}:"
            f"attack={duck_attack}:"
            f"release={duck_release}:"
            f"knee={duck_knee}:"
            f"makeup=1.0"
            f"[bgm_ducked];"
            # Mix voice and ducked BGM
            f"[voice][bgm_ducked]amix=inputs=2:duration=shortest:weights=1.0 0.7[out]"
        )

        return filter_complex

    def normalize_scene_audio(
        self,
        audio_path: str,
        output_path: str,
        target_lufs: float = -16.0,
    ) -> bool:
        """
        Normalize audio to target loudness.

        Args:
            audio_path: Input audio path
            output_path: Output audio path
            target_lufs: Target LUFS loudness

        Returns:
            True if successful
        """
        try:
            run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", audio_path,
                "-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11",
                "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le",
                output_path
            ])

            return os.path.exists(output_path)

        except Exception as exc:
            logger.error(f"Audio normalization failed: {exc}")
            return False

    def add_crossfade_transition(
        self,
        audio1_path: str,
        audio2_path: str,
        output_path: str,
        crossfade_duration: float = 0.5,
    ) -> bool:
        """
        Add smooth crossfade between two audio files.

        Args:
            audio1_path: First audio file
            audio2_path: Second audio file
            output_path: Output path
            crossfade_duration: Crossfade duration in seconds

        Returns:
            True if successful
        """
        try:
            filter_complex = (
                f"[0:a][1:a]acrossfade=d={crossfade_duration:.2f}:c1=tri:c2=tri[out]"
            )

            run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", audio1_path,
                "-i", audio2_path,
                "-filter_complex", filter_complex,
                "-map", "[out]",
                "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le",
                output_path
            ])

            return os.path.exists(output_path)

        except Exception as exc:
            logger.error(f"Crossfade failed: {exc}")
            return False

    def get_audio_profile_for_sentence(
        self, sentence_type: str, has_numbers: bool = False, has_power_words: bool = False
    ) -> Dict:
        """
        Get recommended audio profile for a sentence.

        Args:
            sentence_type: Type of sentence (hook, content, cta)
            has_numbers: Whether sentence contains numbers (likely important)
            has_power_words: Whether sentence contains power words

        Returns:
            Audio profile dictionary
        """
        # Mark as important if contains key indicators
        is_important = has_numbers or has_power_words

        if is_important:
            return AUDIO_PROFILES["important"]
        else:
            return AUDIO_PROFILES.get(sentence_type, AUDIO_PROFILES["content"])

    def reset_normalization(self):
        """Reset scene level tracking (for new video)."""
        self.scene_levels = []
