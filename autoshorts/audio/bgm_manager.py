# FILE: autoshorts/audio/bgm_manager.py
# -*- coding: utf-8 -*-
"""
Background music management - PROFESSIONAL AUDIO MIX
(Performance-safe: unchanged processing chain; relies on ffmpeg threading)
"""
import os
import pathlib
import random
import logging
import requests
from typing import Optional

from autoshorts.config import settings
from autoshorts.utils.ffmpeg_utils import run

logger = logging.getLogger(__name__)

VOICE_EQ = {
    "high_pass": 80,
    "cut_200hz": -3,
    "boost_3khz": 4,
    "high_shelf_8khz": 2,
    "de_ess_threshold": -20
}

BGM_EQ = {
    "cut_500_2khz": -6,
    "low_boost_60hz": 2,
    "high_cut_12khz": -2
}

DUCK_SETTINGS = {
    "threshold_db": -25,
    "ratio": 4.0,
    "attack_ms": 10,
    "release_ms": 250,
    "knee_db": 3.0
}


class BGMManager:
    """Manage background music with professional audio processing."""

    def select_track(self) -> Optional[str]:
        """
        Select a BGM track from available sources.

        Returns:
            Path to BGM file or None
        """
        try:
            # Try local BGM directory first
            if hasattr(settings, 'BGM_DIR'):
                p = pathlib.Path(settings.BGM_DIR)
                if p.exists():
                    files = list(p.glob("*.mp3")) + list(p.glob("*.wav"))
                    if files:
                        return str(random.choice(files))

            # Fallback: Return first URL if available
            if hasattr(settings, 'BGM_URLS') and settings.BGM_URLS:
                return settings.BGM_URLS[0]

            return None
        except Exception as e:
            logger.warning(f"BGM track selection failed: {e}")
            return None

    def mix_bgm(
        self,
        video_path: str,
        bgm_track: str,
        output_path: str,
        duration: float,
        adaptive: bool = True,
    ) -> bool:
        """
        Mix BGM with video with optional adaptive processing.

        Args:
            video_path: Input video path
            bgm_track: BGM audio file path or URL
            output_path: Output video path
            duration: Video duration
            adaptive: Use adaptive ducking (default True)

        Returns:
            True if successful
        """
        try:
            work_dir = pathlib.Path(video_path).parent
            voice_track = work_dir / "video_voice.wav"
            bgm_processed = work_dir / "bgm_loop.wav"
            mixed_audio = work_dir / "audio_mixed.wav"

            # Extract voice from video
            run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", video_path,
                "-vn", "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le",
                str(voice_track)
            ])

            if not voice_track.exists():
                logger.warning("Failed to extract voice track")
                return False

            # Process and loop BGM
            self._loop_and_process_bgm(bgm_track, duration, str(bgm_processed))

            if not bgm_processed.exists():
                logger.warning("Failed to process BGM")
                return False

            # Mix voice and BGM
            self._pro_mix(str(voice_track), str(bgm_processed), str(mixed_audio))

            if not mixed_audio.exists():
                logger.warning("Failed to mix audio")
                return False

            # Combine mixed audio with video
            run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", video_path,
                "-i", str(mixed_audio),
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest", "-movflags", "+faststart",
                output_path
            ])

            # Cleanup temp files
            for temp_file in [voice_track, bgm_processed, mixed_audio]:
                try:
                    temp_file.unlink(missing_ok=True)
                except:
                    pass

            return pathlib.Path(output_path).exists()

        except Exception as e:
            logger.error(f"BGM mixing failed: {e}")
            logger.debug("", exc_info=True)
            return False

    def get_bgm(self, duration: float, output_dir: str) -> str:
        if not settings.BGM_ENABLED:
            logger.info("BGM disabled in settings")
            return ""
        try:
            bgm_src = self._pick_source(output_dir)
            if not bgm_src:
                logger.warning("No BGM source found")
                return ""
            bgm_output = os.path.join(output_dir, "bgm_final.wav")
            self._loop_and_process_bgm(bgm_src, duration, bgm_output)
            logger.info(f"BGM prepared: {bgm_output}")
            return bgm_output
        except Exception as e:
            logger.error(f"BGM error: {e}")
            return ""

    def add_bgm(self, voice_path: str, duration: float, temp_dir: str) -> str:
        bgm_src = self._pick_source(temp_dir)
        if not bgm_src:
            return voice_path
        bgm_processed = os.path.join(temp_dir, "bgm_processed.wav")
        self._loop_and_process_bgm(bgm_src, duration, bgm_processed)
        voice_processed = os.path.join(temp_dir, "voice_processed.wav")
        self._process_voice(voice_path, voice_processed)
        output = os.path.join(temp_dir, "audio_with_bgm.wav")
        self._pro_mix(voice_processed, bgm_processed, output)
        return output

    def add_bgm_to_video(self, video_path: str, duration: float, temp_dir: str) -> str:
        if not settings.BGM_ENABLED:
            return video_path

        work_dir = pathlib.Path(temp_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        voice_track = work_dir / "final_voice.wav"
        mixed_audio = work_dir / "audio_with_bgm.wav"
        final_video = work_dir / "video_with_bgm.mp4"
        processed_voice = work_dir / "voice_processed.wav"
        processed_bgm = work_dir / "bgm_processed.wav"

        try:
            run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", video_path,
                "-vn", "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le",
                str(voice_track)
            ])

            if not voice_track.exists():
                return video_path

            mixed = self.add_bgm(str(voice_track), duration, str(work_dir))
            if (not mixed) or (not pathlib.Path(mixed).exists()):
                return video_path

            run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", video_path,
                "-i", mixed,
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest", "-movflags", "+faststart",
                str(final_video)
            ])

            if final_video.exists():
                return str(final_video)
            return video_path

        except Exception:
            return video_path
        finally:
            for path in (voice_track, processed_voice, processed_bgm, mixed_audio):
                try:
                    pathlib.Path(path).unlink(missing_ok=True)
                except Exception:
                    pass

    def _pick_source(self, temp_dir: str) -> str:
        try:
            p = pathlib.Path(settings.BGM_DIR)
            if p.exists():
                files = list(p.glob("*.mp3")) + list(p.glob("*.wav"))
                if files:
                    return str(random.choice(files))
        except Exception:
            pass

        if settings.BGM_URLS:
            for url in settings.BGM_URLS:
                try:
                    ext = ".mp3" if ".mp3" in url.lower() else ".wav"
                    out_path = os.path.join(temp_dir, f"bgm_src{ext}")
                    with requests.get(url, stream=True, timeout=45) as r:
                        r.raise_for_status()
                        with open(out_path, "wb") as f:
                            for chunk in r.iter_content(64 * 1024):
                                if chunk:
                                    f.write(chunk)
                    if os.path.getsize(out_path) > 100_000:
                        return out_path
                except Exception:
                    continue
        return ""

    def _loop_and_process_bgm(self, src: str, duration: float, output: str):
        fade = settings.BGM_FADE
        endst = max(0.0, duration - fade)
        eq_filter = (
            f"equalizer=f=500:width_type=o:width=2.5:g={BGM_EQ['cut_500_2khz']},"
            f"equalizer=f=60:width_type=o:width=1:g={BGM_EQ['low_boost_60hz']},"
            f"equalizer=f=12000:width_type=o:width=1:g={BGM_EQ['high_cut_12khz']}"
        )
        audio_filter = (
            f"{eq_filter},"
            f"loudnorm=I=-21:TP=-2.0:LRA=11,"
            f"afade=t=in:st=0:d={fade:.2f},afade=t=out:st={endst:.2f}:d={fade:.2f},"
            f"aresample=48000,pan=mono|c0=0.5*FL+0.5*FR"
        )
        run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-stream_loop", "-1", "-i", src,
            "-t", f"{duration:.3f}",
            "-af", audio_filter,
            "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le",
            output
        ])

    def _process_voice(self, voice_in: str, voice_out: str):
        eq_filter = (
            f"highpass=f={VOICE_EQ['high_pass']},"
            f"equalizer=f=200:width_type=o:width=1:g={VOICE_EQ['cut_200hz']},"
            f"equalizer=f=3000:width_type=o:width=1.5:g={VOICE_EQ['boost_3khz']},"
            f"equalizer=f=8000:width_type=h:width=1000:g={VOICE_EQ['high_shelf_8khz']}"
        )
        deess_filter = f"deesser=i={VOICE_EQ['de_ess_threshold']}:m=0.5:f=0.5:s=o"
        comp_filter = "acompressor=threshold=-18dB:ratio=3:attack=5:release=50:makeup=2"
        audio_filter = f"{eq_filter},{deess_filter},{comp_filter}"
        run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", voice_in,
            "-af", audio_filter,
            "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le",
            voice_out
        ])

    def _pro_mix(self, voice: str, bgm: str, output: str):
        gain_db = settings.BGM_GAIN_DB
        thresh = DUCK_SETTINGS["threshold_db"]
        ratio = DUCK_SETTINGS["ratio"]
        attack = DUCK_SETTINGS["attack_ms"]
        release = DUCK_SETTINGS["release_ms"]
        knee = DUCK_SETTINGS["knee_db"]

        sidechain_filter = (
            f"sidechaincompress=threshold={thresh}dB:ratio={ratio}:"
            f"attack={attack}:release={release}:knee={knee}:makeup=1.0"
        )
        filter_complex = (
            f"[1:a]volume={gain_db}dB[bgm];"
            f"[bgm][0:a]{sidechain_filter}[bgm_ducked];"
            f"[0:a][bgm_ducked]amix=inputs=2:duration=shortest:weights=1.0 0.7,aresample=48000[mix]"
        )
        run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", voice,
            "-i", bgm,
            "-filter_complex", filter_complex,
            "-map", "[mix]",
            "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le",
            output
        ])
