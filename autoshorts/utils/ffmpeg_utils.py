# FILE: autoshorts/utils/ffmpeg_utils.py
# -*- coding: utf-8 -*-
"""
FFmpeg utilities: probe, check filters, run commands.
Optimized to auto-inject multi-threading flags into ffmpeg commands.
"""
import os
import re
import subprocess
import pathlib
from typing import Optional, Tuple

# Determine sensible default thread counts
_CPU_COUNT = max(1, os.cpu_count() or 1)
_DEFAULT_THREADS = max(2, _CPU_COUNT - 1)
FFMPEG_THREADS = int(os.getenv("FFMPEG_THREADS", str(_DEFAULT_THREADS)))
FILTER_THREADS = int(os.getenv("FFMPEG_FILTER_THREADS", str(max(1, FFMPEG_THREADS // 2))))

# Ensure BLAS/OpenMP libs don't oversubscribe
os.environ.setdefault("OMP_NUM_THREADS", str(FFMPEG_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def _inject_threads(cmd: list) -> list:
    """If this is an ffmpeg command, ensure -threads/-filter_threads are present."""
    if not cmd:
        return cmd
    exe = pathlib.Path(cmd[0]).name.lower()
    if exe != "ffmpeg":
        return cmd

    # Don't duplicate flags
    if "-threads" not in cmd:
        cmd += ["-threads", str(FFMPEG_THREADS)]
    if "-filter_threads" not in cmd:
        cmd += ["-filter_threads", str(FILTER_THREADS)]
    return cmd


def run(cmd, check=True) -> subprocess.CompletedProcess:
    """Execute command and return result (with automatic ffmpeg threading flags)."""
    cmd = _inject_threads(list(cmd))
    res = subprocess.run(cmd, text=True, capture_output=True)
    if check and res.returncode != 0:
        raise RuntimeError(res.stderr[:4000] or res.stdout[:4000])
    return res


def ffprobe_duration(path: str) -> float:
    """Get video/audio duration in seconds."""
    try:
        out = run([
            "ffprobe", "-v", "quiet", "-show_entries",
            "format=duration", "-of", "csv=p=0", path
        ]).stdout.strip()
        return float(out) if out else 0.0
    except Exception:
        return 0.0


def ffmpeg_has_filter(name: str) -> bool:
    """Check if FFmpeg has a specific filter."""
    try:
        out = run(["ffmpeg", "-hide_banner", "-filters"], check=False).stdout
        return bool(re.search(rf"\b{name}\b", out))
    except Exception:
        return False


def font_path() -> str:
    """Find system font path."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/arial.ttf"
    ]
    for p in candidates:
        if pathlib.Path(p).exists():
            return p
    return ""


def sanitize_font_path(font_path_str: str) -> str:
    """Escape font path for FFmpeg."""
    if not font_path_str:
        return ""
    return font_path_str.replace(":", r"\:").replace(",", r"\,").replace("\\", "/")


def quantize_to_frames(seconds: float, fps: int = 25) -> Tuple[int, float]:
    """Convert seconds to exact frame count and back to seconds."""
    frames = max(2, int(round(seconds * fps)))
    return frames, frames / float(fps)


# Cache filter availability
_HAS_DRAWTEXT = None
_HAS_SUBTITLES = None


def has_drawtext() -> bool:
    """Check if drawtext filter is available (cached)."""
    global _HAS_DRAWTEXT
    if _HAS_DRAWTEXT is None:
        _HAS_DRAWTEXT = ffmpeg_has_filter("drawtext")
    return _HAS_DRAWTEXT


def has_subtitles() -> bool:
    """Check if subtitles filter is available (cached)."""
    global _HAS_SUBTITLES
    if _HAS_SUBTITLES is None:
        _HAS_SUBTITLES = ffmpeg_has_filter("subtitles")
    return _HAS_SUBTITLES
