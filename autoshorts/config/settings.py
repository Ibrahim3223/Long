"""
Settings module - LONG-FORM VIDEO CONFIGURATION
Optimized for 3-10 minute landscape videos (16:9)
"""

import os
import re
from typing import List


def _env_int(key: str, default: int) -> int:
    """Get integer from environment variable."""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    """Get float from environment variable."""
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    """Get boolean from environment variable."""
    val = os.getenv(key, "").strip().lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    return default


def _parse_list(s: str) -> List[str]:
    """Parse comma-separated list from string."""
    s = (s or "").strip()
    if not s:
        return []
    # Try JSON first
    try:
        import json
        data = json.loads(s)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
    except Exception:
        pass
    # Fallback to comma-separated
    s = re.sub(r'^[\[\(]|\s*[\]\)]$', '', s)
    parts = re.split(r'\s*,\s*', s)
    return [p.strip().strip('"').strip("'") for p in parts if p.strip()]


# ============================================================
# CHANNEL CONFIGURATION
# ============================================================

CHANNEL_NAME = os.getenv("CHANNEL_NAME", "DefaultChannel")

# Load channel-specific settings from channels_long.yml
try:
    from .channel_loader import apply_channel_settings
    _channel_settings = apply_channel_settings(CHANNEL_NAME)
    CHANNEL_TOPIC = _channel_settings.get("CHANNEL_TOPIC", "Educational content")
    CHANNEL_MODE = _channel_settings.get("CHANNEL_MODE", "educational")
    CHANNEL_SEARCH_TERMS = _channel_settings.get("CHANNEL_SEARCH_TERMS", [])
    CHANNEL_LANG_OVERRIDE = _channel_settings.get("CHANNEL_LANG", None)
    CHANNEL_VISIBILITY_OVERRIDE = _channel_settings.get("CHANNEL_VISIBILITY", None)
except Exception as e:
    import logging
    logging.warning(f"⚠️ Failed to load channel config: {e}")
    CHANNEL_TOPIC = os.getenv("TOPIC", "Educational content")
    CHANNEL_MODE = "educational"
    CHANNEL_SEARCH_TERMS = []
    CHANNEL_LANG_OVERRIDE = None
    CHANNEL_VISIBILITY_OVERRIDE = None

# Allow ENV override if specified
if os.getenv("TOPIC"):
    CHANNEL_TOPIC = os.getenv("TOPIC")

CONTENT_STYLE = os.getenv("CONTENT_STYLE", "In-depth educational and engaging")

# ============================================================
# API KEYS (from environment)
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or ""
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY") or ""
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY") or ""

# YouTube OAuth
YT_CLIENT_ID = os.getenv("YT_CLIENT_ID") or ""
YT_CLIENT_SECRET = os.getenv("YT_CLIENT_SECRET") or ""
YT_REFRESH_TOKEN = os.getenv("YT_REFRESH_TOKEN") or ""

# ============================================================
# GEMINI SETTINGS
# ============================================================

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "flash")
USE_GEMINI = _env_bool("USE_GEMINI", True)
ADDITIONAL_PROMPT_CONTEXT = os.getenv("ADDITIONAL_PROMPT_CONTEXT", "")

# ============================================================
# VIDEO SETTINGS - LONG-FORM LANDSCAPE (16:9)
# ============================================================

# Duration: 3-10 minutes
TARGET_DURATION = _env_int("TARGET_DURATION", 240)  # 4 minutes default
TARGET_MIN_SEC = _env_float("TARGET_MIN_SEC", 180.0)  # 3 minutes minimum
TARGET_MAX_SEC = _env_float("TARGET_MAX_SEC", 600.0)  # 10 minutes maximum

# Resolution: 16:9 LANDSCAPE (1920x1080 Full HD)
VIDEO_WIDTH = _env_int("VIDEO_WIDTH", 1920)
VIDEO_HEIGHT = _env_int("VIDEO_HEIGHT", 1080)
ASPECT_RATIO = "16:9"

TARGET_FPS = _env_int("TARGET_FPS", 30)
CRF_VISUAL = _env_int("CRF_VISUAL", 18)  # Higher quality for longer videos

# Video motion effects (more subtle for long content)
VIDEO_MOTION = _env_bool("VIDEO_MOTION", True)
MOTION_INTENSITY = _env_float("MOTION_INTENSITY", 1.08)  # Subtle zoom (8% max)

# Scene timing (longer scenes for long-form)
SCENE_MIN_DURATION = _env_float("SCENE_MIN_DURATION", 8.0)  # Shorts: 5.0
SCENE_MAX_DURATION = _env_float("SCENE_MAX_DURATION", 15.0)  # Shorts: 10.0

# ============================================================
# TTS SETTINGS - NATURAL PACING FOR LONG-FORM
# ============================================================

TTS_VOICE = os.getenv("TTS_VOICE", "en-US-GuyNeural")
VOICE = TTS_VOICE
TTS_RATE = os.getenv("TTS_RATE", "+5%")  # Slower, more natural (Shorts: +12%)
TTS_PITCH = os.getenv("TTS_PITCH", "+0Hz")
TTS_STYLE = os.getenv("TTS_STYLE", "narration-professional")

# ============================================================
# PEXELS/PIXABAY SETTINGS - MORE REUSE ALLOWED
# ============================================================

PEXELS_PER_PAGE = _env_int("PEXELS_PER_PAGE", 80)
PEXELS_MAX_USES_PER_CLIP = _env_int("PEXELS_MAX_USES_PER_CLIP", 2)  # Allow reuse
PEXELS_ALLOW_REUSE = _env_bool("PEXELS_ALLOW_REUSE", True)  # CRITICAL for 20+ clips
PEXELS_ALLOW_LANDSCAPE = _env_bool("PEXELS_ALLOW_LANDSCAPE", True)  # 16:9 videos
PEXELS_MIN_DURATION = _env_int("PEXELS_MIN_DURATION", 6)
PEXELS_MAX_DURATION = _env_int("PEXELS_MAX_DURATION", 20)
PEXELS_MIN_HEIGHT = _env_int("PEXELS_MIN_HEIGHT", 720)  # Lower requirement
PEXELS_STRICT_VERTICAL = _env_bool("PEXELS_STRICT_VERTICAL", False)  # Allow landscape
PEXELS_MAX_PAGES = _env_int("PEXELS_MAX_PAGES", 15)  # More search depth

ALLOW_PIXABAY_FALLBACK = _env_bool("ALLOW_PIXABAY_FALLBACK", True)

# Entity filtering (less strict for educational content)
STRICT_ENTITY_FILTER = _env_bool("STRICT_ENTITY_FILTER", False)

# ============================================================
# CAPTION SETTINGS - BOTTOM POSITIONED FOR LANDSCAPE
# ============================================================

KARAOKE_CAPTIONS = _env_bool("KARAOKE_CAPTIONS", True)
KARAOKE_EFFECTS = _env_bool("KARAOKE_EFFECTS", True)
EFFECT_STYLE = os.getenv("EFFECT_STYLE", "subtle")  # More subtle for long-form

CAPTION_FONT = os.getenv("CAPTION_FONT", "Arial")
CAPTION_FONT_SIZE = _env_int("CAPTION_FONT_SIZE", 48)  # Smaller for landscape
CAPTION_MAX_LINE = _env_int("CAPTION_MAX_LINE", 40)  # More characters per line
CAPTION_MAX_LINES = _env_int("CAPTION_MAX_LINES", 2)  # Max 2 lines at bottom
CAPTION_POSITION = os.getenv("CAPTION_POSITION", "bottom")  # BOTTOM for landscape

# Karaoke colors (same as shorts)
CAPTION_PRIMARY_COLOR = os.getenv("CAPTION_PRIMARY_COLOR", "&H00FFFFFF")
CAPTION_OUTLINE_COLOR = os.getenv("CAPTION_OUTLINE_COLOR", "&H00000000")
CAPTION_HIGHLIGHT_COLOR = os.getenv("CAPTION_HIGHLIGHT_COLOR", "&H0000FFFF")

KARAOKE_INACTIVE = os.getenv("KARAOKE_INACTIVE", "#FFFFFF")
KARAOKE_ACTIVE = os.getenv("KARAOKE_ACTIVE", "#00FFFF")
KARAOKE_OUTLINE = os.getenv("KARAOKE_OUTLINE", "#000000")

# Bottom margin for landscape captions
CAPTION_MARGIN_V = _env_int("CAPTION_MARGIN_V", 100)  # 100px from bottom

# ============================================================
# BGM SETTINGS
# ============================================================

BGM_ENABLE = _env_bool("BGM_ENABLE", True)
BGM_VOLUME_DB = _env_float("BGM_DB", -28.0)  # Slightly quieter for long-form
BGM_DUCK_DB = _env_float("BGM_DUCK_DB", -14.0)
BGM_FADE_DURATION = _env_float("BGM_FADE", 1.5)  # Longer fades
BGM_DIR = os.getenv("BGM_DIR", "bgm")
BGM_URLS = _parse_list(os.getenv("BGM_URLS", ""))

# Detailed BGM mixing
BGM_GAIN_DB = _env_float("BGM_GAIN_DB", -28.0)
BGM_DUCK_THRESH = _env_float("BGM_DUCK_THRESH", 0.09)
BGM_DUCK_RATIO = _env_float("BGM_DUCK_RATIO", 4.0)
BGM_DUCK_ATTACK_MS = _env_float("BGM_DUCK_ATTACK_MS", 20.0)
BGM_DUCK_RELEASE_MS = _env_float("BGM_DUCK_RELEASE_MS", 300.0)
BGM_FADE = _env_float("BGM_FADE", 1.5)

# ============================================================
# STATE MANAGEMENT
# ============================================================

STATE_DIR = os.getenv("STATE_DIR", "state")
ENTITY_COOLDOWN_DAYS = _env_int("ENTITY_COOLDOWN_DAYS", 45)  # Longer cooldown

# Novelty settings (less strict for educational long-form)
NOVELTY_ENFORCE = _env_bool("NOVELTY_ENFORCE", True)
NOVELTY_WINDOW = _env_int("NOVELTY_WINDOW", 30)  # Last 30 videos
NOVELTY_JACCARD_MAX = _env_float("NOVELTY_JACCARD_MAX", 0.60)  # More lenient
NOVELTY_RETRIES = _env_int("NOVELTY_RETRIES", 5)

# ============================================================
# QUALITY SETTINGS
# ============================================================

# Higher quality threshold for long-form content
MIN_QUALITY_SCORE = _env_float("MIN_QUALITY_SCORE", 6.5)  # Shorts: 5.0
MAX_GENERATION_ATTEMPTS = _env_int("MAX_GENERATION_ATTEMPTS", 5)

# ============================================================
# UPLOAD SETTINGS - NORMAL VIDEO (NOT SHORTS)
# ============================================================

UPLOAD_TO_YT = _env_bool("UPLOAD_TO_YT", True)
VISIBILITY = os.getenv("VISIBILITY", "public")
UPLOAD_AS_SHORTS = _env_bool("UPLOAD_AS_SHORTS", False)  # FALSE for long-form!

# Chapter support
ENABLE_CHAPTERS = _env_bool("ENABLE_CHAPTERS", True)  # Auto-generate chapters
MIN_CHAPTER_DURATION = _env_int("MIN_CHAPTER_DURATION", 30)  # 30 seconds min

# ============================================================
# CONTENT STRUCTURE - LONG-FORM SPECIFIC
# ============================================================

# Sentence count: 20-35 sentences for 3-10 minutes
MIN_SENTENCES = _env_int("MIN_SENTENCES", 20)
MAX_SENTENCES = _env_int("MAX_SENTENCES", 35)
TARGET_SENTENCES = _env_int("TARGET_SENTENCES", 25)

# Chapter structure (auto-generated from content)
CHAPTERS_ENABLED = _env_bool("CHAPTERS_ENABLED", True)
MIN_CHAPTER_SENTENCES = _env_int("MIN_CHAPTER_SENTENCES", 4)  # 4+ sentences per chapter

# ============================================================
# LANGUAGE SETTINGS
# ============================================================

LANG = os.getenv("LANG", "en")

# ============================================================
# OUTPUT SETTINGS
# ============================================================

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "out")

# Create necessary directories
import pathlib
pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(STATE_DIR).mkdir(parents=True, exist_ok=True)
if BGM_ENABLE and BGM_DIR:
    pathlib.Path(BGM_DIR).mkdir(parents=True, exist_ok=True)


# ============================================================
# SYSTEM TYPE MARKER
# ============================================================
VIDEO_SYSTEM_TYPE = "LONG_FORM"  # vs "SHORTS"
