"""
Settings module - LONG-FORM VIDEO CONFIGURATION
✅ ULTIMATE VERSION: 4-7 minutes, 40-70 sentences, GUARANTEED
"""

import os
import re
from typing import List


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key, "").strip().lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    return default


def _parse_list(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    try:
        import json
        data = json.loads(s)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
    except Exception:
        pass
    s = re.sub(r'^[\[\(]|\s*[\]\)]$', '', s)
    parts = re.split(r'\s*,\s*', s)
    return [p.strip().strip('"').strip("'") for p in parts if p.strip()]


# ============================================================
# CHANNEL CONFIGURATION
# ============================================================

CHANNEL_NAME = os.getenv("CHANNEL_NAME", "DefaultChannel")

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

if os.getenv("TOPIC"):
    CHANNEL_TOPIC = os.getenv("TOPIC")

CONTENT_STYLE = os.getenv("CONTENT_STYLE", "In-depth educational and engaging")

# ============================================================
# API KEYS
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or ""
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY") or ""
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY") or ""

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
# ✅ ULTIMATE VIDEO SETTINGS - 4-7 MINUTE LONG-FORM
# ============================================================

# Duration: 4-7 minutes (FORCED)
TARGET_DURATION = 360  # 6 minutes
TARGET_MIN_SEC = 240.0  # 4 minutes minimum
TARGET_MAX_SEC = 480.0  # 8 minutes maximum

# Resolution: 16:9 LANDSCAPE
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
ASPECT_RATIO = "16:9"

TARGET_FPS = 30
CRF_VISUAL = 18

VIDEO_MOTION = True
MOTION_INTENSITY = 1.08

SCENE_MIN_DURATION = 8.0
SCENE_MAX_DURATION = 15.0

# ============================================================
# TTS SETTINGS (+ compatibility aliases)
# ============================================================

# Provider hint for TTS stack (some handlers read this)
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "edge")  # edge | google

# Canonical names in this config
TTS_VOICE = os.getenv("TTS_VOICE", "en-US-AriaNeural")  # safe default Edge voice
VOICE = TTS_VOICE
TTS_RATE = os.getenv("TTS_RATE", "+5%")
TTS_PITCH = os.getenv("TTS_PITCH", "+0Hz")
TTS_STYLE = os.getenv("TTS_STYLE", "narration-professional")

# Some TTS handlers expect these Edge-specific names:
EDGE_TTS_VOICE = os.getenv("EDGE_TTS_VOICE", TTS_VOICE)
EDGE_TTS_RATE = os.getenv("EDGE_TTS_RATE", TTS_RATE)
GOOGLE_TTS_LANG = os.getenv("GOOGLE_TTS_LANG", "en-US")

# ============================================================
# PEXELS SETTINGS - OPTIMIZED FOR LONG-FORM
# ============================================================

PEXELS_PER_PAGE = 80
PEXELS_MAX_USES_PER_CLIP = 3  # Allow reuse
PEXELS_ALLOW_REUSE = True
PEXELS_ALLOW_LANDSCAPE = True
PEXELS_MIN_DURATION = 5  # Lower for looping
PEXELS_MAX_DURATION = 20
PEXELS_MIN_HEIGHT = 720
PEXELS_STRICT_VERTICAL = False
PEXELS_MAX_PAGES = 25  # More search depth

ALLOW_PIXABAY_FALLBACK = True
STRICT_ENTITY_FILTER = False

# ============================================================
# ✅ ULTIMATE CAPTION SETTINGS - FORCED ENABLED
# ============================================================

KARAOKE_CAPTIONS = True  # FORCED TRUE
KARAOKE_EFFECTS = True
EFFECT_STYLE = "subtle"

CAPTION_FONT = "Arial"
CAPTION_FONT_SIZE = 48
CAPTION_MAX_LINE = 40
CAPTION_MAX_LINES = 2
CAPTION_POSITION = "bottom"

CAPTION_PRIMARY_COLOR = "&H00FFFFFF"
CAPTION_OUTLINE_COLOR = "&H00000000"
CAPTION_HIGHLIGHT_COLOR = "&H0000FFFF"

KARAOKE_INACTIVE = "#FFFFFF"
KARAOKE_ACTIVE = "#00FFFF"
KARAOKE_OUTLINE = "#000000"

CAPTION_MARGIN_V = 100

# ============================================================
# BGM SETTINGS (+ compatibility aliases)
# ============================================================

# Your original flag
BGM_ENABLE = _env_bool("BGM_ENABLE", True)

# Orchestrator expects this exact name:
BGM_ENABLED = _env_bool("BGM_ENABLED", BGM_ENABLE)

BGM_VOLUME_DB = _env_float("BGM_VOLUME_DB", -28.0)
BGM_DUCK_DB = _env_float("BGM_DUCK_DB", -14.0)
BGM_FADE_DURATION = _env_float("BGM_FADE_DURATION", 1.5)  # seconds
BGM_DIR = os.getenv("BGM_DIR", "bgm")
BGM_URLS = _parse_list(os.getenv("BGM_URLS", ""))

# Extra params some mixers/managers may read
BGM_GAIN_DB = _env_float("BGM_GAIN_DB", -28.0)
BGM_DUCK_THRESH = _env_float("BGM_DUCK_THRESH", 0.09)
BGM_DUCK_RATIO = _env_float("BGM_DUCK_RATIO", 4.0)
BGM_DUCK_ATTACK_MS = _env_float("BGM_DUCK_ATTACK_MS", 20.0)
BGM_DUCK_RELEASE_MS = _env_float("BGM_DUCK_RELEASE_MS", 300.0)
BGM_FADE = _env_float("BGM_FADE", 1.5)

# Aliases for other components (if they exist)
BGM_DUCKING_DB = BGM_DUCK_DB
BGM_FADE_MS = int(BGM_FADE_DURATION * 1000)

# ============================================================
# STATE MANAGEMENT
# ============================================================

STATE_DIR = os.getenv("STATE_DIR", "state")
ENTITY_COOLDOWN_DAYS = 45

NOVELTY_ENFORCE = True
NOVELTY_WINDOW = 30
NOVELTY_JACCARD_MAX = 0.60
NOVELTY_RETRIES = 5

# Optional: some modules check this ENV directly, we mirror a bool here
NOVELTY_EMBEDDINGS = _env_bool("NOVELTY_EMBEDDINGS", False)

# ============================================================
# QUALITY SETTINGS
# ============================================================

MIN_QUALITY_SCORE = 6.5
MAX_GENERATION_ATTEMPTS = 5

# ============================================================
# UPLOAD SETTINGS
# ============================================================

UPLOAD_TO_YT = _env_bool("UPLOAD_TO_YT", True)
VISIBILITY = os.getenv("VISIBILITY", "public")
UPLOAD_AS_SHORTS = False  # FALSE for long-form

ENABLE_CHAPTERS = True
MIN_CHAPTER_DURATION = 30

# ============================================================
# ✅ ULTIMATE CONTENT STRUCTURE - 40-70 SENTENCES (FORCED)
# ============================================================

# Sentence count for 4-7 minutes (FORCED VALUES)
MIN_SENTENCES = 40  # FORCED
MAX_SENTENCES = 70  # FORCED
TARGET_SENTENCES = 55  # FORCED

CHAPTERS_ENABLED = True
MIN_CHAPTER_SENTENCES = 5

# ============================================================
# LANGUAGE
# ============================================================

LANG = os.getenv("LANG", "en")

# ============================================================
# RUNTIME / DEVICE
# ============================================================

TORCH_DEVICE = os.getenv("TORCH_DEVICE", "cpu")

# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "out")

# Create directories
import pathlib
pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(STATE_DIR).mkdir(parents=True, exist_ok=True)
if BGM_ENABLED and BGM_DIR:
    pathlib.Path(BGM_DIR).mkdir(parents=True, exist_ok=True)

# ============================================================
# SYSTEM TYPE
# ============================================================
VIDEO_SYSTEM_TYPE = "LONG_FORM"

# ============================================================
# ✅ STARTUP VALIDATION - Print settings on import
# ============================================================
import logging
_logger = logging.getLogger(__name__)
_logger.info("=" * 60)
_logger.info("LONG-FORM SETTINGS LOADED")
_logger.info(f"Sentences: {MIN_SENTENCES}-{MAX_SENTENCES}")
_logger.info(f"Duration: {TARGET_MIN_SEC/60:.1f}-{TARGET_MAX_SEC/60:.1f} min")
_logger.info(f"Captions: {'ENABLED' if KARAOKE_CAPTIONS else 'DISABLED'}")
_logger.info(f"BGM: {'ENABLED' if BGM_ENABLED else 'DISABLED'} | Dir: {BGM_DIR}")
_logger.info(f"TTS: provider={TTS_PROVIDER}, voice={EDGE_TTS_VOICE}, rate={EDGE_TTS_RATE}")
_logger.info("=" * 60)
