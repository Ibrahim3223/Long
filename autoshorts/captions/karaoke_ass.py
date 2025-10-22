# -*- coding: utf-8 -*-
"""
Karaoke ASS subtitle builder - LONG-FORM VERSION (16:9 LANDSCAPE)
Bottom-positioned captions with smaller fonts for landscape videos
"""
import random
from typing import List, Dict, Optional, Any, Tuple


# ============================================================================
# LANDSCAPE CAPTION STYLES - Bottom positioned for 16:9
# Font sizes reduced, margin_v adjusted for bottom positioning
# ============================================================================

CAPTION_STYLES = {
    # Style 1: CLASSIC YELLOW - Safe and proven
    "classic_yellow": {
        "name": "Classic Yellow",
        "fontname": "Arial Black",
        "fontsize_normal": 44,
        "fontsize_hook": 50,
        "fontsize_emphasis": 48,
        "outline": 5,
        "shadow": "3",
        "glow": True,
        "bounce": True,
        "color_inactive": "&H00FFFFFF",  # White
        "color_active": "&H0000FFFF",    # Yellow
        "color_outline": "&H00000000",   # Black
        "color_emphasis": "&H0000FFFF",  # Yellow
        "color_secondary": "&H0000DDFF", # Light yellow
        "margin_v": 90  # Bottom positioning
    },
    
    # Style 2: NEON CYAN - Electric energy
    "neon_cyan": {
        "name": "Neon Cyan",
        "fontname": "Arial Black",
        "fontsize_normal": 42,
        "fontsize_hook": 48,
        "fontsize_emphasis": 46,
        "outline": 4,
        "shadow": "3",
        "glow": True,
        "bounce": True,
        "color_inactive": "&H00FFFFFF",
        "color_active": "&H00FFFF00",    # Cyan
        "color_outline": "&H00000000",
        "color_emphasis": "&H0000FFFF",
        "color_secondary": "&H00FFAA00",
        "margin_v": 90
    },
    
    # Style 3: HOT PINK - Bold and attention-grabbing
    "hot_pink": {
        "name": "Hot Pink",
        "fontname": "Impact",
        "fontsize_normal": 42,
        "fontsize_hook": 48,
        "fontsize_emphasis": 46,
        "outline": 4,
        "shadow": "3",
        "glow": True,
        "bounce": True,
        "color_inactive": "&H00FFFFFF",
        "color_active": "&H00FF1493",    # Deep pink
        "color_outline": "&H00000000",
        "color_emphasis": "&H0000FFFF",
        "color_secondary": "&H00FF69B4",
        "margin_v": 90
    },
    
    # Style 4: LIME GREEN - Fresh and vibrant
    "lime_green": {
        "name": "Lime Green",
        "fontname": "Arial Black",
        "fontsize_normal": 44,
        "fontsize_hook": 50,
        "fontsize_emphasis": 48,
        "outline": 5,
        "shadow": "3",
        "glow": True,
        "bounce": True,
        "color_inactive": "&H00FFFFFF",
        "color_active": "&H0000FF00",    # Lime
        "color_outline": "&H00000000",
        "color_emphasis": "&H0000FFFF",
        "color_secondary": "&H0000DD00",
        "margin_v": 95
    },
    
    # Style 5: ORANGE FIRE - Warm and energetic
    "orange_fire": {
        "name": "Orange Fire",
        "fontname": "Impact",
        "fontsize_normal": 44,
        "fontsize_hook": 50,
        "fontsize_emphasis": 48,
        "outline": 5,
        "shadow": "3",
        "glow": True,
        "bounce": True,
        "color_inactive": "&H00FFFFFF",
        "color_active": "&H000099FF",    # Orange
        "color_outline": "&H00000000",
        "color_emphasis": "&H0000FFFF",
        "color_secondary": "&H0000BBFF",
        "margin_v": 95
    },
    
    # Style 6: PURPLE VIBES - Trendy and modern
    "purple_vibes": {
        "name": "Purple Vibes",
        "fontname": "Montserrat Black",
        "fontsize_normal": 42,
        "fontsize_hook": 48,
        "fontsize_emphasis": 46,
        "outline": 4,
        "shadow": "3",
        "glow": True,
        "bounce": True,
        "color_inactive": "&H00FFFFFF",
        "color_active": "&H00FF00FF",    # Magenta
        "color_outline": "&H00000000",
        "color_emphasis": "&H00FF00FF",
        "color_secondary": "&H00CC00FF",
        "margin_v": 90
    },
    
    # Style 7: TURQUOISE WAVE - Cool and calming
    "turquoise_wave": {
        "name": "Turquoise Wave",
        "fontname": "Arial Black",
        "fontsize_normal": 42,
        "fontsize_hook": 48,
        "fontsize_emphasis": 46,
        "outline": 4,
        "shadow": "3",
        "glow": True,
        "bounce": True,
        "color_inactive": "&H00FFFFFF",
        "color_active": "&H00FFCC00",    # Turquoise
        "color_outline": "&H00000000",
        "color_emphasis": "&H00FFFF00",
        "color_secondary": "&H00FFDD00",
        "margin_v": 90
    },
    
    # Style 8: RED HOT - Intense and dramatic
    "red_hot": {
        "name": "Red Hot",
        "fontname": "Impact",
        "fontsize_normal": 44,
        "fontsize_hook": 50,
        "fontsize_emphasis": 48,
        "outline": 5,
        "shadow": "3",
        "glow": True,
        "bounce": True,
        "color_inactive": "&H00FFFFFF",
        "color_active": "&H000000FF",    # Red
        "color_outline": "&H00000000",
        "color_emphasis": "&H0000FFFF",
        "color_secondary": "&H000033FF",
        "margin_v": 95
    }
}

# Same weights as shorts
STYLE_WEIGHTS = {
    "classic_yellow": 0.25,
    "neon_cyan": 0.15,
    "hot_pink": 0.12,
    "lime_green": 0.12,
    "orange_fire": 0.12,
    "purple_vibes": 0.10,
    "turquoise_wave": 0.08,
    "red_hot": 0.06
}

# Same emphasis keywords
EMPHASIS_KEYWORDS = {
    "NEVER", "ALWAYS", "IMPOSSIBLE", "INSANE", "CRAZY", "SHOCKING",
    "UNBELIEVABLE", "INCREDIBLE", "AMAZING", "STUNNING", "MIND-BLOWING",
    "NOW", "IMMEDIATELY", "INSTANTLY", "URGENT", "BREAKING", "ALERT",
    "STOP", "WAIT", "ATTENTION", "WARNING", "DANGER",
    "SECRET", "HIDDEN", "BANNED", "ILLEGAL", "FORBIDDEN", "RARE",
    "EXCLUSIVE", "LIMITED", "FIRST", "LAST", "ONLY", "UNIQUE",
    "BEST", "WORST", "BIGGEST", "SMALLEST", "FASTEST", "SLOWEST",
    "MOST", "LEAST", "ULTIMATE", "SUPREME", "MAXIMUM",
    "VIRAL", "TRENDING", "POPULAR", "FAMOUS", "EVERYONE", "NOBODY",
    "MILLIONS", "THOUSANDS", "BILLION"
}


def get_random_style() -> str:
    """Get a random caption style based on weights."""
    return random.choices(
        list(STYLE_WEIGHTS.keys()),
        weights=list(STYLE_WEIGHTS.values())
    )[0]


def get_style_info(style_name: str) -> Dict[str, Any]:
    """Get information about a specific style."""
    if style_name in CAPTION_STYLES:
        return CAPTION_STYLES[style_name]
    return CAPTION_STYLES["classic_yellow"]


def list_all_styles() -> List[str]:
    """List all available caption styles."""
    return list(CAPTION_STYLES.keys())


# Backward compatibility
def build_karaoke_ass(
    text: str,
    seg_dur: float,
    words: List[tuple],
    is_hook: bool = False,
    style_name: Optional[str] = None
) -> str:
    """Legacy function for backward compatibility."""
    return """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial Black,44,&H00FFFFFF,&H0000FFFF,&H00000000,&H00000000,-1,0,0,0,100,100,1,0,1,5,3,2,50,50,90,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:10.00,Default,,0,0,0,,{text}
"""
