# FILE: autoshorts/captions/karaoke_ass.py
# -*- coding: utf-8 -*-
"""
Karaoke ASS subtitle builder – long-form, 16:9 landscape.
- Weighted colorful styles
- Bottom-centered alignment
- Safe text escaping for ASS
"""
import random
from typing import List, Dict, Optional, Any, Tuple

# ------------------------- Helpers ------------------------- #

def _ass_escape(text: str) -> str:
    """Escape special chars for ASS override blocks and text."""
    # Order matters: escape backslashes first
    t = (text or "").replace("\\", r"\\")
    t = t.replace("{", r"\{").replace("}", r"\}")
    # Newlines -> ASS hard line breaks
    t = t.replace("\n", r"\N")
    return t


# ------------------------- Styles -------------------------- #
# NOTE: ASS colors are in &H AABBGGRR format (BBGGRR with alpha).

CAPTION_STYLES = {
    # 1) Classic Yellow
    "classic_yellow": {
        "name": "Classic Yellow",
        "fontname": "Arial Black",
        "fontsize_normal": 44,
        "fontsize_hook": 50,
        "fontsize_emphasis": 48,
        "outline": 5,
        "shadow": 3,
        "glow": True,
        "bounce": True,
        "color_inactive": "&H00FFFFFF",  # White
        "color_active": "&H0000FFFF",    # Yellow
        "color_outline": "&H00000000",   # Black
        "color_emphasis": "&H0000FFFF",  # Yellow
        "color_secondary": "&H0000DDFF",
        "margin_v": 90
    },

    # 2) Neon Cyan
    "neon_cyan": {
        "name": "Neon Cyan",
        "fontname": "Arial Black",
        "fontsize_normal": 42,
        "fontsize_hook": 48,
        "fontsize_emphasis": 46,
        "outline": 4,
        "shadow": 3,
        "glow": True,
        "bounce": True,
        "color_inactive": "&H00FFFFFF",
        "color_active": "&H00FFFF00",    # Cyan
        "color_outline": "&H00000000",
        "color_emphasis": "&H00FFFF00",
        "color_secondary": "&H00AAFF00",
        "margin_v": 90
    },

    # 3) Hot Pink (fixed BBGGRR ordering)
    "hot_pink": {
        "name": "Hot Pink",
        "fontname": "Impact",
        "fontsize_normal": 42,
        "fontsize_hook": 48,
        "fontsize_emphasis": 46,
        "outline": 4,
        "shadow": 3,
        "glow": True,
        "bounce": True,
        # DeepPink (#FF1493) => &H009314FF
        "color_inactive": "&H00FFFFFF",
        "color_active": "&H009314FF",
        "color_outline": "&H00000000",
        # HotPink (#FF69B4) => &H00B469FF
        "color_emphasis": "&H00B469FF",
        "color_secondary": "&H00B469FF",
        "margin_v": 90
    },

    # 4) Lime Green
    "lime_green": {
        "name": "Lime Green",
        "fontname": "Arial Black",
        "fontsize_normal": 44,
        "fontsize_hook": 50,
        "fontsize_emphasis": 48,
        "outline": 5,
        "shadow": 3,
        "glow": True,
        "bounce": True,
        "color_inactive": "&H00FFFFFF",
        "color_active": "&H0000FF00",
        "color_outline": "&H00000000",
        "color_emphasis": "&H0000FF00",
        "color_secondary": "&H0000DD00",
        "margin_v": 95
    },

    # 5) Orange Fire
    "orange_fire": {
        "name": "Orange Fire",
        "fontname": "Impact",
        "fontsize_normal": 44,
        "fontsize_hook": 50,
        "fontsize_emphasis": 48,
        "outline": 5,
        "shadow": 3,
        "glow": True,
        "bounce": True,
        "color_inactive": "&H00FFFFFF",
        # Orange (#FF9900) => &H000099FF
        "color_active": "&H000099FF",
        "color_outline": "&H00000000",
        "color_emphasis": "&H0000BBFF",
        "color_secondary": "&H0000BBFF",
        "margin_v": 95
    },

    # 6) Purple Vibes
    "purple_vibes": {
        "name": "Purple Vibes",
        "fontname": "Arial Black",
        "fontsize_normal": 42,
        "fontsize_hook": 48,
        "fontsize_emphasis": 46,
        "outline": 4,
        "shadow": 3,
        "glow": True,
        "bounce": True,
        "color_inactive": "&H00FFFFFF",
        "color_active": "&H00FF00FF",  # Magenta
        "color_outline": "&H00000000",
        "color_emphasis": "&H00FF00FF",
        "color_secondary": "&H00CC00FF",
        "margin_v": 90
    },

    # 7) Turquoise Wave
    "turquoise_wave": {
        "name": "Turquoise Wave",
        "fontname": "Arial Black",
        "fontsize_normal": 42,
        "fontsize_hook": 48,
        "fontsize_emphasis": 46,
        "outline": 4,
        "shadow": 3,
        "glow": True,
        "bounce": True,
        "color_inactive": "&H00FFFFFF",
        # Turquoise (#00CCFF) => &H00FFCC00
        "color_active": "&H00FFCC00",
        "color_outline": "&H00000000",
        # Emphasis as bright yellow for pop
        "color_emphasis": "&H00FFFF00",
        "color_secondary": "&H00FFDD00",
        "margin_v": 90
    },

    # 8) Red Hot
    "red_hot": {
        "name": "Red Hot",
        "fontname": "Impact",
        "fontsize_normal": 44,
        "fontsize_hook": 50,
        "fontsize_emphasis": 48,
        "outline": 5,
        "shadow": 3,
        "glow": True,
        "bounce": True,
        "color_inactive": "&H00FFFFFF",
        "color_active": "&H000000FF",  # Red
        "color_outline": "&H00000000",
        "color_emphasis": "&H000033FF",
        "color_secondary": "&H000033FF",
        "margin_v": 95
    },
}

STYLE_WEIGHTS = {
    "classic_yellow": 0.25,
    "neon_cyan": 0.15,
    "hot_pink": 0.12,
    "lime_green": 0.12,
    "orange_fire": 0.12,
    "purple_vibes": 0.10,
    "turquoise_wave": 0.08,
    "red_hot": 0.06,
}

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
    "MILLIONS", "THOUSANDS", "BILLION",
}


def get_random_style() -> str:
    return random.choices(
        list(STYLE_WEIGHTS.keys()),
        weights=list(STYLE_WEIGHTS.values()),
        k=1,
    )[0]


def get_style_info(style_name: str) -> Dict[str, Any]:
    return CAPTION_STYLES.get(style_name, CAPTION_STYLES["classic_yellow"])


def list_all_styles() -> List[str]:
    return list(CAPTION_STYLES.keys())


def _format_ass_time(seconds: float) -> str:
    """Format seconds to ASS timestamp (H:MM:SS.CS)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centiseconds = int(round((seconds % 1) * 100))
    return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"


def build_karaoke_ass(
    text: str,
    seg_dur: float,
    words: List[Tuple[str, float]],
    is_hook: bool = False,
    style_name: Optional[str] = None,
) -> str:
    """Build a complete ASS file with colorful karaoke captions."""
    if not style_name:
        style_name = get_random_style()
    style = get_style_info(style_name)

    fontsize = style["fontsize_hook"] if is_hook else style["fontsize_normal"]

    # Header
    ass = [
        "[Script Info]",
        "Title: Karaoke Captions",
        "ScriptType: v4.00+",
        "PlayResX: 1920",
        "PlayResY: 1080",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{style['fontname']},{fontsize},{style['color_inactive']},{style['color_active']},"
        f"{style['color_outline']},&H80000000,-1,0,0,0,100,100,1,0,1,{style['outline']},{style['shadow']},2,50,50,{style['margin_v']},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    if not words:
        end_time = _format_ass_time(max(0.01, float(seg_dur)))
        ass.append(f"Dialogue: 0,0:00:00.00,{end_time},Default,,0,0,0,,{_ass_escape(text).upper()}")
        return "\n".join(ass) + "\n"

    # Group words into small chunks for readability
    chunk_size = 2 if is_hook else 3
    chunks: List[Tuple[List[Tuple[str, float]], float]] = []
    buf: List[Tuple[str, float]] = []
    buf_dur = 0.0
    for w, d in words:
        buf.append((w, d))
        buf_dur += float(d)
        if len(buf) >= chunk_size:
            chunks.append((buf, buf_dur))
            buf, buf_dur = [], 0.0
    if buf:
        chunks.append((buf, buf_dur))

    t = 0.0
    for chunk_words, chunk_duration in chunks:
        start_time = _format_ass_time(t)
        end_time = _format_ass_time(min(seg_dur, t + chunk_duration))

        # Build karaoke tagged text
        parts = []
        for w, d in chunk_words:
            wtxt = _ass_escape(w).upper()
            dur_cs = max(1, int(round(float(d) * 100)))  # \k in centiseconds
            if wtxt.strip(".,!?;:") in EMPHASIS_KEYWORDS:
                parts.append(f"{{\\k{dur_cs}\\fs{style['fontsize_emphasis']}\\c{style['color_emphasis']}}}{wtxt} ")
            else:
                parts.append(f"{{\\k{dur_cs}}}{wtxt} ")

        line_text = "".join(parts).strip()

        # Subtle bounce effect
        if style.get("bounce", False):
            line_text = f"{{\\move(960,1000,960,980,0,{int(chunk_duration*1000)})}}{line_text}"

        ass.append(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{line_text}")
        t += chunk_duration
        if t >= seg_dur:
            break

    return "\n".join(ass) + "\n"
