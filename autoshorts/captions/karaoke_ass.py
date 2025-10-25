# -*- coding: utf-8 -*-
"""
Karaoke ASS subtitle builder - LONG-FORM (16:9)
- Varsayılan: vurgu KAPALI, hareket/animasyon KAPALI
- Alt yazılar altta, okunaklı ve sabit konumda
"""
import random
from typing import List, Dict, Optional, Any, Tuple

# =========================
# LANDSCAPE CAPTION STYLES
# =========================
CAPTION_STYLES = {
    "classic_yellow": {
        "name": "Classic Yellow",
        "fontname": "Arial Black",
        "fontsize_normal": 44,
        "fontsize_hook": 50,
        "fontsize_emphasis": 48,
        "outline": 5,
        "shadow": "3",
        "glow": True,
        "color_inactive": "&H00FFFFFF",
        "color_active": "&H0000FFFF",
        "color_outline": "&H00000000",
        "color_emphasis": "&H0000FFFF",
        "color_secondary": "&H0000DDFF",
        "margin_v": 90
    },
    "neon_cyan": {
        "name": "Neon Cyan",
        "fontname": "Arial Black",
        "fontsize_normal": 42,
        "fontsize_hook": 48,
        "fontsize_emphasis": 46,
        "outline": 4,
        "shadow": "3",
        "glow": True,
        "color_inactive": "&H00FFFFFF",
        "color_active": "&H00FFFF00",
        "color_outline": "&H00000000",
        "color_emphasis": "&H0000FFFF",
        "color_secondary": "&H00FFAA00",
        "margin_v": 90
    },
    "hot_pink": {
        "name": "Hot Pink",
        "fontname": "Impact",
        "fontsize_normal": 42,
        "fontsize_hook": 48,
        "fontsize_emphasis": 46,
        "outline": 4,
        "shadow": "3",
        "glow": True,
        "color_inactive": "&H00FFFFFF",
        "color_active": "&H00FF1493",
        "color_outline": "&H00000000",
        "color_emphasis": "&H0000FFFF",
        "color_secondary": "&H00FF69B4",
        "margin_v": 90
    },
    "lime_green": {
        "name": "Lime Green",
        "fontname": "Arial Black",
        "fontsize_normal": 44,
        "fontsize_hook": 50,
        "fontsize_emphasis": 48,
        "outline": 5,
        "shadow": "3",
        "glow": True,
        "color_inactive": "&H00FFFFFF",
        "color_active": "&H0000FF00",
        "color_outline": "&H00000000",
        "color_emphasis": "&H0000FFFF",
        "color_secondary": "&H0000DD00",
        "margin_v": 95
    },
    "orange_fire": {
        "name": "Orange Fire",
        "fontname": "Impact",
        "fontsize_normal": 44,
        "fontsize_hook": 50,
        "fontsize_emphasis": 48,
        "outline": 5,
        "shadow": "3",
        "glow": True,
        "color_inactive": "&H00FFFFFF",
        "color_active": "&H000099FF",
        "color_outline": "&H00000000",
        "color_emphasis": "&H0000FFFF",
        "color_secondary": "&H0000BBFF",
        "margin_v": 95
    },
    "purple_vibes": {
        "name": "Purple Vibes",
        "fontname": "Montserrat Black",
        "fontsize_normal": 42,
        "fontsize_hook": 48,
        "fontsize_emphasis": 46,
        "outline": 4,
        "shadow": "3",
        "glow": True,
        "color_inactive": "&H00FFFFFF",
        "color_active": "&H00FF00FF",
        "color_outline": "&H00000000",
        "color_emphasis": "&H00FF00FF",
        "color_secondary": "&H00CC00FF",
        "margin_v": 90
    },
    "turquoise_wave": {
        "name": "Turquoise Wave",
        "fontname": "Arial Black",
        "fontsize_normal": 42,
        "fontsize_hook": 48,
        "fontsize_emphasis": 46,
        "outline": 4,
        "shadow": "3",
        "glow": True,
        "color_inactive": "&H00FFFFFF",
        "color_active": "&H00FFCC00",
        "color_outline": "&H00000000",
        "color_emphasis": "&H00FFFF00",
        "color_secondary": "&H00FFDD00",
        "margin_v": 90
    },
    "red_hot": {
        "name": "Red Hot",
        "fontname": "Impact",
        "fontsize_normal": 44,
        "fontsize_hook": 50,
        "fontsize_emphasis": 48,
        "outline": 5,
        "shadow": "3",
        "glow": True,
        "color_inactive": "&H00FFFFFF",
        "color_active": "&H000000FF",
        "color_outline": "&H00000000",
        "color_emphasis": "&H0000FFFF",
        "color_secondary": "&H000033FF",
        "margin_v": 95
    }
}

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

# Emphasis list mevcut ama varsayılan davranış vurguyu KULLANMAMAK.
EMPHASIS_KEYWORDS = {
    "NEVER","ALWAYS","IMPOSSIBLE","INSANE","CRAZY","SHOCKING","UNBELIEVABLE",
    "INCREDIBLE","AMAZING","STUNNING","MIND-BLOWING","NOW","IMMEDIATELY",
    "INSTANTLY","URGENT","BREAKING","ALERT","STOP","WAIT","ATTENTION","WARNING",
    "DANGER","SECRET","HIDDEN","BANNED","ILLEGAL","FORBIDDEN","RARE","EXCLUSIVE",
    "LIMITED","FIRST","LAST","ONLY","UNIQUE","BEST","WORST","BIGGEST","SMALLEST",
    "FASTEST","SLOWEST","MOST","LEAST","ULTIMATE","SUPREME","MAXIMUM","VIRAL",
    "TRENDING","POPULAR","FAMOUS","EVERYONE","NOBODY","MILLIONS","THOUSANDS","BILLION"
}

def get_random_style() -> str:
    return random.choices(list(STYLE_WEIGHTS.keys()), weights=list(STYLE_WEIGHTS.values()))[0]

def get_style_info(style_name: str) -> Dict[str, Any]:
    return CAPTION_STYLES.get(style_name, CAPTION_STYLES["classic_yellow"])

def list_all_styles() -> List[str]:
    return list(CAPTION_STYLES.keys())

def build_karaoke_ass(
    text: str,
    seg_dur: float,
    words: List[Tuple[str, float]],
    is_hook: bool = False,
    style_name: Optional[str] = None,
    *,
    emphasize: bool = False,        # ❗ Varsayılan KAPALI
    fancy_motion: bool = False      # ❗ Varsayılan KAPALI (yukarı kayma yok)
) -> str:
    """
    ASS çıktısı üretir. Varsayılan olarak vurgu ve hareket animasyonu devre dışıdır.
    """
    if not style_name:
        style_name = get_random_style()
    style = get_style_info(style_name)
    fontsize = style["fontsize_hook"] if is_hook else style["fontsize_normal"]

    ass_content = f"""[Script Info]
Title: Karaoke Captions
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{style['fontname']},{fontsize},{style['color_inactive']},{style['color_active']},{style['color_outline']},&H80000000,-1,0,0,0,100,100,1,0,1,{style['outline']},{style['shadow']},2,50,50,{style['margin_v']},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    if not words:
        end_time = _format_ass_time(seg_dur)
        ass_content += f"Dialogue: 0,0:00:00.00,{end_time},Default,,0,0,0,,{(text or '').upper()}\n"
        return ass_content

    cumulative_time = 0.0
    chunk_size = 2 if is_hook else 3
    chunks: List[Tuple[List[Tuple[str, float]], float]] = []
    cur, cur_dur = [], 0.0
    for w, d in words:
        cur.append((w, d))
        cur_dur += d
        if len(cur) >= chunk_size:
            chunks.append((cur, cur_dur))
            cur, cur_dur = [], 0.0
    if cur:
        chunks.append((cur, cur_dur))

    for chunk_words, chunk_duration in chunks:
        start_time = _format_ass_time(cumulative_time)
        end_time = _format_ass_time(cumulative_time + chunk_duration)

        karaoke_text = ""
        for w, d in chunk_words:
            cs = max(1, int(d * 100))  # centiseconds
            wu = w.upper().strip('.,!?;:')
            if emphasize and wu in EMPHASIS_KEYWORDS:
                karaoke_text += f"{{\\k{cs}\\fs{style['fontsize_emphasis']}\\c{style['color_emphasis']}}}{wu} "
            else:
                karaoke_text += f"{{\\k{cs}}}{wu} "

        # ❌ Yukarı kayma animasyonu KALDIRILDI (yalnızca fancy_motion True ise eklenir)
        if fancy_motion:
            karaoke_text = f"{{\\move(960,1000,960,980,0,{int(chunk_duration*1000)})}}" + karaoke_text

        ass_content += f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{karaoke_text.strip()}\n"
        cumulative_time += chunk_duration

    return ass_content

def _format_ass_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centiseconds = int((seconds % 1) * 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"
