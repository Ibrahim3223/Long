# -*- coding: utf-8 -*-
"""
Karaoke ASS subtitle builder - LANDSCAPE (16:9)
Sade mod: vurgu YOK, animasyon YOK, yalnızca kelime bazlı \k highlight.
"""

from typing import List, Dict, Optional, Any, Tuple

# 16:9 alt yazı stilleri (alt hizalı)
CAPTION_STYLES: Dict[str, Dict[str, Any]] = {
    "classic_yellow": {
        "name": "Classic Yellow",
        "fontname": "Arial Black",
        "fontsize_normal": 44,
        "fontsize_hook": 50,
        "outline": 5,
        "shadow": "3",
        "color_inactive": "&H00FFFFFF",  # White
        "color_active": "&H0000FFFF",    # Yellow
        "color_outline": "&H00000000",   # Black
        "margin_v": 90
    },
    "neon_cyan": {
        "name": "Neon Cyan",
        "fontname": "Arial Black",
        "fontsize_normal": 42,
        "fontsize_hook": 48,
        "outline": 4,
        "shadow": "3",
        "color_inactive": "&H00FFFFFF",
        "color_active": "&H00FFFF00",    # Cyan
        "color_outline": "&H00000000",
        "margin_v": 90
    },
    "hot_pink": {
        "name": "Hot Pink",
        "fontname": "Impact",
        "fontsize_normal": 42,
        "fontsize_hook": 48,
        "outline": 4,
        "shadow": "3",
        "color_inactive": "&H00FFFFFF",
        "color_active": "&H00FF1493",
        "color_outline": "&H00000000",
        "margin_v": 90
    },
    "lime_green": {
        "name": "Lime Green",
        "fontname": "Arial Black",
        "fontsize_normal": 44,
        "fontsize_hook": 50,
        "outline": 5,
        "shadow": "3",
        "color_inactive": "&H00FFFFFF",
        "color_active": "&H0000FF00",
        "color_outline": "&H00000000",
        "margin_v": 95
    }
}

STYLE_WEIGHTS = {
    "classic_yellow": 0.4,
    "neon_cyan": 0.25,
    "hot_pink": 0.2,
    "lime_green": 0.15,
}

def get_random_style() -> str:
    import random
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
    emphasize: bool = False,     # tutuluyor ama kullanılmıyor (vurgu kapalı)
    fancy_motion: bool = False,  # tutuluyor ama kullanılmıyor (animasyon kapalı)
) -> str:
    """
    DÜZ karaoke: vurgu/animasyon yok, kelime kelime \k akışı.
    """
    style = get_style_info(style_name or "classic_yellow")
    fontsize = style["fontsize_hook"] if is_hook else style["fontsize_normal"]

    header = f"""[Script Info]
Title: Karaoke Captions (Plain)
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
        end_time = _ass_time(seg_dur)
        body = f"Dialogue: 0,0:00:00.00,{end_time},Default,,0,0,0,,{(text or '').strip()}\n"
        return header + body

    # 2-3 kelimelik bloklar
    chunk_size = 2 if is_hook else 3
    chunks: List[Tuple[List[Tuple[str, float]], float]] = []
    cur, cur_d = [], 0.0
    for w, d in words:
        cur.append((w, float(d)))
        cur_d += float(d)
        if len(cur) >= chunk_size:
            chunks.append((cur, cur_d))
            cur, cur_d = [], 0.0
    if cur:
        chunks.append((cur, cur_d))

    body_lines = []
    t = 0.0
    for chunk_words, dur in chunks:
        start = _ass_time(t)
        end = _ass_time(t + dur)
        # yalın \k akışı — vurgu ve hareket yok
        karaoke_text = ""
        for w, d in chunk_words:
            cs = max(1, int(round(float(d) * 100)))
            karaoke_text += f"{{\\k{cs}}}{w} "
        body_lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{karaoke_text.strip()}\n")
        t += dur

    return header + "".join(body_lines)


def _ass_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"
