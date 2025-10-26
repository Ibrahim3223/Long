# -*- coding: utf-8 -*-
"""
Karaoke ASS subtitle builder - LANDSCAPE (16:9)
✅ BÜYÜK HARF - temiz ve okunabilir
✅ SADE geçiş: beyaz → sarı (vurgu YOK)
✅ Kelime bazlı \k timing (hassas senkronizasyon)
"""

from typing import List, Dict, Optional, Any, Tuple

# 16:9 alt yazı stilleri (alt hizalı)
CAPTION_STYLES: Dict[str, Dict[str, Any]] = {
    "clean_yellow": {
        "name": "Clean Yellow",
        "fontname": "Arial Black",
        "fontsize_normal": 46,
        "fontsize_hook": 52,
        "outline": 5,
        "shadow": "3",
        "color_inactive": "&H00FFFFFF",  # Beyaz (başlangıç)
        "color_active": "&H0000FFFF",    # Sarı (aktif)
        "color_outline": "&H00000000",   # Siyah outline
        "margin_v": 90
    },
    "clean_cyan": {
        "name": "Clean Cyan",
        "fontname": "Arial Black",
        "fontsize_normal": 44,
        "fontsize_hook": 50,
        "outline": 4,
        "shadow": "3",
        "color_inactive": "&H00FFFFFF",
        "color_active": "&H00FFFF00",    # Cyan
        "color_outline": "&H00000000",
        "margin_v": 90
    }
}

STYLE_WEIGHTS = {
    "clean_yellow": 0.7,  # Daha çok sarı (en temiz)
    "clean_cyan": 0.3,
}

def get_random_style() -> str:
    import random
    return random.choices(list(STYLE_WEIGHTS.keys()), weights=list(STYLE_WEIGHTS.values()))[0]

def get_style_info(style_name: str) -> Dict[str, Any]:
    return CAPTION_STYLES.get(style_name, CAPTION_STYLES["clean_yellow"])

def list_all_styles() -> List[str]:
    return list(CAPTION_STYLES.keys())


def build_karaoke_ass(
    text: str,
    seg_dur: float,
    words: List[Tuple[str, float]],
    is_hook: bool = False,
    style_name: Optional[str] = None,
    **kwargs  # Geriye dönük uyumluluk
) -> str:
    """
    ✅ SADE karaoke: BÜYÜK HARF, beyaz→sarı geçiş, kelime kelime \k
    ❌ VURGU YOK, animasyon YOK
    """
    # ✅ BÜYÜK HARF
    text = (text or "").strip().upper()
    
    style = get_style_info(style_name or "clean_yellow")
    fontsize = style["fontsize_hook"] if is_hook else style["fontsize_normal"]

    header = f"""[Script Info]
Title: Clean Uppercase Captions
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
        body = f"Dialogue: 0,0:00:00.00,{end_time},Default,,0,0,0,,{text}\n"
        return header + body

    # ✅ Kelimeleri büyük harfe çevir
    words = [(w.strip().upper(), d) for w, d in words if w and w.strip()]
    
    # 3 kelimelik bloklar (uzun video için optimal)
    chunk_size = 3
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
        
        # ✅ SADE \k akışı - VURGU YOK
        karaoke_text = ""
        for w, d in chunk_words:
            cs = max(1, int(round(float(d) * 100)))  # Centisecond
            karaoke_text += f"{{\\k{cs}}}{w} "
        
        body_lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{karaoke_text.strip()}\n")
        t += dur

    return header + "".join(body_lines)


def _ass_time(seconds: float) -> str:
    """ASS format time string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"
