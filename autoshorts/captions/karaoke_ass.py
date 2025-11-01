# -*- coding: utf-8 -*-
"""
CapCut-Style Viral Captions - NO KARAOKE
✅ KALIN font (Impact/Bold)
✅ BÜYÜK gölgelendirme
✅ Sahne sahne CANLI renkler
❌ Karaoke efekti YOK (kelime kelime değişim yok)
"""

from typing import List, Dict, Optional, Any, Tuple
import random

# CapCut tarzı viral renk paleti (ASS format: &H00BBGGRR)
VIRAL_COLORS = {
    "electric_yellow": "&H0000FFFF",   # Elektrik sarısı
    "hot_pink": "&H00FF00FF",          # Pembe
    "neon_cyan": "&H00FFFF00",         # Neon cyan
    "lime_green": "&H0000FF00",        # Lime yeşil
    "electric_blue": "&H00FF6600",     # Elektrik mavisi
    "orange": "&H0000A5FF",            # Turuncu
    "purple": "&H00FF00CC",            # Mor
    "red": "&H000000FF",               # Kırmızı
}

# CapCut stili - MÜTHİŞ KALIN ve GÖLGELİ
CAPTION_STYLES: Dict[str, Dict[str, Any]] = {
    "capcut_impact": {
        "name": "CapCut Impact",
        "fontname": "Impact",  # ✅ En kalın font
        "fontsize_normal": 52,  # ✅ Daha büyük
        "fontsize_hook": 58,    # ✅ Hook için daha da büyük
        "bold": -1,  # ✅ Extra bold (ASS'de -1 = çok kalın)
        "outline": 7,  # ✅ Çok kalın outline (viral görünüm)
        "shadow": "4",  # ✅ Derinlik için kalın gölge
        "color_outline": "&H00000000",  # Siyah outline
        "color_shadow": "&H80000000",   # Yarı saydam siyah gölge
        "margin_v": 80  # Alt kısımdan mesafe
    },
    "capcut_montserrat": {
        "name": "CapCut Montserrat",
        "fontname": "Montserrat Black",  # Modern ve kalın
        "fontsize_normal": 50,
        "fontsize_hook": 56,
        "bold": -1,
        "outline": 6,
        "shadow": "3",
        "color_outline": "&H00000000",
        "color_shadow": "&H80000000",
        "margin_v": 80
    },
    "capcut_bebas": {
        "name": "CapCut Bebas",
        "fontname": "Bebas Neue",  
        "fontsize_normal": 54,
        "fontsize_hook": 60,
        "bold": -1,
        "outline": 7,
        "shadow": "4",
        "color_outline": "&H00000000",
        "color_shadow": "&H80000000",
        "margin_v": 80
    },
    # ✅ SHORT-VIDEO-MAKER MINIMAL STYLES (daha temiz, profesyonel)
    "minimal_clean": {
        "name": "Minimal Clean",
        "fontname": "Arial",  
        "fontsize_normal": 46,  # Daha küçük, dikkat dağıtmıyor
        "fontsize_hook": 52,
        "bold": 0,  # Normal kalınlık
        "outline": 3,  # İnce outline
        "shadow": "2",  # Hafif gölge
        "color_outline": "&H00000000",  
        "color_shadow": "&H50000000",   # Çok hafif gölge
        "margin_v": 100,  
    },
    "modern_subtle": {
        "name": "Modern Subtle",
        "fontname": "Roboto",  
        "fontsize_normal": 48,
        "fontsize_hook": 54,
        "bold": -1,  
        "outline": 4,  
        "shadow": "2",
        "color_outline": "&H00000000",
        "color_shadow": "&H60000000",
        "margin_v": 90,
    },
}

# Font fallback sırası (eğer yoksa)
FONT_FALLBACK = [
    "Impact",           # Windows/Linux
    "Arial Black",      # Evrensel
    "Montserrat Black", # Modern
    "Bebas Neue",      # Viral
    "Anton",           # Alternatif
    "Oswald Bold",     # Alternatif
    "Arial"            # Son çare
]

STYLE_WEIGHTS = {
    "capcut_impact": 0.30,       # Viral, kalın
    "capcut_montserrat": 0.20,   # Modern kalın
    "capcut_bebas": 0.15,        # Yüksek ve dikkat çekici
    "minimal_clean": 0.20,       # ✅ Short-video-maker tarzı (temiz)
    "modern_subtle": 0.15,       # ✅ Modern, profesyonel
}

def get_random_style() -> str:
    """Rastgele CapCut stili seç"""
    return random.choices(list(STYLE_WEIGHTS.keys()), weights=list(STYLE_WEIGHTS.values()))[0]

def get_random_color() -> Tuple[str, str]:
    """Rastgele viral renk seç (renk adı, ASS kodu)"""
    color_name = random.choice(list(VIRAL_COLORS.keys()))
    return color_name, VIRAL_COLORS[color_name]

def get_style_info(style_name: str) -> Dict[str, Any]:
    return CAPTION_STYLES.get(style_name, CAPTION_STYLES["capcut_impact"])


def build_karaoke_ass(
    text: str,
    seg_dur: float,
    words: List[Tuple[str, float]],
    is_hook: bool = False,
    style_name: Optional[str] = None,
    time_offset: float = 0.0,
) -> str:
    """
    ✅ CapCut tarzı viral altyazı:
    - BÜYÜK HARF
    - KALIN font (Impact/Montserrat)
    - Kalın outline + gölge
    - Cümle başına RASTGELE canlı renk
    - ❌ Karaoke efekti YOK (kelime kelime değişim yok)
    """
    # ✅ BÜYÜK HARF
    text = (text or "").strip().upper()
    
    # ✅ Rastgele stil seç (her cümle için)
    if not style_name:
        style_name = get_random_style()
    style = get_style_info(style_name)
    
    # ✅ Rastgele viral renk seç (her cümle için farklı)
    color_name, color_code = get_random_color()
    
    fontsize = style["fontsize_hook"] if is_hook else style["fontsize_normal"]

    # ✅ Font fallback ile dene
    primary_font = style["fontname"]
    
    # ✅ Background color - style'dan al (Python kodu, ASS'den önce!)
    back_color = style.get('color_shadow', '&H80000000')
    
    # ✅ ASS header oluştur (artık back_color tanımlı)
    header = f"""[Script Info]
Title: CapCut Viral Captions - NO KARAOKE
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{primary_font},{fontsize},{color_code},{color_code},{style['color_outline']},{back_color},{style['bold']},0,0,0,100,100,1.5,0,1,{style['outline']},{style['shadow']},2,50,50,{style['margin_v']},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    # ✅ SADE ALTYAZI - Tüm metin aynı anda, aynı renkte
    # Karaoke \k tag'leri YOK
    start_time = _ass_time(time_offset)
    end_time = _ass_time(seg_dur + time_offset)
    
    # ✅ Metin çok uzunsa 2 satıra böl (daha okunabilir)
    words_list = text.split()
    if len(words_list) > 8:
        # Ortadan böl
        mid = len(words_list) // 2
        line1 = " ".join(words_list[:mid])
        line2 = " ".join(words_list[mid:])
        text = f"{line1}\\N{line2}"  # \\N = yeni satır (ASS formatı)
    
    body = f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text}\n"
    
    return header + body


def _ass_time(seconds: float) -> str:
    """ASS format time string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


# ============================================================================
# Geriye dönük uyumluluk için helper fonksiyonlar
# ============================================================================

def get_random_color_name() -> str:
    """Renk adı döndür (debug için)"""
    return random.choice(list(VIRAL_COLORS.keys()))

def list_all_colors() -> List[str]:
    """Tüm renkleri listele"""
    return list(VIRAL_COLORS.keys())

def list_all_styles() -> List[str]:
    """Tüm stilleri listele"""
    return list(CAPTION_STYLES.keys())
