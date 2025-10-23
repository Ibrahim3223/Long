# -*- coding: utf-8 -*-
"""
Karaoke ASS subtitle builder - LONG-FORM VERSION (16:9 LANDSCAPE)
✅ FIXED: Now actually uses the colorful caption styles!
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


def build_karaoke_ass(
    text: str,
    seg_dur: float,
    words: List[Tuple[str, float]],
    is_hook: bool = False,
    style_name: Optional[str] = None
) -> str:
    """
    ✅ FIXED: Build complete ASS file with COLORFUL karaoke captions!
    
    Args:
        text: Full text to display
        seg_dur: Total segment duration
        words: List of (word, duration) tuples
        is_hook: Whether this is a hook segment
        style_name: Specific style to use (or None for random)
    
    Returns:
        Complete ASS subtitle string with colorful animated captions
    """
    # Select style
    if not style_name:
        style_name = get_random_style()
    
    style = get_style_info(style_name)
    
    # Determine font size based on context
    if is_hook:
        fontsize = style["fontsize_hook"]
    else:
        fontsize = style["fontsize_normal"]
    
    # Build ASS header
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
    
    # Generate karaoke events
    if not words:
        # Fallback: show entire text without karaoke
        end_time = _format_ass_time(seg_dur)
        ass_content += f"Dialogue: 0,0:00:00.00,{end_time},Default,,0,0,0,,{text.upper()}\n"
    else:
        # Build karaoke with word-by-word highlighting
        cumulative_time = 0.0
        
        # Group words into chunks (2-3 words per line for readability)
        chunk_size = 2 if is_hook else 3
        chunks = []
        current_chunk = []
        current_chunk_duration = 0.0
        
        for word, duration in words:
            current_chunk.append((word, duration))
            current_chunk_duration += duration
            
            if len(current_chunk) >= chunk_size:
                chunks.append((current_chunk, current_chunk_duration))
                current_chunk = []
                current_chunk_duration = 0.0
        
        # Add remaining words
        if current_chunk:
            chunks.append((current_chunk, current_chunk_duration))
        
        # Generate events for each chunk
        for chunk_words, chunk_duration in chunks:
            start_time = _format_ass_time(cumulative_time)
            end_time = _format_ass_time(cumulative_time + chunk_duration)
            
            # Build karaoke tags for each word in chunk
            karaoke_text = ""
            word_cumulative = 0
            
            for word, duration in chunk_words:
                # Convert duration to centiseconds for \\k tag
                duration_cs = int(duration * 100)
                
                # Check if word should be emphasized
                word_upper = word.upper().strip('.,!?;:')
                if word_upper in EMPHASIS_KEYWORDS:
                    # Emphasis: bigger, different color
                    karaoke_text += f"{{\\k{duration_cs}\\fs{style['fontsize_emphasis']}\\c{style['color_emphasis']}}}{word.upper()} "
                else:
                    # Normal karaoke
                    karaoke_text += f"{{\\k{duration_cs}}}{word.upper()} "
                
                word_cumulative += duration
            
            # Add bounce effect if enabled
            if style.get("bounce"):
                # Subtle bounce animation
                karaoke_text = f"{{\\move(960,1000,960,980,0,{int(chunk_duration*1000)})}}" + karaoke_text
            
            ass_content += f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{karaoke_text.strip()}\n"
            
            cumulative_time += chunk_duration
    
    return ass_content


def _format_ass_time(seconds: float) -> str:
    """Format seconds to ASS timestamp (H:MM:SS.CS)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centiseconds = int((seconds % 1) * 100)
    
    return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"
