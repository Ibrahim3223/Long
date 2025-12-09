# -*- coding: utf-8 -*-
"""
Keyword Highlighter for Captions
Highlights important words for better engagement and readability

Research shows:
- Highlighted captions increase engagement by 60%
- Numbers and emphasis words draw attention
- Mobile viewers rely heavily on visual cues
"""
import re
import logging
from typing import List, Set

logger = logging.getLogger(__name__)


class ShortsKeywordHighlighter:
    """Highlight important keywords in captions for better engagement."""

    # ASS color codes (BGR format)
    COLORS = {
        "yellow": "&H00FFFF&",  # Numbers, facts
        "red": "&H0000FF&",  # Emphasis, power words
        "cyan": "&HFFFF00&",  # Questions
        "white": "&H00FFFFFF&",  # Default
    }

    # Emphasis words (attention-grabbing)
    EMPHASIS_WORDS = [
        # Extremes
        "shocking", "incredible", "unbelievable", "mindblowing",
        "insane", "crazy", "wild", "extreme",
        # Absolutes
        "never", "always", "nobody", "everyone", "impossible",
        # Mystery/Intrigue
        "secret", "hidden", "truth", "mystery", "unknown",
        # Superlatives
        "best", "worst", "fastest", "biggest", "smallest",
        # Emotional
        "amazing", "stunning", "bizarre", "weird", "strange",
    ]

    def __init__(self, additional_words: List[str] = None):
        """
        Initialize highlighter.

        Args:
            additional_words: Additional emphasis words to highlight
        """
        self.emphasis_words = set(self.EMPHASIS_WORDS)

        if additional_words:
            self.emphasis_words.update(w.lower() for w in additional_words)

        logger.debug(f"Keyword highlighter initialized with {len(self.emphasis_words)} emphasis words")

    def highlight(self, text: str) -> str:
        """
        Add ASS formatting to highlight keywords.

        Highlights:
        - Numbers → Yellow, Bold, 1.3x size
        - Emphasis words → Red, Bold
        - Questions → Cyan
        - Exclamations → Slightly larger

        Args:
            text: Plain text caption

        Returns:
            ASS-formatted text with color/size highlights
        """
        result = text

        # Placeholder system to protect ASS tags
        placeholders = {}
        counter = [0]

        def save_tag(match):
            key = f"__TAG{counter[0]}__"
            placeholders[key] = match.group(0)
            counter[0] += 1
            return key

        def protect_tags(text):
            """Save all ASS tags as placeholders."""
            return re.sub(r'\{[^}]+\}', save_tag, text)

        def restore_tags(text):
            """Restore all placeholders to original ASS tags."""
            for key, val in placeholders.items():
                text = text.replace(key, val)
            return text

        # Step 1: Highlight parentheses content: (J), (1)
        def highlight_paren(m):
            return '{\\c&H00FFFF&\\b1\\fscx130\\fscy130}(' + m.group(1) + '){\\r}'

        result = re.sub(r'\(([A-Za-z0-9]+)\)', highlight_paren, result)
        result = protect_tags(result)

        # Step 2: Highlight number-hyphen: 3-MINUTE, 5-STAR
        def highlight_number_hyphen(m):
            return '{\\c&H00FFFF&\\b1\\fscx130\\fscy130}' + m.group(1) + '{\\r}-'

        result = re.sub(r'(\d+)-', highlight_number_hyphen, result)
        result = protect_tags(result)

        # Step 3a: Highlight numbers with percent BEFORE (e.g., "%99")
        def highlight_percent_number(m):
            return '{\\c&H00FFFF&\\b1\\fscx130\\fscy130}' + m.group(1) + '{\\r}'

        result = re.sub(r'\b(%\d+(?:,\d+)*(?:\.\d+)?)\b', highlight_percent_number, result)
        result = protect_tags(result)

        # Step 3b: Highlight standalone numbers with optional suffixes: 100, 5, 3.14, 99%, 5K, 1M
        def highlight_number(m):
            return '{\\c&H00FFFF&\\b1\\fscx130\\fscy130}' + m.group(1) + '{\\r}'

        result = re.sub(r'\b(\d+(?:,\d+)*(?:\.\d+)?[%KMBkmb]?)\b', highlight_number, result)
        result = protect_tags(result)

        # Restore all tags
        result = restore_tags(result)

        # 2. Highlight emphasis words (RED, BOLD)
        for word in self.emphasis_words:
            pattern = rf'\b({re.escape(word)})\b'

            def highlight_emphasis(m):
                return '{\\c&H0000FF&\\b1}' + m.group(1) + '{\\r}'

            result = re.sub(pattern, highlight_emphasis, result, flags=re.IGNORECASE)

        # 3. Highlight questions (CYAN)
        if '?' in result:
            result = result.replace('?', '{\\c&HFFFF00&\\b1}?{\\r}')

        # 4. Highlight exclamations (BOLD, slightly larger)
        if '!' in result:
            result = result.replace('!', '{\\b1\\fscx110\\fscy110}!{\\r}')

        return result

    def add_emphasis_word(self, word: str):
        """Add custom emphasis word to highlight list."""
        word_lower = word.lower()
        if word_lower not in self.emphasis_words:
            self.emphasis_words.add(word_lower)

    def add_emphasis_words(self, words: List[str]):
        """Add multiple emphasis words."""
        for word in words:
            self.add_emphasis_word(word)

    def get_emphasis_words(self) -> Set[str]:
        """Get current set of emphasis words."""
        return self.emphasis_words.copy()


# Backward compatibility alias
KeywordHighlighter = ShortsKeywordHighlighter
