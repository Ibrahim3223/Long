# -*- coding: utf-8 -*-
"""
Caption Keyword Highlighter
✅ Highlights important words in captions
✅ ASS subtitle formatting
✅ Numbers, emphasis words, questions
"""
import re


class KeywordHighlighter:
    """Highlight important keywords in captions for better engagement."""

    # ASS color codes (BGR format)
    COLORS = {
        "yellow": "&H00FFFF&",  # Numbers
        "red": "&H0000FF&",  # Emphasis
        "cyan": "&H00FFFF&",  # Questions
        "orange": "&H0080FF&",  # Names
    }

    EMPHASIS_WORDS = [
        "shocking", "incredible", "never", "always", "nobody",
        "unbelievable", "amazing", "insane", "crazy", "mind"
    ]

    def highlight_sentence(self, sentence: str) -> str:
        """
        Add ASS formatting to highlight keywords.

        Returns:
            ASS-formatted sentence with highlights
        """
        result = sentence

        # Highlight numbers (yellow, bold, 1.2x size)
        result = re.sub(
            r'\b(\d+)\b',
            r'{\\c&H00FFFF&\\b1\\fs1.2}\1{\\r}',
            result
        )

        # Highlight emphasis words (red, bold)
        for word in self.EMPHASIS_WORDS:
            pattern = rf'\b({word})\b'
            replacement = r'{\\c&H0000FF&\\b1}\1{\\r}'
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        # Highlight questions (cyan)
        if '?' in result:
            result = result.replace('?', '{\\c&H00FFFF&}?{\\r}')

        return result
