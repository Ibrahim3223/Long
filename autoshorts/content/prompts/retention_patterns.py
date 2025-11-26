# -*- coding: utf-8 -*-
"""
Retention Patterns for Script Enhancement
✅ Pattern interrupts every 15-20 seconds
✅ Curiosity gaps and cliffhangers
✅ Injects naturally into script flow
"""

RETENTION_PATTERNS = {
    "curiosity_gaps": [
        "But wait...",
        "Here's the crazy part...",
        "You won't believe what happens next...",
        "Plot twist:",
        "But there's more...",
        "The shocking truth?",
        "And then something unexpected happened...",
        "But that's not even the strangest part...",
        "Wait until you see this...",
        "The next part will blow your mind...",
    ],

    "emphasis_markers": [
        "Listen to this:",
        "This is important:",
        "Pay attention:",
        "Remember this:",
        "Here's the key:",
    ],

    "question_hooks": [
        "Want to know why?",
        "Ever wondered how?",
        "Think that's impressive?",
        "Sounds impossible, right?",
    ]
}


# Add these to enhanced_prompts.py as examples for Gemini
RETENTION_LOOP_INSTRUCTIONS = """
RETENTION OPTIMIZATION (CRITICAL):

Every 15-20 seconds (4-5 sentences), include a PATTERN INTERRUPT:

Examples:
- "But wait..." (before revealing key info)
- "Here's the crazy part..." (mid-story)
- "You won't believe what happens next..." (before climax)
- "Plot twist:" (unexpected turn)

These keep viewers watching and boost retention @30s from 55% → 70%+

IMPORTANT: Use these naturally, don't force them. They should feel like part of the story.
"""
