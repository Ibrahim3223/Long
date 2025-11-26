# -*- coding: utf-8 -*-
"""
Hook Patterns for Viral Script Openings
✅ Cold open strategies
✅ Intensity-based variations
✅ No meta-talk ("this video", "today we'll")
"""

# Hook intensity levels
HOOK_PATTERNS = {
    "extreme": [
        "This {entity} {shocking_action} in {timeframe}.",
        "{number} {people/things} {dramatic_outcome} because of this.",
        "Everything you know about {topic} is wrong.",
        "In {year}, {entity} discovered something that changed {field} forever.",
        "{entity} did the impossible: {achievement}.",
    ],

    "high": [
        "{entity} {action} that nobody expected.",
        "This {thing} {attribute} that scientists can't explain.",
        "{location} holds a secret about {topic}.",
        "One {tiny_thing} changed {big_thing} forever.",
        "The truth about {topic} is stranger than you think.",
    ],

    "medium": [
        "Here's what makes {entity} so {adjective}.",
        "{entity} works in a way you've never imagined.",
        "The story behind {topic} is fascinating.",
        "{entity} has a hidden power.",
        "Most people don't know this about {topic}.",
    ],

    "low": [
        "Let's explore {topic}.",
        "{entity} is remarkable for one reason.",
        "There's something special about {topic}.",
        "The science of {topic} is intriguing.",
        "{entity} reveals an interesting pattern.",
    ],
}

# Cold open rules (what to AVOID)
COLD_OPEN_VIOLATIONS = [
    "this video",
    "today we",
    "in this video",
    "let me show you",
    "welcome to",
    "hey guys",
    "before we start",
    "make sure to subscribe",
]

# Mini cliffhanger patterns (for mid-video engagement)
CLIFFHANGER_PATTERNS = [
    "But that's not the strangest part.",
    "And then something unexpected happened.",
    "Wait until you hear what comes next.",
    "The reason why will surprise you.",
    "But here's where it gets interesting.",
    "That's when everything changed.",
    "The truth is even more bizarre.",
    "Scientists discovered something shocking.",
]

# CTA patterns by softness
CTA_PATTERNS = {
    "soft": [
        "That's the story of {topic}.",
        "And that's how {entity} {outcome}.",
        "The mystery of {topic} continues.",
        "Nature never stops surprising us.",
        "The universe is full of wonders.",
    ],

    "medium": [
        "If you found this fascinating, there's more to discover.",
        "The world is full of stories like this.",
        "This is just one of many amazing {category}.",
        "Science keeps revealing new surprises.",
        "Every discovery leads to more questions.",
    ],

    "strong": [
        "Want to learn more? Explore our other videos.",
        "Subscribe to discover more amazing stories.",
        "There's so much more to uncover.",
        "Join us next time for another incredible story.",
        "Don't miss our next exploration.",
    ],
}


def get_hook_pattern(intensity: str = "high") -> str:
    """Get a random hook pattern based on intensity."""
    import random
    patterns = HOOK_PATTERNS.get(intensity, HOOK_PATTERNS["high"])
    return random.choice(patterns)


def get_cliffhanger() -> str:
    """Get a random mid-video cliffhanger."""
    import random
    return random.choice(CLIFFHANGER_PATTERNS)


def get_cta_pattern(softness: str = "medium") -> str:
    """Get a CTA pattern based on softness level."""
    import random
    patterns = CTA_PATTERNS.get(softness, CTA_PATTERNS["medium"])
    return random.choice(patterns)


def validate_cold_open(text: str) -> bool:
    """Check if text violates cold open rules."""
    text_lower = text.lower()
    return not any(violation in text_lower for violation in COLD_OPEN_VIOLATIONS)
