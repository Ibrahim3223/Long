# -*- coding: utf-8 -*-
"""
Script Structure Templates
✅ Modular script framework
✅ Section-based organization
✅ Flow optimization
"""

# Modular script structure
SCRIPT_STRUCTURE = {
    "5_act": {
        "description": "Classic 5-act structure for storytelling",
        "sections": [
            {
                "name": "hook",
                "purpose": "Grab attention in first 5 seconds",
                "duration_percent": 10,  # % of total video
                "sentence_count_range": (1, 2),
                "rules": [
                    "Start with the most shocking/interesting fact",
                    "NO meta-talk (no 'this video', 'today we')",
                    "Immediate action or revelation",
                    "Single powerful image/concept",
                ],
            },
            {
                "name": "context",
                "purpose": "Establish background and why it matters",
                "duration_percent": 20,
                "sentence_count_range": (3, 6),
                "rules": [
                    "Answer: What is this?",
                    "Answer: Why should I care?",
                    "Provide necessary background",
                    "Set up the main story",
                ],
            },
            {
                "name": "mechanism",
                "purpose": "Explain how it works or what happened",
                "duration_percent": 40,
                "sentence_count_range": (8, 15),
                "rules": [
                    "Break down complex concepts",
                    "Use analogies and comparisons",
                    "Step-by-step progression",
                    "Include specific details and examples",
                    "Mini cliffhangers every 20-30 seconds",
                ],
            },
            {
                "name": "impact",
                "purpose": "Show significance and consequences",
                "duration_percent": 20,
                "sentence_count_range": (4, 8),
                "rules": [
                    "Answer: What does this mean?",
                    "Show real-world effects",
                    "Connect to bigger picture",
                    "Emotional or intellectual payoff",
                ],
            },
            {
                "name": "conclusion",
                "purpose": "Wrap up with soft CTA",
                "duration_percent": 10,
                "sentence_count_range": (1, 3),
                "rules": [
                    "Summarize key takeaway",
                    "Soft, natural CTA (no hard sell)",
                    "Leave with wonder or curiosity",
                    "Optional: tease related topic",
                ],
            },
        ],
    },

    "3_act_simple": {
        "description": "Simplified 3-act for shorter content",
        "sections": [
            {
                "name": "hook_and_context",
                "purpose": "Grab attention and set up topic",
                "duration_percent": 25,
                "sentence_count_range": (3, 5),
                "rules": ["Cold open", "Quick context", "Clear setup"],
            },
            {
                "name": "exploration",
                "purpose": "Deep dive into topic",
                "duration_percent": 60,
                "sentence_count_range": (12, 20),
                "rules": ["Multiple angles", "Specific examples", "Maintain pacing"],
            },
            {
                "name": "wrap_up",
                "purpose": "Conclusion and CTA",
                "duration_percent": 15,
                "sentence_count_range": (2, 4),
                "rules": ["Key takeaway", "Soft CTA"],
            },
        ],
    },
}

# Sentence style rules
SENTENCE_RULES = {
    "conversational": {
        "prefer_contractions": True,  # "it's" vs "it is"
        "use_second_person": True,  # "you" vs "one"
        "active_voice": True,  # "X did Y" vs "Y was done by X"
        "short_sentences": True,  # Max 20 words
        "avoid_jargon": True,
        "examples": [
            "Good: It's incredible how fast this happens.",
            "Bad: It is remarkable how expeditiously this occurs.",
            "Good: You can see this pattern everywhere.",
            "Bad: One can observe this pattern ubiquitously.",
        ],
    },

    "evergreen": {
        "no_dates": True,  # Avoid specific dates
        "no_current_events": True,  # No "recently", "this year"
        "no_temporal_refs": True,  # No "today", "now", "modern"
        "timeless_language": True,
        "violations": [
            "in 2024",
            "this year",
            "recently",
            "nowadays",
            "today's world",
            "currently",
            "right now",
            "this month",
            "last week",
        ],
    },

    "show_dont_tell": {
        "concrete_examples": True,  # Specific over abstract
        "analogies_encouraged": True,  # Comparisons help understanding
        "sensory_details": True,  # Visual, tactile descriptions
        "avoid_abstract": True,  # "beautiful" → "crystal-clear blue water"
        "examples": [
            "Good: The star explodes with the force of a billion nuclear bombs.",
            "Bad: The star has a very powerful explosion.",
            "Good: Imagine a grain of sand containing an entire library.",
            "Bad: It stores a lot of information.",
        ],
    },
}

# Pacing guidelines
PACING_RULES = {
    "cliffhanger_triggers": [
        "before_explanation",  # "But here's the weird part..."
        "mid_mechanism",  # "And then something unexpected happened..."
        "before_impact",  # "Wait until you see what this means..."
    ],

    "sentence_rhythm": {
        "vary_length": True,  # Mix short punchy + medium descriptive
        "avoid_monotony": True,  # Don't use same structure 3x in row
        "build_tension": True,  # Short → medium → long at climax
        "release_tension": True,  # Long → short after revelation
    },

    "transition_words": {
        "causation": ["because", "so", "that's why", "this means"],
        "contrast": ["but", "however", "instead", "on the other hand"],
        "sequence": ["first", "then", "next", "finally"],
        "emphasis": ["in fact", "actually", "remarkably", "incredibly"],
    },
}


def calculate_section_sentences(total_sentences: int, structure_name: str = "5_act") -> dict:
    """Calculate sentence distribution across sections."""
    structure = SCRIPT_STRUCTURE[structure_name]
    distribution = {}

    for section in structure["sections"]:
        percent = section["duration_percent"]
        min_range, max_range = section["sentence_count_range"]

        # Calculate based on percentage
        target = int(total_sentences * (percent / 100))

        # Clamp to range
        target = max(min_range, min(max_range, target))

        distribution[section["name"]] = target

    return distribution


def validate_sentence_style(sentence: str, rules: dict) -> tuple:
    """
    Validate sentence against style rules.

    Returns:
        (is_valid, issues_list)
    """
    issues = []

    # Check length
    word_count = len(sentence.split())
    if rules.get("max_sentence_length", 20) and word_count > rules["max_sentence_length"]:
        issues.append(f"Too long: {word_count} words (max {rules['max_sentence_length']})")

    # Check evergreen violations
    if rules.get("evergreen_only"):
        violations = SENTENCE_RULES["evergreen"]["violations"]
        for violation in violations:
            if violation in sentence.lower():
                issues.append(f"Temporal reference: '{violation}'")

    # Check cold open violations (for hook)
    if rules.get("cold_open"):
        from autoshorts.content.prompts.hook_patterns import COLD_OPEN_VIOLATIONS
        for violation in COLD_OPEN_VIOLATIONS:
            if violation in sentence.lower():
                issues.append(f"Meta-talk violation: '{violation}'")

    return len(issues) == 0, issues
