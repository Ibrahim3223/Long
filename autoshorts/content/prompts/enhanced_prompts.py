# -*- coding: utf-8 -*-
"""
Enhanced Prompt Builder for High-Quality Scripts
✅ Integrates hook patterns, script structure, style rules
✅ Config-driven prompt generation
✅ Quality-focused instructions
"""
import logging
from typing import Optional, Dict, Any
from autoshorts.content.prompts.hook_patterns import get_hook_pattern, get_cta_pattern
from autoshorts.content.prompts.script_templates import SCRIPT_STRUCTURE, SENTENCE_RULES

logger = logging.getLogger(__name__)


def build_enhanced_prompt(
    topic: str,
    mode: str,
    sub_topic: Optional[str],
    target_sentences: int,
    script_style_config: Dict[str, Any],
) -> str:
    """
    Build enhanced prompt with quality rules.

    Args:
        topic: Main topic
        mode: Content mode (educational, storytelling, etc.)
        sub_topic: Specific angle
        target_sentences: Target sentence count
        script_style_config: ScriptStyleConfig as dict

    Returns:
        Enhanced prompt string
    """
    # Extract config
    hook_intensity = script_style_config.get("hook_intensity", "high")
    cold_open = script_style_config.get("cold_open", True)
    use_modular = script_style_config.get("use_modular_structure", True)
    max_sentence_length = script_style_config.get("max_sentence_length", 20)
    conversational = script_style_config.get("conversational_tone", True)
    evergreen = script_style_config.get("evergreen_only", True)
    cta_softness = script_style_config.get("cta_softness", "medium")

    # Build prompt sections
    prompt_parts = []

    # ========== ROLE & MISSION ==========
    prompt_parts.append(f"""
You are a WORLD-CLASS video scriptwriter specializing in {mode} content.

Your mission: Create a captivating {target_sentences}-sentence script about {topic}{f' (focus: {sub_topic})' if sub_topic else ''}.

This script will be:
- Read aloud by professional TTS
- Paired with carefully selected stock footage
- Watched by curious viewers seeking engaging, accessible content
- Optimized for retention and watch time

""")

    # ========== STRUCTURE ==========
    if use_modular:
        structure = SCRIPT_STRUCTURE["5_act"]
        from autoshorts.content.prompts.script_templates import calculate_section_sentences
        distribution = calculate_section_sentences(target_sentences, "5_act")

        prompt_parts.append(f"""
SCRIPT STRUCTURE (Modular 5-Act):

1. HOOK ({distribution.get('hook', 2)} sentences) - First 5 seconds
   - Start with the MOST shocking/interesting fact
   - NO meta-talk! Don't say "this video", "today we'll", "let me show you"
   - Jump directly into the topic
   - Create immediate curiosity or surprise
   - Intensity level: {hook_intensity}
   - Example patterns:
     * "This [entity] [shocking action] in [timeframe]."
     * "[Number] [things] [dramatic outcome] because of this."
     * "Everything you know about [topic] is wrong."

2. CONTEXT ({distribution.get('context', 5)} sentences)
   - What is this?
   - Why should viewers care?
   - Necessary background (but don't lecture)
   - Set up the main story

3. MECHANISM ({distribution.get('mechanism', 12)} sentences)
   - How does it work? OR What happened?
   - Break down complexity into simple steps
   - Use analogies: "Imagine...", "It's like..."
   - Include SPECIFIC details (numbers, names, examples)
   - CRITICAL: Every 20-30 seconds, add a mini cliffhanger:
     * "But that's not the strangest part."
     * "Wait until you hear what comes next."
     * "And then something unexpected happened."

4. IMPACT ({distribution.get('impact', 6)} sentences)
   - What does this mean?
   - Real-world consequences
   - Why it matters
   - Emotional or intellectual payoff

5. CONCLUSION ({distribution.get('conclusion', 2)} sentences)
   - Key takeaway sentence (wrap up the topic)
   - CTA sentence with subscribe/like reminder
   - Softness: {cta_softness}

   CRITICAL CTA REQUIREMENTS:
   - MUST include: subscribe reminder ("subscribe for more")
   - SHOULD include: like/comment encouragement
   - Keep natural tone (not pushy/salesy)
   - Examples:
     * "If you found this fascinating, subscribe for more incredible stories like this."
     * "Subscribe to explore more amazing discoveries. And let me know in the comments what fascinates you most."
     * "Want more mind-blowing facts? Hit subscribe and join our journey of discovery."

""")

    # ========== SENTENCE STYLE ==========
    prompt_parts.append(f"""
SENTENCE STYLE RULES (CRITICAL):

✅ DO:
- TARGET 12-15 words per sentence (optimal for pacing & performance!)
- MAX {max_sentence_length} words per sentence (caption-friendly!)
- IMPORTANT: Write full, complete sentences with context
- Each sentence should convey a complete thought
- Use conversational tone: "you", "it's" (contractions), active voice
- Show, don't tell: "crystal-clear blue water" NOT "beautiful water"
- Use specific examples: "explodes with force of billion nuclear bombs" NOT "very powerful"
- Use analogies: "Imagine a grain of sand containing an entire library"
- Vary sentence rhythm: mix short punchy (8-10 words) + medium descriptive (12-15 words)
- Active voice: "Scientists discovered X" NOT "X was discovered"

❌ DON'T:
- Write too-short sentences (less than 8 words) - they feel choppy
- Academic/formal tone: "one can observe" → use "you can see"
- Abstract concepts without examples
- Passive voice
- Jargon without explanation
- Monotonous rhythm (same length/structure 3x in row)

PERFORMANCE NOTE: 12-15 word sentences create better pacing and allow for more detailed, engaging storytelling while keeping processing efficient.

""")

    # ========== EVERGREEN RULES ==========
    if evergreen:
        prompt_parts.append("""
EVERGREEN CONTENT RULES (CRITICAL):

This content must be timeless. ABSOLUTELY FORBIDDEN:
❌ Specific dates: "in 2024", "this year", "last month"
❌ Temporal words: "recently", "nowadays", "currently", "today's world", "right now"
❌ Current events or news
❌ "Modern" unless comparing to historical

✅ Instead use:
- "For centuries..."
- "Scientists discovered..."
- "This phenomenon occurs when..."
- Historical references are OK if context is clear: "In 1969, when humans first..."

""")

    # ========== OUTPUT FORMAT ==========
    prompt_parts.append(f"""
OUTPUT FORMAT:

Return ONLY valid JSON (no markdown, no code blocks):

{{
  "hook": "Your powerful opening sentence (cold open - NO meta-talk!)",
  "script": [
    "Sentence 1",
    "Sentence 2",
    ...
    (total {target_sentences} sentences including hook and CTA)
  ],
  "cta": "Your soft closing sentence",
  "search_queries": ["keyword1", "keyword2", ...],
  "main_visual_focus": "Primary visual theme",
  "chapters": [
    {{
      "title": "Chapter Title",
      "start_sentence_index": 0,
      "end_sentence_index": 10,
      "search_queries": ["chapter-specific", "keywords"]
    }}
  ],
  "metadata": {{
    "title": "Engaging video title (55-65 chars)",
    "description": "SEO-optimized description",
    "tags": ["tag1", "tag2", ...]
  }}
}}

QUALITY CHECKLIST before submitting:
✓ Hook starts IMMEDIATELY with topic (no "this video", "today we")
✓ Every sentence ≤ {max_sentence_length} words
✓ NO temporal references (dates, "recently", "nowadays")
✓ Conversational tone (contractions, "you", active voice)
✓ Specific examples and analogies
✓ Mini cliffhangers every 4-6 sentences
✓ CTA includes "subscribe" + engagement (like/comment)
✓ {target_sentences} total sentences

""")

    # Combine all parts
    final_prompt = "".join(prompt_parts)

    logger.debug(f"Built enhanced prompt: {len(final_prompt)} chars, target={target_sentences} sentences")

    return final_prompt


def add_mode_specific_instructions(prompt: str, mode: str, sub_topic: Optional[str]) -> str:
    """Add mode-specific instructions to base prompt."""

    mode_additions = {
        "country_facts": f"""
MODE: Country Facts {f'(Focus: {sub_topic})' if sub_topic else ''}

Additional Requirements:
- Choose ONE specific country (not covered recently)
- Focus angle: {sub_topic or 'general interesting facts'}
- Include surprising, lesser-known details
- Show modern AND historical perspectives
- Use specific examples: "In 1995, Country X became the first to..."
- Avoid generic statements: "people are friendly" → "locals greet strangers with [specific custom]"

""",
        "history_story": f"""
MODE: Historical Story {f'(Focus: {sub_topic})' if sub_topic else ''}

Additional Requirements:
- Choose ONE specific event, person, or period
- Include dates, names, specific details
- Show cause and effect
- Connect to modern relevance
- Example: "On March 15, 44 BC..." not "A long time ago..."

""",
        "science": f"""
MODE: Science Explainer {f'(Focus: {sub_topic})' if sub_topic else ''}

Additional Requirements:
- Choose ONE specific phenomenon, discovery, or concept
- Explain technical details simply
- Use analogies extensively
- Include scale comparisons: "If X was the size of Earth, Y would be..."
- Avoid jargon or explain it immediately

""",
        "documentary": f"""
MODE: Documentary {f'(Focus: {sub_topic})' if sub_topic else ''}

Additional Requirements:
- Narrative storytelling approach
- Build tension and resolution
- Include expert perspectives (paraphrase, don't quote)
- Show multiple angles
- Cinematic descriptions

""",
    }

    addition = mode_additions.get(mode, "")
    if addition:
        # Insert after role & mission
        parts = prompt.split("SCRIPT STRUCTURE")
        if len(parts) == 2:
            return parts[0] + addition + "SCRIPT STRUCTURE" + parts[1]

    return prompt
