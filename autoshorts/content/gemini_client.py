# -*- coding: utf-8 -*-
"""
LLM Client - Multi-Provider Support (Gemini + Groq)
✅ Kanal mode'una göre içerik üretir (country_facts, history_story, vb.)
✅ Sub-topic rotation ile 6+ ay benzersiz içerik
✅ Her kanal konseptine uygun prompting
✅ Groq desteği (14.4K req/gün free tier!)
"""

import json
import logging
import random
import re
import os
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

# Gemini imports
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None
    types = None

# Groq imports (Alternative LLM - 14.4K req/day free tier!)
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None

# ✅ Import settings for TARGET_DURATION, MIN_SENTENCES, MAX_SENTENCES
from autoshorts.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ContentResponse:
    """Response from Gemini content generation"""
    hook: str
    script: List[str]
    cta: str
    search_queries: List[str]
    main_visual_focus: str
    metadata: Dict[str, Any]
    chapters: List[Dict[str, Any]]


def _build_legacy_mode_prompt(
    mode: str,
    topic: str,
    sub_topic: str,
    target_sentences: int,
    hook: str,
    cta: str
) -> str:
    """
    ✅ Mode ve sub-topic'e göre özelleştirilmiş prompt oluşturur
    """
    
    # Mode'a göre özel talimatlar
    mode_instructions = {
        "country_facts": f"""
You are creating content about a COUNTRY with focus on: {sub_topic}

SPECIFIC REQUIREMENTS:
- Choose ONE specific country (not covered recently)
- Focus angle: {sub_topic} (geography/culture/history/economy/technology/food/traditions/nature/urban_life/innovations)
- Include surprising facts and lesser-known details
- Show modern AND historical perspectives
- Use specific examples, not generalizations

AVOID: Generic statements like "this country is beautiful" or "people are friendly"
DO: "In 1995, Country X became the first to...", "The capital city has 7 metro lines covering..."
""",
        
        "history_story": f"""
You are creating content about a HISTORICAL EVENT/PERIOD with focus on: {sub_topic}

SPECIFIC REQUIREMENTS:
- Choose ONE specific historical event, person, or period
- Focus angle: {sub_topic} (battles/inventions/discoveries/leaders/everyday_life/art/architecture/trade/conflicts/cultural_exchange)
- Include dates, names, specific details
- Show cause and effect relationships
- Connect to modern relevance

AVOID: Vague generalizations
DO: "On March 15, 44 BC...", "This invention changed X industry by..."
""",
        
        "space_news": f"""
You are creating content about SPACE/ASTRONOMY with focus on: {sub_topic}

SPECIFIC REQUIREMENTS:
- Choose ONE specific mission, discovery, or space topic
- Focus angle: {sub_topic} (missions/discoveries/technology/planets/stars/commercial/research/telescopes/astronauts)
- Include technical details but explain simply
- Mention specific dates, organizations, spacecraft names
- Show why it matters

AVOID: Science fiction, speculation without basis
DO: "NASA's James Webb Telescope...", "SpaceX Starship completed..."
""",
        
        "movie_secrets": f"""
You are creating content about FILM PRODUCTION with focus on: {sub_topic}

SPECIFIC REQUIREMENTS:
- Choose ONE specific movie or filmmaking technique
- Focus angle: {sub_topic} (special effects/directing/cinematography/editing/sound design/stunts/production design/makeup)
- Include specific examples from real films
- Explain HOW techniques work
- Show evolution of the craft

AVOID: Plot spoilers, gossip about actors
DO: "In The Matrix (1999), the bullet-time effect was achieved by..."
""",
        
        "design_history": f"""
You are creating content about an EVERYDAY OBJECT'S DESIGN HISTORY with focus on: {sub_topic}

SPECIFIC REQUIREMENTS:
- Choose ONE specific everyday object (chair, pen, cup, etc.)
- Focus angle: {sub_topic} (origin/evolution/materials/famous designs/cultural impact/innovations/modern trends)
- Trace from earliest form to today
- Include specific designer names, years, movements
- Show how design reflects culture/technology

AVOID: "Things got better over time"
DO: "In 1859, Michael Thonet revolutionized chair production with..."
""",
        
        "if_lived_today": f"""
You are creating content about HISTORICAL FIGURES IN MODERN WORLD with focus on: {sub_topic}

SPECIFIC REQUIREMENTS:
- Choose ONE specific historical figure
- Focus angle: {sub_topic} (modern_tech/social_media/current_culture/contemporary_issues/modern_career)
- Imagine realistic scenarios based on their personality/values
- Show parallels between their era and today
- Be respectful and thoughtful

AVOID: Shallow comparisons, jokes at their expense
DO: "Given Leonardo da Vinci's curiosity about human anatomy, in 2025 he might..."
""",
        
        "nostalgia_story": f"""
You are creating content about POP CULTURE NOSTALGIA (1970s-1990s) with focus on: {sub_topic}

SPECIFIC REQUIREMENTS:
- Choose ONE specific trend, technology, or cultural phenomenon
- Focus angle: {sub_topic} (music/tv_shows/toys/fashion/technology/gaming/movies/social_trends)
- Include specific years, brands, names
- Evoke warm memories without being cringy
- Explain why it mattered then and why we remember it

AVOID: "Remember when life was simpler?"
DO: "The Walkman (1979) changed how people experienced music by..."
""",
        
        "cricket_women": f"""
You are creating content about WOMEN'S CRICKET with focus on: {sub_topic}

SPECIFIC REQUIREMENTS:
- Choose ONE specific player, match, or aspect of women's cricket
- Focus angle: {sub_topic} (skills/tactics/records/tournaments/team dynamics/training/equipment/history)
- Include statistics, dates, specific matches
- Celebrate achievement without condescension
- Show technical expertise

AVOID: Generic praise, comparisons to men's game
DO: "In the 2017 World Cup Final, Anya Shrubsole's 6/46 figures..."
"""
    }
    
    # Varsayılan talimat (mode tanımlı değilse)
    default_instruction = f"""
You are creating educational content about: {topic}

SPECIFIC REQUIREMENTS:
- Choose ONE specific aspect within this topic
- Focus angle: {sub_topic}
- Include specific facts, names, dates, examples
- Be educational but engaging
- Show why this matters

AVOID: Vague generalizations
DO: Use specific examples and details
"""
    
    mode_specific = mode_instructions.get(mode, default_instruction)
    
    prompt = f"""Create a HIGH-QUALITY, SEO-OPTIMIZED long-form YouTube video script.

CHANNEL CONCEPT: {topic}
CONTENT MODE: {mode}
TODAY'S FOCUS: {sub_topic}

{mode_specific}

STRUCTURE REQUIREMENTS:
- EXACTLY {target_sentences} sentences (70-90 range - CRITICAL for 1-hour timeout!)
- Each sentence: 15-18 words (REQUIRED - longer sentences reduce scene count!)
- CRITICAL: Write detailed, information-rich sentences with full context
- Pack more information per sentence (not choppy short sentences!)
- Educational but conversational tone
- Divided into 5-7 logical chapters

CRITICAL PERFORMANCE NOTE: 15-18 word sentences are MANDATORY for GitHub Actions timeout prevention. Target ~80 scenes total (not 120+). Longer sentences = fewer scenes = faster processing = no timeout!

VIDEO STRUCTURE:
1. HOOK (1-2 sentences): {hook}
2. INTRODUCTION (5-8 sentences): Set specific context with intrigue
3. MAIN CONTENT (30-55 sentences): 3-5 logical sections with smooth transitions
   - Each section should explore a different aspect
   - Use specific examples and details
   - Build narrative progression
4. CONCLUSION (3-5 sentences): Strong summary + key takeaway
5. CTA (1 sentence): {cta}

✅ CRITICAL SEO REQUIREMENTS:

**metadata.title** (50-60 chars) - VIRAL OPTIMIZATION:
- MUST be SPECIFIC to the exact topic you chose
- NOT generic channel description
- UNIQUE for THIS video
- Use proven viral title formulas:

FORMULA OPTIONS (choose one that fits your content):
  1. NUMBER + ADJECTIVE: "7 Bizarre Facts About [topic]"
  2. HOW/WHY: "How [entity] [shocking action] Without [expected thing]"
  3. TIMEFRAME: "What Happens When You [action] for 30 Days"
  4. COMPARISON: "[Thing] vs [Thing]: The Truth Will Shock You"
  5. REVEALED/EXPOSED: "The [adjective] Truth About [topic] Revealed"
  6. BANNED/FORBIDDEN: "Why [topic] is Banned in [number] Countries"
  7. MISTAKE: "You've Been [doing X] Wrong Your Whole Life"
  8. SECRET: "The Hidden Secret of [famous entity]"
  9. BEFORE/AFTER: "What [place/thing] Looked Like 100 Years Ago"
  10. UNEXPECTED: "[Topic] That Will Change How You See [broader topic]"

Real Examples:
  * country_facts: "7 Countries That Disappeared Overnight"
  * country_facts: "Why Japan's Trains Are Never Late (The Secret)"
  * history_story: "How Cleopatra Won Without Fighting"
  * space_news: "NASA Found Water Where Nobody Expected"
  * movie_secrets: "The $2 Trick That Saved Inception's Budget"
  * design_history: "Why Paper Clips Haven't Changed in 124 Years"

CRITICAL: Title MUST create curiosity gap + promise value + be specific

**chapters** (5-7 chapters):
- TITLE: SHORT (max 50 chars) - keyword-rich
- Use format: "Ancient Origins", "Technical Breakthrough", "Modern Impact"
- NOT: "Ancient Origins of the Thing We're Discussing Including..."

**metadata.description** (400-500 words):
- First 160 chars: Hook + specific topic + value promise
- Include compact chapter outline with keywords
- Add 2-3 related questions viewers might have
- End with CTA
- NO links, NO emojis, NO hashtags

**metadata.tags** (25-35 tags):
- Mix of: main subject, sub-topic, related terms, broader category
- All lowercase, no commas
- Include both broad ("history") and specific ("battle of actium") tags

**search_queries** (30-40 queries):
- SPECIFIC to the subject you chose
- Match the visual needs of YOUR specific content
- For "Japan railway": "shinkansen bullet train", "tokyo station platform", "japanese train conductor"
- NOT generic: "train station", "people commuting"

Return EXACT JSON structure:

{{
  "hook": "Your hook sentence",
  "script": ["sentence 1", "sentence 2", ... {target_sentences} sentences total],
  "cta": "Call to action",
  "search_queries": ["specific term 1", "term 2", ... 30-40 terms],
  "main_visual_focus": "Primary visual theme",
  "chapters": [
    {{"title": "Chapter Title", "start_sentence": 0, "end_sentence": 5, "description": "What this covers"}},
    ...
  ],
  "metadata": {{
    "title": "Your Specific Video Title",
    "description": "400-500 word description...",
    "tags": ["tag1", "tag2", ... 25-35 tags]
  }}
}}

REMEMBER: 
- Be SPECIFIC - choose ONE subject within the focus area
- Use REAL names, dates, numbers, facts
- NO generic content - every video must be about something SPECIFIC
- The sub-topic "{sub_topic}" should guide your angle, not limit your subject choice
"""
    
    return prompt


def _ultra_seo_metadata_fixed(
    metadata: Dict[str, Any],
    topic: str,
    script: List[str],
    chapters: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    ✅ SEO enhancement - başlık kontrolü
    """
    md = dict(metadata or {})
    
    title = (md.get("title") or "Untitled Video").strip()
    
    # Başlık uzunluğu kontrolü (STRICT 60 karakter)
    if len(title) > 60:
        cutoff = title[:57].rfind(' ')
        if cutoff > 45:
            title = title[:cutoff] + "..."
        else:
            title = title[:57] + "..."
    elif len(title) < 45:
        suffix = " Explained"
        if len(title + suffix) <= 60:
            title = title + suffix
    
    desc = (md.get("description") or "").strip()
    tags = md.get("tags") or []
    tags = [t.strip().lower() for t in tags if t and isinstance(t, str)][:35]
    
    md["title"] = title.strip()
    md["description"] = desc.strip()
    md["tags"] = list(dict.fromkeys(tags))[:35]
    
    return md


class GeminiClient:
    """LLM client for content generation - supports Gemini and Groq"""

    # Groq models (RECOMMENDED - 14.4K req/day free tier!)
    GROQ_MODELS = {
        "llama-3.1-8b-instant": "llama-3.1-8b-instant",  # 14.4K req/day - BEST FOR FREE
        "llama-3.3-70b-versatile": "llama-3.3-70b-versatile",  # 1K req/day - Higher quality
        "llama-3.1-70b-versatile": "llama-3.1-70b-versatile",  # Legacy
        "mixtral-8x7b-32768": "mixtral-8x7b-32768",  # Good alternative
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: str = "gemini",  # 'gemini' or 'groq'
        groq_api_key: Optional[str] = None,
        groq_api_keys: Optional[List[str]] = None,  # Multiple keys for rotation
        model: Optional[str] = None
    ):
        """
        Initialize LLM client.

        Args:
            api_key: Gemini API key (for backward compatibility)
            provider: LLM provider ('gemini' or 'groq')
            groq_api_key: Groq API key (if provider is 'groq')
            groq_api_keys: List of Groq API keys for rotation on rate limits
            model: Model name (optional override)
        """
        self.provider = provider

        if provider == "groq":
            # ✅ Support multiple Groq API keys for rate limit rotation
            self.groq_api_keys = groq_api_keys or ([groq_api_key] if groq_api_key else [])
            self.current_groq_key_index = 0

            if not self.groq_api_keys:
                raise ValueError("GROQ_API_KEY is required for Groq provider")
            if not GROQ_AVAILABLE:
                raise ImportError("groq package not installed. Run: pip install groq")

            # Initialize with first key
            current_key = self.groq_api_keys[0]
            self.groq_client = Groq(api_key=current_key)
            self.model = model or settings.GROQ_MODEL
            logger.info(f"✅ [Groq] API key 1/{len(self.groq_api_keys)}: {current_key[:10]}...{current_key[-4:]}")
            logger.info(f"🚀 [Groq] Model: {self.model} (500K tokens/day per key)")

            # ✅ Also initialize Gemini client for fallback
            self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
            if self.api_key and GEMINI_AVAILABLE:
                try:
                    self.client = genai.Client(api_key=self.api_key)
                    logger.info(f"✅ [Gemini] Fallback client ready")
                    # Fallback model chain
                    self.model_chain = ["gemini-2.5-flash-lite", "gemini-2.5-flash"]
                except Exception as e:
                    logger.warning(f"⚠️ Gemini fallback init failed: {e}")
                    self.client = None
                    self.model_chain = []
            else:
                self.client = None
                self.model_chain = []

        else:  # gemini
            self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY is required")
            if not GEMINI_AVAILABLE:
                raise ImportError("google-genai package not installed. Run: pip install google-genai")

            self.client = genai.Client(api_key=self.api_key)
            self.groq_client = None

            # Model zinciri
            env_chain = os.getenv("GEMINI_MODEL_CHAIN", "").strip()
            if env_chain:
                self.model_chain = [m.strip() for m in env_chain.split(",") if m.strip()]
            else:
                self.model_chain = [
                    "gemini-2.5-flash-lite",  # API limit dostu, hızlı
                    "gemini-2.5-flash",       # Fallback
                ]
            self.model = model or self.model_chain[0]

            logger.info(f"✅ [Gemini] API key: {self.api_key[:10]}...{self.api_key[-4:]}")
            logger.info(f"✅ [Gemini] Model: {self.model}")

        self.attempts_per_model = 2
        self.total_attempts = 4
        self.initial_backoff = 2.0
        self.max_backoff = 32.0
    
    def generate(
        self,
        topic: str,
        style: str = "educational",
        duration: Optional[int] = None,  # ✅ FIXED: Read from settings if not provided
        mode: Optional[str] = None,
        sub_topic: Optional[str] = None,
        additional_context: Optional[str] = None,
        script_style_config: Optional[Dict[str, Any]] = None,
    ) -> ContentResponse:
        """
        Generate content for a topic with mode and sub_topic support.

        Args:
            topic: Main channel topic/description
            style: Content style
            duration: Target duration in seconds (defaults to settings.TARGET_DURATION)
            mode: Content mode (country_facts, history_story, etc.)
            sub_topic: Specific angle to focus on
            additional_context: Extra context for prompt
            script_style_config: ScriptStyleConfig as dict (NEW - for enhanced prompts)
        """

        # ✅ FIXED: Read from settings if not provided (supports shorts vs long)
        if duration is None:
            duration = settings.TARGET_DURATION
            logger.debug(f"Using TARGET_DURATION from settings: {duration}s")

        # Calculate target sentences
        target_seconds = duration
        words_per_second = 2.5

        # ✅ PERFORMANCE OPTIMIZATION: Fewer scenes for faster processing
        # Long videos: 70-90 sentences (10 min)
        # Shorts: 10-20 sentences (60s)
        words_per_sentence = 15  # Ask Gemini for 15-18 word sentences
        target_sentences = int((target_seconds * words_per_second) / words_per_sentence)

        # ✅ FIXED: Use MIN_SENTENCES and MAX_SENTENCES from settings
        # This allows shorts (15) vs long (70-90) to work correctly
        min_sentences = settings.MIN_SENTENCES
        max_sentences = settings.MAX_SENTENCES
        target_sentences = max(min_sentences, min(max_sentences, target_sentences))

        logger.info(f"📊 Target sentences: {target_sentences} (range: {min_sentences}-{max_sentences}, duration: {duration}s)")

        # ✅ NEW: Use enhanced prompts if config provided
        use_enhanced_prompts = script_style_config is not None

        if use_enhanced_prompts:
            logger.info("🆕 Using ENHANCED PROMPTS for high-quality script generation")
            from autoshorts.content.prompts.enhanced_prompts import (
                build_enhanced_prompt,
                add_mode_specific_instructions,
            )

            # Build enhanced prompt
            prompt = build_enhanced_prompt(
                topic=topic,
                mode=mode or "educational",
                sub_topic=sub_topic,
                target_sentences=target_sentences,
                script_style_config=script_style_config,
            )

            # Add mode-specific instructions
            if mode:
                prompt = add_mode_specific_instructions(prompt, mode, sub_topic)

            logger.debug(f"Enhanced prompt built: {len(prompt)} chars")
        else:
            # ✅ FALLBACK: Use legacy prompt system
            logger.info("🔄 Using LEGACY PROMPTS (backward compatible)")
            prompt = self._build_legacy_prompt(
                topic=topic,
                mode=mode,
                sub_topic=sub_topic,
                target_sentences=target_sentences,
                style=style,
                additional_context=additional_context,
            )

        # ✅ Generate viral hooks and CTA (only for legacy prompts)
        if not use_enhanced_prompts:
            # Legacy system uses predefined hooks
            hooks = [
                # Shock/Surprise
                "Nobody expected what happened next.",
                "This fact will completely change how you see the world.",
                "What scientists discovered will blow your mind.",

                # Curiosity gap
                "The truth behind this is stranger than fiction.",
                "What if I told you everything you know is wrong?",
                "This secret has been hidden for decades.",

                # Numbers & specifics
                "This changed the lives of 2 billion people.",
                "In just 3 minutes, this will make you smarter.",
                "99% of people don't know this exists.",

                # Controversy/Mystery
                "Experts can't explain why this works.",
                "This is banned in 12 countries, but why?",
                "The government doesn't want you to know this.",

                # Personal/Relatable
                "You've been doing this wrong your entire life.",
                "This happens every day, but nobody notices.",
                "Your parents never told you about this.",

                # Urgency/Timing
                "This will disappear by 2030.",
                "Scientists say we have 10 years left.",
                "This is happening right now and nobody's talking about it."
            ]
            hook = random.choice(hooks)

            ctas = [
                "Thanks for watching! Subscribe to explore more fascinating stories.",
                "If you enjoyed this, hit subscribe for more deep dives.",
                "Want to learn more? Subscribe and join our community.",
                "Subscribe for more in-depth explorations."
            ]
            cta = random.choice(ctas)
        else:
            # Enhanced prompts: Gemini will generate hooks and CTA in the script
            hook = None
            cta = None
        
        # ✅ Build mode-specific prompt (only for legacy system)
        if not use_enhanced_prompts:
            if mode and sub_topic:
                logger.info(f"📝 Generating {mode} content with sub-topic: {sub_topic}")
                prompt = _build_mode_specific_prompt(
                    mode=mode,
                    topic=topic,
                    sub_topic=sub_topic,
                    target_sentences=target_sentences,
                    hook=hook,
                    cta=cta
                )
            else:
                logger.warning("⚠️ No mode/sub_topic provided, using generic prompt")
                prompt = f"""Create a video script about: {topic}
Target: {target_sentences} sentences
Hook: {hook}
CTA: {cta}
Return JSON with: hook, script, cta, search_queries, main_visual_focus, chapters, metadata"""
        
        if additional_context:
            prompt += f"\n\nAdditional context: {additional_context}"
        
        # Call API
        raw_response = self._call_api_with_fallback(prompt)
        
        # Parse response
        content_response = self._parse_response(raw_response, topic)
        
        # Enhance metadata
        content_response.metadata = _ultra_seo_metadata_fixed(
            metadata=content_response.metadata,
            topic=topic,
            script=content_response.script,
            chapters=content_response.chapters
        )
        
        logger.info(f"✅ Generated: {content_response.metadata.get('title', 'Untitled')}")
        
        return content_response
    
    def _call_api_with_fallback(
        self,
        prompt: str,
        max_output_tokens: int = 16000,
        temperature: float = 0.8
    ) -> str:
        """
        Call API with retry + fallback logic.

        Priority: Groq → Gemini (if Groq fails)
        """
        # Try Groq first (if configured)
        if self.provider == "groq" and self.groq_client:
            try:
                return self._call_groq_api(prompt, temperature)
            except Exception as e:
                logger.warning(f"⚠️ Groq failed, falling back to Gemini: {e}")
                # Fallback to Gemini if available
                if GEMINI_AVAILABLE and self.client:
                    logger.info("🔄 Switching to Gemini as fallback...")
                    return self._call_gemini_api(prompt, max_output_tokens, temperature)
                else:
                    raise RuntimeError(f"Groq failed and Gemini not available: {e}")

        # Use Gemini directly
        return self._call_gemini_api(prompt, max_output_tokens, temperature)

    def _call_groq_api(self, prompt: str, temperature: float = 0.8) -> str:
        """
        Make API call to Groq with multi-key rotation on rate limits.

        Rate limit (429) → Try next API key → If all exhausted, raise error
        """
        keys_tried = set()

        while len(keys_tried) < len(self.groq_api_keys):
            current_key = self.groq_api_keys[self.current_groq_key_index]
            key_num = self.current_groq_key_index + 1

            for attempt in range(self.total_attempts):
                try:
                    logger.info(f"[Groq] Key {key_num}/{len(self.groq_api_keys)}, Attempt {attempt+1}/{self.total_attempts}")

                    response = self.groq_client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a viral content creator for YouTube long-form videos. Always respond with valid JSON only, no markdown blocks."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        temperature=temperature,
                        max_tokens=16000,
                        top_p=0.95,
                    )

                    if response.choices and response.choices[0].message.content:
                        text = response.choices[0].message.content.strip()
                        # Sanitize control characters
                        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
                        logger.info(f"✅ [Groq] Success with key {key_num}, model {self.model}")
                        return text

                    raise RuntimeError("Empty response from Groq")

                except Exception as e:
                    error_str = str(e)
                    logger.warning(f"❌ [Groq] Key {key_num}, Attempt {attempt+1} failed: {e}")

                    # ✅ Check for rate limit error (429)
                    is_rate_limit = "429" in error_str or "rate_limit" in error_str.lower()

                    if is_rate_limit:
                        logger.warning(f"⚠️ [Groq] Rate limit hit on key {key_num}")
                        keys_tried.add(self.current_groq_key_index)

                        # Try next key if available
                        if len(keys_tried) < len(self.groq_api_keys):
                            self.current_groq_key_index = (self.current_groq_key_index + 1) % len(self.groq_api_keys)
                            next_key = self.groq_api_keys[self.current_groq_key_index]
                            logger.info(f"🔄 [Groq] Switching to key {self.current_groq_key_index + 1}/{len(self.groq_api_keys)}")
                            self.groq_client = Groq(api_key=next_key)
                            break  # Break inner loop to try new key
                        else:
                            # All keys exhausted
                            raise RuntimeError(f"All {len(self.groq_api_keys)} Groq API keys rate limited")

                    # Not a rate limit error, try again with backoff
                    if attempt < self.total_attempts - 1:
                        time.sleep(self.initial_backoff * (attempt + 1))
                    continue
            else:
                # Inner loop completed without break (all attempts failed, not rate limit)
                keys_tried.add(self.current_groq_key_index)
                if len(keys_tried) < len(self.groq_api_keys):
                    self.current_groq_key_index = (self.current_groq_key_index + 1) % len(self.groq_api_keys)
                    next_key = self.groq_api_keys[self.current_groq_key_index]
                    logger.info(f"🔄 [Groq] Switching to key {self.current_groq_key_index + 1}/{len(self.groq_api_keys)}")
                    self.groq_client = Groq(api_key=next_key)

        raise RuntimeError(f"All Groq API keys exhausted ({len(self.groq_api_keys)} keys tried)")

    def _call_gemini_api(
        self,
        prompt: str,
        max_output_tokens: int = 16000,
        temperature: float = 0.8
    ) -> str:
        """Make API call to Gemini"""
        attempt = 0
        backoff = self.initial_backoff

        for model_name in self.model_chain:
            for _ in range(self.attempts_per_model):
                attempt += 1
                if attempt > self.total_attempts:
                    raise RuntimeError(f"All {self.total_attempts} attempts failed")

                try:
                    logger.info(f"[Gemini] Attempt {attempt}/{self.total_attempts} with {model_name}")

                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=temperature,
                            max_output_tokens=max_output_tokens,
                            response_mime_type="application/json"
                        )
                    )

                    text = response.text.strip()
                    if not text:
                        raise ValueError("Empty response from Gemini")

                    # Sanitize control characters from JSON
                    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)

                    logger.info(f"✅ [Gemini] Success with {model_name}")
                    return text

                except Exception as e:
                    logger.warning(f"❌ [Gemini] Attempt {attempt} failed: {e}")
                    if attempt < self.total_attempts:
                        # 404/NOT_FOUND ise bekleme yapmadan sıradaki modele geç
                        if "404" in str(e) or "NOT_FOUND" in str(e):
                            continue
                        # 429 için sunucunun önerdiği gecikmeyi uygula
                        delay = self._extract_retry_after_seconds(e)
                        if delay:
                            backoff = max(backoff, float(delay))
                        logger.info(f"⏳ Waiting {backoff:.1f}s before retry...")
                        time.sleep(backoff)
                        backoff = min(backoff * 2, self.max_backoff)
                    continue

        raise RuntimeError("All Gemini API attempts exhausted")
    
    @staticmethod
    def _extract_retry_after_seconds(err: Exception) -> Optional[float]:
        """
        429 mesajlarından retry süresini çıkar:
        - "... 'retryDelay': '59s' ..."
        - "... Please retry in 59.55s."
        """
        s = str(err)
        m = re.search(r"retryDelay['\":\s]*'?(?P<sec>\d+)s'?", s)
        if m:
            try:
                return float(m.group("sec"))
            except Exception:
                pass
        m = re.search(r"Please retry in (?P<sec>[0-9]+(?:\.[0-9]+)?)s", s)
        if m:
            try:
                return float(m.group("sec"))
            except Exception:
                pass
        return None

    def _parse_response(self, raw_json: str, topic: str) -> ContentResponse:
        """Parse JSON response from Gemini"""
        try:
            # Clean markdown formatting
            raw_json = raw_json.replace("```json", "").replace("```", "").strip()
            
            data = json.loads(raw_json)
            
            return ContentResponse(
                hook=data.get("hook", ""),
                script=data.get("script", []),
                cta=data.get("cta", ""),
                search_queries=data.get("search_queries", []),
                main_visual_focus=data.get("main_visual_focus", ""),
                metadata=data.get("metadata", {}),
                chapters=data.get("chapters", [])
            )
        except Exception as e:
            logger.error(f"Failed to parse Gemini response: {e}")
            logger.debug(f"Raw response: {raw_json[:500]}")
            raise ValueError(f"Invalid JSON response from Gemini: {e}")

    def _build_legacy_prompt(
        self,
        topic: str,
        mode: Optional[str],
        sub_topic: Optional[str],
        target_sentences: int,
        style: str,
        additional_context: Optional[str],
    ) -> str:
        """Build legacy prompt (backward compatible)."""
        # Generate hooks and CTA
        hooks = [
            "Nobody expected what happened next.",
            "This fact will completely change how you see the world.",
            "The truth behind this is stranger than fiction.",
        ]
        hook = random.choice(hooks)

        ctas = [
            "That's the fascinating story behind this phenomenon.",
            "And that's why this topic continues to amaze us.",
            "The world is full of incredible stories like this.",
        ]
        cta = random.choice(ctas)

        # Build mode-specific prompt if mode provided
        if mode and sub_topic:
            return _build_legacy_mode_prompt(
                mode=mode,
                topic=topic,
                sub_topic=sub_topic,
                target_sentences=target_sentences,
                hook=hook,
                cta=cta,
            )
        else:
            # Generic prompt
            return f"""
You are creating engaging video content about: {topic}

Generate a script with exactly {target_sentences} sentences.

Hook: {hook}
CTA: {cta}

Return JSON with structure matching mode_specific prompts.
"""
