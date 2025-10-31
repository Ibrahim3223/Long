# -*- coding: utf-8 -*-
"""
Gemini Client - ENHANCED VERSION with Mode & Sub-Topic Support
✅ Kanal mode'una göre içerik üretir (country_facts, history_story, vb.)
✅ Sub-topic rotation ile 6+ ay benzersiz içerik
✅ Her kanal konseptine uygun prompting
"""

import json
import logging
import random
import re
import os
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

try:
    from google import genai
    from google.genai import types
except ImportError as e:
    raise ImportError(
        "google-genai paketi bulunamadı. Lütfen yükleyin: pip install google-genai>=0.2.0"
    ) from e

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


def _build_mode_specific_prompt(
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
- EXACTLY {target_sentences} sentences (40-70 range)
- Each sentence: 8-12 words (clear, concise, engaging)
- Educational but conversational tone
- Divided into 5-7 logical chapters

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

**metadata.title** (50-60 chars):
- MUST be SPECIFIC to the exact topic you chose
- NOT generic channel description
- UNIQUE for THIS video
- Include key subject + angle
- Examples for different modes:
  * country_facts: "Japan's Rail System: 99.9% On-Time Record"
  * history_story: "Cleopatra's Naval Victory at Actium"
  * space_news: "Webb Telescope Finds Water on Exoplanet K2-18b"
  * movie_secrets: "How Inception Created Zero-G Hallway Fight"
  * design_history: "Paper Clip: Johan Vaaler's 1899 Revolution"

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
    """Gemini API client for content generation"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini client"""
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required")
        
        self.client = genai.Client(api_key=self.api_key)
        
        # Güncel ve destekli model zinciri
        # (İstersen GEMINI_MODEL_CHAIN env ile "m1,m2,m3" şeklinde override edebilirsin)
        env_chain = os.getenv("GEMINI_MODEL_CHAIN", "").strip()
        if env_chain:
            self.model_chain = [m.strip() for m in env_chain.split(",") if m.strip()]
        else:
            self.model_chain = [
                "gemini-2.5-flash",       # Önerilen varsayılan
                "gemini-2.0-flash-exp",   # Deneysel, farklı kota olabilir
            ]
        
        self.attempts_per_model = 2
        self.total_attempts = 4
        self.initial_backoff = 2.0
        self.max_backoff = 32.0
        
        logger.info(f"✅ GeminiClient initialized with {len(self.model_chain)} models")
    
    def generate(
        self,
        topic: str,
        style: str = "educational",
        duration: int = 180,
        mode: Optional[str] = None,
        sub_topic: Optional[str] = None,
        additional_context: Optional[str] = None
    ) -> ContentResponse:
        """
        Generate content for a topic with mode and sub_topic support
        
        Args:
            topic: Main channel topic/description
            style: Content style
            duration: Target duration in seconds
            mode: Content mode (country_facts, history_story, etc.)
            sub_topic: Specific angle to focus on
            additional_context: Extra context for prompt
        """
        
        # Calculate target sentences
        target_seconds = duration
        words_per_second = 2.5
        words_per_sentence = 10
        target_sentences = int((target_seconds * words_per_second) / words_per_sentence)
        target_sentences = max(40, min(70, target_sentences))
        
        # Generate hooks and CTAs
        hooks = [
            "The story behind this is more fascinating than you'd imagine.",
            "What if I told you this changed the world?",
            "This has a hidden history most people never learn.",
            "The evolution reveals surprising innovations.",
            "Behind this lies an unexpected journey of ingenuity.",
            "The transformation tells a remarkable story.",
            "Few people know the revolutionary origins of this.",
            "This development reveals fascinating insights."
        ]
        hook = random.choice(hooks)
        
        ctas = [
            "Thanks for watching! Subscribe to explore more fascinating stories.",
            "If you enjoyed this, hit subscribe for more deep dives.",
            "Want to learn more? Subscribe and join our community.",
            "Subscribe for more in-depth explorations."
        ]
        cta = random.choice(ctas)
        
        # ✅ Build mode-specific prompt
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
        """Call API with retry + fallback logic"""
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
                    
                    # ✅ Sanitize control characters from JSON
                    import re
                    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)  # Remove control chars
                    
                    logger.info(f"✅ Success with {model_name}")
                    return text
                    
                except Exception as e:
                    logger.warning(f"❌ Attempt {attempt} failed: {e}")
                    if attempt < self.total_attempts:
                        # 404/NOT_FOUND ise bekleme yapmadan sıradaki modele geç
                        if "404" in str(e) or "NOT_FOUND" in str(e):
                            continue
                        # 429 için sunucunun önerdiği gecikmeyi uygula (RetryInfo / 'Please retry in Xs')
                        delay = self._extract_retry_after_seconds(e)
                        if delay:
                            backoff = max(backoff, float(delay))
                        logger.info(f"⏳ Waiting {backoff:.1f}s before retry...")
                        time.sleep(backoff)
                        backoff = min(backoff * 2, self.max_backoff)
                    continue
        
        raise RuntimeError("All API attempts exhausted")
    
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
