# -*- coding: utf-8 -*-
"""
Gemini Client - FIXED VERSION
✅ Başlıklar benzersiz (channel topic KULLANILMAZ)
✅ Chapter başlıkları KISA (max 50 karakter)
✅ SEO-optimized
✅ Import ve class tanımları eklendi
"""

import json
import logging
import random
import re
import os
import time
import difflib
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

# ✅ Gemini SDK imports
try:
    from google import genai
    from google.genai import types
except ImportError as e:
    raise ImportError(
        "google-genai paketi bulunamadı. Lütfen yükleyin: pip install google-genai>=0.2.0"
    ) from e

logger = logging.getLogger(__name__)


# ✅ ContentResponse dataclass tanımı
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


def _ultra_seo_metadata_fixed(
    metadata: Dict[str, Any],
    topic: str,
    script: List[str],
    chapters: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    ✅ FIXED: ULTRA SEO enhancement
    - Başlık UNIQUE (channel topic kullanılmaz)
    - Chapter başlıkları KISA
    """
    md = dict(metadata or {})
    
    # ❌ ESKİ: Channel topic'i başlığa ekleme
    # primary_kw = topic.split("—")[0].strip()
    # title = f"{primary_kw}: {title}"
    
    # ✅ YENİ: Sadece Gemini'den gelen başlığı kullan
    title = (md.get("title") or "Untitled Video").strip()
    
    # ✅ Başlık uzunluğu kontrolü (STRICT 60 karakter)
    if len(title) > 60:
        # Son kelimeyi yarım bırakma, tam kelimede kes
        cutoff = title[:57].rfind(' ')
        if cutoff > 45:
            title = title[:cutoff] + "..."
        else:
            title = title[:57] + "..."
    
    # Çok kısa başlıkları uzat
    elif len(title) < 45:
        suffix = " Explained"
        if len(title + suffix) <= 60:
            title = title + suffix
    
    # ✅ DESCRIPTION
    desc = (md.get("description") or "").strip()
    
    # ✅ TAGS
    tags = md.get("tags") or []
    tags = [t.strip().lower() for t in tags if t and isinstance(t, str)][:35]
    
    md["title"] = title.strip()
    md["description"] = desc.strip()
    md["tags"] = list(dict.fromkeys(tags))[:35]
    
    return md


def _build_ultra_seo_prompt_fixed(
    topic: str,
    style: str,
    target_sentences: int,
    hook: str,
    cta: str
) -> str:
    """
    ✅ FIXED: Ultra SEO prompt
    - Başlık için NET talimat: Channel topic KULLANMA
    - Chapter başlıkları KISA
    """
    
    prompt = f"""Create a HIGH-QUALITY, SEO-OPTIMIZED long-form YouTube video script about: {topic}

CRITICAL REQUIREMENTS:
- EXACTLY {target_sentences} sentences (40-70 range)
- Each sentence: 8-12 words (clear, concise, engaging)
- Educational but conversational tone
- Divided into 5-7 logical chapters

STRUCTURE:
1. HOOK (1-2 sentences): {hook}
2. INTRODUCTION (5-8 sentences): Set context with intrigue
3. MAIN CONTENT (30-55 sentences): 3-5 logical sections with smooth transitions
4. CONCLUSION (3-5 sentences): Strong summary + key takeaway
5. CTA (1 sentence): {cta}

✅ ULTRA SEO REQUIREMENTS (CRITICAL):

**metadata.title** (STRICT 50-60 chars):
- SPECIFIC to THIS EXACT video topic
- DO NOT repeat channel name or channel topic
- UNIQUE for every video (not generic!)
- Mobile-optimized: MAXIMUM 60 characters
- Examples:
  ✅ GOOD: "Chair Evolution: From Throne to Modern Eames"
  ✅ GOOD: "Coffee Discovery: Ethiopian Goat Herders Tale"
  ✅ GOOD: "Paper Clip History: Wire Bending Innovation"
  ❌ BAD: "Design histories of everyday objects: Chair..." (repeating channel topic!)
  ❌ BAD: "Interesting facts about chairs" (too generic!)

**chapters** (5-7 chapters):
- TITLE: SHORT (max 50 chars) - keyword-rich but concise
- TITLE: Use format like "Ancient Origins" or "Medieval Evolution" or "Modern Innovation"
- TITLE: NOT "Ancient Chair Origins: Power and Status Seating – Explore the earliest..."
- DESCRIPTION: Longer explanation (for internal use, not shown to user)
- Examples:
  ✅ GOOD: {{"title": "Ancient Origins", "description": "Explore earliest chair forms"}}
  ✅ GOOD: {{"title": "Industrial Revolution", "description": "Mass production impact"}}
  ❌ BAD: {{"title": "Ancient Chair Origins: Power and Status Seating – Explore...", ...}}

**metadata.description** (400-500 words):
- First 160 chars: Hook + primary keyword + value promise
- Include compact chapter outline with keywords
- Add 2-3 related questions viewers might have
- End with CTA
- NO links, NO emojis, NO hashtags in body

**metadata.tags** (25-35 tags):
- Mix of broad, specific, and related terms
- All lowercase, no commas
- NO generic words ("video", "youtube", "watch")

**search_queries** (30-40 queries):
- SPECIFIC to each scene/chapter
- Mix macro + detail shots
- Examples: "wooden chair close up", "throne ancient egypt", "modern office chair"

Return EXACT JSON structure:

{{
  "hook": "Your hook sentence",
  "script": ["sentence 1", "sentence 2", ... {target_sentences} sentences total],
  "cta": "Call to action",
  "search_queries": ["specific term 1", "term 2", ... 30-40 terms],
  "main_visual_focus": "Primary visual theme",
  "chapters": [
    {{"title": "Introduction", "start_sentence": 0, "end_sentence": 5, "description": "Overview"}},
    {{"title": "Ancient Origins", "start_sentence": 6, "end_sentence": 12, "description": "Earliest forms"}},
    {{"title": "Medieval Era", "start_sentence": 13, "end_sentence": 20, "description": "Middle Ages development"}},
    {{"title": "Industrial Revolution", "start_sentence": 21, "end_sentence": 28, "description": "Mass production"}},
    {{"title": "Modern Design", "start_sentence": 29, "end_sentence": 36, "description": "20th century"}},
    {{"title": "Contemporary Trends", "start_sentence": 37, "end_sentence": 42, "description": "Current innovations"}},
    {{"title": "Conclusion", "start_sentence": 43, "end_sentence": 47, "description": "Summary"}}
  ],
  "metadata": {{
    "title": "Chair Evolution: From Throne to Modern Eames",
    "description": "400-500 word description...",
    "tags": ["chair history", "furniture design", "eames chair", ... 25-35 tags]
  }}
}}

CRITICAL REMINDERS:
- Title must be UNIQUE for THIS video (not generic channel description)
- Chapter titles must be SHORT (max 50 chars)
- NO channel topic in video title
- Quality > Quantity
"""
    return prompt


# ✅ GeminiClient class tanımı
class GeminiClient:
    """Gemini API client for content generation"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini client"""
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required")
        
        # Initialize Gemini client
        self.client = genai.Client(api_key=self.api_key)
        
        # Model chain for fallback
        self.model_chain = [
            "gemini-2.0-flash-exp",
            "gemini-1.5-flash", 
            "gemini-1.5-flash-002"
        ]
        
        # Retry configuration
        self.attempts_per_model = 2
        self.total_attempts = len(self.model_chain) * self.attempts_per_model
        self.initial_backoff = 2.0
        self.max_backoff = 32.0
        
        logger.info(f"✅ GeminiClient initialized with {len(self.model_chain)} models")
    
    def generate(
        self,
        topic: str,
        style: str = "educational",
        duration: int = 180,
        additional_context: Optional[str] = None
    ) -> ContentResponse:
        """Generate content for a topic"""
        
        # Calculate target sentences based on duration
        # ~10-12 words per sentence, ~2.5 words per second
        target_seconds = duration
        words_per_second = 2.5
        words_per_sentence = 10
        target_sentences = int((target_seconds * words_per_second) / words_per_sentence)
        target_sentences = max(40, min(70, target_sentences))
        
        # Generate hooks and CTAs for LONG-FORM content
        hooks = [
            "The story behind this everyday object is more fascinating than you'd imagine.",
            "What if I told you the origin of this common item changed the world?",
            "This object has a hidden history that most people never learn about.",
            "The evolution of this design reveals surprising innovations across centuries.",
            "Behind this familiar object lies an unexpected journey of human ingenuity.",
            "The transformation of this everyday item tells a remarkable story.",
            "Few people know the revolutionary origins of this common design.",
            "This object's development reveals fascinating insights into human creativity."
        ]
        hook = random.choice(hooks)
        
        ctas = [
            "Thanks for watching! Subscribe to explore more fascinating design histories.",
            "If you enjoyed this deep dive, hit subscribe for more untold stories.",
            "Want to learn more hidden histories? Subscribe and join our community.",
            "Subscribe for more in-depth explorations of everyday innovations."
        ]
        cta = random.choice(ctas)
        
        # Build prompt
        prompt = _build_ultra_seo_prompt_fixed(
            topic=topic,
            style=style,
            target_sentences=target_sentences,
            hook=hook,
            cta=cta
        )
        
        if additional_context:
            prompt += f"\n\nAdditional context: {additional_context}"
        
        # Call API with fallback
        raw_response = self._call_api_with_fallback(prompt)
        
        # Parse response
        content_response = self._parse_response(raw_response, topic)
        
        # Enhance metadata with SEO
        content_response.metadata = _ultra_seo_metadata_fixed(
            metadata=content_response.metadata,
            topic=topic,
            script=content_response.script,
            chapters=content_response.chapters
        )
        
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
                            response_mime_type="text/plain",
                        )
                    )

                    if not response or not response.text:
                        raise ValueError("Empty response from API")

                    logger.info(f"[Gemini] ✅ Success with {model_name}")
                    return response.text

                except Exception as e:
                    error_str = str(e).lower()
                    is_retryable = any(
                        code in error_str for code in ["429", "500", "502", "503", "504"]
                    ) or any(
                        status in error_str for status in ["unavailable", "deadline", "resource"]
                    )

                    if is_retryable and attempt < self.total_attempts:
                        logger.warning(f"[Gemini] Retry in {backoff}s: {str(e)[:100]}")
                        time.sleep(backoff)
                        backoff = min(backoff * 1.5, self.max_backoff)
                    else:
                        logger.error(f"[Gemini] Failed with {model_name}: {str(e)[:100]}")
                        if attempt >= self.total_attempts:
                            raise

        raise RuntimeError("All models exhausted")

    def _parse_response(self, raw_text: str, topic: str) -> ContentResponse:
        """Parse and validate response"""
        json_str = self._extract_complete_json(raw_text)
        if not json_str:
            json_str = raw_text.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.startswith("```"):
                json_str = json_str[3:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]
            json_str = json_str.strip()

        # ✅ Clean invalid control characters
        json_str = self._clean_json_string(json_str)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"[Gemini] JSON parse error: {e}")
            logger.debug(f"[Gemini] Problematic JSON (first 500 chars): {json_str[:500]}")
            raise ValueError(f"Invalid JSON: {e}")

        required = ["hook", "script", "cta", "search_queries", "main_visual_focus", "metadata"]
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Missing keys: {missing}")

        if not isinstance(data["script"], list):
            raise ValueError("Script must be a list")

        # ✅ Deduplicate hook/first sentence
        hook_text = (data.get("hook") or "").strip()
        script_list = [s.strip() for s in data["script"] if isinstance(s, str) and s.strip()]
        if script_list and self._is_near_duplicate(hook_text, script_list[0]) and len(script_list) > 40:
            script_list.pop(0)
        data["script"] = script_list

        # ✅ Validate 40-70 range
        sentence_count = len(data["script"])
        if sentence_count < 40:
            raise ValueError(f"Script too short: {sentence_count} (minimum 40)")
        if sentence_count > 70:
            logger.warning(f"[Gemini] Truncating {sentence_count} → 70")
            data["script"] = data["script"][:70]

        # ✅ Auto-generate chapters if missing
        if "chapters" not in data or not data["chapters"]:
            data["chapters"] = self._auto_generate_chapters(data["script"])

        return ContentResponse(
            hook=hook_text,
            script=data["script"],
            cta=data["cta"],
            search_queries=data["search_queries"],
            main_visual_focus=data["main_visual_focus"],
            metadata=data["metadata"],
            chapters=data["chapters"]
        )

    def _clean_json_string(self, json_str: str) -> str:
        """
        Clean JSON string from invalid control characters.
        Preserves valid escape sequences but removes raw control characters.
        """
        # Replace problematic control characters with escaped versions
        replacements = {
            '\n': '\\n',   # Newline
            '\r': '\\r',   # Carriage return
            '\t': '\\t',   # Tab
            '\b': '\\b',   # Backspace
            '\f': '\\f',   # Form feed
        }
        
        result = []
        i = 0
        in_string = False
        escape_next = False
        
        while i < len(json_str):
            char = json_str[i]
            
            # Track if we're inside a string
            if char == '"' and not escape_next:
                in_string = not in_string
                result.append(char)
                i += 1
                continue
            
            # Track escape sequences
            if char == '\\' and not escape_next:
                escape_next = True
                result.append(char)
                i += 1
                continue
            
            # If this was escaped, just add it
            if escape_next:
                result.append(char)
                escape_next = False
                i += 1
                continue
            
            # If we're in a string and hit a control character, replace it
            if in_string and char in replacements:
                result.append(replacements[char])
            # Remove other control characters (ASCII 0-31 except allowed ones)
            elif in_string and ord(char) < 32 and char not in '\n\r\t':
                # Skip invalid control characters
                pass
            else:
                result.append(char)
            
            i += 1
        
        return ''.join(result)

    def _is_near_duplicate(self, a: str, b: str, thresh: float = 0.85) -> bool:
        na = re.sub(r"[^a-z0-9]+", " ", (a or "").lower()).strip()
        nb = re.sub(r"[^a-z0-9]+", " ", (b or "").lower()).strip()
        if not na or not nb:
            return False
        return difflib.SequenceMatcher(None, na, nb).ratio() >= thresh

    def _extract_complete_json(self, text: str) -> Optional[str]:
        """Extract complete JSON with balanced braces"""
        start = text.find('{')
        if start == -1:
            return None

        brace_count = 0
        in_string = False
        escape_next = False

        for i in range(start, len(text)):
            char = text[i]
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return text[start:i+1]
        return None

    def _auto_generate_chapters(self, script: List[str]) -> List[Dict[str, Any]]:
        """Auto-generate chapters"""
        total = len(script)
        if total < 30:
            return [{"title": "Full Content", "start_sentence": 0, "end_sentence": total-1, "description": "Complete video"}]

        num_chapters = min(8, max(6, total // 8))
        chapter_size = total // num_chapters
        chapters = []

        intro_end = max(4, total // 10)
        chapters.append({"title": "Introduction", "start_sentence": 0, "end_sentence": intro_end, "description": "Overview"})

        start = intro_end + 1
        main_chapters = num_chapters - 2

        for i in range(main_chapters):
            end = start + chapter_size - 1
            if i == main_chapters - 1:
                end = int(total * 0.9)
            chapters.append({"title": f"Part {i+1}", "start_sentence": start, "end_sentence": min(end, total-5), "description": f"Section {i+1}"})
            start = min(end, total-5) + 1

        chapters.append({"title": "Conclusion", "start_sentence": chapters[-1]["end_sentence"]+1, "end_sentence": total-1, "description": "Summary"})
        return chapters
