# -*- coding: utf-8 -*-
"""
Gemini Client - FIXED VERSION
✅ Başlıklar benzersiz (channel topic KULLANILMAZ)
✅ Chapter başlıkları KISA (max 50 karakter)
✅ SEO-optimized
"""

import json
import logging
import random
import re
import os
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


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

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"[Gemini] JSON parse error: {e}")
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

    def _ultra_seo_metadata(
        self,
        metadata: Dict[str, Any],
        topic: str,
        script: List[str],
        chapters: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """ULTRA SEO enhancement"""
        md = dict(metadata or {})
        
        # ✅ Extract primary keyword
        primary_kw = topic.split("—")[0].strip() if "—" in topic else topic.split(":")[0].strip()
        
        # ✅ TITLE (60-70 chars, keyword-first)
        title = (md.get("title") or f"{primary_kw} | Complete Guide").strip()
        if not title.lower().startswith(primary_kw.lower()[:15]):
            title = f"{primary_kw}: {title}"
        title = title[:70] if len(title) > 70 else title
        if len(title) < 60:
            title = (title + " | Complete Explanation")[:70]
        
        # ✅ DESCRIPTION (400-500 words)
        desc = (md.get("description") or "").strip()
        
        # Add hook if missing
        if len(desc) < 200:
            desc = f"Discover everything about {primary_kw}. {desc}"
        
        # Add chapters outline
        if chapters and "Chapters" not in desc:
            ch_list = "\n\n📚 Chapters:\n"
            for ch in chapters[:8]:
                ch_list += f"• {ch.get('title', 'Chapter')}\n"
            desc += ch_list
        
        # Add related questions
        if "?" not in desc:
            desc += f"\n\n❓ Questions we answer:\n• What is {primary_kw}?\n• How did it evolve?\n• Why does it matter today?"
        
        # Add CTA
        if "subscribe" not in desc.lower():
            desc += "\n\n👉 Subscribe for more deep dives into fascinating topics!"
        
        # ✅ TAGS (25-35, specific)
        tags = md.get("tags") or []
        tags = [t.strip().lower() for t in tags if t and isinstance(t, str)]
        
        # Add primary keyword variations
        kw_parts = primary_kw.lower().split()
        for i in range(len(kw_parts), 0, -1):
            variant = " ".join(kw_parts[:i])
            if variant and variant not in tags and len(variant) > 3:
                tags.append(variant)
        
        # Add script-based tags (extract nouns)
        script_text = " ".join(script).lower()
        common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        potential_tags = []
        for word in script_text.split():
            word = re.sub(r"[^a-z]", "", word)
            if len(word) > 4 and word not in common_words and word not in tags:
                potential_tags.append(word)
        
        # Add unique potential tags
        for tag in list(dict.fromkeys(potential_tags))[:10]:
            if len(tags) >= 35:
                break
            tags.append(tag)
        
        md["title"] = title.strip()
        md["description"] = desc.strip()
        md["tags"] = list(dict.fromkeys(tags))[:35]
        
        return md
