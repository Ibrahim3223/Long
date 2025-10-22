"""
Gemini Client - LONG-FORM EDUCATIONAL CONTENT (4-7 min)
✅ ULTIMATE: Generates 40-70 sentence content with chapters
"""

import json
import logging
import random
import re
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


@dataclass
class ContentResponse:
    """Structured response from content generation"""
    hook: str
    script: List[str]
    cta: str
    search_queries: List[str]
    main_visual_focus: str
    metadata: Dict[str, Any]
    chapters: List[Dict[str, Any]]


# ============================================================================
# EDUCATIONAL HOOK FORMULAS - For long-form content
# ============================================================================
EDUCATIONAL_HOOKS = [
    "Have you ever wondered {topic_question}?",
    "Today we're diving deep into {topic_keyword} - and what we'll discover is fascinating.",
    "Let's explore the complete story of {topic_keyword}.",
    "The truth about {topic_keyword} is more interesting than you think.",
    "Understanding {topic_keyword} changes everything. Here's the full breakdown.",
    "Three minutes from now, you'll understand {topic_keyword} better than most people.",
    "We're about to uncover the complete truth about {topic_keyword}."
]

EDUCATIONAL_CTA = [
    "Want to learn more? Subscribe for deeper dives into fascinating topics.",
    "If you enjoyed this explanation, hit subscribe for more educational content.",
    "That's the complete picture. Subscribe to explore more topics like this.",
    "Thanks for watching this deep dive. More educational content coming soon!",
    "Drop your questions in the comments - let's discuss this further."
]


class GeminiClient:
    """Long-form educational content generator"""
    
    MODELS = {
        "flash": "gemini-2.5-flash",
        "gemini-2.5-flash": "gemini-2.5-flash",
    }
    
    def __init__(self, api_key: str, model: str = "flash"):
        """Initialize Gemini client for long-form"""
        if not api_key:
            raise ValueError("Gemini API key required")
        
        self.client = genai.Client(api_key=api_key)
        self.model = self.MODELS.get(model, model)
        logger.info(f"[Gemini] Long-form client initialized: {self.model}")
    
    def generate(
        self,
        topic: str,
        style: str,
        duration: int,  # 240-480 seconds (4-8 minutes)
        additional_context: Optional[str] = None
    ) -> ContentResponse:
        """
        Generate long-form educational content (40-70 sentences)
        
        ✅ ULTIMATE FIX: Correct sentence calculation for 4-7 minute videos
        """
        
        # ✅ CRITICAL FIX: New formula for 40-70 sentences
        # Average TTS: ~5-6 seconds per sentence (educational pace)
        seconds_per_sentence = 6
        target_sentences = duration // seconds_per_sentence
        
        # Enforce 40-70 range
        target_sentences = min(70, max(40, target_sentences))
        
        logger.info(f"[Gemini] Target: {duration}s = {target_sentences} sentences (40-70 range)")
        
        # Select hooks
        hook_formula = random.choice(EDUCATIONAL_HOOKS)
        cta = random.choice(EDUCATIONAL_CTA)
        
        prompt = self._build_longform_prompt(
            topic, style, target_sentences, hook_formula, cta, additional_context
        )
        
        try:
            raw_response = self._call_api(prompt)
            content = self._parse_response(raw_response, topic)
            
            logger.info(f"[Gemini] ✅ Generated {len(content.script)} sentences with {len(content.chapters)} chapters")
            return content
            
        except Exception as e:
            logger.error(f"[Gemini] ❌ Generation failed: {e}")
            raise
    
    def _build_longform_prompt(
        self,
        topic: str,
        style: str,
        target_sentences: int,
        hook_formula: str,
        cta: str,
        additional_context: Optional[str] = None
    ) -> str:
        """Build comprehensive long-form prompt"""
        
        return f"""Create an educational long-form YouTube video script about: {topic}

CRITICAL REQUIREMENTS:
- MUST generate EXACTLY {target_sentences} sentences (between 40-70 sentences)
- Each sentence: 8-12 words (clear and concise)
- Educational tone: informative but engaging
- Divided into 5-7 logical chapters

STRUCTURE:
1. HOOK (1-2 sentences): {hook_formula}
2. INTRODUCTION (5-8 sentences): Set context and overview
3. MAIN CONTENT (30-55 sentences): Divided into 3-5 logical sections
4. CONCLUSION (3-5 sentences): Summary and key takeaway
5. CTA (1 sentence): {cta}

CHAPTER STRUCTURE:
Divide content into 5-7 logical chapters for YouTube timestamps

Style: {style}
Additional context: {additional_context or "None"}

YOU MUST RETURN A COMPLETE, VALID JSON OBJECT. DO NOT TRUNCATE.

Return this EXACT JSON structure (no markdown, no code blocks):

{{
  "hook": "Your hook sentence",
  "script": ["sentence 1", "sentence 2", ... EXACTLY {target_sentences} sentences],
  "cta": "Call to action",
  "search_queries": ["visual term 1", "term 2", ... 20-30 terms for variety],
  "main_visual_focus": "Primary visual theme",
  "chapters": [
    {{"title": "Introduction", "start_sentence": 0, "end_sentence": 7, "description": "Overview"}},
    {{"title": "Main Point 1", "start_sentence": 8, "end_sentence": 20, "description": "First section"}},
    ... 5-7 chapters total covering all {target_sentences} sentences
  ],
  "metadata": {{
    "title": "Title (60-70 chars)",
    "description": "Description (300-500 words)",
    "tags": ["tag1", "tag2", ... 20-30 tags]
  }}
}}

CRITICAL: 
- Return ONLY JSON (no markdown, no code blocks)
- Include ALL {target_sentences} sentences
- No truncation
- Complete all fields"""
    
    def _call_api(self, prompt: str) -> str:
        """Call Gemini API"""
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.8,
                    max_output_tokens=16000,  # ✅ INCREASED for 40-70 sentences
                )
            )
            
            if hasattr(response, 'text'):
                text = response.text
                logger.debug(f"[Gemini] Response received: {len(text)} chars")
                return text
            elif hasattr(response, 'candidates') and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    parts = candidate.content.parts
                    if parts and len(parts) > 0:
                        text = parts[0].text
                        logger.debug(f"[Gemini] Response received: {len(text)} chars")
                        return text
            
            raise ValueError("Could not extract text from Gemini response")
            
        except Exception as e:
            logger.error(f"[Gemini] API error: {e}")
            raise
    
    def _parse_response(self, raw_response: str, topic: str) -> ContentResponse:
        """Parse JSON response with robust error handling"""
        
        if not raw_response or not isinstance(raw_response, str):
            raise ValueError(f"Invalid response type: {type(raw_response)}")
        
        logger.debug(f"[Gemini] Raw response length: {len(raw_response)} chars")
        
        # Check truncation
        if len(raw_response) > 5000 and not raw_response.rstrip().endswith('}'):
            logger.warning("[Gemini] Response appears truncated")
        
        # Extract JSON
        json_str = None
        
        # Method 1: Markdown block
        markdown_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_response, re.DOTALL)
        if markdown_match:
            json_str = markdown_match.group(1)
            logger.debug("[Gemini] Extracted JSON from markdown")
        
        # Method 2: Complete JSON object
        if not json_str:
            json_str = self._extract_complete_json(raw_response)
            if json_str:
                logger.debug("[Gemini] Extracted complete JSON")
        
        # Method 3: Entire response
        if not json_str:
            try:
                json.loads(raw_response.strip())
                json_str = raw_response.strip()
                logger.debug("[Gemini] Entire response is JSON")
            except json.JSONDecodeError:
                pass
        
        if not json_str:
            logger.error(f"[Gemini] Response preview: {raw_response[:500]}...")
            raise ValueError("No valid JSON found in response")
        
        # Parse
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"[Gemini] JSON parse error: {e}")
            logger.error(f"[Gemini] Preview: {json_str[:500]}...")
            raise ValueError(f"Invalid JSON: {e}")
        
        # Validate
        required = ["hook", "script", "cta", "search_queries", "main_visual_focus", "metadata"]
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Missing keys: {missing}")
        
        if not isinstance(data["script"], list):
            raise ValueError("Script must be a list")
        
        # ✅ UPDATED: Check for 40-70 range
        sentence_count = len(data["script"])
        if sentence_count < 40:
            raise ValueError(f"Script too short: {sentence_count} sentences (minimum 40)")
        
        if sentence_count > 70:
            logger.warning(f"[Gemini] Script too long: {sentence_count} sentences, truncating to 70")
            data["script"] = data["script"][:70]
        
        # Chapters
        if "chapters" not in data or not data["chapters"]:
            logger.info("[Gemini] Auto-generating chapters...")
            data["chapters"] = self._auto_generate_chapters(data["script"])
        
        logger.info(f"[Gemini] Parsed: {len(data['script'])} sentences, {len(data['chapters'])} chapters")
        
        return ContentResponse(
            hook=data["hook"],
            script=data["script"],
            cta=data["cta"],
            search_queries=data["search_queries"],
            main_visual_focus=data["main_visual_focus"],
            metadata=data["metadata"],
            chapters=data["chapters"]
        )
    
    def _extract_complete_json(self, text: str) -> Optional[str]:
        """Extract complete JSON object with balanced braces"""
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
        """Auto-generate chapter structure for 40-70 sentences"""
        total_sentences = len(script)
        
        if total_sentences < 30:
            return [{
                "title": "Full Content",
                "start_sentence": 0,
                "end_sentence": total_sentences - 1,
                "description": "Complete video content"
            }]
        
        # ✅ UPDATED: Better chapter distribution for 40-70 sentences
        # Aim for 6-8 chapters
        num_chapters = min(8, max(6, total_sentences // 8))
        chapter_size = total_sentences // num_chapters
        
        chapters = []
        
        # Introduction (first 10%)
        intro_end = max(4, total_sentences // 10)
        chapters.append({
            "title": "Introduction",
            "start_sentence": 0,
            "end_sentence": intro_end,
            "description": "Overview and context"
        })
        
        # Main content chapters
        start = intro_end + 1
        main_chapters = num_chapters - 2  # Exclude intro and conclusion
        
        for i in range(main_chapters):
            end = start + chapter_size - 1
            if i == main_chapters - 1:
                # Last main chapter goes to 90% of video
                end = int(total_sentences * 0.9)
            
            chapters.append({
                "title": f"Part {i+1}",
                "start_sentence": start,
                "end_sentence": min(end, total_sentences - 5),
                "description": f"Main content section {i+1}"
            })
            start = min(end, total_sentences - 5) + 1
        
        # Conclusion (last 10%)
        chapters.append({
            "title": "Conclusion",
            "start_sentence": chapters[-1]["end_sentence"] + 1,
            "end_sentence": total_sentences - 1,
            "description": "Summary and takeaways"
        })
        
        return chapters
