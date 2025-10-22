"""
Gemini Client - LONG-FORM EDUCATIONAL CONTENT (3-10 min)
Generates 20-35 sentence structured content with chapter markers
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
    chapters: List[Dict[str, Any]]  # NEW: Chapter structure


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

# CTA for long-form (educational)
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
        duration: int,  # 180-600 seconds (3-10 minutes)
        additional_context: Optional[str] = None
    ) -> ContentResponse:
        """Generate long-form educational content (20-35 sentences)"""
        
        # Calculate sentences based on duration
        # Target: ~8-10 words per sentence, ~165 words per minute
        words_per_minute = 165
        total_words = int((duration / 60) * words_per_minute)
        words_per_sentence = 9  # Average
        target_sentences = min(35, max(20, total_words // words_per_sentence))
        
        logger.info(f"[Gemini] Target: {duration}s = {target_sentences} sentences")
        
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

STRUCTURE (CRITICAL):
1. HOOK (1-2 sentences): {hook_formula}
2. INTRODUCTION (3-4 sentences): Set context and overview
3. MAIN CONTENT (15-25 sentences): Divided into 3-5 logical sections
   - Each section should flow naturally
   - Build complexity gradually
   - Include examples and explanations
4. CONCLUSION (2-3 sentences): Summary and key takeaway
5. CTA (1 sentence): {cta}

REQUIREMENTS:
- Total sentences: {target_sentences} sentences
- Each sentence: 8-12 words (clear and concise)
- Natural pacing: mix of short and medium sentences
- Visual descriptions: every sentence must have clear visual anchor
- No filler words, no meta-instructions
- Logical flow: each sentence builds on previous
- Educational tone: informative but engaging

CHAPTER STRUCTURE:
Divide content into 5-7 logical chapters (for YouTube timestamps):
- Introduction
- 3-5 main topic chapters
- Conclusion

Return STRICT JSON:
{{
  "hook": "...",
  "script": ["sentence1", "sentence2", ...],
  "cta": "...",
  "search_queries": ["query1", "query2", ...],
  "main_visual_focus": "...",
  "chapters": [
    {{
      "title": "Introduction",
      "start_sentence": 0,
      "end_sentence": 4,
      "description": "Brief chapter description"
    }}
  ],
  "metadata": {{
    "title": "...",
    "description": "...",
    "tags": ["tag1", "tag2", ...]
  }}
}}

Style: {style}
Additional context: {additional_context or "None"}

CRITICAL: Return ONLY the JSON object. Do not wrap in markdown code blocks. Do not add any text before or after the JSON."""
    
    def _call_api(self, prompt: str) -> str:
        """Call Gemini API"""
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.8,
                    max_output_tokens=4000,  # More tokens for long-form
                )
            )
            
            # ✅ FIXED: Safely extract text from response
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'candidates') and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    parts = candidate.content.parts
                    if parts and len(parts) > 0:
                        return parts[0].text
            
            raise ValueError("Could not extract text from Gemini response")
            
        except Exception as e:
            logger.error(f"[Gemini] API error: {e}")
            raise
    
    def _parse_response(self, raw_response: str, topic: str) -> ContentResponse:
        """Parse JSON response with robust error handling"""
        
        # ✅ FIXED: Validate input first
        if not raw_response or not isinstance(raw_response, str):
            raise ValueError(f"Invalid response type: {type(raw_response)}")
        
        logger.debug(f"[Gemini] Raw response length: {len(raw_response)} chars")
        
        # Try multiple extraction methods
        json_str = None
        
        # Method 1: Extract from markdown code block
        markdown_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_response, re.DOTALL)
        if markdown_match:
            json_str = markdown_match.group(1)
            logger.debug("[Gemini] Extracted JSON from markdown block")
        
        # Method 2: Find JSON object directly
        if not json_str:
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                logger.debug("[Gemini] Extracted JSON directly")
        
        # Method 3: Try parsing entire response as JSON
        if not json_str:
            try:
                json.loads(raw_response.strip())
                json_str = raw_response.strip()
                logger.debug("[Gemini] Entire response is valid JSON")
            except json.JSONDecodeError:
                pass
        
        if not json_str:
            logger.error(f"[Gemini] Response preview: {raw_response[:500]}...")
            raise ValueError("No valid JSON found in response")
        
        # Parse JSON
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"[Gemini] JSON parse error: {e}")
            logger.error(f"[Gemini] JSON string: {json_str[:500]}...")
            raise ValueError(f"Invalid JSON: {e}")
        
        # Validate required fields
        required = ["hook", "script", "cta", "search_queries", "main_visual_focus", "metadata"]
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Missing required keys: {missing}")
        
        # Validate script is a list
        if not isinstance(data["script"], list):
            raise ValueError("Script must be a list of sentences")
        
        if len(data["script"]) < 15:
            raise ValueError(f"Script too short: {len(data['script'])} sentences (minimum 15)")
        
        # Chapters (optional, auto-generate if missing)
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
    
    def _auto_generate_chapters(self, script: List[str]) -> List[Dict[str, Any]]:
        """Auto-generate chapter structure if not provided"""
        total_sentences = len(script)
        
        if total_sentences < 15:
            # Too short for chapters
            return [{
                "title": "Full Content",
                "start_sentence": 0,
                "end_sentence": total_sentences - 1,
                "description": "Complete video content"
            }]
        
        # Simple chapter division
        chapter_size = total_sentences // 5  # 5 chapters
        
        chapters = [
            {
                "title": "Introduction",
                "start_sentence": 0,
                "end_sentence": min(4, total_sentences // 4),
                "description": "Overview and context"
            }
        ]
        
        # Main content chapters
        start = chapters[0]["end_sentence"] + 1
        for i in range(3):
            end = min(start + chapter_size, total_sentences - 5)
            chapters.append({
                "title": f"Part {i+1}",
                "start_sentence": start,
                "end_sentence": end,
                "description": f"Main content section {i+1}"
            })
            start = end + 1
        
        # Conclusion
        chapters.append({
            "title": "Conclusion",
            "start_sentence": chapters[-1]["end_sentence"] + 1,
            "end_sentence": total_sentences - 1,
            "description": "Summary and takeaways"
        })
        
        return chapters
