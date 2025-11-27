# -*- coding: utf-8 -*-
"""
Metadata Generator - Title, Description, Thumbnail Text
✅ Gemini AI-powered SEO optimization
✅ Multiple title candidates with scoring
✅ SEO-optimized descriptions
✅ Mobile-readable thumbnail text
✅ Viral formula patterns
"""
import re
import logging
import os
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

# Viral title formulas
TITLE_FORMULAS = {
    "number_adjective": "{number} {adjective} {topic}",
    "how_why": "How {entity} {action} Without {expected_thing}",
    "revealed": "The {adjective} Truth About {topic} Revealed",
    "secret": "The Hidden Secret of {entity}",
    "comparison": "{thing1} vs {thing2}: The Truth",
    "before_after": "What {topic} Looked Like {timeframe} Ago",
    "banned": "Why {topic} is Banned in {number} Countries",
    "mistake": "You've Been {doing} Wrong Your Whole Life",
    "unexpected": "{topic} That Will Change How You See {category}",
    "timeframe": "What Happens When You {action} for {duration}",
}


class MetadataGenerator:
    """Generate optimized YouTube metadata."""

    def __init__(self, gemini_api_key: Optional[str] = None):
        """Initialize with optional Gemini API key for AI-powered metadata."""
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.use_gemini = bool(self.gemini_api_key)

        if self.use_gemini:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.gemini_api_key)
                self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")
                logger.info("✅ Gemini AI metadata generation enabled")
            except Exception as e:
                logger.warning(f"⚠️ Gemini init failed: {e}, using template-based fallback")
                self.use_gemini = False
                self.gemini_model = None
        else:
            logger.info("Using template-based metadata generation")
            self.gemini_model = None

    def generate_gemini_metadata(
        self, script: Dict, main_topic: str, lang: str = "en"
    ) -> Optional[Dict[str, any]]:
        """
        Generate SEO-optimized metadata using Gemini AI.

        Args:
            script: Video script with sentences
            main_topic: Main topic/theme
            lang: Language code (en, tr, etc.)

        Returns:
            Dict with title, description, keywords or None if failed
        """
        if not self.use_gemini or not self.gemini_model:
            return None

        try:
            # Extract script content
            sentences = [s.get("text", "") for s in script.get("sentences", [])]
            full_script = "\n".join(sentences[:20])  # First 20 sentences for context

            # Build Gemini prompt for SEO metadata
            prompt = f"""You are a YouTube SEO expert specializing in viral content optimization.

VIDEO SCRIPT PREVIEW (first part):
{full_script}

TOPIC: {main_topic}
LANGUAGE: {lang}

Generate HIGHLY SEO-OPTIMIZED YouTube metadata for this video:

1. TITLE (50-70 characters):
   - Must be attention-grabbing and click-worthy
   - Include power words (shocking, secret, revealed, incredible, etc.)
   - Include numbers if relevant to content
   - Must accurately reflect video content
   - Must be SEO-optimized for search
   - Examples of GOOD titles:
     * "The Shocking Truth About [Topic] Nobody Tells You"
     * "7 Incredible Facts That Will Change How You See [Topic]"
     * "Why [Topic] is More Dangerous Than You Think"
   - Examples of BAD titles:
     * "1 Amazing Facts..." (grammatically incorrect)
     * "Video About [Topic]" (generic, boring)
     * "[Topic] Information" (not engaging)

2. DESCRIPTION (300-500 characters):
   - First 150 chars MUST be a compelling hook (shows in search results)
   - Include key topics covered in the video
   - Add 3-5 relevant questions viewers might search for
   - Natural keyword integration for SEO
   - Include soft CTA (subscribe, like, comment)
   - Must be informative and engaging

3. TAGS/KEYWORDS (5-10 keywords):
   - Mix of broad and specific keywords
   - Include long-tail keywords (3-5 word phrases)
   - Based on actual video content

Return ONLY a valid JSON object with this EXACT structure:
{{
  "title": "Your SEO-optimized title here",
  "description": "Your compelling description here...",
  "keywords": ["keyword1", "keyword2", "keyword3", "long tail keyword phrase"]
}}

CRITICAL RULES:
- Title MUST be grammatically correct
- Title MUST accurately describe video content
- Description first 150 chars MUST hook viewers
- NO placeholders, NO generic content
- Output ONLY valid JSON, nothing else
"""

            # Call Gemini
            response = self.gemini_model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.8,
                    "top_p": 0.95,
                    "max_output_tokens": 1000,
                }
            )

            # Parse response
            response_text = response.text.strip()

            # Extract JSON from response (handle markdown code blocks)
            import json
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            metadata = json.loads(response_text)

            # Validate response
            if not metadata.get("title") or not metadata.get("description"):
                logger.warning("⚠️ Gemini metadata incomplete, using fallback")
                return None

            logger.info(f"✅ Gemini generated metadata:")
            logger.info(f"  Title: {metadata['title']}")
            logger.info(f"  Description: {len(metadata.get('description', ''))} chars")
            logger.info(f"  Keywords: {len(metadata.get('keywords', []))} tags")

            return metadata

        except Exception as e:
            logger.warning(f"⚠️ Gemini metadata generation failed: {e}, using fallback")
            return None

    def generate_title_candidates(
        self, script: Dict, hook: str, main_topic: str
    ) -> List[Dict[str, any]]:
        """
        Generate multiple title candidates and score them.

        Returns:
            List of {title, score, formula} dicts
        """
        candidates = []

        # Extract key elements
        sentences = [s.get("text", "") for s in script.get("sentences", [])]
        full_text = " ".join(sentences)

        # Extract numbers
        numbers = re.findall(r"\b\d+\b", full_text)

        # Extract entities (capitalized words/phrases)
        entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", full_text)

        # Extract adjectives (simple heuristic)
        adjectives = self._extract_adjectives(full_text)

        # Generate candidates from formulas
        if numbers and adjectives:
            candidates.append({
                "title": f"{numbers[0]} {adjectives[0]} Facts About {main_topic}",
                "formula": "number_adjective",
                "score": 0,
            })

        if entities:
            candidates.append({
                "title": f"The Truth About {entities[0]}",
                "formula": "revealed",
                "score": 0,
            })
            candidates.append({
                "title": f"The Hidden Secret of {entities[0]}",
                "formula": "secret",
                "score": 0,
            })

        # Hook-based title
        hook_title = self._hook_to_title(hook)
        if hook_title:
            candidates.append({
                "title": hook_title,
                "formula": "hook_based",
                "score": 0,
            })

        # Topic-based simple title
        candidates.append({
            "title": f"The Fascinating Story of {main_topic}",
            "formula": "simple",
            "score": 0,
        })

        # Score all candidates
        for candidate in candidates:
            candidate["score"] = self._score_title(candidate["title"])

        # Sort by score
        candidates.sort(key=lambda x: x["score"], reverse=True)

        return candidates[:5]  # Top 5

    def _extract_adjectives(self, text: str) -> List[str]:
        """Extract potential adjectives (simple heuristic)."""
        adjective_keywords = [
            "amazing", "shocking", "incredible", "bizarre", "strange",
            "unusual", "remarkable", "extraordinary", "mysterious", "hidden",
            "secret", "ancient", "modern", "powerful", "deadly", "beautiful",
        ]
        found = []
        text_lower = text.lower()
        for adj in adjective_keywords:
            if adj in text_lower:
                found.append(adj.capitalize())
        return found or ["Amazing", "Incredible", "Fascinating"]

    def _hook_to_title(self, hook: str) -> Optional[str]:
        """Convert hook to title (if suitable)."""
        # Clean hook
        hook = hook.strip().rstrip(".!?")

        # Check length
        if len(hook) > 65:
            # Too long, extract key part
            words = hook.split()
            hook = " ".join(words[:10])  # First 10 words

        # Ensure title-case
        hook = hook[0].upper() + hook[1:] if hook else hook

        return hook if 30 <= len(hook) <= 65 else None

    def _score_title(self, title: str) -> float:
        """Score title quality (0-10)."""
        score = 5.0

        # Length check (55-65 optimal)
        length = len(title)
        if 55 <= length <= 65:
            score += 2.0
        elif 50 <= length <= 70:
            score += 1.0
        elif length > 75:
            score -= 2.0

        # Has number
        if re.search(r"\b\d+\b", title):
            score += 1.5

        # Has question word
        if re.search(r"\b(how|why|what|when|where|who)\b", title, re.I):
            score += 1.0

        # Has power words
        power_words = ["secret", "truth", "revealed", "hidden", "shocking", "banned", "never"]
        if any(word in title.lower() for word in power_words):
            score += 1.0

        # Avoid generic
        generic_words = ["video", "content", "channel"]
        if any(word in title.lower() for word in generic_words):
            score -= 1.5

        return min(10.0, max(0.0, score))

    def generate_description(
        self, script: Dict, title: str, chapters: List[Dict]
    ) -> str:
        """
        Generate SEO-optimized description.

        Structure:
        - First 150 chars: Hook + value promise
        - Chapter outline
        - Related questions
        - Soft CTA
        """
        sentences = [s.get("text", "") for s in script.get("sentences", [])]

        # First 150 chars - SEO hook
        hook_sentence = sentences[0] if sentences else title
        seo_hook = f"{hook_sentence} Discover the fascinating details in this video."

        # Truncate to 150 chars
        if len(seo_hook) > 150:
            seo_hook = seo_hook[:147] + "..."

        # Chapter outline
        chapter_section = "\n\nWhat You'll Learn:\n"
        for i, chapter in enumerate(chapters[:7], 1):
            chapter_title = chapter.get("title", f"Part {i}")
            chapter_section += f"{i}. {chapter_title}\n"

        # Related questions
        topic_words = re.findall(r"\b[A-Z][a-z]+\b", title)
        questions_section = "\n\nRelated Questions:\n"
        if topic_words:
            main_topic = topic_words[0] if topic_words else "this topic"
            questions_section += f"- What is {main_topic}?\n"
            questions_section += f"- How does {main_topic} work?\n"
            questions_section += f"- Why is {main_topic} important?\n"

        # CTA
        cta_section = "\n\nExplore more fascinating topics on our channel!"

        # Combine
        description = seo_hook + chapter_section + questions_section + cta_section

        # Limit to 5000 chars (YouTube limit)
        if len(description) > 5000:
            description = description[:4997] + "..."

        return description

    def generate_thumbnail_text(self, title: str, hook: str) -> str:
        """
        Generate mobile-readable thumbnail text (3-5 words max).

        Extracts most impactful words from title/hook.
        """
        # Combine title and hook
        combined = f"{title} {hook}"

        # Extract numbers (high priority)
        numbers = re.findall(r"\b\d+\b", combined)

        # Extract power words
        power_words = [
            "secret", "truth", "revealed", "hidden", "shocking",
            "never", "always", "why", "how", "what"
        ]
        found_power = [word.upper() for word in power_words if word in combined.lower()]

        # Extract capitalized words (entities)
        entities = re.findall(r"\b[A-Z][a-z]+\b", combined)

        # Build thumbnail text
        thumbnail_parts = []

        # Add number if found
        if numbers:
            thumbnail_parts.append(numbers[0])

        # Add power word if found
        if found_power:
            thumbnail_parts.append(found_power[0])

        # Add entity if found
        if entities and len(thumbnail_parts) < 3:
            thumbnail_parts.append(entities[0].upper())

        # If still not enough words, extract from title
        if len(thumbnail_parts) < 3:
            title_words = [w for w in title.split() if len(w) > 4][:3]
            thumbnail_parts.extend([w.upper() for w in title_words])

        # Take first 3-4 words
        thumbnail_text = " ".join(thumbnail_parts[:4])

        # Ensure not too long (max 35 chars for mobile)
        if len(thumbnail_text) > 35:
            thumbnail_text = " ".join(thumbnail_parts[:3])

        return thumbnail_text

    def generate_all_metadata(
        self, script: Dict, main_topic: str, lang: str = "en"
    ) -> Dict[str, any]:
        """
        Generate complete metadata package.

        Args:
            script: Video script
            main_topic: Main topic
            lang: Language code (en, tr, etc.)

        Returns:
            Dict with title, description, thumbnail_text, title_candidates
        """
        hook = script.get("hook", "")
        chapters = script.get("chapters", [])

        # ✅ TRY GEMINI FIRST (SEO-optimized, context-aware)
        gemini_metadata = self.generate_gemini_metadata(script, main_topic, lang)

        if gemini_metadata:
            # Use Gemini-generated metadata
            best_title = gemini_metadata["title"]
            description = gemini_metadata["description"]
            keywords = gemini_metadata.get("keywords", [])

            # Generate thumbnail text from Gemini title
            thumbnail_text = self.generate_thumbnail_text(best_title, hook)

            logger.info(f"📝 Generated metadata (Gemini AI):")
            logger.info(f"  Title: {best_title}")
            logger.info(f"  Thumbnail: {thumbnail_text}")
            logger.info(f"  Description: {len(description)} chars")
            logger.info(f"  Keywords: {', '.join(keywords[:5])}")

            return {
                "title": best_title,
                "description": description,
                "thumbnail_text": thumbnail_text,
                "keywords": keywords,
                "title_candidates": [{"title": best_title, "score": 9.0, "formula": "gemini_ai"}],
                "title_score": 9.0,
                "source": "gemini"
            }

        # ⚠️ FALLBACK: Template-based generation
        logger.info("Using template-based metadata fallback")

        # Generate title candidates
        title_candidates = self.generate_title_candidates(script, hook, main_topic)

        # Select best title
        best_title = title_candidates[0]["title"] if title_candidates else main_topic

        # Generate description
        description = self.generate_description(script, best_title, chapters)

        # Generate thumbnail text
        thumbnail_text = self.generate_thumbnail_text(best_title, hook)

        logger.info(f"📝 Generated metadata (template-based):")
        logger.info(f"  Title: {best_title} (score: {title_candidates[0]['score']:.1f}/10)")
        logger.info(f"  Thumbnail: {thumbnail_text}")
        logger.info(f"  Description: {len(description)} chars")

        return {
            "title": best_title,
            "description": description,
            "thumbnail_text": thumbnail_text,
            "title_candidates": title_candidates,
            "title_score": title_candidates[0]["score"] if title_candidates else 0,
            "source": "template"
        }
