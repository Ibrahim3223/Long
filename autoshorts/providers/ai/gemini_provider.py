# -*- coding: utf-8 -*-
"""
Gemini AI Provider
✅ Wraps existing GeminiClient
✅ Implements BaseAIProvider interface
"""
import logging
from typing import Optional

from autoshorts.providers.base import BaseAIProvider, AIScriptResult
from autoshorts.content.gemini_client import GeminiClient
from autoshorts.config.config_manager import ContentConfig

logger = logging.getLogger(__name__)


class GeminiAIProvider(BaseAIProvider):
    """Gemini AI provider implementation."""

    def __init__(self, api_key: str, config: ContentConfig):
        """
        Initialize Gemini provider.

        Args:
            api_key: Gemini API key
            config: Content configuration
        """
        self.api_key = api_key
        self.config = config
        self.client = GeminiClient(api_key=api_key)
        logger.info("✅ Gemini AI provider initialized")

    def get_priority(self) -> int:
        """Gemini is primary AI provider."""
        return 0

    def is_available(self) -> bool:
        """Check if Gemini is available."""
        return bool(self.api_key) and self.client is not None

    def generate_script(
        self,
        topic: str,
        mode: Optional[str] = None,
        sub_topic: Optional[str] = None,
        min_sentences: int = 40,
        max_sentences: int = 70
    ) -> AIScriptResult:
        """
        Generate video script using Gemini.

        Args:
            topic: Video topic/prompt
            mode: Content mode (educational, storytelling, etc.)
            sub_topic: Sub-topic for variety
            min_sentences: Minimum number of sentences
            max_sentences: Maximum number of sentences

        Returns:
            AIScriptResult with complete script and metadata
        """
        try:
            # Call existing Gemini client
            content_response = self.client.generate(
                topic=topic,
                mode=mode,
                sub_topic=sub_topic
            )

            if not content_response:
                raise ValueError("Gemini returned empty response")

            # Convert to AIScriptResult
            result = AIScriptResult(
                title=content_response.metadata.get("title", "Untitled"),
                description=content_response.metadata.get("description", ""),
                hook=content_response.hook or "",
                script=content_response.script or [],
                cta=content_response.cta or "",
                tags=content_response.metadata.get("tags", []),
                chapters=content_response.chapters or [],
                search_queries=content_response.search_queries or [],
                quality_score=content_response.metadata.get("quality_score", 7.0),
                sub_topic=sub_topic
            )

            logger.info(f"✅ Gemini generated script: {len(result.script)} sentences, quality={result.quality_score:.1f}")
            return result

        except Exception as e:
            logger.error(f"❌ Gemini script generation failed: {e}")
            raise

    def get_name(self) -> str:
        """Get provider name."""
        return "Gemini"
