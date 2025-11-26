# -*- coding: utf-8 -*-
"""
Provider Factory
✅ Centralized provider creation
✅ Automatic fallback chain
✅ Easy to add new providers
✅ Configuration-based provider selection
"""
import logging
from typing import List, Optional, Type

from autoshorts.providers.base import (
    BaseTTSProvider,
    BaseVideoProvider,
    BaseAIProvider,
)
from autoshorts.config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class ProviderFactory:
    """
    Factory for creating and managing providers.

    Usage:
        factory = ProviderFactory(config)
        tts_chain = factory.get_tts_chain()
        video_chain = factory.get_video_chain()
        ai_provider = factory.get_ai_provider()
    """

    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize factory.

        Args:
            config: Configuration manager instance
        """
        self.config = config or ConfigManager.get_instance()
        self._tts_providers: List[BaseTTSProvider] = []
        self._video_providers: List[BaseVideoProvider] = []
        self._ai_provider: Optional[BaseAIProvider] = None

        logger.info("✅ ProviderFactory initialized")

    def get_tts_chain(self) -> List[BaseTTSProvider]:
        """
        Get TTS provider chain with automatic fallback.

        Returns:
            List of TTS providers sorted by priority
        """
        if self._tts_providers:
            return self._tts_providers

        providers: List[BaseTTSProvider] = []

        # Try to load providers based on config
        provider_order = self._get_tts_provider_order()

        for provider_name in provider_order:
            try:
                provider = self._create_tts_provider(provider_name)
                if provider and provider.is_available():
                    providers.append(provider)
                    logger.info(f"✅ TTS provider loaded: {provider.get_name()} (priority={provider.get_priority()})")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load TTS provider {provider_name}: {e}")

        if not providers:
            raise RuntimeError("No TTS providers available")

        # Sort by priority
        providers.sort(key=lambda p: p.get_priority())
        self._tts_providers = providers

        logger.info(f"📊 TTS provider chain: {[p.get_name() for p in providers]}")
        return providers

    def get_video_chain(self) -> List[BaseVideoProvider]:
        """
        Get video provider chain with automatic fallback.

        Returns:
            List of video providers sorted by priority
        """
        if self._video_providers:
            return self._video_providers

        providers: List[BaseVideoProvider] = []

        # Try Pexels first, then Pixabay
        for provider_name in ["pexels", "pixabay"]:
            try:
                provider = self._create_video_provider(provider_name)
                if provider and provider.is_available():
                    providers.append(provider)
                    logger.info(f"✅ Video provider loaded: {provider.get_name()} (priority={provider.get_priority()})")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load video provider {provider_name}: {e}")

        if not providers:
            raise RuntimeError("No video providers available")

        # Sort by priority
        providers.sort(key=lambda p: p.get_priority())
        self._video_providers = providers

        logger.info(f"📊 Video provider chain: {[p.get_name() for p in providers]}")
        return providers

    def get_ai_provider(self) -> BaseAIProvider:
        """
        Get AI provider (currently only Gemini).

        Returns:
            AI provider instance
        """
        if self._ai_provider:
            return self._ai_provider

        # Currently only Gemini is supported
        try:
            provider = self._create_ai_provider("gemini")
            if provider and provider.is_available():
                self._ai_provider = provider
                logger.info(f"✅ AI provider loaded: {provider.get_name()}")
                return provider
        except Exception as e:
            logger.error(f"❌ Failed to load AI provider: {e}")
            raise

        raise RuntimeError("No AI providers available")

    def _get_tts_provider_order(self) -> List[str]:
        """Get TTS provider order based on configuration."""
        provider_pref = self.config.tts.provider.lower()

        if provider_pref == "kokoro":
            return ["kokoro", "edge", "google"]
        elif provider_pref == "edge":
            return ["edge", "kokoro", "google"]
        elif provider_pref == "google":
            return ["google", "edge", "kokoro"]
        else:  # "auto"
            return ["kokoro", "edge", "google"]

    def _create_tts_provider(self, provider_name: str) -> Optional[BaseTTSProvider]:
        """Create TTS provider instance."""
        if provider_name == "kokoro":
            return self._create_kokoro_provider()
        elif provider_name == "edge":
            return self._create_edge_provider()
        elif provider_name == "google":
            return self._create_google_provider()
        return None

    def _create_kokoro_provider(self) -> Optional[BaseTTSProvider]:
        """Create Kokoro TTS provider."""
        if not self.config.tts.kokoro_enabled:
            return None

        try:
            from autoshorts.providers.tts.kokoro_provider import KokoroTTSProvider
            return KokoroTTSProvider(
                voice=self.config.tts.kokoro_voice,
                precision=self.config.tts.kokoro_precision
            )
        except ImportError:
            logger.warning("⚠️ Kokoro TTS not available (import failed)")
            return None

    def _create_edge_provider(self) -> Optional[BaseTTSProvider]:
        """Create Edge TTS provider."""
        try:
            from autoshorts.providers.tts.edge_provider import EdgeTTSProvider
            return EdgeTTSProvider(
                voice=self.config.tts.edge_voice,
                rate=self.config.tts.edge_rate,
                pitch=self.config.tts.edge_pitch
            )
        except ImportError:
            logger.warning("⚠️ Edge TTS not available (import failed)")
            return None

    def _create_google_provider(self) -> Optional[BaseTTSProvider]:
        """Create Google TTS provider."""
        try:
            from autoshorts.providers.tts.google_provider import GoogleTTSProvider
            return GoogleTTSProvider(lang=self.config.tts.google_lang)
        except ImportError:
            logger.warning("⚠️ Google TTS not available (import failed)")
            return None

    def _create_video_provider(self, provider_name: str) -> Optional[BaseVideoProvider]:
        """Create video provider instance."""
        if provider_name == "pexels":
            return self._create_pexels_provider()
        elif provider_name == "pixabay":
            return self._create_pixabay_provider()
        return None

    def _create_pexels_provider(self) -> Optional[BaseVideoProvider]:
        """Create Pexels provider."""
        api_key = self.config.get_api_key("pexels")
        if not api_key:
            logger.warning("⚠️ Pexels API key not configured")
            return None

        try:
            from autoshorts.providers.video.pexels_provider import PexelsVideoProvider
            return PexelsVideoProvider(
                api_key=api_key,
                config=self.config.video_provider
            )
        except ImportError as e:
            logger.warning(f"⚠️ Pexels provider not available: {e}")
            return None

    def _create_pixabay_provider(self) -> Optional[BaseVideoProvider]:
        """Create Pixabay provider."""
        api_key = self.config.get_api_key("pixabay")
        if not api_key:
            logger.debug("Pixabay API key not configured")
            return None

        try:
            from autoshorts.providers.video.pixabay_provider import PixabayVideoProvider
            return PixabayVideoProvider(
                api_key=api_key,
                config=self.config.video_provider
            )
        except ImportError:
            logger.warning("⚠️ Pixabay provider not available (import failed)")
            return None

    def _create_ai_provider(self, provider_name: str) -> Optional[BaseAIProvider]:
        """Create AI provider instance."""
        if provider_name == "gemini":
            return self._create_gemini_provider()
        return None

    def _create_gemini_provider(self) -> Optional[BaseAIProvider]:
        """Create Gemini AI provider."""
        api_key = self.config.get_api_key("gemini")
        if not api_key:
            raise ValueError("Gemini API key not configured")

        try:
            from autoshorts.providers.ai.gemini_provider import GeminiAIProvider
            return GeminiAIProvider(
                api_key=api_key,
                config=self.config.content
            )
        except ImportError as e:
            logger.error(f"❌ Gemini provider not available: {e}")
            raise

    def reset(self):
        """Reset all providers (useful for testing)."""
        self._tts_providers = []
        self._video_providers = []
        self._ai_provider = None
        logger.debug("🔄 Provider factory reset")
