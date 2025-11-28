# -*- coding: utf-8 -*-
"""
Unified TTS Handler - Multi-Provider Support
✅ Kokoro TTS (ultra-realistic, 8 voices)
✅ Edge TTS (fast, reliable, word timings)
✅ Google TTS (fallback)
✅ Automatic fallback chain
✅ Configuration-based provider selection
"""
import logging
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path

from autoshorts.config import settings

logger = logging.getLogger(__name__)


class UnifiedTTSHandler:
    """
    Unified TTS handler with multi-provider support.
    
    Provider priority (configurable):
    1. Kokoro TTS (if enabled, best quality)
    2. Edge TTS (fast, reliable)
    3. Google TTS (last resort)
    """
    
    def __init__(self):
        self.provider = settings.TTS_PROVIDER.lower()  # 'kokoro', 'edge', 'google', 'auto'
        self.kokoro_voice = getattr(settings, 'KOKORO_VOICE', 'af_sarah')
        self.kokoro_precision = getattr(settings, 'KOKORO_PRECISION', 'fp32')
        
        # Initialize providers lazily
        self._kokoro = None
        self._edge = None
        self._last_word_timings = []
        
        logger.info(f"🎙️ Unified TTS initialized: provider={self.provider}")
    
    @property
    def kokoro(self):
        """Lazy load Kokoro TTS."""
        if self._kokoro is None:
            try:
                from autoshorts.tts.kokoro_handler import KokoroTTS
                self._kokoro = KokoroTTS(
                    voice=self.kokoro_voice,
                    precision=self.kokoro_precision
                )
                logger.info("✅ Kokoro TTS loaded")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load Kokoro TTS: {e}")
                self._kokoro = False  # Mark as failed
        return self._kokoro if self._kokoro is not False else None
    
    @property
    def edge(self):
        """Lazy load Edge TTS."""
        if self._edge is None:
            try:
                from autoshorts.tts.edge_handler import TTSHandler
                self._edge = TTSHandler()
                logger.info("✅ Edge TTS loaded")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load Edge TTS: {e}")
                self._edge = False
        return self._edge if self._edge is not False else None
    
    def generate(self, text: str) -> Dict[str, Any]:
        """
        Generate TTS audio with automatic provider selection.
        
        Returns:
            Dict with 'audio' (bytes), 'duration' (float), 'word_timings' (list)
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        text = text.strip()
        
        # Try providers in order
        providers = self._get_provider_order()
        
        for provider_name in providers:
            try:
                result = self._generate_with_provider(provider_name, text)
                if result:
                    logger.info(f"✅ TTS generated with {provider_name}: {result['duration']:.2f}s")
                    self._last_word_timings = result.get('word_timings', [])
                    return result
            except Exception as e:
                logger.warning(f"⚠️ {provider_name} TTS failed: {str(e)[:100]}")
                continue
        
        # All providers failed
        raise RuntimeError("All TTS providers failed")
    
    def synthesize(self, text: str, wav_out: str) -> Tuple[float, List[Tuple[str, float]]]:
        """
        Synthesize text and save to file.
        
        Returns:
            (duration, word_timings)
        """
        result = self.generate(text)
        
        # Save to file
        with open(wav_out, 'wb') as f:
            f.write(result['audio'])
        
        return result['duration'], result.get('word_timings', [])
    
    def get_word_timings(self) -> List[Tuple[str, float]]:
        """Get word timings from last synthesis."""
        return self._last_word_timings
    
    def _get_provider_order(self) -> List[str]:
        """
        Get provider priority order based on configuration.

        ✅ FIXED: No fallback when specific provider selected (prevents voice changes!)
        User feedback: "bazı sahnelerde seslendirme değişiyor" (voice changes in scenes)
        """
        if self.provider == 'kokoro':
            return ['kokoro']  # ✅ NO fallback - same voice throughout!
        elif self.provider == 'edge':
            return ['edge']  # ✅ NO fallback - same voice throughout!
        elif self.provider == 'google':
            return ['google']  # ✅ NO fallback - same voice throughout!
        else:  # 'auto' - smart selection with fallback
            # Use Kokoro if available (best quality), with Edge/Google fallback
            return ['kokoro', 'edge', 'google']
    
    def _generate_with_provider(self, provider: str, text: str) -> Optional[Dict[str, Any]]:
        """Generate with specific provider."""
        if provider == 'kokoro':
            return self._generate_kokoro(text)
        elif provider == 'edge':
            return self._generate_edge(text)
        elif provider == 'google':
            return self._generate_google(text)
        return None
    
    def _generate_kokoro(self, text: str) -> Optional[Dict[str, Any]]:
        """Generate with Kokoro TTS."""
        if self.kokoro is None:
            return None
        
        try:
            result = self.kokoro.generate(text)
            logger.info(f"🎙️ Kokoro: {result['duration']:.2f}s")
            return result
        except Exception as e:
            logger.warning(f"Kokoro generation failed: {e}")
            return None
    
    def _generate_edge(self, text: str) -> Optional[Dict[str, Any]]:
        """Generate with Edge TTS."""
        if self.edge is None:
            return None
        
        try:
            result = self.edge.generate(text)
            logger.info(f"🎙️ Edge TTS: {result['duration']:.2f}s, {len(result.get('word_timings', []))} words")
            return result
        except Exception as e:
            logger.warning(f"Edge TTS generation failed: {e}")
            return None
    
    def _generate_google(self, text: str) -> Optional[Dict[str, Any]]:
        """Generate with Google TTS (via Edge handler's fallback)."""
        if self.edge is None:
            return None
        
        try:
            # Edge handler already has Google TTS fallback
            result = self.edge.generate(text)
            logger.info(f"🎙️ Google TTS: {result['duration']:.2f}s")
            return result
        except Exception as e:
            logger.warning(f"Google TTS generation failed: {e}")
            return None
    
    @classmethod
    def list_kokoro_voices(cls) -> List[str]:
        """Get list of available Kokoro voices."""
        try:
            from autoshorts.tts.kokoro_handler import KokoroTTS
            return KokoroTTS.list_voices()
        except ImportError:
            return []
    
    @classmethod
    def get_voice_info(cls, voice: str) -> Dict[str, str]:
        """Get voice information."""
        try:
            from autoshorts.tts.kokoro_handler import KokoroTTS
            return KokoroTTS.get_voice_info(voice)
        except ImportError:
            return {"name": "Unknown", "gender": "unknown", "style": "unknown"}


# Backward compatibility
TTSHandler = UnifiedTTSHandler
