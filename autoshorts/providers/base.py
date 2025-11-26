# -*- coding: utf-8 -*-
"""
Base provider abstractions
✅ Abstract base classes for all providers
✅ Consistent interface across providers
✅ Easy to add new providers
✅ Easy to mock for testing
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TTSResult:
    """Result from TTS generation."""
    audio_data: bytes
    duration: float
    word_timings: List[Tuple[str, float]]
    provider: str


@dataclass
class VideoResult:
    """Result from video search."""
    video_id: str
    url: str
    duration: float
    width: int
    height: int
    provider: str


@dataclass
class AIScriptResult:
    """Result from AI script generation."""
    title: str
    description: str
    hook: str
    script: List[str]
    cta: str
    tags: List[str]
    chapters: List[Dict[str, Any]]
    search_queries: List[str]
    quality_score: float
    sub_topic: Optional[str] = None


class BaseTTSProvider(ABC):
    """
    Abstract base class for TTS providers.

    All TTS providers must implement these methods.
    """

    @abstractmethod
    def get_priority(self) -> int:
        """
        Get provider priority (lower = higher priority).

        Returns:
            Priority number (0 = highest, 100 = lowest)
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if provider is available and configured.

        Returns:
            True if provider can be used, False otherwise
        """
        pass

    @abstractmethod
    def generate(self, text: str) -> TTSResult:
        """
        Generate TTS audio from text.

        Args:
            text: Text to convert to speech

        Returns:
            TTSResult with audio data and metadata

        Raises:
            Exception if generation fails
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get provider name."""
        pass

    def __repr__(self) -> str:
        return f"<{self.get_name()}TTSProvider priority={self.get_priority()}>"


class BaseVideoProvider(ABC):
    """
    Abstract base class for video providers (Pexels, Pixabay, etc.).

    All video providers must implement these methods.
    """

    @abstractmethod
    def get_priority(self) -> int:
        """
        Get provider priority (lower = higher priority).

        Returns:
            Priority number (0 = highest, 100 = lowest)
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if provider is available and configured.

        Returns:
            True if provider can be used, False otherwise
        """
        pass

    @abstractmethod
    def search_videos(
        self,
        query: str,
        min_duration: float = 5.0,
        max_duration: float = 20.0,
        per_page: int = 20
    ) -> List[VideoResult]:
        """
        Search for videos matching query.

        Args:
            query: Search query
            min_duration: Minimum video duration in seconds
            max_duration: Maximum video duration in seconds
            per_page: Number of results per page

        Returns:
            List of VideoResult objects

        Raises:
            Exception if search fails
        """
        pass

    @abstractmethod
    def download_video(self, video: VideoResult, output_path: str) -> bool:
        """
        Download video to local path.

        Args:
            video: VideoResult to download
            output_path: Path to save video

        Returns:
            True if download successful, False otherwise
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get provider name."""
        pass

    def __repr__(self) -> str:
        return f"<{self.get_name()}VideoProvider priority={self.get_priority()}>"


class BaseAIProvider(ABC):
    """
    Abstract base class for AI providers (Gemini, OpenAI, etc.).

    All AI providers must implement these methods.
    """

    @abstractmethod
    def get_priority(self) -> int:
        """Get provider priority (lower = higher priority)."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available and configured."""
        pass

    @abstractmethod
    def generate_script(
        self,
        topic: str,
        mode: Optional[str] = None,
        sub_topic: Optional[str] = None,
        min_sentences: int = 40,
        max_sentences: int = 70
    ) -> AIScriptResult:
        """
        Generate video script.

        Args:
            topic: Video topic/prompt
            mode: Content mode (educational, storytelling, etc.)
            sub_topic: Sub-topic for variety
            min_sentences: Minimum number of sentences
            max_sentences: Maximum number of sentences

        Returns:
            AIScriptResult with complete script and metadata

        Raises:
            Exception if generation fails
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get provider name."""
        pass

    def __repr__(self) -> str:
        return f"<{self.get_name()}AIProvider priority={self.get_priority()}>"


# Convenience type aliases
TTSProviderType = BaseTTSProvider
VideoProviderType = BaseVideoProvider
AIProviderType = BaseAIProvider
