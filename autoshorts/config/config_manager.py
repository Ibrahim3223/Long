# -*- coding: utf-8 -*-
"""
Centralized Configuration Manager
✅ Single source of truth for all configuration
✅ Validation and type checking
✅ Environment variable support
✅ Channel-specific overrides
✅ Easy testing with mock configs
"""
import os
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class VideoConfig:
    """Video generation configuration."""
    width: int = 1920
    height: int = 1080
    fps: int = 30
    aspect_ratio: str = "16:9"
    crf: int = 18
    min_duration: float = 240.0  # 4 minutes
    max_duration: float = 480.0  # 8 minutes
    target_duration: float = 360.0  # 6 minutes
    scene_min_duration: float = 8.0
    scene_max_duration: float = 15.0


@dataclass
class TTSConfig:
    """Text-to-speech configuration."""
    provider: str = "auto"  # kokoro, edge, google, auto
    kokoro_enabled: bool = True
    kokoro_voice: str = "af_sarah"
    kokoro_precision: str = "fp32"
    edge_voice: str = "en-US-GuyNeural"
    edge_rate: str = "+5%"
    edge_pitch: str = "+0Hz"
    google_lang: str = "en-US"
    timeout: int = 30
    max_retries: int = 2
    cache_enabled: bool = False


@dataclass
class ScriptStyleConfig:
    """Script style and quality configuration."""
    # Hook & Opening
    hook_intensity: str = "high"  # low, medium, high, extreme
    cold_open: bool = True  # Start directly with topic, no meta-talk
    hook_max_words: int = 15  # Max words in opening hook

    # Script Structure
    use_modular_structure: bool = True  # hook → context → mechanism → impact → CTA
    cliffhanger_frequency: int = 25  # Seconds between mini cliffhangers

    # Sentence Style
    max_sentence_length: int = 20  # Max words per sentence (caption-friendly)
    conversational_tone: bool = True  # Use "you" instead of "one", contractions
    use_analogies: bool = True  # Encourage comparisons and metaphors

    # Content Rules
    evergreen_only: bool = True  # No dates, current events, temporal refs
    avoid_academic_tone: bool = True  # No formal/cold language
    show_dont_tell: bool = True  # Concrete examples over abstract concepts

    # CTA Style
    cta_softness: str = "medium"  # soft, medium, strong
    cta_max_words: int = 15


@dataclass
class ContentConfig:
    """Content generation configuration."""
    min_sentences: int = 40
    max_sentences: int = 70
    target_sentences: int = 55
    min_quality_score: float = 6.5
    max_generation_attempts: int = 5
    chapters_enabled: bool = True
    min_chapter_sentences: int = 5

    # Script style (NEW)
    script_style: ScriptStyleConfig = field(default_factory=ScriptStyleConfig)


@dataclass
class VideoProviderConfig:
    """Video provider configuration (Pexels/Pixabay)."""
    pexels_per_page: int = 80
    pexels_max_uses: int = 3
    pexels_allow_reuse: bool = True
    pexels_min_duration: int = 5
    pexels_max_duration: int = 20
    pexels_retry_attempts: int = 3
    pexels_timeout: int = 10
    pexels_random_selection: bool = True
    pixabay_fallback: bool = True


@dataclass
class CaptionConfig:
    """Caption rendering configuration."""
    enabled: bool = True
    karaoke_effects: bool = True
    font: str = "Arial"
    font_size: int = 48
    max_line: int = 40
    max_lines: int = 2
    position: str = "bottom"
    primary_color: str = "&H00FFFFFF"
    outline_color: str = "&H00000000"
    highlight_color: str = "&H0000FFFF"
    margin_v: int = 100
    background_enabled: bool = True
    background_color: str = "&H80000000"


@dataclass
class NoveltyConfig:
    """Novelty/anti-duplicate configuration."""
    enforce: bool = True
    window_days: int = 30
    jaccard_max: float = 0.60
    max_retries: int = 5
    entity_cooldown_days: int = 45
    use_embeddings: bool = False


@dataclass
class UploadConfig:
    """YouTube upload configuration."""
    enabled: bool = True
    visibility: str = "public"
    enable_chapters: bool = False
    min_chapter_duration: int = 30
    as_shorts: bool = False


@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""
    fast_mode: bool = True
    ffmpeg_threads: int = 4
    ffmpeg_preset: str = "veryfast"
    parallel_tts: bool = True
    batch_size: int = 10


@dataclass
class ChannelConfig:
    """Channel-specific configuration."""
    name: str = "DefaultChannel"
    topic: str = "Educational content"
    mode: str = "educational"
    style: str = "Informative and engaging"
    search_terms: List[str] = field(default_factory=list)
    lang: str = "en-US"
    tts_voice: str = "af_sarah"
    visibility: str = "public"


class ConfigManager:
    """
    Centralized configuration manager.

    Usage:
        config = ConfigManager.get_instance()
        api_key = config.get_api_key("gemini")
        video_cfg = config.video
        channel_cfg = config.channel
    """

    _instance: Optional['ConfigManager'] = None

    def __init__(
        self,
        channel_name: Optional[str] = None,
        override_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize configuration manager.

        Args:
            channel_name: Channel name for loading channel-specific config
            override_config: Dictionary to override default configs (for testing)
        """
        self._api_keys: Dict[str, str] = {}
        self._channel_name = channel_name or os.getenv("CHANNEL_NAME", "DefaultChannel")
        self._override_config = override_config or {}

        # Initialize all configs
        self.video = VideoConfig()
        self.tts = TTSConfig()
        self.content = ContentConfig()
        self.video_provider = VideoProviderConfig()
        self.caption = CaptionConfig()
        self.novelty = NoveltyConfig()
        self.upload = UploadConfig()
        self.performance = PerformanceConfig()
        self.channel = ChannelConfig(name=self._channel_name)

        # Load configurations
        self._load_from_environment()
        self._load_from_channel_config()
        self._apply_overrides()

        logger.info(f"✅ ConfigManager initialized for channel: {self._channel_name}")

    @classmethod
    def get_instance(cls, channel_name: Optional[str] = None) -> 'ConfigManager':
        """Get singleton instance."""
        if cls._instance is None or (channel_name and cls._instance._channel_name != channel_name):
            cls._instance = cls(channel_name=channel_name)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset singleton (useful for testing)."""
        cls._instance = None

    def _load_from_environment(self):
        """Load configuration from environment variables."""
        # API Keys
        self._api_keys = {
            "gemini": os.getenv("GEMINI_API_KEY", ""),
            "pexels": os.getenv("PEXELS_API_KEY", ""),
            "pixabay": os.getenv("PIXABAY_API_KEY", ""),
            "yt_client_id": os.getenv("YT_CLIENT_ID", ""),
            "yt_client_secret": os.getenv("YT_CLIENT_SECRET", ""),
            "yt_refresh_token": os.getenv("YT_REFRESH_TOKEN", ""),
        }

        # TTS Config from env
        if os.getenv("TTS_PROVIDER"):
            self.tts.provider = os.getenv("TTS_PROVIDER", "auto")
        if os.getenv("KOKORO_VOICE"):
            self.tts.kokoro_voice = os.getenv("KOKORO_VOICE", "af_sarah")
        if os.getenv("KOKORO_PRECISION"):
            self.tts.kokoro_precision = os.getenv("KOKORO_PRECISION", "fp32")

        # Performance
        if os.getenv("FAST_MODE"):
            self.performance.fast_mode = os.getenv("FAST_MODE", "true").lower() == "true"
        if os.getenv("FFMPEG_THREADS"):
            self.performance.ffmpeg_threads = int(os.getenv("FFMPEG_THREADS", "4"))

        # Upload
        if os.getenv("UPLOAD_TO_YT"):
            self.upload.enabled = os.getenv("UPLOAD_TO_YT", "true").lower() == "true"
        if os.getenv("VISIBILITY"):
            self.upload.visibility = os.getenv("VISIBILITY", "public")

    def _load_from_channel_config(self):
        """Load channel-specific configuration with enhanced features."""
        try:
            # Try new enhanced channel config first
            from autoshorts.config.channel_config import get_channel_config_loader
            loader = get_channel_config_loader()
            channel_data = loader.get_channel_config(self._channel_name, include_enhanced=True)

            # Update basic channel config
            self.channel.name = channel_data.get("name", self._channel_name)
            self.channel.topic = channel_data.get("topic", self.channel.topic)
            self.channel.mode = channel_data.get("mode", self.channel.mode)
            self.channel.style = channel_data.get("style", self.channel.style)
            self.channel.search_terms = channel_data.get("search_terms", [])
            self.channel.lang = channel_data.get("lang", self.channel.lang)
            self.channel.tts_voice = channel_data.get("tts_voice", self.channel.tts_voice)
            self.channel.visibility = channel_data.get("visibility", self.channel.visibility)

            # Update TTS voice from channel
            self.tts.kokoro_voice = self.channel.tts_voice

            # ✅ NEW: Apply enhanced feature configs
            if "enhanced" in channel_data:
                enhanced = channel_data["enhanced"]

                # Script style overrides
                if "script_style" in enhanced:
                    style_cfg = enhanced["script_style"]
                    for key, value in style_cfg.items():
                        if hasattr(self.content.script_style, key):
                            setattr(self.content.script_style, key, value)
                    logger.debug(f"Applied script style overrides: {list(style_cfg.keys())}")

                # Shot variety overrides
                if "shot_variety" in enhanced:
                    # Store for later use
                    self.shot_variety_config = enhanced["shot_variety"]
                    logger.debug(f"Shot variety: {self.shot_variety_config}")

                # Audio overrides
                if "audio" in enhanced:
                    # Store for later use
                    self.audio_config = enhanced["audio"]
                    logger.debug(f"Audio config: {self.audio_config}")

                # Caption overrides
                if "captions" in enhanced:
                    caption_cfg = enhanced["captions"]
                    # Store for later use
                    self.caption_style_config = caption_cfg
                    logger.debug(f"Caption style: {caption_cfg}")

            logger.info(f"✅ Loaded enhanced channel config: {self.channel.name} ({self.channel.mode})")

        except ImportError:
            # Fallback to legacy channel loader
            logger.debug("Enhanced channel config not available, trying legacy...")
            try:
                from autoshorts.config.channel_loader import apply_channel_settings
                channel_settings = apply_channel_settings(self._channel_name)

                self.channel.topic = channel_settings.get("CHANNEL_TOPIC", self.channel.topic)
                self.channel.mode = channel_settings.get("CHANNEL_MODE", self.channel.mode)
                self.channel.style = channel_settings.get("CHANNEL_STYLE", self.channel.style)
                self.channel.search_terms = channel_settings.get("CHANNEL_SEARCH_TERMS", [])
                self.channel.lang = channel_settings.get("CHANNEL_LANG", self.channel.lang)
                self.channel.tts_voice = channel_settings.get("CHANNEL_TTS_VOICE", self.channel.tts_voice)
                self.channel.visibility = channel_settings.get("CHANNEL_VISIBILITY", self.channel.visibility)
                self.tts.kokoro_voice = self.channel.tts_voice

                logger.info(f"✅ Loaded legacy channel config: {self.channel.name}")
            except Exception as e2:
                logger.warning(f"⚠️ Legacy channel loader also failed: {e2}")

        except Exception as e:
            logger.warning(f"⚠️ Failed to load channel config: {e}")
            logger.debug("", exc_info=True)

    def _apply_overrides(self):
        """Apply override configurations (for testing)."""
        if not self._override_config:
            return

        for key, value in self._override_config.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.debug(f"Applied override: {key} = {value}")

    def get_api_key(self, service: str) -> str:
        """
        Get API key for a service.

        Args:
            service: Service name (gemini, pexels, pixabay, etc.)

        Returns:
            API key string or empty string if not found
        """
        return self._api_keys.get(service.lower(), "")

    def validate(self) -> bool:
        """
        Validate configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        errors = []

        # Check required API keys
        if not self.get_api_key("gemini"):
            errors.append("Missing GEMINI_API_KEY")
        if not self.get_api_key("pexels"):
            errors.append("Missing PEXELS_API_KEY")

        # Validate video config
        if self.video.width <= 0 or self.video.height <= 0:
            errors.append("Invalid video dimensions")
        if self.video.min_duration >= self.video.max_duration:
            errors.append("Invalid video duration range")

        # Validate TTS config
        if self.tts.provider not in ["kokoro", "edge", "google", "auto"]:
            errors.append(f"Invalid TTS provider: {self.tts.provider}")

        # Validate content config
        if self.content.min_sentences >= self.content.max_sentences:
            errors.append("Invalid sentence range")

        if errors:
            for error in errors:
                logger.error(f"❌ Config validation error: {error}")
            return False

        logger.info("✅ Configuration validation passed")
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        return {
            "channel": {
                "name": self.channel.name,
                "topic": self.channel.topic,
                "mode": self.channel.mode,
                "lang": self.channel.lang,
            },
            "video": {
                "width": self.video.width,
                "height": self.video.height,
                "fps": self.video.fps,
                "target_duration": self.video.target_duration,
            },
            "tts": {
                "provider": self.tts.provider,
                "kokoro_voice": self.tts.kokoro_voice,
                "edge_voice": self.tts.edge_voice,
            },
            "performance": {
                "fast_mode": self.performance.fast_mode,
                "ffmpeg_threads": self.performance.ffmpeg_threads,
            },
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"<ConfigManager channel={self.channel.name} mode={self.channel.mode}>"


# Convenience function for backward compatibility
def get_config(channel_name: Optional[str] = None) -> ConfigManager:
    """Get global configuration instance."""
    return ConfigManager.get_instance(channel_name=channel_name)
