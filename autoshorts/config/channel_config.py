# -*- coding: utf-8 -*-
"""
Channel-Specific Configuration Loader
✅ Load channel settings from channels.yml
✅ Apply channel-specific overrides
✅ Support for enhanced features (script style, shot variety, audio)
✅ Fallback to defaults
"""
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


# Default configuration for all new features
DEFAULT_ENHANCED_CONFIG = {
    "script_style": {
        "hook_intensity": "high",
        "cold_open": True,
        "hook_max_words": 15,
        "use_modular_structure": True,
        "cliffhanger_frequency": 25,
        "max_sentence_length": 20,
        "conversational_tone": True,
        "use_analogies": True,
        "evergreen_only": True,
        "avoid_academic_tone": True,
        "show_dont_tell": True,
        "cta_softness": "medium",
        "cta_max_words": 15,
    },
    "shot_variety": {
        "enabled": True,
        "variety_strength": "medium",  # low, medium, high
    },
    "audio": {
        "adaptive_mixing": True,
        "bgm_enabled": True,
    },
    "captions": {
        "style": "modern",  # modern, classic, minimal
        "hook_emphasis": True,
    },
}


# Mode-specific overrides (applied after channel config)
MODE_OVERRIDES = {
    "country_facts": {
        "script_style": {
            "hook_intensity": "extreme",  # Geography needs strong hooks
            "use_analogies": True,
        },
        "shot_variety": {
            "variety_strength": "high",  # Lots of visual variety for locations
        },
    },
    "history_story": {
        "script_style": {
            "hook_intensity": "high",
            "use_modular_structure": True,  # Story structure important
            "conversational_tone": True,
        },
        "shot_variety": {
            "variety_strength": "medium",
        },
    },
    "science": {
        "script_style": {
            "hook_intensity": "high",
            "avoid_academic_tone": True,  # Make science accessible
            "use_analogies": True,
        },
        "audio": {
            "adaptive_mixing": True,  # Important for clarity
        },
    },
    "kids_educational": {
        "script_style": {
            "hook_intensity": "extreme",  # Kids need VERY engaging hooks
            "max_sentence_length": 15,  # Shorter for kids
            "conversational_tone": True,
        },
        "shot_variety": {
            "variety_strength": "high",  # Lots of variety for engagement
        },
        "captions": {
            "style": "colorful",
            "hook_emphasis": True,
        },
    },
    "documentary": {
        "script_style": {
            "hook_intensity": "medium",
            "avoid_academic_tone": False,  # Allow more formal tone
            "show_dont_tell": True,
        },
        "shot_variety": {
            "variety_strength": "medium",
        },
    },
    "entertainment": {
        "script_style": {
            "hook_intensity": "extreme",
            "conversational_tone": True,
            "cta_softness": "soft",
        },
        "audio": {
            "bgm_enabled": True,  # Entertainment benefits from BGM
        },
    },
}


class ChannelConfigLoader:
    """Load and manage channel-specific configurations."""

    def __init__(self, channels_file: str = "channels.yml"):
        """
        Initialize channel config loader.

        Args:
            channels_file: Path to channels.yml file
        """
        self.channels_file = channels_file
        self.channels: Dict[str, Dict[str, Any]] = {}
        self._load_channels()

    def _load_channels(self):
        """Load channels from YAML file."""
        try:
            # Try to find channels.yml
            search_paths = [
                self.channels_file,
                "channels.yml",
                "../channels.yml",
                "../../channels.yml",
            ]

            channels_path = None
            for path in search_paths:
                if Path(path).exists():
                    channels_path = path
                    break

            if not channels_path:
                logger.warning("channels.yml not found, using defaults")
                return

            with open(channels_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            if not data or 'channels' not in data:
                logger.warning("No channels found in channels.yml")
                return

            # Index channels by env (identifier)
            for channel in data['channels']:
                env = channel.get('env', '').lower().strip()
                if env:
                    self.channels[env] = channel
                    logger.debug(f"Loaded channel: {env}")

            logger.info(f"Loaded {len(self.channels)} channels from {channels_path}")

        except Exception as e:
            logger.error(f"Failed to load channels.yml: {e}")
            logger.debug("", exc_info=True)

    def get_channel_config(
        self,
        channel_id: str,
        include_enhanced: bool = True
    ) -> Dict[str, Any]:
        """
        Get configuration for a specific channel.

        Args:
            channel_id: Channel identifier (env field)
            include_enhanced: Include enhanced feature configs

        Returns:
            Channel configuration dictionary
        """
        channel_id_lower = channel_id.lower().strip()

        # Get base channel config
        channel_config = self.channels.get(channel_id_lower, {}).copy()

        if not channel_config:
            logger.warning(f"Channel {channel_id} not found, using defaults")
            channel_config = {
                "name": channel_id,
                "mode": "educational",
                "tts_voice": "af_sarah",
                "lang": "en-US",
            }

        # Add enhanced features if requested
        if include_enhanced:
            mode = channel_config.get('mode', 'educational')

            # Start with default config
            enhanced = DEFAULT_ENHANCED_CONFIG.copy()

            # Apply mode-specific overrides
            if mode in MODE_OVERRIDES:
                enhanced = self._deep_merge(enhanced, MODE_OVERRIDES[mode])

            # Apply channel-specific overrides (if defined in channels.yml)
            if 'enhanced' in channel_config:
                enhanced = self._deep_merge(enhanced, channel_config['enhanced'])

            channel_config['enhanced'] = enhanced

        return channel_config

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def get_all_channels(self) -> Dict[str, Dict[str, Any]]:
        """Get all channel configurations."""
        return {
            channel_id: self.get_channel_config(channel_id)
            for channel_id in self.channels.keys()
        }

    def validate_channel_config(self, channel_id: str) -> tuple:
        """
        Validate channel configuration.

        Args:
            channel_id: Channel identifier

        Returns:
            (is_valid: bool, issues: list)
        """
        issues = []
        config = self.get_channel_config(channel_id, include_enhanced=False)

        # Check required fields
        required_fields = ['name', 'mode', 'tts_voice', 'lang']
        for field in required_fields:
            if field not in config:
                issues.append(f"Missing required field: {field}")

        # Check mode is valid
        valid_modes = [
            'educational', 'documentary', 'entertainment', 'storytelling',
            'sports', 'country_facts', 'history_story', 'science', 'kids_educational'
        ]
        if config.get('mode') not in valid_modes:
            issues.append(f"Invalid mode: {config.get('mode')}")

        return len(issues) == 0, issues

    def list_channels(self) -> list:
        """List all channel identifiers."""
        return list(self.channels.keys())


# Global instance
_channel_config_loader = None


def get_channel_config_loader() -> ChannelConfigLoader:
    """Get global channel config loader instance."""
    global _channel_config_loader
    if _channel_config_loader is None:
        _channel_config_loader = ChannelConfigLoader()
    return _channel_config_loader
