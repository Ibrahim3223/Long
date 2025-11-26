# -*- coding: utf-8 -*-
"""
Orchestrator Adapter
✅ Modern interface with ConfigManager
✅ Backward compatible with old orchestrator
✅ Clean dependency injection
✅ Easy to test
"""
import os
import logging
from typing import Optional, Tuple, Dict
from pathlib import Path

from autoshorts.config.config_manager import ConfigManager
from autoshorts.orchestrator import ShortsOrchestrator

logger = logging.getLogger(__name__)


class OrchestratorAdapter:
    """
    Modern adapter for ShortsOrchestrator.

    Usage (New way - recommended):
        config = ConfigManager.get_instance("my_channel")
        adapter = OrchestratorAdapter(config)
        video_path, metadata = adapter.produce_video()

    Usage (Old way - still works):
        orchestrator = ShortsOrchestrator(...)
        video_path, metadata = orchestrator.produce_video(topic)
    """

    def __init__(
        self,
        config: Optional[ConfigManager] = None,
        temp_dir: Optional[str] = None
    ):
        """
        Initialize orchestrator adapter.

        Args:
            config: Configuration manager (auto-loads if None)
            temp_dir: Temporary directory (auto-creates if None)
        """
        # Get or create config
        self.config = config or ConfigManager.get_instance()

        # Setup temp directory
        if temp_dir:
            self.temp_dir = Path(temp_dir)
        else:
            import tempfile
            channel_name = self.config.channel.name.replace(" ", "_")
            self.temp_dir = Path(tempfile.gettempdir()) / f"autoshorts_{channel_name}"

        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Validate config
        if not self.config.validate():
            raise ValueError("Invalid configuration")

        # Create underlying orchestrator
        self.orchestrator = self._create_orchestrator()

        logger.info(f"✅ OrchestratorAdapter initialized for channel: {self.config.channel.name}")

    def _create_orchestrator(self) -> ShortsOrchestrator:
        """Create underlying ShortsOrchestrator with config."""
        return ShortsOrchestrator(
            channel_id=self.config.channel.name,
            temp_dir=str(self.temp_dir),
            api_key=self.config.get_api_key("gemini"),
            pexels_key=self.config.get_api_key("pexels"),
            pixabay_key=self.config.get_api_key("pixabay"),
            use_novelty=self.config.novelty.enforce,
            ffmpeg_preset=self.config.performance.ffmpeg_preset
        )

    def produce_video(
        self,
        topic: Optional[str] = None,
        max_retries: Optional[int] = None
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """
        Produce video using configuration.

        Args:
            topic: Video topic (uses channel topic if None)
            max_retries: Max retries (uses config default if None)

        Returns:
            (video_path, metadata) tuple or (None, None) if failed
        """
        # Use channel topic if not provided
        if topic is None:
            topic = self.config.channel.topic

        # Use config max retries if not provided
        if max_retries is None:
            max_retries = self.config.content.max_generation_attempts

        # Set MODE environment variable for Gemini
        os.environ["MODE"] = self.config.channel.mode

        logger.info(f"🎬 Producing video: {topic[:100]}...")
        logger.info(f"🎯 Channel: {self.config.channel.name} ({self.config.channel.mode})")

        # Call underlying orchestrator
        video_path, metadata = self.orchestrator.produce_video(
            topic_prompt=topic,
            max_retries=max_retries
        )

        return video_path, metadata

    def get_config(self) -> ConfigManager:
        """Get configuration manager."""
        return self.config

    def get_temp_dir(self) -> Path:
        """Get temporary directory."""
        return self.temp_dir

    def __repr__(self) -> str:
        return f"<OrchestratorAdapter channel='{self.config.channel.name}' mode='{self.config.channel.mode}'>"


def create_orchestrator(
    channel_name: Optional[str] = None,
    temp_dir: Optional[str] = None
) -> OrchestratorAdapter:
    """
    Convenience function to create orchestrator.

    Args:
        channel_name: Channel name (uses env CHANNEL_NAME if None)
        temp_dir: Temporary directory (auto-creates if None)

    Returns:
        OrchestratorAdapter instance
    """
    # Get channel name
    if channel_name is None:
        channel_name = os.getenv("CHANNEL_NAME") or os.getenv("ENV") or "DefaultChannel"

    # Load config for channel
    config = ConfigManager.get_instance(channel_name=channel_name)

    # Create adapter
    return OrchestratorAdapter(config=config, temp_dir=temp_dir)
