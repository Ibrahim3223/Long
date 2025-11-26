# -*- coding: utf-8 -*-
"""
Pipeline base classes
✅ Abstract base class for pipeline steps
✅ Context for passing data between steps
✅ Error handling and logging
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """
    Context object passed through pipeline steps.

    Contains all data needed for video generation.
    """
    # Input
    topic: str
    channel_id: str
    temp_dir: Path
    mode: Optional[str] = None
    sub_topic: Optional[str] = None

    # Script generation output
    script: Optional[Dict[str, Any]] = None
    title: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    chapters: List[Dict[str, Any]] = field(default_factory=list)

    # TTS generation output
    tts_results: List[Optional[tuple]] = field(default_factory=list)
    audio_durations: List[float] = field(default_factory=list)

    # Video collection output
    scene_paths: List[str] = field(default_factory=list)

    # Final output
    final_video_path: Optional[str] = None
    thumbnail_path: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Statistics
    stats: Dict[str, Any] = field(default_factory=dict)

    def get_sentences(self) -> List[Dict[str, Any]]:
        """Get sentences from script."""
        if not self.script:
            return []
        return self.script.get("sentences", [])

    def add_stat(self, key: str, value: Any):
        """Add statistics."""
        self.stats[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "description": self.description,
            "tags": self.tags,
            "script": self.script,
            "thumbnail_path": self.thumbnail_path,
            "stats": self.stats,
        }


class BasePipelineStep(ABC):
    """
    Abstract base class for pipeline steps.

    Each step should:
    1. Validate input from context
    2. Perform its specific task
    3. Update context with results
    4. Handle errors gracefully
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize pipeline step.

        Args:
            name: Step name (defaults to class name)
        """
        self.name = name or self.__class__.__name__
        logger.info(f"✅ Pipeline step initialized: {self.name}")

    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute pipeline step.

        Args:
            context: Pipeline context

        Returns:
            Updated context

        Raises:
            Exception if step fails
        """
        logger.info(f"▶️ Executing step: {self.name}")

        try:
            # Validate input
            self.validate(context)

            # Run step
            updated_context = self.run(context)

            # Validate output
            self.validate_output(updated_context)

            logger.info(f"✅ Step completed: {self.name}")
            return updated_context

        except Exception as e:
            logger.error(f"❌ Step failed: {self.name} - {e}")
            raise

    @abstractmethod
    def run(self, context: PipelineContext) -> PipelineContext:
        """
        Run the step logic.

        Args:
            context: Pipeline context

        Returns:
            Updated context
        """
        pass

    def validate(self, context: PipelineContext):
        """
        Validate input before running step.

        Args:
            context: Pipeline context

        Raises:
            ValueError if validation fails
        """
        # Override in subclass if needed
        pass

    def validate_output(self, context: PipelineContext):
        """
        Validate output after running step.

        Args:
            context: Pipeline context

        Raises:
            ValueError if validation fails
        """
        # Override in subclass if needed
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name='{self.name}'>"
