# -*- coding: utf-8 -*-
"""
Pipeline Executor
✅ Orchestrates pipeline steps
✅ Error handling and retry logic
✅ Progress tracking
"""
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from autoshorts.pipeline.base import BasePipelineStep, PipelineContext

logger = logging.getLogger(__name__)


class PipelineExecutor:
    """
    Executes pipeline steps in sequence.

    Usage:
        executor = PipelineExecutor(steps=[
            ScriptGenerationStep(),
            TTSGenerationStep(),
            VideoCollectionStep(),
            CaptionRenderingStep(),
            AudioMixingStep(),
            ConcatenationStep(),
            ThumbnailGenerationStep(),
        ])
        context = executor.execute(topic="...", channel_id="...")
    """

    def __init__(
        self,
        steps: List[BasePipelineStep],
        max_retries: int = 3,
        retry_delay: int = 3
    ):
        """
        Initialize pipeline executor.

        Args:
            steps: List of pipeline steps to execute
            max_retries: Maximum retries for the entire pipeline
            retry_delay: Delay between retries in seconds
        """
        self.steps = steps
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        logger.info(f"✅ PipelineExecutor initialized with {len(steps)} steps")
        for i, step in enumerate(steps, 1):
            logger.info(f"  {i}. {step.name}")

    def execute(
        self,
        topic: str,
        channel_id: str,
        temp_dir: str,
        mode: Optional[str] = None,
        sub_topic: Optional[str] = None
    ) -> Optional[PipelineContext]:
        """
        Execute the pipeline.

        Args:
            topic: Video topic
            channel_id: Channel ID
            temp_dir: Temporary directory
            mode: Content mode
            sub_topic: Sub-topic for variety

        Returns:
            Final pipeline context or None if failed
        """
        logger.info("=" * 70)
        logger.info("🎬 START PIPELINE EXECUTION")
        logger.info("=" * 70)
        logger.info(f"📝 Topic: {topic[:120]}")
        logger.info(f"🎯 Mode: {mode}")
        logger.info(f"📍 Sub-topic: {sub_topic}")

        for attempt in range(1, self.max_retries + 1):
            try:
                if attempt > 1:
                    import time
                    delay = min(self.retry_delay, 1 * attempt)
                    logger.info(f"⏳ Retry {attempt}/{self.max_retries} in {delay}s")
                    time.sleep(delay)

                # Create context
                context = PipelineContext(
                    topic=topic,
                    channel_id=channel_id,
                    temp_dir=Path(temp_dir),
                    mode=mode,
                    sub_topic=sub_topic
                )

                # Execute all steps
                for i, step in enumerate(self.steps, 1):
                    logger.info(f"\n{'='*70}")
                    logger.info(f"📍 Step {i}/{len(self.steps)}: {step.name}")
                    logger.info(f"{'='*70}")

                    context = step.execute(context)

                    if context is None:
                        logger.error(f"❌ Step {step.name} returned None")
                        break

                # Check if pipeline succeeded
                if context and context.final_video_path:
                    logger.info("=" * 70)
                    logger.info("✅ PIPELINE EXECUTION SUCCESSFUL")
                    logger.info("=" * 70)
                    self._log_stats(context)
                    return context

                logger.warning("Pipeline execution incomplete")

            except KeyboardInterrupt:
                raise
            except Exception as exc:
                logger.error(f"Attempt {attempt}/{self.max_retries} failed: {exc}")
                logger.debug("", exc_info=True)

        logger.error("All pipeline attempts failed")
        return None

    def add_step(self, step: BasePipelineStep, position: Optional[int] = None):
        """
        Add a step to the pipeline.

        Args:
            step: Pipeline step to add
            position: Position to insert (None = append at end)
        """
        if position is None:
            self.steps.append(step)
        else:
            self.steps.insert(position, step)

        logger.info(f"✅ Added step: {step.name} at position {position or len(self.steps)}")

    def remove_step(self, name: str) -> bool:
        """
        Remove a step by name.

        Args:
            name: Step name to remove

        Returns:
            True if removed, False if not found
        """
        for i, step in enumerate(self.steps):
            if step.name == name:
                self.steps.pop(i)
                logger.info(f"✅ Removed step: {name}")
                return True

        logger.warning(f"⚠️ Step not found: {name}")
        return False

    def get_step(self, name: str) -> Optional[BasePipelineStep]:
        """
        Get a step by name.

        Args:
            name: Step name

        Returns:
            Step instance or None
        """
        for step in self.steps:
            if step.name == name:
                return step
        return None

    def _log_stats(self, context: PipelineContext):
        """Log pipeline statistics."""
        if not context.stats:
            return

        logger.info("\n📊 Pipeline Statistics:")
        for key, value in context.stats.items():
            logger.info(f"  - {key}: {value}")
