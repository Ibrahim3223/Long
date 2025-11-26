# -*- coding: utf-8 -*-
"""
Script Generation Step
✅ Generates AI-powered scripts
✅ Novelty checking
✅ Quality scoring
"""
import logging
from typing import Optional

from autoshorts.pipeline.base import BasePipelineStep, PipelineContext
from autoshorts.content.gemini_client import GeminiClient
from autoshorts.state.novelty_guard import NoveltyGuard
from autoshorts.content.quality_scorer import QualityScorer

logger = logging.getLogger(__name__)


class ScriptGenerationStep(BasePipelineStep):
    """Generate video script using AI."""

    def __init__(
        self,
        gemini_client: GeminiClient,
        novelty_guard: Optional[NoveltyGuard] = None,
        quality_scorer: Optional[QualityScorer] = None,
        use_novelty: bool = True
    ):
        """
        Initialize script generation step.

        Args:
            gemini_client: Gemini AI client
            novelty_guard: Novelty guard for anti-duplicate
            quality_scorer: Quality scorer for script evaluation
            use_novelty: Whether to use novelty checking
        """
        super().__init__(name="ScriptGeneration")
        self.gemini = gemini_client
        self.novelty_guard = novelty_guard if use_novelty else None
        self.quality_scorer = quality_scorer
        self.use_novelty = use_novelty

    def run(self, context: PipelineContext) -> PipelineContext:
        """Generate script."""
        logger.info("🤖 Generating script via Gemini...")

        # Get or select sub-topic
        sub_topic = context.sub_topic
        if not sub_topic and self.novelty_guard and context.mode:
            try:
                sub_topic = self.novelty_guard.pick_sub_topic(
                    channel=context.channel_id,
                    mode=context.mode
                )
                context.sub_topic = sub_topic
                logger.info(f"🎯 Selected sub-topic: {sub_topic}")
            except Exception as e:
                logger.warning(f"⚠️ Sub-topic selection failed: {e}")

        # Generate script
        content_response = self.gemini.generate(
            topic=context.topic,
            mode=context.mode,
            sub_topic=sub_topic
        )

        if not content_response:
            raise ValueError("Script generation failed")

        # Build script dictionary
        script = {
            "hook": content_response.hook,
            "script": content_response.script,
            "cta": content_response.cta,
            "search_queries": content_response.search_queries,
            "main_visual_focus": content_response.main_visual_focus,
            "title": content_response.metadata.get("title", "Untitled"),
            "description": content_response.metadata.get("description", ""),
            "tags": content_response.metadata.get("tags", []),
            "chapters": content_response.chapters,
            "sentences": [],
            "_sub_topic": sub_topic
        }

        # Build sentences list
        sentences = []
        if script["hook"]:
            sentences.append({"text": script["hook"], "type": "hook"})

        for idx, sentence in enumerate(script["script"]):
            # Skip if duplicate of hook
            if idx == 0 and sentences and sentence.strip().lower() == script["hook"].strip().lower():
                continue
            if sentence.strip():
                sentences.append({"text": sentence, "type": "content"})

        # Add CTA
        if script["cta"]:
            if not sentences or sentences[-1]["text"].strip().lower() != script["cta"].strip().lower():
                sentences.append({"text": script["cta"], "type": "cta"})
            else:
                sentences[-1]["type"] = "cta"

        script["sentences"] = sentences

        if not sentences:
            raise ValueError("Script has no sentences")

        logger.info(f"✅ Script generated: {len(sentences)} sentences")
        logger.info(f"📝 Title: {script['title']}")

        # Quality scoring
        if self.quality_scorer:
            try:
                sentence_texts = [s["text"] for s in sentences if s.get("text")]
                quality_scores = self.quality_scorer.score(
                    sentences=sentence_texts,
                    title=script["title"]
                )
                script["quality_scores"] = quality_scores
                logger.info(f"📊 Quality: {quality_scores.get('overall', 0):.1f}/10")
            except Exception as e:
                logger.warning(f"Quality scoring failed: {e}")

        # Novelty check
        if self.novelty_guard:
            sentences_txt = [s.get("text", "") for s in sentences]
            is_fresh, similarity = self._check_novelty(
                title=script["title"],
                script=sentences_txt,
                channel=context.channel_id
            )
            if not is_fresh:
                raise ValueError(f"Script too similar to recent ones (similarity={similarity:.2f})")

        # Update context
        context.script = script
        context.title = script["title"]
        context.description = script["description"]
        context.tags = script["tags"]
        context.chapters = script["chapters"]
        context.sub_topic = sub_topic

        context.add_stat("script_sentences", len(sentences))
        context.add_stat("script_title", script["title"])

        return context

    def _check_novelty(self, title: str, script: list, channel: str) -> tuple:
        """Check if content is novel."""
        if not self.novelty_guard:
            return True, 0.0

        try:
            is_fresh, sim = self.novelty_guard.is_novel(
                title=title,
                script_text=" ".join(script),
                channel=channel
            )
            return is_fresh, sim
        except Exception as exc:
            logger.warning(f"Novelty check failed: {exc}")
            return True, 0.0

    def validate_output(self, context: PipelineContext):
        """Validate script output."""
        if not context.script:
            raise ValueError("Script not generated")
        if not context.get_sentences():
            raise ValueError("Script has no sentences")
        if not context.title:
            raise ValueError("Script has no title")
