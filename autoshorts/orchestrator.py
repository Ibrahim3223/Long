# FILE: autoshorts/orchestrator.py
# -*- coding: utf-8 -*-
"""
High level orchestration for generating complete videos.
✅ FIXED: Added use_novelty attribute
✅ PERFORMANCE: Reduced retry delays and optimized parallel processing
✅ Path sanitization
✅ Parallel TTS support  
✅ Audio durations collection for YouTube chapters
✅ Thumbnail generation from Pexels
"""
from __future__ import annotations

import os
import re
import sys
import time
import random
import shutil
import hashlib
import logging
import pathlib
import tempfile
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter  # ✅ EKLENDİ
from urllib3.util.retry import Retry  # ✅ EKLENDİ

from autoshorts.config import settings
from autoshorts.content.gemini_client import GeminiClient
from autoshorts.video.pexels_client import PexelsClient
from autoshorts.video.search_optimizer import VideoSearchOptimizer
from autoshorts.video.shot_variety import ShotVarietyManager
from autoshorts.tts.unified_handler import UnifiedTTSHandler as TTSHandler
from autoshorts.captions.renderer import CaptionRenderer
from autoshorts.audio.bgm_manager import BGMManager
from autoshorts.utils.ffmpeg_utils import run, ffprobe_duration
from autoshorts.content.text_utils import extract_keywords, simplify_query
from autoshorts.state.novelty_guard import NoveltyGuard
from autoshorts.state.state_guard import StateGuard
from autoshorts.content.quality_scorer import QualityScorer
from autoshorts.metadata.generator import MetadataGenerator

PEXELS_SEARCH_ENDPOINT = "https://api.pexels.com/videos/search"

logger = logging.getLogger(__name__)


# ============================================================================
# ✅ RATE LIMITING
# ============================================================================

class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, min_interval: float = 0.5):
        """
        Args:
            min_interval: Minimum seconds between API calls
        """
        self.min_interval = min_interval
        self.last_call_time = 0
    
    def wait(self):
        """Wait if necessary to respect rate limit."""
        current_time = time.time()
        time_since_last = current_time - self.last_call_time
        
        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            logger.debug(f"Rate limiting: waiting {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        self.last_call_time = time.time()


# ============================================================================
# ✅ PATH SANITIZATION HELPER
# ============================================================================

def sanitize_path(path: str) -> str:
    """
    Sanitize path to avoid FFmpeg issues with spaces and special characters.
    
    Args:
        path: Input file path
    
    Returns:
        Sanitized path safe for FFmpeg
    """
    # Replace spaces with underscores
    path = path.replace(" ", "_")
    # Remove problematic characters
    path = re.sub(r'[^\w\-_\./\\]', '', path)
    return path


# ============================================================================
# DATACLASSES
# ============================================================================

@dataclass
class ClipCandidate:
    """Candidate video clip from Pexels."""
    pexels_id: str
    path: str
    duration: float
    url: str


class _ClipCache:
    """Simple disk-based cache for Pexels video clips."""
    def __init__(self, cache_dir: pathlib.Path):
        self.cache_dir = cache_dir / ".clip_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, url: str) -> Optional[str]:
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        cached_path = self.cache_dir / f"{url_hash}.mp4"
        return str(cached_path) if cached_path.exists() else None

    def put(self, url: str, path: str) -> None:
        try:
            url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
            dest = self.cache_dir / f"{url_hash}.mp4"
            shutil.copy2(path, dest)
        except Exception as exc:
            logger.debug("Failed to cache clip: %s", exc)


class ShortsOrchestrator:
    """Main orchestrator for video generation with performance improvements."""

    def __init__(
        self,
        channel_id: str,
        temp_dir: str,
        api_key: str,
        pexels_key: str,
        pixabay_key: Optional[str] = None,
        use_novelty: bool = True,
        ffmpeg_preset: str = "veryfast",
    ):
        self.channel_id = channel_id
        self.temp_dir = pathlib.Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.ffmpeg_preset = ffmpeg_preset

        # ✅ Store API keys
        self.pexels_key = pexels_key        

        # API clients
        self.gemini = GeminiClient(api_key=api_key)
        self.pexels = PexelsClient(api_key=pexels_key)
        
        # ✅ Pixabay as backup
        try:
            from autoshorts.video.pixabay_client import PixabayClient
            self.pixabay = PixabayClient(api_key=pixabay_key)
            logger.info("✅ Pixabay client initialized as backup")
        except Exception as e:
            logger.warning(f"⚠️ Pixabay client initialization failed: {e}")
            self.pixabay = None

        # TTS
        self.tts = TTSHandler()

        # Caption renderer
        self.caption_renderer = CaptionRenderer()

        # BGM manager
        self.bgm_manager = BGMManager()

        # Quality scorer
        try:
            self.quality_scorer = QualityScorer()
        except Exception as exc:
            logger.warning("Quality scorer init failed: %s", exc)
            self.quality_scorer = None

        # Metadata generator (pass Gemini API key for AI-powered metadata)
        try:
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            self.metadata_generator = MetadataGenerator(gemini_api_key=gemini_api_key)
            logger.info("Metadata generator initialized")
        except Exception as exc:
            logger.warning("Metadata generator init failed: %s", exc)
            self.metadata_generator = None

        # Video search optimizer
        try:
            self.search_optimizer = VideoSearchOptimizer()
            logger.info("Video search optimizer initialized")
        except Exception as exc:
            logger.warning("Video search optimizer init failed: %s", exc)
            self.search_optimizer = None

        # Shot variety manager
        try:
            variety_strength = getattr(settings, "SHOT_VARIETY_STRENGTH", "medium")
            self.shot_variety = ShotVarietyManager(variety_strength=variety_strength)
            logger.info(f"Shot variety manager initialized (strength: {variety_strength})")
        except Exception as exc:
            logger.warning("Shot variety manager init failed: %s", exc)
            self.shot_variety = None

        # Novelty guard
        self.use_novelty = use_novelty
        if use_novelty:
            try:
                self.novelty_guard = NoveltyGuard()
                logger.info("Novelty guard enabled")
            except Exception as exc:
                logger.warning("Novelty guard init failed: %s", exc)
                self.novelty_guard = None
        else:
            self.novelty_guard = None

        # State guard
        try:
            self.state_guard = StateGuard(channel=channel_id)
            logger.info("State guard initialized")
        except Exception as exc:
            logger.warning("State guard init failed: %s", exc)
            self.state_guard = None

        # HTTP session for downloads
        self._http = requests.Session()
        retry_strategy = Retry(
            total=2,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._http.mount("http://", adapter)
        self._http.mount("https://", adapter)

        # Rate limiter for Pexels
        self._pexels_rate_limiter = RateLimiter(min_interval=0.8)

        # Thread pool for parallel processing
        self._executor = ThreadPoolExecutor(max_workers=4)

        # Track used video IDs to avoid duplicates
        self._used_video_ids: Set[str] = set()
        
        # ✅ Track Pexels rate limit status
        self._pexels_rate_limited = False

        logger.info("ShortsOrchestrator initialized for channel: %s", channel_id)

    def __del__(self):
        """Cleanup thread pool on deletion."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def produce_video(
        self, topic_prompt: str, max_retries: int = 3
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """Produce a full video for topic_prompt."""
        logger.info("=" * 70)
        logger.info("🎬 START VIDEO PRODUCTION")
        logger.info("=" * 70)
        logger.info("📝 Topic: %s", topic_prompt[:120])
    
        # ✅ Track sub_topic for this production session
        selected_sub_topic = None

        for attempt in range(1, max_retries + 1):
            try:
                if attempt > 1:
                    # ✅ PERFORMANCE: Reduced retry delay from 8s to 3s
                    delay = min(3, 1 * attempt)
                    logger.info("⏳ Retry %d/%d in %ds", attempt, max_retries, delay)
                    time.sleep(delay)

                script = self._generate_script(topic_prompt)
                if not script:
                    logger.warning("Script generation failed")
                    continue
            
                # ✅ Extract sub_topic if it was used
                mode = os.getenv("MODE") or getattr(settings, "CHANNEL_MODE", None)
                if mode and not selected_sub_topic and hasattr(self, 'novelty_guard'):
                    try:
                        # Get the sub_topic that was used
                        selected_sub_topic = getattr(script, '_sub_topic', None)
                    except:
                        pass

                sentences_txt = [s.get("text", "") for s in script.get("sentences", [])]
                is_fresh, similarity = self._check_novelty_safe(
                    title=script.get("title", ""), script=sentences_txt
                )
                if not is_fresh:
                    logger.warning(
                        "Script too similar to recent ones (similarity=%.2f), retrying",
                        similarity,
                    )
                    continue

                video_path = self._render_from_script(script)
                if not video_path:
                    continue

                metadata = {
                    "title": script.get("title", ""),
                    "description": script.get("description", ""),
                    "tags": script.get("tags", []),
                    "script": script,
                }

                # ✅ DÜZELTME: Doğru parametrelerle çağır
                self._save_script_state_safe(
                    title=script.get("title", ""),
                    script=sentences_txt,
                    sub_topic=selected_sub_topic
                )
           
                # ✅ Pass sub_topic when registering
                self._novelty_add_used_safe(
                    title=script["title"], 
                    script=sentences_txt,
                    sub_topic=selected_sub_topic
                )

                logger.info("=" * 70)
                logger.info("✅ Video generation successful")
                logger.info("=" * 70)
                return video_path, metadata

            except KeyboardInterrupt:
                raise
            except Exception as exc:
                logger.error("Attempt %d/%d failed: %s", attempt, max_retries, exc)
                logger.debug("", exc_info=True)

        logger.error("All attempts failed")
        return None, None

    # ------------------------------------------------------------------
    # Script & TTS
    # ------------------------------------------------------------------

    def _generate_script(self, topic_prompt: str) -> Optional[Dict]:
        """
        Generate script using GeminiClient with mode and sub-topic support.
    
        ✅ UPDATED: Now passes mode and sub_topic to Gemini for channel-specific content
        """
        try:
            logger.info("🤖 Generating script via Gemini...")
        
            # ✅ Get channel mode from settings/environment
            mode = os.getenv("MODE") or getattr(settings, "CHANNEL_MODE", None)
        
            # ✅ Get or select sub-topic
            sub_topic = None
            if mode and hasattr(self, 'novelty_guard') and self.use_novelty:
                try:
                    sub_topic = self.novelty_guard.pick_sub_topic(
                        channel=self.channel_id,
                        mode=mode
                    )
                    logger.info(f"🎯 Selected sub-topic: {sub_topic}")
                except Exception as e:
                    logger.warning(f"⚠️ Sub-topic selection failed: {e}")
                    # Fallback to random sub-topic
                    from autoshorts.state.novelty_guard import SUB_TOPIC_POOLS
                    pool = SUB_TOPIC_POOLS.get(mode, SUB_TOPIC_POOLS.get("_default", []))
                    if pool:
                        import random
                        sub_topic = random.choice(pool)
                        logger.info(f"🎯 Using fallback sub-topic: {sub_topic}")
        
            # ✅ NEW: Get script style config (if using new system)
            script_style_config = None
            try:
                from autoshorts.config.config_manager import ConfigManager
                config = ConfigManager.get_instance(channel_name=self.channel_id)
                if hasattr(config, 'content') and hasattr(config.content, 'script_style'):
                    # Convert to dict for Gemini
                    script_style_config = {
                        "hook_intensity": config.content.script_style.hook_intensity,
                        "cold_open": config.content.script_style.cold_open,
                        "hook_max_words": config.content.script_style.hook_max_words,
                        "use_modular_structure": config.content.script_style.use_modular_structure,
                        "cliffhanger_frequency": config.content.script_style.cliffhanger_frequency,
                        "max_sentence_length": config.content.script_style.max_sentence_length,
                        "conversational_tone": config.content.script_style.conversational_tone,
                        "use_analogies": config.content.script_style.use_analogies,
                        "evergreen_only": config.content.script_style.evergreen_only,
                        "avoid_academic_tone": config.content.script_style.avoid_academic_tone,
                        "show_dont_tell": config.content.script_style.show_dont_tell,
                        "cta_softness": config.content.script_style.cta_softness,
                        "cta_max_words": config.content.script_style.cta_max_words,
                    }
                    logger.info("✅ Using ENHANCED SCRIPT STYLE from ConfigManager")
            except Exception as e:
                logger.debug(f"ConfigManager not available or no script_style: {e}")
                logger.info("🔄 Falling back to LEGACY script generation")

            # ✅ Pass mode, sub_topic AND script_style_config to Gemini
            content_response = self.gemini.generate(
                topic=topic_prompt,
                mode=mode,
                sub_topic=sub_topic,
                script_style_config=script_style_config,
            )
        
            if not content_response:
                return None
            
            # ✅ Store sub_topic in script for later use
            if sub_topic:
                content_response._sub_topic = sub_topic
            
            # Convert ContentResponse to dict format
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
                "_sub_topic": sub_topic  # Store for later
            }
            
            # Build sentences list
            sentences = []
            
            # Hook'u ekle
            if script["hook"]:
                sentences.append({"text": script["hook"], "type": "hook"})
            
            # Script cümlelerini ekle (ilk cümle hook ile aynıysa atla)
            for idx, sentence in enumerate(script["script"]):
                normalized_sentence = sentence.strip().lower()
                normalized_hook = script["hook"].strip().lower() if script["hook"] else ""
                
                # İlk cümle hook ile benzer mi kontrol et
                if idx == 0 and normalized_sentence == normalized_hook:
                    logger.info("⚠️ Skipping first sentence (duplicate of hook)")
                    continue
                
                # Boş cümleleri de atla
                if not sentence.strip():
                    continue
                    
                sentences.append({"text": sentence, "type": "content"})
            
            # CTA ekle (son cümle ile aynıysa ekleme)
            if script["cta"]:
                normalized_cta = script["cta"].strip().lower()
                
                # Son cümle CTA ile aynı mı kontrol et
                if sentences and sentences[-1]["text"].strip().lower() == normalized_cta:
                    logger.info("⚠️ Skipping duplicate CTA (same as last sentence)")
                    sentences[-1]["type"] = "cta"
                else:
                    sentences.append({"text": script["cta"], "type": "cta"})
            
            script["sentences"] = sentences
            
            # Validate
            if not script.get("sentences"):
                logger.error("Script has no sentences")
                return None
            
            logger.info("✅ Script generated: %d sentences", len(script["sentences"]))
            logger.info(f"📝 Title: {script['title']}")
            
            # ✅ ENHANCED: Comprehensive validation + scoring
            if self.quality_scorer and sentences:
                try:
                    sentence_texts = [s["text"] for s in sentences if s.get("text")]

                    # Run comprehensive validation
                    validation_results = self.quality_scorer.comprehensive_validation(
                        sentences=sentence_texts,
                        title=script["title"],
                        script_style_config=script_style_config,
                    )

                    script["quality_scores"] = validation_results["scores"]
                    script["validation"] = {
                        "valid": validation_results["valid"],
                        "overall_score": validation_results["overall_score"],
                        "issues": validation_results["issues"],
                    }

                    logger.info(f"📊 Quality: {validation_results['overall_score']:.1f}/10")
                    logger.info(f"✅ Valid: {validation_results['valid']}")

                    if validation_results["issues"]:
                        logger.warning(f"⚠️ Quality issues ({len(validation_results['issues'])}):")
                        for issue in validation_results["issues"][:3]:
                            logger.warning(f"  - {issue}")

                    # Reject low-quality scripts
                    if not validation_results["valid"]:
                        logger.error(f"❌ Script rejected: Quality score {validation_results['overall_score']:.1f} < 5.5")
                        return None

                except Exception as e:
                    logger.warning(f"Quality validation failed: {e}")

            # ✅ ENHANCED: Generate viral metadata (titles, description, thumbnail text)
            if self.metadata_generator:
                try:
                    # Get language from settings/environment
                    lang = os.getenv("LANG") or getattr(settings, "LANG", "en")

                    logger.info(f"🎯 Generating viral metadata (lang: {lang})...")
                    enhanced_metadata = self.metadata_generator.generate_all_metadata(
                        script=script,
                        main_topic=topic_prompt,
                        lang=lang
                    )

                    # Use enhanced metadata
                    script["title"] = enhanced_metadata["title"]
                    script["description"] = enhanced_metadata["description"]
                    script["thumbnail_text"] = enhanced_metadata["thumbnail_text"]
                    script["title_candidates"] = enhanced_metadata.get("title_candidates", [])
                    script["title_score"] = enhanced_metadata.get("title_score", 0)
                    if "keywords" in enhanced_metadata:
                        script["keywords"] = enhanced_metadata["keywords"]

                    source = enhanced_metadata.get("source", "unknown")
                    logger.info(f"🎯 Enhanced title ({source}): {enhanced_metadata['title']}")
                    logger.info(f"📊 Title score: {enhanced_metadata.get('title_score', 0):.1f}/10")
                    logger.info(f"🖼️ Thumbnail text: {enhanced_metadata['thumbnail_text']}")

                except Exception as e:
                    logger.warning(f"Metadata generation failed: {e}")
                    logger.debug("", exc_info=True)

            return script
        except Exception as exc:
            logger.error("Script generation error: %s", exc)
            logger.debug("", exc_info=True)
            return None

    def _split_continuous_audio(
        self,
        continuous_audio_path: str,
        segments: List[Dict],
        sentences: Sequence[Dict]
    ) -> List[Optional[Tuple[str, List[Tuple[str, float]]]]]:
        """
        Split continuous audio file into individual sentence files.

        Args:
            continuous_audio_path: Path to continuous audio file
            segments: List of segment dicts with start/end times
            sentences: Original sentence dicts

        Returns:
            List of (audio_path, word_timings) tuples
        """
        from autoshorts.utils.ffmpeg_utils import run

        results = [None] * len(segments)

        for idx, segment in enumerate(segments):
            try:
                start_time = segment["start"]
                duration = segment["duration"]

                # Output path for this segment
                out_path = str(self.temp_dir / f"tts_{idx:03d}.wav")

                # Extract segment using FFmpeg
                run([
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", continuous_audio_path,
                    "-ss", f"{start_time:.3f}",
                    "-t", f"{duration:.3f}",
                    "-ar", "48000", "-ac", "1",
                    out_path
                ])

                # Validate
                if not os.path.exists(out_path):
                    logger.error(f"Segment {idx}: Failed to create audio file")
                    continue

                file_size = os.path.getsize(out_path)
                if file_size < 1000:
                    logger.error(f"Segment {idx}: Audio file too small ({file_size} bytes)")
                    continue

                # For continuous mode, we don't have word timings per segment
                # (they're for the entire continuous audio)
                # This is OK - word timings are optional for most use cases
                results[idx] = (out_path, [])

                logger.debug(f"✅ Segment {idx}: {duration:.2f}s extracted")

            except Exception as e:
                logger.error(f"Failed to extract segment {idx}: {e}")
                results[idx] = None

        return results

    def _generate_all_tts(
        self, sentences: Sequence[Dict]
    ) -> List[Optional[Tuple[str, List[Tuple[str, float]]]]]:
        """
        ✅ Generate TTS with improved reliability and resource management.
        ✅ NEW: Continuous speech mode for natural flow between sentences
        """
        # ⚠️ DISABLED: Continuous speech causing sync issues
        # TODO: Re-enable after fixing timing bugs
        # Reason: Video-audio-caption timing drift causing poor UX

        logger.info("🎙️ Using sentence-by-sentence TTS (stable timing)")

        results = [None] * len(sentences)
        
        def process_sentence_safe(idx: int, sent: Dict) -> Tuple[int, Optional[Tuple[str, List[Tuple[str, float]]]]]:
            """Process single sentence with comprehensive error handling."""
            text = sent.get("text", "")
            
            if not text.strip():
                logger.warning(f"Scene {idx}: Empty text")
                return idx, None
            
            out_path = str(self.temp_dir / f"tts_{idx:03d}.wav")
            out_path = sanitize_path(out_path)
            
            # ✅ Multiple retry attempts
            for attempt in range(3):
                try:
                    # Generate TTS
                    duration, words = self.tts.synthesize(text, out_path)
                    
                    # Validate output
                    if not os.path.exists(out_path):
                        raise ValueError(f"Audio file not created: {out_path}")
                    
                    file_size = os.path.getsize(out_path)
                    if file_size < 1000:
                        raise ValueError(f"Audio file too small: {file_size} bytes")
                    
                    if duration < 0.5:
                        raise ValueError(f"Audio duration too short: {duration}s")
                    
                    logger.debug(f"✅ Scene {idx}: {duration:.2f}s, {len(words)} words")
                    return idx, (out_path, words)
                    
                except Exception as exc:
                    logger.warning(f"⚠️ Scene {idx} attempt {attempt+1}/3 failed: {exc}")
                    
                    # Clean up failed file
                    if os.path.exists(out_path):
                        try:
                            os.remove(out_path)
                        except:
                            pass
                    
                    if attempt < 2:
                        import time
                        time.sleep(0.5 * (attempt + 1))
                    else:
                        logger.error(f"❌ Scene {idx} failed after 3 attempts")
                        return idx, None
            
            return idx, None
        
        # ✅ STRATEGY: Process in small batches to avoid resource exhaustion
        BATCH_SIZE = 10
        total_sentences = len(sentences)
        
        logger.info(f"🎙️ Generating TTS for {total_sentences} sentences in batches of {BATCH_SIZE}")
        
        for batch_start in range(0, total_sentences, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total_sentences)
            batch_indices = range(batch_start, batch_end)
            
            logger.info(f"📊 Processing batch {batch_start//BATCH_SIZE + 1}/{(total_sentences + BATCH_SIZE - 1)//BATCH_SIZE} (sentences {batch_start}-{batch_end-1})")
            
            # Process batch in parallel
            batch_results = {}
            try:
                futures = {}
                for idx in batch_indices:
                    future = self._executor.submit(process_sentence_safe, idx, sentences[idx])
                    futures[future] = idx
                
                # Collect batch results with timeout
                for future in as_completed(futures, timeout=120):
                    try:
                        idx, result = future.result(timeout=60)
                        batch_results[idx] = result
                    except Exception as exc:
                        idx = futures[future]
                        logger.error(f"❌ Batch future failed for scene {idx}: {exc}")
                        batch_results[idx] = None
                
            except Exception as exc:
                logger.error(f"❌ Batch processing failed: {exc}")
                # Fill failed batch with None
                for idx in batch_indices:
                    if idx not in batch_results:
                        batch_results[idx] = None
            
            # ✅ Update results - ensure correct assignment
            for idx in batch_indices:
                if idx in batch_results:
                    results[idx] = batch_results[idx]
                    
                    # Verify assignment worked
                    if results[idx] is not None:
                        audio_path, words = results[idx]
                        if not os.path.exists(audio_path):
                            logger.error(f"❌ Scene {idx}: Audio file missing after assignment: {audio_path}")
                            results[idx] = None
            
            # Log batch progress
            batch_success = sum(1 for idx in batch_indices if results[idx] is not None)
            logger.info(f"✅ Batch complete: {batch_success}/{len(batch_indices)} successful")
        
        # ✅ Sequential fallback for failed sentences
        failed_indices = [i for i, r in enumerate(results) if r is None]
        
        if failed_indices:
            logger.warning(f"⚠️ {len(failed_indices)} scenes failed, retrying sequentially...")
            
            for idx in failed_indices:
                logger.info(f"🔄 Sequential retry {idx+1}/{total_sentences}")
                _, result = process_sentence_safe(idx, sentences[idx])
                results[idx] = result
                
                if result:
                    logger.info(f"✅ Sequential success for scene {idx}")
        
        # ✅ Final validation
        successful = sum(1 for r in results if r is not None)
        success_rate = (successful / total_sentences * 100) if total_sentences > 0 else 0
        
        logger.info(f"📊 TTS Generation Complete: {successful}/{total_sentences} ({success_rate:.1f}%)")
        
        # ✅ DEBUG: Log first few results
        for i in range(min(3, len(results))):
            if results[i]:
                audio_path, words = results[i]
                logger.debug(f"Result[{i}]: {audio_path}, {len(words)} words")
            else:
                logger.debug(f"Result[{i}]: None")
        
        if success_rate < 80:
            logger.error(f"❌ TTS success rate too low: {success_rate:.1f}%")
            return [None] * len(sentences)
        
        return results
    # ------------------------------------------------------------------
    # Main render loop
    # ------------------------------------------------------------------

    def _render_from_script(self, script: Dict) -> Optional[str]:
        """Render complete video from script."""
        sentences = script.get("sentences", [])
        if not sentences:
            logger.error("Script has no sentences")
            return None

        # ✅ Reset shot variety for new video
        if self.shot_variety:
            self.shot_variety.reset()
            logger.debug("Shot variety manager reset for new video")

        logger.info("🎙️ Generating TTS for %d sentences", len(sentences))
        tts_results = self._generate_all_tts(sentences)

        total_audio_duration = 0.0
        for res in tts_results:
            if res:
                path, _ = res
                total_audio_duration += ffprobe_duration(path)

        logger.info("📊 Total audio duration: %.2fs", total_audio_duration)

        # ✅ NEW: Build video pool upfront (user request: collect videos once, not per-scene)
        # User feedback: "videonun ana konusu hakkında sahne sayısı kadar video toplarız"
        video_pool = self._build_video_pool(script, len(sentences))
        if not video_pool:
            logger.error("❌ Failed to build video pool")
            return None
        logger.info(f"✅ Video pool ready: {len(video_pool)} videos for {len(sentences)} scenes")

        chapters = script.get("chapters", [])

        logger.info("🎥 Rendering %d scenes", len(sentences))
        scene_paths = []
        skipped_scenes = 0

        for idx, (sent, tts_result) in enumerate(zip(sentences, tts_results)):
            audio_path = None
            scene_path = None
            captioned = None
            if not tts_result:
                logger.warning("Skipping scene %d (no TTS)", idx)
                skipped_scenes += 1
                continue

            audio_path, words = tts_result
            audio_dur = ffprobe_duration(audio_path)
            text = sent.get("text", "")
            sentence_type = sent.get("type", "content")

            # Find current chapter
            current_chapter = None
            search_queries = None
            for chapter in chapters:
                start_idx = chapter.get("start_sentence_index", 0)
                end_idx = chapter.get("end_sentence_index", len(sentences))
                if start_idx <= idx < end_idx:
                    current_chapter = chapter.get("title")
                    search_queries = chapter.get("search_queries", [])
                    break

            # ✅ MODIFIED: Use video pool instead of per-scene search
            scene_path = self._prepare_scene_clip(
                text=text,
                keywords=extract_keywords(text, lang=getattr(settings, "LANG", "en")),
                duration=audio_dur,
                index=idx,
                sentence_type=sentence_type,
                search_queries=search_queries,
                chapter_title=current_chapter,
                total_sentences=len(sentences),
                video_pool=video_pool,  # ✅ NEW: Pass video pool
            )

            if not scene_path:
                logger.warning("Skipping scene %d (no video clip)", idx)
                skipped_scenes += 1
                continue

            CAPTION_OFFSET = 0.5
            words_with_offset = [(w, d) for w, d in words] if words else []
            
            captioned = self._render_captions(
                video_path=scene_path,
                text=text,
                words=words,
                duration=audio_dur,
                sentence_type=sentence_type,
                caption_offset=0.0,  # ✅ Offset kaldırıldı
            )

            final_scene = self._mux_audio(
                video_path=captioned,
                audio_path=audio_path,
                duration=audio_dur,
                index=idx,
            )

            if final_scene:
                scene_paths.append(final_scene)
                logger.info("✅ Scene %d/%d complete", idx + 1, len(sentences))
                
                # ✅ Clean up intermediate files to save disk space
                try:
                    if os.path.exists(audio_path):
                        os.remove(audio_path)
                    if scene_path and os.path.exists(scene_path):
                        os.remove(scene_path)
                    if captioned != scene_path and os.path.exists(captioned):
                        os.remove(captioned)
                    logger.debug(f"Cleaned up intermediate files for scene {idx}")
                except Exception as cleanup_exc:
                    logger.debug(f"Cleanup warning: {cleanup_exc}")

        # ✅ Collect audio durations for YouTube chapters
        audio_durations = []
        for idx, tts_result in enumerate(tts_results):
            if tts_result:
                audio_path, _ = tts_result
                try:
                    audio_durations.append(ffprobe_duration(audio_path))
                except:
                    audio_durations.append(0.0)
            else:
                audio_durations.append(0.0)
        
        script["audio_durations"] = audio_durations
        logger.info(f"📊 Collected {len(audio_durations)} audio durations for chapters")

        # ✅ Accept video if we have at least 30% of scenes (was 40%)
        if not scene_paths:
            logger.error("No scenes rendered successfully")
            return None
        
        if skipped_scenes > 0:
            success_rate = len(scene_paths) / len(sentences) * 100
            logger.warning(f"⚠️ Skipped {skipped_scenes} scenes - {success_rate:.1f}% success rate")
            
            # ✅ More tolerant: Allow video if we have at least 30% of scenes
            if success_rate < 30:
                logger.error(f"❌ Too many skipped scenes ({success_rate:.1f}% < 30%)")
                return None
            
            logger.info(f"✅ Accepting video with {success_rate:.1f}% scene coverage")

        # Concatenate all scenes
        concat_out = str(self.temp_dir / "concat.mp4")
        logger.info("🔗 Concatenating %d scenes", len(scene_paths))
        self._concat_segments(scene_paths, concat_out)

        if not os.path.exists(concat_out):
            logger.error("Concatenation failed")
            return None
        
        # ✅ Clean up scene files after concatenation
        try:
            for scene_path in scene_paths:
                if os.path.exists(scene_path):
                    os.remove(scene_path)
            logger.debug(f"Cleaned up {len(scene_paths)} scene files")
        except Exception as cleanup_exc:
            logger.debug(f"Scene cleanup warning: {cleanup_exc}")

        # Add background music if enabled
        final_out = str(self.temp_dir / "final.mp4")
        if getattr(settings, "BGM_ENABLED", False):
            logger.info("🎵 Adding background music")
            bgm_track = self.bgm_manager.select_track()
            if bgm_track:
                self.bgm_manager.mix_bgm(concat_out, bgm_track, final_out, total_audio_duration)
            else:
                shutil.copy2(concat_out, final_out)
        else:
            shutil.copy2(concat_out, final_out)
        
        # ✅ Clean up concat file
        try:
            if os.path.exists(concat_out):
                os.remove(concat_out)
            logger.debug("Cleaned up concat file")
        except Exception as cleanup_exc:
            logger.debug(f"Concat cleanup warning: {cleanup_exc}")

        # ✅ Generate thumbnail
        thumbnail_path = self._generate_thumbnail(script, final_out)
        if thumbnail_path:
            script["thumbnail_path"] = thumbnail_path
            logger.info(f"🖼️ Thumbnail saved: {thumbnail_path}")
        
        return final_out if os.path.exists(final_out) else None

    # ------------------------------------------------------------------
    # ✅ NEW METHOD: Thumbnail Generation
    # ------------------------------------------------------------------

    def _generate_thumbnail(self, script: Dict, video_path: str) -> Optional[str]:
        """
        Generate thumbnail from Pexels image with VIRAL text overlay.

        Args:
            script: Script dictionary
            video_path: Path to video file

        Returns:
            Path to thumbnail image or None
        """
        try:
            import requests
            from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
            from io import BytesIO
            import textwrap

            # Extract keywords from title
            title = script.get("title", "")
            keywords = extract_keywords(title, lang=getattr(settings, "LANG", "en"))
            search_query = simplify_query(" ".join(keywords[:3]))

            logger.info(f"🔍 Searching Pexels for thumbnail: {search_query}")

            # Search Pexels for images
            url = "https://api.pexels.com/v1/search"
            params = {
                "query": search_query,
                "per_page": 5,
                "orientation": "landscape"
            }

            response = requests.get(
                url,
                headers={"Authorization": self.pexels_key},
                params=params,
                timeout=10
            )

            if response.status_code != 200:
                logger.warning(f"Pexels image search failed: {response.status_code}")
                return None

            data = response.json()
            photos = data.get("photos", [])

            if not photos:
                logger.warning("No thumbnail images found")
                return None

            # Get first image
            photo = photos[0]
            image_url = photo.get("src", {}).get("large2x") or photo.get("src", {}).get("large")

            if not image_url:
                logger.warning("No image URL found")
                return None

            logger.info(f"⬇️ Downloading thumbnail from: {image_url}")

            # Download image
            img_response = requests.get(image_url, timeout=15)
            img_response.raise_for_status()

            # Open and resize to YouTube thumbnail size (1280x720)
            img = Image.open(BytesIO(img_response.content))
            img = img.resize((1280, 720), Image.Resampling.LANCZOS)
            img = img.convert("RGB")

            # ✅ VIRAL ENHANCEMENT: Increase contrast & saturation for eye-catching effect
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.2)
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(1.15)

            # ✅ Add semi-transparent dark overlay for text readability
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 100))
            img_rgba = img.convert('RGBA')
            img = Image.alpha_composite(img_rgba, overlay).convert('RGB')

            # ✅ Add viral text overlay
            draw = ImageDraw.Draw(img)

            # ✅ Use pre-generated thumbnail_text if available, otherwise extract from title
            thumbnail_text = script.get("thumbnail_text", "")

            if not thumbnail_text:
                # Fallback: Extract key words from title for thumbnail text (max 3-4 words)
                title_words = title.upper().split()

                # ✅ Smart text selection: Pick most impactful words
                # Priority: numbers, "why", "how", "secret", adjectives
                priority_words = []
                for word in title_words:
                    if any(char.isdigit() for char in word):  # Numbers
                        priority_words.insert(0, word)
                    elif word.lower() in ['why', 'how', 'what', 'secret', 'hidden', 'truth']:
                        priority_words.insert(0, word)
                    elif len(word) > 4:  # Longer words are usually more meaningful
                        priority_words.append(word)

                # Take top 3-4 words, max 35 chars
                thumbnail_text = " ".join(priority_words[:4])
                if len(thumbnail_text) > 35:
                    thumbnail_text = " ".join(priority_words[:3])
                if len(thumbnail_text) > 35:
                    thumbnail_text = thumbnail_text[:32] + "..."
            else:
                # Ensure uppercase for viral effect
                thumbnail_text = thumbnail_text.upper()

            # ✅ Try to load system fonts (fallback to default)
            font_size = 90
            stroke_width = 8

            try:
                # Try common bold fonts
                for font_path in [
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                    "/System/Library/Fonts/Helvetica.ttc",
                    "/Windows/Fonts/impact.ttf"
                ]:
                    try:
                        font = ImageFont.truetype(font_path, font_size)
                        break
                    except:
                        continue
                else:
                    # Fallback to default
                    font = ImageFont.load_default()
                    font_size = 40
                    stroke_width = 3
            except:
                font = ImageFont.load_default()
                font_size = 40
                stroke_width = 3

            # ✅ Wrap text if too long
            wrapped_lines = textwrap.wrap(thumbnail_text, width=20)

            # Calculate total text height
            line_heights = []
            for line in wrapped_lines:
                bbox = draw.textbbox((0, 0), line, font=font)
                line_heights.append(bbox[3] - bbox[1])

            total_text_height = sum(line_heights) + (len(wrapped_lines) - 1) * 20

            # ✅ Position text in center (slightly upper for better composition)
            y_position = (720 - total_text_height) // 2 - 50

            # ✅ Draw text with viral styling (white text + thick black outline)
            for line in wrapped_lines:
                # Get text bounding box for centering
                bbox = draw.textbbox((0, 0), line, font=font)
                text_width = bbox[2] - bbox[0]
                x_position = (1280 - text_width) // 2

                # Draw black outline (stroke)
                draw.text(
                    (x_position, y_position),
                    line,
                    font=font,
                    fill='white',
                    stroke_width=stroke_width,
                    stroke_fill='black'
                )

                y_position += line_heights[wrapped_lines.index(line)] + 20

            # ✅ Optional: Add colored accent bar behind text for extra pop
            if len(wrapped_lines) == 1:
                text_bbox = draw.textbbox((0, 0), wrapped_lines[0], font=font)
                bar_width = text_bbox[2] - text_bbox[0] + 60
                bar_height = text_bbox[3] - text_bbox[1] + 30
                bar_x = (1280 - bar_width) // 2
                bar_y = (720 - total_text_height) // 2 - 65

                # Semi-transparent colored bar (red/yellow for attention)
                bar_overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
                bar_draw = ImageDraw.Draw(bar_overlay)

                # Use channel-specific color or default red
                bar_color = (255, 50, 50, 180)  # Red with transparency

                bar_draw.rounded_rectangle(
                    [(bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height)],
                    radius=15,
                    fill=bar_color
                )

                img = img.convert('RGBA')
                img = Image.alpha_composite(img, bar_overlay).convert('RGB')

                # Redraw text on top of bar
                draw = ImageDraw.Draw(img)
                text_x = (1280 - (text_bbox[2] - text_bbox[0])) // 2
                text_y = (720 - total_text_height) // 2 - 50

                draw.text(
                    (text_x, text_y),
                    wrapped_lines[0],
                    font=font,
                    fill='white',
                    stroke_width=stroke_width,
                    stroke_fill='black'
                )

            # Save thumbnail
            thumbnail_path = str(self.temp_dir / "thumbnail_final.jpg")
            img.save(thumbnail_path, "JPEG", quality=95)

            logger.info(f"✅ Viral thumbnail created: {thumbnail_path} (1280x720) - Text: '{thumbnail_text}'")
            return thumbnail_path

        except Exception as exc:
            logger.warning(f"Thumbnail generation failed: {exc}")
            logger.debug("", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Video building
    # ------------------------------------------------------------------

    def _build_video_pool(self, script: Dict, target_count: int) -> List[ClipCandidate]:
        """
        ✅ NEW: Build video pool upfront (user request).

        User feedback: "Şöyle yapabiliriz belki, videonun ana konusu hakkında sahne
        sayısı kadar video toplarız pexels ve pixabaydan. sonra bu videoları her
        sahnede biri olacak şekilde kullanırız."

        Strategy:
        1. Extract 2-3 main keywords from script topic/title
        2. Search Pexels + Pixabay ONCE with these keywords
        3. Collect 100-150 videos into pool
        4. Shuffle for variety
        5. Use throughout video (one per scene)

        Args:
            script: Video script with title, sentences
            target_count: Number of videos needed (scene count)

        Returns:
            List of ClipCandidate objects (shuffled and ready)
        """
        try:
            # 1. Extract main topic keywords (2-3 words)
            # ✅ IMPROVED: Use script content instead of title (title has clickbait words)
            main_visual_focus = script.get("main_visual_focus", "")

            # Get first few sentences for topic extraction (skip hook - it's often clickbait)
            sentences = script.get("sentences", [])
            content_sentences = [
                s.get("text", "") for s in sentences
                if s.get("type") != "hook" and s.get("type") != "cta"
            ][:10]  # First 10 content sentences

            script_sample = " ".join(content_sentences)

            # Combine script content + main_visual_focus (NOT title!)
            search_text = f"{script_sample} {main_visual_focus}".strip()

            keywords = extract_keywords(search_text, lang=getattr(settings, "LANG", "en"))

            # Take top 2-3 keywords
            main_keywords = keywords[:3]

            if not main_keywords:
                # Fallback: use main_visual_focus or title
                if main_visual_focus:
                    main_keywords = main_visual_focus.split()[:3]
                else:
                    # Last resort: use title (remove clickbait words)
                    title = script.get("title", "")
                    # Filter out common clickbait words
                    clickbait_words = {
                        "shocking", "amazing", "incredible", "unbelievable", "mind-blowing",
                        "secret", "hidden", "revealed", "discover", "won't", "believe"
                    }
                    title_words = [
                        w for w in title.lower().split()
                        if w not in clickbait_words and len(w) > 3
                    ]
                    main_keywords = title_words[:3]

            # ✅ IMPROVED: Create multiple diverse search queries
            # User feedback: Better video variety and relevance
            primary_query = simplify_query(" ".join(main_keywords))

            # Create alternative queries for diversity
            search_queries = [primary_query]

            # Add main_visual_focus as separate query if different
            if main_visual_focus and simplify_query(main_visual_focus) != primary_query:
                search_queries.append(simplify_query(main_visual_focus))

            # Add 2-word combinations for broader results
            if len(main_keywords) >= 2:
                search_queries.append(simplify_query(" ".join(main_keywords[:2])))

            # Remove duplicates while preserving order
            search_queries = list(dict.fromkeys(search_queries))[:3]  # Max 3 queries

            logger.info(f"🎬 Building video pool using {len(search_queries)} queries: {search_queries}")
            logger.info(f"🎯 Target: {target_count} videos")

            # 2. Collect videos from Pexels + Pixabay
            video_pool = []

            # Calculate how many videos to fetch per query
            fetch_count = int(target_count * 1.5)
            fetch_count = max(100, min(200, fetch_count))  # Clamp to 100-200
            per_query_count = fetch_count // len(search_queries)  # Distribute across queries

            # Try Pexels first (primary source) - MULTIPLE QUERIES for diversity
            if not self._pexels_rate_limited:
                for query_idx, query in enumerate(search_queries):
                    if len(video_pool) >= fetch_count:
                        break  # Already have enough

                    try:
                        logger.info(f"📥 Pexels query {query_idx+1}/{len(search_queries)}: '{query}' (requesting {per_query_count})")
                        result = self.pexels.search_videos(query, per_page=per_query_count, page=1)

                        if isinstance(result, dict):
                            videos = result.get("videos", [])
                        elif isinstance(result, list):
                            videos = result
                        else:
                            videos = []

                        logger.info(f"  ✅ Returned {len(videos)} videos")

                        # Convert to ClipCandidates
                        for video in videos:
                            video_id = str(video.get("id", ""))

                            # Skip if already used in previous videos or in current pool
                            if video_id in self._used_video_ids:
                                continue

                            video_files = video.get("video_files", [])
                            base_duration = video.get("duration", 0)

                            if not video_files or base_duration <= 0:
                                continue

                            # Find HD quality
                            for vf in video_files:
                                if vf.get("quality") == "hd" and vf.get("width", 0) >= 1080:
                                    candidate = ClipCandidate(
                                        pexels_id=video_id,
                                        path="",
                                        duration=base_duration,
                                        url=vf.get("link", "")
                                    )
                                    video_pool.append(candidate)
                                    self._used_video_ids.add(video_id)
                                    break

                            # Stop if we have enough
                            if len(video_pool) >= fetch_count:
                                break

                    except Exception as e:
                        if "429" in str(e) or "rate limit" in str(e).lower():
                            logger.warning("⚠️ Pexels rate limited during pool build")
                            self._pexels_rate_limited = True
                            break  # Stop trying more queries
                        else:
                            logger.warning(f"⚠️ Pexels query '{query}' failed: {e}")
                            continue  # Try next query

                logger.info(f"✅ Pexels total: {len(video_pool)} HD videos from {len(search_queries)} queries")

            # Try Pixabay if we need more videos - MULTIPLE QUERIES for diversity
            if len(video_pool) < target_count and self.pixabay and self.pixabay.enabled:
                for query_idx, query in enumerate(search_queries):
                    if len(video_pool) >= fetch_count:
                        break  # Already have enough

                    try:
                        needed = fetch_count - len(video_pool)
                        logger.info(f"📥 Pixabay query {query_idx+1}/{len(search_queries)}: '{query}' (requesting {min(needed, per_query_count)})")

                        result = self.pixabay.search_videos(query, per_page=min(needed, per_query_count), page=1)
                        hits = result.get("hits", [])

                        logger.info(f"  ✅ Returned {len(hits)} videos")

                        for video in hits:
                            video_id = f"pixabay_{video.get('id', '')}"

                            if video_id in self._used_video_ids:
                                continue

                            video_url = self.pixabay.get_video_url(video, quality="large")
                            if not video_url:
                                continue

                            candidate = ClipCandidate(
                                pexels_id=video_id,
                                path="",
                                duration=video.get("duration", 0),
                                url=video_url
                            )
                            video_pool.append(candidate)
                            self._used_video_ids.add(video_id)

                            if len(video_pool) >= fetch_count:
                                break

                    except Exception as e:
                        logger.warning(f"⚠️ Pixabay query '{query}' failed: {e}")
                        continue  # Try next query

                logger.info(f"✅ Pixabay total: {len(video_pool) - len([v for v in video_pool if not v.pexels_id.startswith('pixabay')])} videos added")
                logger.info(f"✅ Total pool size: {len(video_pool)} videos")

            # 3. Check if we have enough videos
            if len(video_pool) < target_count:
                logger.warning(f"⚠️ Video pool has only {len(video_pool)}/{target_count} videos")

                # If we have less than 50% of target, fail
                if len(video_pool) < target_count * 0.5:
                    logger.error(f"❌ Insufficient videos in pool: {len(video_pool)} < {target_count * 0.5}")
                    return []

                logger.info("✅ Continuing with reduced pool (will reuse videos if needed)")

            # ✅ NEW: Filter and separate CTA/subscribe videos
            # User request: "subscribe ile ilgili olanları sadece sonunda olacak şekilde"
            # (Subscribe-related videos should only be at the end)

            # We can't filter based on video content (we haven't downloaded yet)
            # But we CAN organize pool to avoid repetitive patterns
            # Strategy: Ensure maximum diversity by shuffling thoroughly

            # 4. Shuffle for maximum variety
            random.shuffle(video_pool)

            # ✅ Additional shuffle to ensure CTA videos don't cluster
            # (If pool has subscribe videos, they'll be distributed randomly, not clustered)
            for _ in range(3):  # Multiple shuffles for better randomization
                random.shuffle(video_pool)

            logger.info(f"🎲 Shuffled pool: {len(video_pool)} videos ready (3x shuffle for variety)")
            logger.info(f"📊 Pool coverage: {len(video_pool)}/{target_count} scenes ({len(video_pool)/target_count*100:.1f}%)")

            return video_pool

        except Exception as exc:
            logger.error(f"❌ Video pool build failed: {exc}")
            logger.debug("", exc_info=True)
            return []

    def _prepare_scene_clip(
        self,
        text: str,
        keywords: List[str],
        duration: float,
        index: int,
        sentence_type: str = "content",
        search_queries: Optional[List[str]] = None,
        chapter_title: Optional[str] = None,
        total_sentences: int = 1,
        video_pool: Optional[List[ClipCandidate]] = None,  # ✅ NEW: Video pool
    ) -> Optional[str]:
        """
        Prepare video clip for a scene.

        ✅ MODIFIED: Now uses video pool instead of per-scene search.
        """

        # ✅ NEW: Use video pool if available
        if video_pool:
            # Calculate which video to use from pool
            # Use modulo to loop if we have fewer videos than scenes
            pool_index = index % len(video_pool)
            candidate = video_pool[pool_index]

            logger.debug(f"Scene {index}: Using video from pool[{pool_index}] (ID: {candidate.pexels_id})")

        else:
            # ⚠️ FALLBACK: Old per-scene search (should rarely happen)
            logger.warning(f"Scene {index}: No video pool, falling back to per-scene search")

            # ✅ ENHANCED: Plan shot variety and pacing
            enhanced_keywords = keywords
            if self.shot_variety:
                try:
                    shot_plan = self.shot_variety.plan_shot(
                        sentence=text,
                        sentence_index=index,
                        sentence_type=sentence_type,
                        total_sentences=total_sentences,
                        keywords=keywords,
                    )
                    # Use shot-specific keywords
                    enhanced_keywords = shot_plan.search_keywords
                    logger.debug(
                        f"Scene {index}: {shot_plan.shot_type.value} shot, "
                        f"{shot_plan.pacing_style.value} pacing"
                    )
                except Exception as e:
                    logger.debug(f"Shot planning failed: {e}")
                    enhanced_keywords = keywords

            # ✅ ENHANCED: Context-aware video search
            candidate = self._find_best_video(
                sentence=text,
                keywords=enhanced_keywords,
                duration=duration,
                search_queries=search_queries,
                chapter_title=chapter_title,
                sentence_type=sentence_type,
                timeout=8
            )

        # ✅ Check if we got a candidate (should always have one from pool)
        if not candidate:
            logger.error(f"❌ Scene {index}: No candidate video available")
            return None

        local_path = self._download_clip(candidate)
        if not local_path:
            return None

        output_path = str(self.temp_dir / f"scene_{index:03d}.mp4")
        output_path = sanitize_path(output_path)

        self._process_video_clip(
            input_path=local_path,
            output_path=output_path,
            target_duration=duration
        )

        # ✅ Clean up downloaded clip immediately
        try:
            if os.path.exists(local_path):
                os.remove(local_path)
        except:
            pass

        return output_path if os.path.exists(output_path) else None

    def _find_best_video(
        self,
        sentence: str,
        keywords: List[str],
        duration: float,
        search_queries: Optional[List[str]] = None,
        chapter_title: Optional[str] = None,
        sentence_type: str = "content",
        timeout: int = 10
    ) -> Optional[ClipCandidate]:
        """Find best matching video clip from Pexels/Pixabay with context awareness."""

        # ✅ ENHANCED: Use VideoSearchOptimizer for better queries
        if self.search_optimizer:
            try:
                search_query_objects = self.search_optimizer.build_search_queries(
                    sentence=sentence,
                    keywords=keywords,
                    chapter_title=chapter_title,
                    search_queries=search_queries,
                    sentence_type=sentence_type,
                )
                queries = [q.text for q in search_query_objects[:5]]
                logger.info(f"🔍 Context-aware search: {len(queries)} queries")
            except Exception as e:
                logger.warning(f"Search optimizer failed: {e}, using fallback")
                # Fallback to legacy query building
                queries = self._build_legacy_queries(keywords, search_queries, chapter_title)
        else:
            # Legacy query building
            queries = self._build_legacy_queries(keywords, search_queries, chapter_title)

        logger.info(f"🔍 Queries: {queries[:3]}")

        # ✅ ENHANCED: Try multi-provider search first (Pexels + Pixabay + Free providers)
        try:
            from autoshorts.video.multi_provider import MultiProviderVideoSearch

            multi_search = MultiProviderVideoSearch(
                pexels_key=self.pexels_client.api_key if hasattr(self, 'pexels_client') and self.pexels_client else None,
                pixabay_key=self.pixabay_client.api_key if hasattr(self, 'pixabay_client') and self.pixabay_client else None
            )

            for query in queries:
                query = simplify_query(query)
                if not query:
                    continue

                videos = multi_search.search(query, min_duration=5, max_results=10)

                if videos:
                    # Get first video (best match)
                    video = videos[0]
                    logger.info(f"✅ Multi-provider: found video from {video.get('provider', 'unknown')}")

                    # Convert to expected format
                    candidate = self._download_video_from_provider(video)
                    if candidate:
                        return candidate

            logger.warning("⚠️ Multi-provider search: no videos found")

        except Exception as e:
            logger.warning(f"⚠️ Multi-provider search failed: {e}, falling back to Pexels only")

        # ✅ Fallback: Try Pexels only (original method)
        for query in queries:
            if self._pexels_rate_limited:
                break

            query = simplify_query(query)
            if not query:
                continue

            try:
                candidate = self._search_pexels_for_query(query)
                if candidate:
                    return candidate
            except Exception as e:
                if "PEXELS_RATE_LIMIT" in str(e):
                    logger.warning("⚠️ Pexels rate limited globally, switching to Pixabay")
                    self._pexels_rate_limited = True
                    break

        return None

    def _build_legacy_queries(
        self,
        keywords: List[str],
        search_queries: Optional[List[str]],
        chapter_title: Optional[str]
    ) -> List[str]:
        """Build search queries using legacy method (fallback)."""
        queries = []

        # 1. Use search queries (highest priority)
        if search_queries:
            queries.extend(search_queries[:3])

        # 2. Use chapter title
        if chapter_title and chapter_title not in queries:
            queries.append(chapter_title)

        # 3. Use keywords
        if keywords:
            # Combine keywords for better results
            queries.append(" ".join(keywords[:2]))
            queries.extend(keywords[:3])

        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in queries:
            normalized = simplify_query(q).lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique_queries.append(q)

        return unique_queries[:5]

    def _download_video_from_provider(self, video: Dict) -> Optional[ClipCandidate]:
        """
        Download video from any provider (Pexels, Pixabay, Mixkit, Videezy, Coverr).

        Args:
            video: Video dict from multi-provider search

        Returns:
            ClipCandidate or None
        """
        try:
            provider = video.get('provider', 'unknown')
            video_url = video.get('url', '')

            if not video_url:
                logger.warning(f"No URL for {provider} video")
                return None

            # Download video with verification
            import requests
            import tempfile
            import os

            # ✅ Get expected file size if available
            head_response = None
            try:
                head_response = requests.head(video_url, timeout=5)
                expected_size = int(head_response.headers.get('Content-Length', 0))
            except:
                expected_size = 0

            # ✅ Download with longer timeout for large files
            response = requests.get(video_url, timeout=60, stream=True)
            response.raise_for_status()

            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                total_downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # filter out keep-alive chunks
                        tmp.write(chunk)
                        total_downloaded += len(chunk)
                temp_path = tmp.name

            # ✅ Verify download completed
            actual_size = os.path.getsize(temp_path)

            # Check if file is too small (likely corrupted)
            if actual_size < 10000:  # Less than 10KB
                logger.warning(f"❌ Downloaded file too small ({actual_size} bytes), likely corrupted")
                os.remove(temp_path)
                return None

            # Check if size matches expected (if we got Content-Length)
            if expected_size > 0 and abs(actual_size - expected_size) > 1000:
                logger.warning(f"⚠️ Size mismatch: expected {expected_size}, got {actual_size}")
                # Don't fail, just warn - some servers don't send accurate Content-Length

            # ✅ Verify moov atom exists (basic MP4 validation)
            try:
                with open(temp_path, 'rb') as f:
                    f.seek(0, 2)  # Seek to end
                    file_size = f.tell()

                    # Check last 1KB for moov atom
                    search_size = min(100000, file_size)  # Check last 100KB
                    f.seek(max(0, file_size - search_size))
                    tail_data = f.read()

                    if b'moov' not in tail_data:
                        logger.warning(f"❌ moov atom not found - file likely incomplete")
                        os.remove(temp_path)
                        return None
            except Exception as verify_err:
                logger.warning(f"⚠️ MP4 verification failed: {verify_err}")
                os.remove(temp_path)
                return None

            # Create ClipCandidate (width/height not needed - ClipCandidate only has 4 fields)
            candidate = ClipCandidate(
                pexels_id=f"{provider}_{hash(video_url)}",
                path=temp_path,
                url=video_url,
                duration=float(video.get('duration', 10))
            )

            logger.info(f"✅ Downloaded video from {provider}: {candidate.duration:.1f}s ({actual_size} bytes)")
            return candidate

        except Exception as e:
            logger.warning(f"Failed to download from {provider}: {e}")
            return None

    def _search_pexels_for_query(self, query: str) -> Optional[ClipCandidate]:
        """Search Pexels for a single query with advanced features."""
        if not settings.PEXELS_API_KEY:
            return None
        
        # Rate limiting
        self._pexels_rate_limiter.wait()
        
        for attempt in range(settings.PEXELS_RETRY_ATTEMPTS):
            try:
                result = self.pexels.search_videos(query, per_page=80, page=1)
                
                if isinstance(result, dict):
                    videos = result.get("videos", [])
                elif isinstance(result, list):
                    videos = result
                else:
                    return None
                
                if not videos:
                    continue
                
                # ✅ FİLTRELEME VE SEÇİM (short-video-maker'dan)
                filtered_videos = []
                
                for video in videos:
                    video_id = str(video.get("id", ""))
                    
                    # Skip used videos
                    if video_id in self._used_video_ids:
                        continue
                    
                    video_files = video.get("video_files", [])
                    base_duration = video.get("duration", 0)
                    
                    if not video_files or base_duration <= 0:
                        continue
                    
                    # ✅ 1. FPS NORMALIZASYONU (25 FPS'e normalize et)
                    fps = video_files[0].get("fps", 25)
                    if settings.PEXELS_FPS_NORMALIZE and fps < 25:
                        # Duration ayarlama (düşük FPS daha uzun görünür)
                        adjusted_duration = base_duration * (fps / 25)
                    else:
                        adjusted_duration = base_duration
                    
                    # ✅ 2. DURATION BUFFER (3 saniye ekstra)
                    required_duration = settings.SCENE_MIN_DURATION
                    if adjusted_duration < (required_duration + settings.PEXELS_DURATION_BUFFER):
                        continue
                    
                    # HD quality check
                    has_hd = False
                    for vf in video_files:
                        if vf.get("quality") == "hd" and vf.get("width", 0) >= 1080:
                            has_hd = True
                            filtered_videos.append({
                                "video": video,
                                "video_file": vf,
                                "adjusted_duration": adjusted_duration
                            })
                            break
                    
                    if not has_hd:
                        continue
                
                # ✅ 3. RANDOM SELECTION (çeşitlilik için)
                if filtered_videos:
                    if settings.PEXELS_RANDOM_SELECTION:
                        selected = random.choice(filtered_videos)
                    else:
                        selected = filtered_videos[0]
                    
                    video = selected["video"]
                    vf = selected["video_file"]
                    video_id = str(video.get("id", ""))
                    
                    candidate = ClipCandidate(
                        pexels_id=video_id,
                        path="",
                        duration=video.get("duration", 0),
                        url=vf.get("link", "")
                    )
                    self._used_video_ids.add(video_id)
                    logger.info(f"✅ Pexels (attempt {attempt+1}): {query} (ID: {video_id}, FPS: {fps}, Duration: {adjusted_duration:.1f}s)")
                    return candidate
            
            except Exception as exc:
                error_msg = str(exc)
                
                # ✅ 4. BETTER ERROR HANDLING
                if "429" in error_msg or "rate limit" in error_msg.lower() or "too many requests" in error_msg.lower():
                    logger.warning(f"⚠️ Pexels rate limited: {query} (attempt {attempt+1}/{settings.PEXELS_RETRY_ATTEMPTS})")
                    raise Exception("PEXELS_RATE_LIMIT")
                
                if "timeout" in error_msg.lower():
                    logger.warning(f"⚠️ Pexels timeout: {query} (attempt {attempt+1}/{settings.PEXELS_RETRY_ATTEMPTS})")
                    if attempt < settings.PEXELS_RETRY_ATTEMPTS - 1:
                        import time
                        time.sleep(1)  # 1 saniye bekle
                        continue
                
                logger.debug(f"Pexels search error for '{query}' (attempt {attempt+1}): {exc}")
                if attempt < settings.PEXELS_RETRY_ATTEMPTS - 1:
                    continue
        
        return None
    
    def _search_pixabay_for_query(self, query: str) -> Optional[ClipCandidate]:
        """Search Pixabay for a single query."""
        try:
            result = self.pixabay.search_videos(query, per_page=20, page=1)
            hits = result.get("hits", [])
            
            if not hits:
                return None
            
            # Try to find unused video
            for video in hits:
                video_id = f"pixabay_{video.get('id', '')}"
                
                if video_id in self._used_video_ids:
                    continue
                
                video_url = self.pixabay.get_video_url(video, quality="large")
                if not video_url:
                    continue
                
                candidate = ClipCandidate(
                    pexels_id=video_id,
                    path="",
                    duration=video.get("duration", 0),
                    url=video_url
                )
                self._used_video_ids.add(video_id)
                logger.info(f"✅ Pixabay: {query} (ID: {video_id})")
                return candidate
            
        except Exception as exc:
            logger.debug(f"Pixabay search error for '{query}': {exc}")
        
        return None
        
    def _download_clip(self, candidate: ClipCandidate) -> Optional[str]:
        """Download video clip."""
        # ✅ Safety check
        if not candidate:
            logger.error("Cannot download clip: candidate is None")
            return None
        
        # ✅ No cache - always download fresh to save disk space
        local_path = str(self.temp_dir / f"clip_{candidate.pexels_id}.mp4")
        local_path = sanitize_path(local_path)
        
        try:
            logger.debug(f"Downloading clip: {candidate.url}")
            # ✅ PERFORMANCE: Reduced timeout from 30s to 20s
            video_response = self._http.get(candidate.url, timeout=20)
            video_response.raise_for_status()
            
            with open(local_path, "wb") as f:
                f.write(video_response.content)
            
            return local_path
            
        except Exception as exc:
            logger.warning(f"Failed to download clip: {exc}")
            return None

    def _process_video_clip(
        self,
        input_path: str,
        output_path: str,
        target_duration: float
    ) -> None:
        """Process video clip to target duration and format."""
        try:
            clip_duration = ffprobe_duration(input_path)
            
            # Determine start point (random for variety)
            max_start = max(0, clip_duration - target_duration)
            start_time = random.uniform(0, max_start) if max_start > 0 else 0
            
            run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{start_time:.3f}",
                "-i", input_path,
                "-t", f"{target_duration:.3f}",
                "-vf", f"scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080",
                "-r", "30",
                "-c:v", "libx264",
                "-preset", self.ffmpeg_preset,
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-an",
                output_path
            ])
        except Exception as exc:
            logger.error("Video processing failed: %s", exc)
            logger.debug("", exc_info=True)

    def _render_captions(
        self,
        video_path: str,
        text: str,
        words: List[Tuple[str, float]],
        duration: float,
        sentence_type: str = "content",
        caption_offset: float = 0.0,
    ) -> str:
        """Render captions on video with optional time offset."""
        if not getattr(settings, "KARAOKE_CAPTIONS", True):
            return video_path

        try:
            output = self.caption_renderer.render(
                video_path=video_path,
                text=text,
                words=words,
                duration=duration,
                sentence_type=sentence_type,
                caption_offset=caption_offset,
            )
            return output if os.path.exists(output) else video_path
        
        except Exception as exc:
            logger.error("Caption render failed: %s", exc)
            logger.debug("", exc_info=True)
            return video_path

    def _mux_audio(
        self,
        video_path: str,
        audio_path: str,
        duration: float,
        index: int,
    ) -> Optional[str]:
        """Mux audio with video - simple and clean."""
        output = self.temp_dir / f"scene_{index:03d}_final.mp4"
        
        try:
            # Video duration'ı kontrol et
            video_duration = ffprobe_duration(video_path)
            
            # ✅ Kaç kere loop etmeli
            loop_count = int(duration / video_duration) + 1
            
            # ✅ Video'yu loop ederek uzat, audio ekle
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
                "-stream_loop", str(loop_count),
                "-i", video_path,
                "-i", audio_path,
                "-t", f"{duration:.3f}",
                "-c:v", "libx264",
                "-preset", self.ffmpeg_preset,
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "192k",
                "-ar", "48000",
                "-ac", "2",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-movflags", "+faststart",
                str(output),
            ]
            
            run(cmd, check=True)
            
            # Verify audio track
            probe_cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=codec_name",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(output)
            ]
            
            from subprocess import run as subprocess_run, PIPE
            result = subprocess_run(probe_cmd, stdout=PIPE, stderr=PIPE, text=True)
            
            if result.stdout.strip():
                logger.debug(f"✅ Scene {index} muxed (audio: {result.stdout.strip()}, duration: {duration:.2f}s)")
            else:
                logger.error(f"❌ Scene {index} has NO AUDIO TRACK!")
                return None
            
        except Exception as exc:
            logger.error(f"❌ Audio mux failed for scene {index}: {exc}")
            logger.debug("", exc_info=True)
            return None
        
        return str(output) if output.exists() else None

    def _concat_segments_with_crossfade(self, segments: List[str], output: str, fade_duration: float = 0.3) -> None:
        """
        Concatenate video segments with crossfade transitions.

        User feedback: "sahneler arası geçiş efektleri yok, daha smooth ve efektli geçiler gerekiyordu"

        Args:
            segments: List of video file paths
            output: Output file path
            fade_duration: Crossfade duration in seconds (default: 0.3s)
        """
        if len(segments) < 2:
            # No transitions needed for single segment
            logger.info("Single segment, skipping transitions")
            import shutil
            shutil.copy(segments[0], output)
            return

        logger.info(f"🎬 Concatenating {len(segments)} scenes with crossfade transitions ({fade_duration}s)")

        try:
            # Build FFmpeg filter_complex for xfade transitions
            # Example: [0:v][1:v]xfade=transition=fade:duration=0.3:offset=4.7[v01];[v01][2:v]xfade=...

            # Get durations of all segments
            durations = []
            for seg in segments:
                try:
                    dur = ffprobe_duration(seg)
                    durations.append(dur)
                except:
                    logger.warning(f"Failed to get duration for {seg}, using 5.0s")
                    durations.append(5.0)

            # Build input list
            inputs = []
            for seg in segments:
                inputs.extend(["-i", seg])

            # Build filter_complex
            filter_parts = []
            video_label = "0:v"
            audio_maps = []

            # Calculate offsets (cumulative duration - fade_duration for overlap)
            offset = 0.0
            for i in range(len(segments) - 1):
                next_label = f"v{i:02d}" if i < len(segments) - 2 else "vout"

                # Offset is when the next clip starts (accounting for fade overlap)
                offset += durations[i] - fade_duration

                # xfade syntax: [input1][input2]xfade=transition=fade:duration=X:offset=Y[output]
                filter_parts.append(
                    f"[{video_label}][{i+1}:v]xfade=transition=fade:duration={fade_duration}:offset={offset:.2f}[{next_label}]"
                )

                video_label = next_label

            # Audio: concatenate all audio streams
            audio_filter = "".join([f"[{i}:a]" for i in range(len(segments))]) + f"concat=n={len(segments)}:v=0:a=1[aout]"
            filter_parts.append(audio_filter)

            filter_complex = ";".join(filter_parts)

            # Build ffmpeg command
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
                *inputs,
                "-filter_complex", filter_complex,
                "-map", "[vout]",
                "-map", "[aout]",
                "-c:v", "libx264",
                "-preset", self.ffmpeg_preset,
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "192k",
                "-ar", "48000",
                "-ac", "2",
                "-movflags", "+faststart",
                output
            ]

            run(cmd, check=True)
            logger.info(f"✅ Crossfade concatenation successful ({len(segments)} scenes)")

        except Exception as exc:
            logger.warning(f"⚠️ Crossfade concat failed: {exc}, falling back to simple concat")
            # Fallback to simple concat
            self._concat_segments_simple(segments, output)

    def _concat_segments_simple(self, segments: Iterable[str], output: str) -> None:
        """Concatenate video segments without transitions (simple concat)."""
        import pathlib

        concat_file = pathlib.Path(output).with_suffix(".txt")
        try:
            with open(concat_file, "w", encoding="utf-8") as handle:
                for segment in segments:
                    handle.write(f"file '{segment}'\n")

            logger.info(f"🔗 Simple concatenation: {len(list(segments))} scenes")

            run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),
                "-c:v", "libx264",
                "-preset", self.ffmpeg_preset,
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "192k",
                "-ar", "48000",
                "-ac", "2",
                "-movflags", "+faststart",
                output
            ], check=True)
            logger.info("✅ Simple concatenation successful")

        except Exception as exc:
            logger.error(f"Concatenation failed: {exc}")
            logger.debug("", exc_info=True)
            raise

    def _concat_segments(self, segments: List[str], output: str) -> None:
        """
        Concatenate video segments (with or without transitions).

        Checks VIDEO_TRANSITIONS environment variable to decide which method to use.
        """
        use_transitions = os.getenv("VIDEO_TRANSITIONS", "1") == "1"
        fade_duration = float(os.getenv("TRANSITION_DURATION", "0.3"))

        if use_transitions and len(segments) > 1:
            logger.info("🎬 Using crossfade transitions")
            self._concat_segments_with_crossfade(segments, output, fade_duration)
        else:
            if not use_transitions:
                logger.info("Video transitions disabled (VIDEO_TRANSITIONS=0)")
            self._concat_segments_simple(segments, output)

    # ------------------------------------------------------------------
    # Helpers for novelty/state
    # ------------------------------------------------------------------

    def _check_novelty_safe(self, title: str, script: List[str]) -> Tuple[bool, float]:
        """Check if content is fresh/novel."""
        if not self.use_novelty or not self.novelty_guard:
            return True, 0.0
            
        try:
            is_fresh, sim = self.novelty_guard.is_novel(
                title=title,
                script_text=" ".join(script),
                channel=self.channel_id
            )
            return is_fresh, sim
        except Exception as exc:
            logger.warning("Novelty check failed: %s", exc)
            return True, 0.0

    def _novelty_add_used_safe(
        self, title: str, script: Union[str, List[str]], sub_topic: Optional[str] = None
    ) -> None:
        """
        ✅ UPDATED: Now tracks sub_topic for rotation
        """
        if not self.use_novelty or not self.novelty_guard:
            return
    
        try:
            # Get mode from settings
            mode = os.getenv("MODE") or getattr(settings, "CHANNEL_MODE", None)
        
            # Register with sub_topic
            self.novelty_guard.add_used_script(
                title=title,
                script=script,
                channel=self.channel_id,
                mode=mode,
                sub_topic=sub_topic
            )
            logger.info("✅ Registered script in novelty guard (sub_topic: %s)", sub_topic)
        except Exception as exc:
            logger.warning("Novelty add_used failed: %s", exc)

    def _save_script_state_safe(self, title: str, script: List[str], sub_topic: Optional[str] = None):
        """Safely save script to state."""
        if not hasattr(self, 'state_guard') or self.state_guard is None:
            return
        
        try:
            entity = sub_topic if sub_topic else title
            script_text = " ".join(script)
            
            # ✅ DÜZELTME: state_guard kullan (novelty_guard değil)
            content_hash = self.state_guard.make_content_hash(
                script_text=script_text,
                video_paths=[],
                audio_path=None
            )
            
            # ✅ Use mark_uploaded instead of save_script
            self.state_guard.mark_uploaded(
                entity=entity,
                script_text=script_text,
                content_hash=content_hash,
                video_path="pending",
                title=title
            )
            
            logger.info("Script state saved successfully")
            
        except Exception as exc:
            logger.warning(f"State save failed: {exc}")
