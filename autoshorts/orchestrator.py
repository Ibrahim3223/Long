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
from autoshorts.tts.unified_handler import UnifiedTTSHandler as TTSHandler
from autoshorts.captions.renderer import CaptionRenderer
from autoshorts.audio.bgm_manager import BGMManager
from autoshorts.utils.ffmpeg_utils import run, ffprobe_duration
from autoshorts.content.text_utils import extract_keywords, simplify_query
from autoshorts.state.novelty_guard import NoveltyGuard
from autoshorts.state.state_guard import StateGuard
from autoshorts.content.quality_scorer import QualityScorer

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
        
            # ✅ Pass mode and sub_topic to Gemini
            content_response = self.gemini.generate(
                topic=topic_prompt,
                mode=mode,
                sub_topic=sub_topic
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
            
            # ✅ Score script quality
            if self.quality_scorer and sentences:
                try:
                    sentence_texts = [s["text"] for s in sentences if s.get("text")]
                    quality_scores = self.quality_scorer.score(
                        sentences=sentence_texts,
                        title=script["title"]
                    )
                    script["quality_scores"] = quality_scores
                    logger.info(f"📊 Quality scores - Overall: {quality_scores.get('overall', 0):.1f}/10")
                except Exception as e:
                    logger.warning(f"Quality scoring failed: {e}")

            return script
        except Exception as exc:
            logger.error("Script generation error: %s", exc)
            logger.debug("", exc_info=True)
            return None

    def _generate_all_tts(
        self, sentences: Sequence[Dict]
    ) -> List[Optional[Tuple[str, List[Tuple[str, float]]]]]:
        """
        ✅ Generate TTS with improved reliability and resource management.
        """
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

        logger.info("🎙️ Generating TTS for %d sentences", len(sentences))
        tts_results = self._generate_all_tts(sentences)

        total_audio_duration = 0.0
        for res in tts_results:
            if res:
                path, _ = res
                total_audio_duration += ffprobe_duration(path)

        logger.info("📊 Total audio duration: %.2fs", total_audio_duration)

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

            scene_path = self._prepare_scene_clip(
                text=text,
                keywords=extract_keywords(text, lang=getattr(settings, "LANG", "en")),
                duration=audio_dur,
                index=idx,
                sentence_type=sentence_type,
                search_queries=search_queries,
                chapter_title=current_chapter,
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
        Generate thumbnail from Pexels image.
        
        Args:
            script: Script dictionary
            video_path: Path to video file
            
        Returns:
            Path to thumbnail image or None
        """
        try:
            import requests
            from PIL import Image, ImageDraw, ImageFont
            from io import BytesIO
            
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
            
            # Save thumbnail
            thumbnail_path = str(self.temp_dir / "thumbnail_final.jpg")
            img.save(thumbnail_path, "JPEG", quality=95)
            
            logger.info(f"✅ Thumbnail created: {thumbnail_path} (1280x720)")
            return thumbnail_path
            
        except Exception as exc:
            logger.warning(f"Thumbnail generation failed: {exc}")
            logger.debug("", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Video building
    # ------------------------------------------------------------------

    def _prepare_scene_clip(
        self,
        text: str,
        keywords: List[str],
        duration: float,
        index: int,
        sentence_type: str = "content",
        search_queries: Optional[List[str]] = None,
        chapter_title: Optional[str] = None,
    ) -> Optional[str]:
        """Prepare video clip for a scene."""
        
        # ✅ PERFORMANCE: Reduce video search timeout
        candidate = self._find_best_video(
            keywords=keywords,
            duration=duration,
            search_queries=search_queries,
            chapter_title=chapter_title,
            timeout=8
        )

        # ✅ Check if rate limited
        pexels_rate_limited = getattr(self, '_pexels_rate_limited', False)
        
        # ✅ IMPROVED: Better fallback with variety
        if not candidate:
            mode = os.getenv("MODE") or "general"
        # ✅ OPTIMIZED: Minimal fallback with cache priority
        if not candidate:
            mode = os.getenv("MODE") or "general"
            
        # ✅ IMPROVED: Diverse fallback categories
        if not candidate:
            mode = os.getenv("MODE") or "general"
            
            # ✅ More diverse fallback terms for each mode
            fallback_terms = {
                "country_facts": [
                    "urban cityscape", "cultural festival", "traditional market",
                    "countryside village", "mountain landscape", "coastal town"
                ],
                "history_story": [
                    "ancient architecture", "historical monument", "old library",
                    "museum artifacts", "vintage documents", "classical art"
                ],
                "science": [
                    "laboratory research", "scientific experiment", "technology innovation",
                    "microscope view", "data visualization", "modern workspace"
                ],
                "kids_educational": [
                    "colorful animation", "nature closeup", "animals wildlife",
                    "underwater scene", "space stars", "playground activity"
                ],
                "_default": [
                    "urban life", "people working", "nature landscape",
                    "technology modern", "creative workspace", "travel destination"
                ]
            }
            
            terms = fallback_terms.get(mode, fallback_terms["_default"])
            
            # ✅ Shuffle for variety
            import random
            random.shuffle(terms)
            
            # Try Pexels fallback only if not rate limited
            if not pexels_rate_limited:
                for fallback_term in terms[:3]:
                    try:
                        logger.info(f"🔄 Pexels fallback: {fallback_term}")
                        candidate = self._search_pexels_for_query(fallback_term)
                        if candidate:
                            break
                    except Exception as e:
                        if "RATE_LIMIT" in str(e):
                            pexels_rate_limited = True
                            self._pexels_rate_limited = True
                            break
            
            # Try Pixabay fallback
            if not candidate and self.pixabay and self.pixabay.enabled:
                for fallback_term in terms[:3]:
                    logger.info(f"🔄 Pixabay fallback: {fallback_term}")
                    candidate = self._search_pixabay_for_query(fallback_term)
                    if candidate:
                        break
            
            # ✅ EMERGENCY: Allow video reuse only as last resort
            if not candidate:
                logger.warning("⚠️ Emergency: Allowing video reuse")
                for fallback_term in terms[:1]:
                    try:
                        result = self.pexels.search_videos(fallback_term, per_page=15, page=1)
                        videos = result.get("videos", []) if isinstance(result, dict) else result
                        
                        if videos:
                            video = videos[0]
                            video_id = str(video.get("id", ""))
                            video_files = video.get("video_files", [])
                            
                            for vf in video_files:
                                if vf.get("quality") == "hd" and vf.get("width", 0) >= 1080:
                                    candidate = ClipCandidate(
                                        pexels_id=video_id,
                                        path="",
                                        duration=video.get("duration", 0),
                                        url=vf.get("link", "")
                                    )
                                    logger.info(f"✅ Reusing video: {fallback_term}")
                                    break
                            if candidate:
                                break
                    except:
                        continue
            
            # ✅ EMERGENCY: Eğer hala video yoksa, kullanılmış videoları tekrar kullan
            if not candidate:
                logger.warning("⚠️ No unused videos found, allowing video reuse for this scene")
                
                for fallback_term in terms[:1]:  # Sadece ilk terimi dene
                    try:
                        result = self.pexels.search_videos(fallback_term, per_page=15, page=1)
                        videos = result.get("videos", []) if isinstance(result, dict) else result
                        
                        if videos:
                            # ✅ İlk bulduğu videoyu kullan (kullanılmış olsa bile)
                            video = videos[0]
                            video_id = str(video.get("id", ""))
                            video_files = video.get("video_files", [])
                            
                            for vf in video_files:
                                if vf.get("quality") == "hd" and vf.get("width", 0) >= 1080:
                                    candidate = ClipCandidate(
                                        pexels_id=video_id,
                                        path="",
                                        duration=video.get("duration", 0),
                                        url=vf.get("link", "")
                                    )
                                    logger.info(f"✅ Reusing video: {fallback_term} (ID: {video_id})")
                                    break
                            
                            if candidate:
                                break
                    except Exception as e:
                        logger.debug(f"Emergency fallback failed: {e}")
                        continue
            
            if not candidate:
                logger.error("❌ All fallback attempts failed (including emergency reuse)")

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
        keywords: List[str],
        duration: float,
        search_queries: Optional[List[str]] = None,
        chapter_title: Optional[str] = None,
        timeout: int = 10
    ) -> Optional[ClipCandidate]:
        """Find best matching video clip from Pexels/Pixabay."""
        
        # ✅ Build diverse query list
        queries = []
        
        # 1. Use search queries (highest priority)
        if search_queries:
            queries.extend(search_queries[:3])  # Use top 3
        
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
        
        queries = unique_queries[:3]  # Limit to 5 queries
        
        logger.info(f"🔍 Searching with {len(queries)} queries: {queries}")
        
        # ✅ Try Pexels first with rate limit handling
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

    def _concat_segments(self, segments: Iterable[str], output: str) -> None:
        """Concatenate video segments."""
        import pathlib
        
        concat_file = pathlib.Path(output).with_suffix(".txt")
        try:
            with open(concat_file, "w", encoding="utf-8") as handle:
                for segment in segments:
                    handle.write(f"file '{segment}'\n")
            
            logger.info(f"🔗 Concatenating {len(list(segments))} scenes")
            
            # Try stream-copy concat (fast)
            try:
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
                logger.info("✅ Concatenation successful (re-encoded for compatibility)")
                
            except Exception as e:
                # Fallback to re-encode
                logger.info("Stream-copy concat failed, re-encoding...")
                run([
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", str(concat_file),
                    "-c:v", "libx264",
                    "-preset", self.ffmpeg_preset,
                    "-crf", "23",
                    "-c:a", "aac",
                    "-b:a", "160k",
                    "-movflags", "+faststart",
                    output
                ], check=True)
                logger.info("✅ Concatenation successful (re-encode)")
                
        except Exception as exc:
            logger.error(f"Concatenation failed: {exc}")
            logger.debug("", exc_info=True)
            raise
            
        finally:
            if concat_file.exists():
                try:
                    concat_file.unlink()
                except Exception:
                    pass

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
