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

import hashlib
import logging
import os
import pathlib
import random
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Set, Any, Union

import requests
from requests.adapters import HTTPAdapter

from autoshorts.audio.bgm_manager import BGMManager
from autoshorts.captions.renderer import CaptionRenderer
from autoshorts.config import settings
from autoshorts.content.gemini_client import GeminiClient
from autoshorts.content.quality_scorer import QualityScorer
from autoshorts.content.text_utils import extract_keywords, simplify_query
from autoshorts.state.novelty_guard import NoveltyGuard
from autoshorts.state.state_guard import StateGuard
from autoshorts.tts.edge_handler import TTSHandler
from autoshorts.utils.ffmpeg_utils import ffprobe_duration, run
from autoshorts.video import PexelsClient

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
        api_key: Optional[str] = None,
        pexels_key: Optional[str] = None,
    ) -> None:
        # ✅ Sanitize temp directory path
        temp_dir = sanitize_path(temp_dir)
        
        self.channel_id = channel_id
        self.temp_dir = pathlib.Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # External services
        self.gemini = GeminiClient(api_key=api_key or settings.GEMINI_API_KEY)
        self.quality_scorer = QualityScorer()
        self.tts = TTSHandler()
        self.caption_renderer = CaptionRenderer()
        self.bgm_manager = BGMManager()
        self.state_guard = StateGuard(channel_id)
        self.novelty_guard = NoveltyGuard()
        
        # ✅ FIX: Add use_novelty attribute
        self.use_novelty = bool(
            getattr(settings, "NOVELTY_ENFORCE", True) or 
            (os.getenv("NOVELTY_ENFORCE", "1") == "1")
        )

        # Keys
        self.pexels_key = pexels_key or settings.PEXELS_API_KEY
        if not self.pexels_key:
            raise ValueError("PEXELS_API_KEY required")

        # Runtime caches
        # ✅ Disable clip cache to save disk space
        self._clip_cache = None
        self._video_candidates: Dict[str, List[ClipCandidate]] = {}
        self._video_url_cache: Dict[str, List[str]] = {}
        self._used_video_ids: Set[str] = set()
        
        # ✅ Rate limiter for Pexels API (0.3s between calls = max 200/min)
        self._pexels_rate_limiter = RateLimiter(min_interval=0.3)

        # Pexels client & HTTP session
        self.pexels = PexelsClient(api_key=self.pexels_key)
        self._http = requests.Session()
        self._http.headers.update({"Authorization": self.pexels_key})
        try:
            adapter = HTTPAdapter(
                pool_connections=8, 
                pool_maxsize=16,
                max_retries=2  # ✅ Reduce retries for faster failures
            )
            self._http.mount("https://", adapter)
            self._http.mount("http://", adapter)
        except Exception as exc:
            logger.debug("Unable to enable HTTP pooling: %s", exc)

        # Performance toggles
        self.fast_mode = bool(getattr(settings, "FAST_MODE", False) or (os.getenv("FAST_MODE", "0") == "1"))
        self.ffmpeg_preset = os.getenv(
            "FFMPEG_PRESET",
            "veryfast" if self.fast_mode else "medium"
        )
        
        # ✅ Thread pool for parallel processing
        max_workers = min(4, (os.cpu_count() or 4))
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        logger.info("🎬 ShortsOrchestrator ready (channel=%s, FAST_MODE=%s, workers=%d, novelty=%s)", 
                   channel_id, self.fast_mode, max_workers, self.use_novelty)

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
            
            # ✅ Add hook if exists
            if script["hook"]:
                sentences.append({"text": script["hook"], "type": "hook"})
            
            # ✅ DÜZELTME: İlk cümle hook ile aynıysa ekleme
            for sentence in script["script"]:
                # Normalize both for comparison
                normalized_sentence = sentence.strip().lower()
                normalized_hook = script["hook"].strip().lower() if script["hook"] else ""
                
                # Skip if this sentence is the same as hook
                if normalized_sentence == normalized_hook:
                    logger.info("⚠️ Skipping duplicate first sentence (same as hook)")
                    continue
                    
                sentences.append({"text": sentence, "type": "content"})
            
            # ✅ DÜZELTME: CTA son cümle ile aynıysa ekleme
            if script["cta"]:
                normalized_cta = script["cta"].strip().lower()
                
                # Check if last sentence is the same as CTA
                if sentences and sentences[-1]["text"].strip().lower() == normalized_cta:
                    logger.info("⚠️ Skipping duplicate CTA (same as last sentence)")
                    # Replace last sentence type to 'cta' instead of adding duplicate
                    sentences[-1]["type"] = "cta"
                else:
                    # CTA is different, add it
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
        ✅ Generate TTS for all sentences in parallel.
        
        Returns:
            List of (audio_path, word_timings) tuples or None for failed sentences
        """
        results = [None] * len(sentences)
        
        def process_sentence(idx: int, sent: Dict) -> Tuple[int, Optional[Tuple[str, List[Tuple[str, float]]]]]:
            """Process single sentence TTS."""
            text = sent.get("text", "")
            
            if not text.strip():
                return idx, None
            
            out_path = str(self.temp_dir / f"tts_{idx:03d}.wav")
            out_path = sanitize_path(out_path)
            
            try:
                duration, words = self.tts.synthesize(text, out_path)
                return idx, (out_path, words)
                
            except Exception as exc:
                logger.error(f"TTS failed for sentence {idx}: {exc}")
                return idx, None
        
        # ✅ PERFORMANCE: Process sentences in parallel with timeout
        futures = {}
        for idx, sent in enumerate(sentences):
            future = self._executor.submit(process_sentence, idx, sent)
            futures[future] = idx
        
        # Collect results with timeout
        for future in as_completed(futures, timeout=90):  # ✅ 90s timeout per batch
            try:
                idx, result = future.result(timeout=30)  # ✅ 30s timeout per sentence
                results[idx] = result
            except Exception as exc:
                idx = futures[future]
                logger.error(f"TTS future failed for sentence {idx}: {exc}")
                results[idx] = None
        
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

            captioned = self._render_captions(
                video_path=scene_path,
                text=text,
                words=words,
                duration=audio_dur,
                sentence_type=sentence_type,
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
        
        # ✅ IMPROVED: Better fallback with variety
        if not candidate:
            mode = os.getenv("MODE") or "general"
            fallback_terms = {
                "country_facts": ["nature landscape", "city skyline", "cultural heritage", "world travel", "countryside"],
                "history_story": ["ancient ruins", "historical", "heritage site", "old architecture"],
                "science": ["technology", "laboratory", "innovation", "research center"],
                "_default": ["world", "people working", "nature beauty", "urban life", "travel destinations"]
            }
            
            # ✅ Get fallback terms for mode
            terms = fallback_terms.get(mode, fallback_terms["_default"])
            
            # ✅ Try multiple fallback terms with random selection
            import random
            random.shuffle(terms)
            
            for fallback_term in terms[:3]:  # Try up to 3 different fallback terms
                logger.info(f"🔄 Trying generic fallback: {fallback_term}")
                
                self._pexels_rate_limiter.wait()
                try:
                    # ✅ Use random page for more variety
                    random_page = random.randint(1, 3)
                    result = self.pexels.search_videos(fallback_term, per_page=10, page=random_page)
                    videos = result.get("videos", []) if isinstance(result, dict) else result
                    
                    if videos:
                        # ✅ Shuffle videos for variety
                        random.shuffle(videos)
                        
                        for video in videos:
                            video_id = str(video.get("id", ""))
                            
                            # ✅ Skip if already used
                            if video_id in self._used_video_ids:
                                continue
                            
                            video_files = video.get("video_files", [])
                            
                            for vf in video_files:
                                if vf.get("quality") == "hd" and vf.get("width", 0) >= 1080:
                                    candidate = ClipCandidate(
                                        pexels_id=video_id,
                                        path="",
                                        duration=video.get("duration", 0),
                                        url=vf.get("link", "")
                                    )
                                    self._used_video_ids.add(video_id)
                                    logger.info(f"✅ Using fallback video: {fallback_term} (ID: {video_id})")
                                    break
                            
                            if candidate:
                                break
                    
                    if candidate:
                        break
                        
                except Exception as e:
                    logger.debug(f"Fallback search failed for '{fallback_term}': {e}")
                    continue
            
            if not candidate:
                logger.warning("❌ All fallback attempts failed")
        
        if not candidate:
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
        keywords: List[str],
        duration: float,
        search_queries: Optional[List[str]] = None,
        chapter_title: Optional[str] = None,
        timeout: int = 10
    ) -> Optional[ClipCandidate]:
        """Find best matching video clip from Pexels."""
        queries = []
        
        # ✅ OPTIMIZATION: Only use most relevant queries
        if search_queries:
            queries.append(search_queries[0])
        
        if chapter_title and not queries:
            queries.append(chapter_title)
        
        if not queries:
            queries.extend(keywords[:2])
        
        queries = queries[:2]
        
        # ✅ Try multiple pages to get more variety
        for page_num in [1, 2]:  # Try first 2 pages
            for query in queries:
                query = simplify_query(query)
                if not query:
                    continue
                    
                self._pexels_rate_limiter.wait()
                
                try:
                    # ✅ Search with page parameter for variety
                    result = self.pexels.search_videos(query, per_page=10, page=page_num)
                    
                    if isinstance(result, dict):
                        videos = result.get("videos", [])
                    elif isinstance(result, list):
                        videos = result
                    else:
                        logger.warning(f"Unexpected Pexels response type: {type(result)}")
                        continue
                    
                    if not videos:
                        logger.debug(f"No videos found for '{query}' (page {page_num})")
                        continue
                    
                    logger.info(f"🔍 Found {len(videos)} videos for '{query}' (page {page_num})")
                    
                    # ✅ Try to find unused video
                    for video in videos:
                        video_id = str(video.get("id", ""))
                        
                        # ✅ Skip if already used
                        if video_id in self._used_video_ids:
                            continue
                        
                        video_files = video.get("video_files", [])
                        for vf in video_files:
                            if vf.get("quality") == "hd" and vf.get("width", 0) >= 1080:
                                candidate = ClipCandidate(
                                    pexels_id=video_id,
                                    path="",
                                    duration=video.get("duration", 0),
                                    url=vf.get("link", "")
                                )
                                self._used_video_ids.add(video_id)
                                logger.info(f"✅ Selected video: {video_id} from '{query}'")
                                return candidate
                    
                except Exception as exc:
                    logger.debug(f"Search error for '{query}': {exc}")
                    continue
        
        return None
        
    def _download_clip(self, candidate: ClipCandidate) -> Optional[str]:
        """Download video clip."""
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
    ) -> str:
        """Render captions on video."""
        if not getattr(settings, "KARAOKE_CAPTIONS", True):
            return video_path

        try:
            output = self.caption_renderer.render(
                video_path=video_path,
                text=text,
                words=words,
                duration=duration,
                sentence_type=sentence_type,
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
        """Mux audio with video."""
        output = self.temp_dir / f"scene_{index:03d}_final.mp4"
        try:
            run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", video_path,
                "-i", audio_path,
                "-t", f"{duration:.3f}",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "160k",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-shortest",
                "-movflags", "+faststart",
                str(output),
            ])
        except Exception as exc:
            logger.error("Audio mux failed: %s", exc)
            logger.debug("", exc_info=True)
            return None

        return str(output) if output.exists() else None

    def _concat_segments(self, segments: Iterable[str], output: str) -> None:
        """Concatenate video segments."""
        concat_file = pathlib.Path(output).with_suffix(".txt")
        try:
            with open(concat_file, "w", encoding="utf-8") as handle:
                for segment in segments:
                    handle.write(f"file '{segment}'\n")

            # Try stream-copy concat (fast)
            try:
                run([
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", str(concat_file),
                    "-c", "copy",
                    output
                ])
            except Exception:
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
                    output
                ])
        finally:
            if concat_file.exists():
                concat_file.unlink()

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
