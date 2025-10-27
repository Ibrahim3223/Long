# FILE: autoshorts/orchestrator.py
# -*- coding: utf-8 -*-
"""
High level orchestration for generating complete videos.
✅ FULL FIXED VERSION v4 - WITH THUMBNAIL & AUDIO DURATIONS SUPPORT
✅ Path sanitization
✅ Parallel TTS support  
✅ get_word_timings() fix
✅ extract_keywords() lang parameter fix
✅ Pexels rate limiting fix (429 error)
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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Set, Any

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

        # Keys
        self.pexels_key = pexels_key or settings.PEXELS_API_KEY
        if not self.pexels_key:
            raise ValueError("PEXELS_API_KEY required")

        # Runtime caches
        self._clip_cache = _ClipCache(self.temp_dir)
        self._video_candidates: Dict[str, List[ClipCandidate]] = {}
        self._video_url_cache: Dict[str, List[str]] = {}
        self._used_video_ids: Set[str] = set()
        
        # ✅ Rate limiter for Pexels API (0.5s between calls = max 120/min)
        self._pexels_rate_limiter = RateLimiter(min_interval=0.5)

        # Pexels client & HTTP session
        self.pexels = PexelsClient(api_key=self.pexels_key)
        self._http = requests.Session()
        self._http.headers.update({"Authorization": self.pexels_key})
        try:
            adapter = HTTPAdapter(pool_connections=8, pool_maxsize=16)
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

        logger.info("🎬 ShortsOrchestrator ready (channel=%s, FAST_MODE=%s, workers=%d)", 
                   channel_id, self.fast_mode, max_workers)

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

        for attempt in range(1, max_retries + 1):
            try:
                if attempt > 1:
                    delay = min(8, 2 * attempt)
                    logger.info("⏳ Retry in %ss", delay)
                    time.sleep(delay)

                script = self._generate_script(topic_prompt)
                if not script:
                    logger.warning("Script generation failed")
                    continue

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

                self._save_script_state_safe(script)
                self._novelty_add_used_safe(title=script["title"], script=sentences_txt)

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
        """Generate script using GeminiClient."""
        try:
            logger.info("🤖 Generating script via Gemini...")
            
            # Call Gemini API - returns ContentResponse dataclass
            content_response = self.gemini.generate(
                topic=topic_prompt,
                style="educational",
                duration=300  # 5 minutes target
            )
            
            if not content_response:
                logger.error("Gemini returned empty response")
                return None
            
            # Convert ContentResponse to dict format expected by orchestrator
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
                "sentences": []  # Will be populated below
            }
            
            # Build sentences list from script parts
            sentences = []
            
            # Add hook
            if script["hook"]:
                sentences.append({"text": script["hook"], "type": "hook"})
            
            # Add main script sentences
            for sentence in script["script"]:
                sentences.append({"text": sentence, "type": "content"})
            
            # Add CTA
            if script["cta"]:
                sentences.append({"text": script["cta"], "type": "cta"})
            
            script["sentences"] = sentences
            
            # Validate script structure
            if not script.get("sentences"):
                logger.error("Script has no sentences")
                return None
            
            logger.info("✅ Script generated: %d sentences", len(script["sentences"]))
            return script
            
        except Exception as exc:
            logger.error("Script generation failed: %s", exc)
            logger.debug("", exc_info=True)
            return None

    def _generate_all_tts(self, sentences: List[Dict]) -> List[Optional[Tuple[str, List[Tuple[str, float]]]]]:
        """
        ✅ PARALLEL TTS GENERATION
        Process multiple sentences concurrently for faster rendering.
        
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
        
        # Process sentences in parallel
        futures = {}
        for idx, sent in enumerate(sentences):
            future = self._executor.submit(process_sentence, idx, sent)
            futures[future] = idx
        
        # Collect results
        for future in as_completed(futures):
            try:
                idx, result = future.result(timeout=60)
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
        skipped_scenes = 0  # ✅ Track skipped scenes

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

            # ✅ FIX: extract_keywords now gets lang parameter
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
                # ✅ Don't give up - continue to next scene
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

        # ✅ Collect audio durations for YouTube chapters
        audio_durations = []
        for idx, tts_result in enumerate(tts_results):
            if tts_result:
                audio_path, _ = tts_result
                audio_durations.append(ffprobe_duration(audio_path))
            else:
                audio_durations.append(0.0)
        
        # ✅ Save audio durations to script for YouTube upload
        script["audio_durations"] = audio_durations
        logger.info(f"📊 Collected {len(audio_durations)} audio durations for chapters")

        # ✅ Accept video if we have at least 50% of scenes
        if not scene_paths:
            logger.error("No scenes rendered successfully")
            return None
        
        if skipped_scenes > 0:
            success_rate = len(scene_paths) / len(sentences) * 100
            logger.warning(f"⚠️ Skipped {skipped_scenes} scenes - {success_rate:.1f}% success rate")
            
            # ✅ Allow video if we have at least 40% of scenes
            if success_rate < 40:
                logger.error(f"❌ Too many skipped scenes ({success_rate:.1f}% < 40%)")
                return None

        # Concatenate all scenes
        concat_out = str(self.temp_dir / "concat.mp4")
        logger.info("🔗 Concatenating %d scenes", len(scene_paths))
        self._concat_segments(scene_paths, concat_out)

        if not os.path.exists(concat_out):
            logger.error("Concatenation failed")
            return None

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
            script: Script dict with search queries
            video_path: Path to final video
        
        Returns:
            Path to thumbnail image or None
        """
        try:
            import pathlib
            from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
            
            # Get search queries from script
            search_queries = script.get("search_queries", [])
            main_visual_focus = script.get("main_visual_focus", "")
            title = script.get("title", "")
            
            # Select thumbnail search query
            thumbnail_query = None
            if main_visual_focus:
                thumbnail_query = main_visual_focus
            elif search_queries:
                # Pick first or most relevant query
                thumbnail_query = search_queries[0]
            else:
                # Fallback to title keywords
                title_words = [w for w in title.split() if len(w) > 4]
                if title_words:
                    thumbnail_query = title_words[0]
            
            if not thumbnail_query:
                logger.warning("No thumbnail query available")
                return None
            
            logger.info(f"🔍 Searching Pexels for thumbnail: {thumbnail_query}")
            
            # Search Pexels for high-resolution image
            # Orientation: landscape for 16:9 thumbnail
            results = self._pexels_search(
                query=thumbnail_query,
                per_page=10,
                orientation="landscape"
            )
            
            if not results:
                logger.warning(f"No Pexels results for thumbnail: {thumbnail_query}")
                return None
            
            # Select best quality image
            best_image = None
            for result in results:
                # Get largest image URL
                if "src" in result and "original" in result["src"]:
                    best_image = result
                    break
            
            if not best_image:
                logger.warning("No suitable image found for thumbnail")
                return None
            
            # Download image
            image_url = best_image["src"]["original"]
            thumbnail_path = str(self.temp_dir / "thumbnail.jpg")
            
            logger.info(f"⬇️ Downloading thumbnail from: {image_url}")
            
            response = self._http.get(image_url, timeout=30)
            response.raise_for_status()
            
            with open(thumbnail_path, "wb") as f:
                f.write(response.content)
            
            # Process thumbnail: resize to 1280x720 (16:9)
            img = Image.open(thumbnail_path)
            
            # Resize and crop to 1280x720
            target_width = 1280
            target_height = 720
            target_ratio = target_width / target_height
            
            img_width, img_height = img.size
            img_ratio = img_width / img_height
            
            if img_ratio > target_ratio:
                # Image is wider, crop width
                new_height = target_height
                new_width = int(new_height * img_ratio)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                left = (new_width - target_width) // 2
                img = img.crop((left, 0, left + target_width, target_height))
            else:
                # Image is taller, crop height
                new_width = target_width
                new_height = int(new_width / img_ratio)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                top = (new_height - target_height) // 2
                img = img.crop((0, top, target_width, top + target_height))
            
            # Optional: Add subtle vignette effect
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.1)  # Slightly brighter
            
            # Save final thumbnail
            final_thumbnail_path = str(self.temp_dir / "thumbnail_final.jpg")
            img.save(final_thumbnail_path, "JPEG", quality=95, optimize=True)
            
            logger.info(f"✅ Thumbnail created: {final_thumbnail_path} (1280x720)")
            
            return final_thumbnail_path
            
        except Exception as exc:
            logger.error(f"Thumbnail generation failed: {exc}")
            logger.debug("", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # ✅ NEW METHOD: Pexels Image Search
    # ------------------------------------------------------------------

    def _pexels_search(
        self,
        query: str,
        per_page: int = 10,
        orientation: str = "landscape"
    ) -> List[Dict[str, Any]]:
        """
        Search Pexels for images.
        
        Args:
            query: Search query
            per_page: Results per page
            orientation: Image orientation (landscape/portrait/square)
        
        Returns:
            List of image results
        """
        try:
            # Apply rate limiting
            self._pexels_rate_limiter.wait()
            
            url = "https://api.pexels.com/v1/search"
            params = {
                "query": query,
                "per_page": per_page,
                "orientation": orientation
            }
            
            response = self._http.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            photos = data.get("photos", [])
            
            logger.debug(f"Pexels image search: {len(photos)} results for '{query}'")
            
            return photos
            
        except Exception as exc:
            logger.warning(f"Pexels image search failed for '{query}': {exc}")
            return []

    # ------------------------------------------------------------------
    # Scene preparation
    # ------------------------------------------------------------------

    def _prepare_scene_clip(
        self,
        text: str,
        keywords: Sequence[str],
        duration: float,
        index: int,
        sentence_type: str = "content",
        search_queries: Optional[List[str]] = None,
        chapter_title: Optional[str] = None,
    ) -> Optional[str]:
        """Download and prepare video clip for a scene."""
        
        # Build search query
        if search_queries:
            query = random.choice(search_queries)
        elif keywords:
            query = " ".join(keywords[:2])
        else:
            query = simplify_query(text)[:50]

        logger.info(f"🔍 Scene {index}: searching '{query}'")

        # Get video candidates
        candidates = self._get_video_candidates(query, duration)
        if not candidates:
            logger.warning(f"No video candidates for query: {query}")
            return None

        # Select best candidate
        candidate = random.choice(candidates[:3])  # Pick from top 3
        
        # Mark as used
        self._used_video_ids.add(candidate.pexels_id)

        # Prepare final clip
        output_path = str(self.temp_dir / f"scene_{index:03d}.mp4")
        output_path = sanitize_path(output_path)
        
        self._process_video_clip(
            input_path=candidate.path,
            output_path=output_path,
            target_duration=duration
        )

        return output_path if os.path.exists(output_path) else None

    def _get_video_candidates(
        self,
        query: str,
        min_duration: float
    ) -> List[ClipCandidate]:
        """Get video candidates from Pexels."""
        
        # Check cache first
        if query in self._video_candidates:
            return self._video_candidates[query]

        candidates = []
        
        try:
            # Apply rate limiting
            self._pexels_rate_limiter.wait()
            
            # Search Pexels
            params = {
                "query": query,
                "per_page": 15,
                "orientation": "portrait"  # Long-form is typically 16:9, but portrait works for shorts
            }
            
            response = self._http.get(PEXELS_SEARCH_ENDPOINT, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            videos = data.get("videos", [])
            
            for video in videos:
                video_id = str(video.get("id", ""))
                
                # Skip already used videos
                if video_id in self._used_video_ids:
                    continue
                
                # Get video duration
                video_duration = float(video.get("duration", 0))
                if video_duration < min_duration:
                    continue
                
                # Get best quality video file
                video_files = video.get("video_files", [])
                if not video_files:
                    continue
                
                # Sort by quality (prefer HD)
                video_files.sort(key=lambda x: x.get("height", 0), reverse=True)
                best_file = video_files[0]
                
                video_url = best_file.get("link")
                if not video_url:
                    continue
                
                # Check cache or download
                cached_path = self._clip_cache.get(video_url)
                if not cached_path:
                    # Download video
                    local_path = str(self.temp_dir / f"clip_{video_id}.mp4")
                    local_path = sanitize_path(local_path)
                    
                    try:
                        logger.debug(f"Downloading clip: {video_url}")
                        video_response = self._http.get(video_url, timeout=30)
                        video_response.raise_for_status()
                        
                        with open(local_path, "wb") as f:
                            f.write(video_response.content)
                        
                        self._clip_cache.put(video_url, local_path)
                        cached_path = local_path
                        
                    except Exception as exc:
                        logger.warning(f"Failed to download clip: {exc}")
                        continue
                
                # Create candidate
                candidate = ClipCandidate(
                    pexels_id=video_id,
                    path=cached_path,
                    duration=video_duration,
                    url=video_url
                )
                
                candidates.append(candidate)
            
            # Cache results
            self._video_candidates[query] = candidates
            
            logger.info(f"Found {len(candidates)} video candidates for '{query}'")
            
        except Exception as exc:
            logger.error(f"Pexels search failed for '{query}': {exc}")
            logger.debug("", exc_info=True)
        
        return candidates

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
        """
        ✅ FIXED: Render captions on video.
        Changed from wrong parameter names (output_path, word_timings, total_duration)
        to correct ones that CaptionRenderer.render() actually accepts.
        """
        if not getattr(settings, "KARAOKE_CAPTIONS", True):
            return video_path

        try:
            # ✅ FIX: caption_renderer.render() already returns the output path
            # No need to pre-create output path or pass it as parameter
            output = self.caption_renderer.render(
                video_path=video_path,
                text=text,
                words=words,  # ✅ CORRECT: 'words' not 'word_timings'
                duration=duration,  # ✅ CORRECT: 'duration' not 'total_duration'  
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

    def _novelty_add_used_safe(self, title: str, script: List[str]):
        """Mark content as used."""
        try:
            self.novelty_guard.add_used(
                title=title,
                script_text=" ".join(script),
                channel=self.channel_id
            )
        except Exception as exc:
            logger.warning("Failed to mark content as used: %s", exc)

    def _save_script_state_safe(self, script: Dict):
        """Save script to state."""
        try:
            self.state_guard.save_successful_script(script)
        except Exception as exc:
            logger.warning("Failed to save script state: %s", exc)
