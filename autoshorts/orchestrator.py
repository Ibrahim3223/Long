# FILE: autoshorts/orchestrator.py
# -*- coding: utf-8 -*-
"""
High level orchestration for generating complete videos.
✅ FULL FIXED VERSION v3 - Ready to use!
✅ Path sanitization
✅ Parallel TTS support
✅ get_word_timings() fix
✅ extract_keywords() lang parameter fix
✅ Pexels rate limiting fix (429 error)
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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Set

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
    Example: "/tmp/autoshorts_Object Origins/file.mp3" -> "/tmp/autoshorts_Object_Origins/file.mp3"
    """
    from pathlib import Path
    p = Path(path)
    
    safe_parts = []
    for part in p.parts:
        if part == '/' or part == p.parts[0]:  # Keep root
            safe_parts.append(part)
        else:
            # Replace problematic characters
            safe_part = re.sub(r'[^\w\-_\.]', '_', part)
            safe_part = re.sub(r'_+', '_', safe_part)  # Remove multiple underscores
            safe_parts.append(safe_part)
    
    return str(Path(*safe_parts))


# ============================================================================
# SMART PEXELS HELPER FUNCTIONS
# ============================================================================

def extract_scene_keywords(text: str, max_keywords: int = 3) -> List[str]:
    """Extract key visual keywords from scene text."""
    if not text:
        return []
    
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were', 'been', 'be',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
    }
    
    words = text.lower().split()
    keywords = [w.strip('.,!?;:()[]{}"\'-') for w in words 
                if len(w) > 3 and w.lower() not in stop_words]
    
    seen = set()
    result = []
    for kw in keywords:
        if kw not in seen and len(result) < max_keywords:
            seen.add(kw)
            result.append(kw)
    
    return result


def build_smart_pexels_query(
    scene_text: str,
    chapter_title: Optional[str] = None,
    search_queries: Optional[List[str]] = None,
    fallback_terms: Optional[List[str]] = None
) -> str:
    """Build an intelligent Pexels search query from scene context."""
    if search_queries and len(search_queries) > 0:
        return random.choice(search_queries)
    
    scene_keywords = extract_scene_keywords(scene_text)
    
    if chapter_title and scene_keywords:
        chapter_kw = extract_scene_keywords(chapter_title, max_keywords=1)
        if chapter_kw:
            combined = chapter_kw + scene_keywords[:1]
            return " ".join(combined)
    
    if scene_keywords:
        return " ".join(scene_keywords[:2])
    
    if fallback_terms and len(fallback_terms) > 0:
        return random.choice(fallback_terms)
    
    return "abstract background"  # Generic fallback


def enhance_pexels_selection(
    pexels_client: PexelsClient,
    query: str,
    duration: float,
    used_urls: Set[str],
    rate_limiter: RateLimiter,
    max_attempts: int = 2  # ✅ Reduced from 3 to 2
) -> Tuple[Optional[str], Optional[str]]:
    """
    Enhanced Pexels video selection with duplicate avoidance and rate limiting.
    ✅ Now with exponential backoff for 429 errors.
    """
    for attempt in range(max_attempts):
        try:
            # ✅ Rate limiting - wait before API call
            rate_limiter.wait()
            
            page = attempt + 1
            results = pexels_client.search_videos(
                query=query,
                per_page=15,
                page=page
            )
            
            if not results or 'videos' not in results:
                continue
            
            candidates = []
            for video in results['videos']:
                video_id = str(video.get('id', ''))
                
                if video_id in used_urls:
                    continue
                
                video_files = video.get('video_files', [])
                if not video_files:
                    continue
                
                suitable_file = None
                for vf in video_files:
                    if vf.get('width', 0) >= 1280 and vf.get('height', 0) >= 720:
                        suitable_file = vf
                        break
                
                if not suitable_file and video_files:
                    suitable_file = video_files[0]
                
                if suitable_file and suitable_file.get('link'):
                    video_duration = video.get('duration', 0)
                    quality_score = suitable_file.get('width', 0) * suitable_file.get('height', 0)
                    candidates.append({
                        'url': suitable_file['link'],
                        'id': video_id,
                        'duration': video_duration,
                        'quality': quality_score
                    })
            
            if candidates:
                candidates.sort(
                    key=lambda x: (
                        abs(x['duration'] - duration * 1.2),
                        -x['quality']
                    )
                )
                
                best = candidates[0]
                return best['url'], best['id']
        
        except requests.exceptions.HTTPError as e:
            # ✅ Handle 429 Rate Limiting with exponential backoff
            if e.response.status_code == 429:
                retry_after = int(e.response.headers.get('Retry-After', 5))
                backoff_time = min(retry_after, 2 ** (attempt + 1))
                logger.warning(f"⏳ Rate limited (429), waiting {backoff_time}s before retry...")
                time.sleep(backoff_time)
                continue
            else:
                logger.debug(f"Pexels HTTP error: {e}")
                continue
        
        except Exception as exc:
            logger.debug(f"Pexels search attempt {attempt + 1} failed: {exc}")
            continue
    
    return None, None


@dataclass
class ClipCandidate:
    """Carry the essential data for a downloaded Pexels clip."""
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
            content_response = self.gemini.generate(
                topic=topic_prompt,
                style=getattr(settings, "CHANNEL_MODE", "educational"),
                duration=getattr(settings, "TARGET_DURATION", 180),
                additional_context=getattr(settings, "GEMINI_PROMPT", None)
            )

            sentences = []
            
            if content_response.hook:
                sentences.append({
                    "text": content_response.hook,
                    "type": "hook"
                })
            
            for idx, text in enumerate(content_response.script):
                sentence_type = "content"
                if idx == 0 and not content_response.hook:
                    sentence_type = "hook"
                elif idx == len(content_response.script) - 1:
                    sentence_type = "conclusion"
                
                sentences.append({
                    "text": text,
                    "type": sentence_type
                })
            
            if content_response.cta:
                sentences.append({
                    "text": content_response.cta,
                    "type": "cta"
                })
            
            chapters = []
            search_queries_list = content_response.search_queries or []
            num_chapters = len(content_response.chapters) if content_response.chapters else 0
            
            for idx, chapter in enumerate(content_response.chapters or []):
                chapter_dict = {
                    "title": chapter.get("title", f"Chapter {idx + 1}"),
                    "start_sentence_index": chapter.get("start_sentence", 0),
                    "end_sentence_index": chapter.get("end_sentence", len(sentences) - 1),
                    "description": chapter.get("description", "")
                }
                
                if search_queries_list and num_chapters > 0:
                    queries_per_chapter = len(search_queries_list) // num_chapters
                    start_query_idx = idx * queries_per_chapter
                    end_query_idx = start_query_idx + queries_per_chapter
                    
                    if idx == num_chapters - 1:
                        end_query_idx = len(search_queries_list)
                    
                    chapter_dict["search_queries"] = search_queries_list[start_query_idx:end_query_idx]
                else:
                    chapter_dict["search_queries"] = []
                
                chapters.append(chapter_dict)
            
            script = {
                "sentences": sentences,
                "title": content_response.metadata.get("title", ""),
                "description": content_response.metadata.get("description", ""),
                "tags": content_response.metadata.get("tags", []),
                "chapters": chapters,
                "search_queries": search_queries_list,
                "main_visual_focus": content_response.main_visual_focus or "",
            }
            
            return script
            
        except Exception as exc:
            logger.error(f"Failed to convert ContentResponse: {exc}")
            return None

    def _generate_all_tts(self, sentences: List[Dict]) -> List[Optional[Tuple[str, List[Tuple[str, float]]]]]:
        """
        ✅ FIXED: Generate TTS for all sentences.
        Now properly handles synthesize() return value (duration, words) tuple.
        """
        # Use parallel for long videos
        if len(sentences) > 20:
            logger.info("🚀 Using parallel TTS generation (%d sentences)", len(sentences))
            return self._generate_all_tts_parallel(sentences)
        else:
            return self._generate_all_tts_sequential(sentences)
    
    def _generate_all_tts_sequential(
        self, 
        sentences: List[Dict]
    ) -> List[Optional[Tuple[str, List[Tuple[str, float]]]]]:
        """Sequential TTS generation."""
        results = []
        
        for idx, sent in enumerate(sentences):
            text = sent.get("text", "")
            
            if not text.strip():
                results.append(None)
                continue
            
            # ✅ Generate safe output path
            out_path = str(self.temp_dir / f"tts_{idx:03d}.wav")
            out_path = sanitize_path(out_path)
            
            try:
                # ✅ FIX: synthesize() returns (duration, words) tuple
                duration, words = self.tts.synthesize(text, out_path)
                results.append((out_path, words))
                
            except Exception as exc:
                logger.error("TTS failed for sentence %d: %s", idx, exc)
                logger.debug("", exc_info=True)
                results.append(None)
        
        return results
    
    def _generate_all_tts_parallel(
        self, 
        sentences: List[Dict]
    ) -> List[Optional[Tuple[str, List[Tuple[str, float]]]]]:
        """Parallel TTS generation for better performance."""
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

        return final_out if os.path.exists(final_out) else None

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
        # Build smart query
        query = build_smart_pexels_query(
            scene_text=text,
            chapter_title=chapter_title,
            search_queries=search_queries,
            fallback_terms=getattr(settings, "FALLBACK_SEARCH_TERMS", None)
        )
        
        logger.info("Scene %d: query='%s' (%.2fs)", index, query, duration)

        # ✅ Get video URL with rate limiting
        video_url, video_id = enhance_pexels_selection(
            self.pexels,
            query,
            duration,
            self._used_video_ids,
            self._pexels_rate_limiter  # Pass rate limiter
        )
        
        if not video_url:
            logger.warning("No suitable video found for query: %s", query)
            # ✅ Don't fail completely - just skip this scene
            return None
        
        if video_id:
            self._used_video_ids.add(video_id)

        # Check cache
        cached_path = self._clip_cache.get(video_url)
        if cached_path and os.path.exists(cached_path):
            logger.debug("Using cached clip")
            clip_path = cached_path
        else:
            # Download
            clip_path = str(self.temp_dir / f"clip_{index:03d}.mp4")
            try:
                response = self._http.get(video_url, timeout=30, stream=True)
                response.raise_for_status()
                with open(clip_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # Cache it
                self._clip_cache.put(video_url, clip_path)
            except Exception as exc:
                logger.error("Failed to download video: %s", exc)
                return None

        # Process clip
        processed = str(self.temp_dir / f"scene_{index:03d}_video.mp4")
        self._process_video_clip(clip_path, processed, duration)
        
        return processed if os.path.exists(processed) else None

    def _process_video_clip(self, input_path: str, output_path: str, target_duration: float):
        """Process video clip to match target duration and format."""
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
