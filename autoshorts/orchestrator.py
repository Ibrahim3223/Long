# FILE: autoshorts/orchestrator.py
# -*- coding: utf-8 -*-
"""
High level orchestration for generating complete videos (OPTIMIZED + SMART PEXELS).
✅ Akıllı Pexels video seçimi
✅ Duplicate önleme
✅ Sahneye özel keywords
✅ Performans optimize
"""
from __future__ import annotations

import hashlib
import logging
import os
import pathlib
import random
import shutil
import time
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
# ✅ SMART PEXELS HELPER FUNCTIONS (inline implementation)
# ============================================================================

def extract_scene_keywords(text: str, max_keywords: int = 3) -> List[str]:
    """Extract key visual keywords from scene text."""
    if not text:
        return []
    
    # Common stop words to filter out
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were', 'been', 'be',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
    }
    
    # Split and clean
    words = text.lower().split()
    keywords = [w.strip('.,!?;:()[]{}"\'-') for w in words 
                if len(w) > 3 and w.lower() not in stop_words]
    
    # Return unique keywords, limited to max_keywords
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
    """
    Build an intelligent Pexels search query from scene context.
    
    Priority:
    1. Search queries (if provided)
    2. Chapter title + scene keywords
    3. Scene keywords only
    4. Fallback terms
    """
    # 1. If explicit search queries provided, use them
    if search_queries and len(search_queries) > 0:
        # Pick a random query from the list for variety
        return random.choice(search_queries).strip()
    
    # 2. Extract keywords from scene text
    scene_keywords = extract_scene_keywords(scene_text, max_keywords=2)
    
    # 3. If we have chapter title, combine with scene keywords
    if chapter_title:
        chapter_keywords = extract_scene_keywords(chapter_title, max_keywords=1)
        if chapter_keywords:
            all_keywords = chapter_keywords + scene_keywords[:1]
            return " ".join(all_keywords[:2])
    
    # 4. Use scene keywords
    if scene_keywords:
        return " ".join(scene_keywords[:2])
    
    # 5. Fallback to provided terms or generic
    if fallback_terms and len(fallback_terms) > 0:
        return random.choice(fallback_terms)
    
    return "nature landscape"


def enhance_pexels_selection(
    pexels_client: PexelsClient,
    query: str,
    duration: float,
    used_urls: Set[str],
    max_attempts: int = 3
) -> Tuple[Optional[str], Optional[str]]:
    """
    Enhanced Pexels video selection with duplicate avoidance.
    
    Returns:
        (video_url, video_id) or (None, None) if no suitable video found
    """
    for attempt in range(max_attempts):
        try:
            # Search with pagination
            page = attempt + 1
            results = pexels_client.search_videos(
                query=query,
                per_page=15,
                page=page
            )
            
            if not results or 'videos' not in results:
                continue
            
            # Filter and sort videos
            candidates = []
            for video in results['videos']:
                video_id = str(video.get('id', ''))
                
                # Skip if already used
                if video_id in used_urls:
                    continue
                
                # Get video files
                video_files = video.get('video_files', [])
                if not video_files:
                    continue
                
                # Find suitable quality (HD preferred)
                suitable_file = None
                for vf in video_files:
                    if vf.get('width', 0) >= 1280 and vf.get('height', 0) >= 720:
                        suitable_file = vf
                        break
                
                # Fallback to any available file
                if not suitable_file and video_files:
                    suitable_file = video_files[0]
                
                if suitable_file and suitable_file.get('link'):
                    video_duration = video.get('duration', 0)
                    # Prefer videos longer than needed duration
                    quality_score = suitable_file.get('width', 0) * suitable_file.get('height', 0)
                    candidates.append({
                        'url': suitable_file['link'],
                        'id': video_id,
                        'duration': video_duration,
                        'quality': quality_score
                    })
            
            # Sort by quality and duration match
            if candidates:
                # Prioritize videos that are slightly longer than needed
                candidates.sort(
                    key=lambda x: (
                        abs(x['duration'] - duration * 1.2),  # Prefer 20% longer
                        -x['quality']  # Higher quality better
                    )
                )
                
                best = candidates[0]
                return best['url'], best['id']
        
        except Exception as exc:
            logger.debug(f"Pexels search attempt {attempt + 1} failed: {exc}")
            continue
    
    return None, None


# ============================================================================
# Original orchestrator code continues below
# ============================================================================

@dataclass
class ClipCandidate:
    """Carry the essential data for a downloaded Pexels clip."""
    url: str
    duration: float
    video_id: str


class _ClipCache:
    """Manage short lived and persistent clip caches."""

    def __init__(self, temp_dir: pathlib.Path) -> None:
        self._runtime_cache: Dict[str, pathlib.Path] = {}
        self._cache_dir = temp_dir / "clip-cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        shared_root = pathlib.Path(
            os.getenv("SHARED_CLIP_CACHE_DIR", os.path.join(".state", "clip_cache"))
        )
        shared_root.mkdir(parents=True, exist_ok=True)
        self._shared_root = shared_root
        self._shared_limit = int(os.getenv("PERSISTENT_CLIP_CACHE_LIMIT", "200"))

    def cache_paths(self, url: str) -> Tuple[pathlib.Path, pathlib.Path]:
        digest = hashlib.sha1(url.encode("utf-8")).hexdigest()
        runtime = self._cache_dir / f"{digest}.mp4"
        shared = self._shared_root / f"{digest}.mp4"
        return runtime, shared

    def try_copy(self, url: str, destination: pathlib.Path) -> bool:
        cached = self._runtime_cache.get(url)
        if cached and cached.exists():
            shutil.copy2(cached, destination)
            return True

        runtime, shared = self.cache_paths(url)
        for path in (runtime, shared):
            if path.exists():
                shutil.copy2(path, destination)
                self._runtime_cache[url] = path
                return True
        return False

    def store(self, url: str, source: pathlib.Path) -> None:
        runtime, shared = self.cache_paths(url)
        try:
            if not runtime.exists():
                shutil.copy2(source, runtime)
            self._runtime_cache[url] = runtime
        except Exception as exc:
            logger.debug("Unable to store runtime cache for %s: %s", url, exc)

        try:
            if not shared.exists():
                shutil.copy2(source, shared)
            self._enforce_shared_budget()
        except Exception as exc:
            logger.debug("Unable to store shared cache for %s: %s", url, exc)

    def _enforce_shared_budget(self) -> None:
        if self._shared_limit <= 0:
            return
        entries = sorted(
            (
                (p.stat().st_mtime, p)
                for p in self._shared_root.glob("*.mp4")
                if p.is_file()
            ),
            key=lambda item: item[0],
        )
        overflow = len(entries) - self._shared_limit
        for _, path in entries[: max(0, overflow)]:
            try:
                path.unlink()
            except Exception:
                logger.debug("Unable to evict cached clip: %s", path)


class ShortsOrchestrator:
    """Coordinate script, audio, and video generation into a final render."""

    def __init__(
        self,
        channel_id: str,
        temp_dir: str,
        api_key: Optional[str] = None,
        pexels_key: Optional[str] = None,
    ) -> None:
        self.channel_id = channel_id
        self.temp_dir = pathlib.Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # External services / helpers
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
        
        # ✅ SMART PEXELS: Track used video IDs to avoid duplicates
        self._used_video_ids: Set[str] = set()

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

        logger.info("🎬 ShortsOrchestrator ready for channel %s (FAST_MODE=%s)", channel_id, self.fast_mode)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def produce_video(
        self, topic_prompt: str, max_retries: int = 3
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """Produce a full video for *topic_prompt*."""
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
        """Generate script using GeminiClient and convert to expected format."""
        # Get target duration from settings (default to 5 minutes for long-form)
        target_duration = int(os.getenv("TARGET_DURATION_SECONDS", "300"))  # 5 minutes default
        style = getattr(settings, "CONTENT_STYLE", "educational and engaging")
        
        for attempt in range(3):
            try:
                logger.info("Generating script (attempt %d)", attempt + 1)
                
                # Call GeminiClient.generate() with proper parameters
                content_response = self.gemini.generate(
                    topic=topic_prompt,
                    style=style,
                    duration=target_duration,
                    additional_context=f"Channel: {self.channel_id}"
                )
                
                if not content_response:
                    continue
                
                # Convert ContentResponse to expected dict format
                script = self._convert_content_response_to_script(content_response)
                
                if not script:
                    continue

                # Score script quality using correct method
                sentences_text = [s.get("text", "") for s in script.get("sentences", [])]
                score_dict = self.quality_scorer.score(sentences_text, title=script.get("title", ""))
                score = score_dict.get("overall", 0) * 10  # Convert 0-10 to 0-100
                
                if score < 50:
                    logger.warning("Low quality script (score=%d), retrying", score)
                    continue

                logger.info("✅ Script generated (score=%d)", score)
                return script

            except Exception as exc:
                logger.warning("Script generation attempt %d failed: %s", attempt + 1, exc)
                if attempt < 2:
                    time.sleep(2)

        return None
    
    def _convert_content_response_to_script(self, content_response) -> Optional[Dict]:
        """Convert GeminiClient ContentResponse to orchestrator script format."""
        try:
            # Build sentences list from script (list of strings -> list of dicts)
            sentences = []
            
            # Add hook as first sentence
            if content_response.hook:
                sentences.append({
                    "text": content_response.hook,
                    "type": "hook"
                })
            
            # Add main script sentences
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
            
            # Add CTA as last sentence
            if content_response.cta:
                sentences.append({
                    "text": content_response.cta,
                    "type": "cta"
                })
            
            # Process chapters - fix field names and add search_queries
            chapters = []
            search_queries_list = content_response.search_queries or []
            num_chapters = len(content_response.chapters) if content_response.chapters else 0
            
            for idx, chapter in enumerate(content_response.chapters or []):
                # Convert field names to match orchestrator expectations
                chapter_dict = {
                    "title": chapter.get("title", f"Chapter {idx + 1}"),
                    "start_sentence_index": chapter.get("start_sentence", 0),
                    "end_sentence_index": chapter.get("end_sentence", len(sentences) - 1),
                    "description": chapter.get("description", "")
                }
                
                # Distribute search_queries across chapters
                if search_queries_list and num_chapters > 0:
                    queries_per_chapter = len(search_queries_list) // num_chapters
                    start_query_idx = idx * queries_per_chapter
                    end_query_idx = start_query_idx + queries_per_chapter
                    
                    # Last chapter gets remaining queries
                    if idx == num_chapters - 1:
                        end_query_idx = len(search_queries_list)
                    
                    chapter_dict["search_queries"] = search_queries_list[start_query_idx:end_query_idx]
                else:
                    chapter_dict["search_queries"] = []
                
                chapters.append(chapter_dict)
            
            # Build script dict with all required fields
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
        results = []
        for idx, sent in enumerate(sentences):
            text = sent.get("text", "")
            if not text.strip():
                results.append(None)
                continue

            out_path = str(self.temp_dir / f"tts_{idx:03d}.mp3")
            try:
                self.tts.synthesize(text, out_path)
                words = self.tts.get_word_timings() or []
                results.append((out_path, words))
            except Exception as exc:
                logger.error("TTS failed for sentence %d: %s", idx, exc)
                results.append(None)

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

        # Get chapters and search queries if available
        chapters = script.get("chapters", [])
        
        logger.info("🎥 Rendering %d scenes", len(sentences))
        scene_paths = []

        for idx, (sent, tts_result) in enumerate(zip(sentences, tts_results)):
            if not tts_result:
                logger.warning("Skipping scene %d (no TTS)", idx)
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
                keywords=extract_keywords(text),
                duration=audio_dur,
                index=idx,
                sentence_type=sentence_type,
                search_queries=search_queries,
                chapter_title=current_chapter,
            )

            if not scene_path:
                logger.warning("Skipping scene %d (no video clip)", idx)
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

        if not scene_paths:
            logger.error("No scenes rendered successfully")
            return None

        # Concatenate all scenes
        concat_out = str(self.temp_dir / "concat.mp4")
        logger.info("🔗 Concatenating %d scenes", len(scene_paths))
        self._concat_segments(scene_paths, concat_out)

        if not os.path.exists(concat_out):
            logger.error("Concatenation failed")
            return None

        # Add background music
        logger.info("🎵 Adding background music")
        final_video = self._maybe_add_bgm(concat_out, total_audio_duration)

        return final_video or concat_out

    # ------------------------------------------------------------------
    # Pexels / Video selection
    # ------------------------------------------------------------------

    def _prepare_scene_clip(
        self,
        text: str,
        keywords: Sequence[str],
        duration: float,
        index: int,
        sentence_type: str,
        search_queries: Optional[List[str]] = None,
        chapter_title: Optional[str] = None,
    ) -> Optional[str]:
        """
        ✅ SMART PEXELS: Prepare clip with intelligent video selection
        """
        logger.info("Selecting clip for scene %s", index)
        
        # ✅ Build smart Pexels query
        fallback_terms = getattr(settings, 'SEARCH_TERMS', None)
        query = build_smart_pexels_query(
            scene_text=text,
            chapter_title=chapter_title,
            search_queries=search_queries,
            fallback_terms=fallback_terms
        )
        
        logger.info("       🔍 Searching: '%s'", query)
        
        # ✅ Enhanced Pexels selection
        video_url, video_id = enhance_pexels_selection(
            pexels_client=self.pexels,
            query=query,
            duration=max(duration, 5.0),
            used_urls=self._used_video_ids,
            max_attempts=3
        )
        
        if not video_url:
            logger.warning("       ⚠️ No suitable video found, using fallback")
            # Fallback to old method
            candidate = self._next_candidate_fallback(text, keywords)
            if not candidate:
                logger.error("No clip candidate found for scene %s", index)
                return None
            video_url = candidate.url
            video_id = candidate.video_id
        
        # ✅ Track used video
        if video_id:
            self._used_video_ids.add(video_id)
            logger.info("       ✅ Selected video ID: %s", video_id)

        raw_clip = self.temp_dir / f"raw_{index:03d}.mp4"

        if self._clip_cache.try_copy(video_url, raw_clip):
            logger.info("       ♻️ Using cached clip")
        else:
            logger.info("       📥 Downloading...")
            try:
                resp = self._http.get(video_url, timeout=60, stream=True)
                resp.raise_for_status()
                with open(raw_clip, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1048576):
                        fh.write(chunk)
                self._clip_cache.store(video_url, raw_clip)
            except Exception as exc:
                logger.error("Download failed: %s", exc)
                return None

        processed = self.temp_dir / f"scene_{index:03d}_video.mp4"
        self._process_clip(raw_clip, processed, duration)

        return str(processed) if processed.exists() else None

    def _next_candidate_fallback(
        self, text: str, keywords: Sequence[str]
    ) -> Optional[ClipCandidate]:
        """Fallback method for finding video clips."""
        simplified = simplify_query(text) or (keywords[0] if keywords else "nature")
        
        if simplified not in self._video_candidates:
            try:
                results = self.pexels.search_videos(simplified, per_page=10)
                candidates = []
                for video in results.get("videos", []):
                    video_id = str(video.get("id", ""))
                    video_files = video.get("video_files", [])
                    if not video_files:
                        continue
                    
                    # Find HD quality
                    selected_file = None
                    for vf in video_files:
                        if vf.get("width", 0) >= 1280:
                            selected_file = vf
                            break
                    if not selected_file:
                        selected_file = video_files[0]
                    
                    if selected_file.get("link"):
                        candidates.append(
                            ClipCandidate(
                                url=selected_file["link"],
                                duration=video.get("duration", 15),
                                video_id=video_id,
                            )
                        )
                
                self._video_candidates[simplified] = candidates
            except Exception as exc:
                logger.error("Pexels search failed for '%s': %s", simplified, exc)
                return None

        pool = self._video_candidates.get(simplified, [])
        if not pool:
            return None

        # Filter out already used videos
        available = [c for c in pool if c.video_id not in self._used_video_ids]
        if not available:
            available = pool  # If all used, reuse

        return random.choice(available) if available else None

    def _process_clip(
        self, source: pathlib.Path, output: pathlib.Path, target_duration: float
    ) -> None:
        """Process raw clip: resize, crop, loop, etc."""
        clip_dur = ffprobe_duration(str(source))
        if clip_dur <= 0:
            clip_dur = target_duration

        loops = max(1, int((target_duration / clip_dur) + 0.999))

        # Build video filters
        target_w = int(getattr(settings, "TARGET_WIDTH", 1080))
        target_h = int(getattr(settings, "TARGET_HEIGHT", 1920))
        
        filters = [
            f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase",
            f"crop={target_w}:{target_h}",
        ]

        crf = str(getattr(settings, "CRF_VISUAL", 22))
        fps = int(getattr(settings, "TARGET_FPS", 30))

        input_opts = []
        if loops > 1:
            input_opts = ["-stream_loop", str(loops - 1)]

        run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                *input_opts,
                "-i",
                str(source),
                "-vf",
                ",".join(filters),
                "-r",
                str(fps),
                "-vsync",
                "cfr",
                "-c:v",
                "libx264",
                "-preset",
                self.ffmpeg_preset,
                "-crf",
                crf,
                "-pix_fmt",
                "yuv420p",
                "-an",
                "-movflags",
                "+faststart",
                str(output),
            ]
        )

    def _render_captions(
        self,
        video_path: str,
        text: str,
        words: Sequence[Tuple[str, float]],
        duration: float,
        sentence_type: str,
    ) -> str:
        logger.info("Rendering captions")
        try:
            return self.caption_renderer.render(
                video_path=video_path,
                text=text,
                words=list(words),
                duration=duration,
                is_hook=(sentence_type == "hook"),
                sentence_type=sentence_type,
                temp_dir=str(self.temp_dir),
            )
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
        output = self.temp_dir / f"scene_{index:03d}_final.mp4"
        try:
            run(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    video_path,
                    "-i",
                    audio_path,
                    "-t",
                    f"{duration:.3f}",
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "160k",
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-shortest",
                    "-movflags",
                    "+faststart",
                    str(output),
                ]
            )
        except Exception as exc:
            logger.error("Audio mux failed: %s", exc)
            logger.debug("", exc_info=True)
            return None

        return str(output) if output.exists() else None

    def _concat_segments(self, segments: Iterable[str], output: str) -> None:
        concat_file = pathlib.Path(output).with_suffix(".txt")
        try:
            with open(concat_file, "w", encoding="utf-8") as handle:
                for segment in segments:
                    handle.write(f"file '{segment}'\n")

            # First try stream-copy concat (very fast)
            try:
                run(
                    [
                        "ffmpeg",
                        "-y",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-f",
                        "concat",
                        "-safe",
                        "0",
                        "-i",
                        str(concat_file),
                        "-c",
                        "copy",
                        "-movflags",
                        "+faststart",
                        output,
                    ]
                )
                if os.path.exists(output):
                    return
            except Exception:
                logger.debug("Concat copy failed, falling back to re-encode")

            # Fallback: re-encode concat
            crf = str(getattr(settings, "CRF_VISUAL", 22))
            fps = int(getattr(settings, "TARGET_FPS", 30))
            run(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(concat_file),
                    "-c:v",
                    "libx264",
                    "-preset",
                    self.ffmpeg_preset,
                    "-crf",
                    crf,
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "160k",
                    "-r",
                    str(fps),
                    "-vsync",
                    "cfr",
                    "-movflags",
                    "+faststart",
                    output,
                ]
            )
        finally:
            concat_file.unlink(missing_ok=True)

    # ----------------------- Helpers (robustness) -----------------------

    def _maybe_add_bgm(self, final_path: str, total_duration: float) -> Optional[str]:
        candidates = [
            ("add_bgm_to_video", (final_path, total_duration, str(self.temp_dir))),
            ("add_to_video", (final_path, total_duration)),
            ("apply_to_video", (final_path, total_duration)),
            ("mix_to_video", (final_path, total_duration, str(self.temp_dir))),
        ]
        for method_name, args in candidates:
            method = getattr(self.bgm_manager, method_name, None)
            if callable(method):
                try:
                    out = method(*args)
                    if out and isinstance(out, str) and os.path.exists(out):
                        return out
                except Exception as exc:
                    logger.debug("BGMManager.%s failed: %s", method_name, exc)
        logger.info("BGM step skipped (no compatible method)")
        return None

    def _save_script_state_safe(self, script: Dict) -> None:
        for name in ["save_successful_script", "save_script", "record_script", "save"]:
            method = getattr(self.state_guard, name, None)
            if callable(method):
                try:
                    method(script)
                    return
                except Exception as exc:
                    logger.debug("StateGuard.%s failed: %s", name, exc)
        logger.debug("No StateGuard save method available")

    def _check_novelty_safe(self, title: str, script: List[str]) -> Tuple[bool, float]:
        try:
            return self.novelty_guard.check_novelty(title=title, script=script)
        except Exception as exc:
            logger.debug("novelty_guard.check_novelty failed: %s", exc)
            return True, 0.0

    def _novelty_add_used_safe(self, title: str, script: List[str]) -> None:
        for name in ["add_used_script", "add_script", "add"]:
            method = getattr(self.novelty_guard, name, None)
            if callable(method):
                try:
                    method(title=title, script=script)
                    return
                except Exception as exc:
                    logger.debug("novelty_guard.%s failed: %s", name, exc)
        logger.debug("No novelty guard add method available")
