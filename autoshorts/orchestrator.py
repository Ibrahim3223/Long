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

# ✅ SMART PEXELS IMPORT
from autoshorts.orchestrator_enhancements import (
    build_smart_pexels_query,
    enhance_pexels_selection,
    extract_scene_keywords
)

PEXELS_SEARCH_ENDPOINT = "https://api.pexels.com/videos/search"

logger = logging.getLogger(__name__)


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
                    "hook": script.get("hook", ""),
                    "script": script,
                }

                self._save_script_state_safe(script)
                self._novelty_add_used_safe(
                    title=script.get("title", ""), script=sentences_txt
                )

                logger.info("✅ Video ready: %s", video_path)
                return video_path, metadata

            except Exception as exc:
                logger.error("Attempt %s failed: %s", attempt, exc)
                logger.debug("", exc_info=True)

        logger.error("❌ All attempts failed")
        return None, None

    # ------------------------------------------------------------------
    # Script generation
    # ------------------------------------------------------------------

    def _generate_script(self, topic_prompt: str) -> Optional[Dict]:
        logger.info("🧠 Generating script via Gemini")
        try:
            response = self.gemini.generate(
                topic=topic_prompt,
                style="educational, informative, engaging",
                duration=getattr(settings, "TARGET_DURATION", 210),
                additional_context=None,
            )
        except Exception as exc:
            logger.error("Gemini error: %s", exc)
            logger.debug("", exc_info=True)
            return None

        script = {
            "title": response.metadata.get("title", ""),
            "description": response.metadata.get("description", ""),
            "tags": response.metadata.get("tags", []),
            "hook": response.hook,
            "sentences": [],
            "chapters": response.chapters,
            # ✅ SMART PEXELS: Store search_queries from Gemini
            "search_queries": response.search_queries,
        }

        script["sentences"].append(
            {
                "text": response.hook,
                "type": "hook",
                "visual_keywords": [response.main_visual_focus]
                if getattr(response, "main_visual_focus", None)
                else [],
            }
        )

        for idx, sentence in enumerate(response.script):
            if idx < len(response.search_queries):
                keywords = [response.search_queries[idx]]
            elif response.search_queries:
                keywords = [response.search_queries[idx % len(response.search_queries)]]
            else:
                keywords = []
            script["sentences"].append(
                {"text": sentence, "type": "buildup", "visual_keywords": keywords}
            )

        script["sentences"].append(
            {"text": response.cta, "type": "conclusion", "visual_keywords": []}
        )

        quality = self.quality_scorer.score(
            [s.get("text", "") for s in script["sentences"]],
            script.get("title", ""),
        )
        overall = quality.get("overall", 0.0)
        if overall < 4.0:
            logger.warning("Script quality too low (%.1f)", overall)
            return None

        logger.info("Script generated with score %.1f", overall)
        return script

    # ------------------------------------------------------------------
    # Rendering pipeline
    # ------------------------------------------------------------------

    def _render_from_script(self, script: Dict) -> Optional[str]:
        logger.info("🎬 Rendering scenes")
        sentences = script.get("sentences", [])
        if not sentences:
            logger.error("Script missing sentences")
            return None

        # ✅ SMART PEXELS: Get search queries and chapters
        search_queries = script.get("search_queries", [])
        chapters = script.get("chapters", [])

        rendered: List[str] = []
        total_duration = 0.0

        for index, sentence in enumerate(sentences, 1):
            logger.info("—" * 60)
            logger.info("Scene %s/%s", index, len(sentences))
            
            # ✅ SMART PEXELS: Determine chapter for this scene
            chapter_title = self._get_chapter_title_for_sentence(index - 1, chapters)
            
            scene_path = self._produce_scene(
                sentence, 
                index, 
                search_queries=search_queries,
                chapter_title=chapter_title
            )
            if not scene_path:
                logger.error("Scene %s failed", index)
                continue

            rendered.append(scene_path)
            # We already know target duration from TTS, but probe in case of trims
            scene_duration = ffprobe_duration(scene_path)
            total_duration += scene_duration
            logger.info("Scene %s duration %.2fs", index, scene_duration)

        if not rendered:
            logger.error("No scenes rendered")
            return None

        final_path = str(self.temp_dir / f"final_{int(time.time())}.mp4")
        try:
            self._concat_segments(rendered, final_path)
        except Exception as exc:
            logger.error("Concatenation failed: %s", exc)
            return None

        if not os.path.exists(final_path):
            logger.error("Final video missing after concat")
            return None

        if getattr(settings, "BGM_ENABLED", True):
            logger.info("Adding BGM to final video")
            with_bgm = self._maybe_add_bgm(final_path, total_duration)
            if with_bgm and os.path.exists(with_bgm):
                final_path = with_bgm

        logger.info("Final video assembled: %s", final_path)
        return final_path

    def _get_chapter_title_for_sentence(
        self, sentence_index: int, chapters: List[Dict]
    ) -> Optional[str]:
        """Get chapter title for a given sentence index."""
        for chapter in chapters:
            start = chapter.get("start_sentence", 0)
            end = chapter.get("end_sentence", 0)
            if start <= sentence_index <= end:
                return chapter.get("title")
        return None

    def _produce_scene(
        self, 
        sentence: Dict, 
        index: int,
        search_queries: Optional[List[str]] = None,
        chapter_title: Optional[str] = None
    ) -> Optional[str]:
        text = sentence.get("text", "").strip()
        sentence_type = sentence.get("type", "buildup")
        keywords = sentence.get("visual_keywords", []) or []

        if not text:
            logger.warning("Empty text for scene %s", index)
            return None

        logger.info("Text: %s", text[:120])
        audio_path, words, duration = self._generate_audio(text, index)
        if not audio_path:
            return None

        # ✅ SMART PEXELS: Pass search_queries and chapter_title
        clip_path = self._prepare_clip(
            text, 
            keywords, 
            duration, 
            index, 
            sentence_type,
            search_queries=search_queries,
            chapter_title=chapter_title
        )
        if not clip_path:
            return None

        captioned = self._render_captions(
            clip_path, text, words, duration, sentence_type
        )

        return self._mux_audio(captioned, audio_path, duration, index)

    def _generate_audio(
        self, text: str, index: int
    ) -> Tuple[Optional[str], List[Tuple[str, float]], float]:
        logger.info("Generating TTS for scene %s", index)
        target = self.temp_dir / f"scene_{index:03d}_voice.wav"
        try:
            duration, words = self.tts.synthesize(text=text, wav_out=str(target))
        except Exception as exc:
            logger.error("TTS failure: %s", exc)
            logger.debug("", exc_info=True)
            return None, [], 0.0

        if not target.exists():
            logger.error("TTS file missing for scene %s", index)
            return None, [], 0.0

        return str(target), words, duration

    def _prepare_clip(
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
        
        local_raw = self.temp_dir / f"scene_{index:03d}_raw.mp4"
        if not self._download_clip_from_url(video_url, local_raw):
            logger.error("Download failed for %s", video_url)
            return None

        processed = self.temp_dir / f"scene_{index:03d}_proc.mp4"
        try:
            self._process_clip(local_raw, processed, duration, sentence_type)
        except Exception as exc:
            logger.error("Processing failed: %s", exc)
            logger.debug("", exc_info=True)
            return None

        return str(processed)

    # ----------------------- Keyword utils (SAFE) -----------------------

    def _extract_keywords_safe(self, text: str) -> List[str]:
        lang = getattr(settings, "LANG", "en")
        try:
            return [kw for kw in extract_keywords(text, lang) if kw]
        except TypeError:
            try:
                return [kw for kw in extract_keywords(text) if kw]
            except Exception:
                return []
        except Exception:
            return []

    def _simplify_query_safe(self, text: str) -> str:
        lang = getattr(settings, "LANG", "en")
        try:
            return simplify_query(text, lang) or ""
        except TypeError:
            try:
                return simplify_query(text) or ""
            except Exception:
                return ""
        except Exception:
            return ""

    def _choose_keyword(self, text: str, keywords: Sequence[str]) -> str:
        pool: List[str] = []
        pool.extend([kw for kw in keywords if kw])
        pool.extend(self._extract_keywords_safe(text))

        simplified = self._simplify_query_safe(text)
        if simplified:
            pool.append(simplified)

        pool.append(getattr(settings, "CHANNEL_TOPIC", "interesting facts"))
        pool.append("interesting landscape")

        choice = next((kw for kw in pool if kw), text[:40] or "interesting")
        logger.info("Search keyword: %s", choice)
        return choice

    # ----------------------- Candidate selection (FALLBACK) -----------------------

    def _next_candidate_fallback(
        self,
        text: str,
        fallback_keywords: Sequence[str],
    ) -> Optional[ClipCandidate]:
        """Fallback method for video selection (old approach)"""
        queries = list(fallback_keywords)
        
        simplified = self._simplify_query_safe(text)
        if simplified and simplified not in queries:
            queries.append(simplified)

        queries.append(getattr(settings, "CHANNEL_TOPIC", "nature"))
        queries.append("dynamic landscape video")

        for query in queries:
            pool = self._get_candidates(query)
            if not pool:
                continue
            candidate = next(
                (clip for clip in pool if clip.duration >= getattr(settings, "PEXELS_MIN_DURATION", 3.0)),
                None,
            )
            if not candidate:
                continue
            pool.remove(candidate)
            logger.info("Using clip %s for query '%s'", candidate.url, query)
            return candidate

        logger.warning("No clip candidate located")
        return None

    def _get_candidates(self, query: str) -> List[ClipCandidate]:
        cache = self._video_candidates.get(query)
        if cache is not None:
            return cache

        results: List[ClipCandidate] = []
        try:
            per_page_default = getattr(settings, "PEXELS_PER_PAGE", 80)
            per_page = min(per_page_default, (50 if self.fast_mode else 80))
            params = {
                "query": query,
                "per_page": per_page,
                "orientation": "landscape",
            }
            response = self._http.get(PEXELS_SEARCH_ENDPOINT, params=params, timeout=(8 if self.fast_mode else 12))
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.error("Pexels lookup failed for '%s': %s", query, exc)
            logger.debug("", exc_info=True)
            payload = {}

        for video in payload.get("videos", []) or []:
            video_id = str(video.get("id", ""))
            duration = float(video.get("duration", 0.0))
            url = self._select_video_file(video)
            if not url:
                continue
            results.append(ClipCandidate(url=url, duration=duration, video_id=video_id))

        random.shuffle(results)
        self._video_candidates[query] = results
        return results

    def _select_video_file(self, video: Dict) -> Optional[str]:
        files = video.get("video_files", []) or []
        landscape = [fd for fd in files if fd.get("link") and int(fd.get("width", 0)) > int(fd.get("height", 0))]
        if not landscape:
            return None

        # Prefer 'sd' in FAST_MODE to download smaller files; otherwise prefer 'hd'
        preferred_quality = "sd" if self.fast_mode else "hd"
        preferred = [f for f in landscape if (f.get("quality") or "").lower() == preferred_quality]
        pool = preferred or landscape

        # Sort by width ascending (smaller first) to speed downloads
        try:
            pool.sort(key=lambda f: int(f.get("width", 0)))
        except Exception:
            pass

        # Shuffle lightly to avoid overusing the same file
        random.shuffle(pool)
        return pool[0].get("link")

    def _download_clip_from_url(self, url: str, destination: pathlib.Path) -> bool:
        """Download clip from URL with caching"""
        if self._clip_cache.try_copy(url, destination):
            logger.info("       ✅ Reused cached clip")
            return True

        try:
            with self._http.get(url, stream=True, timeout=45) as response:
                response.raise_for_status()
                with open(destination, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=64 * 1024):
                        if chunk:
                            handle.write(chunk)
        except Exception as exc:
            logger.error("Clip download failed: %s", exc)
            logger.debug("", exc_info=True)
            return False

        if destination.exists():
            self._clip_cache.store(url, destination)
            return True
        return False

    def _process_clip(
        self,
        source: pathlib.Path,
        output: pathlib.Path,
        target_duration: float,
        sentence_type: str,
    ) -> None:
        duration = ffprobe_duration(str(source))
        if duration <= 0:
            raise RuntimeError("invalid clip duration")

        loops = max(1, int(target_duration // duration) + 1)

        # Build filter chain (lighter zoompan), no 'loop' filter (use -stream_loop instead)
        filters: List[str] = [
            "scale=1920:1080:force_original_aspect_ratio=increase",
            "crop=1920:1080",
        ]

        fps = int(getattr(settings, "TARGET_FPS", 30))
        if sentence_type == "hook":
            filters.append(f"zoompan=z='min(zoom+0.0005,1.10)':d=1:s=1920x1080:fps={fps}")
        else:
            filters.append(f"fps={fps}")

        total_frames = max(2, int(round(target_duration * fps)))
        filters.extend([f"trim=start_frame=0:end_frame={total_frames}", "setpts=PTS-STARTPTS"])

        crf = str(getattr(settings, "CRF_VISUAL", 22))
        input_opts: List[str] = []
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
