# -*- coding: utf-8 -*-
"""High level orchestration for generating complete videos."""
from __future__ import annotations

import hashlib
import logging
import os
import pathlib
import random
import shutil
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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

logger = logging.getLogger(__name__)


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
        """Return runtime and shared cache locations for a URL."""
        digest = hashlib.sha1(url.encode("utf-8")).hexdigest()
        runtime = self._cache_dir / f"{digest}.mp4"
        shared = self._shared_root / f"{digest}.mp4"
        return runtime, shared

    def try_copy(self, url: str, destination: pathlib.Path) -> bool:
        """Copy a cached clip into *destination* if available."""
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
        """Persist *source* in the runtime and shared caches."""
        runtime, shared = self.cache_paths(url)
        try:
            if not runtime.exists():
                shutil.copy2(source, runtime)
            self._runtime_cache[url] = runtime
        except Exception as exc:  # pragma: no cover - best effort cache copy
            logger.debug("Unable to store runtime cache for %s: %s", url, exc)

        try:
            if not shared.exists():
                shutil.copy2(source, shared)
            self._enforce_shared_budget()
        except Exception as exc:  # pragma: no cover - best effort cache copy
            logger.debug("Unable to store shared cache for %s: %s", url, exc)

    def _enforce_shared_budget(self) -> None:
        """Trim the shared cache directory to the configured limit."""
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

        self.gemini = GeminiClient(api_key=api_key or settings.GEMINI_API_KEY)
        self.quality_scorer = QualityScorer()
        self.tts = TTSHandler()
        self.caption_renderer = CaptionRenderer()
        self.bgm_manager = BGMManager()
        self.state_guard = StateGuard(channel_id)
        self.novelty_guard = NoveltyGuard()

        self.pexels = PexelsClient(api_key=pexels_key or settings.PEXELS_API_KEY)
        self._clip_cache = _ClipCache(self.temp_dir)
        self._video_candidates: Dict[str, List[Dict[str, str]]] = {}

        self._http = requests.Session()
        try:
            adapter = HTTPAdapter(pool_connections=6, pool_maxsize=12)
            self._http.mount("https://", adapter)
            self._http.mount("http://", adapter)
        except Exception as exc:  # pragma: no cover - adapter install best effort
            logger.debug("Unable to enable HTTP pooling: %s", exc)

        if not self.pexels.api_key:
            raise ValueError("PEXELS_API_KEY required")

        logger.info("🎬 ShortsOrchestrator ready for channel %s", channel_id)

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

                sentences = [s.get("text", "") for s in script.get("sentences", [])]
                is_fresh, similarity = self.novelty_guard.check_novelty(
                    title=script.get("title", ""), script=sentences
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

                self.state_guard.save_successful_script(script)
                self.novelty_guard.add_used_script(
                    title=script.get("title", ""), script=sentences
                )

                logger.info("✅ Video ready: %s", video_path)
                return video_path, metadata

            except Exception as exc:  # pragma: no cover - defensive logging
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
                duration=settings.TARGET_DURATION,
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
        }

        script["sentences"].append(
            {
                "text": response.hook,
                "type": "hook",
                "visual_keywords": [response.main_visual_focus]
                if response.main_visual_focus
                else [],
            }
        )

        for idx, sentence in enumerate(response.script):
            keywords = []
            if idx < len(response.search_queries):
                keywords = [response.search_queries[idx]]
            elif response.search_queries:
                keywords = [response.search_queries[idx % len(response.search_queries)]]

            script["sentences"].append(
                {
                    "text": sentence,
                    "type": "buildup",
                    "visual_keywords": keywords,
                }
            )

        script["sentences"].append(
            {
                "text": response.cta,
                "type": "conclusion",
                "visual_keywords": [],
            }
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

        rendered: List[str] = []
        total_duration = 0.0

        for index, sentence in enumerate(sentences, 1):
            logger.info("—" * 60)
            logger.info("Scene %s/%s", index, len(sentences))
            scene_path = self._produce_scene(sentence, index)
            if not scene_path:
                logger.error("Scene %s failed", index)
                continue

            rendered.append(scene_path)
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

        if settings.BGM_ENABLED:
            logger.info("Adding BGM to final video")
            with_bgm = self.bgm_manager.add_bgm_to_video(
                final_path, total_duration, str(self.temp_dir)
            )
            if with_bgm and os.path.exists(with_bgm):
                final_path = with_bgm

        logger.info("Final video assembled: %s", final_path)
        return final_path

    def _produce_scene(self, sentence: Dict, index: int) -> Optional[str]:
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

        clip_path = self._prepare_clip(text, keywords, duration, index, sentence_type)
        if not clip_path:
            return None

        captioned = self._render_captions(
            clip_path,
            text,
            words,
            duration,
            sentence_type,
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
    ) -> Optional[str]:
        logger.info("Selecting clip for scene %s", index)
        keyword = self._choose_keyword(text, keywords)
        candidate = self._next_candidate(keyword, keywords, text)
        if not candidate:
            logger.error("No clip candidate found for scene %s", index)
            return None

        local_raw = self.temp_dir / f"scene_{index:03d}_raw.mp4"
        if not self._download_clip(candidate["url"], local_raw):
            logger.error("Download failed for %s", candidate["url"])
            return None

        processed = self.temp_dir / f"scene_{index:03d}_proc.mp4"
        try:
            self._process_clip(local_raw, processed, duration, sentence_type)
        except Exception as exc:
            logger.error("Processing failed: %s", exc)
            logger.debug("", exc_info=True)
            return None

        return str(processed)

    def _choose_keyword(self, text: str, keywords: Sequence[str]) -> str:
        pool: List[str] = []
        pool.extend([kw for kw in keywords if kw])
        pool.extend(extract_keywords(text))
        simplified = simplify_query(text)
        if simplified:
            pool.append(simplified)

        pool.append(settings.CHANNEL_TOPIC)
        pool.append("interesting landscape")

        choice = next((kw for kw in pool if kw), text[:40])
        logger.info("Search keyword: %s", choice)
        return choice

    def _next_candidate(
        self,
        primary: str,
        fallbacks: Sequence[str],
        text: str,
    ) -> Optional[Dict[str, str]]:
        queries = [primary]
        queries.extend([kw for kw in fallbacks if kw and kw != primary])

        simplified = simplify_query(text)
        if simplified and simplified not in queries:
            queries.append(simplified)

        queries.append(settings.CHANNEL_TOPIC)
        queries.append("dynamic landscape video")

        for query in queries:
            pool = self._get_candidates(query)
            while pool:
                candidate = pool.pop(0)
                url = candidate.get("url")
                if not url:
                    continue
                if candidate.get("duration", 0) < settings.PEXELS_MIN_DURATION:
                    continue
                logger.info("Using clip %s for query '%s'", url, query)
                return candidate
        return None

    def _get_candidates(self, query: str) -> List[Dict[str, str]]:
        cache = self._video_candidates.get(query)
        if cache is not None:
            return cache

        videos = self.pexels.search_videos(query, per_page=settings.PEXELS_PER_PAGE)
        results: List[Dict[str, str]] = []
        for video in videos:
            url = self.pexels.get_video_file_url(video, quality="hd")
            if not url:
                continue
            results.append(
                {
                    "url": url,
                    "id": str(video.get("id", "")),
                    "duration": float(video.get("duration", 0.0)),
                }
            )

        random.shuffle(results)
        self._video_candidates[query] = results
        return results

    def _download_clip(self, url: str, destination: pathlib.Path) -> bool:
        if self._clip_cache.try_copy(url, destination):
            logger.info("Reused cached clip for %s", url)
            return True

        try:
            with self._http.get(url, stream=True, timeout=45) as response:
                response.raise_for_status()
                with open(destination, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=8192):
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
        filters: List[str] = []
        if loops > 1:
            filters.append(f"loop={loops}:size=1:start=0")

        filters.extend(
            [
                "scale=1920:1080:force_original_aspect_ratio=increase",
                "crop=1920:1080",
            ]
        )

        if sentence_type == "hook":
            filters.append(
                "zoompan=z='min(zoom+0.0006,1.12)':d=1:s=1920x1080:fps=%d"
                % settings.TARGET_FPS
            )
        else:
            filters.append(
                "zoompan=z='1.06':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                "d=1:s=1920x1080:fps=%d" % settings.TARGET_FPS
            )

        total_frames = max(2, int(round(target_duration * settings.TARGET_FPS)))
        filters.extend(
            [
                f"trim=start_frame=0:end_frame={total_frames}",
                "setpts=PTS-STARTPTS",
            ]
        )

        run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-vf",
                ",".join(filters),
                "-r",
                str(settings.TARGET_FPS),
                "-vsync",
                "cfr",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                str(settings.CRF_VISUAL),
                "-pix_fmt",
                "yuv420p",
                "-an",
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
                is_hook=sentence_type == "hook",
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
                    "medium",
                    "-crf",
                    str(settings.CRF_VISUAL),
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "160k",
                    "-r",
                    str(settings.TARGET_FPS),
                    "-vsync",
                    "cfr",
                    output,
                ]
            )
        finally:
            concat_file.unlink(missing_ok=True)
