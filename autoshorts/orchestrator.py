# -*- coding: utf-8 -*-
"""
Main orchestrator - FIXED VERSION
✅ Landscape-only videos
✅ Better scene-to-video matching
✅ Colorful karaoke captions
"""
import os
import pathlib
import random
import logging
import time
import re
from typing import List, Dict, Optional, Tuple

import requests

from autoshorts.config import settings
from autoshorts.content.gemini_client import GeminiClient
from autoshorts.content.quality_scorer import QualityScorer
from autoshorts.content.text_utils import normalize_sentence, extract_keywords, simplify_query
from autoshorts.tts.edge_handler import TTSHandler
from autoshorts.captions.renderer import CaptionRenderer
from autoshorts.captions.karaoke_ass import build_karaoke_ass, get_random_style
from autoshorts.audio.bgm_manager import BGMManager
from autoshorts.state.state_guard import StateGuard
from autoshorts.state.novelty_guard import NoveltyGuard
from autoshorts.utils.ffmpeg_utils import (
    run,
    ffprobe_duration
)

logger = logging.getLogger(__name__)


class ShortsOrchestrator:
    """
    Complete orchestrator for automated YouTube Shorts/Long-form production.
    ✅ FIXED: Landscape videos, better relevance, colorful captions
    """

    def __init__(
        self,
        channel_id: str,
        temp_dir: str,
        api_key: str = None,
        pexels_key: str = None
    ):
        """Initialize orchestrator."""
        self.channel_id = channel_id
        self.temp_dir = pathlib.Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.gemini = GeminiClient(api_key=api_key or settings.GEMINI_API_KEY)
        self.quality_scorer = QualityScorer()
        self.tts = TTSHandler()
        self.caption_renderer = CaptionRenderer()
        self.bgm_manager = BGMManager()

        self.pexels_key = pexels_key or settings.PEXELS_API_KEY
        if not self.pexels_key:
            raise ValueError("PEXELS_API_KEY required")

        # State guards
        self.state_guard = StateGuard(channel_id)
        self.novelty_guard = NoveltyGuard()

        logger.info(f"🎬 ShortsOrchestrator initialized for channel: {channel_id}")

    # ============================================================================
    # HELPER METHODS - FFmpeg utilities
    # ============================================================================
    
    def _concat_videos(self, video_paths: List[str], output_path: str, fps: int = 25):
        """Concatenate multiple videos."""
        if not video_paths:
            raise ValueError("No videos to concatenate")
        
        # Create concat file
        concat_file = output_path.replace(".mp4", "_concat.txt")
        
        with open(concat_file, 'w') as f:
            for vp in video_paths:
                f.write(f"file '{vp}'\n")
        
        try:
            run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "concat", "-safe", "0", "-i", concat_file,
                "-c:v", "libx264", "-preset", "medium",
                "-crf", str(settings.CRF_VISUAL),
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "128k",
                "-r", str(fps), "-vsync", "cfr",
                output_path
            ])
        finally:
            pathlib.Path(concat_file).unlink(missing_ok=True)
    
    def _overlay_audio(
        self, 
        video_path: str, 
        audio_path: str, 
        output_path: str,
        video_duration: float
    ):
        """Overlay audio on video."""
        run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", video_path,
            "-i", audio_path,
            "-t", str(video_duration),
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "128k",
            "-map", "0:v:0", "-map", "1:a:0",
            "-shortest",
            output_path
        ])
    
    # ============================================================================
    # END HELPER METHODS
    # ============================================================================

    def produce_video(
        self,
        topic_prompt: str,
        max_retries: int = 3
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """
        Produce complete video with metadata.
        
        Returns:
            (video_path, metadata) or (None, None) on failure
        """
        logger.info("=" * 70)
        logger.info("🎬 STARTING VIDEO PRODUCTION")
        logger.info("=" * 70)
        logger.info(f"📝 Topic: {topic_prompt[:100]}...")

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"\n🔄 Attempt {attempt}/{max_retries}")

                # ✅ Add delay between retries (avoid API rate limits)
                if attempt > 1:
                    delay = 2 * attempt  # Progressive delay: 2s, 4s, 6s
                    logger.info(f"   ⏳ Waiting {delay}s before retry...")
                    time.sleep(delay)

                # 1. Generate script
                script = self._generate_script(topic_prompt)
                if not script:
                    logger.error("❌ Script generation failed")
                    continue

                # 2. Check novelty
                sentences_text = [s.get("text", "") for s in script.get("sentences", [])]
                is_novel, similarity = self.novelty_guard.check_novelty(
                    title=script.get("title", ""),
                    script=sentences_text
                )
                if not is_novel:
                    logger.warning(f"⚠️ Script too similar to recent ones (similarity: {similarity:.2f}), regenerating...")
                    continue

                # 3. Produce video
                video_path = self._produce_video_from_script(script)
                if not video_path:
                    logger.error("❌ Video production failed")
                    continue

                # 4. Prepare metadata
                metadata = {
                    "title": script.get("title", ""),
                    "description": script.get("description", ""),
                    "tags": script.get("tags", []),
                    "hook": script.get("hook", ""),
                    "script": script
                }

                # 5. Update state
                self.state_guard.save_successful_script(script)
                
                # Update novelty guard with title and sentences
                sentences_text = [s.get("text", "") for s in script.get("sentences", [])]
                self.novelty_guard.add_used_script(
                    title=script.get("title", ""),
                    script=sentences_text
                )

                logger.info(f"\n✅ VIDEO PRODUCTION COMPLETE")
                logger.info(f"📁 Output: {video_path}")
                return video_path, metadata

            except Exception as e:
                logger.error(f"❌ Attempt {attempt} failed: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                continue

        logger.error("❌ All attempts failed")
        return None, None

    def _generate_script(self, topic_prompt: str) -> Optional[Dict]:
        """Generate script from topic."""
        logger.info("\n📝 Generating script...")

        try:
            # ✅ Use GeminiClient.generate() with correct parameters
            # Long-form videos: 4-7 minutes = 240-420 seconds
            target_duration = 300  # 5 minutes default
            style = "educational, informative, engaging"
            
            response = self.gemini.generate(
                topic=topic_prompt,
                style=style,
                duration=target_duration,
                additional_context=None
            )
            
            # ✅ Convert ContentResponse to dict format for orchestrator
            script = {
                "title": response.metadata.get("title", ""),
                "description": response.metadata.get("description", ""),
                "tags": response.metadata.get("tags", []),
                "hook": response.hook,
                "sentences": [],
                "chapters": response.chapters
            }
            
            # ✅ Convert script list to sentence format
            # First sentence is hook
            script["sentences"].append({
                "text": response.hook,
                "type": "hook",
                "visual_keywords": [response.main_visual_focus] if response.main_visual_focus else []
            })
            
            # Main script sentences with visual keywords
            for idx, sentence_text in enumerate(response.script):
                # Distribute visual keywords across sentences
                keywords = []
                if idx < len(response.search_queries):
                    # Use corresponding search query as keyword
                    keywords = [response.search_queries[idx]]
                elif response.search_queries:
                    # Cycle through available keywords
                    keywords = [response.search_queries[idx % len(response.search_queries)]]
                
                script["sentences"].append({
                    "text": sentence_text,
                    "type": "buildup",
                    "visual_keywords": keywords
                })
            
            # Last sentence is CTA
            script["sentences"].append({
                "text": response.cta,
                "type": "conclusion",
                "visual_keywords": []
            })
            
            if not script["sentences"]:
                logger.error("   ❌ No sentences generated")
                return None

            # Quality check
            sentences = [s.get("text", "") for s in script["sentences"]]
            title = script.get("title", "")
            
            scores = self.quality_scorer.score(sentences, title)
            overall_score = scores.get("overall", 0.0)
            
            # ✅ LOWERED: Long-form content has different quality criteria than shorts
            if overall_score < 4.0:
                logger.warning(f"   ⚠️ Low quality script (score: {overall_score:.1f}), skipping")
                return None

            logger.info(f"   ✅ Script generated")
            logger.info(f"      Quality score: {overall_score:.1f}/10")
            logger.info(f"      Title: {script.get('title', 'N/A')[:60]}...")
            logger.info(f"      Scenes: {len(script.get('sentences', []))}")

            return script

        except Exception as e:
            logger.error(f"   ❌ Script generation error: {e}")
            
            # Check if it's a temporary API error
            error_str = str(e)
            if "503" in error_str or "UNAVAILABLE" in error_str or "overloaded" in error_str.lower():
                logger.warning(f"   ⚠️ Gemini API temporarily unavailable, will retry...")
            elif "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                logger.warning(f"   ⚠️ Rate limit hit, will retry after delay...")
            
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def _produce_video_from_script(self, script: Dict) -> Optional[str]:
        """Produce video from script."""
        logger.info("\n🎬 Producing video from script...")

        sentences = script.get("sentences", [])
        if not sentences:
            logger.error("   ❌ No sentences in script")
            return None

        scene_videos = []
        total_duration = 0.0

        # Process each scene
        for idx, sentence_data in enumerate(sentences, 1):
            logger.info(f"\n{'='*70}")
            logger.info(f"🎞️  SCENE {idx}/{len(sentences)}")
            logger.info(f"{'='*70}")

            try:
                scene_path = self._produce_scene(sentence_data, idx)

                if scene_path and os.path.exists(scene_path):
                    scene_videos.append(scene_path)
                    scene_duration = ffprobe_duration(scene_path)
                    total_duration += scene_duration
                    logger.info(f"   ✅ Scene {idx} completed ({scene_duration:.2f}s)")
                else:
                    logger.error(f"   ❌ Scene {idx} failed")

            except Exception as e:
                logger.error(f"   ❌ Scene {idx} error: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                continue

        if not scene_videos:
            logger.error("   ❌ No scenes produced")
            return None

        logger.info(f"\n{'='*70}")
        logger.info(f"🎬 FINAL ASSEMBLY")
        logger.info(f"{'='*70}")
        logger.info(f"   📹 Scenes: {len(scene_videos)}")
        logger.info(f"   ⏱️  Duration: {total_duration:.1f}s ({total_duration/60:.1f}min)")

        # Concatenate scenes
        output_name = f"final_video_{int(time.time())}.mp4"
        output_path = str(self.temp_dir / output_name)

        try:
            self._concat_videos(scene_videos, output_path, fps=settings.TARGET_FPS)

            if not os.path.exists(output_path):
                logger.error("   ❌ Concatenation failed")
                return None

            # Add BGM if enabled
            if settings.BGM_ENABLED:
                logger.info(f"\n   🎵 Adding background music...")
                final_with_bgm = self.bgm_manager.add_bgm_to_video(
                    output_path,
                    total_duration,
                    str(self.temp_dir)
                )

                if final_with_bgm and os.path.exists(final_with_bgm):
                    output_path = final_with_bgm
                    logger.info(f"   ✅ BGM added")

            logger.info(f"\n   ✅ Final video ready")
            logger.info(f"   📊 Size: {os.path.getsize(output_path) / (1024*1024):.1f}MB")

            return output_path

        except Exception as e:
            logger.error(f"   ❌ Final assembly error: {e}")
            return None

    def _produce_scene(
        self,
        sentence_data: Dict,
        scene_idx: int
    ) -> Optional[str]:
        """Produce a single scene."""
        text = sentence_data.get("text", "").strip()
        scene_type = sentence_data.get("type", "buildup")
        keywords = sentence_data.get("visual_keywords", [])

        if not text:
            logger.warning(f"   ⚠️ Empty text for scene {scene_idx}")
            return None

        logger.info(f"   📝 Text: {text[:100]}...")
        logger.info(f"   🎯 Type: {scene_type}")
        logger.info(f"   🔑 Keywords: {keywords}")

        # 1. Generate audio
        audio_path, words, audio_duration = self._generate_audio(text, scene_idx, scene_type)
        if not audio_path:
            logger.error(f"   ❌ Audio generation failed")
            return None

        # 2. Get video clip
        video_path = self._get_video_clip(keywords, text, audio_duration, scene_idx, scene_type)
        if not video_path:
            logger.error(f"   ❌ Video selection failed")
            return None

        # 3. Add captions
        video_with_captions = self._render_captions_on_scene(
            video_path, text, words, audio_duration, scene_type
        )

        # 4. Overlay audio
        final_scene = self._overlay_audio_on_video(
            video_with_captions, audio_path, audio_duration, scene_idx
        )

        return final_scene

    def _generate_audio(
        self,
        text: str,
        scene_idx: int,
        scene_type: str
    ) -> Tuple[Optional[str], List[Tuple[str, float]], float]:
        """Generate TTS audio."""
        logger.info(f"   🎤 Generating audio...")

        audio_filename = f"scene_{scene_idx:03d}_audio.wav"
        audio_path = str(self.temp_dir / audio_filename)

        try:
            # Use synthesize method which returns (duration, words)
            duration, words = self.tts.synthesize(
                text=text,
                wav_out=audio_path
            )

            if not os.path.exists(audio_path):
                logger.error(f"      ❌ TTS failed - file not created")
                return None, [], 0.0

            logger.info(f"      ✅ Audio: {duration:.2f}s, {len(words)} words")

            return audio_path, words, duration

        except Exception as e:
            logger.error(f"      ❌ Audio error: {e}")
            return None, [], 0.0

    def _get_video_clip(
        self,
        keywords: List[str],
        text: str,
        duration: float,
        scene_idx: int,
        scene_type: str
    ) -> Optional[str]:
        """
        Get video clip with BETTER relevance.
        
        ✅ FIXED: Landscape-only videos with smarter keyword extraction
        """
        logger.info(f"   🎥 Getting video clip...")

        # ✅ Choose best search keyword
        search_keyword = self._choose_video_keyword(keywords, text)
        logger.info(f"      🔍 Search: '{search_keyword}'")

        # ✅ Fetch video URL (landscape only)
        video_url = self._fetch_pexels_video(search_keyword, fallback_keywords=keywords)

        if not video_url:
            logger.error(f"      ❌ No video found")
            return None

        # Download and process
        raw_video = str(self.temp_dir / f"scene_{scene_idx:03d}_raw.mp4")
        
        if not self._download_video(video_url, raw_video):
            logger.error(f"      ❌ Download failed")
            return None

        # Process video
        processed_video = self._download_and_process_clip(
            raw_video, duration, scene_idx, scene_type
        )

        return processed_video

    def _choose_video_keyword(
        self,
        keywords: List[str],
        text: str
    ) -> str:
        """
        ✅ IMPROVED: Choose best keyword for video search.
        
        Priority:
        1. Use provided keywords (already optimized)
        2. Extract key nouns from text
        3. Fallback to simplified query
        """
        # Use first keyword if available
        if keywords and keywords[0]:
            return keywords[0]

        # Extract from text
        text_lower = text.lower()

        # Stop words to exclude
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'can', 'this', 'that', 'these', 'those', 'it', 'its', 'their', 'them'
        }

        # Extract meaningful words
        words = re.findall(r'\b[a-z]+\b', text_lower)
        important_words = [w for w in words if len(w) > 3 and w not in stop_words]

        if important_words:
            # Take first 2-3 important words
            return " ".join(important_words[:3])

        # Fallback to simplified query
        return simplify_query(text, keep=3)

    def _fetch_pexels_video(
        self,
        query: str,
        per_page: int = 15,
        fallback_keywords: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        ✅ FIXED: Fetch video from Pexels with LANDSCAPE-ONLY filtering.
        """
        # Try main query first
        all_queries = [query]
        
        # Add fallback keywords if provided
        if fallback_keywords:
            all_queries.extend([kw for kw in fallback_keywords if kw and kw != query][:2])
        
        # Generic fallbacks
        all_queries.extend(["nature landscape", "abstract motion"])

        for attempt, current_query in enumerate(all_queries, 1):
            try:
                logger.debug(f"         Attempt {attempt}: '{current_query}'")

                # ✅ CRITICAL: Request landscape orientation
                params = {
                    "query": current_query,
                    "per_page": per_page,
                    "orientation": "landscape"  # ✅ LANDSCAPE ONLY
                }

                headers = {"Authorization": self.pexels_key}
                
                response = requests.get(
                    "https://api.pexels.com/videos/search",
                    params=params,
                    headers=headers,
                    timeout=10
                )
                response.raise_for_status()

                data = response.json()
                videos = data.get("videos", [])

                if not videos:
                    logger.debug(f"         No videos for '{current_query}'")
                    continue

                # ✅ DOUBLE-CHECK: Filter landscape videos
                landscape_videos = []
                for video in videos:
                    width = video.get("width", 0)
                    height = video.get("height", 0)

                    # Ensure width > height (horizontal)
                    if width > height:
                        landscape_videos.append(video)

                if not landscape_videos:
                    logger.debug(f"         No landscape videos found")
                    continue

                # Pick random video
                video = random.choice(landscape_videos)
                video_files = video.get("video_files", [])

                # Get best quality landscape video
                for vf in video_files:
                    if vf.get("quality") == "hd":
                        vf_width = vf.get("width", 0)
                        vf_height = vf.get("height", 0)
                        
                        if vf_width > vf_height:  # Double check
                            logger.info(f"      ✅ Video found (attempt {attempt})")
                            return vf.get("link")

                # Fallback to any landscape file
                for vf in video_files:
                    vf_width = vf.get("width", 0)
                    vf_height = vf.get("height", 0)
                    
                    if vf_width > vf_height:
                        return vf.get("link")

            except Exception as e:
                logger.debug(f"         Query {attempt} error: {e}")
                continue

        logger.warning(f"      ⚠️ No landscape video found after {len(all_queries)} attempts")
        return None

    def _download_video(self, url: str, output_path: str) -> bool:
        """Download video from URL."""
        try:
            logger.info(f"      ⬇️  Downloading...")

            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            if os.path.exists(output_path):
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                logger.info(f"         ✅ Downloaded: {size_mb:.1f}MB")
                return True

            return False

        except Exception as e:
            logger.error(f"         ❌ Download error: {e}")
            return False

    def _download_and_process_clip(
        self,
        video_path: str,
        target_duration: float,
        scene_idx: int,
        scene_type: str
    ) -> Optional[str]:
        """Process video clip: loop, crop to 16:9, effects."""
        logger.info(f"      🎬 Processing video...")

        output_name = f"scene_{scene_idx:03d}_processed.mp4"
        output_path = str(self.temp_dir / output_name)

        try:
            source_duration = ffprobe_duration(video_path)

            if source_duration <= 0:
                logger.error(f"         ❌ Invalid duration")
                return None

            # Calculate loops
            loops_needed = int(target_duration / source_duration) + 1

            # Build filter chain
            filters = []

            # 1. Loop
            if loops_needed > 1:
                filters.append(f"loop={loops_needed}:size=1:start=0")

            # 2. Scale and crop to 1920x1080
            filters.append("scale=1920:1080:force_original_aspect_ratio=increase")
            filters.append("crop=1920:1080")

            # 3. Subtle effects based on scene type
            if scene_type == "hook":
                # Gentle zoom for hooks
                filters.append(
                    f"zoompan=z='min(zoom+0.0005,1.1)':d={int(target_duration * settings.TARGET_FPS)}"
                    f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1920x1080:fps={settings.TARGET_FPS}"
                )
            else:
                # Very subtle pan
                filters.append(
                    f"zoompan=z='1.05':d={int(target_duration * settings.TARGET_FPS)}"
                    f":x='if(gte(on,1),x+2,0)':y='ih/2-(ih/zoom/2)':s=1920x1080:fps={settings.TARGET_FPS}"
                )

            # 4. Trim to exact frames
            target_frames = int(target_duration * settings.TARGET_FPS)
            filters.append(f"trim=start_frame=0:end_frame={target_frames}")
            filters.append("setpts=PTS-STARTPTS")

            filter_chain = ",".join(filters)

            # Execute
            run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", video_path,
                "-vf", filter_chain,
                "-r", str(settings.TARGET_FPS),
                "-vsync", "cfr",
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", str(settings.CRF_VISUAL),
                "-pix_fmt", "yuv420p",
                "-an",  # Remove source audio
                output_path
            ])

            if os.path.exists(output_path):
                logger.info(f"         ✅ Processed: {target_duration:.2f}s")
                return output_path
            else:
                logger.error(f"         ❌ Processing failed")
                return None

        except Exception as e:
            logger.error(f"         ❌ Processing error: {e}")
            return None

    def _render_captions_on_scene(
        self,
        video_path: str,
        text: str,
        words: List[Tuple[str, float]],
        duration: float,
        sentence_type: str
    ) -> str:
        """
        ✅ FIXED: Render captions with COLORFUL karaoke style!
        """
        logger.info(f"   📝 Adding captions...")

        try:
            # Check if captions enabled
            if not settings.KARAOKE_CAPTIONS:
                logger.info(f"      ⚠️ Captions disabled")
                return video_path

            # Generate colorful ASS file
            ass_path = video_path.replace(".mp4", ".ass")
            
            # ✅ NEW: Use colorful karaoke caption system!
            style_name = get_random_style()
            logger.info(f"      🎨 Caption style: {style_name}")

            ass_content = build_karaoke_ass(
                text=text,
                seg_dur=duration,
                words=words,
                is_hook=(sentence_type == "hook"),
                style_name=style_name
            )

            # Write ASS file
            with open(ass_path, 'w', encoding='utf-8') as f:
                f.write(ass_content)

            if not os.path.exists(ass_path):
                logger.error(f"      ❌ ASS file creation failed")
                return video_path

            # Burn captions
            output = video_path.replace(".mp4", "_caption.mp4")
            tmp_out = output.replace(".mp4", ".tmp.mp4")

            try:
                # Burn subtitles
                run([
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", video_path,
                    "-vf", f"subtitles='{ass_path}':force_style='Kerning=1',setsar=1,fps={settings.TARGET_FPS}",
                    "-r", str(settings.TARGET_FPS), "-vsync", "cfr",
                    "-c:v", "libx264", "-preset", "medium",
                    "-crf", str(settings.CRF_VISUAL),
                    "-pix_fmt", "yuv420p",
                    "-an",  # No audio yet
                    tmp_out
                ])

                if not os.path.exists(tmp_out):
                    logger.error(f"      ❌ Caption burn failed")
                    return video_path

                # Trim to exact duration
                frames = int(duration * settings.TARGET_FPS)
                
                run([
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", tmp_out,
                    "-vf", f"setsar=1,fps={settings.TARGET_FPS},trim=start_frame=0:end_frame={frames}",
                    "-r", str(settings.TARGET_FPS), "-vsync", "cfr",
                    "-c:v", "libx264", "-preset", "medium",
                    "-crf", str(settings.CRF_VISUAL),
                    "-pix_fmt", "yuv420p",
                    output
                ])

                if os.path.exists(output):
                    logger.info(f"      ✅ Captions added with colorful style!")
                    return output

            finally:
                pathlib.Path(ass_path).unlink(missing_ok=True)
                pathlib.Path(tmp_out).unlink(missing_ok=True)

            return video_path

        except Exception as e:
            logger.error(f"      ❌ Caption error: {e}")
            return video_path

    def _overlay_audio_on_video(
        self,
        video_path: str,
        audio_path: str,
        duration: float,
        scene_idx: int
    ) -> Optional[str]:
        """Overlay audio on video."""
        logger.info(f"   🔊 Overlaying audio...")

        output_name = f"scene_{scene_idx:03d}_final.mp4"
        output_path = str(self.temp_dir / output_name)

        try:
            self._overlay_audio(
                video_path=video_path,
                audio_path=audio_path,
                output_path=output_path,
                video_duration=duration
            )

            if os.path.exists(output_path):
                logger.info(f"      ✅ Audio overlaid")
                return output_path
            else:
                logger.error(f"      ❌ Overlay failed")
                return None

        except Exception as e:
            logger.error(f"      ❌ Overlay error: {e}")
            return None
