# -*- coding: utf-8 -*-
"""
Video Orchestrator - ULTIMATE LONG-FORM VERSION
✅ FIXED: Landscape-only videos with better scene relevance
✅ Improved search queries for scene-video matching
"""
import os
import pathlib
import logging
import random
import re
from typing import List, Dict, Optional, Tuple

from autoshorts.config import settings
from autoshorts.content.text_utils import split_into_sentences
from autoshorts.tts.edge_handler import EdgeTTSHandler
from autoshorts.captions.renderer import CaptionRenderer
from autoshorts.audio.bgm_manager import BGMManager
from autoshorts.utils.ffmpeg_utils import (
    run, concat_videos, overlay_audio, apply_zoom_pan, ffprobe_duration
)

logger = logging.getLogger(__name__)


class VideoOrchestrator:
    """Orchestrate complete video production with better scene relevance."""
    
    def __init__(
        self,
        pexels_client,
        temp_dir: str,
        aspect_ratio: str = "16:9"
    ):
        """Initialize video orchestrator."""
        self.pexels = pexels_client
        self.temp_dir = pathlib.Path(temp_dir)
        self.aspect_ratio = aspect_ratio
        
        self.tts_handler = EdgeTTSHandler()
        self.caption_renderer = CaptionRenderer()
        self.bgm_manager = BGMManager()
        
        # Video dimensions for 16:9
        self.width = 1920
        self.height = 1080
        
        logger.info(f"🎬 Video orchestrator initialized ({aspect_ratio})")
    
    def produce_complete_video(
        self,
        script: Dict,
        video_title: str,
        enable_bgm: bool = True
    ) -> str:
        """
        Produce complete video from script.
        
        ✅ FIXED: Better scene-to-video matching with landscape-only videos
        """
        try:
            logger.info("=" * 70)
            logger.info("🎬 STARTING VIDEO PRODUCTION")
            logger.info("=" * 70)
            
            sentences = script.get("sentences", [])
            if not sentences:
                raise ValueError("No sentences in script")
            
            scene_videos = []
            total_duration = 0.0
            
            # ✅ Process each scene with better video selection
            for idx, sentence_data in enumerate(sentences, 1):
                logger.info(f"\n{'='*70}")
                logger.info(f"🎞️  SCENE {idx}/{len(sentences)}")
                logger.info(f"{'='*70}")
                
                text = sentence_data.get("text", "").strip()
                scene_type = sentence_data.get("type", "buildup")
                keywords = sentence_data.get("visual_keywords", [])
                
                if not text:
                    logger.warning(f"⚠️  Scene {idx}: Empty text, skipping")
                    continue
                
                logger.info(f"   📝 Text: {text[:100]}...")
                logger.info(f"   🎯 Type: {scene_type}")
                logger.info(f"   🔑 Keywords: {keywords}")
                
                try:
                    # ✅ Generate audio with word timings
                    audio_path, words, duration = self._generate_scene_audio(
                        text, idx, scene_type
                    )
                    
                    if not audio_path or duration <= 0:
                        logger.error(f"   ❌ Scene {idx}: Audio generation failed")
                        continue
                    
                    # ✅ Select and prepare video with better relevance
                    video_path = self._select_and_prepare_scene_video(
                        keywords, text, duration, idx, scene_type
                    )
                    
                    if not video_path:
                        logger.error(f"   ❌ Scene {idx}: Video selection failed")
                        continue
                    
                    # ✅ Add captions
                    video_with_captions = self.caption_renderer.render(
                        video_path=video_path,
                        text=text,
                        words=words,
                        duration=duration,
                        is_hook=(scene_type == "hook"),
                        sentence_type=scene_type,
                        temp_dir=str(self.temp_dir)
                    )
                    
                    # ✅ Overlay audio
                    final_scene = self._overlay_audio_on_video(
                        video_with_captions, audio_path, duration, idx
                    )
                    
                    if final_scene and os.path.exists(final_scene):
                        scene_videos.append(final_scene)
                        total_duration += duration
                        logger.info(f"   ✅ Scene {idx} completed ({duration:.2f}s)")
                    else:
                        logger.error(f"   ❌ Scene {idx}: Final scene not created")
                    
                except Exception as e:
                    logger.error(f"   ❌ Scene {idx} error: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    continue
            
            if not scene_videos:
                raise ValueError("No scenes were successfully created")
            
            logger.info(f"\n{'='*70}")
            logger.info(f"🎬 FINAL ASSEMBLY")
            logger.info(f"{'='*70}")
            logger.info(f"   📹 Scenes: {len(scene_videos)}")
            logger.info(f"   ⏱️  Total: {total_duration:.1f}s ({total_duration/60:.1f}min)")
            
            # ✅ Concatenate all scenes
            output_name = f"{video_title}_video.mp4"
            output_path = str(self.temp_dir / output_name)
            
            concat_videos(scene_videos, output_path, fps=settings.TARGET_FPS)
            
            if not os.path.exists(output_path):
                raise ValueError("Final concatenation failed")
            
            # ✅ Add BGM if enabled
            if enable_bgm and settings.BGM_ENABLED:
                logger.info(f"\n   🎵 Adding background music...")
                final_with_bgm = self.bgm_manager.add_bgm_to_video(
                    output_path,
                    total_duration,
                    str(self.temp_dir)
                )
                
                if final_with_bgm and os.path.exists(final_with_bgm):
                    output_path = final_with_bgm
                    logger.info(f"   ✅ BGM added successfully")
            
            logger.info(f"\n{'='*70}")
            logger.info(f"✅ VIDEO PRODUCTION COMPLETE")
            logger.info(f"{'='*70}")
            logger.info(f"   📁 Output: {output_path}")
            logger.info(f"   📊 Size: {os.path.getsize(output_path) / (1024*1024):.1f}MB")
            
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Video production failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def _generate_scene_audio(
        self,
        text: str,
        scene_idx: int,
        scene_type: str
    ) -> Tuple[str, List[Tuple[str, float]], float]:
        """Generate audio for a scene."""
        logger.info(f"   🎤 Generating audio...")
        
        audio_filename = f"scene_{scene_idx:03d}_audio.mp3"
        audio_path = str(self.temp_dir / audio_filename)
        
        # Select voice based on language
        voice = settings.VOICE_NAME
        
        # Generate with word timings
        success, words = self.tts_handler.generate_with_timings(
            text=text,
            output_path=audio_path,
            voice=voice
        )
        
        if not success or not os.path.exists(audio_path):
            logger.error(f"      ❌ Audio generation failed")
            return None, [], 0.0
        
        duration = ffprobe_duration(audio_path)
        logger.info(f"      ✅ Audio: {duration:.2f}s, {len(words)} words")
        
        return audio_path, words, duration
    
    def _select_and_prepare_scene_video(
        self,
        keywords: List[str],
        text: str,
        duration: float,
        scene_idx: int,
        scene_type: str
    ) -> Optional[str]:
        """
        Select and prepare video with BETTER relevance to scene.
        
        ✅ FIXED: Smarter keyword extraction and landscape-only filtering
        """
        logger.info(f"   🎥 Selecting video...")
        
        # ✅ IMPROVED: Extract better search keywords from text and keywords
        search_query = self._extract_best_search_query(keywords, text)
        
        logger.info(f"      🔍 Search: '{search_query}'")
        
        # ✅ Try to get landscape videos with fallback queries
        video_url = self._choose_pexels_video(
            search_query,
            fallback_queries=[
                self._get_fallback_query(keywords, text, 1),
                self._get_fallback_query(keywords, text, 2),
                "nature landscape",  # Generic fallback
            ]
        )
        
        if not video_url:
            logger.error(f"      ❌ No video found for scene")
            return None
        
        # ✅ Download video
        video_filename = f"scene_{scene_idx:03d}_raw.mp4"
        raw_video_path = str(self.temp_dir / video_filename)
        
        if not self._download_video(video_url, raw_video_path):
            logger.error(f"      ❌ Video download failed")
            return None
        
        # ✅ Process video (loop, crop, effects)
        processed_path = self._process_scene_video(
            raw_video_path, duration, scene_idx, scene_type
        )
        
        return processed_path
    
    def _extract_best_search_query(
        self,
        keywords: List[str],
        text: str,
        max_words: int = 3
    ) -> str:
        """
        ✅ IMPROVED: Extract most relevant search terms from keywords and text.
        
        Priority:
        1. Use provided keywords (most relevant)
        2. Extract nouns from text
        3. Use important action words
        """
        # Start with provided keywords
        if keywords:
            # Take top 2-3 most important keywords
            search_words = keywords[:max_words]
            return " ".join(search_words)
        
        # Fallback: Extract from text
        # Remove common filler words
        text_lower = text.lower()
        
        # Common words to exclude
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'can', 'this', 'that', 'these', 'those', 'it', 'its', 'their', 'them'
        }
        
        # Extract words
        words = re.findall(r'\b[a-z]+\b', text_lower)
        
        # Filter and prioritize
        important_words = []
        for word in words:
            if len(word) > 3 and word not in stop_words:
                important_words.append(word)
        
        # Take first few important words
        if important_words:
            return " ".join(important_words[:max_words])
        
        # Last resort: use first few words of text
        first_words = text.split()[:max_words]
        return " ".join(first_words)
    
    def _get_fallback_query(
        self,
        keywords: List[str],
        text: str,
        fallback_level: int
    ) -> str:
        """Generate fallback search queries."""
        if fallback_level == 1 and keywords:
            # Try different keyword combination
            return keywords[0] if keywords else "landscape"
        
        # More generic fallback
        generic_terms = [
            "nature", "scenery", "landscape", "sky", "water",
            "forest", "mountain", "ocean", "sunset", "clouds"
        ]
        
        return random.choice(generic_terms)
    
    def _choose_pexels_video(
        self,
        query: str,
        fallback_queries: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Choose best video from Pexels with landscape-only filtering.
        
        ✅ FIXED: Always request landscape orientation
        """
        all_queries = [query] + (fallback_queries or [])
        
        for attempt, current_query in enumerate(all_queries, 1):
            try:
                # ✅ CRITICAL: Always use landscape orientation
                videos = self.pexels.search_videos(
                    query=current_query,
                    per_page=15,
                    orientation="landscape"  # ✅ LANDSCAPE ONLY
                )
                
                if not videos:
                    logger.debug(f"      ⚠️ No videos for query {attempt}: '{current_query}'")
                    continue
                
                # ✅ Pick random video from results
                video = random.choice(videos)
                video_url = self.pexels.get_video_file_url(video, quality="hd")
                
                if video_url:
                    logger.info(f"      ✅ Video found (query {attempt}: '{current_query}')")
                    return video_url
                
            except Exception as e:
                logger.debug(f"      ⚠️ Query {attempt} error: {e}")
                continue
        
        logger.warning(f"      ⚠️ No video found after {len(all_queries)} attempts")
        return None
    
    def _download_video(self, url: str, output_path: str) -> bool:
        """Download video from URL."""
        try:
            import requests
            
            logger.info(f"      ⬇️  Downloading video...")
            
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
    
    def _process_scene_video(
        self,
        video_path: str,
        target_duration: float,
        scene_idx: int,
        scene_type: str
    ) -> str:
        """Process video: loop, crop to 16:9, add effects."""
        logger.info(f"      🎬 Processing video...")
        
        output_name = f"scene_{scene_idx:03d}_processed.mp4"
        output_path = str(self.temp_dir / output_name)
        
        # Get video info
        source_duration = ffprobe_duration(video_path)
        
        if source_duration <= 0:
            logger.error(f"         ❌ Invalid source duration")
            return None
        
        # Calculate loops needed
        loops_needed = int(target_duration / source_duration) + 1
        
        # Build filter chain
        filters = []
        
        # 1. Loop video
        if loops_needed > 1:
            filters.append(f"loop={loops_needed}:size=1:start=0")
        
        # 2. Scale and crop to exact 1920x1080
        filters.append(f"scale=1920:1080:force_original_aspect_ratio=increase")
        filters.append("crop=1920:1080")
        
        # 3. Subtle zoom/pan effect based on scene type
        if scene_type == "hook":
            # Gentle zoom for hooks
            filters.append(
                f"zoompan=z='min(zoom+0.0005,1.1)':d={int(target_duration * settings.TARGET_FPS)}"
                f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1920x1080:fps={settings.TARGET_FPS}"
            )
        else:
            # Very subtle pan for other scenes
            filters.append(
                f"zoompan=z='1.05':d={int(target_duration * settings.TARGET_FPS)}"
                f":x='if(gte(on,1),x+2,0)':y='ih/2-(ih/zoom/2)':s=1920x1080:fps={settings.TARGET_FPS}"
            )
        
        # 4. Set exact frame count
        target_frames = int(target_duration * settings.TARGET_FPS)
        filters.append(f"trim=start_frame=0:end_frame={target_frames}")
        filters.append("setpts=PTS-STARTPTS")
        
        filter_chain = ",".join(filters)
        
        # Execute FFmpeg
        try:
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
                "-an",  # Remove audio from source
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
    
    def _overlay_audio_on_video(
        self,
        video_path: str,
        audio_path: str,
        duration: float,
        scene_idx: int
    ) -> str:
        """Overlay audio onto video."""
        logger.info(f"   🔊 Overlaying audio...")
        
        output_name = f"scene_{scene_idx:03d}_final.mp4"
        output_path = str(self.temp_dir / output_name)
        
        try:
            overlay_audio(
                video_path=video_path,
                audio_path=audio_path,
                output_path=output_path,
                video_duration=duration
            )
            
            if os.path.exists(output_path):
                logger.info(f"      ✅ Audio overlaid")
                return output_path
            else:
                logger.error(f"      ❌ Audio overlay failed")
                return None
                
        except Exception as e:
            logger.error(f"      ❌ Audio overlay error: {e}")
            return None
