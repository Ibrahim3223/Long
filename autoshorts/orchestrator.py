"""
Orchestrator - LONG-FORM VIDEO PIPELINE (3-10 minutes)
Handles 20-35 sentence content with chapter support
"""

import os
import tempfile
import shutil
import logging
from typing import Optional, Dict, Any, List

from .config import settings
from .content.gemini_client import GeminiClient
from .tts.edge_handler import TTSHandler
from .video.pexels_client import PexelsClient
from .captions.renderer import CaptionRenderer
from .audio.bgm_manager import BGMManager
from .upload.youtube_uploader import YouTubeUploader
from .state.novelty_guard import NoveltyGuard

logger = logging.getLogger(__name__)


class LongFormOrchestrator:
    """Main orchestrator for long-form YouTube videos"""
    
    def __init__(self):
        """Initialize all components"""
        logger.info("🚀 Initializing Long-Form Orchestrator...")
        
        self.gemini = GeminiClient(
            api_key=settings.GEMINI_API_KEY,
            model=settings.GEMINI_MODEL
        )
        self.tts = TTSHandler()  # Voice is read from settings internally
        self.pexels = PexelsClient()  # ✅ FIXED: No api_key parameter needed
        self.caption_renderer = CaptionRenderer()
        self.bgm_manager = BGMManager()
        self.uploader = YouTubeUploader()
        self.novelty_guard = NoveltyGuard()
        
        self.temp_dir = None
        logger.info("✅ Long-Form Orchestrator ready")
    
    def run(self) -> Optional[str]:
        """Execute full pipeline for long-form video"""
        self.temp_dir = tempfile.mkdtemp(prefix="longform_")
        
        try:
            # Phase 1: Generate content (20-35 sentences + chapters)
            logger.info("\n📝 Phase 1: Generating long-form content...")
            content = self._generate_content()
            if not content:
                return None
            
            # Phase 2: TTS (20-35 audio segments)
            logger.info("\n🎤 Phase 2: Text-to-speech (20-35 segments)...")
            audio_segments = self._generate_tts(content['script'])
            if not audio_segments:
                return None
            
            # Phase 3: Video production (20-35 video clips)
            logger.info("\n🎬 Phase 3: Video production...")
            video_path = self._produce_video(
                audio_segments,
                content['search_queries'],
                content['chapters']
            )
            if not video_path:
                return None
            
            # Phase 4: Upload with chapters
            logger.info("\n📤 Phase 4: Uploading to YouTube...")
            if settings.UPLOAD_TO_YT:
                video_id = self._upload(
                    video_path,
                    content,
                    audio_segments
                )
                logger.info(f"✅ Success! Video ID: {video_id}")
                return video_id
            else:
                logger.info(f"⏭️ Upload skipped. Video saved: {video_path}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise
        finally:
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
    
    def _generate_content(self) -> Optional[Dict[str, Any]]:
        """Generate 20-35 sentence content with chapters"""
        
        max_attempts = settings.MAX_GENERATION_ATTEMPTS
        
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"   Attempt {attempt}/{max_attempts}")
                
                # Generate content
                content_response = self.gemini.generate(
                    topic=settings.CHANNEL_TOPIC,
                    style=settings.CONTENT_STYLE,
                    duration=settings.TARGET_DURATION,
                    additional_context=settings.ADDITIONAL_PROMPT_CONTEXT
                )
                
                # Validate sentence count
                sentence_count = len(content_response.script)
                if not (settings.MIN_SENTENCES <= sentence_count <= settings.MAX_SENTENCES):
                    logger.warning(f"   ⚠️ Sentence count {sentence_count} out of range")
                    continue
                
                # Build content dict
                content = {
                    'hook': content_response.hook,
                    'script': content_response.script,
                    'cta': content_response.cta,
                    'search_queries': content_response.search_queries,
                    'main_visual_focus': content_response.main_visual_focus,
                    'chapters': content_response.chapters,
                    'metadata': content_response.metadata
                }
                
                logger.info(f"   ✅ Content: {sentence_count} sentences, {len(content['chapters'])} chapters")
                return content
                
            except Exception as e:
                logger.error(f"   ❌ Attempt {attempt} failed: {e}")
                if attempt == max_attempts:
                    return None
        
        return None
    
    def _generate_tts(self, sentences: List[str]) -> Optional[List[Dict[str, Any]]]:
        """Generate TTS for all sentences"""
        
        audio_segments = []
        total_duration = 0.0
        
        for i, sentence in enumerate(sentences):
            try:
                logger.info(f"   Processing sentence {i+1}/{len(sentences)}")
                
                # Generate audio
                audio_data = self.tts.generate(sentence)
                
                # Save to temp file
                audio_path = os.path.join(self.temp_dir, f"audio_{i:03d}.mp3")
                with open(audio_path, 'wb') as f:
                    f.write(audio_data['audio'])
                
                segment = {
                    'text': sentence,
                    'audio_path': audio_path,
                    'duration': audio_data['duration'],
                    'word_timings': audio_data.get('word_timings', []),
                    'type': 'hook' if i == 0 else 'body'
                }
                
                audio_segments.append(segment)
                total_duration += audio_data['duration']
                
            except Exception as e:
                logger.error(f"   ❌ TTS failed for sentence {i+1}: {e}")
                return None
        
        logger.info(f"   ✅ Total duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")
        
        # Check duration constraints
        if not (settings.TARGET_MIN_SEC <= total_duration <= settings.TARGET_MAX_SEC):
            logger.warning(f"   ⚠️ Duration {total_duration:.1f}s out of range")
        
        return audio_segments
    
    def _produce_video(
        self,
        audio_segments: List[Dict[str, Any]],
        search_queries: List[str],
        chapters: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Produce video with 20-35 clips"""
        
        # Calculate how many video clips we need
        clips_needed = len(audio_segments)
        
        logger.info(f"   Need {clips_needed} video clips for {len(audio_segments)} sentences")
        
        # Search and download videos using PexelsClient methods
        logger.info("   🔍 Searching for video clips...")
        all_videos = []
        
        # Use the search_simple method for better results
        for term in search_queries[:20]:  # Use more search terms
            try:
                # search_simple returns List[Tuple[video_id, download_url]]
                results = self.pexels.search_simple(query=term, count=3)
                all_videos.extend(results)
                
                if len(all_videos) >= clips_needed:
                    break
            except Exception as e:
                logger.warning(f"   ⚠️ Search failed for '{term}': {e}")
                continue
        
        # If not enough, allow reuse
        if len(all_videos) < clips_needed:
            logger.warning(f"   ⚠️ Only found {len(all_videos)} clips, need {clips_needed}")
            logger.info("   🔄 Enabling video reuse...")
            # Duplicate videos to reach target
            while len(all_videos) < clips_needed:
                all_videos.extend(all_videos[:clips_needed - len(all_videos)])
        
        logger.info(f"   ✅ Collected {len(all_videos)} video clips")
        
        # Create video segments with captions
        logger.info("   🎨 Creating video segments with captions...")
        video_segments = []
        
        for i, (audio_seg, (video_id, download_url)) in enumerate(zip(audio_segments, all_videos)):
            try:
                # Download video
                video_path = self._download_video(download_url, video_id, i)
                
                # Add captions
                captioned_path = self.caption_renderer.render(
                    video_path=video_path,
                    audio_segment=audio_seg,
                    output_dir=self.temp_dir,
                    index=i
                )
                
                video_segments.append(captioned_path)
                
            except Exception as e:
                logger.error(f"   ❌ Segment {i} failed: {e}")
                return None
        
        # Concatenate all segments
        logger.info("   🔗 Concatenating video segments...")
        final_video = os.path.join(self.temp_dir, "final_longform.mp4")
        
        # Use FFmpeg to concatenate
        concat_list = os.path.join(self.temp_dir, "concat.txt")
        with open(concat_list, 'w') as f:
            for segment in video_segments:
                f.write(f"file '{segment}'\n")
        
        import subprocess
        cmd = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', concat_list,
            '-c', 'copy',
            final_video
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Add BGM if enabled
        if settings.BGM_ENABLE:
            logger.info("   🎵 Adding background music...")
            final_with_bgm = self.bgm_manager.add_bgm(final_video, self.temp_dir)
            if final_with_bgm:
                final_video = final_with_bgm
        
        # Move to output dir
        output_path = os.path.join(settings.OUTPUT_DIR, f"longform_{os.path.basename(final_video)}")
        shutil.copy(final_video, output_path)
        
        logger.info(f"   ✅ Video saved: {output_path}")
        return output_path
    
    def _download_video(self, url: str, video_id: int, index: int) -> str:
        """Download a single video clip"""
        import requests
        
        output_path = os.path.join(self.temp_dir, f"clip_{index:03d}.mp4")
        
        try:
            logger.info(f"      Downloading clip {index+1} (ID: {video_id})...")
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return output_path
            
        except Exception as e:
            logger.error(f"      ❌ Download failed: {e}")
            raise
    
    def _upload(
        self,
        video_path: str,
        content: Dict[str, Any],
        audio_segments: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Upload video with chapter timestamps"""
        
        # Extract durations for chapter timestamps
        audio_durations = [seg['duration'] for seg in audio_segments]
        
        try:
            video_id = self.uploader.upload(
                video_path=video_path,
                title=content['metadata']['title'],
                description=content['metadata']['description'],
                tags=content['metadata']['tags'],
                topic=settings.CHANNEL_TOPIC,
                chapters=content['chapters'],
                audio_durations=audio_durations
            )
            
            return video_id
            
        except Exception as e:
            logger.error(f"   ❌ Upload failed: {e}")
            return None


# ============================================================================
# BACKWARD COMPATIBILITY ALIAS
# ============================================================================
# main.py expects 'ShortsOrchestrator' class name
# Provide alias so existing code works with long-form orchestrator
ShortsOrchestrator = LongFormOrchestrator
