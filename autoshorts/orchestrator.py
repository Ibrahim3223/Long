"""
Orchestrator - LONG-FORM VIDEO PIPELINE (3-10 minutes)
Handles 20-35 sentence content with chapter support
✅ FIXED: Proper video-audio sync, caption with audio, correct concatenation
"""

import os
import tempfile
import shutil
import logging
import subprocess
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
        self.pexels = PexelsClient()
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
        """Produce video with 20-35 clips - FIXED VERSION"""
        
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
                
                if len(all_videos) >= clips_needed * 2:  # Get 2x for variety
                    break
                    
            except Exception as e:
                logger.warning(f"      ⚠️ Search failed for '{term}': {e}")
                continue
        
        if len(all_videos) < clips_needed:
            logger.error(f"   ❌ Not enough videos: {len(all_videos)}/{clips_needed}")
            return None
        
        logger.info(f"   ✅ Found {len(all_videos)} video clips")
        
        # Download video clips
        logger.info("   📥 Downloading video clips...")
        downloaded_clips = []
        
        for i in range(min(clips_needed, len(all_videos))):
            try:
                video_id, url = all_videos[i]
                clip_path = self._download_video(url, video_id, i)
                downloaded_clips.append(clip_path)
                
            except Exception as e:
                logger.error(f"   ❌ Failed to download clip {i+1}: {e}")
                return None
        
        if len(downloaded_clips) < clips_needed:
            logger.error(f"   ❌ Not enough downloaded clips: {len(downloaded_clips)}/{clips_needed}")
            return None
        
        logger.info(f"   ✅ Downloaded {len(downloaded_clips)} clips")
        
        # ✅ CRITICAL FIX: Process each segment properly
        # 1. Trim video to audio duration
        # 2. Add audio to video
        # 3. Add captions (while preserving audio)
        logger.info("   🎬 Processing video segments with audio and captions...")
        video_segments = []
        
        for i, (clip_path, audio_seg) in enumerate(zip(downloaded_clips, audio_segments)):
            try:
                logger.info(f"      Processing segment {i+1}/{len(audio_segments)}")
                
                # Step 1: Trim video to exact audio duration
                duration = audio_seg['duration']
                trimmed_path = os.path.join(self.temp_dir, f"trimmed_{i:03d}.mp4")
                
                logger.info(f"         Trimming to {duration:.2f}s...")
                subprocess.run([
                    'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                    '-i', clip_path,
                    '-t', str(duration),
                    '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                    '-an',  # Remove original audio (we'll add our own)
                    trimmed_path
                ], check=True, capture_output=True)
                
                if not os.path.exists(trimmed_path):
                    logger.error(f"         ❌ Trim failed")
                    return None
                
                # Step 2: Add TTS audio to trimmed video
                audio_path = audio_seg['audio_path']
                with_audio_path = os.path.join(self.temp_dir, f"with_audio_{i:03d}.mp4")
                
                logger.info(f"         Adding audio...")
                subprocess.run([
                    'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                    '-i', trimmed_path,
                    '-i', audio_path,
                    '-c:v', 'copy',
                    '-c:a', 'aac', '-b:a', '192k',
                    '-shortest',
                    with_audio_path
                ], check=True, capture_output=True)
                
                if not os.path.exists(with_audio_path):
                    logger.error(f"         ❌ Audio merge failed")
                    return None
                
                # Step 3: Add captions (preserving audio!)
                logger.info(f"         Rendering captions...")
                captioned_path = self.caption_renderer.render(
                    video_path=with_audio_path,
                    text=audio_seg['text'],
                    words=audio_seg.get('word_timings', []),
                    duration=audio_seg['duration'],
                    is_hook=(i == 0),
                    sentence_type=audio_seg.get('type', 'body'),
                    temp_dir=self.temp_dir
                )
                
                if not captioned_path or not os.path.exists(captioned_path):
                    logger.warning(f"         ⚠️ Caption failed, using video with audio")
                    captioned_path = with_audio_path
                
                video_segments.append(captioned_path)
                logger.info(f"      ✅ Segment {i+1} complete: {duration:.2f}s")
                
            except Exception as e:
                logger.error(f"   ❌ Segment {i+1} failed: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                return None
        
        # ✅ CRITICAL FIX: Concatenate with proper codec settings
        logger.info("   🔗 Concatenating video segments...")
        final_video = os.path.join(self.temp_dir, "final_longform.mp4")
        
        # Create concat file
        concat_list = os.path.join(self.temp_dir, "concat.txt")
        with open(concat_list, 'w') as f:
            for segment in video_segments:
                # Use absolute path for safety
                abs_path = os.path.abspath(segment)
                f.write(f"file '{abs_path}'\n")
        
        # ✅ FIX: Re-encode instead of copy to ensure compatibility
        try:
            subprocess.run([
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
                '-f', 'concat', '-safe', '0',
                '-i', concat_list,
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
                '-c:a', 'aac', '-b:a', '192k',
                '-movflags', '+faststart',
                final_video
            ], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"   ❌ Concatenation failed: {e.stderr.decode()}")
            return None
        
        if not os.path.exists(final_video):
            logger.error("   ❌ Final video not created")
            return None
        
        # Verify final video has audio
        probe_result = subprocess.run([
            'ffprobe', '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=codec_type',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            final_video
        ], capture_output=True, text=True)
        
        if 'audio' not in probe_result.stdout:
            logger.error("   ❌ Final video has no audio stream!")
            return None
        
        logger.info("   ✅ Final video has audio stream")
        
        # Add BGM if enabled
        if settings.BGM_ENABLE:
            logger.info("   🎵 Adding background music...")
            final_with_bgm = self._add_bgm_to_video(final_video, audio_segments)
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
            
            # Verify download
            if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
                raise ValueError("Downloaded file is empty or invalid")
            
            return output_path
            
        except Exception as e:
            logger.error(f"      ❌ Download failed: {e}")
            raise
    
    def _add_bgm_to_video(self, video_path: str, audio_segments: List[Dict[str, Any]]) -> Optional[str]:
        """
        Add background music to video file.
        
        Args:
            video_path: Path to video file (with voice audio)
            audio_segments: Audio segments with durations
            
        Returns:
            Path to video with BGM, or None if failed
        """
        try:
            # Calculate total duration
            total_duration = sum(seg['duration'] for seg in audio_segments)
            
            # Check if video has audio track
            logger.info("      🔍 Checking for audio track...")
            
            probe_cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'a:0',
                '-show_entries', 'stream=codec_type',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                video_path
            ]
            
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
            
            if probe_result.returncode != 0 or 'audio' not in probe_result.stdout:
                logger.warning("      ⚠️ Video has no audio track - skipping BGM")
                return None
            
            # Extract audio from video
            logger.info("      📤 Extracting audio from video...")
            voice_audio = os.path.join(self.temp_dir, "voice_only.wav")
            
            extract_result = subprocess.run([
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-i', video_path,
                '-vn', '-ar', '48000', '-ac', '1', '-c:a', 'pcm_s16le',
                voice_audio
            ], capture_output=True)
            
            if extract_result.returncode != 0:
                logger.warning(f"      ⚠️ Audio extraction failed: {extract_result.stderr.decode()[:100]}")
                return None
            
            # Verify extracted audio exists and is not empty
            if not os.path.exists(voice_audio) or os.path.getsize(voice_audio) < 1000:
                logger.warning("      ⚠️ Extracted audio is empty - skipping BGM")
                return None
            
            # Add BGM to audio
            logger.info("      🎵 Mixing background music...")
            mixed_audio = self.bgm_manager.add_bgm(
                voice_path=voice_audio,
                duration=total_duration,
                temp_dir=self.temp_dir
            )
            
            if not mixed_audio or mixed_audio == voice_audio:
                logger.warning("      ⚠️ BGM mixing failed or skipped")
                return None
            
            if not os.path.exists(mixed_audio):
                logger.warning("      ⚠️ Mixed audio file not found")
                return None
            
            # Combine video with new audio
            logger.info("      🎬 Combining video with BGM...")
            output_path = os.path.join(self.temp_dir, "final_with_bgm.mp4")
            
            subprocess.run([
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-i', video_path,
                '-i', mixed_audio,
                '-c:v', 'copy',
                '-c:a', 'aac', '-b:a', '192k',
                '-map', '0:v:0', '-map', '1:a:0',
                '-shortest',
                output_path
            ], check=True, capture_output=True)
            
            if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
                logger.warning("      ⚠️ Final video creation failed")
                return None
            
            logger.info(f"      ✅ BGM added successfully")
            return output_path
            
        except Exception as e:
            logger.error(f"      ❌ BGM addition failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
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
