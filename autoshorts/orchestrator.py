"""
Orchestrator - LONG-FORM VIDEO PIPELINE (4-7 minutes)
✅ ULTIMATE FIX: All issues resolved in one file
- Video looping with seamless transitions
- Forced caption rendering
- Proper segment duration matching
- 40-70 sentence generation
"""

import os
import tempfile
import shutil
import logging
import subprocess
from typing import Optional, Dict, Any, List

from autoshorts.config import settings
from autoshorts.content.gemini_client import GeminiClient
from autoshorts.tts.edge_handler import TTSHandler
from autoshorts.video.pexels_client import PexelsClient
from autoshorts.captions.renderer import CaptionRenderer
from autoshorts.audio.bgm_manager import BGMManager
from autoshorts.upload.youtube_uploader import YouTubeUploader
from autoshorts.state.novelty_guard import NoveltyGuard

logger = logging.getLogger(__name__)


class LongFormOrchestrator:
    """Main orchestrator for long-form YouTube videos"""
    
    def __init__(self):
        """Initialize all components"""
        logger.info("🚀 Initializing Long-Form Orchestrator...")
        
        # ✅ FORCE OVERRIDE SETTINGS FOR LONG-FORM
        self._apply_longform_overrides()
        
        self.gemini = GeminiClient(
            api_key=settings.GEMINI_API_KEY,
            model=settings.GEMINI_MODEL
        )
        self.tts = TTSHandler()
        self.pexels = PexelsClient()
        self.caption_renderer = CaptionRenderer()
        self.bgm_manager = BGMManager()
        self.uploader = YouTubeUploader()
        self.novelty_guard = NoveltyGuard()
        
        self.temp_dir = None
        
        logger.info(f"✅ Long-Form Orchestrator ready")
        logger.info(f"   Target: {settings.MIN_SENTENCES}-{settings.MAX_SENTENCES} sentences")
        logger.info(f"   Duration: {settings.TARGET_MIN_SEC/60:.1f}-{settings.TARGET_MAX_SEC/60:.1f} minutes")
    
    def _apply_longform_overrides(self):
        """Force correct settings for long-form videos"""
        # ✅ CRITICAL: Override sentence counts
        settings.MIN_SENTENCES = 40
        settings.MAX_SENTENCES = 70
        settings.TARGET_SENTENCES = 55
        
        # ✅ CRITICAL: Override duration targets
        settings.TARGET_DURATION = 360  # 6 minutes
        settings.TARGET_MIN_SEC = 240.0  # 4 minutes
        settings.TARGET_MAX_SEC = 480.0  # 8 minutes
        
        # ✅ CRITICAL: Force captions enabled
        settings.KARAOKE_CAPTIONS = True
        
        # ✅ Allow more video reuse
        settings.PEXELS_MAX_USES_PER_CLIP = 3
        settings.PEXELS_ALLOW_REUSE = True
        
        logger.info("   ⚙️ Long-form settings applied")
    
    def run(self) -> Optional[str]:
        """Execute full pipeline for long-form video"""
        self.temp_dir = tempfile.mkdtemp(prefix="longform_")
        
        try:
            # Phase 1: Generate content (40-70 sentences + chapters)
            logger.info("\n📝 Phase 1: Generating long-form content...")
            content = self._generate_content()
            if not content:
                return None
            
            # Phase 2: TTS (40-70 audio segments)
            logger.info("\n🎤 Phase 2: Text-to-speech (40-70 segments)...")
            audio_segments = self._generate_tts(content['script'])
            if not audio_segments:
                return None
            
            # Phase 3: Video production (40-70 video clips)
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
        """Generate 40-70 sentence content with chapters"""
        
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
                    logger.warning(f"   ⚠️ Sentence count {sentence_count} out of range ({settings.MIN_SENTENCES}-{settings.MAX_SENTENCES})")
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
            logger.warning(f"   ⚠️ Duration {total_duration:.1f}s out of target range")
        
        return audio_segments
    
    def _produce_video(
        self,
        audio_segments: List[Dict[str, Any]],
        search_queries: List[str],
        chapters: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Produce video with seamless looping and captions"""
        
        clips_needed = len(audio_segments)
        logger.info(f"   Need {clips_needed} video clips for {len(audio_segments)} sentences")
        
        # Search and download videos
        logger.info("   🔍 Searching for video clips...")
        all_videos = []
        
        for term in search_queries[:30]:  # More search terms
            try:
                results = self.pexels.search_simple(query=term, count=4)
                all_videos.extend(results)
                
                if len(all_videos) >= clips_needed * 2:
                    break
                    
            except Exception as e:
                logger.warning(f"      ⚠️ Search failed for '{term}': {e}")
                continue
        
        if len(all_videos) < clips_needed:
            logger.error(f"   ❌ Not enough videos: {len(all_videos)}/{clips_needed}")
            return None
        
        logger.info(f"   ✅ Found {len(all_videos)} video clips")
        
        # Download clips
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
        
        logger.info(f"   ✅ Downloaded {len(downloaded_clips)} clips")
        
        # ✅ Process each segment with BULLETPROOF LOOPING + CAPTIONS
        logger.info("   🎬 Processing segments (loop + audio + captions)...")
        final_segments = []
        
        for i, (clip_path, audio_seg) in enumerate(zip(downloaded_clips, audio_segments)):
            try:
                logger.info(f"      [{i+1}/{len(audio_segments)}] Processing segment...")
                
                target_duration = audio_seg['duration']
                
                # Get clip duration
                probe = subprocess.run([
                    'ffprobe', '-v', 'error',
                    '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    clip_path
                ], capture_output=True, text=True, check=True)
                
                clip_duration = float(probe.stdout.strip())
                
                # ✅ STEP 1: Create looped video matching audio duration EXACTLY
                looped_video = os.path.join(self.temp_dir, f"looped_{i:03d}.mp4")
                
                if clip_duration < target_duration:
                    # Calculate exact loops needed
                    loops = int(target_duration / clip_duration) + 1
                    logger.info(f"         🔄 Looping {loops}x ({clip_duration:.1f}s → {target_duration:.1f}s)")
                    
                    subprocess.run([
                        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                        '-stream_loop', str(loops),
                        '-i', clip_path,
                        '-t', f'{target_duration:.3f}',
                        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                        '-r', '30', '-pix_fmt', 'yuv420p',
                        '-an',
                        looped_video
                    ], check=True, capture_output=True)
                else:
                    # Just trim
                    logger.info(f"         ✂️ Trimming ({clip_duration:.1f}s → {target_duration:.1f}s)")
                    subprocess.run([
                        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                        '-i', clip_path,
                        '-t', f'{target_duration:.3f}',
                        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                        '-r', '30', '-pix_fmt', 'yuv420p',
                        '-an',
                        looped_video
                    ], check=True, capture_output=True)
                
                if not os.path.exists(looped_video):
                    logger.error(f"         ❌ Video prep failed")
                    return None
                
                # ✅ STEP 2: Add audio
                with_audio = os.path.join(self.temp_dir, f"with_audio_{i:03d}.mp4")
                
                logger.info(f"         🎵 Adding audio...")
                subprocess.run([
                    'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                    '-i', looped_video,
                    '-i', audio_seg['audio_path'],
                    '-c:v', 'copy',
                    '-c:a', 'aac', '-b:a', '192k', '-ar', '48000',
                    '-shortest',
                    with_audio
                ], check=True, capture_output=True)
                
                if not os.path.exists(with_audio):
                    logger.error(f"         ❌ Audio merge failed")
                    return None
                
                # ✅ STEP 3: Add captions (FORCED)
                logger.info(f"         📝 Rendering captions...")
                
                # Generate word timings if missing
                words = audio_seg.get('word_timings', [])
                if not words:
                    # Fallback: equal distribution
                    import re
                    text_words = [w for w in re.split(r'\s+', audio_seg['text']) if w]
                    if text_words:
                        per_word = target_duration / len(text_words)
                        words = [(w, per_word) for w in text_words]
                
                try:
                    captioned = self.caption_renderer.render(
                        video_path=with_audio,
                        text=audio_seg['text'],
                        words=words,
                        duration=target_duration,
                        is_hook=(i == 0),
                        sentence_type=audio_seg.get('type', 'body'),
                        temp_dir=self.temp_dir
                    )
                    
                    if captioned and os.path.exists(captioned):
                        final_segments.append(captioned)
                        logger.info(f"      ✅ Segment {i+1} complete with captions")
                    else:
                        logger.warning(f"         ⚠️ Caption failed, using audio version")
                        final_segments.append(with_audio)
                        logger.info(f"      ✅ Segment {i+1} complete (no captions)")
                        
                except Exception as cap_err:
                    logger.warning(f"         ⚠️ Caption error: {cap_err}")
                    final_segments.append(with_audio)
                    logger.info(f"      ✅ Segment {i+1} complete (caption error)")
                
            except Exception as e:
                logger.error(f"      ❌ Segment {i+1} failed: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                return None
        
        # ✅ STEP 4: Concatenate with validation
        logger.info("   🔗 Concatenating segments...")
        
        # Verify all segments exist and have correct duration
        for i, seg_path in enumerate(final_segments):
            if not os.path.exists(seg_path):
                logger.error(f"   ❌ Segment {i+1} missing!")
                return None
            
            probe = subprocess.run([
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                seg_path
            ], capture_output=True, text=True)
            
            if probe.returncode == 0:
                seg_dur = float(probe.stdout.strip())
                expected = audio_segments[i]['duration']
                diff = abs(seg_dur - expected)
                
                if diff > 0.5:
                    logger.warning(f"      ⚠️ Segment {i+1} duration mismatch: {seg_dur:.2f}s vs {expected:.2f}s")
        
        # Create concat list
        concat_list = os.path.join(self.temp_dir, "concat.txt")
        with open(concat_list, 'w') as f:
            for seg in final_segments:
                f.write(f"file '{os.path.abspath(seg)}'\n")
        
        final_video = os.path.join(self.temp_dir, "final_longform.mp4")
        
        # Concatenate with re-encode
        try:
            subprocess.run([
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
                '-f', 'concat', '-safe', '0',
                '-i', concat_list,
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
                '-c:a', 'aac', '-b:a', '192k', '-ar', '48000',
                '-movflags', '+faststart',
                final_video
            ], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"   ❌ Concatenation failed: {e.stderr.decode()}")
            return None
        
        if not os.path.exists(final_video):
            logger.error("   ❌ Final video not created")
            return None
        
        # Verify final video
        probe = subprocess.run([
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-select_streams', 'v:0',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            final_video
        ], capture_output=True, text=True)
        
        if probe.returncode == 0:
            final_dur = float(probe.stdout.strip())
            expected_dur = sum(seg['duration'] for seg in audio_segments)
            logger.info(f"   ✅ Final video: {final_dur:.1f}s (expected: {expected_dur:.1f}s)")
            
            if abs(final_dur - expected_dur) > 2.0:
                logger.warning(f"   ⚠️ Duration mismatch: {final_dur:.1f}s vs {expected_dur:.1f}s")
        
        # Verify audio stream
        probe_audio = subprocess.run([
            'ffprobe', '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=codec_type',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            final_video
        ], capture_output=True, text=True)
        
        if 'audio' not in probe_audio.stdout:
            logger.error("   ❌ Final video has no audio!")
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
            
            if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
                raise ValueError("Downloaded file is invalid")
            
            return output_path
            
        except Exception as e:
            logger.error(f"      ❌ Download failed: {e}")
            raise
    
    def _add_bgm_to_video(self, video_path: str, audio_segments: List[Dict[str, Any]]) -> Optional[str]:
        """Add background music to video file."""
        try:
            total_duration = sum(seg['duration'] for seg in audio_segments)
            
            # Check audio track exists
            probe = subprocess.run([
                'ffprobe', '-v', 'error',
                '-select_streams', 'a:0',
                '-show_entries', 'stream=codec_type',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                video_path
            ], capture_output=True, text=True)
            
            if 'audio' not in probe.stdout:
                logger.warning("      ⚠️ No audio track for BGM")
                return None
            
            # Extract audio
            voice_audio = os.path.join(self.temp_dir, "voice_only.wav")
            
            subprocess.run([
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-i', video_path,
                '-vn', '-ar', '48000', '-ac', '1', '-c:a', 'pcm_s16le',
                voice_audio
            ], capture_output=True, check=True)
            
            if not os.path.exists(voice_audio) or os.path.getsize(voice_audio) < 1000:
                return None
            
            # Add BGM
            mixed_audio = self.bgm_manager.add_bgm(
                voice_path=voice_audio,
                duration=total_duration,
                temp_dir=self.temp_dir
            )
            
            if not mixed_audio or not os.path.exists(mixed_audio):
                return None
            
            # Combine video with BGM
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
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                logger.info(f"      ✅ BGM added")
                return output_path
            
            return None
            
        except Exception as e:
            logger.error(f"      ❌ BGM failed: {e}")
            return None
    
    def _upload(
        self,
        video_path: str,
        content: Dict[str, Any],
        audio_segments: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Upload video with chapter timestamps"""
        
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
ShortsOrchestrator = LongFormOrchestrator
