#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for autoshorts.
Run this file to generate a YouTube Video.
✅ WITH THUMBNAIL SUPPORT
✅ WITH MODE ENV VAR SETUP
✅ WITH OUTPUT DIRECTORY COPY
✅ WITH YOUTUBE UPLOAD - FIXED
"""
import sys
import os
import re
import json
import shutil
import logging
import tempfile
from datetime import datetime

# CRITICAL: Clear Python cache before starting
def clear_cache():
    """Clear all Python cache files"""
    project_root = os.path.dirname(os.path.abspath(__file__))
    autoshorts_path = os.path.join(project_root, 'autoshorts')
    
    if os.path.exists(autoshorts_path):
        for root, dirs, files in os.walk(autoshorts_path):
            # Remove __pycache__ directories
            if '__pycache__' in dirs:
                cache_dir = os.path.join(root, '__pycache__')
                try:
                    shutil.rmtree(cache_dir)
                    print(f"[CACHE] Cleared: {cache_dir}")
                except Exception as e:
                    print(f"[CACHE] Warning: Could not clear {cache_dir}: {e}")
            
            # Remove .pyc files
            for file in files:
                if file.endswith('.pyc'):
                    pyc_file = os.path.join(root, file)
                    try:
                        os.remove(pyc_file)
                        print(f"[CACHE] Removed: {pyc_file}")
                    except Exception as e:
                        print(f"[CACHE] Warning: Could not remove {pyc_file}: {e}")

# Clear cache first
print("[CACHE] Clearing Python cache...")
clear_cache()
print("[CACHE] Cache cleared successfully\n")

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"[DEBUG] Python path: {project_root}")
print(f"[DEBUG] Checking autoshorts module...")

# Verify autoshorts exists
autoshorts_path = os.path.join(project_root, 'autoshorts')
if not os.path.exists(autoshorts_path):
    print(f"❌ ERROR: autoshorts directory not found at {autoshorts_path}")
    sys.exit(1)

init_file = os.path.join(autoshorts_path, '__init__.py')
if not os.path.exists(init_file):
    print(f"❌ ERROR: autoshorts/__init__.py not found")
    sys.exit(1)

print(f"✅ autoshorts module found at {autoshorts_path}")

# ============================================================
# 🆕 NEW: Modern import with adapter pattern
# ============================================================
# Try to use new ConfigManager-based system first
USE_NEW_SYSTEM = os.getenv("USE_NEW_SYSTEM", "true").lower() == "true"

# Now safe to import
try:
    if USE_NEW_SYSTEM:
        # 🆕 NEW: Modern approach with ConfigManager
        from autoshorts.orchestrator_adapter import create_orchestrator
        from autoshorts.config.config_manager import ConfigManager
        print("✅ Successfully imported OrchestratorAdapter (NEW SYSTEM)")
    else:
        # 🔄 OLD: Legacy approach (still supported)
        from autoshorts.orchestrator import ShortsOrchestrator
        from autoshorts.config.channel_loader import apply_channel_settings
        from autoshorts.config import settings
        print("✅ Successfully imported ShortsOrchestrator (LEGACY SYSTEM)")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\n[DEBUG] Directory structure:")
    for root, dirs, files in os.walk(autoshorts_path):
        level = root.replace(autoshorts_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f'{subindent}{file}')
    sys.exit(1)


def _safe_slug(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value)
    return value.strip("-") or "channel"


def _import_youtube_uploader():
    """Import YouTube uploader with fallbacks."""
    try:
        from autoshorts.upload.youtube_uploader import YouTubeUploader
        return YouTubeUploader
    except Exception as e1:
        print(f"[YouTube] Uploader not at 'autoshorts.upload.youtube_uploader': {e1}")
        try:
            from autoshorts.uploader.youtube_long import YouTubeUploader
            return YouTubeUploader
        except Exception as e2:
            print(f"[YouTube] Uploader not at 'autoshorts.uploader.youtube_long': {e2}")
            return None


def main():
    """Main entry point"""
    print("=" * 60)
    print("  YouTube Video Generator v2.1 (Hybrid System)")
    print("=" * 60)

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )

    try:
        # Get channel name
        channel_name = os.environ.get("CHANNEL_NAME") or os.environ.get("ENV")
        if not channel_name:
            print("⚠️ No CHANNEL_NAME or ENV variable found, using 'default'")
            channel_name = "default"
        print(f"\n📺 Channel: {channel_name}")

        # ============================================================
        # 🆕 NEW SYSTEM: ConfigManager-based approach
        # ============================================================
        if USE_NEW_SYSTEM:
            print("🆕 Using NEW SYSTEM (ConfigManager + Adapter)")

            # Create orchestrator using new system
            print("\n🔧 Creating orchestrator adapter...")
            orchestrator_adapter = create_orchestrator(
                channel_name=channel_name,
                temp_dir=None  # Auto-creates
            )

            # Get config for info
            config = orchestrator_adapter.get_config()

            print(f"🎯 Mode: {config.channel.mode}")
            print(f"🌐 Language: {config.channel.lang}")
            print(f"📝 Topic: {config.channel.topic[:80]}...")
            print(f"📁 Temp dir: {orchestrator_adapter.get_temp_dir()}")

            # ✅ Extract channel settings for YouTube upload
            channel_settings = {
                "CHANNEL_MODE": config.channel.mode,
                "CHANNEL_LANG": config.channel.lang,
                "CHANNEL_TOPIC": config.channel.topic,
                "CHANNEL_NAME": config.channel.name,
            }

            print("\n🎬 Starting video generation...\n")

            # Generate video (uses channel topic from config)
            video_path, metadata = orchestrator_adapter.produce_video()

        # ============================================================
        # 🔄 LEGACY SYSTEM: Old approach (still supported)
        # ============================================================
        else:
            print("🔄 Using LEGACY SYSTEM (backward compatible)")

            # ✅ Load channel-specific settings
            channel_settings = apply_channel_settings(channel_name)

            # ✅ Set MODE as environment variable (CRITICAL for Gemini prompting)
            channel_mode = channel_settings.get("CHANNEL_MODE", "general")
            os.environ["MODE"] = channel_mode
            print(f"🎯 Mode: {channel_mode}")

            # ✅ Set other channel-specific env vars
            os.environ["LANG"] = channel_settings.get("CHANNEL_LANG", "en")
            os.environ["TOPIC"] = channel_settings.get("CHANNEL_TOPIC", "Interesting content")

            print(f"🌐 Language: {os.environ['LANG']}")
            print(f"📝 Topic: {os.environ['TOPIC'][:80]}...")

            # Create temp directory
            temp_dir_name = channel_name.replace(" ", "_")  # ✅ Boşlukları kaldır
            temp_dir = os.path.join(tempfile.gettempdir(), f"autoshorts_{temp_dir_name}")
            os.makedirs(temp_dir, exist_ok=True)
            print(f"📁 Temp dir: {temp_dir}")

            print("\n🔧 Creating orchestrator...")

            # Initialize orchestrator
            orchestrator = ShortsOrchestrator(
                channel_id=channel_name,
                temp_dir=temp_dir,
                api_key=settings.GEMINI_API_KEY,
                pexels_key=settings.PEXELS_API_KEY,
                pixabay_key=settings.PIXABAY_API_KEY
            )

            print("\n🎬 Starting video generation...\n")

            # Generate video using channel topic
            topic_prompt = channel_settings.get("CHANNEL_TOPIC", "Create an interesting video")
            video_path, metadata = orchestrator.produce_video(topic_prompt)
        
        if video_path and metadata:
            # ✅ Create output directory
            out_dir = os.path.join(project_root, "out")
            os.makedirs(out_dir, exist_ok=True)
            
            # ✅ Generate output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            slug = _safe_slug(metadata.get('title', 'video'))[:50]
            output_filename = f"{channel_name}_{slug}_{timestamp}.mp4"
            output_path = os.path.join(out_dir, output_filename)
            
            # ✅ Copy video to output directory
            print(f"\n📦 Copying video to output directory...")
            shutil.copy2(video_path, output_path)
            print(f"✅ Video copied to: {output_path}")
            
            # ✅ Copy thumbnail if exists
            thumbnail_src = metadata.get('script', {}).get('thumbnail_path')
            thumbnail_dest = None
            if thumbnail_src and os.path.exists(thumbnail_src):
                thumbnail_dest = output_path.replace('.mp4', '_thumbnail.jpg')
                shutil.copy2(thumbnail_src, thumbnail_dest)
                print(f"✅ Thumbnail copied to: {thumbnail_dest}")
            
            # ✅ Save metadata
            metadata_path = output_path.replace('.mp4', '_metadata.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"✅ Metadata saved to: {metadata_path}")
            
            print("\n" + "=" * 60)
            print(f"✅ SUCCESS!")
            print(f"📹 Video: {output_path}")
            print(f"📝 Title: {metadata.get('title', 'N/A')}")
            print(f"🏷️ Tags: {len(metadata.get('tags', []))} tags")
            print("=" * 60)
            
            # ✅ YouTube Upload
            upload_enabled = os.environ.get("YOUTUBE_UPLOAD", "true").lower() == "true"
            if upload_enabled:
                print("\n🎥 Starting YouTube upload...")
                
                YouTubeUploader = _import_youtube_uploader()
                if YouTubeUploader is None:
                    print("⚠️ YouTube uploader not available, skipping upload")
                else:
                    try:
                        # ✅ Initialize uploader WITHOUT channel_name parameter
                        uploader = YouTubeUploader()
                        
                        # ✅ Extract data from metadata
                        script_data = metadata.get("script", {})
                        
                        # ✅ Upload video with correct parameters
                        video_id = uploader.upload(
                            video_path=output_path,
                            title=metadata.get("title", ""),
                            description=metadata.get("description", ""),
                            tags=metadata.get("tags", []),
                            category_id="22",  # People & Blogs
                            privacy_status=os.environ.get("YOUTUBE_PRIVACY", "public"),  # ✅ DEĞIŞTI: "private" -> "public"
                            topic=channel_settings.get("CHANNEL_TOPIC", ""),
                            chapters=None,
                            audio_durations=None,
                            thumbnail_path=thumbnail_dest
                        )
                        
                        if video_id:
                            print(f"✅ Uploaded to YouTube: https://youtube.com/watch?v={video_id}")
                            
                            # Set output for GitHub Actions
                            if os.getenv("GITHUB_OUTPUT"):
                                with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                                    f.write(f"youtube_id={video_id}\n")
                                    f.write(f"youtube_url=https://youtube.com/watch?v={video_id}\n")
                        else:
                            print("⚠️ YouTube upload failed - no video ID returned")
                            
                    except Exception as upload_exc:
                        print(f"⚠️ YouTube upload error: {upload_exc}")
                        logging.exception("YouTube upload failed")
            else:
                print("\n⚠️ YouTube upload disabled (YOUTUBE_UPLOAD=false)")
            
            # Set output for GitHub Actions
            if os.getenv("GITHUB_OUTPUT"):
                with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                    f.write(f"video_path={output_path}\n")
                    f.write(f"title={metadata.get('title', '')}\n")
                    description = metadata.get('description', '').replace('\n', ' ')
                    f.write(f"description={description}\n")
                    tags_str = ",".join(metadata.get("tags", []))
                    f.write(f"tags={tags_str}\n")
            
            return 0
        else:
            print("\n" + "=" * 60)
            print("❌ Video generation failed")
            print("=" * 60)
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        return 130
    except Exception as exc:
        print(f"\n\n❌ Fatal error: {exc}")
        logging.exception("Fatal error in main()")
        return 1


if __name__ == "__main__":
    sys.exit(main())
