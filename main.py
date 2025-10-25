#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for autoshorts.
Run this file to generate a YouTube Short.
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

# Now safe to import
try:
    from autoshorts.orchestrator import ShortsOrchestrator
    from autoshorts.config.channel_loader import apply_channel_settings
    from autoshorts.config import settings
    print("✅ Successfully imported ShortsOrchestrator")
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


def _import_long_uploader():
    """Import YouTube long-form uploader with fallbacks."""
    # 1) Eski yol (bazı repolarda mevcut olabilir)
    try:
        from autoshorts.uploader.youtube_long import YouTubeUploader as LongUploader
        return LongUploader, "autoshorts.uploader.youtube_long"
    except Exception as e1:
        print(f"[YouTube] Long-form uploader not at 'autoshorts.uploader.youtube_long': {e1}")

    # 2) Senin dosyan: autoshorts/upload/youtube_uploader.py
    try:
        from autoshorts.upload.youtube_uploader import YouTubeUploader as LongUploader
        return LongUploader, "autoshorts.upload.youtube_uploader"
    except Exception as e2:
        print(f"[YouTube] Long-form uploader not at 'autoshorts.upload.youtube_uploader': {e2}")

    return None, None


def main():
    """Main entry point."""
    print("=" * 60)
    print("  YouTube Shorts Generator v2.0")
    print("=" * 60)
    
    # ✅ Set up logging properly
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    try:
        # ✅ Get channel name from environment
        channel_name = os.environ.get("CHANNEL_NAME") or os.environ.get("ENV")
        if not channel_name:
            print("⚠️ No CHANNEL_NAME or ENV variable found, using 'default'")
            channel_name = "default"

        print(f"\n📺 Channel: {channel_name}")
        
        # ✅ Load channel-specific settings
        channel_settings = apply_channel_settings(channel_name)
        
        # ✅ Create temp directory
        temp_dir = os.path.join(tempfile.gettempdir(), f"autoshorts_{channel_name}")
        os.makedirs(temp_dir, exist_ok=True)
        print(f"📁 Temp dir: {temp_dir}")
        
        print("\n🔧 Creating orchestrator...")
        
        # ✅ Initialize orchestrator with proper parameters
        orchestrator = ShortsOrchestrator(
            channel_id=channel_name,
            temp_dir=temp_dir,
            api_key=settings.GEMINI_API_KEY,
            pexels_key=settings.PEXELS_API_KEY
        )
        
        print("\n🎬 Starting video generation...\n")
        
        # ✅ Generate video using channel topic
        topic_prompt = channel_settings.get("CHANNEL_TOPIC", "Create an interesting video")
        video_path, metadata = orchestrator.produce_video(topic_prompt)
        
        if video_path and metadata:
            print("\n" + "=" * 60)
            print(f"✅ SUCCESS! Video created: {video_path}")
            print(f"   Title: {metadata.get('title', 'N/A')[:60]}...")
            print("=" * 60)

            safe_channel = _safe_slug(channel_name)
            out_root = os.path.join(project_root, "out", safe_channel)
            os.makedirs(out_root, exist_ok=True)

            timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            destination_name = f"{safe_channel}_{timestamp}.mp4"
            destination_path = os.path.join(out_root, destination_name)

            print(f"\n📦 Copying final video to {destination_path}")
            shutil.copy2(video_path, destination_path)

            meta_payload = {
                "channel": channel_name,
                "channel_slug": safe_channel,
                "generated_at": timestamp,
                "source_video": video_path,
                "output_video": destination_path,
                "topic_prompt": topic_prompt,
                "metadata": metadata,
            }

            metadata_path = destination_path.replace(".mp4", ".json")
            with open(metadata_path, "w", encoding="utf-8") as fh:
                json.dump(meta_payload, fh, ensure_ascii=False, indent=2)

            print(f"🗒️ Metadata saved to {metadata_path}")

            # =========================
            # YouTube UPLOAD (optional)
            # =========================
            upload_flag = str(os.getenv("UPLOAD_TO_YT", "0")).lower() not in ("0", "false", "")
            if upload_flag:
                print("\n🚀 UPLOAD_TO_YT=1 → Starting YouTube upload…")

                LongUploader, import_path = _import_long_uploader()
                if LongUploader is None:
                    print("[YouTube] ❌ No uploader module found. "
                          "Ensure file exists at autoshorts/upload/youtube_uploader.py "
                          "and that autoshorts/upload/__init__.py is present.")
                else:
                    try:
                        uploader = LongUploader()
                        # Chapters + audio durations metadata (varsa) eklensin
                        script = metadata.get("script", {}) or {}
                        chapters = script.get("chapters")
                        audio_durations = script.get("audio_durations")

                        yt_visibility = os.getenv("VISIBILITY", "public")
                        topic_env = os.getenv("TOPIC")

                        vid = uploader.upload(
                            video_path=destination_path,
                            title=metadata.get("title", "Untitled"),
                            description=metadata.get("description", ""),
                            tags=metadata.get("tags"),
                            category_id="27",
                            privacy_status=yt_visibility,
                            topic=topic_env,
                            chapters=chapters,
                            audio_durations=audio_durations
                        )
                        print(f"📺 YouTube Video ID: {vid}")
                    except Exception as e:
                        print(f"[YouTube] ❌ Upload failed: {e}")
            else:
                print("\n⏭️ UPLOAD_TO_YT is disabled; skipping YouTube upload.")

            return 0
        else:
            print("\n" + "=" * 60)
            print("❌ Video generation failed")
            print("=" * 60)
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        return 130
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ ERROR: {e}")
        print("=" * 60)
        
        # Always print full traceback for debugging
        import traceback
        print("\n[DEBUG] Full traceback:")
        traceback.print_exc()
        
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
