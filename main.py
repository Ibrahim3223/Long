#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for autoshorts.
Run this file to generate a YouTube Video.
✅ WITH THUMBNAIL SUPPORT
✅ WITH MODE ENV VAR SETUP
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
    """Main entry point"""
    print("=" * 60)
    print("  YouTube Video Generator v2.0")
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
        temp_dir = os.path.join(tempfile.gettempdir(), f"autoshorts_{channel_name}")
        os.makedirs(temp_dir, exist_ok=True)
        print(f"📁 Temp dir: {temp_dir}")
        
        print("\n🔧 Creating orchestrator...")
        
        # Initialize orchestrator
        orchestrator = ShortsOrchestrator(
            channel_id=channel_name,
            temp_dir=temp_dir,
            api_key=settings.GEMINI_API_KEY,
            pexels_key=settings.PEXELS_API_KEY
        )
        
        print("\n🎬 Starting video generation...\n")
        
        # Generate video using channel topic
        topic_prompt = channel_settings.get("CHANNEL_TOPIC", "Create an interesting video")
        video_path, metadata = orchestrator.produce_video(topic_prompt)
        
        if video_path and metadata:
            print("\n" + "=" * 60)
            print(f"✅ SUCCESS!")
            print(f"📹 Video: {video_path}")
            print(f"📝 Title: {metadata.get('title', 'N/A')}")
            print(f"🏷️ Tags: {len(metadata.get('tags', []))} tags")
            print("=" * 60)
            
            # Set output for GitHub Actions
            if os.getenv("GITHUB_OUTPUT"):
                with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                    f.write(f"video_path={video_path}\n")
                    f.write(f"title={metadata.get('title', '')}\n")
                    f.write(f"description={metadata.get('description', '')}\n")
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
