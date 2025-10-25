#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for autoshorts.
Generates the video and (optionally) uploads to YouTube.
"""
import sys
import os
import re
import json
import shutil
import logging
import tempfile
from datetime import datetime

# ---------------- cache clean ----------------
def clear_cache():
    project_root = os.path.dirname(os.path.abspath(__file__))
    autoshorts_path = os.path.join(project_root, 'autoshorts')
    if os.path.exists(autoshorts_path):
        for root, dirs, files in os.walk(autoshorts_path):
            if '__pycache__' in dirs:
                cache_dir = os.path.join(root, '__pycache__')
                try:
                    shutil.rmtree(cache_dir)
                    print(f"[CACHE] Cleared: {cache_dir}")
                except Exception as e:
                    print(f"[CACHE] Warning: Could not clear {cache_dir}: {e}")
            for file in files:
                if file.endswith('.pyc'):
                    pyc_file = os.path.join(root, file)
                    try:
                        os.remove(pyc_file)
                        print(f"[CACHE] Removed: {pyc_file}")
                    except Exception as e:
                        print(f"[CACHE] Warning: Could not remove {pyc_file}: {e}")

print("[CACHE] Clearing Python cache...")
clear_cache()
print("[CACHE] Cache cleared successfully\n")

# ---------------- sys.path ----------------
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"[DEBUG] Python path: {project_root}")
print(f"[DEBUG] Checking autoshorts module...")

autoshorts_path = os.path.join(project_root, 'autoshorts')
if not os.path.exists(autoshorts_path):
    print(f"❌ ERROR: autoshorts directory not found at {autoshorts_path}")
    sys.exit(1)

init_file = os.path.join(autoshorts_path, '__init__.py')
if not os.path.exists(init_file):
    print(f"❌ ERROR: autoshorts/__init__.py not found")
    sys.exit(1)

print(f"✅ autoshorts module found at {autoshorts_path}")

# ---------------- imports ----------------
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

# ---------------- helpers ----------------
def _safe_slug(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value)
    return value.strip("-") or "channel"

def _truthy_env(name: str, default: str = "") -> bool:
    val = (os.getenv(name, default) or "").strip().lower()
    return val in ("1", "true", "yes", "on")

def _build_description_fallback(meta: dict) -> str:
    parts = []
    title = (meta.get("title") or "").strip()
    desc = (meta.get("description") or "").strip()
    hook = (meta.get("hook") or "").strip()
    tags = meta.get("tags") or []
    if desc:
        parts.append(desc)
    elif hook:
        parts.append(hook)
    if title and title not in parts:
        parts.append(f"\n— {title}")
    if tags:
        hashtags = ["#" + t.replace(" ", "")[:28] for t in tags[:10]]
        parts.append("\n" + " ".join(hashtags))
    parts.append("\n\nGenerated with autoshorts.")
    return "\n".join(p for p in parts if p).strip()

# ---------------- main ----------------
def main():
    print("=" * 60)
    print("  YouTube Shorts/Long Generator")
    print("=" * 60)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        channel_name = os.environ.get("CHANNEL_NAME") or os.environ.get("ENV") or "default"
        print(f"\n📺 Channel: {channel_name}")

        channel_settings = apply_channel_settings(channel_name)

        temp_dir = os.path.join(tempfile.gettempdir(), f"autoshorts_{channel_name}")
        os.makedirs(temp_dir, exist_ok=True)
        print(f"📁 Temp dir: {temp_dir}")

        print("\n🔧 Creating orchestrator...")
        orchestrator = ShortsOrchestrator(
            channel_id=channel_name,
            temp_dir=temp_dir,
            api_key=settings.GEMINI_API_KEY,
            pexels_key=settings.PEXELS_API_KEY
        )

        print("\n🎬 Starting video generation...\n")
        topic_prompt = channel_settings.get("CHANNEL_TOPIC", "Create an interesting video")
        video_path, metadata = orchestrator.produce_video(topic_prompt)

        if not (video_path and metadata):
            print("\n" + "=" * 60)
            print("❌ Video generation failed")
            print("=" * 60)
            return 1

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

        # ---------------- YouTube Upload (RE-ENABLED) ----------------
        if _truthy_env("UPLOAD_TO_YT", os.getenv("UPLOAD_TO_YT", "")):
            print("\n🚀 UPLOAD_TO_YT=1 → Starting YouTube upload…")

            visibility = (os.getenv("VISIBILITY") or "unlisted").strip().lower()
            if visibility not in ("public", "private", "unlisted"):
                visibility = "unlisted"
            category_id = os.getenv("YT_CATEGORY_ID", "27")  # Education default

            title = metadata.get("title") or "Untitled"
            description = metadata.get("description") or ""
            tags = metadata.get("tags") or []

            # Chapters (long-form uploader bunları destekliyor)
            script = metadata.get("script") or {}
            chapters = script.get("chapters")
            audio_durations = script.get("audio_durations")  # varsa kullanır; yoksa None

            video_id = None

            # 1) Önce long-form uploader’ı dene (senin dosyan)
            try:
                from autoshorts.uploader.youtube_long import YouTubeUploader as LongUploader
                logger = logging.getLogger(__name__)
                logger.info("[YouTube] Using long-form uploader with chapter support")
                yt = LongUploader()
                video_id = yt.upload(
                    video_path=destination_path,
                    title=title,
                    description=description or _build_description_fallback(metadata),
                    tags=tags,
                    category_id=category_id,
                    privacy_status=visibility,
                    topic=topic_prompt,
                    chapters=chapters,
                    audio_durations=audio_durations
                )
            except Exception as e:
                print(f"[YouTube] Long-form uploader failed or not found: {e}")

            # 2) Gerekirse kısa uploader’a (opsiyonel) düş
            if not video_id:
                try:
                    # Eğer projende kısa uploader varsa kullan; yoksa bu blok atlanır.
                    from autoshorts.uploader.youtube import YouTubeUploader as ShortUploader, UploadOptions
                    print("[YouTube] Falling back to short uploader")
                    yt2 = ShortUploader.from_env() if hasattr(ShortUploader, "from_env") else ShortUploader()
                    video_id = yt2.upload(
                        destination_path,
                        UploadOptions(
                            title=title,
                            description=description or _build_description_fallback(metadata),
                            tags=tags,
                            category_id=category_id,
                            privacy_status=visibility,
                            made_for_kids=False
                        )
                    ) if hasattr(yt2, "upload") else None
                except Exception as e:
                    print(f"[YouTube] Short uploader not available: {e}")

            if video_id:
                print(f"✅ Uploaded to YouTube. Video ID: {video_id}")
                meta_payload["youtube_video_id"] = video_id
                with open(metadata_path, "w", encoding="utf-8") as fh:
                    json.dump(meta_payload, fh, ensure_ascii=False, indent=2)
            else:
                print("❌ YouTube upload did not complete; see logs above.")

        else:
            print("\n⏭️ UPLOAD_TO_YT disabled or not set; skipping YouTube upload.")

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        return 130

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ ERROR: {e}")
        print("=" * 60)
        import traceback
        print("\n[DEBUG] Full traceback:")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
