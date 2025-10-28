# -*- coding: utf-8 -*-
"""
State management and novelty detection for autoshorts
✅ FIXED: SQLite INTEGER overflow for hash values
"""

import os
import re
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDINGS_FILE = Path(".state/embeddings.json")
USED_ENTITIES_FILE = Path(".state/used_entities.json")
UPLOADS_FILE = Path(".state/uploads.json")

def _save_json(filepath: Path, data):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _load_json(filepath: Path, default):
    if filepath.exists():
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load {filepath}: {e}")
    return default

def _norm_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text


class StateGuard:
    """
    Manages channel state, novelty detection, and upload history.
    ✅ Fixed SQLite INTEGER overflow by using string hashes
    """

    def __init__(self, channel: str, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.channel = channel
        self.model = None
        try:
            self.model = SentenceTransformer(embedding_model)
        except Exception as e:
            logging.warning(f"[state_guard] Could not load embedding model: {e}")

        # Load state files
        self.embeddings = _load_json(EMBEDDINGS_FILE, {})
        self.used_entities = _load_json(USED_ENTITIES_FILE, {})
        self.uploads = _load_json(UPLOADS_FILE, [])

        if self.channel not in self.embeddings:
            self.embeddings[self.channel] = {"vectors": [], "texts": []}
        if self.channel not in self.used_entities:
            self.used_entities[self.channel] = {}

    def check_entity_used(self, entity: str, cooldown_days: int = 30) -> bool:
        """Check if entity was used recently."""
        entity = _norm_text(entity)
        if entity not in self.used_entities[self.channel]:
            return False
        last_used = datetime.fromisoformat(self.used_entities[self.channel][entity])
        return (datetime.now() - last_used).days < cooldown_days

    def check_script_novelty(self, script: List[str], threshold: float = 0.75) -> Tuple[bool, float]:
        """
        Check if script is sufficiently different from previous scripts.
        Returns (is_novel, max_similarity)
        """
        if self.model is None:
            return True, 0.0

        script_text = _norm_text(" ".join(script))
        if not script_text:
            return True, 0.0

        try:
            new_vec = self.model.encode(script_text)
            channel_vecs = self.embeddings[self.channel]["vectors"]

            if not channel_vecs:
                return True, 0.0

            similarities = [
                float(np.dot(new_vec, old_vec) / (np.linalg.norm(new_vec) * np.linalg.norm(old_vec)))
                for old_vec in channel_vecs
            ]
            max_sim = max(similarities) if similarities else 0.0

            is_novel = max_sim < threshold
            return is_novel, max_sim

        except Exception as e:
            logging.warning(f"[state_guard] Novelty check error: {e}")
            return True, 0.0

    def make_content_hash(self, script_text: str, video_paths: List[str], audio_path: Optional[str]) -> str:
        """
        Generate a unique hash for content.
        ✅ Returns hex string instead of integer to avoid SQLite overflow
        """
        content = script_text
        for vp in video_paths:
            if os.path.exists(vp):
                content += str(os.path.getsize(vp))
        if audio_path and os.path.exists(audio_path):
            content += str(os.path.getsize(audio_path))
        
        # ✅ Return hex string (not int) to avoid SQLite INTEGER overflow
        return hashlib.sha256(content.encode()).hexdigest()

    def record_upload(self, video_id: str, content: Dict):
        """
        Record a successful upload.
        Called by orchestrator after successful YouTube upload.
        
        Args:
            video_id: YouTube video ID
            content: Content dict with metadata, script, etc.
        """
        try:
            # Extract key information
            title = content.get("metadata", {}).get("title", "")
            script_text = " ".join([
                content.get("hook", ""),
                *content.get("script", []),
                content.get("cta", "")
            ])
            
            # Generate content hash (hex string)
            content_hash = self.make_content_hash(
                script_text=script_text,
                video_paths=[],
                audio_path=None
            )
            
            # Extract main entity from title or first search query
            entity = title
            if not entity and content.get("search_queries"):
                entity = content["search_queries"][0]
            
            # Record in state
            self.mark_uploaded(
                entity=entity,
                script_text=script_text,
                content_hash=content_hash,
                video_path=f"youtube:{video_id}",
                title=title
            )
            
            logging.info(f"[state_guard] Recorded upload: {video_id} - {title}")
            
        except Exception as e:
            logging.error(f"[state_guard] Failed to record upload: {e}")
            import traceback
            logging.debug(traceback.format_exc())

    def mark_uploaded(self, entity: str, script_text: str, content_hash: str,
                      video_path: str, title: str = ""):
        """Mark content as uploaded and save to state."""
        # entities
        self.used_entities[self.channel][entity] = datetime.now().isoformat()
        _save_json(USED_ENTITIES_FILE, self.used_entities)

        # embeddings (script)
        if self.model is not None:
            try:
                vec = self.model.encode(_norm_text(script_text)).tolist()
                self.embeddings[self.channel]["vectors"].append(vec)
                self.embeddings[self.channel]["texts"].append(_norm_text(script_text)[:200])
                _save_json(EMBEDDINGS_FILE, self.embeddings)
            except Exception as e:
                logging.warning(f"[state_guard] embed save error: {e}")

        # uploads
        self.uploads.append({
            "channel": self.channel,
            "entity": entity,
            "title": title,
            "content_hash": content_hash,  # ✅ Now a hex string, not integer
            "video_path": video_path,
            "uploaded_at": datetime.now().isoformat()
        })
        _save_json(UPLOADS_FILE, self.uploads)

    def get_recent_uploads(self, days: int = 7) -> List[Dict]:
        """Get uploads from last N days."""
        cutoff = datetime.now() - timedelta(days=days)
        return [
            u for u in self.uploads
            if u.get("channel") == self.channel
            and datetime.fromisoformat(u["uploaded_at"]) > cutoff
        ]

    def cleanup_old_data(self, days: int = 90):
        """Remove data older than N days."""
        cutoff = datetime.now() - timedelta(days=days)
        
        # Cleanup entities
        self.used_entities[self.channel] = {
            k: v for k, v in self.used_entities[self.channel].items()
            if datetime.fromisoformat(v) > cutoff
        }
        _save_json(USED_ENTITIES_FILE, self.used_entities)
        
        # Cleanup uploads
        self.uploads = [
            u for u in self.uploads
            if datetime.fromisoformat(u["uploaded_at"]) > cutoff
        ]
        _save_json(UPLOADS_FILE, self.uploads)
        
        logging.info(f"[state_guard] Cleaned up data older than {days} days")
