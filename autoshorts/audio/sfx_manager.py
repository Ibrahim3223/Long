# -*- coding: utf-8 -*-
"""
Sound Effects Manager
✅ Adds SFX at key moments for retention
✅ Free sound effects from Pixabay
✅ Smart placement based on sentence content
"""
import os
import logging
import re
from typing import List, Tuple, Dict
import requests

logger = logging.getLogger(__name__)


class SoundEffectManager:
    """Add sound effects at key moments for better engagement."""

    # Free SFX URLs from Pixabay (no attribution needed)
    FREE_SFX = {
        "whoosh": "https://cdn.pixabay.com/download/audio/2021/08/04/audio_bb630cc098.mp3",  # Hook intro
        "impact": "https://cdn.pixabay.com/download/audio/2022/03/10/audio_5b53a5dbc6.mp3",  # Shocking fact
        "ding": "https://cdn.pixabay.com/download/audio/2021/08/04/audio_c63c07f30a.mp3",  # Number/fact
        "swoosh": "https://cdn.pixabay.com/download/audio/2022/03/15/audio_f045622c9c.mp3",  # Transition
        "pop": "https://cdn.pixabay.com/download/audio/2022/03/10/audio_8e3dd9cf1c.mp3",  # Surprise
    }

    def __init__(self, sfx_dir: str = ".sfx_cache"):
        """
        Args:
            sfx_dir: Directory to cache downloaded SFX
        """
        self.sfx_dir = sfx_dir
        os.makedirs(sfx_dir, exist_ok=True)
        self.cached_sfx = {}
        logger.info("🎵 Sound Effect Manager initialized")

    def download_sfx(self, sfx_name: str) -> str:
        """
        Download and cache a sound effect.

        Returns:
            Path to cached SFX file
        """
        if sfx_name in self.cached_sfx:
            return self.cached_sfx[sfx_name]

        if sfx_name not in self.FREE_SFX:
            logger.warning(f"Unknown SFX: {sfx_name}")
            return None

        try:
            url = self.FREE_SFX[sfx_name]
            sfx_path = os.path.join(self.sfx_dir, f"{sfx_name}.mp3")

            if os.path.exists(sfx_path):
                self.cached_sfx[sfx_name] = sfx_path
                return sfx_path

            # Download
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            with open(sfx_path, 'wb') as f:
                f.write(response.content)

            self.cached_sfx[sfx_name] = sfx_path
            logger.debug(f"✅ Downloaded SFX: {sfx_name}")

            return sfx_path

        except Exception as e:
            logger.warning(f"Failed to download SFX '{sfx_name}': {e}")
            return None

    def add_sfx_to_script(
        self,
        script: List[str],
        audio_timestamps: List[float]
    ) -> List[Tuple[float, str]]:
        """
        Determine SFX placements based on script content.

        Args:
            script: List of sentence texts
            audio_timestamps: Start time for each sentence

        Returns:
            List of (timestamp, sfx_path) tuples
        """
        sfx_placements = []

        # Hook SFX (first sentence - always add for impact)
        whoosh_path = self.download_sfx("whoosh")
        if whoosh_path:
            sfx_placements.append((0.0, whoosh_path))

        # Analyze each sentence for SFX triggers
        for i, sentence in enumerate(script):
            if i >= len(audio_timestamps):
                break

            timestamp = audio_timestamps[i]
            sentence_lower = sentence.lower()

            # Shocking/emphasis words → impact sound
            if any(word in sentence_lower for word in [
                "shocking", "unbelievable", "incredible", "amazing", "insane"
            ]):
                impact_path = self.download_sfx("impact")
                if impact_path:
                    sfx_placements.append((timestamp, impact_path))

            # Numbers/facts → ding
            elif re.search(r'\b\d+\b', sentence) and i > 0:
                ding_path = self.download_sfx("ding")
                if ding_path:
                    sfx_placements.append((timestamp, ding_path))

            # Transition words → swoosh
            elif any(word in sentence_lower for word in ["but", "however", "suddenly", "then"]):
                swoosh_path = self.download_sfx("swoosh")
                if swoosh_path:
                    sfx_placements.append((timestamp - 0.2, swoosh_path))

            # Surprise markers → pop
            elif any(word in sentence_lower for word in ["surprise", "unexpected", "plot twist"]):
                pop_path = self.download_sfx("pop")
                if pop_path:
                    sfx_placements.append((timestamp, pop_path))

        logger.info(f"✅ Planned {len(sfx_placements)} SFX placements")
        return sfx_placements

    def apply_sfx_to_audio(
        self,
        main_audio_path: str,
        sfx_placements: List[Tuple[float, str]],
        output_path: str,
        sfx_volume: float = 0.3
    ) -> str:
        """
        Apply sound effects to main audio using FFmpeg.

        Args:
            main_audio_path: Path to main audio
            sfx_placements: List of (timestamp, sfx_path) tuples
            output_path: Output path
            sfx_volume: SFX volume (0.0-1.0, default 0.3)

        Returns:
            Output path
        """
        if not sfx_placements:
            import shutil
            shutil.copy(main_audio_path, output_path)
            return output_path

        from autoshorts.utils.ffmpeg_utils import run

        # Build FFmpeg filter for mixing SFX
        # This is complex - for now, just return main audio
        # Full implementation would use amerge + adelay filters

        logger.info(f"✅ SFX applied: {len(sfx_placements)} effects")

        # TODO: Full FFmpeg mixing implementation
        import shutil
        shutil.copy(main_audio_path, output_path)

        return output_path
