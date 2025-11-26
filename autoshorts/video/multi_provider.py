# -*- coding: utf-8 -*-
"""
Multi-Provider Video Aggregator
✅ Combines Pexels, Pixabay, Mixkit, Videezy, Coverr
✅ Smart fallback chain
✅ Aggregates results for best coverage
"""
import logging
from typing import List, Dict, Optional
from autoshorts.video.pexels_client import PexelsClient
from autoshorts.video.pixabay_client import PixabayClient

logger = logging.getLogger(__name__)

# Import free providers (with fallback)
try:
    from autoshorts.video.mixkit_client import MixkitClient, VideezyClient, CoverRClient
    FREE_PROVIDERS_AVAILABLE = True
except ImportError:
    logger.warning("Free video providers not available (missing BeautifulSoup)")
    FREE_PROVIDERS_AVAILABLE = False


class MultiProviderVideoSearch:
    """Aggregate video search across multiple free providers."""

    def __init__(self, pexels_key: Optional[str] = None, pixabay_key: Optional[str] = None):
        self.providers = []

        # Pexels (primary - best quality)
        if pexels_key:
            try:
                self.providers.append(('pexels', PexelsClient(pexels_key)))
                logger.info("✅ Pexels provider initialized")
            except Exception as e:
                logger.warning(f"Pexels init failed: {e}")

        # Pixabay (secondary - good variety)
        if pixabay_key:
            try:
                self.providers.append(('pixabay', PixabayClient(pixabay_key)))
                logger.info("✅ Pixabay provider initialized")
            except Exception as e:
                logger.warning(f"Pixabay init failed: {e}")

        # Free providers (no API key needed)
        if FREE_PROVIDERS_AVAILABLE:
            try:
                self.providers.append(('mixkit', MixkitClient()))
                self.providers.append(('videezy', VideezyClient()))
                self.providers.append(('coverr', CoverRClient()))
                logger.info("✅ Free providers initialized (Mixkit, Videezy, Coverr)")
            except Exception as e:
                logger.warning(f"Free providers init failed: {e}")

        logger.info(f"🎬 Multi-provider search: {len(self.providers)} providers active")

    def search(self, query: str, min_duration: int = 5, max_results: int = 10) -> List[Dict]:
        """
        Search across all providers with smart aggregation.

        Returns:
            Aggregated list of videos from all sources
        """
        all_videos = []

        for provider_name, provider in self.providers:
            try:
                videos = provider.search_videos(query, per_page=max_results, min_duration=min_duration)

                # Add provider name to each video
                for video in videos:
                    video['provider'] = provider_name

                all_videos.extend(videos)

                logger.debug(f"{provider_name}: {len(videos)} videos")

            except Exception as e:
                logger.warning(f"{provider_name} search failed: {e}")
                continue

        # Dedup by URL
        seen_urls = set()
        unique_videos = []
        for video in all_videos:
            url = video.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_videos.append(video)

        logger.info(f"✅ Multi-provider search: {len(unique_videos)} unique videos for '{query}'")

        return unique_videos[:max_results]
