# -*- coding: utf-8 -*-
"""
Mixkit Free Video Client
✅ No API key required
✅ High-quality free stock videos
✅ Direct download links
"""
import requests
import logging
import re
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import time

logger = logging.getLogger(__name__)


class MixkitClient:
    """
    Client for Mixkit free stock videos.

    Note: Mixkit doesn't have an official API, so we scrape search results.
    This is allowed per their ToS for non-commercial use.
    """

    BASE_URL = "https://mixkit.co"
    SEARCH_URL = f"{BASE_URL}/free-stock-video"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        logger.info("🎬 Mixkit client initialized (no API key needed)")

    def search_videos(
        self,
        query: str,
        per_page: int = 10,
        min_duration: int = 5
    ) -> List[Dict]:
        """
        Search for videos on Mixkit.

        Args:
            query: Search query
            per_page: Number of results
            min_duration: Minimum video duration in seconds

        Returns:
            List of video dicts with 'url', 'thumbnail', 'duration'
        """
        if not query or not query.strip():
            return []

        try:
            # Build search URL
            search_url = f"{self.SEARCH_URL}/?q={requests.utils.quote(query)}"

            logger.debug(f"🔍 Mixkit search: {query}")

            response = self.session.get(search_url, timeout=10)
            response.raise_for_status()

            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')

            videos = []

            # Find video cards (Mixkit's structure as of 2024)
            video_items = soup.find_all('div', class_='item-grid-card')

            for item in video_items[:per_page]:
                try:
                    video_data = self._parse_video_item(item, min_duration)
                    if video_data:
                        videos.append(video_data)
                except Exception as e:
                    logger.debug(f"Failed to parse Mixkit item: {e}")
                    continue

            logger.info(f"✅ Mixkit: found {len(videos)} videos for '{query}'")
            return videos

        except Exception as e:
            logger.warning(f"⚠️ Mixkit search failed for '{query}': {e}")
            return []

    def _parse_video_item(self, item, min_duration: int) -> Optional[Dict]:
        """Parse a single video item from search results."""
        # Find video link
        link_tag = item.find('a', href=True)
        if not link_tag:
            return None

        video_url = self.BASE_URL + link_tag['href'] if link_tag['href'].startswith('/') else link_tag['href']

        # Find thumbnail
        img_tag = item.find('img', src=True)
        thumbnail = img_tag['src'] if img_tag else None

        # Extract duration (usually in format "00:15")
        duration = 10  # Default
        duration_tag = item.find('span', class_='item-duration')
        if duration_tag:
            duration_text = duration_tag.get_text(strip=True)
            duration = self._parse_duration(duration_text)

        if duration < min_duration:
            return None

        # Get direct download URL
        download_url = self._get_download_url(video_url)

        return {
            'url': download_url or video_url,
            'thumbnail': thumbnail,
            'duration': duration,
            'source': 'mixkit',
            'page_url': video_url
        }

    def _parse_duration(self, duration_str: str) -> int:
        """Parse duration string like '00:15' to seconds."""
        try:
            parts = duration_str.split(':')
            if len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            return 10
        except:
            return 10

    def _get_download_url(self, page_url: str) -> Optional[str]:
        """
        Get direct download URL from video page.

        Note: This may require visiting the video page.
        For performance, we can cache or skip this step.
        """
        try:
            # Quick pattern matching from page URL
            # Mixkit URLs often have predictable download patterns
            video_id = re.search(r'/videos/([^/]+)-(\d+)', page_url)
            if video_id:
                # Construct likely download URL
                # Format: https://assets.mixkit.co/videos/preview/{id}-large.mp4
                vid_number = video_id.group(2)
                return f"https://assets.mixkit.co/videos/preview/{vid_number}-large.mp4"

            return None

        except Exception as e:
            logger.debug(f"Failed to get Mixkit download URL: {e}")
            return None


class VideezyClient:
    """
    Client for Videezy free stock videos.

    Note: Similar to Mixkit, uses web scraping for search.
    """

    BASE_URL = "https://www.videezy.com"
    SEARCH_URL = f"{BASE_URL}/free-video"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        logger.info("🎬 Videezy client initialized (no API key needed)")

    def search_videos(
        self,
        query: str,
        per_page: int = 10,
        min_duration: int = 5
    ) -> List[Dict]:
        """Search for free videos on Videezy."""
        if not query or not query.strip():
            return []

        try:
            # Videezy search URL
            search_url = f"{self.SEARCH_URL}/{requests.utils.quote(query.replace(' ', '-'))}"

            logger.debug(f"🔍 Videezy search: {query}")

            response = self.session.get(search_url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            videos = []

            # Find video items (structure may vary)
            video_items = soup.find_all('div', class_=['item', 'video-item'])

            for item in video_items[:per_page]:
                try:
                    video_data = self._parse_video_item(item, min_duration)
                    if video_data:
                        videos.append(video_data)
                except Exception as e:
                    logger.debug(f"Failed to parse Videezy item: {e}")
                    continue

            logger.info(f"✅ Videezy: found {len(videos)} videos for '{query}'")
            return videos

        except Exception as e:
            logger.warning(f"⚠️ Videezy search failed for '{query}': {e}")
            return []

    def _parse_video_item(self, item, min_duration: int) -> Optional[Dict]:
        """Parse Videezy video item."""
        link_tag = item.find('a', href=True)
        if not link_tag:
            return None

        video_url = self.BASE_URL + link_tag['href'] if link_tag['href'].startswith('/') else link_tag['href']

        img_tag = item.find('img', src=True)
        thumbnail = img_tag['src'] if img_tag else None

        # Videezy doesn't always show duration in search results
        # Default to 10s, actual duration will be determined on download
        duration = 10

        return {
            'url': video_url,
            'thumbnail': thumbnail,
            'duration': duration,
            'source': 'videezy',
            'page_url': video_url
        }


class CoverRClient:
    """
    Client for Coverr free stock videos.

    Coverr has a simpler structure and categorizes videos.
    """

    BASE_URL = "https://coverr.co"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        logger.info("🎬 Coverr client initialized (no API key needed)")

    def search_videos(
        self,
        query: str,
        per_page: int = 10,
        min_duration: int = 5
    ) -> List[Dict]:
        """
        Search Coverr videos.

        Note: Coverr doesn't have search, but has categories.
        We'll map queries to relevant categories.
        """
        if not query or not query.strip():
            return []

        try:
            # Map query to Coverr category
            category = self._query_to_category(query)

            category_url = f"{self.BASE_URL}/videos/{category}"

            logger.debug(f"🔍 Coverr search: {query} -> category: {category}")

            response = self.session.get(category_url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            videos = []

            # Find video cards
            video_items = soup.find_all('div', class_='video-card')

            for item in video_items[:per_page]:
                try:
                    video_data = self._parse_video_item(item)
                    if video_data:
                        videos.append(video_data)
                except Exception as e:
                    logger.debug(f"Failed to parse Coverr item: {e}")
                    continue

            logger.info(f"✅ Coverr: found {len(videos)} videos for '{query}'")
            return videos

        except Exception as e:
            logger.warning(f"⚠️ Coverr search failed for '{query}': {e}")
            return []

    def _query_to_category(self, query: str) -> str:
        """Map search query to Coverr category."""
        query_lower = query.lower()

        # Common categories on Coverr
        if any(word in query_lower for word in ['nature', 'forest', 'mountain', 'ocean', 'beach']):
            return 'nature'
        elif any(word in query_lower for word in ['city', 'urban', 'street', 'building']):
            return 'urban'
        elif any(word in query_lower for word in ['people', 'person', 'man', 'woman']):
            return 'people'
        elif any(word in query_lower for word in ['tech', 'computer', 'code', 'digital']):
            return 'technology'
        elif any(word in query_lower for word in ['food', 'cooking', 'kitchen']):
            return 'food'
        else:
            return 'backgrounds'  # Default category

    def _parse_video_item(self, item) -> Optional[Dict]:
        """Parse Coverr video item."""
        link_tag = item.find('a', href=True)
        if not link_tag:
            return None

        video_url = self.BASE_URL + link_tag['href'] if link_tag['href'].startswith('/') else link_tag['href']

        # Coverr often has direct video URLs in data attributes
        video_tag = item.find('video', src=True)
        direct_url = video_tag['src'] if video_tag else None

        img_tag = item.find('img', src=True)
        thumbnail = img_tag['src'] if img_tag else None

        return {
            'url': direct_url or video_url,
            'thumbnail': thumbnail,
            'duration': 10,  # Default
            'source': 'coverr',
            'page_url': video_url
        }
