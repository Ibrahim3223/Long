# FILE: autoshorts/video/pexels_client.py
# -*- coding: utf-8 -*-
"""
Pexels API client for fetching stock videos and photos.
Enhanced with:
- Advanced rate limiting (200 requests per minute)
- Intelligent caching system
- Exponential backoff retry mechanism
- Request history tracking
- Automatic quota management
"""
import requests
import logging
import time
import json
import hashlib
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from collections import deque
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Advanced rate limiter for Pexels API.
    Enforces 200 requests per minute limit with intelligent throttling.
    """
    
    def __init__(self, max_requests: int = 200, time_window: int = 60, min_interval: float = 0.5):
        """
        Args:
            max_requests: Maximum requests allowed in time window (default: 200)
            time_window: Time window in seconds (default: 60)
            min_interval: Minimum seconds between requests (default: 0.5)
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.min_interval = min_interval
        self.request_times = deque(maxlen=max_requests)
        self.last_request_time = 0
        
    def wait(self):
        """Wait if necessary to respect rate limits."""
        now = time.time()
        
        # Enforce minimum interval between requests
        time_since_last = now - self.last_request_time
        if time_since_last < self.min_interval:
            wait_time = self.min_interval - time_since_last
            logger.debug(f"⏳ Throttling: waiting {wait_time:.2f}s")
            time.sleep(wait_time)
            now = time.time()
        
        # Enforce time window limit
        if len(self.request_times) >= self.max_requests:
            oldest_request = self.request_times[0]
            time_elapsed = now - oldest_request
            
            if time_elapsed < self.time_window:
                wait_time = self.time_window - time_elapsed + 1  # +1 for safety margin
                logger.warning(f"⏳ Rate limit reached: waiting {wait_time:.1f}s ({len(self.request_times)} requests in last {time_elapsed:.1f}s)")
                time.sleep(wait_time)
                now = time.time()
        
        # Record this request
        self.request_times.append(now)
        self.last_request_time = now
    
    def get_stats(self) -> Dict:
        """Get current rate limiter statistics."""
        now = time.time()
        recent_requests = sum(1 for t in self.request_times if now - t < self.time_window)
        
        return {
            "total_requests": len(self.request_times),
            "requests_last_minute": recent_requests,
            "remaining_quota": self.max_requests - recent_requests,
            "time_until_reset": max(0, self.time_window - (now - self.request_times[0])) if self.request_times else 0
        }


class PexelsCache:
    """
    Intelligent caching system for Pexels API responses.
    Prevents duplicate requests and saves API quota.
    """
    
    def __init__(self, cache_dir: str = ".pexels_cache", ttl_hours: int = 24):
        """
        Args:
            cache_dir: Directory to store cache files
            ttl_hours: Time to live for cached data in hours
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.ttl = timedelta(hours=ttl_hours)
        
        # Memory cache for faster access
        self._memory_cache: Dict[str, Tuple[Dict, datetime]] = {}
        
        logger.info(f"💾 Cache initialized: {self.cache_dir} (TTL: {ttl_hours}h)")
    
    def _get_cache_key(self, query: str, orientation: str = "landscape", page: int = 1) -> str:
        """Generate unique cache key from search parameters."""
        data = f"{query.lower()}_{orientation}_{page}"
        return hashlib.md5(data.encode()).hexdigest()
    
    def get(self, query: str, orientation: str = "landscape", page: int = 1) -> Optional[Dict]:
        """Retrieve cached results if available and not expired."""
        cache_key = self._get_cache_key(query, orientation, page)
        
        # Check memory cache first
        if cache_key in self._memory_cache:
            data, timestamp = self._memory_cache[cache_key]
            if datetime.now() - timestamp < self.ttl:
                logger.debug(f"💾 Memory cache HIT: {query} (page {page})")
                return data
            else:
                # Expired, remove from memory
                del self._memory_cache[cache_key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Check if expired
            timestamp = datetime.fromisoformat(cache_data['timestamp'])
            if datetime.now() - timestamp > self.ttl:
                logger.debug(f"💾 Cache EXPIRED: {query} (page {page})")
                cache_file.unlink()  # Delete expired cache
                return None
            
            # Update memory cache
            data = cache_data['data']
            self._memory_cache[cache_key] = (data, timestamp)
            
            logger.debug(f"💾 Disk cache HIT: {query} (page {page})")
            return data
            
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return None
    
    def set(self, query: str, data: Dict, orientation: str = "landscape", page: int = 1):
        """Store results in cache."""
        cache_key = self._get_cache_key(query, orientation, page)
        timestamp = datetime.now()
        
        # Update memory cache
        self._memory_cache[cache_key] = (data, timestamp)
        
        # Update disk cache
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            cache_data = {
                'query': query,
                'orientation': orientation,
                'page': page,
                'timestamp': timestamp.isoformat(),
                'data': data
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
            
            logger.debug(f"💾 Cached: {query} (page {page})")
            
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    def clear_expired(self):
        """Remove expired cache files."""
        removed = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                timestamp = datetime.fromisoformat(cache_data['timestamp'])
                if datetime.now() - timestamp > self.ttl:
                    cache_file.unlink()
                    removed += 1
            except Exception:
                continue
        
        if removed > 0:
            logger.info(f"💾 Removed {removed} expired cache files")
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total_files = len(list(self.cache_dir.glob("*.json")))
        memory_items = len(self._memory_cache)
        
        return {
            "disk_cache_files": total_files,
            "memory_cache_items": memory_items,
            "cache_dir": str(self.cache_dir)
        }


class PexelsClient:
    """
    Enhanced Pexels API client with advanced features:
    - Rate limiting (200 req/min)
    - Intelligent caching
    - Exponential backoff retry
    - Request history tracking
    """

    BASE_URL = "https://api.pexels.com/v1"
    VIDEO_URL = "https://api.pexels.com/videos"

    def __init__(
        self, 
        api_key: Optional[str] = None,
        enable_cache: bool = True,
        cache_ttl_hours: int = 168,
        max_retries: int = 3
    ):
        """
        Initialize Pexels client with enhanced features.
        
        Args:
            api_key: Pexels API key
            enable_cache: Enable intelligent caching
            cache_ttl_hours: Cache time to live in hours
            max_retries: Maximum retry attempts for failed requests
        """
        # Try to get API key from environment if not provided
        if not api_key:
            from autoshorts.config import settings
            api_key = settings.PEXELS_API_KEY
        
        if not api_key:
            raise ValueError("PEXELS_API_KEY is required")

        self.api_key = api_key
        self.headers = {"Authorization": self.api_key}
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(
            max_requests=200,  # Pexels free tier limit
            time_window=60,    # 1 minute
            min_interval=0.5   # Minimum 0.5s between requests
        )
        
        # Initialize cache
        self.cache_enabled = enable_cache
        if self.cache_enabled:
            self.cache = PexelsCache(ttl_hours=cache_ttl_hours)
            # Clear expired cache on init
            self.cache.clear_expired()
        
        # Setup session with retry strategy
        self.session = self._create_session_with_retries(max_retries)
        self.session.headers.update(self.headers)
        
        # Request tracking
        self.total_requests = 0
        self.cache_hits = 0
        self.api_errors = 0
        
        logger.info("🎬 PexelsClient initialized (cache=%s, max_retries=%d)", 
                   enable_cache, max_retries)
    
    def _create_session_with_retries(self, max_retries: int) -> requests.Session:
        """Create session with exponential backoff retry strategy."""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=2,  # 1s, 2s, 4s, 8s...
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=20
        )
        
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def search_videos(
        self,
        query: str,
        per_page: int = 15,
        page: int = 1,
        orientation: str = "landscape",
        size: str = "medium",
        use_cache: bool = True
    ) -> Dict:
        """
        Search for videos on Pexels with caching and rate limiting.
        
        Args:
            query: Search query
            per_page: Results per page (1-80)
            page: Page number
            orientation: 'landscape', 'portrait', or 'square'
            size: 'large', 'medium', or 'small'
            use_cache: Use cached results if available
        
        Returns:
            Dictionary containing search results
        """
        # Check cache first
        if self.cache_enabled and use_cache:
            cached = self.cache.get(query, orientation, page)
            if cached is not None:
                self.cache_hits += 1
                logger.info(f"💾 Cache hit: '{query}' (page {page})")
                return cached
        
        # Rate limiting
        self.rate_limiter.wait()
        
        # Make API request
        url = f"{self.VIDEO_URL}/search"
        params = {
            "query": query,
            "per_page": per_page,
            "page": page,
            "orientation": orientation,
            "size": size
        }
        
        try:
            self.total_requests += 1
            logger.info(f"🔍 Pexels search: '{query}' (page {page}, orientation={orientation})")
            
            response = self.session.get(url, params=params, timeout=15)
            
            # Handle rate limiting specifically
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 10))
                logger.warning(f"⏳ Rate limited (429)! Waiting {retry_after}s...")
                time.sleep(retry_after)
                
                # Retry once after waiting
                response = self.session.get(url, params=params, timeout=15)
            
            response.raise_for_status()
            data = response.json()
            
            # Log quota info
            self._log_quota_info(response.headers)
            
            # Cache the result
            if self.cache_enabled:
                self.cache.set(query, data, orientation, page)
            
            return data
            
        except requests.exceptions.HTTPError as e:
            self.api_errors += 1
            logger.error(f"❌ Pexels API error: {e.response.status_code} {e}")
            
            if e.response.status_code == 429:
                logger.error("💡 Tip: You've hit the rate limit. Consider:")
                logger.error("   • Reducing requests per minute")
                logger.error("   • Enabling cache (already enabled)" if self.cache_enabled else "   • Enabling cache")
                logger.error("   • Upgrading to Pexels paid plan")
            
            return {"videos": [], "total_results": 0}
            
        except Exception as e:
            self.api_errors += 1
            logger.error(f"❌ Request failed: {e}")
            return {"videos": [], "total_results": 0}
    
    def _log_quota_info(self, headers: Dict):
        """Log API quota information from response headers."""
        limit = headers.get('X-Ratelimit-Limit')
        remaining = headers.get('X-Ratelimit-Remaining')
        reset = headers.get('X-Ratelimit-Reset')
        
        if limit and remaining:
            logger.debug(f"📊 API Quota: {remaining}/{limit} remaining")
            
            if reset:
                reset_time = datetime.fromtimestamp(int(reset))
                logger.debug(f"   Reset at: {reset_time.strftime('%H:%M:%S')}")
    
    def get_statistics(self) -> Dict:
        """Get comprehensive client statistics."""
        stats = {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "api_errors": self.api_errors,
            "cache_hit_rate": f"{(self.cache_hits / max(self.total_requests, 1)) * 100:.1f}%"
        }
        
        # Add rate limiter stats
        stats["rate_limiter"] = self.rate_limiter.get_stats()
        
        # Add cache stats
        if self.cache_enabled:
            stats["cache"] = self.cache.get_stats()
        
        return stats
    
    def print_statistics(self):
        """Print formatted statistics."""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("📊 PEXELS CLIENT STATISTICS")
        print("="*60)
        print(f"Total Requests:    {stats['total_requests']}")
        print(f"Cache Hits:        {stats['cache_hits']}")
        print(f"API Errors:        {stats['api_errors']}")
        print(f"Cache Hit Rate:    {stats['cache_hit_rate']}")
        print("\n📈 RATE LIMITER:")
        rl = stats['rate_limiter']
        print(f"  Requests (last min): {rl['requests_last_minute']}/{rl['total_requests']}")
        print(f"  Remaining Quota:     {rl['remaining_quota']}")
        print(f"  Time Until Reset:    {rl['time_until_reset']:.1f}s")
        
        if self.cache_enabled:
            print("\n💾 CACHE:")
            cache = stats['cache']
            print(f"  Disk Files:      {cache['disk_cache_files']}")
            print(f"  Memory Items:    {cache['memory_cache_items']}")
            print(f"  Location:        {cache['cache_dir']}")
        print("="*60 + "\n")
    
    def clear_cache(self):
        """Clear all cached data."""
        if self.cache_enabled:
            for cache_file in self.cache.cache_dir.glob("*.json"):
                cache_file.unlink()
            self.cache._memory_cache.clear()
            logger.info("💾 Cache cleared")
        else:
            logger.warning("Cache is not enabled")


# Convenience function for backward compatibility
def create_pexels_client(api_key: Optional[str] = None) -> PexelsClient:
    """Create a Pexels client with default settings."""
    return PexelsClient(api_key=api_key)


if __name__ == "__main__":
    # Test the client
    import os
    
    api_key = os.getenv("PEXELS_API_KEY")
    if not api_key:
        print("❌ Please set PEXELS_API_KEY environment variable")
        exit(1)
    
    client = PexelsClient(api_key=api_key)
    
    # Test search
    print("Testing Pexels search...")
    results = client.search_videos("ocean waves", per_page=5)
    print(f"Found {results.get('total_results', 0)} videos")
    
    # Test cache (should be instant)
    print("\nTesting cache...")
    results2 = client.search_videos("ocean waves", per_page=5)
    print(f"Second search (cached): {results2.get('total_results', 0)} videos")
    
    # Print statistics
    client.print_statistics()
