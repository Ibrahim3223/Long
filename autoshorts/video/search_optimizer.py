# -*- coding: utf-8 -*-
"""
Video Search Optimizer - Context-Aware Search with Relevance Scoring
✅ Keyword expansion with synonyms
✅ Relevance scoring for video selection
✅ Contextual query building
✅ Multi-source search coordination
"""
import re
import logging
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Keyword expansion database (synonyms and related terms)
KEYWORD_EXPANSIONS = {
    # Nature & Landscapes
    "mountain": ["peak", "summit", "alpine", "hill", "terrain"],
    "ocean": ["sea", "water", "waves", "coast", "beach"],
    "forest": ["woods", "trees", "jungle", "woodland", "nature"],
    "desert": ["sand", "dunes", "arid", "wasteland", "dry"],
    "river": ["stream", "waterway", "creek", "flow", "water"],
    "sky": ["clouds", "atmosphere", "heaven", "aerial", "air"],

    # Urban & Architecture
    "city": ["urban", "metropolis", "downtown", "skyline", "buildings"],
    "building": ["architecture", "structure", "tower", "skyscraper"],
    "street": ["road", "avenue", "boulevard", "pathway", "urban"],
    "bridge": ["crossing", "overpass", "viaduct", "structure"],

    # People & Activities
    "people": ["crowd", "person", "human", "individual", "group"],
    "working": ["office", "desk", "computer", "business", "professional"],
    "walking": ["pedestrian", "strolling", "moving", "street"],
    "running": ["jogging", "sprint", "athlete", "fitness", "sport"],
    "talking": ["conversation", "speaking", "communication", "discussion"],

    # Technology & Science
    "technology": ["innovation", "digital", "modern", "computer", "tech"],
    "science": ["research", "laboratory", "experiment", "scientific"],
    "computer": ["laptop", "technology", "digital", "screen", "device"],
    "robot": ["automation", "mechanical", "android", "machine"],
    "space": ["cosmos", "galaxy", "stars", "universe", "astronomy"],

    # Abstract & Concepts
    "ancient": ["historical", "old", "antique", "past", "heritage"],
    "modern": ["contemporary", "current", "new", "innovative"],
    "mysterious": ["enigmatic", "unknown", "secret", "hidden", "strange"],
    "powerful": ["strong", "mighty", "intense", "forceful", "dynamic"],
    "beautiful": ["stunning", "gorgeous", "attractive", "scenic", "picturesque"],

    # Animals & Nature
    "animal": ["wildlife", "creature", "beast", "fauna"],
    "bird": ["avian", "flying", "feather", "wings"],
    "fish": ["marine", "aquatic", "underwater", "ocean"],
    "dog": ["canine", "pet", "puppy", "animal"],
    "cat": ["feline", "kitten", "pet", "animal"],

    # Time & Events
    "sunset": ["dusk", "evening", "twilight", "golden hour"],
    "sunrise": ["dawn", "morning", "daybreak", "sunup"],
    "night": ["evening", "darkness", "nighttime", "nocturnal"],
    "day": ["daytime", "daylight", "afternoon", "morning"],

    # Weather & Seasons
    "rain": ["rainfall", "storm", "weather", "precipitation", "wet"],
    "snow": ["winter", "ice", "cold", "snowfall", "frozen"],
    "storm": ["thunder", "lightning", "tempest", "weather"],
    "wind": ["breeze", "gust", "air", "windy", "blow"],
}


# Context-based query templates
CONTEXT_TEMPLATES = {
    "action": [
        "{subject} {action}",
        "{action} in {location}",
        "person {action}",
    ],
    "description": [
        "{adjective} {subject}",
        "{subject} {environment}",
    ],
    "location": [
        "{location} landscape",
        "{location} view",
        "{location} scenery",
    ],
    "abstract": [
        "{concept} visualization",
        "{concept} concept",
        "abstract {concept}",
    ],
}


@dataclass
class VideoSearchQuery:
    """Enhanced search query with metadata."""
    text: str
    priority: int  # 1 (highest) to 5 (lowest)
    source: str  # "chapter", "sentence", "keyword", "expansion"
    relevance_keywords: Set[str]  # Keywords to check relevance


class VideoSearchOptimizer:
    """Optimize video search with context awareness and relevance scoring."""

    def __init__(self, use_simple_queries: bool = True):
        """
        Initialize search optimizer.

        Args:
            use_simple_queries: If True, use simple 1-2 keyword queries (better match rates).
                                If False, use complex contextual queries.
        """
        self.expansion_cache: Dict[str, List[str]] = {}
        self.use_simple_queries = use_simple_queries

        if use_simple_queries:
            logger.info("🔍 Using SIMPLE search queries (1-2 keywords for better matches)")
        else:
            logger.info("🔍 Using COMPLEX search queries (contextual phrases)")

    def build_simple_queries(
        self,
        keywords: List[str],
        sentence: str,
    ) -> List[VideoSearchQuery]:
        """
        Build SIMPLE search queries (1-2 keywords).

        User feedback: "her sahneyi basit bir kelime veya 2 kelime ile aratırsak
        daha iyi sonuçlara ulaşacağımızdan eminim"

        Args:
            keywords: Extracted keywords from sentence
            sentence: Original sentence (for context)

        Returns:
            List of simple VideoSearchQuery objects (max 5 queries, 1-2 words each)
        """
        queries: List[VideoSearchQuery] = []
        seen_queries: Set[str] = set()

        # Extract only NOUNS (most important words) from keywords
        nouns = []
        for kw in keywords[:5]:  # Top 5 keywords max
            # Filter out common verbs/adjectives (keep nouns)
            kw_lower = kw.lower()
            if kw_lower not in {"is", "are", "was", "were", "become", "has", "have", "been", "being"}:
                if len(kw) >= 3:  # Min 3 chars
                    nouns.append(kw)

        # 1. Single most important keyword (highest priority)
        if nouns:
            main_keyword = nouns[0]
            normalized = self._normalize_query(main_keyword)
            if normalized and normalized not in seen_queries:
                seen_queries.add(normalized)
                queries.append(VideoSearchQuery(
                    text=main_keyword,
                    priority=1,
                    source="simple_main",
                    relevance_keywords={main_keyword.lower()},
                ))

        # 2. Two-keyword combination (if available)
        if len(nouns) >= 2:
            combined = f"{nouns[0]} {nouns[1]}"
            normalized = self._normalize_query(combined)
            if normalized and normalized not in seen_queries:
                seen_queries.add(normalized)
                queries.append(VideoSearchQuery(
                    text=combined,
                    priority=2,
                    source="simple_combined",
                    relevance_keywords={nouns[0].lower(), nouns[1].lower()},
                ))

        # 3. Alternative single keywords (lower priority)
        for noun in nouns[1:3]:  # 2nd and 3rd keywords
            normalized = self._normalize_query(noun)
            if normalized and normalized not in seen_queries:
                seen_queries.add(normalized)
                queries.append(VideoSearchQuery(
                    text=noun,
                    priority=3,
                    source="simple_alternative",
                    relevance_keywords={noun.lower()},
                ))

        # Limit to 5 queries max (simple is better)
        queries = queries[:5]

        logger.debug(f"🔍 Simple queries: {[q.text for q in queries]}")
        return queries

    def build_search_queries(
        self,
        sentence: str,
        keywords: List[str],
        chapter_title: Optional[str] = None,
        search_queries: Optional[List[str]] = None,
        sentence_type: str = "content",
    ) -> List[VideoSearchQuery]:
        """
        Build optimized search queries with context awareness.

        Args:
            sentence: The sentence text
            keywords: Extracted keywords
            chapter_title: Chapter title for context
            search_queries: Pre-defined search queries
            sentence_type: Type of sentence (hook, content, cta)

        Returns:
            List of VideoSearchQuery objects, ordered by priority
        """
        # ✅ USE SIMPLE QUERIES (user requested)
        if self.use_simple_queries:
            return self.build_simple_queries(keywords, sentence)

        # ⚠️ COMPLEX QUERIES (legacy, not recommended)
        queries: List[VideoSearchQuery] = []
        seen_queries: Set[str] = set()

        # 1. Pre-defined search queries (highest priority)
        if search_queries:
            for query in search_queries[:3]:
                normalized = self._normalize_query(query)
                if normalized and normalized not in seen_queries:
                    seen_queries.add(normalized)
                    queries.append(VideoSearchQuery(
                        text=query,
                        priority=1,
                        source="chapter",
                        relevance_keywords=set(self._extract_words(query)),
                    ))

        # 2. Chapter title (high priority)
        if chapter_title:
            normalized = self._normalize_query(chapter_title)
            if normalized and normalized not in seen_queries:
                seen_queries.add(normalized)
                queries.append(VideoSearchQuery(
                    text=chapter_title,
                    priority=2,
                    source="chapter",
                    relevance_keywords=set(self._extract_words(chapter_title)),
                ))

        # 3. Sentence-based contextual queries (medium priority)
        sentence_queries = self._extract_contextual_queries(sentence)
        for query in sentence_queries[:2]:
            normalized = self._normalize_query(query)
            if normalized and normalized not in seen_queries:
                seen_queries.add(normalized)
                queries.append(VideoSearchQuery(
                    text=query,
                    priority=3,
                    source="sentence",
                    relevance_keywords=set(self._extract_words(sentence)),
                ))

        # 4. Combined keywords (medium priority)
        if keywords:
            combined = " ".join(keywords[:2])
            normalized = self._normalize_query(combined)
            if normalized and normalized not in seen_queries:
                seen_queries.add(normalized)
                queries.append(VideoSearchQuery(
                    text=combined,
                    priority=3,
                    source="keyword",
                    relevance_keywords=set(keywords),
                ))

        # 5. Individual keywords with expansions (lower priority)
        for keyword in keywords[:3]:
            # Original keyword
            normalized = self._normalize_query(keyword)
            if normalized and normalized not in seen_queries:
                seen_queries.add(normalized)
                queries.append(VideoSearchQuery(
                    text=keyword,
                    priority=4,
                    source="keyword",
                    relevance_keywords=set(keywords),
                ))

            # Expanded keywords (synonyms)
            expansions = self._expand_keyword(keyword)
            for expanded in expansions[:2]:
                normalized = self._normalize_query(expanded)
                if normalized and normalized not in seen_queries:
                    seen_queries.add(normalized)
                    queries.append(VideoSearchQuery(
                        text=expanded,
                        priority=5,
                        source="expansion",
                        relevance_keywords=set(keywords + [keyword]),
                    ))

        # Sort by priority (lower number = higher priority)
        queries.sort(key=lambda q: q.priority)

        logger.info(f"🔍 Built {len(queries)} search queries (priorities 1-5)")
        return queries[:10]  # Limit to top 10

    def _extract_contextual_queries(self, sentence: str) -> List[str]:
        """
        Extract contextual queries from sentence structure.

        Examples:
        - "The mountain rises above the clouds" -> "mountain clouds", "mountain landscape"
        - "People walking in the city" -> "people walking city", "urban walking"
        """
        queries = []

        # Extract patterns
        # Pattern 1: Subject + Verb
        match = re.search(r"(\w+)\s+(is|are|was|were|becomes?|rises?|falls?|grows?)\s+(\w+)", sentence, re.I)
        if match:
            subject, verb, complement = match.groups()
            queries.append(f"{subject} {complement}")

        # Pattern 2: Adjective + Noun
        match = re.search(r"(amazing|incredible|beautiful|massive|tiny|ancient|modern|mysterious)\s+(\w+)", sentence, re.I)
        if match:
            adjective, noun = match.groups()
            queries.append(f"{adjective} {noun}")

        # Pattern 3: Action phrases (verb + in/on/at + location)
        match = re.search(r"(\w+ing)\s+(?:in|on|at|through)\s+(?:the\s+)?(\w+)", sentence, re.I)
        if match:
            action, location = match.groups()
            queries.append(f"{action} {location}")

        # Pattern 4: Time-based
        match = re.search(r"(during|at|in)\s+(?:the\s+)?(sunrise|sunset|night|day|morning|evening)", sentence, re.I)
        if match:
            _, time = match.groups()
            queries.append(f"{time} landscape")

        return queries

    def _expand_keyword(self, keyword: str) -> List[str]:
        """
        Expand keyword with synonyms and related terms.

        Args:
            keyword: Original keyword

        Returns:
            List of expanded keywords
        """
        keyword_lower = keyword.lower()

        # Check cache
        if keyword_lower in self.expansion_cache:
            return self.expansion_cache[keyword_lower]

        expansions = []

        # Direct lookup
        if keyword_lower in KEYWORD_EXPANSIONS:
            expansions.extend(KEYWORD_EXPANSIONS[keyword_lower])

        # Partial match (e.g., "mountains" matches "mountain")
        for key, values in KEYWORD_EXPANSIONS.items():
            if key in keyword_lower or keyword_lower in key:
                expansions.extend(values)

        # Remove duplicates and limit
        expansions = list(dict.fromkeys(expansions))[:3]

        # Cache result
        self.expansion_cache[keyword_lower] = expansions

        return expansions

    def score_video_relevance(
        self,
        video_title: str,
        video_tags: List[str],
        query: VideoSearchQuery,
    ) -> float:
        """
        Score video relevance to query (0.0 to 1.0).

        Args:
            video_title: Video title or description
            video_tags: Video tags (if available)
            query: Search query with relevance keywords

        Returns:
            Relevance score (0.0 = not relevant, 1.0 = highly relevant)
        """
        score = 0.0

        # Normalize text
        video_text = f"{video_title.lower()} {' '.join(video_tags).lower()}"

        # Check exact query match
        if query.text.lower() in video_text:
            score += 0.4

        # Check relevance keywords
        matched_keywords = 0
        for keyword in query.relevance_keywords:
            if keyword.lower() in video_text:
                matched_keywords += 1

        if query.relevance_keywords:
            keyword_ratio = matched_keywords / len(query.relevance_keywords)
            score += keyword_ratio * 0.4

        # Bonus for high-priority queries
        priority_bonus = (6 - query.priority) * 0.04  # 0.04 to 0.2
        score += priority_bonus

        return min(1.0, score)

    def _normalize_query(self, query: str) -> str:
        """Normalize and clean query."""
        if not query:
            return ""

        # Remove special characters
        query = re.sub(r'[^\w\s-]', '', query)

        # Remove extra whitespace
        query = " ".join(query.split())

        # Remove very short queries
        if len(query) < 3:
            return ""

        return query.strip()

    def _extract_words(self, text: str) -> List[str]:
        """Extract meaningful words from text."""
        # Remove stopwords
        stopwords = {
            "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
            "in", "on", "at", "to", "for", "of", "with", "by", "from", "this", "that",
        }

        words = re.findall(r'\b\w+\b', text.lower())
        return [w for w in words if w not in stopwords and len(w) > 2]
