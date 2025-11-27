# -*- coding: utf-8 -*-
"""
Universal content quality scoring.
✅ ENHANCED: Hook strength, evergreen check, cold open validation
Works for all topics without topic-specific rules.
"""
import re
from typing import List, Dict, Tuple, Optional


class QualityScorer:
    """Score content quality, viral potential, and retention."""

    def score(self, sentences: List[str], title: str = "") -> Dict[str, float]:
        """
        Score content across multiple dimensions.
        
        Args:
            sentences: List of script sentences
            title: Optional video title
            
        Returns:
            Dict with keys: quality, viral, retention, overall
        """
        text_all = (" ".join(sentences) + " " + title).lower()
        
        scores = {
            'quality': 5.0,
            'viral': 5.0,
            'retention': 5.0
        }
        
        # ===== QUALITY SIGNALS =====
        scores['quality'] += self._score_quality(sentences, text_all)
        
        # ===== VIRAL SIGNALS =====
        scores['viral'] += self._score_viral(sentences, text_all)
        
        # ===== RETENTION SIGNALS =====
        scores['retention'] += self._score_retention(sentences, text_all)
        
        # ===== NEGATIVE SIGNALS =====
        penalty = self._score_penalties(text_all)
        scores['quality'] -= penalty
        scores['viral'] -= penalty
        scores['retention'] -= penalty * 0.5
        
        # Normalize to 0-10
        for key in scores:
            scores[key] = max(0.0, min(10.0, scores[key]))
        
        # Calculate overall (weighted)
        scores['overall'] = (
            scores['quality'] * 0.4 + 
            scores['viral'] * 0.35 + 
            scores['retention'] * 0.25
        )
        
        return scores
    
    def _score_quality(self, sentences: List[str], text_all: str) -> float:
        """Score content quality."""
        score = 0.0
        
        # Conciseness (short sentences = clearer)
        avg_words = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
        if avg_words <= 12:
            score += 1.5
        elif avg_words > 15:
            score -= 1.0
        
        # Specificity (numbers = concrete)
        num_count = len(re.findall(r'\b\d+\b', text_all))
        score += min(2.0, num_count * 0.5)
        
        # Active voice (action verbs)
        action_verbs = [
            'is', 'does', 'makes', 'shows', 'reveals', 
            'changes', 'breaks', 'creates', 'moves', 
            'stops', 'starts', 'turns'
        ]
        action_count = sum(1 for v in action_verbs if v in text_all)
        score += min(1.5, action_count * 0.3)
        
        return score
    
    def _score_viral(self, sentences: List[str], text_all: str) -> float:
        """Score viral potential."""
        score = 0.0
        
        if not sentences:
            return score
        
        hook = sentences[0].lower()
        
        # Hook strength
        if '?' in hook:
            score += 1.0
        if re.search(r'\b\d+\b', hook):
            score += 0.8
        
        mystery_words = ['secret', 'hidden', 'never', 'nobody', 'why', 'how']
        if any(w in hook for w in mystery_words):
            score += 0.6
        
        # Curiosity gap
        question_marks = text_all.count('?')
        score += min(1.2, question_marks * 0.4)
        
        # Emotional triggers
        triggers = [
            'shocking', 'insane', 'crazy', 'mind', 
            'unbelievable', 'secret', 'hidden'
        ]
        score += sum(0.3 for t in triggers if t in text_all)
        
        # Contrast markers
        contrasts = [
            'but', 'however', 'actually', 'surprisingly', 
            'turns out', 'wait'
        ]
        score += sum(0.25 for c in contrasts if c in text_all)
        
        return score
    
    def _score_retention(self, sentences: List[str], text_all: str) -> float:
        """Score retention potential."""
        score = 0.0
        
        # Pattern interrupts
        interrupts = [
            'wait', 'stop', 'look', 'watch', 
            'check', 'see', 'notice'
        ]
        score += sum(0.4 for i in interrupts if i in text_all)
        
        # Temporal cues (urgency)
        temporal = [
            'now', 'right now', 'immediately', 
            'seconds', 'instantly'
        ]
        score += sum(0.3 for t in temporal if t in text_all)
        
        # Visual references
        visual_refs = [
            'look at', 'watch', 'see', 'notice', 'spot', 'check'
        ]
        score += sum(0.35 for v in visual_refs if v in text_all)
        
        # Callback to hook (narrative closure)
        if len(sentences) >= 2 and sentences[-1] and sentences[0]:
            hook_words = set(sentences[0].lower().split()[:5])
            end_words = set(sentences[-1].lower().split())
            if hook_words & end_words:
                score += 1.0
        
        # Too long penalty
        if any(len(s.split()) > 18 for s in sentences):
            score -= 1.0
        
        return score
    
    def _score_penalties(self, text_all: str) -> float:
        """Calculate penalties for bad patterns."""
        penalty = 0.0
        
        # Generic filler
        bad_words = [
            'interesting', 'amazing', 'great', 'nice', 
            'good', 'cool', 'awesome'
        ]
        penalty += sum(0.5 for b in bad_words if b in text_all)
        
        # Meta references (breaks immersion)
        meta = [
            'this video', 'in this', 'today we', 
            "i'm going", "we're going", 'subscribe', 'like'
        ]
        penalty += sum(0.6 for m in meta if m in text_all)

        return penalty

    # ========== NEW: Enhanced Validation Methods ==========

    def validate_hook(self, hook: str) -> Tuple[bool, List[str]]:
        """
        Validate hook strength and cold open rules.

        Returns:
            (is_valid, issues_list)
        """
        issues = []
        hook_lower = hook.lower().strip()

        # Cold open violations
        cold_open_violations = [
            "this video",
            "today we",
            "in this video",
            "let me show you",
            "welcome to",
            "hey guys",
            "before we start",
        ]

        for violation in cold_open_violations:
            if violation in hook_lower:
                issues.append(f"Cold open violation: '{violation}'")

        # Hook too long
        word_count = len(hook.split())
        if word_count > 20:
            issues.append(f"Hook too long: {word_count} words (max 20)")

        # Hook too generic
        generic_patterns = [
            r"^let'?s (explore|talk about|discuss|look at)",
            r"^(have you|did you) ever (wonder|think)",
            r"^(today|this time)",
        ]
        for pattern in generic_patterns:
            if re.search(pattern, hook_lower):
                issues.append(f"Generic hook pattern: {pattern}")

        # Hook has no hook (too bland) - RELAXED validation
        # ✅ EXPANDED: Added more engagement markers (secret, hidden, discover, etc.)
        engagement_markers = [
            "?",  # Question
            "!",  # Exclamation
            " but ",  # Contrast
            "never",
            "nobody",
            "no one",
            "impossible",
            "unbelievable",
            "shocking",
            # NEW: Additional valid engagement words
            "secret",
            "hidden",
            "discover",
            "reveal",
            "unveil",
            "truth",
            "mystery",
            "incredible",
            "amazing",
            "astonishing",
            "extraordinary",
            "bizarre",
            "strange",
            "unusual",
            "what ",  # "What happens", "What if", etc.
            "why ",   # "Why does", "Why is", etc.
            "how ",   # "How can", "How does", etc.
        ]

        has_engagement = any(marker in hook_lower for marker in engagement_markers)
        has_number = bool(re.search(r"\d+", hook))

        if not has_engagement and not has_number:
            # ⚠️ RELAXED: This is now a warning, not a critical failure
            issues.append("Hook could be more engaging (add ?, !, numbers, or power words)")

        return len(issues) == 0, issues

    def validate_evergreen(self, text: str) -> Tuple[bool, List[str]]:
        """
        Validate content is evergreen (no temporal references).

        Returns:
            (is_evergreen, temporal_refs_found)
        """
        temporal_violations = [
            # Specific dates
            r"\b20\d{2}\b",  # Years like 2024
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}",
            # Temporal words
            r"\b(recently|nowadays|currently|today'?s|right now|this year|last year|next year)\b",
            r"\b(this month|last month|next month|this week|last week)\b",
            r"\b(modern|contemporary)\b",
            # Implied recency (context-aware)
            r"\b(just (released|announced|discovered))\b",
            r"\b(breaking (news|story|update)|latest (news|update|release)|brand new|newly (released|announced|discovered)|upcoming (event|release))\b",
        ]

        violations_found = []
        text_lower = text.lower()

        for pattern in temporal_violations:
            matches = re.findall(pattern, text_lower)
            if matches:
                violations_found.extend(matches)

        return len(violations_found) == 0, violations_found

    def validate_sentence_length(
        self, sentences: List[str], max_length: int = 20
    ) -> Tuple[bool, List[Tuple[int, int]]]:
        """
        Validate all sentences are within length limit.

        Returns:
            (all_valid, [(sentence_index, word_count)])
        """
        violations = []

        for idx, sentence in enumerate(sentences):
            word_count = len(sentence.split())
            if word_count > max_length:
                violations.append((idx, word_count))

        return len(violations) == 0, violations

    def validate_flow(self, sentences: List[str]) -> Tuple[bool, float, List[str]]:
        """
        Validate sentence flow and coherence.

        Returns:
            (is_good_flow, flow_score, issues)
        """
        issues = []
        flow_score = 10.0

        if len(sentences) < 5:
            return True, flow_score, []

        # Check for monotonous structure (same start word 3x in a row)
        for i in range(len(sentences) - 2):
            first_words = [
                s.split()[0].lower() if s.split() else "" for s in sentences[i : i + 3]
            ]
            if len(set(first_words)) == 1 and first_words[0]:
                issues.append(f"Monotonous structure at sentences {i}-{i+2}: all start with '{first_words[0]}'")
                flow_score -= 2.0

        # Check for missing transitions
        transition_words = {
            "but",
            "however",
            "so",
            "because",
            "therefore",
            "then",
            "next",
            "first",
            "finally",
            "meanwhile",
            "instead",
        }

        transition_count = 0
        for sentence in sentences:
            if any(word in sentence.lower().split()[:3] for word in transition_words):
                transition_count += 1

        expected_transitions = len(sentences) // 4  # Expect ~25% transition sentences
        if transition_count < expected_transitions:
            issues.append(f"Low transition usage: {transition_count}/{expected_transitions} expected")
            flow_score -= 1.5

        # Check for variety in sentence length
        lengths = [len(s.split()) for s in sentences]
        if max(lengths) - min(lengths) < 5:
            issues.append("Low sentence length variety (all similar length)")
            flow_score -= 1.0

        flow_score = max(0.0, min(10.0, flow_score))

        return flow_score >= 7.0, flow_score, issues

    def comprehensive_validation(
        self, sentences: List[str], title: str = "", script_style_config: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        Run comprehensive validation on script.

        Args:
            sentences: Script sentences
            title: Video title
            script_style_config: Script style configuration

        Returns:
            Dict with validation results
        """
        if not sentences:
            return {
                "valid": False,
                "overall_score": 0.0,
                "issues": ["No sentences provided"],
                "scores": {},
            }

        # Get config values
        config = script_style_config or {}
        max_length = config.get("max_sentence_length", 20)
        require_evergreen = config.get("evergreen_only", True)
        require_cold_open = config.get("cold_open", True)

        # Run validations
        results = {
            "valid": True,
            "overall_score": 10.0,
            "issues": [],
            "scores": {},
        }

        # 1. Hook validation
        if require_cold_open and sentences:
            hook_valid, hook_issues = self.validate_hook(sentences[0])
            if not hook_valid:
                results["issues"].extend([f"Hook: {issue}" for issue in hook_issues])
                # ✅ REDUCED PENALTY: 2.0 → 1.0 (less punitive for hook issues)
                results["overall_score"] -= 1.0

        # 2. Evergreen validation
        if require_evergreen:
            full_text = " ".join(sentences) + " " + title
            is_evergreen, violations = self.validate_evergreen(full_text)
            if not is_evergreen:
                results["issues"].append(f"Temporal references found: {violations[:3]}")
                results["overall_score"] -= 3.0

        # 3. Sentence length validation
        length_valid, length_violations = self.validate_sentence_length(sentences, max_length)
        if not length_valid:
            results["issues"].append(f"{len(length_violations)} sentences exceed {max_length} words")
            results["overall_score"] -= min(2.0, len(length_violations) * 0.3)

        # 4. Flow validation
        flow_valid, flow_score, flow_issues = self.validate_flow(sentences)
        results["scores"]["flow"] = flow_score
        if not flow_valid:
            results["issues"].extend([f"Flow: {issue}" for issue in flow_issues])
            results["overall_score"] -= (10.0 - flow_score) * 0.3

        # 5. Quality scoring
        quality_scores = self.score(sentences, title)
        results["scores"].update(quality_scores)

        # Final score
        results["overall_score"] = max(
            0.0, min(10.0, results["overall_score"] * (quality_scores["overall"] / 10.0))
        )

        # ✅ LOWERED THRESHOLD: 6.5 → 5.5 → 4.5 (production calibration for longer scripts)
        # Longer scripts (100-130 sentences) naturally score lower, so we need a lower threshold
        results["valid"] = results["overall_score"] >= 4.5 and len(results["issues"]) < 5

        return results
