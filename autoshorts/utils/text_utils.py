# -*- coding: utf-8 -*-
"""
Text processing utilities: normalize, clean, tokenize, simple hashtags.
Bu dosya youtube_uploader tarafından import ediliyor.
"""
import re
from typing import List, Set
try:
    # varsa kendi sabitlerinizi kullanın
    from autoshorts.config.constants import GENERIC_SKIP, STOP_EN, STOP_TR
except Exception:
    GENERIC_SKIP, STOP_EN, STOP_TR = set(), set(), set()

def normalize_sentence(raw: str) -> str:
    s = (raw or "").strip()
    s = s.replace("\\n", "\n").replace("\r\n", "\n").replace("\r", "\n")
    s = "\n".join(re.sub(r"\s+", " ", ln).strip() for ln in s.split("\n"))
    s = s.replace("—", "-").replace("–", "-")
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    s = s.replace("´", "'").replace("`", "'")
    s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)
    return s

def clean_caption_text(s: str) -> str:
    t = (s or "").strip()
    t = t.replace("—", "-").replace("–", "-")
    t = t.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    t = t.replace("´", "'").replace("`", "")
    t = re.sub(r"\s+", " ", t).strip()
    if t and t[0].islower():
        t = t[0].upper() + t[1:]
    return t

def tokenize_words_loose(s: str) -> List[str]:
    s = re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower())
    return [w for w in s.split() if len(w) >= 3]

def tokenize_words(s: str) -> List[str]:
    s = re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower())
    return [w for w in s.split() if len(w) >= 3 and w not in STOP_EN]

def trigrams(words: List[str]) -> Set[str]:
    if len(words) < 3:
        return set()
    return {" ".join(words[i:i+3]) for i in range(len(words)-2)}

def sentences_fingerprint(sentences: List[str]) -> Set[str]:
    ws = tokenize_words(" ".join(sentences or []))
    return trigrams(ws)

def hashtags_from_tags(tags: List[str], title: str, limit: int = 5) -> List[str]:
    """Basit, güvenli hashtag üretimi."""
    pool = [*(tags or [])]
    if title:
        pool.extend(tokenize_words_loose(title))
    seen = set()
    out: List[str] = []
    for t in pool:
        key = re.sub(r"[^a-z0-9]+", "", (t or "").lower())
        if not key or key in seen:
            continue
        seen.add(key)
        out.append("#" + key[:28])
        if len(out) >= limit:
            break
    return out
