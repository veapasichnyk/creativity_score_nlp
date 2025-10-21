# -*- coding: utf-8 -*-
"""
Feature engineering for text creativity / topic adherence.
Language-agnostic heuristics (works for English & Ukrainian).
Dependencies: numpy, pandas, scikit-learn (for TF-IDF only), re
"""
from __future__ import annotations
import math
import re
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Minimal bilingual stopword list (ENG+UKR). Extend if needed.
STOPWORDS = set(
    """
    a an the and or but if while of on in to for with at from by as is are was were be been being
    i me my we our you your he she it they them this that those these who whom whose which what
    do does did doing have has had having not no nor so than too very can could may might must shall should will would
    about above below under over again further then once here there when where why how all any both each few more most other some such
    own same so than too very s t can will just don should now
    я ми ви ти він вона ми вони мене мене наш ваш їх це той ті ці хто кого чий який що
    та і або але якщо коли то ж ні не а у в на з до від по за як щоб чи бо
    """.split()
)

_WORD_RE = re.compile(r"[\w’']+", re.UNICODE)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")


def tokenize_words(text: str) -> List[str]:
    text = text.lower()
    tokens = _WORD_RE.findall(text)
    # keep tokens that contain at least one letter (avoid pure digits)
    return [t for t in tokens if re.search(r"[a-zа-ящґєіїʼ’]", t)]


def split_sentences(text: str) -> List[str]:
    sents = re.split(r"(?<=[.!?…])\s+|\n+", text.strip())
    return [s for s in sents if s]


# ---- Lexical diversity & rarity ----

def type_token_ratio(tokens: List[str]) -> float:
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def hapax_ratio(tokens: List[str]) -> float:
    if not tokens:
        return 0.0
    cnt = Counter(tokens)
    hapax = sum(1 for w, c in cnt.items() if c == 1)
    return hapax / len(cnt)


def yules_i_inverse(tokens: List[str]) -> float:
    """Yule's I (inverse of Yule's K) variant.
    I = (N^2) / (sum_i f_i^2 - N)  (common form); we return I normalized.
    If denominator <= 0, return 0.
    Normalization: I' = I / (I + 1000) to map into (0,1).
    """
    if not tokens:
        return 0.0
    N = len(tokens)
    cnt = Counter(tokens)
    sum_f2 = sum(c * c for c in cnt.values())
    denom = sum_f2 - N
    if denom <= 0:
        return 0.0
    I = (N * N) / denom
    return I / (I + 1000.0)


def max_repetition_ratio(tokens: List[str]) -> float:
    """Max single-token dominance: max_count / N (penalizes repetition)."""
    if not tokens:
        return 0.0
    N = len(tokens)
    cnt = Counter(tokens)
    return max(cnt.values()) / N


def stopword_ratio(tokens: List[str]) -> float:
    if not tokens:
        return 0.0
    sw = sum(1 for t in tokens if t in STOPWORDS)
    return sw / len(tokens)


# ---- Structural complexity ----

def avg_word_len(tokens: List[str]) -> float:
    if not tokens:
        return 0.0
    # letters only length proxy
    return float(np.mean([len(re.sub(r"[^a-zа-ящґєіїʼ’]", "", t)) or 0 for t in tokens]))


def avg_sent_len_words(sentences: List[str]) -> float:
    if not sentences:
        return 0.0
    return float(np.mean([len(tokenize_words(s)) for s in sentences]))


def punctuation_density(text: str) -> float:
    if not text:
        return 0.0
    punct = re.findall(r"[,:;—–-]", text)
    return len(punct) / max(1, len(text))


def bigram_entropy(tokens: List[str]) -> float:
    if len(tokens) < 2:
        return 0.0
    bigrams = list(zip(tokens[:-1], tokens[1:]))
    cnt = Counter(bigrams)
    total = sum(cnt.values())
    probs = [c / total for c in cnt.values()]
    return float(-sum(p * math.log(p + 1e-12) for p in probs))


# ---- Topic adherence ----

def topic_cosine_similarity(text: str, topic: str) -> float:
    """Cosine similarity between the text and topic prompt via TF-IDF."""
    if not topic.strip() or not text.strip():
        return 0.0
    vec = TfidfVectorizer(min_df=1, max_df=0.9, ngram_range=(1, 2))
    X = vec.fit_transform([topic, text])
    a = X[0].toarray()[0]
    b = X[1].toarray()[0]
    num = float(np.dot(a, b))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(num / denom) if denom else 0.0


# ---- Master feature extractor ----

def extract_features(text: str, topic: str | None = None) -> Dict[str, float]:
    tokens = tokenize_words(text)
    sents = split_sentences(text)

    feats = {
        "ttr": type_token_ratio(tokens),
        "hapax_ratio": hapax_ratio(tokens),
        "yules_i_inv": yules_i_inverse(tokens),
        "avg_word_len": avg_word_len(tokens),
        "avg_sent_len_words": avg_sent_len_words(sents),
        "max_repetition_ratio": max_repetition_ratio(tokens),
        "stopword_ratio": stopword_ratio(tokens),
        "punctuation_density": punctuation_density(text),
        "bigram_entropy": bigram_entropy(tokens),
    }
    if topic is not None:
        feats["topic_similarity"] = topic_cosine_similarity(text, topic)
    return feats


def feature_vector_and_names(texts: List[str], topic: str | None = None) -> Tuple[np.ndarray, List[str]]:
    F = []
    names = None
    for t in texts:
        f = extract_features(t, topic=topic)
        if names is None:
            names = list(f.keys())
        F.append([f[k] for k in names])
    return np.asarray(F, dtype=float), names