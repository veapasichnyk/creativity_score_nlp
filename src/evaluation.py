# -*- coding: utf-8 -*-
"""
Evaluation utilities: correlation with labels if available; sanity tests.
"""
from __future__ import annotations
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

from src.features import extract_features, feature_vector_and_names
from src.model import RuleBasedScorer, LinearCalibrator


def evaluate_rule_based(texts: List[str], topics: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    scorer = RuleBasedScorer(use_topic=topics is not None)
    scores = []
    feats_list = []
    for i, t in enumerate(texts):
        topic = topics[i] if topics is not None else None
        feats = extract_features(t, topic)
        feats_list.append(feats)
        scores.append(scorer.score(feats))
    return {"scores": np.array(scores), "features": feats_list}


def correlation_with_labels(pred: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    rho, _ = spearmanr(pred, y)
    r, _ = pearsonr(pred, y)
    return {"spearman_rho": float(rho), "pearson_r": float(r)}


def sanity_checks(text: str, topic: Optional[str] = None) -> Dict[str, float]:
    """Generate controlled variants and ensure ordering behaves sensibly."""
    base_feats = extract_features(text, topic)
    # Variant 1: repetition (worse)
    rep_text = text + "\n" + " ".join(text.split()[:30]) * 3
    rep_feats = extract_features(rep_text, topic)
    # Variant 2: shuffled sentences (usually worse coherence)
    sents = [s for s in text.split('.') if s.strip()]
    np.random.seed(0)
    np.random.shuffle(sents)
    shuf_text = '. '.join(sents) + '.'
    shuf_feats = extract_features(shuf_text, topic)

    rb = RuleBasedScorer(use_topic=topic is not None)
    base = rb.score(base_feats)
    rep = rb.score(rep_feats)
    shuf = rb.score(shuf_feats)

    return {"base": base, "repetition_variant": rep, "shuffled_variant": shuf}


def fit_linear_calibrator(texts: List[str], labels: np.ndarray, topic: Optional[str] = None):
    X, names = feature_vector_and_names(texts, topic=topic)
    cal = LinearCalibrator().fit(X, labels, names)
    return cal, names


def plot_scores_distribution(scores: np.ndarray, title: str = "Rule-based score distribution"):
    plt.figure()
    plt.hist(scores, bins=20)
    plt.title(title)
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.tight_layout()