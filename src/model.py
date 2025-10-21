# -*- coding: utf-8 -*-
"""
Rule-based scorer + optional linear regressor to calibrate on labeled data.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


@dataclass
class RuleWeights:
    # Positive contributors
    w_ttr: float = 0.25
    w_hapax: float = 0.10
    w_yule: float = 0.15
    w_avg_word_len: float = 0.08
    w_avg_sent_len: float = 0.10
    w_bigram_entropy: float = 0.12
    w_topic_similarity: float = 0.20  # only if provided

    # Penalties (subtracted)
    p_repetition: float = 0.20
    p_stopword: float = 0.10


class RuleBasedScorer:
    def __init__(self, weights: RuleWeights | None = None, use_topic: bool = True):
        self.w = weights or RuleWeights()
        self.use_topic = use_topic

    @staticmethod
    def _z(x, lo, hi):
        """Clamp+normalize x to [0,1] using [lo,hi] band."""
        if hi == lo:
            return 0.0
        x = max(lo, min(hi, x))
        return (x - lo) / (hi - lo)

    def score(self, feats: Dict[str, float]) -> float:
        # Normalization bands (empirical; adjust after EDA)
        z_ttr = self._z(feats.get("ttr", 0), 0.2, 0.8)
        z_hapax = self._z(feats.get("hapax_ratio", 0), 0.1, 0.6)
        z_yule = self._z(feats.get("yules_i_inv", 0), 0.0, 0.6)
        z_awl = self._z(feats.get("avg_word_len", 0), 3.5, 7.5)
        z_asl = self._z(feats.get("avg_sent_len_words", 0), 8, 30)
        z_entropy = self._z(feats.get("bigram_entropy", 0), 0.5, 5.0)
        z_topic = self._z(feats.get("topic_similarity", 0), 0.1, 0.8)

        z_rep = self._z(feats.get("max_repetition_ratio", 0), 0.0, 0.2)
        z_stop = self._z(feats.get("stopword_ratio", 0), 0.1, 0.7)

        pos = (
            self.w.w_ttr * z_ttr
            + self.w.w_hapax * z_hapax
            + self.w.w_yule * z_yule
            + self.w.w_avg_word_len * z_awl
            + self.w.w_avg_sent_len * z_asl
            + self.w.w_bigram_entropy * z_entropy
            + (self.w.w_topic_similarity * z_topic if self.use_topic else 0.0)
        )

        neg = (
            self.w.p_repetition * z_rep
            + self.w.p_stopword * z_stop
        )

        raw = pos - neg
        # Map to [0,100]
        score = 100.0 * max(0.0, min(1.0, raw))
        return float(score)


class LinearCalibrator:
    """Optional: fit a linear model on labeled data (targets in [0,100])."""
    def __init__(self):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("reg", Ridge(alpha=1.0))
        ])
        self.feature_names_: List[str] | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
        self.pipeline.fit(X, y)
        self.feature_names_ = feature_names
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict(X)

    def coefficients(self) -> List[Tuple[str, float]]:
        reg = self.pipeline.named_steps["reg"]
        assert self.feature_names_ is not None
        return list(zip(self.feature_names_, reg.coef_.tolist()))