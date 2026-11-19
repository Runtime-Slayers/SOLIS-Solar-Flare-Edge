"""
QCAIE v2.2 - EXPERIENCE LAYER (Case-Based Reasoning, CBR)
==========================================================
"Logics and mathematics won't work in space as we thought.
 Data can say a little, but what says more? Experience."

This module implements a Case-Based Reasoning (CBR) layer on top of the
existing Kalman-Neupert system. CBR is a classical AI technique (Kolodner
1993; Aamodt & Plaza 1994) — NOT machine learning. There are no weights,
no gradient descent, no neural networks. It is a structured lookup table
of past "experiences" (cases) with a similarity-based retrieval mechanism.

WHY CBR FITS THE USER'S INTENT:
- The user observed that pure mathematics doesn't capture real space conditions.
- Each active region, each flare, each instrument glitch is unique.
- Instead of trying to fit a universal mathematical model, CBR says:
  "Have we seen something like this before? What happened? Use that."
- This is exactly how human operators at ISRO/NASA/NOAA actually work —
  they compare new observations to past events they remember.

ARCHITECTURE:
1. CASE: A stored experience = (feature_signature, context, outcome, lesson)
   - feature_signature: numerical vector characterizing the event
     [alpha_final, alpha_peak, alpha_slope, maha_peak, maha_slope,
      sxr_hxr_ratio, integral_growth_rate, duration, background_level]
   - context: source (SoLEXS/GOES/HEL1OS), date, flare_class, cadence
   - outcome: was_alert_correct, was_false_alarm, was_missed, lead_time
   - lesson: free-text "what we learned"

2. RETRIEVE: Given a live feature_signature, find the K nearest neighbors
   in the case base using a weighted Euclidean distance (Mahalanobis-like
   with diagonal weights — we don't have enough cases for full covariance).

3. REUSE: Combine the K retrieved outcomes via similarity-weighted voting
   to produce:
   - prior_P_flare: Bayesian prior for P(Pre-Flare)
   - recommended_alpha_ceiling: from similar active regions
   - recommended_confirm_steps: from similar signal-strength situations
   - false_alarm_risk: how often similar situations led to FAs in the past

4. REVISE: After the run completes, the actual outcome is recorded.

5. RETAIN: New cases are appended to the case base for future use.

This is the classic CBR cycle: Retrieve → Reuse → Revise → Retain.
(Aamodt & Plaza, "Case-Based Reasoning: Foundational Issues, Methodological
Variations, and System Approaches", AI Communications, 1994.)

NO ML. NO NEURAL NETWORKS. NO GRADIENT DESCENT.
Pure nearest-neighbor + Bayesian voting + structured experience storage.
"""
from __future__ import annotations

import json
import os
import math
import time
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any


# =============================================================================
# FEATURE SIGNATURE
# =============================================================================
# These are the dimensions of the "experience space" — the features that
# characterize a solar event from the filter's perspective. Each running
# window produces a feature vector; we compare it to past windows.

FEATURE_NAMES = [
    "alpha_current",          # current Neupert alpha estimate
    "alpha_slope_30",         # how fast alpha is changing (per 30 steps)
    "maha_peak_30",           # peak Mahalanobis distance in last 30 steps
    "maha_slope_30",          # is Maha rising or falling
    "mu_flare_mean_30",       # mean P(Pre-Flare) over last 30 steps
    "mu_flare_max_30",        # max P(Pre-Flare) over last 30 steps
    "sxr_hxr_ratio",          # SXR/HXR ratio (thermal vs non-thermal balance)
    "hxr_integral_growth",    # d(integral)/dt — energy injection rate
    "background_level",       # current baseline SXR level (solar-max awareness)
    "background_variability", # MAD of recent SXR baseline (noise level)
    "qpp_signif_max_30",      # max QPP significance in last 30 steps
    "health_degraded_frac",   # fraction of last 30 steps marked UNHEALTHY
    "dropout_frac_30",        # fraction of last 30 steps that were dropouts
    "cadence_seconds",        # instrument cadence (1 for SoLEXS, 60 for GOES)
]

FEATURE_WEIGHTS = np.array([
    1.0,   # alpha_current — important: tells us the active region type
    0.8,   # alpha_slope — important: tells us if we're converging or windup
    1.2,   # maha_peak — most important: signal strength
    0.7,   # maha_slope — context
    0.9,   # mu_flare_mean — context
    0.6,   # mu_flare_max — context
    1.0,   # sxr_hxr_ratio — important: thermal/non-thermal balance
    1.0,   # integral_growth — important: energy injection rate
    0.8,   # background — solar-max awareness
    0.5,   # variability — noise context
    0.6,   # qpp_signif — pulsation context
    0.5,   # health_degraded — sensor context
    0.5,   # dropout_frac — data quality context
    0.3,   # cadence — only matters for cross-instrument comparison
])

# Normalization scales (used to make features dimensionless for distance)
# These were derived from the 3 real SoLEXS + 1 live GOES datasets.
FEATURE_SCALES = np.array([
    0.5,    # alpha ranges ~0-2
    0.05,   # alpha slope
    5.0,    # maha peak
    0.5,    # maha slope
    0.3,    # mu mean
    0.5,    # mu max
    2.0,    # sxr/hxr ratio
    1.0,    # integral growth
    1.0,    # background level (after normalization)
    0.5,    # variability
    5.0,    # qpp signif
    0.3,    # degraded fraction
    0.3,    # dropout fraction
    30.0,   # cadence seconds
])


# =============================================================================
# CASE DATA STRUCTURE
# =============================================================================
@dataclass
class Case:
    """A single stored experience.

    Attributes:
        case_id: unique identifier (e.g. "solexs_20250328_flare")
        source: instrument ("SoLEXS", "GOES-18", "HEL1OS", "synthetic")
        date: ISO date string ("2025-03-28")
        flare_class: B/C/M/X or "QUIET"
        cadence_seconds: 1.0 for SoLEXS, 60.0 for GOES
        feature_signature: numerical vector (length = len(FEATURE_NAMES))
        outcome: dict with keys
            - correct: bool (was the alert decision correct?)
            - false_alarm: bool
            - missed: bool
            - lead_time_steps: int (negative = pre-flare, positive = lag)
            - peak_hxr: float
            - peak_sxr: float
            - alpha_final: float
        lesson: free-text lesson learned
        timestamp: when this case was added (epoch seconds)
    """
    case_id: str
    source: str
    date: str
    flare_class: str  # "B", "C", "M", "X", "QUIET"
    cadence_seconds: float
    feature_signature: List[float]
    outcome: Dict[str, Any]
    lesson: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Case":
        return cls(**d)


# =============================================================================
# CASE BASE
# =============================================================================
@dataclass
class CaseBase:
    """In-memory case base with persistence (JSON).

    The case base is loaded once at startup and persists across runs.
    New cases are appended after each run (RETAIN step of CBR cycle).
    """
    cases: List[Case] = field(default_factory=list)
    db_path: Optional[str] = None

    # ---- Persistence ----
    @classmethod
    def load(cls, path: str) -> "CaseBase":
        """Load case base from a JSON file. Returns empty case base if
        the file does not exist (first run)."""
        if not os.path.exists(path):
            return cls(cases=[], db_path=path)
        with open(path, "r") as f:
            data = json.load(f)
        cases = [Case.from_dict(c) for c in data.get("cases", [])]
        return cls(cases=cases, db_path=path)

    def save(self, path: Optional[str] = None) -> None:
        """Persist case base to JSON. If path is None, uses db_path."""
        path = path or self.db_path
        if path is None:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "version": "2.2",
                "n_cases": len(self.cases),
                "feature_names": FEATURE_NAMES,
                "cases": [c.to_dict() for c in self.cases],
            }, f, indent=2)

    # ---- Mutation ----
    def add(self, case: Case) -> None:
        # Replace if case_id already exists
        self.cases = [c for c in self.cases if c.case_id != case.case_id]
        self.cases.append(case)

    def __len__(self) -> int:
        return len(self.cases)


# =============================================================================
# FEATURE EXTRACTOR (RUNTIME)
# =============================================================================
class FeatureExtractor:
    """Extracts the feature signature from the live filter state.

    Used at every alert-evaluation step to query the case base.
    """

    def __init__(self, cadence_seconds: float = 1.0):
        self.cadence = float(cadence_seconds)
        self.alpha_hist: List[float] = []
        self.maha_hist: List[float] = []
        self.mu_hist: List[float] = []
        self.sxr_hist: List[float] = []
        self.hxr_hist: List[float] = []
        self.integral_hist: List[float] = []
        self.bg_hist: List[float] = []
        self.qpp_hist: List[float] = []
        self.health_hist: List[str] = []
        self.dropout_hist: List[bool] = []

    def update(
        self,
        alpha: float,
        maha: float,
        mu_flare: float,
        sxr: float,
        hxr: float,
        integral: float,
        background: float,
        qpp_signif: float,
        health: str,
        is_dropout: bool,
    ) -> None:
        self.alpha_hist.append(float(alpha))
        self.maha_hist.append(float(maha))
        self.mu_hist.append(float(mu_flare))
        self.sxr_hist.append(float(sxr))
        self.hxr_hist.append(float(hxr))
        self.integral_hist.append(float(integral))
        self.bg_hist.append(float(background))
        self.qpp_hist.append(float(qpp_signif))
        self.health_hist.append(health)
        self.dropout_hist.append(bool(is_dropout))

        # Keep last 60 samples max (one minute at 1 Hz, one hour at 1-min cadence)
        max_len = 60
        if len(self.alpha_hist) > max_len:
            for attr in [
                "alpha_hist", "maha_hist", "mu_hist", "sxr_hist",
                "hxr_hist", "integral_hist", "bg_hist", "qpp_hist",
                "health_hist", "dropout_hist",
            ]:
                lst = getattr(self, attr)
                del lst[: len(lst) - max_len]

    def extract(self) -> np.ndarray:
        """Extract the current feature signature (length = len(FEATURE_NAMES)).

        Returns zeros if not enough history.
        """
        n = len(self.alpha_hist)
        if n < 5:
            return np.zeros(len(FEATURE_NAMES))

        # Last 30 samples (or all if fewer)
        w = min(30, n)
        alpha_w = np.array(self.alpha_hist[-w:])
        maha_w = np.array(self.maha_hist[-w:])
        mu_w = np.array(self.mu_hist[-w:])
        sxr_w = np.array(self.sxr_hist[-w:])
        hxr_w = np.array(self.hxr_hist[-w:])
        integ_w = np.array(self.integral_hist[-w:])
        bg_w = np.array(self.bg_hist[-w:])
        qpp_w = np.array(self.qpp_hist[-w:])
        health_w = self.health_hist[-w:]
        drop_w = self.dropout_hist[-w:]

        # Alpha slope (per 30 steps)
        if w >= 2:
            alpha_slope = (alpha_w[-1] - alpha_w[0]) / max(w - 1, 1) * 30.0
        else:
            alpha_slope = 0.0

        # Mahalanobis slope
        if w >= 2:
            maha_slope = (maha_w[-1] - maha_w[0]) / max(w - 1, 1) * 30.0
        else:
            maha_slope = 0.0

        # SXR/HXR ratio (thermal vs non-thermal balance)
        hxr_mean = float(np.mean(hxr_w))
        sxr_mean = float(np.mean(sxr_w))
        sxr_hxr_ratio = sxr_mean / max(hxr_mean, 1e-6)

        # Integral growth rate (energy injection)
        if w >= 2:
            integ_growth = (integ_w[-1] - integ_w[0]) / max(w - 1, 1)
        else:
            integ_growth = 0.0

        # Background variability (MAD)
        bg_med = float(np.median(bg_w))
        bg_mad = float(np.median(np.abs(bg_w - bg_med)))

        # QPP max
        qpp_max = float(np.max(qpp_w))

        # Health degraded fraction
        degraded_frac = sum(1 for h in health_w if h == "UNHEALTHY") / w

        # Dropout fraction
        dropout_frac = sum(1 for d in drop_w if d) / w

        feature = np.array([
            float(alpha_w[-1]),         # alpha_current
            float(alpha_slope),         # alpha_slope_30
            float(np.max(maha_w)),      # maha_peak_30
            float(maha_slope),          # maha_slope_30
            float(np.mean(mu_w)),       # mu_flare_mean_30
            float(np.max(mu_w)),        # mu_flare_max_30
            float(sxr_hxr_ratio),       # sxr_hxr_ratio
            float(integ_growth),        # hxr_integral_growth
            float(bg_med),              # background_level
            float(bg_mad),              # background_variability
            float(qpp_max),             # qpp_signif_max_30
            float(degraded_frac),       # health_degraded_frac
            float(dropout_frac),        # dropout_frac_30
            float(self.cadence),        # cadence_seconds
        ])
        return feature


# =============================================================================
# RETRIEVAL (K-NEAREST NEIGHBORS)
# =============================================================================
def _normalize_features(v: np.ndarray) -> np.ndarray:
    """Normalize features to dimensionless units using FEATURE_SCALES."""
    return v / (FEATURE_SCALES + 1e-9)


def _weighted_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Weighted Euclidean distance in normalized feature space.

    Each feature is normalized by its scale, then weighted by FEATURE_WEIGHTS.
    """
    n1 = _normalize_features(v1)
    n2 = _normalize_features(v2)
    diff = n1 - n2
    return float(np.sqrt(np.sum(FEATURE_WEIGHTS * diff * diff)))


def retrieve(
    query: np.ndarray,
    case_base: CaseBase,
    k: int = 5,
    max_distance: float = 8.0,
) -> List[Case]:
    """Retrieve the K most similar cases from the case base.

    Args:
        query: feature signature of the current situation
        case_base: the case base to search
        k: max number of cases to retrieve
        max_distance: ignore cases farther than this (8.0 ≈ weak similarity)

    Returns:
        List of Cases, sorted by similarity (most similar first).
        May return fewer than k if not enough cases are within max_distance.
    """
    if len(case_base) == 0:
        return []

    dists = []
    for c in case_base.cases:
        v = np.array(c.feature_signature, dtype=float)
        # Pad/truncate to match query length
        if len(v) != len(query):
            v = np.resize(v, len(query))
        d = _weighted_distance(query, v)
        dists.append((d, c))

    dists.sort(key=lambda x: x[0])
    out = [c for d, c in dists[:k] if d <= max_distance]
    return out


# =============================================================================
# REUSE (BAYESIAN-LIKE VOTING FROM RETRIEVED CASES)
# =============================================================================
@dataclass
class ExperienceAdvice:
    """Recommendations produced by the experience layer.

    These are ADVISORY — the main pipeline uses them to bias its decisions,
    but does not blindly trust them. The main pipeline always retains the
    final authority (the math still runs; experience just shifts priors).
    """
    n_similar: int = 0
    similarity_scores: List[float] = field(default_factory=list)

    # Prior on P(Pre-Flare) — Bayesian update from similar past situations.
    # If similar situations often led to flares, this raises the prior.
    prior_P_flare: float = 0.0

    # Recommended alpha ceiling — adaptive based on similar active regions.
    # If similar regions needed alpha > 2.0, raise the ceiling.
    recommended_alpha_ceiling: float = 2.0

    # Recommended confirmation steps — adaptive based on signal strength patterns.
    recommended_confirm_steps: int = 4

    # False-alarm risk estimate (0 = no risk, 1 = high risk based on past FAs).
    false_alarm_risk: float = 0.0

    # Lessons learned (free text) — for logging and operator display.
    lessons: List[str] = field(default_factory=list)

    # Source cases (for audit trail)
    source_case_ids: List[str] = field(default_factory=list)


def reuse(retrieved: List[Case], query: np.ndarray) -> ExperienceAdvice:
    """Combine retrieved cases into an ExperienceAdvice via similarity-weighted voting.

    Each case votes with weight = exp(-distance / scale).
    The vote is its outcome (correct, false_alarm, missed, lead_time, alpha_final).
    """
    advice = ExperienceAdvice()
    if not retrieved:
        return advice

    advice.n_similar = len(retrieved)

    # Compute similarity weights (Gaussian kernel in feature space)
    weights = []
    for c in retrieved:
        v = np.array(c.feature_signature, dtype=float)
        if len(v) != len(query):
            v = np.resize(v, len(query))
        d = _weighted_distance(query, v)
        # sigma = 4.0 means: distance 4 → weight 0.37, distance 8 → weight 0.14
        w = math.exp(-(d * d) / (2.0 * 4.0 * 4.0))
        weights.append(w)
        advice.similarity_scores.append(w)
        advice.source_case_ids.append(c.case_id)
        if c.lesson:
            advice.lessons.append(f"[{c.case_id}] {c.lesson}")

    weights = np.array(weights)
    total = float(weights.sum())
    if total <= 0:
        return advice
    norm_w = weights / total

    # ---- Prior P(Pre-Flare) ----
    # Bayesian-style update: prior = similarity-weighted mean of
    # (1 - false_alarm_rate) for similar cases.
    fa_rates = []
    flare_rates = []
    for c in retrieved:
        oc = c.outcome
        # Was this situation followed by a real flare?
        is_flare = (not oc.get("false_alarm", False)) and \
                   (not oc.get("missed", False)) and \
                   (oc.get("lead_time_steps") is not None)
        fa_rates.append(1.0 if oc.get("false_alarm", False) else 0.0)
        flare_rates.append(1.0 if is_flare else 0.0)
    advice.prior_P_flare = float(np.dot(norm_w, flare_rates))
    advice.false_alarm_risk = float(np.dot(norm_w, fa_rates))

    # ---- Recommended alpha ceiling ----
    # Look at the alpha_final of similar cases; recommend a ceiling
    # at 95th percentile + 50% margin (the experience-based adaptive ceiling).
    alpha_finals = []
    for c in retrieved:
        af = c.outcome.get("alpha_final", 2.0)
        alpha_finals.append(float(af))
    if alpha_finals:
        p95 = float(np.percentile(alpha_finals, 95))
        # Adaptive ceiling: max(2.0, p95 * 1.5)
        advice.recommended_alpha_ceiling = max(2.0, p95 * 1.5)

    # ---- Recommended confirmation steps ----
    # Look at how strong the signal was in similar cases that were CONFIRMED
    # vs how strong it was in cases that turned out to be FALSE ALARMS.
    # If similar-looking situations mostly led to FAs, increase confirm steps.
    if advice.false_alarm_risk > 0.4:
        advice.recommended_confirm_steps = 6  # cautious
    elif advice.prior_P_flare > 0.6:
        advice.recommended_confirm_steps = 3  # confident
    else:
        advice.recommended_confirm_steps = 4  # standard

    return advice


# =============================================================================
# HIGH-LEVEL API
# =============================================================================
class ExperienceLayer:
    """The full CBR experience layer.

    Usage:
        layer = ExperienceLayer.load("/path/to/cases.json", cadence_seconds=1.0)
        layer.observe(alpha=..., maha=..., mu_flare=..., ...)  # each step
        advice = layer.advise()  # get current recommendations
        # ... run filter, decide alert using advice + math ...
        layer.retain(case)  # after run completes, store new case
        layer.save()
    """

    def __init__(self, case_base: CaseBase, cadence_seconds: float = 1.0):
        self.case_base = case_base
        self.extractor = FeatureExtractor(cadence_seconds=cadence_seconds)
        self._last_advice: Optional[ExperienceAdvice] = None
        self._last_query: Optional[np.ndarray] = None

    @classmethod
    def load(cls, path: str, cadence_seconds: float = 1.0) -> "ExperienceLayer":
        cb = CaseBase.load(path)
        return cls(case_base=cb, cadence_seconds=cadence_seconds)

    def observe(
        self,
        alpha: float,
        maha: float,
        mu_flare: float,
        sxr: float,
        hxr: float,
        integral: float,
        background: float,
        qpp_signif: float,
        health: str,
        is_dropout: bool,
    ) -> None:
        """Called at every filter step. Records the observation into the
        feature extractor's rolling window."""
        self.extractor.update(
            alpha, maha, mu_flare, sxr, hxr, integral, background,
            qpp_signif, health, is_dropout,
        )

    def advise(self, k: int = 5) -> ExperienceAdvice:
        """Query the case base with the current feature signature.

        Returns an ExperienceAdvice containing:
        - prior_P_flare (Bayesian prior on P(Pre-Flare))
        - recommended_alpha_ceiling
        - recommended_confirm_steps
        - false_alarm_risk
        - lessons learned (for operator display)
        """
        query = self.extractor.extract()
        self._last_query = query
        retrieved = retrieve(query, self.case_base, k=k)
        advice = reuse(retrieved, query)
        self._last_advice = advice
        return advice

    def retain(self, case: Case) -> None:
        """Store a new case after a run completes (RETAIN step of CBR)."""
        self.case_base.add(case)

    def save(self, path: Optional[str] = None) -> None:
        self.case_base.save(path)

    def __len__(self) -> int:
        return len(self.case_base)
