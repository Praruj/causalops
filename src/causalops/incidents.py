"""Incident generator. Every incident has one known true cause.

Five kinds:
  unit_shift       a feature arrives in the wrong unit (for example 0-100 becomes 0-1)
  null_spike       an upstream job starts sending missing values for a feature
  covariate_shift  a new customer segment pushes a feature outside the training range
  label_shift      the fraud rate goes up; P(features | label) is unchanged
  trap             a small real fault hidden behind a large harmless drift in another feature
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import NewSegment, sample

KINDS = ["unit_shift", "null_spike", "covariate_shift", "label_shift", "trap"]
LABEL_CAUSE = "label_prior"

# Multiplier applied when a feature is delivered in the wrong unit.
UNIT_FACTORS = {
    "device_risk_score": 0.01,  # 0-100 -> 0-1
    "amount": 100.0,  # dollars -> cents
    "account_age_days": 1 / 365,  # days -> years
    "distance_km": 1000.0,  # km -> m
    "merchant_risk": 100.0,  # 0-1 -> percent
}
NULLABLE = ["device_risk_score", "account_age_days", "n_txn_24h", "distance_km"]
SEGMENT_RANGES = {"amount": (8, 25), "distance_km": (15, 40), "n_txn_24h": (10, 16)}
DECOYS = ["is_weekend", "txn_hour", "account_age_days", "merchant_risk"]


@dataclass
class Incident:
    kind: str
    true_cause: str
    X: pd.DataFrame
    y: np.ndarray
    decoy: str | None = None
    detail: dict = field(default_factory=dict)


def apply_unit_fault(
    X: pd.DataFrame, feature: str, rng: np.random.Generator, share: float = 1.0
) -> pd.DataFrame:
    X = X.copy()
    hit = rng.random(len(X)) < share
    X.loc[hit, feature] = X.loc[hit, feature] * UNIT_FACTORS[feature]
    return X


def apply_null_fault(
    X: pd.DataFrame, feature: str, rng: np.random.Generator, share: float
) -> pd.DataFrame:
    X = X.copy()
    X.loc[rng.random(len(X)) < share, feature] = np.nan
    return X


def _decoy_kwargs(decoy: str, rng: np.random.Generator) -> dict:
    """A large shift that does not hurt: either a feature with no effect on fraud,
    or a move towards safer customers inside the range the model was trained on."""
    if decoy == "is_weekend":
        return {"weekend_rate": rng.uniform(0.6, 0.85)}
    if decoy == "txn_hour":
        return {"night_share": rng.uniform(0.5, 0.8)}
    if decoy == "account_age_days":
        return {"age_scale": rng.uniform(700, 1000)}
    if decoy == "merchant_risk":
        return {"merchant_a": rng.uniform(1.0, 1.4)}
    raise ValueError(decoy)


def _background(rng: np.random.Generator) -> dict:
    """Ordinary week-to-week movement that every window has. None of it raises fraud:
    the two bystander columns wander, and the customer mix leans a little safer."""
    return {
        "weekend_rate": rng.uniform(0.2, 0.5),
        "night_share": rng.uniform(0.1, 0.35),
        "age_scale": rng.uniform(400, 550),
        "merchant_a": rng.uniform(1.7, 2.0),
    }


def make_incident(kind: str, rng: np.random.Generator, n: int = 20000) -> Incident:
    bg = _background(rng)
    if kind == "unit_shift":
        f = str(rng.choice(list(UNIT_FACTORS)))
        share = float(rng.uniform(0.1, 1.0))  # a partial rollout hits only some rows
        X, y = sample(n, rng, **bg)
        return Incident(kind, f, apply_unit_fault(X, f, rng, share), y, detail={"share": share})

    if kind == "null_spike":
        f = str(rng.choice(NULLABLE))
        share = float(rng.uniform(0.1, 0.9))
        X, y = sample(n, rng, **bg)
        return Incident(kind, f, apply_null_fault(X, f, rng, share), y, detail={"share": share})

    if kind == "covariate_shift":
        f = str(rng.choice(list(SEGMENT_RANGES)))
        seg = NewSegment(f, float(rng.uniform(0.1, 0.6)), float(rng.uniform(*SEGMENT_RANGES[f])))
        X, y = sample(n, rng, segment=seg, **bg)
        return Incident(kind, f, X, y, detail={"share": seg.share, "factor": seg.factor})

    if kind == "label_shift":
        mult = float(rng.uniform(1.5, 4.0))
        X, y = sample(n, rng, fraud_mult=mult, **bg)
        return Incident(kind, LABEL_CAUSE, X, y, detail={"fraud_mult": mult})

    if kind == "trap":
        decoy = str(rng.choice(DECOYS))
        X, y = sample(n, rng, **{**bg, **_decoy_kwargs(decoy, rng)})
        fault = str(rng.choice(["null", "unit"]))
        if fault == "null":
            f = str(rng.choice([c for c in NULLABLE if c != decoy]))
            share = float(rng.uniform(0.1, 0.6))
            X = apply_null_fault(X, f, rng, share)
        else:
            f = str(rng.choice([c for c in UNIT_FACTORS if c != decoy]))
            share = float(rng.uniform(0.1, 0.5))
            X = apply_unit_fault(X, f, rng, share)
        return Incident(kind, f, X, y, decoy=decoy, detail={"fault": fault, "share": share})

    raise ValueError(f"unknown incident kind: {kind}")
