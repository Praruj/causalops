"""Synthetic card-fraud data with a known causal structure.

A hidden "risk" variable drives both the device score and the transaction count,
and all three feed the fraud label. `is_weekend` and `txn_hour` have no effect on
fraud at all, so any drift in them is harmless by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURES = [
    "amount",
    "device_risk_score",
    "account_age_days",
    "n_txn_24h",
    "distance_km",
    "merchant_risk",
    "txn_hour",
    "is_weekend",
]

# Features that enter the fraud label. The other two are bystanders.
CAUSAL_FEATURES = FEATURES[:6]
BYSTANDERS = ["txn_hour", "is_weekend"]

# Intercept chosen so that about 5% of clean transactions are fraud.
INTERCEPT = -6.7


@dataclass
class NewSegment:
    """A share of traffic that comes from a kind of customer not seen in training."""

    feature: str
    share: float
    factor: float


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _draw(
    n: int,
    rng: np.random.Generator,
    weekend_rate: float,
    night_share: float,
    age_scale: float,
    merchant_a: float,
    segment: NewSegment | None,
) -> tuple[pd.DataFrame, np.ndarray]:
    u = rng.normal(size=n)  # hidden risk, never shown to the model
    amount = np.exp(rng.normal(3.5, 1.0, n))
    device = 100 * _sigmoid(0.9 * u + rng.normal(size=n))
    age = rng.exponential(age_scale, n)
    n_txn = rng.poisson(np.exp(0.7 + 0.4 * u)).astype(float)
    distance = np.exp(rng.normal(1.5, 1.2, n))
    merchant = rng.beta(merchant_a, 5.0, n)
    night = rng.random(n) < night_share
    hour = np.where(night, rng.integers(0, 6, n), rng.integers(6, 24, n)).astype(float)
    weekend = (rng.random(n) < weekend_rate).astype(float)

    if segment is not None:
        in_seg = rng.random(n) < segment.share
        if segment.feature == "amount":
            amount = np.where(in_seg, amount * segment.factor, amount)
        elif segment.feature == "distance_km":
            distance = np.where(in_seg, distance * segment.factor, distance)
        elif segment.feature == "n_txn_24h":
            n_txn = np.where(in_seg, n_txn + rng.poisson(segment.factor, n), n_txn)
        else:
            raise ValueError(f"no new-segment rule for {segment.feature}")

    z = (
        INTERCEPT
        + 0.03 * device
        + 0.5 * np.log(amount)
        + 0.18 * n_txn
        + 3.0 * merchant
        - 0.6 * np.log1p(age)
        + 0.6 * np.log1p(distance)
        + 0.8 * u
    )
    y = (rng.random(n) < _sigmoid(z)).astype(int)
    X = pd.DataFrame(
        {
            "amount": amount,
            "device_risk_score": device,
            "account_age_days": age,
            "n_txn_24h": n_txn,
            "distance_km": distance,
            "merchant_risk": merchant,
            "txn_hour": hour,
            "is_weekend": weekend,
        }
    )
    return X, y


def sample(
    n: int,
    rng: np.random.Generator,
    *,
    weekend_rate: float = 0.28,
    night_share: float = 0.15,
    age_scale: float = 400.0,
    merchant_a: float = 2.0,
    segment: NewSegment | None = None,
    fraud_mult: float = 1.0,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Draw n transactions. With fraud_mult != 1 the fraud rate is changed by
    re-sampling within each class, so P(features | label) stays the same."""
    kwargs = dict(
        weekend_rate=weekend_rate,
        night_share=night_share,
        age_scale=age_scale,
        merchant_a=merchant_a,
        segment=segment,
    )
    if fraud_mult == 1.0:
        return _draw(n, rng, **kwargs)

    X, y = _draw(int(n * 4 * max(fraud_mult, 1.0)), rng, **kwargs)
    n_pos = int(round(n * min(y.mean() * fraud_mult, 0.9)))
    pos = rng.choice(np.flatnonzero(y == 1), n_pos, replace=False)
    neg = rng.choice(np.flatnonzero(y == 0), n - n_pos, replace=False)
    keep = rng.permutation(np.concatenate([pos, neg]))
    return X.iloc[keep].reset_index(drop=True), y[keep]
