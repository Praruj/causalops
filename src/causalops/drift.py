"""Per-column drift statistics: PSI, KS and Wasserstein distance."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .schemas import DriftMetric

EPS = 1e-4


def bin_edges(ref_ok: np.ndarray, bins: int) -> np.ndarray:
    """Cut points from reference quantiles. A column with only a few distinct values
    (a flag, a count) gets one bin per value instead."""
    values = np.unique(ref_ok)
    if len(values) <= bins:
        return (values[:-1] + values[1:]) / 2
    return np.unique(np.quantile(ref_ok, np.linspace(0, 1, bins + 1)[1:-1]))


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """Population stability index. Missing values get a bin of their own, so a
    null spike shows up here too."""
    ref_ok, cur_ok = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    edges = bin_edges(ref_ok, bins)
    p = np.bincount(np.searchsorted(edges, ref_ok, side="right"), minlength=len(edges) + 1)
    q = np.bincount(np.searchsorted(edges, cur_ok, side="right"), minlength=len(edges) + 1)
    p = np.append(p, len(ref) - len(ref_ok)) / len(ref)
    q = np.append(q, len(cur) - len(cur_ok)) / len(cur)
    p, q = np.clip(p, EPS, None), np.clip(q, EPS, None)
    return float(np.sum((q - p) * np.log(q / p)))


def drift_metric(name: str, ref: np.ndarray, cur: np.ndarray) -> DriftMetric:
    ref = np.asarray(ref, dtype=float)
    cur = np.asarray(cur, dtype=float)
    ref_ok, cur_ok = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    scale = float(np.std(ref_ok)) or 1.0
    return DriftMetric(
        feature=name,
        psi=psi(ref, cur),
        ks=float(stats.ks_2samp(ref_ok, cur_ok, method="asymp").statistic),
        wasserstein=float(stats.wasserstein_distance(ref_ok / scale, cur_ok / scale)),
        null_rate_ref=float(np.isnan(ref).mean()),
        null_rate_cur=float(np.isnan(cur).mean()),
    )


def drift_table(
    ref_X: pd.DataFrame,
    cur_X: pd.DataFrame,
    ref_y: np.ndarray | None = None,
    cur_y: np.ndarray | None = None,
    label_name: str = "label_prior",
) -> list[DriftMetric]:
    """Drift for every feature, plus the label rate when labels are given.
    Sorted with the largest PSI first."""
    rows = [drift_metric(c, ref_X[c].to_numpy(), cur_X[c].to_numpy()) for c in ref_X.columns]
    if ref_y is not None and cur_y is not None:
        rows.append(drift_metric(label_name, ref_y, cur_y))
    return sorted(rows, key=lambda m: m.psi, reverse=True)
