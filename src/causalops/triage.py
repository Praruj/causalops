"""Incident triage: drift tests propose suspects, an intervention check tests them.

The check is simple. We own the model and we have the labels for the incident
window, so for each suspect we undo that one change, score again and measure how
much of the extra loss goes away. A column that drifted but is not the cause
recovers nothing.

The ways of undoing a change come from a fixed menu. No code is generated or run.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .drift import bin_edges, drift_table
from .schemas import CausalValidation, DriftMetric, Hypothesis, RCAReport

LABEL_CAUSE = "label_prior"
MIN_PSI = 0.01  # below this a column is not treated as a suspect
MIN_RECOVERY = 0.25  # undoing the change must win back at least this share of the extra loss
MIN_EFFECTIVE_SHARE = 0.1  # reweighting is skipped when fewer rows than this still count
MIN_LABEL_EXPLAINED = 0.5  # a feature "explains" a label-rate move if it undoes this much
N_BINS = 20

Predict = Callable[[pd.DataFrame], np.ndarray]


def row_log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


# --- the menu ------------------------------------------------------------------


def reference_weights(ref: np.ndarray, cur: np.ndarray, bins: int = N_BINS) -> np.ndarray:
    """Row weights that give the incident column its reference distribution again.

    Rows in a value range that is now over-represented are weighted down. Rows that
    are missing, or outside the range seen in the reference data, get weight zero.
    The rows themselves are not changed, so their labels stay valid."""
    ref_ok = ref[~np.isnan(ref)]
    edges = bin_edges(ref_ok, bins)
    edges = np.concatenate([[ref_ok.min()], edges, [np.nextafter(ref_ok.max(), np.inf)]])
    n_bins = len(edges) + 2  # below range, the bins, above range, missing

    def assign(v: np.ndarray) -> np.ndarray:
        b = np.searchsorted(edges, np.nan_to_num(v), side="right")
        return np.where(np.isnan(v), n_bins - 1, b)

    ref_bin, cur_bin = assign(ref), assign(cur)
    p = np.bincount(ref_bin, minlength=n_bins) / len(ref)
    q = np.bincount(cur_bin, minlength=n_bins) / len(cur)
    return (p / np.where(q > 0, q, 1))[cur_bin]


def quantile_map(
    cur: np.ndarray, cur_y: np.ndarray, ref: np.ndarray, ref_y: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Map the column back onto its reference distribution within each class, keeping
    the order of the values. Exact for a unit change, because that keeps the order too.
    Used when the whole column has moved and reweighting has nothing left to work with."""
    out = cur.astype(float).copy()
    for c in np.unique(cur_y):
        idx = np.flatnonzero(cur_y == c)
        pool = ref[(ref_y == c) & ~np.isnan(ref)]
        if len(pool) == 0:
            pool = ref[~np.isnan(ref)]
        v = cur[idx].astype(float)
        known = v[~np.isnan(v)]
        if len(known):
            v = np.where(np.isnan(v), rng.choice(known, len(v)), v)
        else:
            v = rng.random(len(v))
        order = np.lexsort((rng.random(len(v)), v))
        ranks = np.empty(len(v), dtype=int)
        ranks[order] = np.arange(len(v))
        out[idx] = np.quantile(pool, (ranks + 0.5) / len(v))
    return out


def weighted_gain(rows: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Per-row values whose mean is (plain loss - weighted loss)."""
    return rows - w * rows / w.mean()


def effective_share(w: np.ndarray) -> float:
    """Effective sample size of the weighted rows, as a share of all rows."""
    return float(w.sum() ** 2 / (len(w) * (w**2).sum())) if w.sum() > 0 else 0.0


# --- triage --------------------------------------------------------------------


def _recovery(
    gain: np.ndarray, extra_loss: float, rng: np.random.Generator, n_boot: int = 200
) -> tuple[float, float, float]:
    """Recovered share of the extra loss, with a bootstrap interval over incident rows."""
    point = gain.mean() / extra_loss
    boots = rng.choice(gain, (n_boot, len(gain))).mean(axis=1) / extra_loss
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def pick_tests(m: DriftMetric) -> str:
    """Rule-based choice from the menu, from what the drift numbers look like."""
    if m.feature == LABEL_CAUSE:
        return "reweight_classes"
    tests = ["quantile_map", "reweight"]
    if m.null_rate_cur - m.null_rate_ref > 0.01:
        tests.append("replay_nulls")
    return ", ".join(tests)


def guess_cause_type(ref: np.ndarray, cur: np.ndarray) -> str:
    if np.isnan(cur).mean() - np.isnan(ref).mean() > 0.05:
        return "null_spike"
    ref_ok, cur_ok = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    qs = [0.1, 0.5, 0.9]
    ref_q, cur_q = np.quantile(ref_ok, qs), np.quantile(cur_ok, qs)
    if ref_q[0] > 0 and cur_q[0] > 0:
        ratio = cur_q[1] / ref_q[1]
        # A pure unit change: one rescale makes the column look normal again.
        if abs(np.log(ratio)) > np.log(1.5) and np.abs(cur_q / ratio / ref_q - 1).max() < 0.1:
            return "scale_change"
    return "distribution_shift"


def triage(
    ref_X: pd.DataFrame,
    ref_y: np.ndarray,
    cur_X: pd.DataFrame,
    cur_y: np.ndarray,
    predict: Predict,
    seed: int = 0,
) -> RCAReport:
    rng = np.random.default_rng(seed)
    ref_p, cur_p = predict(ref_X), predict(cur_X)
    ref_rows = row_log_loss(ref_y, ref_p)
    cur_rows = row_log_loss(cur_y, cur_p)
    loss_ref, loss_cur = float(ref_rows.mean()), float(cur_rows.mean())
    extra = loss_cur - loss_ref
    # The drop must be clear of sampling noise before we go looking for a cause.
    noise = np.sqrt(cur_rows.var() / len(cur_rows) + ref_rows.var() / len(ref_rows))
    dropped = extra > 3 * noise

    drift = drift_table(ref_X, cur_X, ref_y, cur_y, LABEL_CAUSE)
    hypotheses = [
        Hypothesis(
            cause=m.feature,
            test=pick_tests(m),
            reason=(
                f"PSI {m.psi:.3f}, KS {m.ks:.3f}, "
                f"nulls {m.null_rate_ref:.1%} -> {m.null_rate_cur:.1%}"
            ),
        )
        for m in drift
        if m.psi >= MIN_PSI
    ]

    validations: list[CausalValidation] = []
    rate_move = float(cur_y.mean() - ref_y.mean())
    if dropped:
        for h in hypotheses:
            label_explained = None
            if h.cause == LABEL_CAUSE:
                w = reference_weights(ref_y.astype(float), cur_y.astype(float))
                test, gain = "reweight_classes", weighted_gain(cur_rows, w)
            else:
                ref_col, cur_col = ref_X[h.cause].to_numpy(float), cur_X[h.cause].to_numpy(float)
                fixed = cur_X.copy()
                fixed[h.cause] = quantile_map(cur_col, cur_y, ref_col, ref_y, rng)
                test, gain = "quantile_map", cur_rows - row_log_loss(cur_y, predict(fixed))
                # Reweighting only means something if enough rows keep some weight.
                w = reference_weights(ref_col, cur_col)
                if effective_share(w) >= MIN_EFFECTIVE_SHARE:
                    reweighted = weighted_gain(cur_rows, w)
                    if reweighted.mean() > gain.mean():
                        test, gain = "reweight", reweighted
                    if rate_move != 0:
                        rate_w = float((w * cur_y).sum() / w.sum())
                        label_explained = (cur_y.mean() - rate_w) / rate_move
                # New missing values can be replayed: blank out the same share of the
                # reference column and see whether that alone produces the extra loss.
                # Same rows before and after, so this is much less noisy.
                new_nulls = np.isnan(cur_col).mean() - np.isnan(ref_col).mean()
                if new_nulls > 0.01:
                    broken = ref_X.copy()
                    broken.loc[rng.random(len(ref_X)) < new_nulls, h.cause] = np.nan
                    replayed = row_log_loss(ref_y, predict(broken)) - ref_rows
                    if replayed.mean() > gain.mean():
                        test, gain = "replay_nulls", replayed
            point, lo, hi = _recovery(gain, extra, rng)
            validations.append(
                CausalValidation(
                    cause=h.cause,
                    test=test,
                    loss_after_repair=loss_cur - float(gain.mean()),
                    recovered_fraction=point,
                    ci_low=lo,
                    ci_high=hi,
                    verified=point >= MIN_RECOVERY and lo > 0,
                    label_shift_explained=label_explained,
                )
            )
        validations.sort(key=lambda v: v.recovered_fraction, reverse=True)

    winner = next((v for v in validations if v.verified), None)
    if winner is not None and winner.cause == LABEL_CAUSE:
        # Features come before the label. If putting one feature back also puts the
        # label rate back, the label rate moved because of that feature.
        upstream = [
            v
            for v in validations
            if v.verified and (v.label_shift_explained or 0) >= MIN_LABEL_EXPLAINED
        ]
        winner = upstream[0] if upstream else winner
    root, cause_type = None, None
    if not dropped:
        summary = "No clear performance drop, so nothing to explain."
    elif winner is None:
        summary = (
            "Performance dropped, but undoing any single drifted column or the label rate "
            "does not bring it back. This may be concept drift or more than one cause."
        )
    else:
        root = winner.cause
        if root == LABEL_CAUSE:
            cause_type = "label_shift"
        else:
            cause_type = guess_cause_type(ref_X[root].to_numpy(float), cur_X[root].to_numpy(float))
        summary = (
            f"Log loss went from {loss_ref:.4f} to {loss_cur:.4f}. Undoing the change in "
            f"'{root}' wins back {winner.recovered_fraction:.0%} of the extra loss "
            f"(95% interval {winner.ci_low:.0%} to {winner.ci_high:.0%}). "
            f"It looks like a {cause_type.replace('_', ' ')}."
        )
        if winner.recovered_fraction > 1.3:
            summary += (
                " A share well above 100% means something else was pulling the loss down"
                " at the same time, or the estimate is unstable; read it as a pointer."
            )
        if drift[0].feature != root:
            summary += f" The column that drifted most, '{drift[0].feature}', is not the cause."

    return RCAReport(
        loss_reference=loss_ref,
        loss_incident=loss_cur,
        auc_reference=float(roc_auc_score(ref_y, ref_p)),
        auc_incident=float(roc_auc_score(cur_y, cur_p)),
        performance_dropped=bool(dropped),
        drift=drift,
        hypotheses=hypotheses,
        validations=validations,
        root_cause=root,
        cause_type=cause_type,
        summary=summary,
    )
