"""The same benchmark on real rows: the Elec2 electricity-price data from OpenML.

Elec2 has 45,312 half-hourly records from the New South Wales electricity market
(1996-1998). The label says whether the price went up against its 24-hour average.
The Victoria columns are a constant for the first 17,760 rows, so those rows are left out.

Real data has no known root cause, so the faults are still injected. What is real
here is the rows: skewed columns, real correlations between them, a noisy label.
One extra run looks at the natural drift over time, where the truth is not known.
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmark import MAX_TRIES, baseline_picks, summarise, to_markdown
from .incidents import LABEL_CAUSE, Incident, apply_null_fault
from .model import importances, predict, train_model
from .triage import triage

FEATURES = ["day", "period", "nswprice", "nswdemand", "vicprice", "vicdemand", "transfer"]
FIRST_LIVE_ROW = 17760
REAL_KINDS = ["unit_shift", "null_spike", "label_shift", "trap"]
# The columns are stored scaled to 0-1; a wrong scale stands in for a wrong unit.
SCALE_FACTORS = {
    "nswprice": 1000.0,
    "nswdemand": 100.0,
    "vicprice": 1000.0,
    "vicdemand": 100.0,
    "transfer": 100.0,
}


def load_elec2(data_home: str | Path = "data") -> tuple[pd.DataFrame, np.ndarray]:
    from sklearn.datasets import fetch_openml

    raw = fetch_openml(data_id=151, as_frame=True, data_home=str(data_home), parser="auto")
    X = raw.data[FEATURES].astype(float).iloc[FIRST_LIVE_ROW:].reset_index(drop=True)
    y = (raw.target.astype(str) == "UP").to_numpy().astype(int)[FIRST_LIVE_ROW:]
    return X, y


def _take(X: pd.DataFrame, y: np.ndarray, idx: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    return X.iloc[idx].reset_index(drop=True), y[idx]


def make_real_incident(
    kind: str, X: pd.DataFrame, y: np.ndarray, rng: np.random.Generator, n: int
) -> Incident:
    idx = rng.choice(len(X), n, replace=False)
    if kind == "label_shift":
        # Far more or far fewer "price up" periods, features within each class untouched.
        up, down = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        rate = float(rng.choice([rng.uniform(0.1, 0.25), rng.uniform(0.65, 0.85)]))
        n_up = int(n * rate)
        idx = np.concatenate([rng.choice(up, n_up), rng.choice(down, n - n_up)])
        wX, wy = _take(X, y, rng.permutation(idx))
        return Incident(kind, LABEL_CAUSE, wX, wy, detail={"up_rate": rate})

    if kind == "trap":
        # Decoy: a window with far more weekend rows than usual. A real change of mix.
        weekend = X["day"].isin([1.0, 7.0]).to_numpy()
        p = np.where(weekend, 6.0, 1.0)
        idx = rng.choice(len(X), n, replace=False, p=p / p.sum())
    wX, wy = _take(X, y, idx)

    fault = kind if kind != "trap" else str(rng.choice(["unit_shift", "null_spike"]))
    f = str(rng.choice(list(SCALE_FACTORS)))
    share = float(rng.uniform(0.1, 0.9))
    if fault == "unit_shift":
        hit = rng.random(n) < share
        wX.loc[hit, f] = wX.loc[hit, f] * SCALE_FACTORS[f]
    else:
        wX = apply_null_fault(wX, f, rng, share)
    decoy = "day" if kind == "trap" else None
    return Incident(kind, f, wX, wy, decoy=decoy, detail={"fault": fault, "share": share})


def run(n_incidents: int = 120, seed: int = 0, data_home: str | Path = "data") -> dict:
    X, y = load_elec2(data_home)
    rng = np.random.default_rng(seed)

    # Random split: train on 50%, and the rest is shared by the reference set and the
    # incident pool. Nothing but the injected fault differs, so the true cause is known.
    order = rng.permutation(len(X))
    cut_a, cut_b = int(0.5 * len(X)), int(0.75 * len(X))
    train_X, train_y = _take(X, y, order[:cut_a])
    ref_X, ref_y = _take(X, y, order[cut_a:cut_b])
    pool_X, pool_y = _take(X, y, order[cut_b:])
    model = train_model(train_X, train_y, seed)
    score = partial(predict, model)
    imp = importances(model, ref_X, ref_y, seed)
    n_window = int(0.7 * len(pool_X))

    rows = []
    for i in range(n_incidents):
        kind = REAL_KINDS[i % len(REAL_KINDS)]
        inc_rng = np.random.default_rng([seed, i])
        for tries in range(1, MAX_TRIES + 1):  # noqa: B007
            inc = make_real_incident(kind, pool_X, pool_y, inc_rng, n_window)
            rep = triage(ref_X, ref_y, inc.X, inc.y, score, seed=i)
            if rep.performance_dropped:
                break
        picks = baseline_picks(rep, imp)
        row = {
            "kind": kind,
            "tries": tries,
            "true_cause": inc.true_cause,
            "decoy": inc.decoy,
            "dropped": rep.performance_dropped,
            "extra_loss": rep.loss_incident - rep.loss_reference,
            "intervention": rep.root_cause,
            "cause_type": rep.cause_type,
        }
        for m, ranked in picks.items():
            row[m] = ranked[0]
            row[m + "_top3"] = inc.true_cause in ranked[:3]
        rows.append(row)

    false_alarms, n_controls = 0, 40
    for i in range(n_controls):
        idx = np.random.default_rng([seed, 10_000 + i]).choice(len(pool_X), n_window, replace=False)
        cX, cy = _take(pool_X, pool_y, idx)
        false_alarms += triage(ref_X, ref_y, cX, cy, score, seed=i).root_cause is not None

    res = summarise(pd.DataFrame(rows), n_controls, false_alarms, seed)
    res["natural_drift"] = natural_drift(X, y, seed)
    return res


def natural_drift(X: pd.DataFrame, y: np.ndarray, seed: int = 0) -> dict:
    """Train on the first half in time order and triage the last 30% against the
    20% in between. Nothing is injected, and nobody knows the true cause."""
    a, b = int(0.5 * len(X)), int(0.7 * len(X))
    model = train_model(X.iloc[:a], y[:a], seed)
    rep = triage(
        X.iloc[a:b].reset_index(drop=True),
        y[a:b],
        X.iloc[b:].reset_index(drop=True),
        y[b:],
        partial(predict, model),
        seed,
    )
    return json.loads(rep.model_dump_json())

def save(res: dict, out_dir: str | Path = "results") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "real_elec2.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (out / "real_elec2.md").write_text(to_markdown(res) + "\n", encoding="utf-8")
    return out / "real_elec2.json"
