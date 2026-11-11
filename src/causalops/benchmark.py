"""Benchmark: generate many incidents with a known cause and count how often
each method names it."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from .data import sample
from .incidents import KINDS, make_incident
from .model import importances, predict, train_model
from .schemas import RCAReport
from .triage import LABEL_CAUSE, triage

MAX_TRIES = 25
METHODS = ["psi_rank", "ks_rank", "psi_x_importance", "intervention"]
TYPE_NAMES = {
    "unit_shift": "scale_change",
    "null_spike": "null_spike",
    "covariate_shift": "distribution_shift",
    "label_shift": "label_shift",
}


def baseline_picks(report: RCAReport, imp: dict[str, float]) -> dict[str, list[str]]:
    """Ranked suspects for the three drift-only baselines. The label rate is a
    candidate in all of them; for the importance baseline it gets weight 1."""
    d = report.drift
    ks = sorted(d, key=lambda m: m.ks + (m.null_rate_cur - m.null_rate_ref), reverse=True)
    pxi = sorted(d, key=lambda m: m.psi * imp.get(m.feature, 1.0), reverse=True)
    return {
        "psi_rank": [m.feature for m in d],
        "ks_rank": [m.feature for m in ks],
        "psi_x_importance": [m.feature for m in pxi],
    }


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    z, p = 1.96, k / n
    mid = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(mid - half), float(mid + half)


def run_benchmark(
    n_incidents: int = 300,
    n_controls: int = 60,
    seed: int = 0,
    n_train: int = 40000,
    n_ref: int = 20000,
    n_window: int = 20000,
) -> dict:
    rng = np.random.default_rng(seed)
    X_train, y_train = sample(n_train, rng)
    ref_X, ref_y = sample(n_ref, rng)
    model = train_model(X_train, y_train, seed)
    score = partial(predict, model)
    imp = importances(model, ref_X, ref_y, seed)

    rows = []
    for i in range(n_incidents):
        kind = KINDS[i % len(KINDS)]
        # A fault that does not hurt the model is not an incident, so draw again.
        inc_rng = np.random.default_rng([seed, i])
        for tries in range(1, MAX_TRIES + 1):  # noqa: B007
            inc = make_incident(kind, inc_rng, n_window)
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

    false_alarms = 0
    for i in range(n_controls):
        cX, cy = sample(n_window, np.random.default_rng([seed, 10_000 + i]))
        false_alarms += triage(ref_X, ref_y, cX, cy, score, seed=i).root_cause is not None

    return summarise(pd.DataFrame(rows), n_controls, false_alarms, seed)


def summarise(df: pd.DataFrame, n_controls: int, false_alarms: int, seed: int) -> dict:
    by_kind = {}
    for kind, g in [*df.groupby("kind", sort=False), ("all", df)]:
        entry = {
            "n": len(g),
            "performance_dropped": int(g["dropped"].sum()),
            "faults_drawn": int(g["tries"].sum()),
        }
        for m in METHODS:
            k = int((g[m] == g["true_cause"]).sum())
            lo, hi = wilson(k, len(g))
            entry[m] = {"correct": k, "accuracy": k / len(g), "ci_low": lo, "ci_high": hi}
            if m != "intervention":
                entry[m]["top3"] = float(g[m + "_top3"].mean())
        # The intervention method may say "no cause found" instead of guessing.
        entry["intervention"]["abstained"] = int(g["intervention"].isna().sum())
        by_kind[kind] = entry

    traps = df[df["kind"] == "trap"]
    pure = df[df["kind"].isin(TYPE_NAMES) & (df["intervention"] == df["true_cause"])]
    return {
        "seed": seed,
        "n_incidents": len(df),
        "by_kind": by_kind,
        "trap_blamed_decoy": {
            m: float((traps[m] == traps["decoy"]).mean()) if len(traps) else 0.0 for m in METHODS
        },
        "cause_type_accuracy": (
            float((pure["cause_type"] == pure["kind"].map(TYPE_NAMES)).mean()) if len(pure) else 0.0
        ),
        "controls": {"n": n_controls, "false_alarms": false_alarms},
        "incidents": json.loads(df.to_json(orient="records")),
    }


def to_markdown(res: dict) -> str:
    head = "| incident kind | n | PSI rank | KS rank | PSI x importance | drift + intervention |"
    lines = [head, "|---|---|---|---|---|---|"]
    for kind, e in res["by_kind"].items():
        cells = [f"{e[m]['accuracy']:.0%}" for m in METHODS]
        if kind == "all":
            cells = [
                f"**{e[m]['accuracy']:.0%}** ({e[m]['ci_low']:.0%}-{e[m]['ci_high']:.0%})"
                for m in METHODS
            ]
        lines.append(f"| {kind} | {e['n']} | " + " | ".join(cells) + " |")
    c = res["controls"]
    total = res["by_kind"]["all"]
    missed = total["n"] - total["intervention"]["correct"]
    abstained = total["intervention"]["abstained"]
    lines += [
        "",
        f"Intervention misses: {missed}, of which {abstained} said 'no cause found' "
        f"and {missed - abstained} named a wrong cause.",
        "Trap incidents where the method blamed the harmless decoy: "
        + ", ".join(f"{m} {v:.0%}" for m, v in res["trap_blamed_decoy"].items()),
        f"Clean windows with no fault: {c['false_alarms']} false alarms out of {c['n']}.",
        f"Cause type named correctly (when the cause was right, trap excluded): "
        f"{res['cause_type_accuracy']:.0%}.",
    ]
    return "\n".join(lines)


def save(res: dict, out_dir: str | Path = "results") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (out / "benchmark.md").write_text(to_markdown(res) + "\n", encoding="utf-8")
    return out / "benchmark.json"


__all__ = ["LABEL_CAUSE", "run_benchmark", "save", "to_markdown"]
