"""Small dashboard: break the data in a chosen way, then watch the triage.

    streamlit run app.py
"""

import json
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from causalops.benchmark import METHODS
from causalops.data import sample
from causalops.incidents import KINDS, make_incident
from causalops.model import predict, train_model
from causalops.triage import triage

st.set_page_config(page_title="CausalOps", layout="wide")
st.title("CausalOps: which change broke the model?")


@st.cache_resource
def world():
    rng = np.random.default_rng(0)
    X_train, y_train = sample(40000, rng)
    ref_X, ref_y = sample(20000, rng)
    return ref_X, ref_y, partial(predict, train_model(X_train, y_train, 0))


one, bench = st.tabs(["One incident", "Benchmark"])

with one:
    left, right = st.columns([1, 3])
    kind = left.selectbox("Incident kind", KINDS, index=KINDS.index("trap"))
    seed = left.number_input("Seed", 0, 9999, 3)
    ref_X, ref_y, score = world()
    inc = make_incident(kind, np.random.default_rng(int(seed)))
    rep = triage(ref_X, ref_y, inc.X, inc.y, score, int(seed))

    extra = rep.loss_incident - rep.loss_reference
    left.metric("Log loss", f"{rep.loss_incident:.4f}", f"{extra:+.4f}", delta_color="inverse")
    left.metric("AUC", f"{rep.auc_incident:.3f}", f"{rep.auc_incident - rep.auc_reference:+.3f}")
    decoy = f", decoy: **{inc.decoy}**" if inc.decoy else ""
    left.caption(f"True cause: **{inc.true_cause}**{decoy}")

    right.subheader("Verdict")
    found = rep.root_cause == inc.true_cause
    (right.success if found else right.warning)(rep.summary)

    a, b = right.columns(2)
    a.markdown("**Step 1. Drift: who moved?**")
    drift = pd.DataFrame([m.model_dump() for m in rep.drift])
    a.dataframe(drift[["feature", "psi", "ks", "null_rate_cur"]], hide_index=True)
    b.markdown("**Step 2. Undo each change: how much loss comes back?**")
    if rep.validations:
        v = pd.DataFrame([x.model_dump() for x in rep.validations])
        b.bar_chart(v.set_index("cause")["recovered_fraction"])
        b.dataframe(v[["cause", "test", "recovered_fraction", "ci_low", "ci_high", "verified"]],
                    hide_index=True)
    else:
        b.info("No clear performance drop, so no checks were run.")

with bench:
    path = Path(__file__).parent / "results" / "benchmark.json"
    if not path.exists():
        st.info("Run `python -m causalops benchmark` first.")
    else:
        res = json.loads(path.read_text(encoding="utf-8"))
        table = pd.DataFrame(
            {m: {k: e[m]["accuracy"] for k, e in res["by_kind"].items()} for m in METHODS}
        )
        st.caption(f"Share of {res['n_incidents']} incidents where the method named the cause.")
        st.dataframe(table.style.format("{:.0%}"))
        st.bar_chart(table.drop(index="all"), stack=False)
