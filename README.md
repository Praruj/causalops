# CausalOps

When a deployed model gets worse, the usual first step is to look at which input
columns drifted. That gives a list of suspects, not an answer: plenty of columns
drift every week without hurting anything.

This project adds a second step. For each suspect I undo that one change in the
incident data, score the model again, and measure how much of the lost performance
comes back. Then I check how often that names the true cause, on 300 generated
incidents where the cause is known.

Everything runs offline on a laptop. There is no LLM and no API key.

## The short version

On 300 synthetic incidents (card-fraud data, five kinds of fault):

| incident kind | n | PSI rank | KS rank | PSI x importance | drift + intervention |
|---|---|---|---|---|---|
| unit shift | 60 | 100% | 100% | 100% | 100% |
| null spike | 60 | 100% | 95% | 100% | 90% |
| covariate shift | 60 | 95% | 92% | 95% | 100% |
| label shift | 60 | 47% | 23% | 100% | 100% |
| trap (harmless drift hides a real fault) | 60 | 40% | 32% | 100% | 97% |
| all | 300 | 76% (71-81) | 68% (63-73) | 99% (97-100) | 97% (95-99) |

Numbers are the share of incidents where the method's first pick was the true cause.
Brackets are 95% Wilson intervals. Full output is in [results/](results/).

What I take from this:

- **Ranking columns by drift alone is not enough.** It was right 76% of the time (PSI)
  and 68% (KS). In the trap incidents it blamed the harmless column 60% of the time.
- **The intervention check fixes that**, 97% overall. Its 8 misses were all
  "no cause found". It never named a wrong column, and it raised no alarm on 60 clean
  windows.
- **A much simpler heuristic did just as well here.** Multiplying drift by feature
  importance scored 99%. I did not expect that, and I am reporting it as it came out.
  On this benchmark the intervention check does not beat it on accuracy.
- What the check adds is different: a number ("undoing this wins back 93% of the
  loss"), a tested "no" for the other suspects, an honest "I don't know", and a guess
  at the kind of fault (right 76% of the time).

## How it works

1. **Drift.** For every column and for the label rate: PSI, KS and Wasserstein
   distance, plus the change in missing values.
2. **Suspects.** Every column with PSI of 0.01 or more becomes a hypothesis. Simple
   rules pick which tests to run on it.
3. **Intervention check.** The tests come from a fixed menu:
   - `reweight`: weight the incident rows so the column has its old distribution again.
     Missing and out-of-range rows get weight zero. Rows are not edited, so labels stay valid.
   - `quantile_map`: map the column back to its old distribution, keeping the order of
     values. This exactly undoes a unit change.
   - `replay_nulls`: blank out the same share of the column in the reference data and
     see if that alone produces the loss.
   - `reweight_classes`: put the label rate back.
4. **Verdict.** The suspect that recovers the most loss wins, if it recovers at least
   25% and its bootstrap interval stays above zero. If a feature repair also puts the
   label rate back, the feature is named, not the label. Otherwise the answer is
   "no cause found".

It is "causal" in a narrow sense: I own the model, so I can change one input and
re-run it. That is an intervention, not a correlation. It is not causal discovery,
and it does not learn a causal graph.

## The incidents

Synthetic transactions with a known structure: a hidden risk variable drives the
device score and the transaction count, six features feed the fraud label, and
`is_weekend` and `txn_hour` have no effect at all. Every window also gets some
ordinary harmless movement, so no column is ever perfectly still.

| kind | what happens | true cause |
|---|---|---|
| unit shift | a column arrives in the wrong unit (0-100 becomes 0-1) for 10-100% of rows | that column |
| null spike | 10-90% of a column goes missing | that column |
| covariate shift | a new customer segment pushes a column outside the training range | that column |
| label shift | fraud rate rises 1.5-4x, features within each class unchanged | the label rate |
| trap | a large harmless drift in one column, plus a smaller real fault in another | the faulty column |

A fault that does not hurt the model is not an incident, so it is drawn again. It took
820 draws to get 300 incidents. Most of the rejected ones were null spikes: the
tree model shrugs off a lot of missing values.

## Real data

I ran the same thing on real rows: the [Elec2](https://www.openml.org/d/151)
electricity-price data (27,552 rows after dropping the early part where three columns
are a constant). Real data has no known root cause, so faults are still injected; what
is real is the columns and how they relate.

With a random split, all four methods are at 98-100% on 120 incidents. It is an easy
setting, because the only thing that differs between the windows is the fault.

The hard setting is the natural one. Train on the first half in time order and compare
the next 20% with the last 30%: log loss goes from 0.668 to 0.768. The tool blames
`nswprice` and reports a 383% recovery. A price-level change is a plausible cause, but
nobody knows the true answer, and a share far above 100% means the estimate is not a
clean fraction there. I treat that result as a pointer, not a finding.

## Run it

```bash
pip install -e ".[dev,dashboard]"
python -m causalops benchmark          # 300 incidents, about 13 minutes on my laptop
python -m causalops benchmark --n 50   # quick version, about 2 minutes
python -m causalops triage --kind trap # one incident, printed step by step
python -m causalops real               # Elec2, downloads the data once
streamlit run app.py                   # dashboard
pytest -q
```

The dashboard lets you pick an incident kind and a seed, and shows the drift table
next to the recovery for each suspect.

## Limits

- **Synthetic faults are cleaner than real ones.** One fault at a time, on top of mild
  background drift. Two real faults at once are not tested.
- **It needs labels for the incident window.** In fraud, labels arrive weeks late. This
  is a tool for after the fact, not for live alerting.
- **It needs large windows.** I used 20,000 rows. With 5,000 the loss is too noisy to
  see anything but big faults.
- **A fault that leaves the column's distribution unchanged is invisible.** A bad join
  that gives rows the wrong values would not be flagged by any of the four methods.
- **Weak null spikes get "no cause found".** That is where most of the misses come from.
- **Partial unit faults are typed as "distribution shift"**, not "scale change". The
  column is right, the label for the kind of fault is not.
- **Concept drift (the meaning of the label changing) is not in the benchmark.** The
  tool can only say "no cause found" for it.
- **The decoys were built to be harmless.** That is what makes the truth known, but
  real "harmless" drift is rarely that clean.
- **One model type** (gradient-boosted trees) and one seed for the main table.
- **CI only runs a 20-incident benchmark**, to check the pipeline end to end. The
  numbers above come from the full run on my laptop.

## Layout

```
src/causalops/
  data.py        synthetic transactions
  incidents.py   the five incident kinds
  drift.py       PSI, KS, Wasserstein
  triage.py      suspects, the menu of tests, the verdict
  benchmark.py   the benchmark and the three baselines
  realdata.py    Elec2
  schemas.py     Pydantic models for the report
app.py           Streamlit dashboard
tests/           21 tests
```

MIT licence.
