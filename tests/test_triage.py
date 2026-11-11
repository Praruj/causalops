import numpy as np

from causalops.data import sample
from causalops.incidents import LABEL_CAUSE, apply_null_fault, apply_unit_fault
from causalops.schemas import RCAReport
from causalops.triage import quantile_map, reference_weights, triage


def test_quantile_map_undoes_a_unit_change():
    rng = np.random.default_rng(0)
    ref = rng.gamma(2.0, 10.0, 5000)
    truth = rng.gamma(2.0, 10.0, 5000)
    y = np.zeros(5000, dtype=int)
    fixed = quantile_map(truth / 100, y, ref, y, rng)
    assert np.corrcoef(fixed, truth)[0, 1] > 0.99
    assert abs(np.median(fixed) / np.median(truth) - 1) < 0.1


def test_weights_drop_missing_and_out_of_range_rows():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=5000)
    cur = rng.normal(size=5000)
    cur[:500] = np.nan
    cur[500:1000] = 50.0
    w = reference_weights(ref, cur)
    assert w[:1000].max() == 0
    assert (w[1000:] > 0).mean() > 0.99


def test_clean_window_gets_no_root_cause(world):
    ref_X, ref_y, score = world
    X, y = sample(20000, np.random.default_rng(11))
    rep = triage(ref_X, ref_y, X, y, score)
    assert rep.root_cause is None


def test_unit_change_is_found_and_typed(world):
    ref_X, ref_y, score = world
    rng = np.random.default_rng(12)
    X, y = sample(20000, rng)
    rep = triage(ref_X, ref_y, apply_unit_fault(X, "device_risk_score", rng), y, score)
    assert rep.performance_dropped
    assert rep.root_cause == "device_risk_score"
    assert rep.cause_type == "scale_change"


def test_label_shift_is_not_blamed_on_a_feature(world):
    ref_X, ref_y, score = world
    X, y = sample(20000, np.random.default_rng(13), fraud_mult=3.0)
    rep = triage(ref_X, ref_y, X, y, score)
    assert rep.root_cause == LABEL_CAUSE


def test_trap_big_harmless_drift_does_not_win(world):
    ref_X, ref_y, score = world
    rng = np.random.default_rng(14)
    X, y = sample(20000, rng, weekend_rate=0.8)
    X = apply_null_fault(X, "device_risk_score", rng, 0.5)
    rep = triage(ref_X, ref_y, X, y, score)
    assert rep.root_cause == "device_risk_score"
    assert rep.cause_type == "null_spike"
    weekend = next(v for v in rep.validations if v.cause == "is_weekend")
    assert not weekend.verified


def test_report_survives_a_json_round_trip(world):
    ref_X, ref_y, score = world
    rng = np.random.default_rng(15)
    X, y = sample(20000, rng)
    rep = triage(ref_X, ref_y, apply_unit_fault(X, "account_age_days", rng), y, score)
    assert RCAReport.model_validate_json(rep.model_dump_json()) == rep
