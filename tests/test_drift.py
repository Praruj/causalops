import numpy as np

from causalops.data import sample
from causalops.drift import drift_table, psi


def test_psi_is_small_for_two_samples_of_the_same_thing():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(size=5000), rng.normal(size=5000)) < 0.01


def test_psi_sees_a_shift_and_a_null_spike():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=5000)
    assert psi(ref, rng.normal(1.0, 1.0, 5000)) > 0.25
    holes = rng.normal(size=5000)
    holes[:1500] = np.nan
    assert psi(ref, holes) > 0.25


def test_psi_works_on_a_binary_column():
    rng = np.random.default_rng(0)
    ref = (rng.random(5000) < 0.05).astype(float)
    cur = (rng.random(5000) < 0.20).astype(float)
    assert psi(ref, cur) > 0.1
    assert psi(ref, (rng.random(5000) < 0.05).astype(float)) < 0.01


def test_drift_table_is_sorted_and_includes_the_label():
    rng = np.random.default_rng(1)
    ref_X, ref_y = sample(4000, rng)
    cur_X, cur_y = sample(4000, rng, weekend_rate=0.8)
    table = drift_table(ref_X, cur_X, ref_y, cur_y)
    assert table[0].feature == "is_weekend"
    assert [m.psi for m in table] == sorted((m.psi for m in table), reverse=True)
    assert "label_prior" in {m.feature for m in table}
