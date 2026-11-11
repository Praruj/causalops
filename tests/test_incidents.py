import numpy as np
import pytest

from causalops.data import FEATURES, sample
from causalops.incidents import KINDS, LABEL_CAUSE, make_incident


@pytest.mark.parametrize("kind", KINDS)
def test_every_kind_builds_and_names_a_real_cause(kind):
    inc = make_incident(kind, np.random.default_rng(3), n=2000)
    assert list(inc.X.columns) == FEATURES
    assert len(inc.X) == len(inc.y) == 2000
    assert inc.true_cause in FEATURES or inc.true_cause == LABEL_CAUSE
    if kind == "trap":
        assert inc.decoy is not None and inc.decoy != inc.true_cause


def test_same_seed_gives_the_same_incident():
    a = make_incident("trap", np.random.default_rng(5), n=1000)
    b = make_incident("trap", np.random.default_rng(5), n=1000)
    assert a.true_cause == b.true_cause
    assert a.X.equals(b.X)


def test_label_shift_raises_the_fraud_rate():
    base = sample(20000, np.random.default_rng(0))[1].mean()
    inc = make_incident("label_shift", np.random.default_rng(0), n=5000)
    assert inc.y.mean() > 1.3 * base


def test_null_spike_only_touches_the_named_column():
    inc = make_incident("null_spike", np.random.default_rng(2), n=2000)
    nulls = inc.X.isna().mean()
    assert nulls[inc.true_cause] > 0.05
    assert nulls.drop(inc.true_cause).sum() == 0
