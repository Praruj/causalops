from causalops.benchmark import METHODS, run_benchmark, to_markdown, wilson
from causalops.incidents import KINDS


def test_wilson_interval_contains_the_point():
    lo, hi = wilson(8, 10)
    assert 0 < lo < 0.8 < hi < 1
    assert wilson(0, 0) == (0.0, 0.0)


def test_small_benchmark_runs_end_to_end():
    res = run_benchmark(
        n_incidents=5, n_controls=2, seed=1, n_train=20000, n_ref=10000, n_window=10000
    )
    assert set(res["by_kind"]) == {*KINDS, "all"}
    assert res["by_kind"]["all"]["n"] == 5
    for m in METHODS:
        assert 0 <= res["by_kind"]["all"][m]["accuracy"] <= 1
    assert "drift + intervention" in to_markdown(res)
