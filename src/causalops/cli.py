from __future__ import annotations

import argparse
from functools import partial

import numpy as np

from . import benchmark as bm
from .data import sample
from .incidents import KINDS, make_incident
from .model import predict, train_model
from .triage import triage


def demo_report(kind: str, seed: int):
    """Train a model, create one incident and triage it."""
    rng = np.random.default_rng(seed)
    X_train, y_train = sample(40000, rng)
    ref_X, ref_y = sample(20000, rng)
    model = train_model(X_train, y_train, seed)
    # As in the benchmark: a fault that does not hurt the model is drawn again.
    for _ in range(bm.MAX_TRIES):
        inc = make_incident(kind, rng)
        rep = triage(ref_X, ref_y, inc.X, inc.y, partial(predict, model), seed)
        if rep.performance_dropped:
            break
    return inc, rep


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="causalops")
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("benchmark", help="run the synthetic benchmark and write results/")
    b.add_argument("--n", type=int, default=300, help="number of incidents")
    b.add_argument("--controls", type=int, default=60, help="clean windows with no fault")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", default="results")

    t = sub.add_parser("triage", help="create one incident and print the report")
    t.add_argument("--kind", choices=KINDS, default="trap")
    t.add_argument("--seed", type=int, default=1)

    r = sub.add_parser("real", help="run the benchmark on real rows (downloads Elec2 once)")
    r.add_argument("--n", type=int, default=120)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", default="results")

    args = parser.parse_args(argv)
    if args.cmd == "benchmark":
        res = bm.run_benchmark(args.n, args.controls, args.seed)
        path = bm.save(res, args.out)
        print(bm.to_markdown(res))
        print(f"\nsaved {path}")
    elif args.cmd == "triage":
        inc, rep = demo_report(args.kind, args.seed)
        print(f"true cause: {inc.true_cause}   decoy: {inc.decoy}   {inc.detail}\n")
        print("drift (largest PSI first)")
        for m in rep.drift[:5]:
            print(f"  {m.feature:20s} PSI {m.psi:7.3f}  KS {m.ks:.3f}")
        print("\nintervention check")
        for v in rep.validations[:5]:
            mark = "yes" if v.verified else "no"
            print(f"  {v.cause:20s} recovers {v.recovered_fraction:6.0%}  verified: {mark}")
        print("\n" + rep.summary)
    else:
        from . import realdata

        res = realdata.run(args.n, args.seed)
        path = realdata.save(res, args.out)
        print(realdata.to_markdown(res))
        print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
