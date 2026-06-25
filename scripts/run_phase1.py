"""Phase 1 scoreboard - QJL unbiased inner-product estimator.

Run:  python scripts/run_phase1.py  [--dim 128] [--seeds 4000]

Demonstrates the two properties that define Phase 1:
  1. **Unbiasedness** - averaged over many independent projections the QJL estimate of
     <q, x> converges to the true inner product (bias ~ 0 within a few standard errors).
  2. **1/m variance** - the estimator variance halves each time the number of sign bits
     m doubles, and matches the closed form (pi/2 ||q||^2||x||^2 - <q,x>^2) / m.

This is the "scoreboard" for Phase 1 referenced in docs/03.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from turboquant import QJL, qjl_estimator_variance  # noqa: E402


def estimates_over_seeds(q, x, m, n_seeds, base_seed=0):
    d = q.shape[0]
    out = np.empty(n_seeds)
    for i in range(n_seeds):
        qjl = QJL(d, m, seed=base_seed + i)
        out[i] = qjl.estimate(q, qjl.sketch(x))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=4000)
    args = ap.parse_args()

    d, n_seeds = args.dim, args.seeds
    rng = np.random.default_rng(123)
    q = rng.standard_normal(d)
    x = rng.standard_normal(d)
    truth = float(q @ x)

    print(f"\nTurboQuant Phase 1 - QJL unbiased inner-product estimator  (d={d})\n")
    print(f"true <q, x> = {truth:.5f}    (averaging over {n_seeds:,} random projections)\n")

    header = (
        f"{'m bits':>7} | {'mean est':>9} | {'bias':>9} | {'4*SE':>8} | "
        f"{'emp var':>9} | {'closed var':>10} | {'var*m':>9} | ok"
    )
    print(header)
    print("-" * len(header))

    ms = [16, 32, 64, 128, 256]
    all_ok = True
    var_times_m = []
    for m in ms:
        ests = estimates_over_seeds(q, x, m, n_seeds)
        mean = ests.mean()
        bias = mean - truth
        se = ests.std(ddof=1) / np.sqrt(n_seeds)
        emp_var = ests.var(ddof=1)
        closed = qjl_estimator_variance(q, x, m)
        unbiased_ok = abs(bias) <= 4.0 * se
        var_ok = abs(emp_var / closed - 1.0) <= 0.15
        ok = unbiased_ok and var_ok
        all_ok &= ok
        var_times_m.append(emp_var * m)
        print(
            f"{m:>7} | {mean:>9.5f} | {bias:>9.5f} | {se*4:>8.5f} | "
            f"{emp_var:>9.5f} | {closed:>10.5f} | {emp_var*m:>9.2f} | "
            f"{'PASS' if ok else 'FAIL'}"
        )

    # var*m should be ~constant (the 1/m law). Report its spread.
    vtm = np.array(var_times_m)
    spread = vtm.std() / vtm.mean()
    print(
        f"\nvar*m across m: mean={vtm.mean():.2f}  rel-spread={spread:.3f}  "
        f"(flat => variance ~ 1/m)"
    )

    print(f"\nresult: {'ALL PASS - QJL unbiased with 1/m variance' if all_ok else 'FAIL'}\n")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
