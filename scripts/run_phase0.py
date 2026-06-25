"""Phase 0 scoreboard - reproduce TurboQuant's synthetic distortion table.

Run:  python scripts/run_phase0.py  [--dim 128] [--n 50000]

Prints, for b = 1..4 bits/coordinate, the measured normalized MSE of the
rotation + Lloyd-Max quantizer against the paper's upper bound (~2.7 * 2^-2b) and
the optimal Gaussian Lloyd-Max distortion, plus a Haar-vs-Hadamard rotation check.
This is the "scoreboard" referenced in docs/01 and docs/03.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

# Allow running straight from the repo without installing the package.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from turboquant import (  # noqa: E402
    TurboQuantMSE,
    LloydMaxQuantizer,
    normalized_distortion,
    paper_distortion_bound,
    random_signs,
    randomized_hadamard,
    inverse_randomized_hadamard,
)

GAUSSIAN_LLOYD_MAX = {1: 0.3634, 2: 0.1175, 3: 0.03454, 4: 0.009497}


def unit_vectors(n: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d, n = args.dim, args.n
    x = unit_vectors(n, d, seed=123)

    print(f"\nTurboQuant Phase 0 - synthetic distortion  (d={d}, n={n:,} unit vectors)\n")
    header = f"{'bits':>4} | {'measured':>9} | {'optimal(LM)':>11} | {'bound 2.7*2^-2b':>15} | {'meas/bound':>10} | {'meas/floor':>10} | ok"
    print(header)
    print("-" * len(header))

    all_ok = True
    for bits in (1, 2, 3, 4):
        tq = TurboQuantMSE(d, bits, seed=args.seed)
        dist = normalized_distortion(x, tq.reconstruct(x))
        bound = paper_distortion_bound(bits)
        floor = 2.0 ** (-2.0 * bits)
        ok = dist <= bound
        all_ok &= ok
        print(
            f"{bits:>4} | {dist:>9.5f} | {GAUSSIAN_LLOYD_MAX[bits]:>11.5f} | "
            f"{bound:>15.5f} | {dist / bound:>10.3f} | {dist / floor:>10.3f} | "
            f"{'PASS' if ok else 'FAIL'}"
        )

    # Fast-rotation parity check.
    bits = 3
    codebook = LloydMaxQuantizer.for_dimension(d, bits, seed=args.seed)
    haar = TurboQuantMSE(d, bits, seed=args.seed, codebook=codebook)
    haar_dist = normalized_distortion(x, haar.reconstruct(x))
    signs = random_signs(d, seed=args.seed)
    y_hat = codebook.reconstruct(randomized_hadamard(x, signs))
    had_dist = normalized_distortion(x, inverse_randomized_hadamard(y_hat, signs))
    print(
        f"\nrotation parity @3-bit:  Haar={haar_dist:.5f}  Hadamard={had_dist:.5f}  "
        f"(diff {abs(haar_dist - had_dist):.5f})"
    )

    print(f"\nresult: {'ALL PASS - core idea reproduced' if all_ok else 'FAIL'}\n")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
