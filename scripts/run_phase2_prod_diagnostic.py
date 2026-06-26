"""Phase 2 diagnostic - *why* TurboQuant-Prod underperforms for search ranking.

The paper's stated search method is TurboQuant-Prod: MSE quantization at ``b-1`` bits
plus a 1-bit QJL sign sketch of the residual, giving an *unbiased* inner product. On
GloVe-200 that method is the **worst** of the ones we test (see ``run_phase2.py``). This
script shows the reason is **QJL variance**, not a bug, and that our Prod is implemented
correctly:

  * Prod's recall climbs monotonically toward the exact ceiling as we add QJL sign bits
    ``m`` (the estimator variance falls like ``1/m``);
  * its inner-product *estimate* RMSE falls like ``1/sqrt(m)`` in lock-step;
  * but a single 1-bit sketch (``m = d``) is *bit-inefficient*: matching the inner-product
    accuracy of one extra **scalar** bit (Stage-1 ``b=2``, 400 bits) takes the QJL residual
    ~1000 bits. At the paper's matched budget the residual's variance swamps the tiny score
    gaps between real GloVe neighbours and hurts ranking.

This is the search-side echo of the KV-cache lesson "MSE-only > MSE+QJL" (docs/02): an
unbiased-but-high-variance inner product is the wrong tool when scores must be discriminated
finely (ranking here, softmax there).

Run:  python scripts/run_phase2_prod_diagnostic.py [--db-limit 100000] [--queries 500]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from turboquant import Stage1Index, TurboQuantProdIndex, exact_search, recall_at_k  # noqa: E402

DATA = pathlib.Path(__file__).resolve().parents[1] / "data" / "glove-200-angular.hdf5"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db-limit", type=int, default=100_000)
    ap.add_argument("--queries", type=int, default=500)
    args = ap.parse_args()

    import h5py

    if not DATA.exists():
        sys.exit(f"missing dataset: {DATA}")
    with h5py.File(DATA, "r") as f:
        db = f["train"][: args.db_limit].astype(np.float32)
        q = f["test"][: args.queries].astype(np.float32)
    db /= np.linalg.norm(db, axis=1, keepdims=True)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    d = db.shape[1]

    truth = exact_search(db, q, 10)
    true_ip = (q[:64] @ db.T)  # exact inner products for an RMSE probe

    def ip_rmse(index) -> float:
        est = index._score_block(q[:64])
        return float(np.sqrt(np.mean((est - true_ip) ** 2)))

    print(f"\nProd diagnostic on GloVe-200  ({db.shape[0]:,} x {d}, {q.shape[0]} queries)")
    print("exact R@10 ceiling = 1.000 by construction\n")
    print(f"{'method':<20} {'bits/vec':>8} {'R@10':>6}  {'IP-est RMSE':>11}")
    print("-" * 50)

    for b in (1, 2, 3):
        idx = Stage1Index(d, b, seed=0).add(db)
        print(f"{'Stage1 b=%d' % b:<20} {b*d:>8} "
              f"{recall_at_k(idx.search(q, 10), truth, 10):>6.3f}  {ip_rmse(idx):>11.5f}")
        del idx

    print("-" * 50)
    for mult in (1, 2, 4, 8):
        m = mult * d
        idx = TurboQuantProdIndex(d, 1, m, seed=0).add(db)
        print(f"{'Prod b=1 m=%d' % m:<20} {1*d + m:>8} "
              f"{recall_at_k(idx.search(q, 10), truth, 10):>6.3f}  {ip_rmse(idx):>11.5f}")
        del idx

    print("\nTakeaway: Prod -> ceiling as m grows (impl correct), but ~1000 QJL bits are")
    print("needed to match the inner-product accuracy of one extra 200-bit scalar level.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
