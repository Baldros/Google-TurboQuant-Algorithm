"""Phase 2 scoreboard - vector search recall@k on GloVe-200 vs FAISS Product Quantization.

Run:  python scripts/run_phase2.py [--queries 1000] [--db-limit N] [--quick]

The claim under test (docs/03 Phase 2 DoD): at a **matched bit-rate**, TurboQuant's
data-*oblivious* quantization reaches **recall@k >= FAISS PQ** (whose codebooks are *learned*
from the data). Every method here scans the whole database exhaustively with its approximate
score, so the only difference being measured is the compression, not any ANN graph.

Dataset: ``data/glove-200-angular.hdf5`` (ann-benchmarks; 1.18M x 200, angular/cosine).
We L2-normalize so cosine-NN == MIPS == nearest-L2, and take our own exact top-k as the
ground truth (so the "exact" row is 1.0 by construction and isolates quantization error;
agreement with the dataset's own neighbour list is reported as an external sanity check).

Bit accounting (d=200, vectors normalized so the per-vector norm costs ~0 bits):
  * TurboQuant Stage-1 at b bits/coord -> b*d code bits/vector.
  * QJL sign sketch with m bits        -> m bits/vector (m=d -> 1 bit/coord).
  * TurboQuantProd (Stage-1+2)         -> b*d + m bits/vector.
  * FAISS PQ with M subquantizers      -> M*nbits bits/vector (nbits=8).
Matched operating points: 1, 2 and 4 bits/coordinate (200 / 400 / 800 bits/vector).
"""

from __future__ import annotations

import argparse
import gc
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from turboquant import (  # noqa: E402
    QJLIndex,
    Stage1Index,
    TurboQuantProdIndex,
    exact_search,
    recall_at_k,
)

DATA = pathlib.Path(__file__).resolve().parents[1] / "data" / "glove-200-angular.hdf5"
KS = (1, 10, 100)


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    np.divide(x, np.where(n == 0.0, 1.0, n), out=x)
    return x


def load_glove(db_limit: int | None, n_queries: int):
    import h5py

    if not DATA.exists():
        sys.exit(f"missing dataset: {DATA}\n  download: http://ann-benchmarks.com/glove-200-angular.hdf5")
    with h5py.File(DATA, "r") as f:
        train = f["train"][:] if db_limit is None else f["train"][:db_limit]
        test = f["test"][:n_queries]
        gt = f["neighbors"][:n_queries]  # external cross-check only
    return _normalize(train), _normalize(test), gt


# --------------------------------------------------------------------------- #
# FAISS Product Quantization baseline (L2 on normalized vectors == cosine/MIPS).
# --------------------------------------------------------------------------- #
def faiss_pq_search(train, queries, k, M, nbits, train_subsample):
    import faiss

    d = train.shape[1]
    index = faiss.IndexPQ(d, M, nbits)  # default metric L2
    xt = train if train.shape[0] <= train_subsample else train[
        np.random.default_rng(0).choice(train.shape[0], train_subsample, replace=False)
    ]
    index.train(np.ascontiguousarray(xt))
    index.add(np.ascontiguousarray(train))
    _, idx = index.search(np.ascontiguousarray(queries), k)
    del index
    return idx


def _bits(label, d):
    return {"per_vec": label, "per_dim": label / d}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", type=int, default=1000, help="number of test queries to evaluate")
    ap.add_argument("--db-limit", type=int, default=None, help="cap database size (quick runs)")
    ap.add_argument("--train-subsample", type=int, default=100_000, help="vectors used to train PQ")
    ap.add_argument("--quick", action="store_true", help="500 queries, 100k DB - fast smoke test")
    args = ap.parse_args()
    if args.quick:
        args.queries, args.db_limit = 500, 100_000

    t0 = time.time()
    train, queries, gt_external = load_glove(args.db_limit, args.queries)
    n, d = train.shape
    kmax = max(KS)
    print(f"\nTurboQuant Phase 2 - vector search recall on GloVe-200")
    print(f"  database {n:,} x {d}   queries {queries.shape[0]:,}   loaded in {time.time()-t0:.1f}s\n")

    # Ground truth = our own exact top-kmax (recall is then purely quantization error).
    t = time.time()
    truth = exact_search(train, queries, kmax)
    ext = recall_at_k(truth, gt_external, 100)
    print(f"  exact ground truth computed in {time.time()-t:.1f}s  "
          f"(agrees with dataset neighbours @100: {ext:.3f})\n")

    # ---- methods grouped by matched bit-rate -------------------------------- #
    # Each entry: (group_bits_per_dim, label, builder -> index-with .search, is_pq, bits_per_vec)
    groups = [
        (1, [
            ("Stage1 b=1",        lambda: Stage1Index(d, 1, seed=0).add(train),      1 * d),
            ("QJL m=200",         lambda: QJLIndex(d, m=d, seed=0).add(train),       d),
            ("FAISS PQ M=25",     ("pq", 25, 8),                                     25 * 8),
        ]),
        (2, [
            ("Stage1 b=2",        lambda: Stage1Index(d, 2, seed=0).add(train),      2 * d),
            ("Prod b=1,m=200",    lambda: TurboQuantProdIndex(d, 1, d, seed=0).add(train), 1 * d + d),
            ("FAISS PQ M=50",     ("pq", 50, 8),                                     50 * 8),
        ]),
        (4, [
            ("Stage1 b=4",        lambda: Stage1Index(d, 4, seed=0).add(train),      4 * d),
            ("Prod b=3,m=200",    lambda: TurboQuantProdIndex(d, 3, d, seed=0).add(train), 3 * d + d),
            ("FAISS PQ M=100",    ("pq", 100, 8),                                    100 * 8),
        ]),
    ]

    hdr = f"{'method':<16} {'b/vec':>6} {'b/dim':>6} | " + " ".join(f"R@{k:<4}" for k in KS) + f" | {'sec':>6}"
    all_pass = True
    for bpd, methods in groups:
        print(f"--- matched bit-rate: {bpd} bit/dim ({bpd*d} bits/vector) " + "-" * 20)
        print(hdr)
        print("-" * len(hdr))
        tq_best = {k: 0.0 for k in KS}
        pq_recall = {k: None for k in KS}
        for label, builder, bits_per_vec in methods:
            ts = time.time()
            if isinstance(builder, tuple) and builder[0] == "pq":
                _, M, nbits = builder
                found = faiss_pq_search(train, queries, kmax, M, nbits, args.train_subsample)
                is_pq = True
            else:
                index = builder()
                found = index.search(queries, kmax)
                del index
                is_pq = False
            recalls = {k: recall_at_k(found, truth, k) for k in KS}
            dt = time.time() - ts
            row = f"{label:<16} {bits_per_vec:>6} {bits_per_vec/d:>6.2f} | "
            row += " ".join(f"{recalls[k]:>5.3f}" for k in KS) + f" | {dt:>6.1f}"
            print(row)
            if is_pq:
                pq_recall = recalls
            else:
                for k in KS:
                    tq_best[k] = max(tq_best[k], recalls[k])
            gc.collect()

        passed = tq_best[10] >= pq_recall[10] - 1e-6
        all_pass &= passed
        print(f"  -> best TurboQuant R@10={tq_best[10]:.3f} vs FAISS PQ R@10={pq_recall[10]:.3f}  "
              f"=> {'PASS' if passed else 'FAIL'}\n")

    print(f"result: {'ALL PASS - TurboQuant matches/beats PQ at every matched bit-rate' if all_pass else 'FAIL'}")
    print(f"total {time.time()-t0:.1f}s\n")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
