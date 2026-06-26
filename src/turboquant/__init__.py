"""turboquant - a validated replication of TurboQuant.

Phase 0 surface: random rotation + per-coordinate Lloyd-Max scalar quantization,
enough to reproduce the paper's synthetic distortion scoreboard.
"""

from .rotation import (
    random_orthogonal,
    apply_rotation,
    apply_inverse_rotation,
    fast_hadamard_transform,
    random_signs,
    randomized_hadamard,
    inverse_randomized_hadamard,
)
from .scalar_quant import (
    LloydMaxQuantizer,
    fit_lloyd_max,
    unit_vector_coordinate_samples,
)
from .quantizers import TurboQuantMSE, QuantizedVectors
from .qjl import (
    QJL,
    QJLSketch,
    gaussian_projection,
    qjl_estimator_variance,
)
from .search import (
    Stage1Index,
    QJLIndex,
    TurboQuantProdIndex,
    exact_search,
)
from .attention import (
    softmax,
    causal_additive_mask,
    attention,
    attention_from_scores,
    attention_from_prescaled,
    mse_key_scores,
    prod_key_scores,
)
from .metrics import (
    normalized_distortion,
    mean_squared_error,
    paper_distortion_bound,
    recall_at_k,
    cosine_similarity_rows,
    kl_divergence_rows,
)

__version__ = "0.0.1"

__all__ = [
    "random_orthogonal",
    "apply_rotation",
    "apply_inverse_rotation",
    "fast_hadamard_transform",
    "random_signs",
    "randomized_hadamard",
    "inverse_randomized_hadamard",
    "LloydMaxQuantizer",
    "fit_lloyd_max",
    "unit_vector_coordinate_samples",
    "TurboQuantMSE",
    "QuantizedVectors",
    "QJL",
    "QJLSketch",
    "gaussian_projection",
    "qjl_estimator_variance",
    "Stage1Index",
    "QJLIndex",
    "TurboQuantProdIndex",
    "exact_search",
    "softmax",
    "causal_additive_mask",
    "attention",
    "attention_from_scores",
    "attention_from_prescaled",
    "mse_key_scores",
    "prod_key_scores",
    "normalized_distortion",
    "mean_squared_error",
    "paper_distortion_bound",
    "recall_at_k",
    "cosine_similarity_rows",
    "kl_divergence_rows",
    "__version__",
]
