from __future__ import annotations

import numpy as np

VELOCITY_WEIGHT = 1.0  # w_1 of the smoothness metric: squared first differences
ACCELERATION_WEIGHT = 1.0  # w_2: squared second differences
METRIC_RIDGE = 1e-3  # keeps the metric positive definite where both differences vanish
SPREAD_STRENGTH = 5.0  # mu: how far a correction is spread along the chunk (0 leaves it as it is)


def smoothness_metric(horizon: int) -> np.ndarray:
    """The metric ``A = w_1 K_1^T K_1 + w_2 K_2^T K_2 + ridge I`` (H, H) on a sequence of ``horizon`` values (CHOMP,
    Ratliff et al., 2009, Eq. 1), ``K_d`` the finite difference of order d with the value before the sequence fixed at zero:
    the norm of a change of the sequence is its size in velocity and acceleration, so a spiky change is large and a
    smooth one small."""
    first = np.eye(horizon) - np.eye(horizon, k=-1)
    second = first @ first
    return VELOCITY_WEIGHT * first.T @ first + ACCELERATION_WEIGHT * second.T @ second + METRIC_RIDGE * np.eye(horizon)


class SmoothSpread:
    """Spreads a correction of a sequence along it by the smoothness metric (CHOMP, Sec. II-A and II-B).

    The covariant update ``-A^-1 g`` spreads the gradient ``g`` over the whole trajectory instead of changing one point of
    it; ``(I + mu A)^-1`` does so in a bounded way: it is the change ``d`` minimizing ``||d - g||^2 + mu ||d||_A^2``, a
    smoothed ``g`` of at most its size.
    """

    def __init__(self, horizon: int, strength: float = SPREAD_STRENGTH) -> None:
        self.matrix = np.linalg.inv(np.eye(horizon) + strength * smoothness_metric(horizon))

    def __call__(self, correction: np.ndarray) -> np.ndarray:
        """The correction (H, columns) spread along its first axis."""
        return self.matrix @ correction

