from __future__ import annotations

import numpy as np


class RestProjection:
    """Brings the final velocity and acceleration of a chunk of jerks toward rest by the smallest change of the jerks
    (CHOMP, Zucker et al., 2013, Sec. 6.1, Eq. 39).

    The final velocity and acceleration of one joint are affine in its jerks u, ``h(u) = C u + b``, with C the (2, H) map
    from the jerks to ``[v_H, a_H]``. CHOMP's update under the constraint ``h = 0`` takes the unconstrained step and
    projects it onto the zero set of ``C`` with the metric A of the trajectory, then removes the offset ``b``; with the
    jerks as the trajectory, ``A = I`` is the squared norm of the change of the third derivative of the positions
    (Eq. 13 with k = 3), and the whole update reduces to

        du = -C^T (C C^T)^-1 w h(u),

    the smallest ``||du||`` that scales the final velocity and acceleration by ``1 - w``: ``w = 1`` is the constraint,
    ``w = 0`` leaves the chunk as it is.
    """

    def __init__(self, terminal_map: np.ndarray) -> None:
        """``terminal_map``: the (2, H) map from the jerks of one joint to its final velocity and acceleration."""
        self._projector = terminal_map.T @ np.linalg.inv(terminal_map @ terminal_map.T)

    def __call__(
        self, chunk: np.ndarray, final_velocity: np.ndarray, final_acceleration: np.ndarray, weight: float
    ) -> np.ndarray:
        """The chunk of jerks (H, n) whose final velocity and acceleration (n,) are scaled by ``1 - weight``."""
        return chunk - weight * (self._projector @ np.stack([final_velocity, final_acceleration]))
