from __future__ import annotations

import numpy as np

from learning.models.smoothness import smoothness_metric


class RestProjection:
    """Brings the final velocity and acceleration of a chunk of positions toward rest by the smallest change of the positions
    in the smoothness metric of CHOMP (Zucker et al., 2013, Sec. 3.4 and 6.1, Eq. 39).

    The final velocity and acceleration of one joint are linear in its last three positions (backward differences),
    ``h(q) = C q`` with the (2, H) map C onto ``[v_H, a_H]``. CHOMP's update under the constraint ``h = 0`` takes the
    unconstrained step and projects it onto the zero set of C with the metric A of the trajectory, then removes the offset;
    with no unconstrained step the whole update is the offset term, and with the metric A of ``smoothness_metric`` (the squared
    first and second differences of the change of the positions, the smoothness of the trajectory of Sec. 3.4) it is

        dq = -A^-1 C^T (C A^-1 C^T)^-1 w h(q),

    the smooth change of the positions that scales the final velocity and acceleration by ``1 - w``: ``w = 1`` is the
    constraint, ``w = 0`` leaves the chunk as it is. The change is spread along the chunk by A^-1, as in CHOMP's goal-set
    update (Sec. 6.3), instead of landing on its last positions.
    """

    def __init__(self, horizon: int, dt: float) -> None:
        if horizon < 3:
            raise ValueError("A chunk needs at least three positions to have a final acceleration.")
        self._rest_map = np.zeros((2, horizon))
        self._rest_map[0, -2:] = np.array([-1.0, 1.0]) / dt
        self._rest_map[1, -3:] = np.array([1.0, -2.0, 1.0]) / dt**2
        metric_inverse_transpose = np.linalg.solve(smoothness_metric(horizon), self._rest_map.T)
        self._projector = metric_inverse_transpose @ np.linalg.inv(self._rest_map @ metric_inverse_transpose)

    def __call__(self, chunk: np.ndarray, weight: float) -> np.ndarray:
        """The chunk of positions (H, n) whose final velocity and acceleration are scaled by ``1 - weight``."""
        return chunk - weight * (self._projector @ (self._rest_map @ chunk))
