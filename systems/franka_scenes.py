"""Obstacle scenes for the Franka arm: random ones that block the nominal motion and hand-made ones.

A scene is a list of spheres ``(center, radius)``; ``pad_scene`` fills it up to the fixed number of obstacle slots the
policy observes and the planners are built for (an unused slot is a tiny sphere far below the floor). Every scene is
generated from a start configuration and a goal configuration, and keeps ``OBSTACLE_MARGIN`` away from the arm at both,
so the start and the goal stay free. The nominal motion is the straight line in joint space from start to goal, the
path an arm takes when nothing is in its way:

``blocking``  ``count`` random spheres, the first one always on the arm's nominal sweep, each other one with
              probability ``BLOCKING_FRACTION`` (the rest anywhere in the workspace, as distractors).
``behind``    one or two spheres on the straight line of the hand from start to goal: the goal is behind them.
``passage``   a 3 x 3 wall of spheres across the hand's line with the center sphere left out: a passage of the given
              free width the hand has to go through (the wall is about 0.6 m wide, the arm can also go around it).
``corner``    two perpendicular rows of four spheres forming an L: one across the line, one running back along it from
              one of its ends, so the straight way is closed on two sides.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .franka_system import FLOOR_CLEARANCE, WORKSPACE_BOUNDS, FrankaSystem

Scene = list[tuple[np.ndarray, float]]

OBSTACLE_MARGIN = 0.03  # m: the gap an obstacle keeps to the arm at the start and at the goal configuration
NOMINAL_NODES = 25  # nodes of the nominal joint-space path
BLOCKING_FRACTION = 0.5
RADIUS_RANGE = (0.05, 0.09)
UNUSED_SLOT = (np.array([0.0, 0.0, -5.0]), 0.01)
PLACEMENT_ATTEMPTS = 100


def pad_scene(scene: Scene, slots: int) -> Scene:
    if len(scene) > slots:
        raise ValueError(f"A scene of {len(scene)} obstacles does not fit in {slots} slots.")
    return [(position.copy(), radius) for position, radius in scene] + [(UNUSED_SLOT[0].copy(), UNUSED_SLOT[1])] * (
        slots - len(scene)
    )


def clearance(simulator: FrankaSystem, q: np.ndarray, scene: Scene) -> float:
    """Smallest gap [m] between a bounding sphere of the arm at ``q`` and an obstacle of ``scene``."""
    centers, _, radii = simulator.collision_spheres(q)
    return min((float(np.min(np.linalg.norm(centers - position, axis=1) - radii - radius)) for position, radius in scene), default=np.inf)


def nominal_path(q_start: np.ndarray, q_goal: np.ndarray) -> np.ndarray:
    return q_start + np.linspace(0.0, 1.0, NOMINAL_NODES)[:, None] * (q_goal - q_start)


def blocks_nominal_path(simulator: FrankaSystem, q_start: np.ndarray, q_goal: np.ndarray, scene: Scene) -> bool:
    return any(clearance(simulator, q, scene) < 0.0 for q in nominal_path(q_start, q_goal))


def _is_placeable(simulator: FrankaSystem, position: np.ndarray, radius: float, q_start: np.ndarray, q_goal: np.ndarray) -> bool:
    """Above the floor, inside the workspace box (grown by the radius), and clear of the arm at the start and the goal."""
    low = np.array([WORKSPACE_BOUNDS[0][0], WORKSPACE_BOUNDS[1][0], max(WORKSPACE_BOUNDS[2][0], FLOOR_CLEARANCE)]) - radius
    high = np.array([bound[1] for bound in WORKSPACE_BOUNDS]) + radius
    if np.any(position < low) or np.any(position > high):
        return False
    sphere = [(position, radius)]
    return all(clearance(simulator, q, sphere) >= OBSTACLE_MARGIN for q in (q_start, q_goal))


def _line_frame(simulator: FrankaSystem, q_start: np.ndarray, q_goal: np.ndarray):
    """The hand's start and goal position and the orthonormal frame (line direction, horizontal, vertical) across it."""
    start, goal = simulator.tcp_position(q_start), simulator.tcp_position(q_goal)
    direction = (goal - start) / np.linalg.norm(goal - start)
    horizontal = np.cross(direction, [0.0, 0.0, 1.0])
    horizontal = horizontal / np.linalg.norm(horizontal) if np.linalg.norm(horizontal) > 1e-6 else np.array([0.0, 1.0, 0.0])
    return start, goal, direction, horizontal, np.cross(direction, horizontal)


def blocking_scene(simulator: FrankaSystem, q_start: np.ndarray, q_goal: np.ndarray, rng: np.random.Generator, count: int) -> Scene:
    path = nominal_path(q_start, q_goal)
    sweep = [simulator.collision_spheres(q)[0:3:2] for q in path]  # the arm's sphere centers and radii at every node
    scene: Scene = []
    for index in range(count):
        blocking = index == 0 or rng.random() < BLOCKING_FRACTION
        for _ in range(PLACEMENT_ATTEMPTS):
            radius = rng.uniform(*RADIUS_RANGE)
            if blocking:
                centers, radii = sweep[rng.integers(2, NOMINAL_NODES - 2)]
                arm_sphere = rng.integers(len(radii))
                offset = rng.normal(size=3)
                position = centers[arm_sphere] + offset / np.linalg.norm(offset) * rng.uniform(0.0, 0.5 * (radii[arm_sphere] + radius))
            else:
                position = rng.uniform(
                    [WORKSPACE_BOUNDS[0][0], WORKSPACE_BOUNDS[1][0], WORKSPACE_BOUNDS[2][0]],
                    [bound[1] for bound in WORKSPACE_BOUNDS],
                )
            if _is_placeable(simulator, position, radius, q_start, q_goal):
                scene.append((position, radius))
                break
        else:
            raise RuntimeError("Unable to place an obstacle.")
    return scene


def behind_scene(simulator: FrankaSystem, q_start: np.ndarray, q_goal: np.ndarray, rng: np.random.Generator, count: int = 0) -> Scene:
    start, goal, *_ = _line_frame(simulator, q_start, q_goal)
    scene: Scene = []
    for _ in range(rng.integers(1, 3)):
        radius = rng.uniform(0.07, 0.11)
        position = start + rng.uniform(0.4, 0.75) * (goal - start) + rng.normal(scale=0.02, size=3)
        if not _is_placeable(simulator, position, radius, q_start, q_goal):
            raise RuntimeError("Unable to place an obstacle.")
        scene.append((position, radius))
    return scene


def passage_scene(simulator: FrankaSystem, q_start: np.ndarray, q_goal: np.ndarray, rng: np.random.Generator, count: int = 0) -> Scene:
    start, goal, direction, horizontal, vertical = _line_frame(simulator, q_start, q_goal)
    radius, width = 0.10, rng.uniform(0.18, 0.26)  # the free width of the passage between opposite spheres
    spacing = (width + 2 * radius) / 2
    center = start + rng.uniform(0.4, 0.6) * (goal - start) + rng.normal(scale=0.02, size=3)
    scene: Scene = []
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            if (i, j) != (0, 0):
                position = center + spacing * (i * horizontal + j * vertical)
                if _is_placeable(simulator, position, radius, q_start, q_goal):
                    scene.append((position, radius))
    if len(scene) < 6:
        raise RuntimeError("Unable to place a wall.")
    return scene


def corner_scene(simulator: FrankaSystem, q_start: np.ndarray, q_goal: np.ndarray, rng: np.random.Generator, count: int = 0) -> Scene:
    start, goal, direction, horizontal, vertical = _line_frame(simulator, q_start, q_goal)
    radius = rng.uniform(0.06, 0.08)
    apex = start + rng.uniform(0.4, 0.6) * (goal - start) + rng.normal(scale=0.02, size=3)
    side = rng.choice([-1.0, 1.0])
    scene: Scene = []
    for k in range(4):
        scene.append((apex + side * k * 1.6 * radius * horizontal, radius))  # across the line
        if k > 0:
            scene.append((apex + side * 3 * 1.6 * radius * horizontal + k * 1.6 * radius * direction, radius))  # back along it
    scene = [(position, r) for position, r in scene if _is_placeable(simulator, position, r, q_start, q_goal)]
    if len(scene) < 6:
        raise RuntimeError("Unable to place a corner.")
    return scene


SCENE_FAMILIES: dict[str, Callable[..., Scene]] = {
    "blocking": blocking_scene,
    "behind": behind_scene,
    "passage": passage_scene,
    "corner": corner_scene,
}


def sample_scene(
    family: str, simulator: FrankaSystem, q_start: np.ndarray, q_goal: np.ndarray, rng: np.random.Generator, count: int = 0
) -> Scene:
    """A scene of ``family`` for the given start and goal; ``count`` is the number of obstacles of the random family."""
    for _ in range(PLACEMENT_ATTEMPTS):
        try:
            scene = SCENE_FAMILIES[family](simulator, q_start, q_goal, rng, count)
        except RuntimeError:
            continue
        if scene:
            return scene
    raise RuntimeError(f"Unable to generate a '{family}' scene.")
