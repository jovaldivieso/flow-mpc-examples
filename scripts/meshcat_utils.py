"""Shared Meshcat drawing helpers for the visualization scripts."""

import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np

GREEN, ORANGE = np.array([0.1, 0.8, 0.2]), np.array([1.0, 0.55, 0.0])


def draw_frame(viz, name: str, length: float = 0.1, radius: float = 0.005) -> None:
    """Create a 3-D RGB coordinate frame (X=red, Y=green, Z=blue) with solid cylinders at ``viz.viewer[name]``."""
    axes = (
        ("x", 0xFF2020, [length / 2, 0, 0], tf.rotation_matrix(-np.pi / 2, [0, 0, 1])),
        ("y", 0x20D040, [0, length / 2, 0], np.eye(4)),
        ("z", 0x2060FF, [0, 0, length / 2], tf.rotation_matrix(np.pi / 2, [1, 0, 0])),
    )
    for axis, color, offset, rotation in axes:
        viz.viewer[name][axis].set_object(g.Cylinder(length, radius), g.MeshLambertMaterial(color=color))
        viz.viewer[name][axis].set_transform(tf.translation_matrix(offset) @ rotation)


def plan_tcp_points(dyn, plan: np.ndarray) -> np.ndarray:
    """TCP positions (H, 3) of a joint-position plan (H, n)."""
    return np.array([dyn.tcp_position(q) for q in plan])


def action_tcp_points(dyn, state: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """TCP positions (H + 1, 3) of the motion of the plant applying the (H, nu) ``actions`` from ``state``."""
    return plan_tcp_points(dyn, dyn.rollout_positions(state, actions))


def draw_plan(viz, name: str, points: np.ndarray, color: int) -> None:
    """Draw (or replace) the polyline ``name`` through the (H, 3) points, ``color`` as 0xRRGGBB."""
    viz.viewer[name].set_object(g.Line(g.PointsGeometry(points.T), g.LineBasicMaterial(color=color, linewidth=3)))


def draw_tcp_paths(viz, paths: list[np.ndarray], goals: list[np.ndarray], reached: list[bool]) -> None:
    """Draw every TCP path (green if it reached its goal, orange otherwise), the starts (blue) and the goals (red).

    paths: one (T, 3) TCP position array per episode; goals: one goal TCP position (3,) per episode.
    """
    starts_of_segments = np.concatenate([path[:-1] for path in paths])
    ends_of_segments = np.concatenate([path[1:] for path in paths])
    colors = np.concatenate([np.tile(GREEN if ok else ORANGE, (len(path) - 1, 1)) for path, ok in zip(paths, reached)])
    segments = np.empty((2 * len(starts_of_segments), 3))
    segments[0::2], segments[1::2] = starts_of_segments, ends_of_segments
    vertex_colors = np.repeat(colors, 2, axis=0)
    viz.viewer["paths"].set_object(
        g.LineSegments(g.PointsGeometry(segments.T, color=vertex_colors.T), g.LineBasicMaterial(vertexColors=True))
    )
    for name, points, color, size in (
        ("starts", [path[0] for path in paths], [0.1, 0.4, 1.0], 0.02),
        ("goals", list(goals), [1.0, 0.1, 0.1], 0.03),
    ):
        cloud = np.array(points).T
        colors_ = np.tile(np.array(color)[:, None], (1, cloud.shape[1]))
        viz.viewer[name].set_object(g.Points(g.PointsGeometry(cloud, color=colors_), g.PointsMaterial(size=size)))


def draw_obstacles(viz, obstacles: list[tuple[np.ndarray, float]]) -> None:
    """Draw spherical obstacles ``[(position, radius), ...]`` in translucent red at ``viz.viewer["obstacles"]``."""
    viz.viewer["obstacles"].delete()
    for i, (position, radius) in enumerate(obstacles):
        pos_3d = np.pad(np.asarray(position, dtype=float)[:3], (0, max(0, 3 - len(position))))
        viz.viewer["obstacles"][str(i)].set_object(
            g.Sphere(float(radius)),
            g.MeshLambertMaterial(color=0xE03030, opacity=0.5),
        )
        viz.viewer["obstacles"][str(i)].set_transform(tf.translation_matrix(pos_3d))
