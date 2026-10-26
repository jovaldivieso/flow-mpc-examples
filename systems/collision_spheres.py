"""Bounding spheres of a robot's collision meshes, fitted from the meshes themselves.

For every moving joint, the surface points of the meshes attached to it (in the joint frame) are split recursively along
their principal axis, at the median, until the smallest sphere enclosing a part has a radius of at most ``max_radius``.
The spheres overlap and together cover the surface; a larger ``max_radius`` gives fewer, looser spheres, and a smaller one
hugs the robot more tightly at the cost of more spheres (every sphere costs a slack variable per obstacle and node in the
planners).
"""

import coal
import numpy as np
import pinocchio as pin
from scipy.spatial import ConvexHull

SURFACE_SAMPLES = 4000  # random surface points per mesh, besides its vertices
RADIUS_MARGIN = 0.002  # [m] added to every fitted radius
MIN_POINTS = 20
MAX_DEPTH = 6
ENCLOSING_ITERATIONS = 400


def mesh_triangles(geometry: coal.CollisionGeometry) -> tuple[np.ndarray, np.ndarray]:
    """The vertices (V, 3) and triangle indices (T, 3) of a mesh or box."""
    if isinstance(geometry, coal.Box):
        half = np.asarray(geometry.halfSide)
        corners = np.array([[x * half[0], y * half[1], z * half[2]] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        return corners, ConvexHull(corners).simplices
    triangles = np.array([[geometry.tri_indices(i)[k] for k in range(3)] for i in range(geometry.num_tris)])
    return np.asarray(geometry.vertices()), triangles


def surface_points(geometry: coal.CollisionGeometry, placement: pin.SE3, rng: np.random.Generator) -> np.ndarray:
    """The vertices and area-weighted random surface points of a mesh, in the frame of its joint."""
    vertices, triangles = mesh_triangles(geometry)
    a, b, c = vertices[triangles[:, 0]], vertices[triangles[:, 1]], vertices[triangles[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    chosen = rng.choice(len(triangles), SURFACE_SAMPLES, p=area / area.sum())
    u, v = rng.random(SURFACE_SAMPLES), rng.random(SURFACE_SAMPLES)
    fold = u + v > 1
    u[fold], v[fold] = 1 - u[fold], 1 - v[fold]
    points = np.vstack([vertices, a[chosen] + u[:, None] * (b[chosen] - a[chosen]) + v[:, None] * (c[chosen] - a[chosen])])
    return points @ placement.rotation.T + placement.translation


def enclosing_ball(points: np.ndarray) -> tuple[np.ndarray, float]:
    """The (approximately) smallest ball containing the points (Badoiu-Clarkson), as center and radius."""
    center = points.mean(axis=0)
    for i in range(1, ENCLOSING_ITERATIONS + 1):
        farthest = points[np.argmax(np.linalg.norm(points - center, axis=1))]
        center = center + (farthest - center) / (i + 1)
    return center, float(np.linalg.norm(points - center, axis=1).max())


def split_into_spheres(points: np.ndarray, max_radius: float, depth: int = 0) -> list[tuple[np.ndarray, float]]:
    """Enclosing balls of the points, split along the principal axis until each has a radius of at most ``max_radius``."""
    center, radius = enclosing_ball(points)
    if radius <= max_radius or len(points) < MIN_POINTS or depth >= MAX_DEPTH:
        return [(center, radius)]
    offset = points - points.mean(axis=0)
    projection = offset @ np.linalg.svd(offset, full_matrices=False)[2][0]
    below = projection <= np.median(projection)
    return split_into_spheres(points[below], max_radius, depth + 1) + split_into_spheres(points[~below], max_radius, depth + 1)


def fit_collision_spheres(
    collision_model: pin.GeometryModel, max_radius: float, seed: int = 0
) -> list[tuple[int, np.ndarray, float]]:
    """The bounding spheres of the meshes of every moving joint (the base is skipped) as ``(joint, center, radius)``, the
    center in the frame of the joint. Deterministic for a given ``seed``."""
    rng = np.random.default_rng(seed)
    points_of_joint: dict[int, list[np.ndarray]] = {}
    for geometry in collision_model.geometryObjects:
        if geometry.parentJoint != 0:
            points_of_joint.setdefault(geometry.parentJoint, []).append(surface_points(geometry.geometry, geometry.placement, rng))
    return [
        (joint, center, radius + RADIUS_MARGIN)
        for joint in sorted(points_of_joint)
        for center, radius in split_into_spheres(np.vstack(points_of_joint[joint]), max_radius)
    ]
