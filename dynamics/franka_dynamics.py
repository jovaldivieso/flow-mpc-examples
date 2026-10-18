"""Franka FR3 arm (7 joints) with the gripper as fixed geometry.

State  x = [q, v, a]  (joint positions, velocities, accelerations), nx = 21
Action u = j          (joint jerks), nu = 7
Goal   g = target joint configuration q_goal
Observation o = [g, x] (obs_dim = 28); the LeRobot frame stores g as ``observation.environment_state`` and x as
``observation.state``.

Kinematic model: every joint is a triple integrator with piecewise constant jerk, with SafeFlowMPC's limits (the
joint position limits narrowed by ``JOINT_LIMIT_MARGIN``, the velocity limits scaled by ``VELOCITY_LIMIT_SCALE``,
acceleration and jerk limits from the config). No masses or torques: a joint tracking controller is assumed to
follow the motion.

The model comes from the franka_description xacro via Pinocchio, the finger joints locked open. Start and goal
configurations are sampled well inside the joint limits, and there is no collision checking.
"""

import functools
from pathlib import Path
from typing import Any, Mapping

import casadi as ca
import numpy as np
import pinocchio as pin
import xacro
from pinocchio.robot_wrapper import RobotWrapper
from xacro import substitution_args

from .dynamics import DynamicsSimulator, constant_jerk_step, saturated_step

FRANKA_NUM_JOINTS = 7
FRANKA_DEFAULT_MAX_ACCELERATION = (5.0,) * 7
FRANKA_DEFAULT_MAX_JERK = (15.0,) * 7
FRANKA_DEFAULT_WORKSPACE_BOUNDS = ((0.2, 1.0), (-1.0, 1.0), (0.1, 0.8))

# SafeFlowMPC's joint limits: positions 3 degrees inside the description's, velocities at 92 %
JOINT_LIMIT_MARGIN = np.deg2rad(3.0)
VELOCITY_LIMIT_SCALE = 0.92
# Random start/goal configurations keep this fraction of every joint range away from the limits
JOINT_SAMPLING_MARGIN = 0.1
SAMPLING_ATTEMPTS = 256

# Official franka_description, pinned as a git submodule
FRANKA_DESCRIPTION = Path(__file__).resolve().parents[1] / "third_party" / "franka_description"
XACRO_PATH = FRANKA_DESCRIPTION / "robots" / "fr3" / "fr3.urdf.xacro"

# xacro resolves $(find pkg) via ament_index_python (ROS-only, not on PyPI)
substitution_args._eval_find = lambda pkg: str(FRANKA_DESCRIPTION)

# The finger joints are locked at this opening, the hand and fingers are fixed geometry
FINGER_OPEN = 0.04
FINGER_JOINTS = ("fr3_finger_joint1", "fr3_finger_joint2")
Q_READY = np.array([0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4])
TCP_FRAME = "fr3_hand_tcp"


@functools.lru_cache(maxsize=None)
def _urdf() -> str:
    return xacro.process_file(XACRO_PATH, mappings={"hand": "true"}).toxml()


def _finger_locks(full_model: pin.Model) -> tuple[list[int], np.ndarray]:
    """Ids of the finger joints and the full 9-D configuration (arm at its ready pose, fingers open) they are locked at."""
    ids = [full_model.getJointId(name) for name in FINGER_JOINTS]
    return ids, np.concatenate([Q_READY, [FINGER_OPEN] * len(FINGER_JOINTS)])


def build_model() -> pin.Model:
    """The 7-DoF FR3 with the finger joints locked open and the limits of SafeFlowMPC (kinematics only)."""
    full = pin.buildModelFromXML(_urdf())
    ids, reference = _finger_locks(full)
    model = pin.buildReducedModel(full, ids, reference)
    model.lowerPositionLimit = model.lowerPositionLimit + JOINT_LIMIT_MARGIN
    model.upperPositionLimit = model.upperPositionLimit - JOINT_LIMIT_MARGIN
    model.velocityLimit = VELOCITY_LIMIT_SCALE * model.velocityLimit
    return model


def build_robot() -> RobotWrapper:
    """The 7-DoF FR3 with collision and visual geometry (for Meshcat), the hand and fingers fixed on the last joint."""
    urdf = _urdf()
    # meshes are package://franka_description/..., resolved from the clone's parent
    pkg = [str(FRANKA_DESCRIPTION.parent)]
    full = pin.buildModelFromXML(urdf)
    visual = pin.buildGeomFromUrdfString(full, urdf, pin.GeometryType.VISUAL, package_dirs=pkg)
    collision = pin.buildGeomFromUrdfString(full, urdf, pin.GeometryType.COLLISION, package_dirs=pkg)
    ids, reference = _finger_locks(full)
    model, (collision, visual) = pin.buildReducedModel(full, [collision, visual], ids, reference)
    return RobotWrapper(model, collision, visual)


class FrankaDynamics(DynamicsSimulator):
    """Triple-integrator 7-joint FR3 with jerk as the action; the goal is a joint configuration."""

    task_description = "reach target configuration"

    def __init__(self, config: Mapping[str, Any]):
        super().__init__(config)
        self.model = build_model()
        self.data = self.model.createData()
        self._robot: RobotWrapper | None = None
        self.tcp_id = self.model.getFrameId(TCP_FRAME)
        self.n = self.model.nv
        if self.n != FRANKA_NUM_JOINTS:
            raise ValueError(f"Expected {FRANKA_NUM_JOINTS} joints, the reduced franka_description has {self.n}.")

        self.nx = 3 * self.n
        self.nu = self.n
        self.obs_dim = self.n + self.nx
        per_joint = lambda key, default: np.broadcast_to(np.asarray(config.get(key, default), dtype=float), (self.n,)).copy()
        self.max_acceleration = per_joint("max_acceleration", FRANKA_DEFAULT_MAX_ACCELERATION)
        self.max_action = per_joint("max_jerk", FRANKA_DEFAULT_MAX_JERK)
        self.state_lower_bounds = np.concatenate([self.model.lowerPositionLimit, -self.model.velocityLimit, -self.max_acceleration])
        self.state_upper_bounds = np.concatenate([self.model.upperPositionLimit, self.model.velocityLimit, self.max_acceleration])

        self.goal = np.asarray(config["goal"], dtype=float) if "goal" in config else Q_READY.copy()
        self.randomize_goal = config.get("randomize_goal", "goal" not in config)
        self.min_start_goal_distance = float(config.get("min_start_goal_distance", 0.25))
        self.pos_tol = float(config.get("pos_tol", 2e-3))
        self.vel_tol = float(config.get("vel_tol", 2e-2))

    @property
    def rest_state_indices(self) -> tuple[int, ...]:
        return tuple(range(self.n, self.nx))  # velocity and acceleration

    @property
    def robot(self) -> RobotWrapper:
        """Model with collision and visual geometry, built on first use (for Meshcat)."""
        if self._robot is None:
            self._robot = build_robot()
        return self._robot

    def casadi_dynamics(self, x: Any, u: Any) -> Any:
        """Exact step under constant jerk, matching ``predict_next_state`` away from the limits."""
        return ca.vertcat(*constant_jerk_step(x[: self.n], x[self.n : 2 * self.n], x[2 * self.n :], u, self.dt))

    # --- numeric simulation ------------------------------------------------

    def predict_next_state(self, state: np.ndarray, action: np.ndarray, validate: bool = True) -> np.ndarray:
        state_array = self.validate_state(state) if validate else np.asarray(state, dtype=float)
        action_array = self.validate_action(action) if validate else np.asarray(action, dtype=float)
        return saturated_step(
            state_array, action_array, self.dt, self.state_lower_bounds, self.state_upper_bounds, self.max_action
        )

    def step(self, state: np.ndarray, action: np.ndarray, validate: bool = True) -> np.ndarray:
        return self.predict_next_state(state, action, validate=validate)

    def goal_errors(self, state: np.ndarray) -> tuple[float, float]:
        """Joint-position error [rad] of ``state`` with respect to the goal, and orientation error (none: 0)."""
        return float(np.linalg.norm(np.asarray(state, dtype=float)[: self.n] - self.goal)), 0.0

    def is_done(self, state: np.ndarray, validate: bool = True) -> bool:
        """At the goal (within ``pos_tol``) and at rest (every joint velocity within ``vel_tol``)."""
        state_array = self.validate_state(state) if validate else np.asarray(state, dtype=float)
        return bool(
            self.goal_errors(state_array)[0] < self.pos_tol and np.abs(state_array[self.n : 2 * self.n]).max() < self.vel_tol
        )

    # --- kinematics, goal and dataset -----------------------------------------

    def tcp_placement(self, q: np.ndarray) -> pin.SE3:
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.tcp_id].copy()

    def tcp_position(self, q: np.ndarray) -> np.ndarray:
        """The 3-D position of the TCP in configuration ``q`` (what the viewers draw)."""
        return self.tcp_placement(q).translation

    def get_dataset_features(self, prediction_horizon: int) -> dict[str, Any]:
        """LeRobot features: state [q, v, a], the goal joint configuration as environment state, the expert's planned
        jerks as action (its next ``prediction_horizon`` jerks, flattened row by row)."""
        joints = list(self.model.names[1:])
        return {
            "observation.environment_state": {
                "dtype": "float32",
                "shape": (self.n,),
                "names": [f"goal.{j}" for j in joints],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (self.nx,),
                "names": [f"{j}.{kind}" for kind in ("pos", "vel", "acc") for j in joints],
            },
            "action": {
                "dtype": "float32",
                "shape": (prediction_horizon * self.nu,),
                "names": [f"{j}.jerk[t+{k}]" for k in range(prediction_horizon) for j in joints],
            },
        }

    def _sample_configuration(self, rng: np.random.Generator) -> np.ndarray:
        """Random arm configuration inside the joint limits (by ``JOINT_SAMPLING_MARGIN``)."""
        lower, upper = self.model.lowerPositionLimit, self.model.upperPositionLimit
        margin = JOINT_SAMPLING_MARGIN * (upper - lower)
        return rng.uniform(lower + margin, upper - margin)

    def random_initial_state(self, rng: np.random.Generator) -> np.ndarray:
        """Rest state from a random configuration at least ``min_start_goal_distance`` from the goal."""
        for _ in range(SAMPLING_ATTEMPTS):
            q = self._sample_configuration(rng)
            if np.linalg.norm(q - self.goal) > self.min_start_goal_distance:
                return np.concatenate([q, np.zeros(2 * self.n)])
        raise RuntimeError(
            f"Unable to sample a start configuration {self.min_start_goal_distance} away from the goal "
            f"in {SAMPLING_ATTEMPTS} attempts."
        )

    def randomize_goal_for_reset(self, rng: np.random.Generator) -> None:
        if self.randomize_goal:
            self.set_goal(self._sample_configuration(rng))
