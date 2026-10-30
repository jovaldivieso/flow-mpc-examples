from __future__ import annotations

import os
import sys
import unittest
import warnings

import numpy as np
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from planning.config import validate_system_config
from systems.franka_system import Q_READY, FrankaSystem
from planning.casadi_planner import CasadiPlanner
from planning.casadi_graph import GraphSolution
from planning.casadi_projector import CasadiTrajectoryProjector
from planning.planner import PlannerSolveError
from planning.planner_config import PlannerConfig
from test.planner_fixtures import pose_goal

CONFIG_PATH = os.path.join(PROJECT_ROOT, "scripts", "config", "franka_casadi_config.yaml")

# Behaviour tests pin the settings they depend on, so tuning scripts/config/franka_casadi_config.yaml cannot break them
PINNED_EXPERT_SETTINGS = {
    "Q_diag": [20.0] * 7 + [2.0] * 7 + [0.2] * 7,
    "R_diag": [4.4e-3] * 7,
    "terminal_cost_multiplier": 10.0,
    "pos_tol": 0.002,
    "vel_tol": 0.02,
}
HORIZON = 10


def _validated_config(**overrides) -> dict:
    with open(CONFIG_PATH) as f:
        raw_config = yaml.safe_load(f)
    raw_config.update(PINNED_EXPERT_SETTINGS)
    raw_config.update(overrides)
    return validate_system_config(raw_config=raw_config)


def _rest_state(simulator: FrankaSystem) -> np.ndarray:
    """Ready pose with the wrist turned by 0.3 rad, at rest."""
    q = Q_READY.copy()
    q[6] += 0.3
    return np.concatenate([q, np.zeros(2 * simulator.n)])


def _zero_chunk(horizon: int = HORIZON) -> np.ndarray:
    """A chunk of zero jerks (horizon, nu): from rest, it stays at rest."""
    return np.zeros((horizon, 7))


def _moving_chunk(horizon: int = HORIZON) -> np.ndarray:
    """A proposal that starts a motion of joint 1 and does not come to rest within the horizon."""
    chunk = _zero_chunk(horizon)
    chunk[:3, 0] = 8.0
    return chunk


class ProjectorFailClosedFallbackTests(unittest.TestCase):
    """Returning the unprojected proposal on any solve failure bypasses every
    constraint the projector exists to enforce. The projector falls back to
    the last known-safe trajectory (SafeFlowMPC Theorem 2) whenever one
    exists. On a true cold-start failure there is no such trajectory --
    exactly the situation the paper's Assumption 2 assumes away -- so this
    raises PlannerSolveError instead of fabricating a fallback with no
    safety basis.
    """

    def setUp(self) -> None:
        config = _validated_config(terminal_velocity_weight=500.0)
        # An amplified terminal_velocity_weight makes the projector settle at
        # rest: the terminal-rest condition is a soft cost, not a hard
        # constraint, and these tests need a genuinely resting cached
        # trajectory to fall back on.
        self.simulator = FrankaSystem(config)
        self.projector = CasadiTrajectoryProjector(self.simulator, PlannerConfig.from_mapping(config), horizon=HORIZON)
        self.state0 = _rest_state(self.simulator)
        self.q_ref = _zero_chunk()

    def _perturbed_state(self, offset: float) -> np.ndarray:
        # Perturb joint positions only: leaves the certified-zero terminal velocity untouched
        state = self.state0.copy()
        state[: self.simulator.n] += offset
        return state

    def test_falls_back_to_last_safe_trajectory_when_one_exists(self) -> None:
        self.projector.project(self.state0, self.q_ref)
        self.assertIsNotNone(self.projector._prev_U_sol)
        # Matches project()'s own fallback padding: the shifted plan's
        # extra final slot is zero action (holds the certified-at-rest
        # terminal state), not a duplicate of the last applied action.
        shifted_actions = np.hstack([self.projector._prev_U_sol[:, 1:], np.zeros((self.simulator.nu, 1))])
        perturbed_state = self._perturbed_state(0.005)
        expected_plan = self.projector._revalidate_fallback(perturbed_state, shifted_actions)
        self.assertIsNotNone(expected_plan)

        self.projector._graph.opti.solve = lambda: (_ for _ in ()).throw(RuntimeError("forced failure"))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fallback_chunk = self.projector.project(perturbed_state, self.q_ref)

        np.testing.assert_allclose(fallback_chunk, shifted_actions.T)
        self.assertTrue(any("last known-safe" in str(w.message) for w in caught))

    def test_raises_on_cold_start_failure_instead_of_returning_the_proposal(self) -> None:
        # No previously-solved safe trajectory exists (first call ever, or
        # right after reset()) -- SafeFlowMPC's safety guarantee (Theorem 2)
        # assumes one always exists (Assumption 2), so there is no
        # paper-faithful fallback action here. Must fail loudly rather than
        # silently returning the unverified, unprojected proposal.
        self.projector.reset()
        self.projector._graph.opti.solve = lambda: (_ for _ in ()).throw(RuntimeError("forced failure"))
        with self.assertRaises(PlannerSolveError):
            self.projector.project(self.state0, self.q_ref)

    def test_consecutive_fallbacks_stay_at_rest_instead_of_drifting_away(self) -> None:
        # Regression test for padding the fallback's extra control slot with
        # a duplicate of the *last applied* action instead of zero (the
        # action that actually sustains the certified at-rest terminal
        # state). A run of consecutive failures across genuinely different
        # ticks (a new x0 each time, so each hits the shift-and-pad branch,
        # not the same-tick reuse-as-is branch) must keep the fallback's own
        # terminal velocity bounded, not drifting further away with each
        # additional fallback.
        self.projector.project(self.state0, self.q_ref)
        self.projector._graph.opti.solve = lambda: (_ for _ in ()).throw(RuntimeError("forced failure"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(1, 6):
                self.projector.project(self._perturbed_state(0.001 * i), self.q_ref)
        terminal_velocity = self.projector._prev_X_sol[self.projector.rest_idx, self.projector.N]
        # Not 0.0 to machine precision: the terminal-rest condition is a
        # soft cost, so the cached solve only ever settles *close to*
        # rest, never exactly -- fallback_terminal_velocity_tol is precisely
        # the bound _revalidate_fallback itself already enforces on this
        # quantity every time it accepts a fallback.
        np.testing.assert_allclose(terminal_velocity, 0.0, atol=self.projector.fallback_terminal_velocity_tol)

    def test_exceeding_max_consecutive_fallbacks_raises_instead_of_coasting_indefinitely(self) -> None:
        # A fallback accepted within fallback_terminal_velocity_tol is
        # coasting at a small residual velocity, not genuinely at rest; each
        # individual reuse is only checked through its own horizon. An
        # unbroken run of them would otherwise let that drift keep
        # accumulating for as long as real solves keep failing.
        # max_consecutive_fallbacks bounds it: one more consecutive fallback
        # than the default (5) must raise rather than be accepted.
        self.assertEqual(self.projector.max_consecutive_fallbacks, 5)
        self.projector.project(self.state0, self.q_ref)
        self.projector._graph.opti.solve = lambda: (_ for _ in ()).throw(RuntimeError("forced failure"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(1, 6):
                self.projector.project(self._perturbed_state(0.001 * i), self.q_ref)
            with self.assertRaises(PlannerSolveError):
                self.projector.project(self._perturbed_state(0.006), self.q_ref)

    def test_repeated_same_tick_calls_do_not_each_consume_the_fallback_budget(self) -> None:
        # select_action() calls project() once per flow/Euler denoising step
        # -- several times per real control tick, all at the *same* x0 (only
        # the proposal changes as the flow network refines it). None of
        # those represent additional elapsed time or drift, so repeating
        # far more of them than max_consecutive_fallbacks allows must not
        # raise, as long as it's genuinely the same tick throughout.
        self.assertEqual(self.projector.max_consecutive_fallbacks, 5)
        self.projector.project(self.state0, self.q_ref)
        self.projector._graph.opti.solve = lambda: (_ for _ in ()).throw(RuntimeError("forced failure"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(20):  # far more than max_consecutive_fallbacks (5)
                self.projector.project(self.state0, self.q_ref)
        self.assertEqual(self.projector._consecutive_fallback_count, 0)


class FallbackRevalidationTests(unittest.TestCase):
    """_revalidate_fallback re-checks a reused plan against everything the
    live solve enforces, not only the state bounds and the terminal rest."""

    def setUp(self) -> None:
        config = _validated_config(terminal_velocity_weight=500.0)
        self.simulator = FrankaSystem(config)
        self.projector = CasadiTrajectoryProjector(self.simulator, PlannerConfig.from_mapping(config), horizon=HORIZON)
        self.state0 = _rest_state(self.simulator)
        self.u_rest = np.zeros((self.simulator.nu, HORIZON))

    def test_accepts_a_resting_plan(self) -> None:
        trajectory = self.projector._revalidate_fallback(self.state0, self.u_rest)
        self.assertIsNotNone(trajectory)
        self.assertEqual(trajectory.shape, (self.simulator.nx, HORIZON + 1))

    def test_rejects_a_plan_that_does_not_end_at_rest(self) -> None:
        moving = self.u_rest.copy()
        moving[0, :] = 5.0
        self.assertIsNone(self.projector._revalidate_fallback(self.state0, moving))


class ProjectionFeasibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = _validated_config()
        cls.simulator = FrankaSystem(cls.config)
        cls.projector = CasadiTrajectoryProjector(cls.simulator, PlannerConfig.from_mapping(cls.config), horizon=HORIZON)

    def test_projection_of_a_resting_reference_at_rest_stays_put(self) -> None:
        self.projector.reset()
        state = _rest_state(self.simulator)
        projected = self.projector.project(state, _zero_chunk())
        np.testing.assert_allclose(projected, _zero_chunk(), atol=1e-3)

    def test_projected_chunks_are_feasible_for_the_plant_even_for_infeasible_proposals(self) -> None:
        simulator = self.simulator
        tolerance = 1e-5  # the constraints hold up to the solver's tolerance
        for seed, scale in enumerate((0.05, 0.5, 3.0)):
            rng = np.random.default_rng(seed)
            simulator.randomize_goal_for_reset(rng)
            x0 = simulator.random_initial_state(rng)
            self.projector.reset()
            u_ref = rng.normal(0.0, scale, (HORIZON, simulator.nu)) * simulator.max_action

            projected = self.projector.project(x0, u_ref)

            self.assertEqual(projected.shape, u_ref.shape)
            self.assertTrue(np.all(np.abs(projected) <= simulator.max_action + tolerance), scale)
            state = x0.copy()
            for action in projected:  # the plant follows the projected actions
                state = simulator.predict_next_state(state, action)
                self.assertTrue(np.all(state >= simulator.state_lower_bounds - tolerance), scale)
                self.assertTrue(np.all(state <= simulator.state_upper_bounds + tolerance), scale)
            np.testing.assert_allclose(state[: simulator.n], self.projector._prev_X_sol[: simulator.n, -1], atol=1e-4)

    def test_unit_gaussian_noise_is_projected_without_failing(self) -> None:
        # The flow starts from noise like this (scaled by the action limits): the cold start must not begin inside
        # the constraints, where the solver ends in a locally infeasible point
        simulator = self.simulator
        state = _rest_state(simulator)
        for seed in range(6):
            self.projector.reset()
            noise = np.random.default_rng(seed).normal(size=(HORIZON, simulator.nu)) * simulator.max_action

            projected = self.projector.project(state, noise)

            self.assertEqual(projected.shape, noise.shape)

    def test_the_projection_stays_inside_the_joint_limits_the_proposal_ignores(self) -> None:
        # A proposal that drives joint 5 into its upper limit at full jerk
        joint = 4
        simulator = FrankaSystem(self.config)
        projector = CasadiTrajectoryProjector(simulator, PlannerConfig.from_mapping(self.config), horizon=HORIZON)
        state = _rest_state(simulator)
        upper = simulator.state_upper_bounds[joint]
        state[joint] = upper - 0.3
        proposal = _zero_chunk()
        proposal[:, joint] = simulator.max_action[joint]

        projector.project(state, proposal)

        self.assertLessEqual(projector._prev_X_sol[joint].max(), upper + 1e-6)

    def test_a_feasible_plan_of_the_expert_is_tracked_closely(self) -> None:
        # The projector only corrects what is unsafe: an action chunk the expert planned stays almost untouched
        simulator = self.simulator
        simulator.set_goal(pose_goal(simulator, Q_READY))
        state = _rest_state(simulator)
        reference = CasadiPlanner(simulator, PlannerConfig.from_mapping(self.config)).plan(simulator.observe(state)).actions[:HORIZON]
        self.projector.reset()

        projected = self.projector.project(state, reference)

        np.testing.assert_allclose(projected, reference, atol=0.15 * simulator.max_action.max())

    def test_the_next_chunk_guess_is_the_actions_shifted_by_one_step(self) -> None:
        projector = CasadiTrajectoryProjector(self.simulator, PlannerConfig.from_mapping(self.config), horizon=HORIZON)
        self.assertIsNone(projector.next_chunk_guess())
        state = _rest_state(self.simulator)
        projected = projector.project(state, _moving_chunk())

        guess = projector.next_chunk_guess()

        self.assertEqual(guess.shape, (HORIZON, self.simulator.nu))
        np.testing.assert_allclose(guess[:-1], projected[1:])
        np.testing.assert_array_equal(guess[-1], 0.0)  # zero jerk keeps the rest state the plan ends in
        projector.reset()
        self.assertIsNone(projector.next_chunk_guess())

    def test_a_chunk_of_the_wrong_shape_is_rejected(self) -> None:
        state = _rest_state(self.simulator)
        with self.assertRaises(ValueError):
            self.projector.project(state, _zero_chunk().T)

    def test_short_horizon_constructs_and_solves(self) -> None:
        projector = CasadiTrajectoryProjector(self.simulator, PlannerConfig.from_mapping(self.config), horizon=2)
        state = _rest_state(self.simulator)
        projected = projector.project(state, _zero_chunk(2))
        self.assertEqual(projected.shape, (2, self.simulator.nu))

    def test_the_projector_ignores_the_goal_of_the_simulator_it_was_built_on(self) -> None:
        # The projector takes no goal, and no projector cost or constraint reads it, so whatever goal the simulator
        # holds (here none that is a pose at all) must not change the projection.
        simulator = FrankaSystem(self.config)
        other = FrankaSystem(self.config)
        other.set_goal(np.zeros_like(other.goal))
        projector = CasadiTrajectoryProjector(other, PlannerConfig.from_mapping(self.config), horizon=HORIZON)
        reference = CasadiTrajectoryProjector(self.simulator, PlannerConfig.from_mapping(self.config), horizon=HORIZON)
        state = _rest_state(simulator)
        u_ref = np.random.default_rng(5).normal(0.0, 0.3, (HORIZON, simulator.nu)) * simulator.max_action
        np.testing.assert_allclose(
            projector.project(state, u_ref),
            reference.project(state, u_ref),
            atol=1e-6,
        )

    def test_a_projector_needs_a_horizon(self) -> None:
        # Its scalar settings are validated when the config is loaded (test_planner_config)
        with self.assertRaises(ValueError):
            CasadiTrajectoryProjector(self.simulator, PlannerConfig.from_mapping(self.config), horizon=0)

    def test_projector_pushes_an_obstacle_colliding_proposal_outside_the_sphere(self) -> None:
        state = _rest_state(self.simulator)
        proposal = _moving_chunk()
        rollout_q = self.simulator.rollout_positions(state, proposal)
        target_tcp = self.simulator.tcp_position(rollout_q[-1]).tolist()

        obs_config = _validated_config(obstacles=[{"position": target_tcp, "radius": 0.05}])
        obs_sim = FrankaSystem(obs_config)
        self.assertTrue(obs_sim.collision.is_collision_free(state))
        self.assertFalse(obs_sim.collision.is_collision_free(rollout_q[-1]))

        projector = CasadiTrajectoryProjector(obs_sim, PlannerConfig.from_mapping(obs_config), horizon=HORIZON)
        projected = projector.project(state, proposal)
        projected_q = obs_sim.rollout_positions(state, projected)
        for q in projected_q[1:]:
            self.assertTrue(obs_sim.collision.is_collision_free(q), "projected trajectory collided with obstacle")


class PerStepProjectionTests(unittest.TestCase):
    """Every Euler step is a converged solve, linearized along its own proposal, whose solution has to clear the obstacles."""

    def setUp(self) -> None:
        self.state = _rest_state(FrankaSystem(_validated_config()))
        rollout_q = FrankaSystem(_validated_config()).rollout_positions(self.state, _moving_chunk())
        self.config = _validated_config(
            terminal_velocity_weight=500.0,
            obstacles=[{"position": FrankaSystem(_validated_config()).tcp_position(rollout_q[-1]).tolist(), "radius": 0.05}],
        )
        self.simulator = FrankaSystem(self.config)
        self.projector = CasadiTrajectoryProjector(self.simulator, PlannerConfig.from_mapping(self.config), horizon=HORIZON)

    def test_the_obstacles_are_linearized_at_every_step(self) -> None:
        real_halfspaces = self.simulator.collision.halfspaces
        calls: list[int] = []
        self.simulator.collision.halfspaces = lambda q: calls.append(len(q)) or real_halfspaces(q)

        for _ in range(3):  # the Euler steps of one tick
            self.projector.project(self.state, _moving_chunk())

        self.assertGreaterEqual(len(calls), 3)

    def test_every_projection_clears_the_obstacles(self) -> None:
        for _ in range(3):
            projected = self.projector.project(self.state, _moving_chunk())
            for q in self.simulator.rollout_positions(self.state, projected)[1:]:
                self.assertTrue(self.simulator.collision.is_collision_free(q))

    def test_a_solution_that_still_penetrates_an_obstacle_is_a_failed_projection(self) -> None:
        simulator = self.simulator
        simulator.collision.obstacles[:] = [(simulator.tcp_position(self.state[: simulator.n]), 0.05)]  # on the TCP of the start
        at_start = np.tile(self.state[:, None], (1, HORIZON + 1))
        self.projector._graph.solve = lambda: GraphSolution(at_start, np.zeros((simulator.nu, HORIZON)))

        with self.assertRaises(PlannerSolveError):
            self.projector.project(self.state, _moving_chunk())


if __name__ == "__main__":
    unittest.main()
