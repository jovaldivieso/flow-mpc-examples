# flow-mpc-examples

Imitation learning of a CasADi MPC expert on the Franka FR3 arm with flow-matching policies (`flow`, `safeflow`), trained with DAgger.

The system is the 7-joint FR3 arm (the gripper is fixed geometry), modelled as a kinematic triple integrator per joint (jerk as the action, as in SafeFlowMPC) with the TCP pose as the goal. The policy is conditioned directly on the goal pose as seen from the end effector and the robot state, with no observation encoder.

## Setup (macOS)

```bash
git clone --recurse-submodules <repo-url>   # or: git submodule update --init
python3.13 -m venv venv                     # OMPL's wheels do not support Python 3.14 yet
venv/bin/pip install -r requirements.txt
venv/bin/python -c "import pinocchio, ompl; print(pinocchio.__version__)"
```

Pinocchio (the `pin` wheel, with `eigenpy` and `coal`) and OMPL both come from PyPI, so nothing is needed from Homebrew. All commands below run from the repo root.

If DAgger prints `Class AVFFrameReceiver is implemented in both ...` warnings, `torchcodec` is installed in the venv: it and PyAV each load their own copy of ffmpeg's libavdevice. This repo's datasets have no videos, so `requirements.txt` leaves `torchcodec` out (LeRobot then decodes with PyAV); remove it with `venv/bin/pip uninstall torchcodec`.

## Systems

Both systems inherit from the abstract `TripleIntegratorBase` (`systems/triple_integrator.py`: the plant, limits, goal and tolerances, start sampling) and, through it, from `KinematicCollisionSystem` (`systems/kinematic_collision_system.py`: what the planners need beyond a simulator: the obstacles, the bounding spheres, the path-planning box and the goal configuration, with the sphere checks and linearized half-spaces built on them). Each axis/joint is a kinematic 3rd-order triple integrator (`q' = q + v dt + a dt^2/2 + j dt^3/6`, `v' = v + a dt + j dt^2/2`, `a' = a + j dt`), assuming a low-level trajectory-tracking controller underneath. For the Franka FR3 (`systems/franka_system.py`) the URDF is expanded at runtime from [third_party/franka_description](third_party/franka_description) via `xacro` and Pinocchio, with the two finger joints locked open (`pin.buildReducedModel`).

| System (`system`) | Class | State `x` (`nx`) | Goal `g` (`goal_dim`) | Simulator Obs `o = [g, x]` (`obs_dim`) | Policy Input / Dataset Features (`state_dim`) | Policy Output / Action (`action_dim`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`franka`** | `FrankaSystem` (7-DoF FR3) | `[q, v, a]` (**21**) | Target TCP pose `[p, R]` in base frame (**12**) | `[g, x]` (**33**) | **40-D**:<br>- `observation.environment_state` (**19**): `[T_tcp^-1 T_goal (12), q_goal - q (7)]`<br>- `observation.state` (**21**): `[q (7), v (7), a (7)]` | Action chunk `(H, 7)` of joint jerks `u = j` (**7**), flattened to `(H * 7,)` in dataset `action` |
| **`triple_integrator`** | `PointMassSystem` ($d$-D point mass, default $d=2$) | `[p, v, a]` (**3d**, default **6**) | Target position `g` (**d**, default **2**) | `[g, x]` (**4d**, default **8**) | **3d-D** (default **6**):<br>- `observation.environment_state` (**d**): `g - p`<br>- `observation.state` (**2d**): `[v (d), a (d)]` | Action chunk `(H, d)` of jerks `u = j` (**d**, default **2**), flattened to `(H * d,)` in dataset `action` |

## Core file structure

```text
flow-mpc-examples/
├── systems/
│   ├── system.py                # SystemSimulator: the plant, goal, observations and the policy's view of them (no dataset or rollout state)
│   ├── kinematic_collision_system.py  # KinematicCollisionSystem: the interface the planners use (obstacles, bounding spheres, path box, goal configuration)
│   ├── rollout_termination.py   # RolloutTerminationTracker: ends a rollout once the goal has held for done_hold_steps
│   ├── system_factory.py        # SystemFactory: registry of the systems the config chooses from
│   ├── franka_system.py         # 7-joint FR3, gripper as fixed geometry: kinematic plant, dataset schema, sampling
│   ├── franka_model.py          # The FR3 as a Pinocchio model: xacro/URDF expansion, locked fingers, SafeFlowMPC's joint limits
│   ├── franka_kinematics.py     # CasADi kinematics generated from that model (built once): TCP forward kinematics, link bounding spheres, SafeFlowMPC's IK problem (IPOPT)
│   ├── collision_spheres.py     # Bounding spheres fitted to the collision meshes (they cover the real geometry, hand and fingers included)
│   ├── triple_integrator.py     # TripleIntegratorBase: the shared triple-integrator core (plant, limits, goal, obstacles, start sampling)
│   ├── point_mass_system.py     # PointMassSystem: a point mass, a toy system to calibrate the pipeline
│   ├── initial_state_utils.py   # Initial/goal-state parsing and normalization
│   └── seed_utils.py            # Seed defaults and deterministic rollout seeding
├── planning/
│   ├── config.py                # Typed validation and normalized YAML loading of the expert config
│   ├── planner.py               # The one abstract Planner (observation [, initial plan] -> Plan: joint positions and jerks)
│   ├── casadi_graph.py          # CasadiKinematicGraph: the stage-wise OCP skeleton (FATROP) the MPC and the projector share (dynamics, limits, obstacle half-spaces)
│   ├── initial_guess.py         # InitialGuessGenerator: the strategy interface of the optional initial guess of the MPC
│   ├── casadi_planner.py        # CasADi MPC (FATROP) in joint space: goal-seeking cost, optional initial guess generator (strategy)
│   ├── ompl_planner.py          # OMPL RRT-Connect planner: collision-free joint-space path to the goal configuration; initialize_ompl (OMPL's seed and log level, called once at an entry point)
│   ├── ompl_initial_guess.py    # OmplInitialGuess: the OMPL path as the goal configuration and first guess of the MPC (the strategy of `ompl_casadi`)
│   ├── planner_factory.py       # PlannerFactory: registry of the experts the config chooses from
│   └── casadi_projector.py      # CasADi safety projection for SafeFlowMPC
├── learning/
│   ├── config_loaders.py        # YAML and policy configuration helpers
│   ├── config/                  # Policy + DAgger schedule configs (flow, safeflow)
│   ├── models/
│   │   ├── temporal_unet.py     # SafeFlowMPC's temporal U-Net (1D convolutions over the horizon), without its attention layers
│   │   ├── flow_velocity_net.py # Flow-matching loss and Euler integration around the U-Net
│   │   ├── flow_policy.py       # FlowVelocityNet conditioned directly on [goal, state]
│   │   ├── exponential_moving_average.py  # Exponential moving average of the weights
│   │   ├── safe_flow_policy.py  # Flow policy + CasADi safety projection at inference (SafeFlowMPC)
│   │   ├── policy.py            # Shared policy interface
│   │   └── policy_factory.py    # PolicyFactory: builds the flow or safeflow policy from its type
│   ├── data_utils.py            # LeRobot dataset adapter: features and frames of a system's dataset, policy batch formatting (flat dataset action -> (prediction_horizon, nu) chunk)
│   ├── dagger/                  # config, trainer, metrics, utils; rollouts: RolloutCollector, RolloutEvaluator and the RolloutWorker of a process pool
│   └── train_dagger.py          # CLI: parses args, builds DaggerConfig, runs DaggerTrainer
├── scripts/
│   ├── config/                  # Expert configs (casadi, ompl_casadi, the cluttered ompl_casadi evaluation environment; point mass)
│   ├── evaluate_policy.py       # Expert vs policy rollouts and metrics
│   ├── meshcat_expert.py        # Expert check before DAgger: random rollouts, statistics, Meshcat playback
│   ├── meshcat_utils.py         # Shared Meshcat drawing helpers
│   ├── meshcat_dataset.py       # Dataset episodes (and their labelled plans) played back in Meshcat
│   └── meshcat_policy.py        # A checkpoint's closed-loop rollout in Meshcat, next to the expert's plan
├── test/                        # Unit tests only
└── third_party/franka_description  # Submodule
```

Policy types (`model.policy_type` in a `learning/config/` policy config): `flow` (conditional flow-matching policy of the action chunk) and `safeflow` (`flow` wrapped in the CasADi trajectory-safety projection at inference, SafeFlowMPC, Oelerich et al., 2026). They share the same weights, so a `flow`-trained checkpoint can be evaluated as either.

What the two do, following SafeFlowMPC:

- **Flow network** (both): SafeFlowMPC's temporal U-Net (`TemporalUnet`, the same outputs as `safe_flow_mpc/Models/models.py` for the same weights, without its unused attention layers): a 1D convolution over the chunk's horizon with the action dimensions as channels, `dim` 32 and `dim_mults` (1, 2, 4, 8) (about 4 M parameters), every residual block conditioned on the sum of an embedding of the flow time and of the observation `[goal, state]`. `prediction_horizon` has to be divisible by `2 ** (len(dim_mults) - 1)`.
- **Flow training** (both): rectified flow matching `x_t = t x_1 + (1 - t) x_0` against Gaussian noise, with the flow time drawn as `t = 1 - Beta(1.5, 1)` (more samples near the noisy end), the Euler integration of the velocity network at inference, and an **exponential moving average (EMA)** of the weights (`ema <- 0.995 ema + 0.005 weights` after every optimizer step), which is what the DAgger rollouts, the evaluation and the saved `ema_model_state_dict` use (`evaluate_policy.py` prefers it over `model_state_dict`).
- **Safe inference** (`safeflow`): after every Euler step the proposed action chunk is projected by `CasadiTrajectoryProjector`: it tracks the proposal (squared difference of the actions, each divided by its limit), keeps the plan within the joint, velocity, acceleration and jerk limits and clear of the obstacles (linearized sphere half-spaces with a penalized slack, see "Obstacles") and ends at rest (soft terminal velocity and acceleration cost, `terminal_velocity_weight`); if a solve fails it falls back to the last plan that still validates (Theorem 2), otherwise it raises `PlannerSolveError` (collection then executes the expert's action of that step, evaluation counts a solve failure). The integration starts from the previous tick's safe actions shifted by one step, with zero jerk appended (`safeflow.init: previous_safe`; on the first tick from zero actions) or from Gaussian noise (`gaussian`). SafeFlowMPC's goal guidance acted on a position chunk and is not part of this version.

### Toy system: `triple_integrator`

Set `system: triple_integrator` in the expert config (`scripts/config/triple_integrator_casadi_config.yaml` with `learning/config/triple_integrator_casadi_flow_config.yaml` or `..._safeflow_config.yaml`) to calibrate the pipeline on a point mass before running on the arm:

```bash
venv/bin/python learning/train_dagger.py --experiment-name toy_flow \
  --expert-config scripts/config/triple_integrator_casadi_config.yaml \
  --policy-config learning/config/triple_integrator_casadi_flow_config.yaml
```

`scripts.meshcat_policy` also plays a point-mass checkpoint (pass `--config scripts/config/triple_integrator_casadi_config.yaml`).

### Expert config (`scripts/config/franka_casadi_config.yaml`)

The expert config holds the system and planner parameters, validated through [planning/config.py](planning/config.py) (unknown keys, wrong types and impossible values are rejected before anything runs; the config does not know the system's dimensions, so `CasadiPlanner` checks the lengths of `Q_diag` and `R_diag` and completes them when omitted):

- planner: `dt`, `horizon` (the expert MPC's own horizon), `Q_diag` (positive, 21 for the arm, one per state: the position, velocity and acceleration error to the goal state; default 10 each), `R_diag` (positive, 7, one per joint: the squared jerk; default `R_weight` each), `terminal_cost_multiplier`
- projector: `terminal_velocity_weight`, `fallback_terminal_velocity_tol`, `max_consecutive_fallbacks` (read by `CasadiTrajectoryProjector`, defaults in `planning/config.py`)
- `expert`: the expert planner, a name of the registry in [planning/planner_factory.py](planning/planner_factory.py) (`PlannerFactory`): `casadi` (the MPC) or `ompl_casadi` (the same MPC with the end of an OMPL path as its goal configuration, starting from the path); a new planner is one more entry
- OMPL path (`OmplConfig`, for `ompl_casadi`): the `ompl_*` settings (`ompl_speed`, `ompl_replan_distance`, `ompl_timeout`, `ompl_seed`), see below
- arm: `max_acceleration` and `max_jerk` (each a scalar or 7 per-joint limits; the joint position and velocity limits come from the description)
- tolerances: `pos_tol` [m, TCP to goal position], `vel_tol` [rad/s] (what counts as "reached the goal and at rest"), `done_hold_steps`
- `obstacles`: static spherical obstacles in the world frame, a list of `position` and `radius` [m] (see "Obstacles")
- sampling: `min_start_goal_distance` [m, TCP to goal position], `goal` (a pose: position and a row-major rotation matrix, 12 numbers) / `randomize_goal` (omit `goal` to randomize it every episode), `initial_state_seed`, `action_noise_seed`

**The experts.** `casadi` is the MPC alone: it heads for the goal configuration `q_goal` (the inverse kinematics solution of the goal pose seeded from `Q_READY`), solved as a convex QP with FATROP at every step from the previous solution shifted by one step. With `expert: ompl_casadi` the expert is the same `CasadiPlanner`, composed with an `OmplInitialGuess` (an `InitialGuessGenerator` strategy): the goal configuration is the end of an OMPL path and the first solve starts from that path. For a goal, `OmplPlanner` plans an RRT-Connect path to `q_goal` (inside the joint limits, collision-free, simplified, deterministic given `ompl_seed`, which the entry points pass to `initialize_ompl` once) and returns it as a `Plan`, timed at the constant joint speed `ompl_speed`. Later solves start from the previous solution again, unless the arm is further than `ompl_replan_distance` rad from it (the learner drove it away): then a new path is planned from where it is. If the MPC solve that started from a path fails, the path is the plan (`trajectory_fallbacks`); if no path is found (for example the arm starts in collision) the MPC runs as it does alone (`ompl_path_failures`). The path alone is no expert: a jerk-limited arm cannot follow it step by step.

### Obstacles

The obstacles are static spheres. The arm is covered by bounding spheres fitted to its collision meshes (`systems/collision_spheres.py`: the meshes are split recursively along their principal axis until a minimal enclosing ball has a radius of at most 8 cm, so the hand and the fingers are covered too). Everything that plans or filters keeps these spheres clear of the obstacles: the MPC and the projector as linearized half-spaces `n^T J_s(q_ref) q >= b` per sphere, obstacle and node with a slack per sphere and node (shared by its obstacles) penalized by 1e5 (the solution is linearized again once if it still penetrates), and OMPL as its validity check. The evaluation counts a collision as a touch of the real collision meshes with an obstacle, which the spheres cover; the DAgger collection does not check for collisions. The policy does not observe the obstacles: they act through the expert's labels and the projector, so one checkpoint can be evaluated with any `obstacles` list. FATROP's time grows linearly with the obstacles (an MPC step takes about 3 ms without obstacles, 12 with 1, 19 with 3 and 29 with 6 on an Apple-silicon laptop; ProxQP, which this replaced, took seconds from 4 obstacles on). `scripts/config/franka_ompl_casadi_cluttered_config.yaml` is `franka_ompl_casadi_config.yaml` with 5 obstacles (its 2 and 3 more, at other depths, left-right displacements and heights), to evaluate how a policy trained in the latter generalizes, as `flow` or as `safeflow` (`--policy-type safeflow`).

### Training and curriculum config (`learning/config/`)

`learning/train_dagger.py`'s `--policy-config` points at a YAML file with a `model:` block (`policy_type`, `flow.num_inference_steps`, `safeflow.init`, `prediction_horizon`, `unet.dim` / `unet.dim_mults`) and a `training:` block with the DAgger schedule (rounds, `trajectories_per_iteration`, `target_epochs_per_round`, `expert_beta_per_iteration`, `num_workers`, action noise, evaluation, `tolerance_overrides` / `eval_tolerance_overrides`, optional `workspace_bounds`, `initial_states` / `goal_states` / `training_curriculum`). Every `training:` key has a matching `--flag` that overrides it for a single run (`train_dagger.py --help`).

The training recipe is SafeFlowMPC's: AdamW (`learning_rate` 1e-4, weight decay 1e-7), `batch_size` 1024, a cosine decay of the learning rate in every round over twice the round's steps down to 1e-7 (half a period, restarted each round) and an EMA of 0.995.

`franka_casadi_flow_config.yaml` and `franka_casadi_safeflow_config.yaml` differ only in `policy_type` and the `safeflow:` block (initialization, ignored by `flow`), and define the per-round expert action mixing probability directly via `expert_beta_per_iteration` (either a single value broadcast across all rounds or one value per round).

## Pipeline

### Train

```bash
venv/bin/python learning/train_dagger.py \
  --experiment-name franka_flow \
  --expert-config scripts/config/franka_casadi_config.yaml \
  --policy-config learning/config/franka_casadi_flow_config.yaml
```

Fresh-start DAgger collects its own expert-labelled rollouts, so no offline dataset is needed (`--dagger-iterations 0` with `--repo-id`/`--dataset-root` trains offline on an existing dataset). The dataset is written to `data/lerobot_dataset_franka_dagger_<timestamp>` (LeRobot format: `observation.environment_state` = `[T_tcp^-1 T_goal (12), q_goal - q (7)]` (19), `observation.state` = `[q, v, a]` (21), `action` = the expert's planned jerks from that state, `prediction_horizon x 7` flattened). Checkpoints land in `outputs/train_dagger/<experiment-name>/` as `flow_dagger_checkpoint.pt` (latest) and one `flow_dagger_iter_NNN.pt` per round, next to `training_summary.yaml` (per-round success rates, the evaluation collision / timeout / solve-failure breakdown, the collection's attempted and discarded episodes, expert planner failures and policy-solve fallbacks, and the expert's own counters in `expert_statistics` / `total_expert_statistics`: `ompl_paths`, `ompl_replans`, `ompl_path_failures`, `trajectory_optimizations`, `trajectory_fallbacks`) and copies of the configs and run arguments.

A smoke test of the whole loop (collection, dataset, training, evaluation, checkpoints) that takes under a minute, with `--policy-config` set to either policy; it leaves a throwaway dataset in `data/` and checkpoints in `outputs/train_dagger/smoke/`:

```bash
venv/bin/python learning/train_dagger.py --experiment-name smoke \
  --expert-config scripts/config/franka_casadi_config.yaml \
  --policy-config learning/config/franka_casadi_flow_config.yaml \
  --dagger-iterations 2 --trajectories-per-iteration 2 --steps-per-trajectory 8 \
  --target-epochs-per-round 1 --eval-episodes 1 --eval-steps 6 --expert-beta-per-iteration 0.5
```

Each round executes a mix of expert and learner actions (`expert_beta_per_iteration`) and labels every visited state with the expert's plan. An episode whose expert planner fails to solve (`planner_failures`) is dropped as a whole and retried from a fresh start and goal, at most `3 x trajectories_per_iteration` attempts per round. Nothing else drops an episode: a learner that collides keeps going, and the expert labels the states it reaches. A policy solve that fails (`safeflow`) executes the expert's action of that step instead (`policy_solve_fallbacks`).

### Evaluate

```bash
venv/bin/python -m scripts.evaluate_policy \
  --model-dir outputs/train_dagger/franka_flow/flow_dagger_checkpoint.pt \
  --config scripts/config/franka_ompl_casadi_config.yaml \
  --num-steps 125 --seeds "[0, 1, 2, 3, 4]" \
  --tolerance-overrides '{"pos_tol": 0.02, "vel_tol": 0.1}' \
  [--output-path outputs/eval.npz]
```

`evaluate_policy` and `meshcat_policy` share their options (`--model-dir`, `--policy-type`, `--config`, `--seeds`, `--num-steps`, `--tolerance-overrides`, `--device`, with the same defaults), so a rollout looked at in Meshcat can be evaluated with the same command line. It rolls out the expert and the policy from the same seeded start and goal and reports the success rate, the collision / timeout / solve-failure rates, steps, the final error, and the solve times of the expert and the policy. `--policy-type` is the checkpoint's own by default; pass `flow` or `safeflow` to evaluate it as the other.

### Check the expert, visualize

Run the expert check before a DAgger run: it samples starts and goals like the collection does, prints per rollout whether the goal was reached, the final error, and how smooth the jerks are (`jerkiness` = mean change of the normalized jerk per step, `at jerk limit` = share of steps on the limit), then plays all rollouts in Meshcat with their TCP paths (green = reached, orange = not). `--no-viewer` prints the statistics only; `--tolerance-overrides`, `--action-noise-std` and `--seeds` mirror the training options.

```bash
venv/bin/python -m scripts.meshcat_expert --num-traj 10   # expert on 10 seeded random start/goal pairs
venv/bin/python -m scripts.meshcat_dataset --root data/lerobot_dataset_franka_dagger_<timestamp> \
  --first --episodes 10 --show-plan   # the first 10 episodes with the expert's labelled plan of every frame
venv/bin/python -m scripts.meshcat_policy --model-dir outputs/train_dagger/<name>/flow_dagger_checkpoint.pt \
  --seeds "[10000, 10001]" --show-expert   # a policy rollout next to the expert's plan from the same states
```

`meshcat_dataset` prints a table per episode first (frames, final error, share of frames at rest, share of "stay" labels, i.e. a labelled plan whose motion stays within 0.05 rad of the current configuration) and takes `--indices "[0, 5, 8]"` or random episodes (`--seed`); `--no-viewer` stops after the table. It reads the parquet files directly, so it also works on a dataset a DAgger run is still writing to (unfinished files are skipped). `meshcat_policy` runs the checkpoint closed loop from seeded starts, draws the motion of the policy's action chunk (cyan) and, with `--show-expert`, of the expert's plan from the same state (green), both integrated from the current state, and prints the outcome of each rollout; `--no-viewer` prints only the outcomes.

`meshcat_policy --show-collision-geometry` draws the bounding spheres on the robot (translucent blue). Meshcat opens at `http://127.0.0.1:7001/static/`; the small red ball marks the goal TCP position (and, for datasets, the blue dot the start), the translucent red spheres are the obstacles.

## Notes

- Every projection and every MPC step is a convex CasADi/FATROP solve with the linear triple-integrator dynamics, box constraints on joint positions, velocities, accelerations and jerks and the linearized obstacle half-spaces (~3 ms per MPC step on an Apple-silicon laptop at `dt` 0.1 and horizon 20 without obstacles, see above for obstacles), and DAgger rollout collection and evaluation parallelize across episodes via `num_workers`.

## Tests

```bash
venv/bin/python -m unittest discover -s test -t . -v
```

The tests cover the DAgger episode collection (including multiprocessing, dropping and retrying episodes whose expert fails), the policies (flow-time sampling, EMA, flow initialization), the config loaders, the projector (action tracking, feasibility, fallback logic) and the expert's plan labels, the planners (MPC, OMPL path, obstacle avoidance), and check the arm-specific pieces against references: forward kinematics and the plant's triple-integrator step and its limits.
