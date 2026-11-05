# flow-mpc-examples

Imitation learning of an OMPL + time-optimal-trajectory-generation expert on the Franka FR3 arm with flow-matching policies over chunks of joint positions (`flow`, `safeflow`), trained with DAgger.

The system is the 7-joint FR3 arm (the gripper is fixed geometry), modelled as a kinematic triple integrator per joint (jerk as the plant's input) with the TCP pose as the goal. The policy outputs joint positions, like SafeFlowMPC: a safety projection (or, without one, a tracking within the limits) turns them into the jerks. It is conditioned directly on the goal pose as seen from the end effector, where the arm's body is and the history of its last positions, with no observation encoder and no velocity or acceleration.

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

Both systems inherit from the abstract `TripleIntegratorBase` (`systems/triple_integrator.py`: the plant, limits, goal and tolerances, start sampling, and what the planners need of a system: the path-planning box, the goal configuration and the bounding spheres) and hold a `SphereCollisionModel` (`systems/sphere_collision.py`, `system.collision`: the obstacles, the sphere collision check and the linearized half-spaces, built on the system's bounding spheres). Each axis/joint is a kinematic 3rd-order triple integrator (`q' = q + v dt + a dt^2/2 + j dt^3/6`, `v' = v + a dt + j dt^2/2`, `a' = a + j dt`), assuming a low-level trajectory-tracking controller underneath. For the Franka FR3 (`systems/franka_system.py`) the URDF is expanded at runtime from [third_party/franka_description](third_party/franka_description) via `xacro` and Pinocchio, with the two finger joints locked open (`pin.buildReducedModel`).

| System (`system`) | Class | State `x` (`nx`) | Goal `g` (`goal_dim`) | Simulator Obs `o = [g, x]` (`obs_dim`) | Policy Input / Dataset Features | Policy Output |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`franka`** | `FrankaSystem` (7-DoF FR3) | `[q, v, a]` (**21**) | Target TCP pose `[p, R]` in base frame (**12**) | `[g, x]` (**33**) | Conditioning (**151-D** with a history of 10):<br>- `observation.environment_state` (**81**): `[T_tcp^-1 T_goal (12), centers of the 23 bounding spheres in the base frame (69)]`<br>- `observation.history` (**70**): the last 10 joint positions, the current one last<br>Also stored: `observation.state` (**21**): `[q, v, a]`, for the guidance and the projection | Chunk `(H, 7)` of joint positions (the next `H` control periods), flattened to `(H * 7,)` in dataset `action`; the jerks `u = j` (**7**) come from the projection |
| **`triple_integrator`** | `PointMassSystem` ($d$-D point mass, default $d=2$) | `[p, v, a]` (**3d**, default **6**) | Target position `g` (**d**, default **2**) | `[g, x]` (**4d**, default **8**) | Conditioning:<br>- `observation.environment_state` (**d**): `g - p`<br>- `observation.history` (**10 d**)<br>Also stored: `observation.state` (**3d**) | Chunk `(H, d)` of positions, flattened to `(H * d,)` in dataset `action` |

## Core file structure

```text
flow-mpc-examples/
├── systems/
│   ├── system.py                # SystemSimulator: the plant, goal, observations and the policy's view of them (no dataset or rollout state)
│   ├── sphere_collision.py      # SphereCollisionModel: the obstacles, the sphere collision check and the linearized free-space half-spaces of a system
│   ├── free_space_sets.py       # FreeSpaceSets: the convex free-space set of a bounding sphere around a step of its motion (BoundPlanner Eq. 1 and 13)
│   ├── rollout_termination.py   # RolloutTerminationTracker: ends a rollout once the goal has held for done_hold_steps
│   ├── position_history.py      # PositionHistory: the last positions of a rollout, what a policy observes of where the arm has been
│   ├── system_factory.py        # SystemFactory: registry of the systems the config chooses from, which also validates their config fields
│   ├── config_fields.py         # The readers of the fields of a config mapping (types, ranges, known keys) and ConfigurationError
│   ├── franka_system.py         # 7-joint FR3, gripper as fixed geometry: kinematic plant, dataset schema, start and goal sampling
│   ├── franka_config.py         # The config of the FR3: its constants and the validation of its fields (limits, tolerances, obstacles, goal)
│   ├── franka_sampling.py       # Where an FR3 start and goal are drawn: the workspace in front of the arm (a TCP box and a hand cone) and its sampler
│   ├── franka_model.py          # The FR3 as a Pinocchio model: xacro/URDF expansion, locked fingers, SafeFlowMPC's joint limits
│   ├── franka_kinematics.py     # CasADi kinematics generated from that model (built once): TCP forward kinematics, link bounding spheres, SafeFlowMPC's IK problem (IPOPT)
│   ├── collision_spheres.py     # Bounding spheres fitted to the collision meshes (they cover the real geometry, hand and fingers included)
│   ├── triple_integrator.py     # TripleIntegratorBase: the shared triple-integrator core (plant limits, goal, obstacles, start sampling)
│   ├── point_mass_system.py     # PointMassSystem: a point mass, a toy system to calibrate the pipeline
│   ├── initial_state_utils.py   # Initial/goal-state parsing and normalization
│   └── seed_utils.py            # Seed defaults and deterministic rollout seeding
├── planning/
│   ├── config.py                # Validation and normalized YAML loading of the expert config: the planner fields, and the system's through SystemFactory
│   ├── planner_config.py        # ExpertConfig and ProjectionConfig: the typed settings of the expert and of the projector
│   ├── planner.py               # The abstract Planner (observation -> Plan: collision-free joint positions and the jerks that track them)
│   ├── totg.py                  # Time-optimal trajectory generation along a joint-space path (Kunz and Stilman 2012): BlendedPath and TimedPath
│   ├── ompl_planner.py          # OMPL RRT-Connect planner: collision-free joint-space path to the goal configuration; initialize_ompl (OMPL's seed and log level, called once at an entry point)
│   ├── ompl_casadi_planner.py      # OmplCasadiPlanner: the OMPL path timed by TOTG, the positions ahead of the arm as the plan (trimmed at obstacles), tracked by the projector
│   ├── planner_factory.py       # PlannerFactory: registry of the experts the config chooses from
│   ├── casadi_graph.py          # CasadiKinematicGraph: the stage-wise OCP skeleton (FATROP) of the projector (dynamics, limits, free-space half-spaces)
│   └── casadi_projector.py      # CasADi safety projection for SafeFlowMPC: tracks a chunk of positions, returns the safe positions and the jerks
├── learning/
│   ├── config_loaders.py        # YAML and policy configuration helpers
│   ├── config/                  # Policy + DAgger schedule configs (flow, safeflow)
│   ├── models/
│   │   ├── temporal_unet.py     # SafeFlowMPC's temporal U-Net (1D convolutions over the horizon), without its attention layers
│   │   ├── flow_velocity_net.py # Flow-matching loss and the one Euler integration (with a per-step correction) around the U-Net
│   │   ├── flow_policy.py       # FlowVelocityNet conditioned on [environment, history of positions], with the tracking of its final chunk
│   │   ├── exponential_moving_average.py  # Exponential moving average of the weights
│   │   ├── pose_guidance.py     # Goal-pose guidance of the flow's position proposals (SafeFlowMPC, Eq. 9)
│   │   ├── rest_projection.py   # Brings a chunk's final velocity and acceleration to rest (CHOMP, Eq. 39, on the positions)
│   │   ├── safe_source.py       # The safe path of a labeled state, from a safe source to a safe target, for the finetuning on the safe set
│   │   ├── smoothness.py        # CHOMP's smoothness metric and the smooth spread of a correction along a chunk
│   │   ├── safe_flow_policy.py  # Flow policy + CasADi safety projection at inference (SafeFlowMPC)
│   │   ├── policy.py            # Shared policy interface
│   │   └── policy_factory.py    # PolicySpec and PolicyFactory: a registry that builds the flow or safeflow policy from a spec
│   ├── data_utils.py            # LeRobot dataset adapter: features and frames of a system's dataset, policy batch formatting (flat dataset action -> (prediction_horizon, n) chunk of positions)
│   ├── dagger/                  # config, trainer, round summary, metrics, utils; safe_pairs.py (the safe paths of a dataset, kept in its folder, and the pool that computes them); coverage.py (how well the dataset covers the evaluation starts), collection.py (RolloutCollector), evaluation.py (RolloutEvaluator), workers.py (the RolloutWorker of a process pool), episode_starts.py, policy_action.py
│   └── train_dagger.py          # CLI: parses args, builds DaggerConfig, runs DaggerTrainer
├── scripts/
│   ├── config/                  # Expert configs (free space, obstacles, the table scene; point mass)
│   ├── evaluate_policy.py       # Expert vs policy rollouts and metrics
│   ├── meshcat_expert.py        # Expert check before DAgger: random rollouts, statistics, Meshcat playback
│   ├── meshcat_utils.py         # Shared Meshcat drawing helpers
│   ├── meshcat_dataset.py       # Dataset episodes (and their labelled plans) played back in Meshcat
│   └── meshcat_policy.py        # A checkpoint's closed-loop rollout in Meshcat, next to the expert's plan
├── test/                        # Unit tests only
└── third_party/franka_description  # Submodule
```

Policy types (`model.policy_type` in a `learning/config/` policy config): `flow` (conditional flow-matching policy of the chunk of joint positions, executed by the tracking of the plant's limits alone) and `safeflow` (`flow` wrapped in the CasADi trajectory-safety projection at inference, SafeFlowMPC, Oelerich et al., 2026). They share the same weights, so a `flow`-trained checkpoint can be evaluated as either with `evaluate_policy`; a DAgger run evaluates the policy type it trains.

What the two do, following SafeFlowMPC:

- **Flow network** (both): SafeFlowMPC's temporal U-Net (`TemporalUnet`, the same outputs as `safe_flow_mpc/Models/models.py` for the same weights, without its unused attention layers): a 1D convolution over the chunk's horizon with the joint positions as channels, `dim` 32 and `dim_mults` (1, 2, 4, 8) (about 4 M parameters), every residual block conditioned on the sum of an embedding of the flow time and of the observation `[goal in the TCP frame, bounding-sphere centers, history of the last positions]`: the policy is not conditioned on the velocity and acceleration of the arm (`observation.state`), which are noisy to learn from; the guidance and the projection read them. `prediction_horizon` has to be divisible by `2 ** (len(dim_mults) - 1)`.
- **The chunk** (both): the flow runs over the joint positions of the next `prediction_horizon` control periods (row k is the configuration k + 1 periods ahead), normalized by half the range of each joint (`TripleIntegratorBase.position_scale`) against the unit noise prior; the jerks are what the safety projection (or, for a `flow` policy, the tracking within the limits) returns for the final chunk, as in SafeFlowMPC where the filter computes the jerks.
- **Flow training** (both): rectified flow matching `x_t = t x_1 + (1 - t) x_0` against Gaussian noise, with the flow time drawn as `t = 1 - Beta(1.5, 1)` (more samples near the noisy end), the Euler integration of the velocity network at inference, and an **exponential moving average (EMA)** of the weights (`ema <- 0.995 ema + 0.005 weights` after every optimizer step), which is what the DAgger rollouts, the evaluation and the saved `ema_model_state_dict` use (`evaluate_policy.py` prefers it over `model_state_dict`).
- **Execution** (both): the policy is asked once every `execution_horizon` control periods (`model.execution_horizon`, at most the `prediction_horizon`) and the first `execution_horizon` jerks of the chunk it selects are executed one after the other, as Diffusion Policy executes half of its predicted chunk (Chi et al., 2025). The jerks come from one plan, so they stay within its safe horizon; the warm start of the projection and the initial chunk of the next tick shift by `execution_horizon` control periods (`tick_steps` of the projector). DAgger collection draws the expert-or-policy mix once per block (`expert_mixing_beta`) and the expert labels every step.
- **Safe prior** (`safeflow` policy type, SafeFlowMPC Sec. II-B): the source distribution of the flow is the distribution over safe chunks, as the safe flow policy starts its integration on the safe set. A safeflow run trains on safe rows only: every row starts the flow from a safe source chunk of its state and learns towards a safe target; a flow run trains from Gaussian noise towards the expert's chunk throughout. The authors' two steps are a run with the flow config followed by one with the safeflow config from its checkpoint (`--init-checkpoint`). As in the authors' dataset creation, a source is a chunk of jerks drawn uniformly within the jerk limits and rolled out from the state, and a target is the expert's label (clear of the obstacles by its trimming), both made safe by the safety projection (`SafeSource`: a `CasadiTrajectoryProjector` with the limits, the dynamics, the obstacles and the terminal rest; safe is on or off, so a source or target is clear of the obstacles, which is what the guarantee of the safety projection at inference assumes). The flow learns along the projected path between the two (Eq. 7), kept at 11 points: a point of the straight line between source and target is kept when it is clear of the obstacles (limits, dynamics and rest are convex, so only the obstacles can make it unsafe) and is otherwise projected by the same projection; the flow follows the polyline through the 11 points, its velocity the slope of the segment at the flow time. A frame whose source, target or a point has no safe plan has no path and is left out of the training, as in the authors' dataset (about a seventh of the frames of the table scene). The paths of a frame are computed once and kept in `safe_pairs.npz` in the dataset's folder (about 60 ms a frame in all on the table scene, split over the `num_workers` processes when a round adds more than 200 frames; a frame's source depends on its index alone, so the paths do not depend on the number of workers), so one script covers both training steps and DAgger's own states are the ones the safe starts are drawn from.
- **Goal guidance** (both, always on; `PoseGuidance` in `learning/models/pose_guidance.py`, Eq. 9, 13 and 14 of SafeFlowMPC): after every Euler step the proposed chunk of positions is moved toward the goal pose. The positions of the chunk get a damped Gauss-Newton step of the TCP pose error (the se(3) logarithm `log(T_tcp^-1 T_goal)` of the pose of the goal in the TCP frame, like the policy observes it, with the exact Jacobian of that error: the body Jacobian times the left Jacobian inverse `J_l^-1` of SE(3), the goal read from the policy observation) with the weight `min(1, exp(-10 (d_k / d_0 - 0.09))) k / H`, so it only acts once the TCP is within roughly a tenth of its start distance `d_0` from the goal and more on the later positions of the horizon; the moves are first spread along the chunk by the smoothness metric of CHOMP (`SmoothSpread` in `learning/models/smoothness.py`: the covariant update `A^-1 g` of Ratliff et al. 2009, with `A` the squared first and second differences, applied as `(I + mu A)^-1` so that it never enlarges a move). Close to the goal the chunk is then brought to rest: `RestProjection` (`learning/models/rest_projection.py`) scales its final velocity and acceleration (backward differences of the last three positions) by one minus the closeness of its last position to the goal, CHOMP's projection onto a constraint (Eq. 39 of Zucker et al. 2013) on the positions with the smoothness metric of the positions, so the change is spread smoothly along the chunk. A `flow` policy applies no other correction to the iterates (only its final chunk is looked at; the tracking that executes it keeps the plant's limits), a `safeflow` policy projects every iterate.
- **Safe inference** (`safeflow`): after every Euler step the proposed chunk of positions is projected by `CasadiTrajectoryProjector`: each step is a converged FATROP solve of a convex QP linearized along its own reference (SafeFlowMPC solves one QP per step with acados SQP-RTI, which linearizes at each step too and rejects a step whose slack exceeds a tolerance; here a solution that still penetrates an obstacle after one relinearization counts as a failed projection). It tracks the reference positions (squared position difference with weight 1, small weights on the velocities, accelerations and jerks, as the authors' safety filter does), keeps the plan within the joint, velocity, acceleration and jerk limits and clear of the obstacles (linearized sphere half-spaces with a penalized slack, see "Obstacles") and ends at rest (soft terminal velocity and acceleration cost, `terminal_velocity_weight`); its safe positions are the next iterate of the flow and its jerks what the arm executes. If a projection fails it falls back to the last plan that still validates (Theorem 2; at the start of an episode, holding still, which validates when the arm is at rest and clear of the obstacles), otherwise it raises `PlannerSolveError` (collection then executes the expert's action of that step, evaluation counts a solve failure). The fallbacks are counted per control tick (a policy call, also while the arm is held at rest), so a policy whose projection keeps failing ends the episode after `max_consecutive_fallbacks` ticks instead of holding the arm still until the timeout. The integration starts from the previous tick's safe positions `execution_horizon` control periods later (`safeflow.init: previous_safe`; on the first tick from the projection of the arm coasting at zero jerk, which brakes a moving arm and keeps it within its limits and clear of the obstacles: the flow starts on the safe set, as in SafeFlowMPC) or from Gaussian noise (`gaussian`).

### Toy system: `triple_integrator`

Set `system: triple_integrator` in the expert config (`scripts/config/triple_integrator_config.yaml` with `learning/config/triple_integrator_flow_config.yaml` or `..._safeflow_config.yaml`) to calibrate the pipeline on a point mass before running on the arm:

```bash
venv/bin/python learning/train_dagger.py --experiment-name toy_flow \
  --expert-config scripts/config/triple_integrator_config.yaml \
  --policy-config learning/config/triple_integrator_flow_config.yaml
```

`scripts.meshcat_policy` also plays a point-mass checkpoint (pass `--config scripts/config/triple_integrator_config.yaml`).

### Expert config (`scripts/config/franka_table_config.yaml`)

The expert config holds the system and planner parameters, validated through [planning/config.py](planning/config.py) (the system's own fields by the system class, `validate_config`) (unknown keys, wrong types and impossible values are rejected before anything runs):

- `dt`, `expert`: the expert planner, a name of the registry in [planning/planner_factory.py](planning/planner_factory.py) (`PlannerFactory`): `ompl_casadi`; a new planner is one more entry
- `horizon`: the rows of positions the expert plans per state (DAgger labels a state with the first `prediction_horizon` of them, so it must not exceed this)
- OMPL path and timing (`ExpertConfig`): `ompl_timeout`, `ompl_candidates`, `ompl_seed`, `replan_distance`, `progress_window`, `totg_velocity_scale`, `totg_acceleration_scale`, `totg_max_deviation`, see below
- projector (`ProjectionConfig`, read by `CasadiTrajectoryProjector`, defaults in `planning/planner_config.py`): `projection_velocity_weight`, `projection_acceleration_weight`, `projection_jerk_weight`, `terminal_velocity_weight`, `fallback_terminal_velocity_tol`, `max_consecutive_fallbacks`
- arm: `max_acceleration` and `max_jerk` (each a scalar or 7 per-joint limits; the joint position and velocity limits come from the description)
- tolerances: `pos_tol` [m, TCP to goal position], `vel_tol` [rad/s] (what counts as "reached the goal and at rest"), `done_hold_steps`
- `obstacles`: static spherical obstacles in the world frame, a list of `position` and `radius` [m] (see "Obstacles")
- sampling: `min_start_goal_distance` [m, TCP to goal position], `goal` (a pose: position and a row-major rotation matrix, 12 numbers) / `randomize_goal` (omit `goal` to randomize it every episode), `initial_state_seed`, `action_noise_seed`. A start (at rest) and a goal are drawn anywhere in front of the arm (a TCP box x 0.2–0.8, y ±0.6, z 0.1–0.8 m, the hand between pointing forward and pointing down). A start is the configuration of a pose, a goal a pose, both found by inverse kinematics from up to 6 random configurations (the first valid solution is taken, so a pose is reached in different configurations from one draw to the next), inside the joint margins and free of obstacles. `scripts/config/franka_table_config.yaml` is a table of 18 spheres, too wide for the arm to go down at its sides.

**The expert** (`expert: ompl_casadi`, `OmplCasadiPlanner`). For a goal, `OmplPlanner` plans an RRT-Connect path to the goal configuration `q_goal` (the inverse kinematics solution of the goal pose nearest to the start; inside the joint limits, collision-free, simplified to a few waypoints, deterministic given `ompl_seed`, which the entry points pass to `initialize_ompl` once); with `ompl_candidates` > 1 it plans that many paths and keeps the one along which the end effector travels the shortest. The path is timed by time-optimal trajectory generation (`planning/totg.py`, Kunz and Stilman, RSS 2012, written here): `BlendedPath` rounds every corner of the waypoints with a circular blend that cuts it by at most `totg_max_deviation` rad, which makes the path differentiable, and `TimedPath` finds the fastest rest-to-rest trajectory that follows it exactly within `totg_velocity_scale` of the joint velocity limits and `totg_acceleration_scale` of the acceleration limits (the arm tracks it within its jerk limit), by integrating the path velocity in the phase plane forward with the largest acceleration and backward with the smallest from the switching points of the limit curves. The expert labels a state with the positions of the next `horizon` control periods along that trajectory from the point of the path nearest to the arm (the progress is searched within `progress_window` rad of path length and never goes back); there is no rest at a lookahead point to head for, the arm follows the timed path. A new path is planned at the first state of an episode and again from where the arm is when it is further than `replan_distance` rad from the path (the learner drove it away). A label never penetrates an obstacle: its rows from the first one at which a bounding sphere does are replaced by the last row before it (the current configuration if there is none), and the expert has no relaxation of the obstacle slack (`trimmed_labels` counts those labels). The expert executes its plan with the jerks of the safe `CasadiTrajectoryProjector` tracking the positions; when that finds no safe plan from the arm's state (an arm too fast to brake in time) the expert fails to solve there, as it does when no path exists.

### Obstacles

The obstacles are static spheres. The arm is covered by bounding spheres fitted to its collision meshes (`systems/collision_spheres.py`: the meshes are split recursively along their principal axis until a minimal enclosing ball has a radius of at most 8 cm, so the hand and the fingers are covered too). Everything that plans or filters keeps these spheres clear of the obstacles: the projector as the free-space sets of BoundPlanner (`systems/free_space_sets.py`, Eq. 1 and 13): at every node each sphere gets at most 4 half-spaces that touch the obstacles (inflated by the sphere radius and a 5 mm buffer) nearest to the step the sphere makes into that node, skipping obstacles an earlier half-space already excludes, so the rows per node do not grow with the obstacles; they are linearized in the joints along the reference trajectory (`n^T J_s(q_ref) q >= b`) with a slack per sphere and node (shared by its half-spaces) penalized by 1e3 per meter plus 1 per square meter, and a solution that penetrates an obstacle or has a slack above 1 mm is linearized again within the same sets and, if still rejected, counts as a failed solve; OMPL uses the sphere check as its validity check. The evaluation counts a collision as a touch of the real collision meshes with an obstacle, which the spheres cover; the DAgger collection does not check for collisions, but no label penetrates an obstacle (the expert's plan is trimmed at the first row at which a bounding sphere does). The policy does not observe the obstacles: they act through the expert's labels and the projector, so one checkpoint can be evaluated with any `obstacles` list. FATROP's time is linear in the horizon and the stage has one slack per sphere, and the half-spaces per sphere are capped, so it does not grow with the obstacles.

### Training and curriculum config (`learning/config/`)

`learning/train_dagger.py`'s `--policy-config` points at a YAML file with a `model:` block (`policy_type`, `flow.num_inference_steps`, `flow.history_length`, `safeflow.init`, `prediction_horizon`, `execution_horizon`, `unet.dim` / `unet.dim_mults`) and a `training:` block with the DAgger schedule (rounds, `trajectories_per_iteration`, `target_epochs_per_round`, `expert_beta_per_iteration`, `num_workers`, action noise, evaluation, `tolerance_overrides` / `eval_tolerance_overrides`, optional `workspace_bounds`, `initial_states` / `goal_states` / `training_curriculum`). Every `training:` key has a matching `--flag` that overrides it for a single run (`train_dagger.py --help`).

A `training_curriculum` entry per round is `random` (random starts and goals in the workspace) or `config` (the explicit `initial_states` / `goal_states`); without a curriculum every round is `random`. After every round `training_summary.yaml` has, under `coverage`, how far the starts of the first 50 evaluation episodes are from the dataset: the median and 90th percentile of the distance of their observation (the goal in the TCP frame, the sphere centers and the history of positions) to the nearest observation of the dataset, in standard deviations of each component, and `far_fraction`, the share of the starts farther than the 90th percentile of that distance for the first observations of the dataset's own episodes to the other episodes.

The training recipe is SafeFlowMPC's: AdamW (`learning_rate` 1e-4, weight decay 1e-7), `batch_size` 1024, a cosine decay of the learning rate in every round over twice the round's steps down to 1e-7 (half a period, restarted each round) and an EMA of 0.995.

`franka_flow_config.yaml` and `franka_safeflow_config.yaml` differ in `policy_type`, the `safeflow:` block (initialization, ignored by `flow`) and the expert mixing, and define the per-round expert action mixing probability directly via `expert_beta_per_iteration` (either a single value broadcast across all rounds or one value per round).

## Pipeline

### Train

```bash
venv/bin/python learning/train_dagger.py \
  --experiment-name franka_flow \
  --expert-config scripts/config/franka_table_config.yaml \
  --policy-config learning/config/franka_flow_config.yaml
```

Fresh-start DAgger collects its own expert-labelled rollouts, so no offline dataset is needed (`--dagger-iterations 0` with `--repo-id`/`--dataset-root` trains offline on an existing dataset). The dataset is written to `data/lerobot_dataset_franka_dagger_<timestamp>` (LeRobot format: `observation.environment_state` = `[T_tcp^-1 T_goal (12), bounding-sphere centers in the base frame (69)]` (81; the goal pose in the TCP frame, no goal joint configuration since inverse kinematics is the experts' privilege, and where the arm's own body is, from forward kinematics, so the policy can relate it to fixed obstacles it does not observe), `observation.history` = the last `flow.history_length` joint positions, the current one last (70 for 10), `observation.state` = `[q, v, a]` (21; what the guidance and the projection start from, not a conditioning of the policy), `action` = the expert's planned joint positions of the next `prediction_horizon` control periods, `prediction_horizon x 7` flattened). Checkpoints land in `outputs/train_dagger/<experiment-name>/` as `flow_dagger_checkpoint.pt` (latest) and one `flow_dagger_iter_NNN.pt` per round, next to `training_summary.yaml` (per-round success rates and the wall-clock seconds of the round's phases under `timing_seconds` (the collection, the training, every evaluation by policy type, the coverage measurement and the whole round), and the mean wall-clock time of a policy call over the evaluation (`eval_mean_inference_ms`; the pool's workers run side by side, so it is a little more than a call alone takes), the evaluation collision / timeout / solve-failure breakdown, the collection's attempted, discarded and truncated episodes, expert planner failures and policy-solve fallbacks, and the expert's own counters in `expert_statistics` / `total_expert_statistics`: `ompl_paths`, `ompl_replans`, `ompl_path_failures`, `trimmed_labels`) and copies of the configs and run arguments.

A smoke test of the whole loop (collection, dataset, training, evaluation, checkpoints) that takes about a minute, with `--policy-config` set to either policy; it leaves a throwaway dataset in `data/` and checkpoints in `outputs/train_dagger/smoke/`:

```bash
venv/bin/python learning/train_dagger.py --experiment-name smoke \
  --expert-config scripts/config/franka_table_config.yaml \
  --policy-config learning/config/franka_flow_config.yaml \
  --dagger-iterations 2 --trajectories-per-iteration 6 6 --steps-per-trajectory 30 \
  --target-epochs-per-round 1 --eval-episodes 3 --eval-steps 30 --expert-beta-per-iteration 1.0 0.5 \
  --batch-size 64 --max-train-steps 20
```

Each round executes a mix of expert and learner actions (`expert_beta_per_iteration`) and labels every visited state with the expert's plan: the positions of the next control periods along the timed path, up to the last collision-free one. An episode whose expert planner fails to solve, to find a path or to track it safely (`planner_failures`) ends there and keeps the states it labeled before (`truncated_episodes`); one that fails at its very first state has none and is dropped and retried from a fresh start and goal (`discarded_episodes`), at most `3 x trajectories_per_iteration` attempts per round. The round summary counts the failures by the solver's error under `expert_failure_causes`. Nothing else ends or drops an episode: a learner that collides keeps going, and the expert labels the states it reaches. A policy solve that fails (`safeflow`) executes the expert's action of that step instead (`policy_solve_fallbacks`).

### Evaluate

```bash
venv/bin/python -m scripts.evaluate_policy \
  --model-dir outputs/train_dagger/franka_flow/flow_dagger_checkpoint.pt \
  --config scripts/config/franka_obstacles_config.yaml \
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

- Every projection is a convex CasADi/FATROP solve with the linear triple-integrator dynamics, box constraints on joint positions, velocities, accelerations and jerks and the linearized free-space half-spaces, and DAgger rollout collection and evaluation parallelize across episodes via `num_workers`.

## Tests

```bash
venv/bin/python -m unittest discover -s test -t . -v
```

The tests cover the safe sources and the rest projection, the execution of blocks of jerks, the time-optimal trajectory generation (blends, limits, rest-to-rest, random paths), the expert (OMPL path, timing, progress, label trimming, replanning), the DAgger episode collection (including multiprocessing, and episodes whose expert fails: truncated, or dropped and retried when it fails at the first state), the policies (flow-time sampling, EMA, flow initialization, guidance, conditioning on the history), the config loaders, the projector (position tracking, feasibility, fallback logic), and check the arm-specific pieces against references: forward kinematics and the plant's triple-integrator step and its limits.
