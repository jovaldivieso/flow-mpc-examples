# flow-mpc-examples

Imitation learning of a CasADi MPC expert on the Franka FR3 arm with flow-matching policies (`flow`, `safeflow`), trained with DAgger.

The system is the 7-joint FR3 arm (the gripper is fixed geometry), modelled as a kinematic triple integrator per joint (jerk as the action, as in SafeFlowMPC) with a TCP-pose goal. The only collision checked is the **floor** the base stands on. The policy is conditioned directly on the robot state and the goal, with no observation encoder.

## Setup (macOS)

```bash
git clone --recurse-submodules <repo-url>   # or: git submodule update --init
python3.13 -m venv venv                     # OMPL's wheels do not support Python 3.14 yet
venv/bin/pip install -r requirements.txt
venv/bin/python -c "import pinocchio, ompl; print(pinocchio.__version__)"
```

Pinocchio (the `pin` wheel, with `eigenpy` and `coal`) and OMPL both come from PyPI, so nothing is needed from Homebrew. All commands below run from the repo root.

If DAgger prints `Class AVFFrameReceiver is implemented in both ...` warnings, `torchcodec` is installed in the venv: it and PyAV each load their own copy of ffmpeg's libavdevice. This repo's datasets have no videos, so `requirements.txt` leaves `torchcodec` out (LeRobot then decodes with PyAV); remove it with `venv/bin/pip uninstall torchcodec`.

## Robot model

[third_party/franka_description](third_party/franka_description) is the official Franka description, pinned as a git submodule. Nothing is copied: the FR3 xacro (with gripper) is expanded to a URDF at runtime with `xacro`, and Pinocchio loads meshes straight from the submodule.

`dynamics/franka_dynamics.py` (7 DoF: the arm joints; the two finger joints are locked at their open width with `pin.buildReducedModel`, so the hand and the fingers are fixed geometry on the last joint):

| | |
| --- | --- |
| state `x` | `[q, v, a]`, joint positions, velocities and accelerations (21) |
| action `u` | joint jerks (7) |
| goal | TCP pose `[position (3), rotation matrix row-major (9)]` |
| observation | `[goal, x]` (33) |

The model is kinematic: every joint is a triple integrator with piecewise constant jerk, the same exact step in the planner and in the plant (`q' = q + v dt + a dt^2/2 + j dt^3/6`, `v' = v + a dt + j dt^2/2`, `a' = a + j dt`). The limits are SafeFlowMPC's: joint positions 3 degrees inside those of the description, velocities at 92 % of them, accelerations `max_acceleration` (5 rad/s^2) and jerks `max_jerk` (15 rad/s^3) from the config (the description has none of the last two). The plant (`predict_next_state`) clips the jerk, the acceleration and the velocity and stops a joint at its position limit. SafeFlowMPC's state also carries the previous jerk (its discretization interpolates it); here the jerk is constant over a step, so the state is `[q, v, a]`. No masses or torques are modelled: an underlying joint tracking controller is assumed to follow the planned motion, so a plan can ask for motions the real arm could not track under its torque limits.

The goal is a TCP pose, and the CasADi planner works in joint space: its goal is the joint configuration of the goal pose (inverse kinematics, nearest to the start for `expert: casadi`, the end of the OMPL path for `ompl_casadi`), and its cost is the state error to it (`Q_diag`, terminal multiplier) plus the jerk effort (`R_diag`); see the docstring of `CasadiPlanner` for the variables and constraints.

### Policy output: an action chunk

The policy outputs the action directly, as an action chunk `U` of `prediction_horizon` jerks (`model.prediction_horizon` of the policy config, default 16), row `k` being the action at step `t + k`: the first row is what the system executes (for `safeflow`, the first row of the projected chunk). The label of a visited state is the expert's own plan from that state (the first `prediction_horizon` jerks of the `Plan` its `Planner` returns: the optimal controls of its MPC solve), so the dataset `action` is that chunk flattened row by row (`prediction_horizon x 7` values, named `<joint>.jerk[t+k]`). The network works on the jerks divided by their limits (`action_scale`, saved in the checkpoint), so that the unit-scale noise it starts from is of the order of the actions. `dt` is 0.1 s and `horizon` 20 in the shipped expert config; `prediction_horizon` must not exceed the expert's `horizon` (the trainer checks): DAgger labels every state with the first `prediction_horizon` jerks of the expert's plan. This differs from SafeFlowMPC, whose network outputs a position trajectory and whose safety filter solves for the jerks; here the policy's own output is the control.

### Collision: the floor

Only the floor is checked, by the OMPL path planner (the validity of a configuration) and the simulator (`is_collision`, used by the DAgger rollouts and the evaluation): the joint frames and the TCP of the arm keep `floor_clearance` (0.05 m) above the plane `z = 0` the base stands on (`FrankaDynamics.clearance` is the height of the lowest of them). Self-collision is not checked: start and goal configurations are sampled at least 10 % of every joint range away from the joint limits with their TCP in `workspace_bounds`, where the arm hardly ever hits itself, and the CasADi planner and the projector check no collisions at all.

## Core file structure

```text
flow-mpc-examples/
├── dynamics/
│   ├── dynamics.py              # Simulator protocol and base class
│   ├── franka_dynamics.py       # 7-joint FR3, gripper as fixed geometry: kinematic plant, dataset schema, sampling
│   ├── triple_integrator.py     # Point mass with jerk as the action: a toy system to calibrate the pipeline
│   ├── initial_state_utils.py   # Initial/goal-state parsing and normalization
│   └── seed_utils.py            # Seed defaults and deterministic rollout seeding
├── planning/
│   ├── config.py                # Typed validation and normalized YAML loading of the expert config
│   ├── planner.py               # The one abstract Planner (observation [, initial plan] -> Plan: joint positions and jerks)
│   ├── casadi_planner.py        # CasADi MPC (IPOPT) in joint space: state, variables, cost and constraints in its docstring
│   ├── ompl_planner.py          # OMPL RRT-Connect planner: collision-free joint-space path to a TCP goal pose (IK goal states)
│   ├── ompl_casadi_planner.py   # CasadiPlanner with the end of the OMPL path as its goal configuration and the path as its first guess
│   ├── factory.py               # DynamicsFactory and PlannerFactory: the registries of the systems and the experts the config chooses from
│   └── casadi_projector.py      # CasADi safety projection for SafeFlowMPC
├── learning/
│   ├── config_loaders.py        # YAML and policy configuration helpers
│   ├── config/                  # Policy + DAgger schedule configs (flow, safeflow)
│   ├── models/
│   │   ├── flow_velocity_net.py # Flow-matching loss and Euler integration around the MLP
│   │   ├── flow_policy.py       # FlowVelocityNet conditioned directly on [goal, state]
│   │   ├── ema.py               # Exponential moving average of the weights
│   │   ├── safe_flow_policy.py  # Flow policy + CasADi safety projection at inference (SafeFlowMPC)
│   │   └── policy.py            # Shared interface and factory
│   ├── data_utils.py            # Policy batch formatting (flat dataset action -> (prediction_horizon, nu) chunk)
│   ├── dagger/                  # beta controller, config, trainer, metrics, rollouts, utils
│   ├── train_dagger.py          # CLI: parses args, builds DaggerConfig, runs DaggerTrainer
│   └── finetune_safe.py         # CLI: safety fine-tuning of a trained flow policy on projected paths
├── scripts/
│   ├── config/franka_casadi_config.yaml   # Expert config
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

- **Flow network** (both): an MLP on the concatenation of the flattened chunk, the observation `[goal, state]` and a sinusoidal embedding of the flow time (64 values, the time scaled to [0, 1000]), `hidden_dims` 256-256-256 with Mish activations, (it replaces SafeFlowMPC's temporal U-Net, to keep the model simple).
- **Flow training** (both): rectified flow matching `x_t = t x_1 + (1 - t) x_0` against Gaussian noise, with the flow time drawn as `t = 1 - Beta(1.5, 1)` (more samples near the noisy end), the Euler integration of the velocity network at inference, and an **exponential moving average (EMA)** of the weights, `ema <- 0.995 ema + 0.005 weights` after every optimizer step: the averaged weights are less noisy than those of the last step and are the ones used for the DAgger rollouts, the evaluation and the saved `ema_model_state_dict` (`evaluate_policy.py` prefers it over `model_state_dict`).
- **Safe inference** (`safeflow`): after every Euler step the proposed action chunk is projected by `CasadiTrajectoryProjector`: it tracks the proposal (squared difference of the actions, each divided by its limit), keeps the plan within the joint, velocity, acceleration and jerk limits (no collision constraints) and ends at rest (soft terminal velocity and acceleration cost, `terminal_velocity_weight`); if a solve fails it falls back to the last plan that still validates (Theorem 2), otherwise it raises and the episode is dropped. The integration starts from the previous tick's safe actions shifted by one step, with zero jerk appended (`safeflow.init: previous_safe`; on the first tick from zero actions) or from Gaussian noise (`gaussian`). SafeFlowMPC's goal guidance acted on a position chunk and is not part of this version.
- **Safety fine-tuning**: a network trained only on straight noise-to-plan lines has never seen the projected, previous-plan starts. `learning/finetune_safe.py` (below) trains it on projected paths.

### Toy system: a triple-integrator point mass

`system: triple_integrator` in the expert config (`DynamicsFactory`, `planning/factory.py`, chooses the simulator from `system`, default `franka`) swaps the arm for a point mass in `dimension` (2) dimensions: state `[p, v, a]`, action the jerk, goal a position, the same plant, MPC, projector, dataset schema and policies, with no collisions (its task space is the position, so the inverse kinematics the planners call is the identity; `expert: ompl_casadi` works too, its path being a straight line until obstacles exist). It is for calibrating the pipeline before committing to the arm: `scripts/config/triple_integrator_casadi_config.yaml` (expert) with `learning/config/triple_integrator_casadi_flow_config.yaml` or `..._safeflow_config.yaml`:

```bash
venv/bin/python learning/train_dagger.py --experiment-name toy_flow \
  --expert-config scripts/config/triple_integrator_casadi_config.yaml \
  --policy-config learning/config/triple_integrator_casadi_flow_config.yaml
```

`scripts.meshcat_policy` also plays a point-mass checkpoint (pass `--config scripts/config/triple_integrator_casadi_config.yaml`): the point is a sphere, padded to three axes, with the same policy and expert lines; the other Meshcat scripts are for the arm only.

### Expert config (`scripts/config/franka_casadi_config.yaml`)

The expert config holds the dynamics and planner parameters, validated through [planning/config.py](planning/config.py) (unknown keys, wrong lengths and impossible values are rejected before anything runs):

- planner: `dt`, `horizon` (the expert MPC's own horizon), `Q_diag` (21, one per state: the position, velocity and acceleration error to the goal state), `R_diag` (7, one per joint: the squared jerk), `terminal_cost_multiplier`
- projector: `terminal_velocity_weight`, `fallback_terminal_velocity_tol`, `max_consecutive_fallbacks` (read by `CasadiTrajectoryProjector`, defaults in `planning/config.py`)
- `expert`: the expert planner, a name of the registry in [planning/factory.py](planning/factory.py) (`PlannerFactory`): `casadi` (the MPC, heading for the joint configuration of the goal pose nearest to the start) or `ompl_casadi` (the same MPC with the end of that path as its goal configuration, starting from the path, see below); a new planner is one more entry
- OMPL path (`OmplConfig`, for `ompl_casadi`): the `ompl_*` settings (`ompl_speed`, `ompl_replan_distance`, `ompl_timeout`, `ompl_seed`, `ompl_ik_solutions`), see below
- arm: `max_acceleration` and `max_jerk` (each a scalar or 7 per-joint limits; the joint position and velocity limits come from the description), `floor_clearance` (the height the arm's joint frames and TCP keep above the floor)
- tolerances: `pos_tol` [m], `rot_tol` [rad], `vel_tol` [rad/s] (what counts as "reached the goal and at rest"), `done_hold_steps`
- sampling: `workspace_bounds` (the TCP box `[[x], [y], [z]]` random start and goal poses are drawn from), `min_start_goal_distance`, `goal` / `randomize_goal` (omit `goal` to randomize it every episode), `initial_state_seed`, `action_noise_seed`

**The experts.** `casadi` is the MPC alone: it heads for the joint configuration of the goal pose nearest to the start (inverse kinematics from the start, from random configurations if that does not converge), solved at every step from the previous solution shifted by one step. With `expert: ompl_casadi` the expert is an `OmplCasadiPlanner`, a `CasadiPlanner` whose goal configuration is the end of an OMPL path and whose first solve starts from that path: for a goal, `OmplPlanner` plans an RRT-Connect path to the inverse-kinematics solutions of the goal pose clear of the floor (inside the joint limits, simplified, deterministic given `ompl_seed`) and returns it as a `Plan`, timed at the constant joint speed `ompl_speed`. Later solves start from the previous solution again, unless the arm is further than `ompl_replan_distance` rad from it (the learner drove it away): then a new path is planned from where it is. If the MPC solve that started from a path fails the path is the plan (`trajectory_fallbacks`); if no path is found the MPC runs as it does alone (`path_failures`). The path alone is no expert (it is not in the registry): a jerk-limited arm cannot follow it step by step, so the plain OMPL expert of the double-integrator setup was dropped.

On 32 random episodes (shipped config, goals above the floor): `casadi` and `ompl_casadi` reach the goal in 32 of 32 (mean 27 and 28 steps), in about 1 s and 1.3 s per episode.

A start pose is a random joint configuration whose TCP lies in the box; the goal is the forward kinematics of another such configuration, so it is reachable by construction.

### Training and curriculum config (`learning/config/`)

`learning/train_dagger.py`'s `--policy-config` points at a YAML file with a `model:` block (`policy_type`, `flow.num_inference_steps`, `safeflow.init`, `prediction_horizon`, `hidden_dims`) and a `training:` block with the DAgger schedule (rounds, trajectories and epochs per round, action noise, expert mixing `expert_mix_beta_*`, evaluation, `tolerance_overrides` / `eval_tolerance_overrides`, optional `workspace_bounds`, `initial_states` / `goal_states` / `training_curriculum`). Every `training:` key has a matching `--flag` that overrides it for a single run (`train_dagger.py --help`).

The training recipe is SafeFlowMPC's: AdamW (`learning_rate` 1e-4, weight decay 1e-7), `batch_size` 1024, a cosine decay of the learning rate in every round over twice the round's steps down to 1e-7 (half a period, restarted each round) and an EMA of 0.995. Unlike theirs, the network is an MLP and the policy is conditioned on `[goal pose, q, v, a]` only: SafeFlowMPC's conditioning vector also holds a 10-step joint-position history, the end-effector pose and the collision-sphere centers, and no camera data.

`franka_casadi_flow_config.yaml` and `franka_casadi_safeflow_config.yaml` differ only in `policy_type` and the `safeflow:` block (initialization, ignored by `flow`). `beta` decays by `expert_mix_beta_decay_rate` per round, but only once the evaluation success passes `expert_mix_decay_after_eval_success`: while the policy never reaches that success rate, `beta` stays at its start value.

## Pipeline

### Train

```bash
venv/bin/python learning/train_dagger.py \
  --experiment-name franka_flow \
  --expert-config scripts/config/franka_casadi_config.yaml \
  --policy-config learning/config/franka_casadi_flow_config.yaml
```

Fresh-start DAgger collects its own expert-labelled rollouts, so no offline dataset is needed (`--dagger-iterations 0` with `--repo-id`/`--dataset-root` trains offline on an existing dataset). The dataset is written to `data/lerobot_dataset_franka_dagger_<timestamp>` (LeRobot format: `observation.environment_state` = goal TCP pose (12), `observation.state` = `[q, v, a]` (21), `action` = the expert's planned jerks from that state, `prediction_horizon x 7` flattened). Checkpoints land in `outputs/train_dagger/<experiment-name>/` as `flow_dagger_checkpoint.pt` (latest) and one `flow_dagger_iter_NNN.pt` per round, next to `training_summary.yaml` (per-round success rates, the evaluation collision / timeout / solve-failure breakdown, and per round and in total the collection's attempted and discarded episodes, expert planner failures and policy-solve fallbacks, and `expert_statistics` / `total_expert_statistics`: the expert's own counters, with the OMPL path `ompl_paths`, `ompl_replans` (the learner drove the arm off the trajectory), `ompl_path_failures`, `trajectory_optimizations` and `trajectory_fallbacks`) and copies of the configs and run arguments.

A smoke test of the whole loop (collection, dataset, training, evaluation, checkpoints) that takes under a minute, with `--policy-config` set to either policy; it leaves a throwaway dataset in `data/` and checkpoints in `outputs/train_dagger/smoke/`:

```bash
venv/bin/python learning/train_dagger.py --experiment-name smoke \
  --expert-config scripts/config/franka_casadi_config.yaml \
  --policy-config learning/config/franka_casadi_flow_config.yaml \
  --dagger-iterations 2 --trajectories-per-iteration 2 --steps-per-trajectory 8 \
  --target-epochs-per-round 1 --eval-episodes 1 --eval-steps 6 --expert-mix-beta-start 0.5
```

Each round executes a mix of expert and learner actions (`expert_mix_beta`) and labels every visited state with the expert's plan. An episode whose expert solve fails, whose expert label would collide or whose executed step collides is dropped as a whole and retried from a fresh start and goal (at most `3 x trajectories_per_iteration` attempts per round).

### Safety fine-tuning (second stage)

```bash
venv/bin/python -m learning.finetune_safe \
  --checkpoint outputs/train_dagger/franka_flow/flow_dagger_checkpoint.pt \
  --repo-id local/franka_dagger_<timestamp> --dataset-root data/lerobot_dataset_franka_dagger_<timestamp> \
  --num-frames 500 --noise-samples 2 --path-steps 10 --iterations 3000 --num-workers 4
```

For `--num-frames` random frames of the dataset it projects `--noise-samples` paths each from projected Gaussian noise (scaled by the jerk limits) to the projected expert plan (`--path-steps + 1` projections per path, spread over `--num-workers` processes: this is the expensive part, about 0.1 s per projection), then trains the velocity network on them: half of the batches regress the velocity at Beta-distributed grid times onto the paths' finite differences (`FlowVelocityNet.compute_path_loss`), the other half are the ordinary Gaussian batches, with AdamW at learning rate `1e-6` and the EMA kept, as in the original. The result is written next to the checkpoint as `<stem>_safe_finetuned.pt` (same format; `--policy-type safeflow` in `evaluate_policy.py`); the base checkpoint is not changed.

### Evaluate

```bash
venv/bin/python scripts/evaluate_policy.py \
  --policy-type safeflow \
  --model-dir outputs/train_dagger/franka_flow/flow_dagger_checkpoint.pt \
  --num-steps 125 --seeds "[0, 1, 2, 3, 4]" \
  --tolerance-overrides '{"pos_tol": 0.02, "rot_tol": 0.2, "vel_tol": 0.1}' \
  [--output-path outputs/eval.npz]
```

Rolls out the expert and the policy from the same seeded start and goal and reports success rate, collision / timeout / solve-failure rates, steps, the final TCP position and rotation error, and the solve times of the expert and the policy. `--policy-type` supports `flow` and `safeflow`; they share the same weights, so a `flow`-trained checkpoint can be evaluated as either.

### Check the expert, visualize

Run the expert check before a DAgger run: it samples starts and goals like the collection does, prints per rollout whether the goal was reached, the final TCP error, the minimum floor clearance and how smooth the jerks are (`jerkiness` = mean change of the normalized jerk per step, `at jerk limit` = share of steps on the limit), then plays all rollouts in Meshcat with their TCP paths (green = reached, orange = not). `--no-viewer` prints the statistics only; `--tolerance-overrides`, `--action-noise-std` and `--seeds` mirror the training options.

```bash
venv/bin/python -m scripts.meshcat_expert --num-traj 10   # expert on 10 seeded random start/goal pairs
venv/bin/python -m scripts.meshcat_dataset --root data/lerobot_dataset_franka_dagger_<timestamp> \
  --first --episodes 10 --show-plan   # the first 10 episodes with the expert's labelled plan of every frame
venv/bin/python -m scripts.meshcat_policy --model-dir outputs/train_dagger/<name>/flow_dagger_checkpoint.pt \
  --seeds "[10000, 10001]" --show-expert   # a policy rollout next to the expert's plan from the same states
```

`meshcat_dataset` prints a table per episode first (frames, final TCP error, share of frames at rest, share of "stay" labels, i.e. a labelled plan whose motion stays within 0.05 rad of the current configuration) and takes `--indices "[0, 5, 8]"` or random episodes (`--seed`); `--no-viewer` stops after the table. It reads the parquet files directly, so it also works on a dataset a DAgger run is still writing to (unfinished files are skipped). `meshcat_policy` runs the checkpoint closed loop from seeded starts, draws the motion of the policy's action chunk (cyan) and, with `--show-expert`, of the expert's plan from the same state (green), both integrated from the current state, and prints the outcome of each rollout; `--no-viewer` prints only the outcomes.

Meshcat opens at `http://127.0.0.1:7001/static/`; the red ball marks the goal TCP position (and, for datasets, the blue dot the start).

## Notes

- Every projection and every MPC step is a CasADi/IPOPT solve of an NLP with the dynamics and the joint, velocity, acceleration and jerk limits: on an Apple-silicon laptop at `dt` 0.1 and the shipped horizons (20 for the expert, 16 for the projector) expect on the order of 30 ms per expert step and about 0.1 s per `safeflow` projection (one per flow step, plus one on the first tick of an episode), so a `safeflow` call takes a few tenths of a second, which is what dominates DAgger wall-clock time and is longer than the 0.1 s control step. `flow` inference itself takes a few tens of milliseconds or less.
- The MPC does not check collisions, so an expert rollout could end on the floor (none of 64 test episodes did); the DAgger collection drops such an episode like any other failed one.

## Tests

```bash
venv/bin/python -m unittest discover -s test -t . -v
```

The tests cover the DAgger episode collection (including dropping and retrying failed episodes), the policies (flow-time sampling, EMA, projected-path loss, flow initialization), the config loaders, the projector (action tracking, feasibility, fallback logic), the safety fine-tuning helpers and the expert's plan labels, and check the arm-specific pieces against references: the CasADi forward kinematics against Pinocchio, the plant's triple-integrator step and its limits, and the floor check.
