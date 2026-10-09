# flow-mpc-examples

Examples combining rigid body dynamics (Pinocchio), MPC and flow matching on the Franka FR3 arm.

## Setup (macOS)

```bash
git clone --recurse-submodules <repo-url>   # or: git submodule update --init
brew install pinocchio                      # Python bindings for Python 3.14
python3.14 -m venv venv
venv/bin/pip install -r requirements.txt
```

Pinocchio comes from Homebrew, so the venv needs a `.pth` file pointing to it:

```bash
cat > venv/lib/python3.14/site-packages/homebrew_pinocchio.pth <<'EOF'
/opt/homebrew/opt/pinocchio/lib/python3.14/site-packages
/opt/homebrew/lib/python3.14/site-packages
EOF
venv/bin/python -c "import pinocchio; print(pinocchio.__version__)"
```

The second path provides Pinocchio's dependencies (`eigenpy`, `coal`). Packages pip-installed into the venv (numpy, ...) take precedence over the Homebrew ones.

## Robot model

[third_party/franka_description](third_party/franka_description) is the official Franka description, pinned as a git submodule. Nothing is copied: the FR3 xacro (with gripper) is expanded to a URDF at runtime with `xacro`, and Pinocchio loads meshes straight from the submodule.

## Run

```bash
venv/bin/python -m test.meshcat_viz
```

Closed-loop CasADi/IPOPT MPC on the FR3 (`dynamics/franka_dynamics.py`, 9 DoF: 7 arm joints + 2 gripper fingers), played back in Meshcat. Start and goal are TCP poses (SE(3)): the start is turned into a joint state by one IK solve, the goal enters the cost through a CasADi forward-kinematics pose error. The red sphere marks the goal position (visual only).

Planning uses an inverse-dynamics formulation: the decision variables are joint accelerations `a`, and the torques `tau = RNEA(q, v, a)` are constrained to the effort limits. Brew's Pinocchio has no CasADi bindings, so the RNEA and FK are written symbolically in CasADi from the Pinocchio model data and checked against Pinocchio; the simulated plant uses `pin.aba`.


## Dataset (LeRobot)

`FrankaDynamics.get_dataset_features()` / `format_dataset_frame()` follow the `LeRobotDataset` schema (requires `pip install "lerobot[dataset]"`): `observation.state` = `[q, v]` (18), `observation.environment_state` = goal TCP pose (position + row-major rotation, 12), `action` = joint accelerations (9). Callers add the required `task` string per frame:

```python
ds = LeRobotDataset.create(repo_id, fps=int(round(1 / dyn.dt)), features=dyn.get_dataset_features(), robot_type="franka_fr3", use_videos=False)
ds.add_frame({**dyn.format_dataset_frame(obs, action), "task": "reach goal TCP pose"})
ds.save_episode(); ds.finalize()
```

### Collecting expert data

```bash
venv/bin/python -m learning.collect --root data/franka_expert --episodes 10 [--noise-std 0.5] [--seed N]
```

`--root` must not exist yet (LeRobot refuses to overwrite it), so delete it or pick a new name to re-collect. Expect about 15 s per saved episode.

### Visualizing collected episodes

```bash
venv/bin/python -m test.meshcat_dataset --root data/franka_expert --episodes 10 [--seed N]
```

Opens Meshcat at `http://127.0.0.1:7001/static/` and loops over randomly picked episodes until Ctrl+C. The blue dot marks the start TCP position and the red ball the goal. Both commands run from the repo root.

Each episode samples a random joint configuration (start) and the FK pose of another one (goal, reachable by construction), then runs the closed-loop MPC (seeded with `seed + attempt`). Episodes where the solver fails or the goal is not reached within `--steps` (e.g. the MPC stalls in a local minimum) are discarded; at most 3x`--episodes` attempts are made. With `--noise-std > 0`, Gaussian noise is added to the executed accelerations, but the stored label is always the expert action.
