# flow-mpc-examples

Examples combining rigid body dynamics (Pinocchio), MPC and flow matching on the Franka FR3 arm.

## Setup (macOS)

```bash
git clone --recurse-submodules <repo-url>   # or: git submodule update --init
brew install pinocchio                      # Python bindings for Python 3.14
python3.14 -m venv venv
venv/bin/pip install xacro pyyaml
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
