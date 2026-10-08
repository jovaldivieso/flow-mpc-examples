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
venv/bin/python rigid_body_dynamics/franka_dynamics.py
```

Loads the model and prints gravity torques (RNEA), mass matrix (CRBA), forward dynamics (ABA) and forward kinematics at the neutral configuration.
