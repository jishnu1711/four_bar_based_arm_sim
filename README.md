# Shoulder-Mounted Parallelogram Arm Simulator

This project contains simple 2D kinematic simulators for a 2-DOF planar arm
with both actuators mounted at the shoulder/base.

It only simulates the linkage motion up to the wrist. It does not model a
turntable, gripper, PCB probe, tool, dynamics, motor drivers, or contacts.

## Mechanism

Both motor axes are at the shoulder/base point `A`:

- shoulder motor drives the upper arm `A-E`
- elbow motor drives the input rocker `A-P`

The elbow motion is transmitted through a parallelogram:

```text
        P ---------------- Q
       /                  /
      /                  /
     A ---------------- E -------- W
    shoulder           elbow      wrist
```

Where:

- `A-E` is the upper arm
- `A-P` is the elbow actuator input rocker
- `P-Q` is the parallel coupler
- `E-Q` is the elbow output rocker
- `E-W` is the forearm

The two independent commanded angles are:

- `shoulder_angle_deg`
- `elbow_motor_angle_deg`

The dependent linkage points `P`, `Q`, `E`, and `W` are computed from the
parallelogram geometry.

## Install

Use a virtual environment. This avoids problems with Ubuntu's externally
managed system Python.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install numpy matplotlib tornado
```

If `python3 -m venv .venv` fails because `venv` is missing, install it first:

```bash
sudo apt install python3-venv
```

## Forward-Kinematics Simulator

```bash
.venv/bin/python fourbar_leg_sim.py
```

The simulator uses Matplotlib's `WebAgg` backend by default. After running the
script, open the exact URL printed in the terminal, for example:

```text
To view figure, visit http://127.0.0.1:8988
```

If that port is busy, Matplotlib may print `8989` or another nearby port. Use
whatever URL it prints.

Stop the simulator with `Ctrl+C` in the terminal.

### Controls

The plot has:

- a `shoulder motor` slider for the upper arm angle
- an `elbow motor at shoulder` slider for the parallelogram input rocker
- a `Play/Pause` button for the built-in animation

The readout shows:

- shoulder motor angle
- elbow motor angle
- effective elbow angle
- elbow position `E`
- wrist position `W`

## Inverse-Kinematics Simulator

Run the coordinate-input simulator with:

```bash
.venv/bin/python fourbar_leg_ik.py
```

This simulator takes the desired wrist point `W = (X, Y)` as input and solves
the two base-mounted motor angles:

- shoulder motor angle
- elbow motor rocker angle

The plot has:

- a `target W X` slider
- a `target W Y` slider
- a branch button to switch between the two possible IK solutions
- a trace of accepted target positions

If the requested point is unreachable, the red target marker still moves, but
the linkage stays at the last valid pose and the readout explains why.

The IK function to reuse later is:

```python
solve_inverse_kinematics(target_xy, branch="elbow_down")
```

The UI method to replace later with coordinate commands is:

```python
app.set_target_xy(x_mm, y_mm)
```

## Geometry Knobs

All dimensions and limits are near the top of `fourbar_leg_sim.py`. The IK
script imports these values, so changing the geometry in one place updates both
simulators.

```python
UPPER_ARM_LENGTH = 145.0
FOREARM_LENGTH = 130.0
PARALLELOGRAM_ROCKER_LENGTH = 48.0
FOREARM_TO_ROCKER_ANGLE_DEG = 0.0
SHOULDER_PIVOT = np.array([0.0, 0.0])

SHOULDER_ANGLE_MIN_DEG = -10.0
SHOULDER_ANGLE_MAX_DEG = 115.0
ELBOW_MOTOR_ANGLE_MIN_DEG = -120.0
ELBOW_MOTOR_ANGLE_MAX_DEG = 60.0
```

All lengths are in millimeters. Angles are in degrees.

## Motor Integration

The function to reuse later with real motor commands is:

```python
app.set_motor_angles(shoulder_angle_deg, elbow_motor_angle_deg)
```

The kinematic solve itself is in:

```python
solve_arm_pose(shoulder_angle_deg, elbow_motor_angle_deg)
```
