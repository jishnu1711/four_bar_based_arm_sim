#!/usr/bin/env python3
r"""Simple 2D simulation of a shoulder-mounted parallelogram arm.

This is a kinematic visualization only: NumPy + Matplotlib, no robotics
toolboxes and no physics engine.

Mechanism
---------

Both actuators are at the shoulder/base point A:

    shoulder motor:       drives upper arm A-E
    elbow motor at A:     drives input rocker A-P

The elbow is driven through a parallelogram:

        P ---------------- Q
       /                  /
      /                  /
     A ---------------- E -------- W
    shoulder           elbow      wrist

    A-E = upper arm
    A-P = elbow actuator input rocker
    P-Q = parallel coupler, always parallel to A-E
    E-Q = elbow output rocker, always parallel to A-P
    E-W = forearm, rigidly attached to the elbow output rocker

There is no turntable, gripper, PCB probe, or tool model here.
Only the arm mechanism up to the wrist is shown.

The method to reuse later with motor commands is:

    app.set_motor_angles(shoulder_angle_deg, elbow_motor_angle_deg)
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

MPL_CONFIG_DIR = Path(__file__).with_name(".mplconfig")
MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib


def configure_matplotlib_backend() -> None:
    """Use a backend that can show Matplotlib sliders reliably."""
    if os.environ.get("MPLBACKEND"):
        return
    matplotlib.use("WebAgg", force=True)


configure_matplotlib_backend()
matplotlib.rcParams["webagg.address"] = "127.0.0.1"
matplotlib.rcParams["webagg.port"] = 8988
matplotlib.rcParams["webagg.port_retries"] = 20
matplotlib.rcParams["webagg.open_in_browser"] = False

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider


# ---------------------------------------------------------------------------
# User-configurable geometry
# ---------------------------------------------------------------------------

# Main arm lengths in millimeters.
UPPER_ARM_LENGTH = 145.0
FOREARM_LENGTH = 130.0

# Parallelogram rocker length. A-P and E-Q have this same length.
PARALLELOGRAM_ROCKER_LENGTH = 48.0

# Angle between the elbow output rocker E-Q and the forearm E-W.
# 0 deg means the forearm is collinear with the output rocker.
FOREARM_TO_ROCKER_ANGLE_DEG = 0.0

# Fixed shoulder/base position.
SHOULDER_PIVOT = np.array([0.0, 0.0])

# Motor angle limits in degrees, measured from +X in the plot.
SHOULDER_ANGLE_MIN_DEG = -10.0
SHOULDER_ANGLE_MAX_DEG = 115.0
ELBOW_MOTOR_ANGLE_MIN_DEG = -120.0
ELBOW_MOTOR_ANGLE_MAX_DEG = 60.0

# Starting motor commands.
START_SHOULDER_ANGLE_DEG = 45.0
START_ELBOW_MOTOR_ANGLE_DEG = -45.0

# Animation parameters. Sliders still work while animation is paused.
PLAY_ANIMATION_ON_START = True
ANIMATION_SPEED = 0.75


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmPose:
    """All mechanism joint locations for one pair of motor commands."""

    A: np.ndarray  # shoulder/base pivot
    E: np.ndarray  # elbow pivot
    W: np.ndarray  # wrist point
    P: np.ndarray  # input rocker tip
    Q: np.ndarray  # output rocker tip
    shoulder_angle_deg: float
    elbow_motor_angle_deg: float
    effective_elbow_angle_deg: float


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------


def unit_from_angle(angle_deg: float) -> np.ndarray:
    """Unit vector at angle_deg from +X."""
    theta = math.radians(angle_deg)
    return np.array([math.cos(theta), math.sin(theta)])


def rot2(theta_rad: float) -> np.ndarray:
    """2D rotation matrix."""
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)
    return np.array([[c, -s], [s, c]], dtype=float)


def solve_arm_pose(shoulder_angle_deg: float, elbow_motor_angle_deg: float) -> ArmPose:
    """Compute all dependent parallelogram joints.

    The two independent inputs are:
      - shoulder_angle_deg for upper arm A-E
      - elbow_motor_angle_deg for input rocker A-P

    Parallelogram closure is direct:
      - P = A + rocker vector
      - E = A + upper-arm vector
      - Q = E + same rocker vector

    This makes A-P-Q-E a parallelogram, so E-Q is always parallel to A-P and
    P-Q is always parallel to A-E. The forearm is rigidly attached to E-Q.
    """
    A = SHOULDER_PIVOT.copy()

    upper_dir = unit_from_angle(shoulder_angle_deg)
    rocker_dir = unit_from_angle(elbow_motor_angle_deg)
    forearm_dir = rot2(math.radians(FOREARM_TO_ROCKER_ANGLE_DEG)) @ rocker_dir

    E = A + UPPER_ARM_LENGTH * upper_dir
    P = A + PARALLELOGRAM_ROCKER_LENGTH * rocker_dir
    Q = E + PARALLELOGRAM_ROCKER_LENGTH * rocker_dir
    W = E + FOREARM_LENGTH * forearm_dir

    effective_elbow_angle_deg = elbow_motor_angle_deg - shoulder_angle_deg

    return ArmPose(
        A=A,
        E=E,
        W=W,
        P=P,
        Q=Q,
        shoulder_angle_deg=shoulder_angle_deg,
        elbow_motor_angle_deg=elbow_motor_angle_deg,
        effective_elbow_angle_deg=effective_elbow_angle_deg,
    )


# ---------------------------------------------------------------------------
# Visualization app
# ---------------------------------------------------------------------------


class ShoulderParallelogramArmApp:
    """Matplotlib UI wrapper around the shoulder-mounted parallelogram arm."""

    def __init__(self) -> None:
        self.is_playing = PLAY_ANIMATION_ON_START

        self.fig, self.ax = plt.subplots(figsize=(8.5, 7.5))
        plt.subplots_adjust(left=0.08, right=0.98, bottom=0.24, top=0.94)

        self.ax.set_title("2-DOF shoulder-mounted parallelogram arm")
        self.ax.set_xlabel("X (mm)")
        self.ax.set_ylabel("Y (mm)")
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True, color="0.90", linewidth=0.8)

        reach = UPPER_ARM_LENGTH + FOREARM_LENGTH + PARALLELOGRAM_ROCKER_LENGTH
        self.ax.set_xlim(-reach * 0.55, reach * 1.05)
        self.ax.set_ylim(-reach * 0.80, reach * 0.95)

        self.base_line, = self.ax.plot([], [], color="0.25", linewidth=7, solid_capstyle="round", label="fixed base")
        self.upper_arm_line, = self.ax.plot([], [], color="#111827", linewidth=5, label="upper arm A-E")
        self.forearm_line, = self.ax.plot([], [], color="#111827", linewidth=5, label="forearm E-W")
        self.input_rocker_line, = self.ax.plot([], [], color="#2563eb", linewidth=4, label="elbow motor rocker A-P")
        self.output_rocker_line, = self.ax.plot([], [], color="#f97316", linewidth=4, label="elbow output rocker E-Q")
        self.coupler_line, = self.ax.plot([], [], color="#16a34a", linewidth=4, label="parallel coupler P-Q")
        self.parallelogram_outline, = self.ax.plot([], [], color="#64748b", linewidth=1.4, linestyle="--", label="parallelogram loop")

        self.joints_scatter = self.ax.scatter([], [], s=64, color="white", edgecolor="black", zorder=5)
        self.joint_labels = {
            name: self.ax.text(0, 0, name, fontsize=10, weight="bold", ha="center", va="center")
            for name in ("A", "P", "E", "Q", "W")
        }
        self.coord_text = self.ax.text(
            0.02,
            0.98,
            "",
            transform=self.ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            bbox=dict(facecolor="white", edgecolor="0.85", boxstyle="round,pad=0.35"),
        )
        self.ax.legend(loc="lower right")

        shoulder_slider_ax = self.fig.add_axes([0.20, 0.14, 0.67, 0.035])
        elbow_slider_ax = self.fig.add_axes([0.20, 0.085, 0.67, 0.035])
        button_ax = self.fig.add_axes([0.20, 0.025, 0.16, 0.04])

        self.shoulder_slider = Slider(
            ax=shoulder_slider_ax,
            label="shoulder motor (deg)",
            valmin=SHOULDER_ANGLE_MIN_DEG,
            valmax=SHOULDER_ANGLE_MAX_DEG,
            valinit=START_SHOULDER_ANGLE_DEG,
        )
        self.elbow_slider = Slider(
            ax=elbow_slider_ax,
            label="elbow motor at shoulder (deg)",
            valmin=ELBOW_MOTOR_ANGLE_MIN_DEG,
            valmax=ELBOW_MOTOR_ANGLE_MAX_DEG,
            valinit=START_ELBOW_MOTOR_ANGLE_DEG,
        )
        self.play_button = Button(button_ax, "Pause" if self.is_playing else "Play")

        self.shoulder_slider.on_changed(self.update_from_sliders)
        self.elbow_slider.on_changed(self.update_from_sliders)
        self.play_button.on_clicked(self.toggle_play)

        self.draw_pose(solve_arm_pose(START_SHOULDER_ANGLE_DEG, START_ELBOW_MOTOR_ANGLE_DEG))
        self.animation: FuncAnimation | None = None

    @staticmethod
    def set_line(line, *points: np.ndarray, closed: bool = False) -> None:
        arr = np.array(points)
        if closed:
            arr = np.vstack([arr, arr[0]])
        line.set_data(arr[:, 0], arr[:, 1])

    def update_joint_labels(self, pose: ArmPose) -> None:
        positions = {
            "A": pose.A,
            "P": pose.P,
            "E": pose.E,
            "Q": pose.Q,
            "W": pose.W,
        }
        offsets = {
            "A": np.array([-10.0, -12.0]),
            "P": np.array([0.0, -14.0]),
            "E": np.array([12.0, -12.0]),
            "Q": np.array([12.0, 10.0]),
            "W": np.array([12.0, -10.0]),
        }
        for name, text in self.joint_labels.items():
            p = positions[name] + offsets[name]
            text.set_position((p[0], p[1]))

    def draw_pose(self, pose: ArmPose) -> None:
        """Draw an already-solved pose."""
        base_left = pose.A + np.array([-42.0, -22.0])
        base_right = pose.A + np.array([42.0, -22.0])

        self.set_line(self.base_line, base_left, base_right)
        self.set_line(self.upper_arm_line, pose.A, pose.E)
        self.set_line(self.forearm_line, pose.E, pose.W)
        self.set_line(self.input_rocker_line, pose.A, pose.P)
        self.set_line(self.output_rocker_line, pose.E, pose.Q)
        self.set_line(self.coupler_line, pose.P, pose.Q)
        self.set_line(self.parallelogram_outline, pose.A, pose.P, pose.Q, pose.E, closed=True)

        joints = np.vstack([pose.A, pose.P, pose.E, pose.Q, pose.W])
        self.joints_scatter.set_offsets(joints)
        self.update_joint_labels(pose)

        self.coord_text.set_text(
            f"Shoulder motor: {pose.shoulder_angle_deg:8.2f} deg\n"
            f"Elbow motor:    {pose.elbow_motor_angle_deg:8.2f} deg\n"
            f"Elbow relative: {pose.effective_elbow_angle_deg:8.2f} deg\n"
            f"Elbow E:        ({pose.E[0]:7.1f}, {pose.E[1]:7.1f}) mm\n"
            f"Wrist W:        ({pose.W[0]:7.1f}, {pose.W[1]:7.1f}) mm"
        )

    def set_motor_angles(self, shoulder_angle_deg: float, elbow_motor_angle_deg: float) -> None:
        """Update the simulated mechanism from the two shoulder-mounted motors."""
        pose = solve_arm_pose(shoulder_angle_deg, elbow_motor_angle_deg)
        self.draw_pose(pose)
        self.fig.canvas.draw_idle()

    def update_from_sliders(self, _value: float) -> None:
        self.is_playing = False
        self.play_button.label.set_text("Play")
        self.set_motor_angles(self.shoulder_slider.val, self.elbow_slider.val)

    def toggle_play(self, _event) -> None:
        self.is_playing = not self.is_playing
        self.play_button.label.set_text("Pause" if self.is_playing else "Play")

    def animate(self, frame: int):
        if not self.is_playing:
            return self.artists()

        t = frame * 0.035 * ANIMATION_SPEED
        shoulder_mid = 0.5 * (SHOULDER_ANGLE_MIN_DEG + SHOULDER_ANGLE_MAX_DEG)
        shoulder_amp = 0.38 * (SHOULDER_ANGLE_MAX_DEG - SHOULDER_ANGLE_MIN_DEG)
        elbow_mid = 0.5 * (ELBOW_MOTOR_ANGLE_MIN_DEG + ELBOW_MOTOR_ANGLE_MAX_DEG)
        elbow_amp = 0.38 * (ELBOW_MOTOR_ANGLE_MAX_DEG - ELBOW_MOTOR_ANGLE_MIN_DEG)

        shoulder_angle = shoulder_mid + shoulder_amp * math.sin(t)
        elbow_angle = elbow_mid + elbow_amp * math.sin(t + math.pi / 2.0)

        self.shoulder_slider.eventson = False
        self.elbow_slider.eventson = False
        self.shoulder_slider.set_val(shoulder_angle)
        self.elbow_slider.set_val(elbow_angle)
        self.shoulder_slider.eventson = True
        self.elbow_slider.eventson = True

        self.set_motor_angles(shoulder_angle, elbow_angle)
        return self.artists()

    def artists(self):
        return (
            self.base_line,
            self.upper_arm_line,
            self.forearm_line,
            self.input_rocker_line,
            self.output_rocker_line,
            self.coupler_line,
            self.parallelogram_outline,
            self.joints_scatter,
        )

    def run(self) -> None:
        self.animation = FuncAnimation(
            self.fig,
            self.animate,
            interval=30,
            blit=False,
            cache_frame_data=False,
        )
        if matplotlib.get_backend().lower() == "webagg":
            print("Matplotlib will print the exact simulator URL below.")
            print("Press Ctrl+C in this terminal to stop it.")
        plt.show()


def main() -> None:
    app = ShoulderParallelogramArmApp()
    app.run()


if __name__ == "__main__":
    main()
