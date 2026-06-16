#!/usr/bin/env python3
r"""Inverse-kinematics UI for the shoulder-mounted parallelogram arm.

This file takes the current wrist point W as the input and solves the two
base-mounted motor angles:

    shoulder motor at A -> upper arm A-E
    elbow motor at A    -> parallelogram rocker A-P

The parallelogram itself does not need a separate IK solve. Once the target
wrist point gives us the shoulder angle and the forearm absolute angle, the
elbow motor angle is just the forearm angle corrected by the rigid offset
between the forearm E-W and output rocker E-Q.

The standalone function to reuse later is:

    solve_inverse_kinematics(target_xy, branch="elbow_down")

The UI wrapper method to replace later with coordinate commands is:

    app.set_target_xy(x_mm, y_mm)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fourbar_leg_sim import (
    ELBOW_MOTOR_ANGLE_MAX_DEG,
    ELBOW_MOTOR_ANGLE_MIN_DEG,
    FOREARM_LENGTH,
    FOREARM_TO_ROCKER_ANGLE_DEG,
    PARALLELOGRAM_ROCKER_LENGTH,
    SHOULDER_ANGLE_MAX_DEG,
    SHOULDER_ANGLE_MIN_DEG,
    SHOULDER_PIVOT,
    START_ELBOW_MOTOR_ANGLE_DEG,
    START_SHOULDER_ANGLE_DEG,
    UPPER_ARM_LENGTH,
    ArmPose,
    solve_arm_pose,
)

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Button, Slider


# ---------------------------------------------------------------------------
# User-configurable IK UI settings
# ---------------------------------------------------------------------------

# "elbow_down" reproduces the current default pose. The other valid option is
# "elbow_up". A reachable target normally has one solution in each branch.
DEFAULT_IK_BRANCH = "elbow_down"

# Leave these as None to start from the wrist position produced by the forward
# simulator's START_* motor angles.
START_TARGET_X_MM: float | None = None
START_TARGET_Y_MM: float | None = None

# Slider limits are based on the maximum reach plus this margin.
TARGET_SLIDER_MARGIN_MM = 20.0

# How many accepted target positions to keep in the faint trace.
TRACE_MAX_POINTS = 500


# ---------------------------------------------------------------------------
# IK data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IKResult:
    """Inverse-kinematics result for one wrist target."""

    target: np.ndarray
    branch: str
    reachable: bool
    within_angle_limits: bool
    shoulder_angle_deg: float | None
    elbow_motor_angle_deg: float | None
    pose: ArmPose | None
    message: str


# ---------------------------------------------------------------------------
# Inverse kinematics
# ---------------------------------------------------------------------------


def normalize_degrees(angle_deg: float) -> float:
    """Normalize an angle to [-180, 180) degrees."""
    return (angle_deg + 180.0) % 360.0 - 180.0


def motor_angles_within_limits(shoulder_angle_deg: float, elbow_motor_angle_deg: float) -> bool:
    """Check the solved angles against the configured motor ranges."""
    shoulder_ok = SHOULDER_ANGLE_MIN_DEG <= shoulder_angle_deg <= SHOULDER_ANGLE_MAX_DEG
    elbow_ok = ELBOW_MOTOR_ANGLE_MIN_DEG <= elbow_motor_angle_deg <= ELBOW_MOTOR_ANGLE_MAX_DEG
    return shoulder_ok and elbow_ok


def solve_inverse_kinematics(target_xy: np.ndarray, branch: str = DEFAULT_IK_BRANCH) -> IKResult:
    """Solve shoulder and base-mounted elbow motor angles for wrist point W.

    The wrist kinematics reduce to a planar two-link triangle:

        A ---- upper arm ---- E ---- forearm ---- W

    where A is fixed and W is the requested target. The parallelogram
    A-P-Q-E is reconstructed afterward by solve_arm_pose().

    branch controls which of the two possible elbow configurations is used:
      - "elbow_down": negative signed elbow bend
      - "elbow_up": positive signed elbow bend
    """
    target = np.asarray(target_xy, dtype=float)
    if target.shape != (2,):
        raise ValueError("target_xy must contain exactly two values: x_mm, y_mm")
    if branch not in {"elbow_down", "elbow_up"}:
        raise ValueError('branch must be "elbow_down" or "elbow_up"')

    base_to_target = target - SHOULDER_PIVOT
    x = float(base_to_target[0])
    y = float(base_to_target[1])
    distance_sq = x * x + y * y
    distance = math.sqrt(distance_sq)

    l1 = UPPER_ARM_LENGTH
    l2 = FOREARM_LENGTH
    outer_reach = l1 + l2
    inner_reach = abs(l1 - l2)
    reach_tolerance = 1e-9

    if distance > outer_reach + reach_tolerance:
        return IKResult(
            target=target,
            branch=branch,
            reachable=False,
            within_angle_limits=False,
            shoulder_angle_deg=None,
            elbow_motor_angle_deg=None,
            pose=None,
            message=f"Target is too far. Max reach is {outer_reach:.1f} mm.",
        )
    if distance < inner_reach - reach_tolerance:
        return IKResult(
            target=target,
            branch=branch,
            reachable=False,
            within_angle_limits=False,
            shoulder_angle_deg=None,
            elbow_motor_angle_deg=None,
            pose=None,
            message=f"Target is too close. Min reach is {inner_reach:.1f} mm.",
        )

    # Law of cosines for the signed angle between upper arm A-E and forearm E-W.
    # Clamping only protects against tiny floating-point overshoot at the edge
    # of the workspace.
    cos_elbow = (distance_sq - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    cos_elbow = min(1.0, max(-1.0, cos_elbow))
    unsigned_elbow_rad = math.acos(cos_elbow)
    signed_elbow_rad = -unsigned_elbow_rad if branch == "elbow_down" else unsigned_elbow_rad

    # Standard two-link IK. The first atan2 points from A to W; the second
    # removes the triangle angle between A-W and the upper arm A-E.
    target_angle_rad = math.atan2(y, x)
    triangle_angle_rad = math.atan2(
        l2 * math.sin(signed_elbow_rad),
        l1 + l2 * math.cos(signed_elbow_rad),
    )
    shoulder_rad = target_angle_rad - triangle_angle_rad
    forearm_abs_rad = shoulder_rad + signed_elbow_rad

    shoulder_angle_deg = normalize_degrees(math.degrees(shoulder_rad))

    # In the forward model:
    #   forearm_dir = rot(FOREARM_TO_ROCKER_ANGLE_DEG) @ rocker_dir
    # so:
    #   elbow_motor_angle = forearm_absolute_angle - fixed_offset
    elbow_motor_angle_deg = normalize_degrees(
        math.degrees(forearm_abs_rad) - FOREARM_TO_ROCKER_ANGLE_DEG
    )

    pose = solve_arm_pose(shoulder_angle_deg, elbow_motor_angle_deg)
    within_limits = motor_angles_within_limits(shoulder_angle_deg, elbow_motor_angle_deg)
    message = "Target solved."
    if not within_limits:
        message = "Target solved, but motor angles are outside the configured limits."

    return IKResult(
        target=target,
        branch=branch,
        reachable=True,
        within_angle_limits=within_limits,
        shoulder_angle_deg=shoulder_angle_deg,
        elbow_motor_angle_deg=elbow_motor_angle_deg,
        pose=pose,
        message=message,
    )


# ---------------------------------------------------------------------------
# Visualization app
# ---------------------------------------------------------------------------


class ParallelogramArmIKApp:
    """Matplotlib UI that accepts wrist X/Y and displays the solved mechanism."""

    def __init__(self) -> None:
        self.branch = DEFAULT_IK_BRANCH
        self.trace_points: list[np.ndarray] = []

        initial_pose = solve_arm_pose(START_SHOULDER_ANGLE_DEG, START_ELBOW_MOTOR_ANGLE_DEG)
        initial_target = initial_pose.W.copy()
        if START_TARGET_X_MM is not None:
            initial_target[0] = START_TARGET_X_MM
        if START_TARGET_Y_MM is not None:
            initial_target[1] = START_TARGET_Y_MM

        self.last_valid_pose = initial_pose

        self.fig, self.ax = plt.subplots(figsize=(8.5, 7.5))
        plt.subplots_adjust(left=0.08, right=0.98, bottom=0.26, top=0.94)

        self.ax.set_title("Inverse kinematics: shoulder parallelogram arm")
        self.ax.set_xlabel("X (mm)")
        self.ax.set_ylabel("Y (mm)")
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True, color="0.90", linewidth=0.8)

        reach = UPPER_ARM_LENGTH + FOREARM_LENGTH
        slider_extent = reach + TARGET_SLIDER_MARGIN_MM
        extra = PARALLELOGRAM_ROCKER_LENGTH + 35.0
        self.ax.set_xlim(-slider_extent - extra, slider_extent + extra)
        self.ax.set_ylim(-slider_extent - extra, slider_extent + extra)

        outer_workspace = Circle(
            SHOULDER_PIVOT,
            reach,
            fill=False,
            color="0.80",
            linestyle="--",
            linewidth=1.0,
            label="max wrist reach",
        )
        inner_reach = abs(UPPER_ARM_LENGTH - FOREARM_LENGTH)
        inner_workspace = Circle(
            SHOULDER_PIVOT,
            inner_reach,
            fill=False,
            color="0.86",
            linestyle=":",
            linewidth=1.0,
            label="min wrist reach",
        )
        self.ax.add_patch(outer_workspace)
        if inner_reach > 0.0:
            self.ax.add_patch(inner_workspace)

        self.base_line, = self.ax.plot([], [], color="0.25", linewidth=7, solid_capstyle="round", label="fixed base")
        self.upper_arm_line, = self.ax.plot([], [], color="#111827", linewidth=5, label="upper arm A-E")
        self.forearm_line, = self.ax.plot([], [], color="#111827", linewidth=5, label="forearm E-W")
        self.input_rocker_line, = self.ax.plot([], [], color="#2563eb", linewidth=4, label="elbow motor rocker A-P")
        self.output_rocker_line, = self.ax.plot([], [], color="#f97316", linewidth=4, label="elbow output rocker E-Q")
        self.coupler_line, = self.ax.plot([], [], color="#16a34a", linewidth=4, label="parallel coupler P-Q")
        self.parallelogram_outline, = self.ax.plot([], [], color="#64748b", linewidth=1.4, linestyle="--", label="parallelogram loop")
        self.trace_line, = self.ax.plot([], [], color="#ef4444", linewidth=1.2, alpha=0.45, label="accepted target trace")
        self.target_marker, = self.ax.plot([], [], marker="x", color="#dc2626", markersize=10, markeredgewidth=2.0, linestyle="None", label="requested W")

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

        x_slider_ax = self.fig.add_axes([0.20, 0.15, 0.67, 0.035])
        y_slider_ax = self.fig.add_axes([0.20, 0.095, 0.67, 0.035])
        branch_button_ax = self.fig.add_axes([0.20, 0.03, 0.20, 0.04])
        clear_button_ax = self.fig.add_axes([0.43, 0.03, 0.16, 0.04])

        self.x_slider = Slider(
            ax=x_slider_ax,
            label="target W X (mm)",
            valmin=-slider_extent,
            valmax=slider_extent,
            valinit=float(initial_target[0]),
        )
        self.y_slider = Slider(
            ax=y_slider_ax,
            label="target W Y (mm)",
            valmin=-slider_extent,
            valmax=slider_extent,
            valinit=float(initial_target[1]),
        )
        self.branch_button = Button(branch_button_ax, self.branch_label())
        self.clear_button = Button(clear_button_ax, "Clear trace")

        self.x_slider.on_changed(self.update_from_sliders)
        self.y_slider.on_changed(self.update_from_sliders)
        self.branch_button.on_clicked(self.toggle_branch)
        self.clear_button.on_clicked(self.clear_trace)

        self.set_target_xy(float(initial_target[0]), float(initial_target[1]))

    @staticmethod
    def set_line(line, *points: np.ndarray, closed: bool = False) -> None:
        arr = np.array(points)
        if closed:
            arr = np.vstack([arr, arr[0]])
        line.set_data(arr[:, 0], arr[:, 1])

    def branch_label(self) -> str:
        return f"Branch: {self.branch.replace('_', ' ')}"

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
        """Draw an already-solved forward pose."""
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

    def draw_trace(self) -> None:
        if not self.trace_points:
            self.trace_line.set_data([], [])
            return
        trace = np.vstack(self.trace_points)
        self.trace_line.set_data(trace[:, 0], trace[:, 1])

    def set_target_xy(self, x_mm: float, y_mm: float) -> IKResult:
        """Update the mechanism from a desired wrist coordinate."""
        target = np.array([x_mm, y_mm], dtype=float)
        result = solve_inverse_kinematics(target, self.branch)

        self.target_marker.set_data([x_mm], [y_mm])

        pose_to_draw = result.pose if result.pose is not None else self.last_valid_pose
        self.draw_pose(pose_to_draw)

        if result.pose is not None:
            self.last_valid_pose = result.pose
            self.trace_points.append(result.pose.W.copy())
            if len(self.trace_points) > TRACE_MAX_POINTS:
                self.trace_points = self.trace_points[-TRACE_MAX_POINTS:]
        self.draw_trace()

        self.update_readout(result)
        self.fig.canvas.draw_idle()
        return result

    def update_readout(self, result: IKResult) -> None:
        x_mm = float(result.target[0])
        y_mm = float(result.target[1])
        if result.pose is None:
            self.coord_text.set_text(
                f"Requested W: ({x_mm:7.1f}, {y_mm:7.1f}) mm\n"
                f"Branch:      {result.branch.replace('_', ' ')}\n"
                f"{result.message}"
            )
            self.coord_text.get_bbox_patch().set_edgecolor("#ef4444")
            return

        limit_text = "inside limits" if result.within_angle_limits else "outside angle limits"
        self.coord_text.set_text(
            f"Requested W: ({x_mm:7.1f}, {y_mm:7.1f}) mm\n"
            f"Solved W:    ({result.pose.W[0]:7.1f}, {result.pose.W[1]:7.1f}) mm\n"
            f"Shoulder:    {result.shoulder_angle_deg:8.2f} deg\n"
            f"Elbow motor: {result.elbow_motor_angle_deg:8.2f} deg\n"
            f"Branch:      {result.branch.replace('_', ' ')}\n"
            f"{limit_text}"
        )
        edgecolor = "#22c55e" if result.within_angle_limits else "#f97316"
        self.coord_text.get_bbox_patch().set_edgecolor(edgecolor)

    def update_from_sliders(self, _value: float) -> None:
        self.set_target_xy(self.x_slider.val, self.y_slider.val)

    def toggle_branch(self, _event) -> None:
        self.branch = "elbow_up" if self.branch == "elbow_down" else "elbow_down"
        self.branch_button.label.set_text(self.branch_label())
        self.set_target_xy(self.x_slider.val, self.y_slider.val)

    def clear_trace(self, _event) -> None:
        self.trace_points.clear()
        self.draw_trace()
        self.fig.canvas.draw_idle()

    def run(self) -> None:
        if matplotlib.get_backend().lower() == "webagg":
            print("Matplotlib will print the exact IK simulator URL below.")
            print("Press Ctrl+C in this terminal to stop it.")
        plt.show()


def main() -> None:
    app = ParallelogramArmIKApp()
    app.run()


if __name__ == "__main__":
    main()
