#!/usr/bin/env python3
"""Generate differential-drive odometry drift and encoder pulse examples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("numpy is required to generate the odometry figure") from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("matplotlib is required to generate the odometry figure") from exc

from figure_style import finalize_figure


WHEEL_RADIUS = 0.032  # m
WHEEL_BASE = 0.280  # m
COUNTS_PER_REV = 2048
DT = 0.02  # s
T_FINAL = 10.0  # s
SLIP_START = 3.8  # s
SLIP_END = 6.7  # s
RIGHT_SLIP_SCALE = 0.86
LEFT_SCALE_BIAS = 1.004
RIGHT_SCALE_BIAS = 0.997
SAMPLE_DT = 0.10  # s
SAMPLE_LEFT_COUNTS = 360
SAMPLE_RIGHT_COUNTS = 404


@dataclass
class Simulation:
    time: np.ndarray
    true_pose: np.ndarray
    odom_pose: np.ndarray
    left_counts: np.ndarray
    right_counts: np.ndarray


def commanded_body_velocity(time: float) -> tuple[float, float]:
    speed = 0.34 + 0.06 * np.sin(0.45 * time)
    turn_rate = 0.20 * np.sin(0.85 * time) + 0.10 * np.sin(0.18 * time + 0.4)
    return speed, turn_rate


def integrate_pose(pose: np.ndarray, left_distance: float, right_distance: float) -> np.ndarray:
    distance = 0.5 * (left_distance + right_distance)
    heading_change = (right_distance - left_distance) / WHEEL_BASE
    mid_heading = pose[2] + 0.5 * heading_change
    return np.array(
        [
            pose[0] + distance * np.cos(mid_heading),
            pose[1] + distance * np.sin(mid_heading),
            pose[2] + heading_change,
        ]
    )


def simulate() -> Simulation:
    time = np.arange(0.0, T_FINAL + DT, DT)
    true_pose = np.zeros((time.size, 3))
    odom_pose = np.zeros((time.size, 3))
    left_count_cumulative = np.zeros(time.size, dtype=int)
    right_count_cumulative = np.zeros(time.size, dtype=int)

    ideal_left_rotation = 0.0
    ideal_right_rotation = 0.0

    for index, current_time in enumerate(time[:-1]):
        speed, turn_rate = commanded_body_velocity(float(current_time))
        left_rate = speed - 0.5 * WHEEL_BASE * turn_rate
        right_rate = speed + 0.5 * WHEEL_BASE * turn_rate
        left_command = left_rate * DT
        right_command = right_rate * DT

        slip_scale = RIGHT_SLIP_SCALE if SLIP_START <= current_time <= SLIP_END else 1.0
        true_left_distance = LEFT_SCALE_BIAS * left_command
        true_right_distance = RIGHT_SCALE_BIAS * slip_scale * right_command
        true_pose[index + 1] = integrate_pose(
            true_pose[index], true_left_distance, true_right_distance
        )

        ideal_left_rotation += left_command / WHEEL_RADIUS
        ideal_right_rotation += right_command / WHEEL_RADIUS
        left_count_cumulative[index + 1] = int(
            np.rint(ideal_left_rotation * COUNTS_PER_REV / (2.0 * np.pi))
        )
        right_count_cumulative[index + 1] = int(
            np.rint(ideal_right_rotation * COUNTS_PER_REV / (2.0 * np.pi))
        )

        left_delta = left_count_cumulative[index + 1] - left_count_cumulative[index]
        right_delta = right_count_cumulative[index + 1] - right_count_cumulative[index]
        meters_per_count = 2.0 * np.pi * WHEEL_RADIUS / COUNTS_PER_REV
        odom_left_distance = left_delta * meters_per_count
        odom_right_distance = right_delta * meters_per_count
        odom_pose[index + 1] = integrate_pose(
            odom_pose[index], odom_left_distance, odom_right_distance
        )

    return Simulation(
        time=time,
        true_pose=true_pose,
        odom_pose=odom_pose,
        left_counts=left_count_cumulative,
        right_counts=right_count_cumulative,
    )


def draw_robot_marker(axis, pose: np.ndarray, color: str, label: str) -> None:
    x, y, heading = pose
    body = Circle((x, y), 0.055, facecolor="white", edgecolor=color, linewidth=1.7, zorder=4)
    axis.add_patch(body)
    axis.arrow(
        x,
        y,
        0.105 * np.cos(heading),
        0.105 * np.sin(heading),
        color=color,
        width=0.006,
        head_width=0.035,
        length_includes_head=True,
        zorder=5,
    )
    axis.text(x, y - 0.105, label, ha="center", va="top", color=color, fontsize=8)


def draw_path_panel(axis, simulation: Simulation) -> None:
    true_pose = simulation.true_pose
    odom_pose = simulation.odom_pose
    axis.plot(true_pose[:, 0], true_pose[:, 1], color="#1f77b4", linewidth=2.2, label="true path")
    axis.plot(
        odom_pose[:, 0],
        odom_pose[:, 1],
        color="#d62728",
        linestyle="--",
        linewidth=2.0,
        label="encoder odometry",
    )
    draw_robot_marker(axis, true_pose[-1], "#1f77b4", "true")
    draw_robot_marker(axis, odom_pose[-1], "#d62728", "odom")
    axis.axvspan(
        true_pose[np.searchsorted(simulation.time, SLIP_START), 0],
        true_pose[np.searchsorted(simulation.time, SLIP_END), 0],
        color="#ffe8cc",
        alpha=0.8,
        label="right-wheel slip interval",
    )
    axis.set_xlabel(r"$x$ [m]")
    axis.set_ylabel(r"$y$ [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)
    axis.legend(loc="upper left", fontsize=8, frameon=True)
    axis.text(
        0.98,
        0.04,
        "encoders report wheel rotation;\nslip changes ground motion",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.95},
    )


def draw_pulse_panel(axis) -> None:
    meters_per_count = 2.0 * np.pi * WHEEL_RADIUS / COUNTS_PER_REV
    left_distance = SAMPLE_LEFT_COUNTS * meters_per_count
    right_distance = SAMPLE_RIGHT_COUNTS * meters_per_count
    average_distance = 0.5 * (left_distance + right_distance)
    heading_change = (right_distance - left_distance) / WHEEL_BASE

    pulse_count = 18
    sketch_window = 0.035
    pulse_time = np.linspace(0.0, sketch_window, pulse_count * 2, endpoint=False)
    left_train = np.zeros_like(pulse_time)
    right_train = np.zeros_like(pulse_time)
    left_train[1::2] = 1.0
    right_train[1::2] = 1.0
    right_train_time = pulse_time * SAMPLE_LEFT_COUNTS / SAMPLE_RIGHT_COUNTS

    axis.step(pulse_time, left_train + 1.45, where="post", color="#1f77b4", linewidth=1.4)
    axis.step(
        np.clip(right_train_time, 0.0, SAMPLE_DT),
        right_train + 0.15,
        where="post",
        color="#d62728",
        linewidth=1.4,
    )
    label_box = {"facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.95}
    axis.text(
        0.002,
        2.62,
        rf"left: $\Delta n_L={SAMPLE_LEFT_COUNTS}$",
        color="#1f77b4",
        bbox=label_box,
    )
    axis.text(
        0.002,
        1.06,
        rf"right: $\Delta n_R={SAMPLE_RIGHT_COUNTS}$",
        color="#d62728",
        bbox=label_box,
    )
    axis.text(
        0.55,
        0.56,
        (
            rf"$2\pi r/N={meters_per_count*1000.0:.3f}$ mm/count" "\n"
            rf"$\Delta s_L={left_distance:.4f}$ m" "\n"
            rf"$\Delta s_R={right_distance:.4f}$ m" "\n"
            rf"$\Delta s={average_distance:.4f}$ m, "
            rf"$\Delta\theta={heading_change:.4f}$ rad"
        ),
        transform=axis.transAxes,
        ha="left",
        va="center",
        fontsize=8.4,
        bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.95},
    )
    axis.text(
        0.002,
        0.02,
        "first pulses drawn; total counts are annotated",
        ha="left",
        va="bottom",
        fontsize=8.0,
        color="#555555",
    )
    axis.set_xlim(0.0, SAMPLE_DT)
    axis.set_ylim(-0.1, 2.75)
    axis.set_yticks([])
    axis.set_xlabel(r"sample time [s]")
    axis.set_ylabel("pulse level [-]")
    axis.grid(True, axis="x", color="#d0d0d0", linewidth=0.7, alpha=0.7)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "ja" / "figures" / "flow6_odometry_examples.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 180,
            "savefig.dpi": 180,
            "axes.unicode_minus": False,
        }
    )

    simulation = simulate()
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.3), constrained_layout=False)
    draw_path_panel(axes[0], simulation)
    draw_pulse_panel(axes[1])
    finalize_figure(figure, output_path)
    plt.close(figure)
    print(output_path)


if __name__ == "__main__":
    main()
