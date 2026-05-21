#!/usr/bin/env python3
"""Generate Flow 6 IMU axes and gyro bias examples."""

from __future__ import annotations

from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("numpy is required to generate the IMU gyro figure") from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc, Circle, Polygon
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("matplotlib is required to generate the IMU gyro figure") from exc

from figure_style import finalize_figure


HEADING_DEG = 32.0
SAMPLE_PERIOD = 0.01
SIMULATION_TIME = 60.0
BIAS_DEG_PER_S = (0.1, 0.5, 1.5)


def rotation_matrix(angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.array([[c, -s], [s, c]])


def rotate_translate(points: np.ndarray, angle_rad: float, center: np.ndarray) -> np.ndarray:
    return points @ rotation_matrix(angle_rad).T + center


def add_arrow(axis, start: np.ndarray, vector: np.ndarray, color: str, label: str) -> None:
    end = start + vector
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "linewidth": 2.1,
            "color": color,
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )
    label_pos = start + 1.08 * vector
    axis.text(
        label_pos[0],
        label_pos[1],
        label,
        color=color,
        fontsize=11,
        fontweight="bold",
        ha="center",
        va="center",
    )


def draw_imu_axes(axis) -> None:
    heading = np.deg2rad(HEADING_DEG)
    center = np.array([0.0, 0.0])
    body_color = "#4c78a8"
    imu_color = "#e45756"
    inertial_color = "#444444"
    body_x_color = "#1f77b4"
    body_y_color = "#2ca02c"

    chassis = np.array(
        [
            [-1.20, -0.55],
            [1.20, -0.55],
            [1.20, 0.55],
            [-1.20, 0.55],
        ]
    )
    wheel_left = np.array(
        [
            [-0.75, 0.58],
            [0.75, 0.58],
            [0.75, 0.78],
            [-0.75, 0.78],
        ]
    )
    wheel_right = np.array(
        [
            [-0.75, -0.78],
            [0.75, -0.78],
            [0.75, -0.58],
            [-0.75, -0.58],
        ]
    )

    axis.add_patch(
        Polygon(
            rotate_translate(chassis, heading, center),
            closed=True,
            facecolor="#d8e7f5",
            edgecolor=body_color,
            linewidth=1.6,
        )
    )
    for wheel in (wheel_left, wheel_right):
        axis.add_patch(
            Polygon(
                rotate_translate(wheel, heading, center),
                closed=True,
                facecolor="#333333",
                edgecolor="#111111",
                linewidth=1.1,
            )
        )

    nose = rotate_translate(np.array([[1.35, 0.0]]), heading, center)[0]
    axis.plot([center[0], nose[0]], [center[1], nose[1]], color=body_color, linewidth=2.0)

    imu_square = np.array(
        [
            [-0.20, -0.20],
            [0.20, -0.20],
            [0.20, 0.20],
            [-0.20, 0.20],
        ]
    )
    axis.add_patch(
        Polygon(
            rotate_translate(imu_square, heading, center),
            closed=True,
            facecolor="#ffe7e5",
            edgecolor=imu_color,
            linewidth=1.4,
            zorder=5,
        )
    )
    axis.text(center[0], center[1], "IMU", color=imu_color, ha="center", va="center", fontsize=9)

    body_x = rotation_matrix(heading) @ np.array([1.05, 0.0])
    body_y = rotation_matrix(heading) @ np.array([0.0, 0.88])
    add_arrow(axis, center, body_x, body_x_color, r"$x_B$")
    add_arrow(axis, center, body_y, body_y_color, r"$y_B$")
    axis.add_patch(Circle(center, 0.09, facecolor="white", edgecolor="#111111", linewidth=1.2, zorder=8))
    axis.text(center[0] - 0.10, center[1] - 0.22, r"$z_B$ up", ha="center", va="center", fontsize=9)

    origin = np.array([-2.05, -1.45])
    add_arrow(axis, origin, np.array([0.90, 0.0]), inertial_color, r"$x_I$")
    add_arrow(axis, origin, np.array([0.0, 0.90]), inertial_color, r"$y_I$")
    axis.add_patch(Arc(origin, 0.85, 0.85, theta1=0.0, theta2=HEADING_DEG, color="#9b59b6", linewidth=1.5))
    axis.text(origin[0] + 0.55, origin[1] + 0.20, r"$\psi$", color="#9b59b6", fontsize=11)

    axis.text(
        0.03,
        0.96,
        "A  IMU axes fixed to body frame",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.95, "pad": 2.5},
    )
    axis.text(
        0.03,
        0.08,
        r"gyro measures $\omega_z^B$ about $z_B$",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.9, "pad": 2.0},
    )
    axis.set_xlim(-2.35, 2.35)
    axis.set_ylim(-1.75, 1.65)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Inertial x [m]")
    axis.set_ylabel("Inertial y [m]")
    axis.grid(True, color="#dddddd", linewidth=0.7, alpha=0.7)


def draw_bias_trace(axis) -> None:
    time = np.arange(0.0, SIMULATION_TIME + SAMPLE_PERIOD, SAMPLE_PERIOD)
    colors = ("#1f77b4", "#ff7f0e", "#d62728")
    for bias, color in zip(BIAS_DEG_PER_S, colors):
        error_deg = bias * time
        axis.plot(time, error_deg, color=color, linewidth=2.0, label=rf"$b_g={bias:.1f}^\circ/s$")
        axis.plot(time[-1], error_deg[-1], marker="o", color=color, markersize=4.5)
        axis.text(
            time[-1] + 1.2,
            error_deg[-1],
            f"{error_deg[-1]:.0f} deg",
            color=color,
            va="center",
            fontsize=9,
        )

    axis.axhline(0.0, color="#777777", linewidth=0.8)
    axis.set_xlim(0.0, SIMULATION_TIME + 12.0)
    axis.set_ylim(-3.0, 96.0)
    axis.set_xlabel("Time t [s]")
    axis.set_ylabel(r"Heading error $e_\psi$ [deg]")
    axis.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.75)
    axis.legend(loc="upper left", frameon=True)
    axis.text(
        0.50,
        1.035,
        "B  gyro-bias drift",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        clip_on=False,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.95, "pad": 2.5},
    )
    axis.text(
        0.97,
        0.18,
        rf"$T_s={SAMPLE_PERIOD:.2f}\,s$",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.9, "pad": 2.0},
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "ja" / "figures" / "flow6_imu_gyro_examples.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 9,
            "font.family": ["DejaVu Sans"],
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 180,
            "savefig.dpi": 180,
            "axes.unicode_minus": False,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.5), constrained_layout=False)
    draw_imu_axes(axes[0])
    draw_bias_trace(axes[1])
    finalize_figure(fig, output_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
