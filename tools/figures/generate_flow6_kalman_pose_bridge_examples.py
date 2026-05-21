#!/usr/bin/env python3
"""Generate Flow 6 Kalman-to-pose-kinematics bridge examples."""

from __future__ import annotations

from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("numpy is required to generate the Flow 6 pose bridge figure") from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("matplotlib is required to generate the Flow 6 pose bridge figure") from exc

from figure_style import finalize_figure


STEP_LENGTH = 0.12
STEP_HEADING = 0.075
STEP_COUNT = 24
START_POSE = np.array([0.0, 0.0, np.deg2rad(22.0)])
INITIAL_COVARIANCE = np.diag([0.025**2, 0.018**2, np.deg2rad(4.0) ** 2])
ODOMETRY_NOISE = np.diag([0.010**2, np.deg2rad(0.75) ** 2])
HEADING_SIGMA = np.deg2rad(8.0)
LOCAL_STEP_LENGTH = 0.85
LOCAL_HEADING = np.deg2rad(34.0)


def propagate_pose(pose: np.ndarray, step_length: float, heading_change: float) -> np.ndarray:
    x_position, y_position, heading = pose
    return np.array(
        [
            x_position + step_length * np.cos(heading),
            y_position + step_length * np.sin(heading),
            heading + heading_change,
        ]
    )


def prediction_jacobians(
    heading: float, step_length: float
) -> tuple[np.ndarray, np.ndarray]:
    state_jacobian = np.array(
        [
            [1.0, 0.0, -step_length * np.sin(heading)],
            [0.0, 1.0, step_length * np.cos(heading)],
            [0.0, 0.0, 1.0],
        ]
    )
    input_jacobian = np.array(
        [
            [np.cos(heading), 0.0],
            [np.sin(heading), 0.0],
            [0.0, 1.0],
        ]
    )
    return state_jacobian, input_jacobian


def propagate_covariance(
    covariance: np.ndarray, heading: float, step_length: float
) -> np.ndarray:
    state_jacobian, input_jacobian = prediction_jacobians(heading, step_length)
    return (
        state_jacobian @ covariance @ state_jacobian.T
        + input_jacobian @ ODOMETRY_NOISE @ input_jacobian.T
    )


def simulate_path() -> tuple[np.ndarray, np.ndarray]:
    poses = [START_POSE.copy()]
    covariances = [INITIAL_COVARIANCE.copy()]
    pose = START_POSE.copy()
    covariance = INITIAL_COVARIANCE.copy()

    for _ in range(STEP_COUNT):
        covariance = propagate_covariance(covariance, pose[2], STEP_LENGTH)
        pose = propagate_pose(pose, STEP_LENGTH, STEP_HEADING)
        poses.append(pose.copy())
        covariances.append(covariance.copy())

    return np.asarray(poses), np.asarray(covariances)


def covariance_ellipse_patch(
    mean: np.ndarray,
    covariance: np.ndarray,
    *,
    scale: float,
    edgecolor: str,
    linestyle: str,
    linewidth: float,
    label: str | None = None,
) -> Ellipse:
    values, vectors = np.linalg.eigh(covariance[:2, :2])
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    width, height = 2.0 * scale * np.sqrt(np.maximum(values, 0.0))
    return Ellipse(
        xy=mean[:2],
        width=width,
        height=height,
        angle=angle,
        fill=False,
        edgecolor=edgecolor,
        linestyle=linestyle,
        linewidth=linewidth,
        label=label,
    )


def draw_pose(axis, pose: np.ndarray, *, color: str, scale: float = 0.18) -> None:
    position = pose[:2]
    heading = pose[2]
    forward = np.array([np.cos(heading), np.sin(heading)])
    lateral = np.array([-np.sin(heading), np.cos(heading)])
    body = np.array(
        [
            position + scale * forward,
            position - 0.55 * scale * forward + 0.42 * scale * lateral,
            position - 0.55 * scale * forward - 0.42 * scale * lateral,
            position + scale * forward,
        ]
    )
    axis.plot(body[:, 0], body[:, 1], color=color, linewidth=1.5)
    axis.arrow(
        position[0],
        position[1],
        0.7 * scale * forward[0],
        0.7 * scale * forward[1],
        color=color,
        width=0.006,
        head_width=0.05,
        head_length=0.07,
        length_includes_head=True,
    )


def add_panel_label(axis, label: str) -> None:
    axis.text(
        0.02,
        0.97,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="#222222",
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "#cccccc",
            "alpha": 0.92,
        },
    )


def configure_xy_axis(axis) -> None:
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(r"$x$ [m]")
    axis.set_ylabel(r"$y$ [m]")
    axis.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.72)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "ja" / "figures" / "flow6_kalman_pose_bridge_examples.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 9,
            "font.family": ["Noto Sans CJK JP", "Noto Sans", "DejaVu Sans"],
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 7,
            "figure.dpi": 180,
            "savefig.dpi": 180,
            "axes.unicode_minus": False,
        }
    )

    poses, covariances = simulate_path()
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 3.35), constrained_layout=False)
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.19, top=0.985, wspace=0.32)
    arc_axis, covariance_axis, tangent_axis = axes

    arc_axis.plot(poses[:, 0], poses[:, 1], color="#1f77b4", linewidth=2.0, label="予測軌跡")
    for sigma_sign, linestyle, label in (
        (-1.0, "--", r"$\theta_0-\sigma_\theta$"),
        (1.0, ":", r"$\theta_0+\sigma_\theta$"),
    ):
        shifted_pose = START_POSE.copy()
        shifted_pose[2] += sigma_sign * HEADING_SIGMA
        shifted_poses = [shifted_pose.copy()]
        for _ in range(STEP_COUNT):
            shifted_pose = propagate_pose(shifted_pose, STEP_LENGTH, STEP_HEADING)
            shifted_poses.append(shifted_pose.copy())
        shifted_poses = np.asarray(shifted_poses)
        arc_axis.plot(
            shifted_poses[:, 0],
            shifted_poses[:, 1],
            color="#777777",
            linestyle=linestyle,
            linewidth=1.4,
            label=label,
        )
    draw_pose(arc_axis, poses[0], color="#222222")
    draw_pose(arc_axis, poses[-1], color="#d62728")
    arc_axis.text(0.08, -0.18, "start", fontsize=8, color="#222222")
    arc_axis.text(1.30, 1.98, "predicted pose", fontsize=8, color="#d62728")
    arc_axis.legend(loc="upper left")
    arc_axis.set_xlim(-0.28, 2.35)
    arc_axis.set_ylim(-0.35, 2.38)
    configure_xy_axis(arc_axis)
    add_panel_label(arc_axis, "A")

    covariance_axis.plot(
        poses[:, 0],
        poses[:, 1],
        color="#1f77b4",
        linewidth=1.6,
        label="平均予測",
    )
    ellipse_indices = (0, 8, 16, 24)
    ellipse_colors = ("#555555", "#2ca02c", "#ff7f0e", "#d62728")
    for index, color in zip(ellipse_indices, ellipse_colors):
        patch = covariance_ellipse_patch(
            poses[index],
            covariances[index],
            scale=2.0,
            edgecolor=color,
            linestyle="-" if index == ellipse_indices[-1] else "--",
            linewidth=1.5,
            label=rf"$k={index}$",
        )
        covariance_axis.add_patch(patch)
        draw_pose(covariance_axis, poses[index], color=color, scale=0.12)
    covariance_axis.text(
        0.04,
        0.05,
        r"$2\sigma$ covariance ellipses",
        transform=covariance_axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": "#cccccc",
            "alpha": 0.92,
        },
    )
    covariance_axis.legend(loc="upper left")
    covariance_axis.set_xlim(-0.28, 2.35)
    covariance_axis.set_ylim(-0.35, 2.38)
    configure_xy_axis(covariance_axis)
    add_panel_label(covariance_axis, "B")

    base = np.array([0.0, 0.0])
    forward = np.array([np.cos(LOCAL_HEADING), np.sin(LOCAL_HEADING)])
    lateral = np.array([-np.sin(LOCAL_HEADING), np.cos(LOCAL_HEADING)])
    nominal_end = base + LOCAL_STEP_LENGTH * forward
    minus_end = base + LOCAL_STEP_LENGTH * np.array(
        [np.cos(LOCAL_HEADING - HEADING_SIGMA), np.sin(LOCAL_HEADING - HEADING_SIGMA)]
    )
    plus_end = base + LOCAL_STEP_LENGTH * np.array(
        [np.cos(LOCAL_HEADING + HEADING_SIGMA), np.sin(LOCAL_HEADING + HEADING_SIGMA)]
    )
    tangent_axis.arrow(
        base[0],
        base[1],
        nominal_end[0],
        nominal_end[1],
        color="#1f77b4",
        width=0.006,
        head_width=0.045,
        head_length=0.06,
        length_includes_head=True,
        label=r"$s[\cos\theta,\sin\theta]^\mathsf{T}$",
    )
    tangent_axis.plot(
        [base[0], minus_end[0]],
        [base[1], minus_end[1]],
        color="#777777",
        linestyle="--",
        linewidth=1.3,
    )
    tangent_axis.plot(
        [base[0], plus_end[0]],
        [base[1], plus_end[1]],
        color="#777777",
        linestyle=":",
        linewidth=1.3,
    )
    tangent_axis.plot(
        [minus_end[0], plus_end[0]],
        [minus_end[1], plus_end[1]],
        color="#d62728",
        linewidth=2.0,
        label=r"$s\,\delta\theta[-\sin\theta,\cos\theta]^\mathsf{T}$",
    )
    tangent_axis.arrow(
        nominal_end[0],
        nominal_end[1],
        0.32 * lateral[0],
        0.32 * lateral[1],
        color="#d62728",
        width=0.004,
        head_width=0.04,
        head_length=0.055,
        length_includes_head=True,
    )
    tangent_axis.scatter(
        [base[0], nominal_end[0], minus_end[0], plus_end[0]],
        [base[1], nominal_end[1], minus_end[1], plus_end[1]],
        color=["#222222", "#1f77b4", "#777777", "#777777"],
        s=[24, 28, 20, 20],
        zorder=4,
    )
    tangent_axis.text(-0.02, -0.08, r"$p_k$", fontsize=8, ha="right")
    tangent_axis.text(
        nominal_end[0] + 0.05,
        nominal_end[1] - 0.02,
        r"$p^-_{k+1}$",
        fontsize=8,
        color="#1f77b4",
    )
    tangent_axis.text(
        0.02,
        0.93,
        r"$F_\theta=[-s\sin\theta,\ s\cos\theta]^\mathsf{T}$",
        transform=tangent_axis.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": "#cccccc",
            "alpha": 0.92,
        },
    )
    tangent_axis.legend(loc="lower right")
    tangent_axis.set_xlim(-0.20, 1.10)
    tangent_axis.set_ylim(-0.20, 0.95)
    configure_xy_axis(tangent_axis)
    add_panel_label(tangent_axis, "C")

    finalize_figure(fig, output_path, layout="none")
    print(output_path)


if __name__ == "__main__":
    main()
