#!/usr/bin/env python3
"""Generate Flow 6 encoder slip examples for the Japanese text."""

from __future__ import annotations

from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("numpy is required to generate the encoder slip figure") from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("matplotlib is required to generate the encoder slip figure") from exc

from figure_style import finalize_figure


TRACK_WIDTH = 0.32
WHEEL_RADIUS_TRUE = 0.050
WHEEL_RADIUS_NOMINAL = 0.050
WHEEL_RADIUS_LEFT_BIASED = 0.05075
WHEEL_RADIUS_RIGHT_BIASED = 0.0495
PULSES_PER_REVOLUTION = 360
DT = 0.02
TIME = np.arange(0.0, 8.0 + DT, DT)
SLIP_START = 3.20
SLIP_END = 3.80
LEFT_SLIP_GROUND_RATIO = 0.60


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def commanded_body_velocity(time: float) -> tuple[float, float]:
    center_speed = 0.34 + 0.035 * np.sin(0.55 * time)
    yaw_rate = 0.10 + 0.22 * np.sin(0.75 * time)
    return center_speed, yaw_rate


def integrate_pose(pose: np.ndarray, left_distance: float, right_distance: float) -> np.ndarray:
    x_position, y_position, heading = pose
    center_distance = 0.5 * (left_distance + right_distance)
    heading_change = (right_distance - left_distance) / TRACK_WIDTH
    midpoint_heading = heading + 0.5 * heading_change
    return np.array(
        [
            x_position + center_distance * np.cos(midpoint_heading),
            y_position + center_distance * np.sin(midpoint_heading),
            float(wrap_angle(heading + heading_change)),
        ]
    )


def simulate() -> dict[str, np.ndarray | float]:
    true_pose = np.zeros(3)
    encoder_pose = np.zeros(3)
    biased_pose = np.zeros(3)

    true_path = [true_pose.copy()]
    encoder_path = [encoder_pose.copy()]
    biased_path = [biased_pose.copy()]
    position_error_encoder = [0.0]
    position_error_biased = [0.0]
    heading_error_encoder = [0.0]
    heading_error_biased = [0.0]

    continuous_left_distance = [0.0]
    quantized_left_distance = [0.0]
    cumulative_left_ground = [0.0]
    cumulative_left_encoder = [0.0]

    wheel_angle_left = 0.0
    wheel_angle_right = 0.0
    last_left_count = 0
    last_right_count = 0
    slip_loss_left = 0.0
    commanded_left_during_slip = 0.0

    for step in range(1, len(TIME)):
        time_midpoint = 0.5 * (TIME[step - 1] + TIME[step])
        center_speed, yaw_rate = commanded_body_velocity(time_midpoint)
        left_wheel_speed = center_speed - 0.5 * TRACK_WIDTH * yaw_rate
        right_wheel_speed = center_speed + 0.5 * TRACK_WIDTH * yaw_rate

        left_wheel_arc = left_wheel_speed * DT
        right_wheel_arc = right_wheel_speed * DT

        is_slipping = SLIP_START <= time_midpoint <= SLIP_END
        left_ground_arc = (
            LEFT_SLIP_GROUND_RATIO * left_wheel_arc if is_slipping else left_wheel_arc
        )
        right_ground_arc = right_wheel_arc

        if is_slipping:
            commanded_left_during_slip += left_wheel_arc
            slip_loss_left += left_wheel_arc - left_ground_arc

        true_pose = integrate_pose(true_pose, left_ground_arc, right_ground_arc)

        wheel_angle_left += left_wheel_arc / WHEEL_RADIUS_TRUE
        wheel_angle_right += right_wheel_arc / WHEEL_RADIUS_TRUE
        left_count = int(np.rint(wheel_angle_left / (2.0 * np.pi) * PULSES_PER_REVOLUTION))
        right_count = int(np.rint(wheel_angle_right / (2.0 * np.pi) * PULSES_PER_REVOLUTION))
        left_count_delta = left_count - last_left_count
        right_count_delta = right_count - last_right_count
        last_left_count = left_count
        last_right_count = right_count

        left_encoder_arc = (
            left_count_delta * 2.0 * np.pi * WHEEL_RADIUS_NOMINAL / PULSES_PER_REVOLUTION
        )
        right_encoder_arc = (
            right_count_delta * 2.0 * np.pi * WHEEL_RADIUS_NOMINAL / PULSES_PER_REVOLUTION
        )
        left_biased_arc = (
            left_count_delta * 2.0 * np.pi * WHEEL_RADIUS_LEFT_BIASED / PULSES_PER_REVOLUTION
        )
        right_biased_arc = (
            right_count_delta * 2.0 * np.pi * WHEEL_RADIUS_RIGHT_BIASED / PULSES_PER_REVOLUTION
        )

        encoder_pose = integrate_pose(encoder_pose, left_encoder_arc, right_encoder_arc)
        biased_pose = integrate_pose(biased_pose, left_biased_arc, right_biased_arc)

        true_path.append(true_pose.copy())
        encoder_path.append(encoder_pose.copy())
        biased_path.append(biased_pose.copy())
        position_error_encoder.append(float(np.linalg.norm(encoder_pose[:2] - true_pose[:2])))
        position_error_biased.append(float(np.linalg.norm(biased_pose[:2] - true_pose[:2])))
        heading_error_encoder.append(float(wrap_angle(encoder_pose[2] - true_pose[2])))
        heading_error_biased.append(float(wrap_angle(biased_pose[2] - true_pose[2])))

        continuous_left_distance.append(continuous_left_distance[-1] + left_wheel_arc)
        quantized_left_distance.append(quantized_left_distance[-1] + left_encoder_arc)
        cumulative_left_ground.append(cumulative_left_ground[-1] + left_ground_arc)
        cumulative_left_encoder.append(cumulative_left_encoder[-1] + left_encoder_arc)

    return {
        "true_path": np.asarray(true_path),
        "encoder_path": np.asarray(encoder_path),
        "biased_path": np.asarray(biased_path),
        "position_error_encoder": np.asarray(position_error_encoder),
        "position_error_biased": np.asarray(position_error_biased),
        "heading_error_encoder": np.asarray(heading_error_encoder),
        "heading_error_biased": np.asarray(heading_error_biased),
        "continuous_left_distance": np.asarray(continuous_left_distance),
        "quantized_left_distance": np.asarray(quantized_left_distance),
        "cumulative_left_ground": np.asarray(cumulative_left_ground),
        "cumulative_left_encoder": np.asarray(cumulative_left_encoder),
        "slip_loss_left": slip_loss_left,
        "commanded_left_during_slip": commanded_left_during_slip,
    }


def add_panel_label(axis, label: str) -> None:
    axis.text(
        0.025,
        0.965,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        color="#222222",
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "#c8c8c8",
            "alpha": 0.92,
        },
    )


def shade_slip_interval(axis) -> None:
    axis.axvspan(SLIP_START, SLIP_END, color="#f4a261", alpha=0.18, linewidth=0.0)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "ja" / "figures" / "flow6_encoder_slip_examples.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 9,
            "font.family": ["Noto Sans", "DejaVu Sans"],
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 7.5,
            "figure.dpi": 180,
            "savefig.dpi": 180,
            "axes.unicode_minus": False,
        }
    )

    data = simulate()
    true_path = data["true_path"]
    encoder_path = data["encoder_path"]
    biased_path = data["biased_path"]

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.2), constrained_layout=False)

    path_axis = axes[0, 0]
    path_axis.plot(true_path[:, 0], true_path[:, 1], color="#111111", linewidth=2.0, label="true path")
    path_axis.plot(
        encoder_path[:, 0],
        encoder_path[:, 1],
        color="#1f77b4",
        linewidth=1.6,
        linestyle="--",
        label="encoder odometry",
    )
    path_axis.plot(
        biased_path[:, 0],
        biased_path[:, 1],
        color="#d62728",
        linewidth=1.6,
        linestyle="-.",
        label="biased odometry",
    )
    slip_mask = (SLIP_START <= TIME) & (TIME <= SLIP_END)
    path_axis.plot(
        true_path[slip_mask, 0],
        true_path[slip_mask, 1],
        color="#f28e2b",
        linewidth=4.0,
        alpha=0.75,
        label="slip interval",
    )
    path_axis.plot(true_path[0, 0], true_path[0, 1], marker="o", color="#111111", markersize=4)
    path_axis.plot(true_path[-1, 0], true_path[-1, 1], marker="s", color="#111111", markersize=4)
    path_axis.set_xlabel("x position [m]")
    path_axis.set_ylabel("y position [m]")
    path_axis.set_aspect("equal", adjustable="box")
    path_axis.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.75)
    path_axis.legend(loc="lower right", frameon=True)
    add_panel_label(path_axis, "A path comparison")

    quant_axis = axes[0, 1]
    zoom_mask = TIME <= 0.60
    quant_axis.plot(
        TIME[zoom_mask],
        1000.0 * data["continuous_left_distance"][zoom_mask],
        color="#111111",
        linewidth=1.8,
        label="continuous wheel arc",
    )
    quant_axis.plot(
        TIME[zoom_mask],
        1000.0 * data["quantized_left_distance"][zoom_mask],
        color="#1f77b4",
        linewidth=1.5,
        drawstyle="steps-post",
        label="integer pulse distance",
    )
    distance_per_pulse = 2.0 * np.pi * WHEEL_RADIUS_NOMINAL / PULSES_PER_REVOLUTION
    quant_axis.text(
        0.04,
        0.82,
        f"q = {1000.0 * distance_per_pulse:.2f} mm/count",
        transform=quant_axis.transAxes,
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#c8c8c8"},
    )
    quant_axis.set_xlabel("time [s]")
    quant_axis.set_ylabel("left wheel distance [mm]")
    quant_axis.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.75)
    quant_axis.legend(loc="lower right", frameon=True)
    add_panel_label(quant_axis, "B pulse quantization")

    position_axis = axes[1, 0]
    shade_slip_interval(position_axis)
    position_axis.plot(
        TIME,
        100.0 * data["position_error_encoder"],
        color="#1f77b4",
        linewidth=1.7,
        label="encoder odometry",
    )
    position_axis.plot(
        TIME,
        100.0 * data["position_error_biased"],
        color="#d62728",
        linewidth=1.7,
        linestyle="-.",
        label="biased odometry",
    )
    position_axis.set_xlabel("time [s]")
    position_axis.set_ylabel("position error [cm]")
    position_axis.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.75)
    position_axis.legend(loc="upper left", frameon=True)
    add_panel_label(position_axis, "C accumulated position error")

    heading_axis = axes[1, 1]
    shade_slip_interval(heading_axis)
    heading_axis.plot(
        TIME,
        np.rad2deg(data["heading_error_encoder"]),
        color="#1f77b4",
        linewidth=1.7,
        label="encoder odometry",
    )
    heading_axis.plot(
        TIME,
        np.rad2deg(data["heading_error_biased"]),
        color="#d62728",
        linewidth=1.7,
        linestyle="-.",
        label="biased odometry",
    )
    heading_axis.set_xlabel("time [s]")
    heading_axis.set_ylabel("heading error [deg]")
    heading_axis.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.75)
    heading_axis.legend(loc="lower left", frameon=True)
    add_panel_label(heading_axis, "D accumulated heading error")

    finalize_figure(fig, output_path)

    final_encoder_position_error = float(data["position_error_encoder"][-1])
    final_biased_position_error = float(data["position_error_biased"][-1])
    final_encoder_heading_error = float(np.rad2deg(data["heading_error_encoder"][-1]))
    final_biased_heading_error = float(np.rad2deg(data["heading_error_biased"][-1]))
    slip_loss_left = float(data["slip_loss_left"])
    slip_heading = float(np.rad2deg(slip_loss_left / TRACK_WIDTH))

    print(f"wrote {output_path}")
    print(f"distance_per_pulse_mm={1000.0 * distance_per_pulse:.3f}")
    print(f"quantization_distance_bound_mm={500.0 * distance_per_pulse:.3f}")
    print(f"quantization_heading_bound_deg={np.rad2deg(distance_per_pulse / TRACK_WIDTH):.3f}")
    print(f"commanded_left_during_slip_m={float(data['commanded_left_during_slip']):.3f}")
    print(f"left_slip_loss_m={slip_loss_left:.3f}")
    print(f"slip_heading_difference_deg={slip_heading:.2f}")
    print(f"final_encoder_position_error_m={final_encoder_position_error:.3f}")
    print(f"final_encoder_heading_error_deg={final_encoder_heading_error:.2f}")
    print(f"final_biased_position_error_m={final_biased_position_error:.3f}")
    print(f"final_biased_heading_error_deg={final_biased_heading_error:.2f}")


if __name__ == "__main__":
    main()
