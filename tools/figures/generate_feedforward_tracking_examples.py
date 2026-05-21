#!/usr/bin/env python3
"""Generate feedforward tracking examples for motor and differential-drive systems."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("matplotlib is required to generate the feedforward tracking figure") from exc

from figure_style import finalize_figure


JAPANESE_FONT_CANDIDATES = (
    "Noto Sans CJK JP",
    "Noto Sans JP",
    "Harano Aji Gothic",
    "UDEV Gothic",
    "IPAexGothic",
    "IPAGothic",
    "Yu Gothic",
    "YuGothic",
    "Hiragino Sans",
    "Hiragino Kaku Gothic ProN",
    "Meiryo",
    "TakaoGothic",
)

COLORS = {
    "reference": "#303030",
    "feedback": "#d62728",
    "feedforward": "#1f77b4",
    "disturbance": "#2ca02c",
    "limit": "#8c1d18",
    "neutral": "#666666",
}

MOTOR_TIME_STEP = 0.002
MOTOR_TIME_STOP = 4.2
MOTOR_J = 0.020
MOTOR_B = 0.018
MOTOR_KT = 0.170
MOTOR_J_NOMINAL = 0.018
MOTOR_B_NOMINAL = 0.015
MOTOR_KT_NOMINAL = 0.180
MOTOR_CURRENT_LIMIT = 4.8
MOTOR_KP = 0.36
MOTOR_KI = 1.20

TRACK_WIDTH = 0.34
WHEEL_RADIUS = 0.052
WHEEL_SPEED_LIMIT = 19.0
BODY_SPEED = 0.72
ROBOT_TIME_STEP = 0.01
ROBOT_TIME_STOP = 9.0


def configure_fonts() -> None:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    selected_fonts = [
        font_name
        for font_name in JAPANESE_FONT_CANDIDATES
        if font_name in available_fonts
    ]
    if not selected_fonts:
        raise SystemExit(
            "A Japanese-capable Matplotlib font is required; install "
            "Noto Sans CJK JP, Harano Aji Gothic, or another listed font."
        )
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = selected_fonts + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def smoothstep(s: np.ndarray | float) -> np.ndarray:
    s_array = np.clip(np.asarray(s, dtype=float), 0.0, 1.0)
    return 3.0 * s_array**2 - 2.0 * s_array**3


def motor_reference(time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    speed = np.zeros_like(time)
    rise = smoothstep((time - 0.25) / 1.20)
    fall = smoothstep((time - 2.75) / 0.95)
    speed = 15.0 * rise - 5.5 * fall
    acceleration = np.gradient(speed, time)
    return speed, acceleration


def load_torque(time: float) -> float:
    return 0.16 if time >= 2.18 else 0.0


def measured_load_torque(time: float) -> float:
    delay = 0.035
    return load_torque(time - delay)


def saturation_would_worsen(command: float, limited: float, error: float) -> bool:
    return (command > limited and error > 0.0) or (command < limited and error < 0.0)


def simulate_motor(mode: str) -> dict[str, np.ndarray]:
    time = np.arange(0.0, MOTOR_TIME_STOP + 0.5 * MOTOR_TIME_STEP, MOTOR_TIME_STEP)
    reference, acceleration = motor_reference(time)

    omega = 0.0
    integral_error = 0.0
    output = np.zeros_like(time)
    current = np.zeros_like(time)
    current_ff = np.zeros_like(time)
    margin = np.zeros_like(time)

    for index, t in enumerate(time):
        error = reference[index] - omega
        load_measurement = measured_load_torque(float(t)) if mode == "disturbance" else 0.0
        if mode in {"feedforward", "disturbance"}:
            feedforward = (
                MOTOR_J_NOMINAL * acceleration[index]
                + MOTOR_B_NOMINAL * reference[index]
                + load_measurement
            ) / MOTOR_KT_NOMINAL
        else:
            feedforward = 0.0

        trial_integral = integral_error + error * MOTOR_TIME_STEP
        command_trial = feedforward + MOTOR_KP * error + MOTOR_KI * trial_integral
        limited_trial = float(
            np.clip(command_trial, -MOTOR_CURRENT_LIMIT, MOTOR_CURRENT_LIMIT)
        )
        if not saturation_would_worsen(command_trial, limited_trial, error):
            integral_error = trial_integral

        command = feedforward + MOTOR_KP * error + MOTOR_KI * integral_error
        limited = float(np.clip(command, -MOTOR_CURRENT_LIMIT, MOTOR_CURRENT_LIMIT))

        output[index] = omega
        current[index] = limited
        current_ff[index] = feedforward
        margin[index] = 1.0 - abs(command) / MOTOR_CURRENT_LIMIT

        if index == len(time) - 1:
            break

        omega_dot = (MOTOR_KT * limited - MOTOR_B * omega - load_torque(float(t))) / MOTOR_J
        omega += MOTOR_TIME_STEP * omega_dot

    return {
        "time": time,
        "reference": reference,
        "acceleration": acceleration,
        "output": output,
        "current": current,
        "current_ff": current_ff,
        "margin": margin,
    }


def curvature_profile(time: np.ndarray) -> np.ndarray:
    envelope = smoothstep(time / 1.0) * (1.0 - 0.18 * smoothstep((time - 7.8) / 1.0))
    return envelope * (
        1.05 * np.sin(2.0 * np.pi * time / 5.4)
        + 0.30 * np.sin(2.0 * np.pi * time / 2.7)
    )


def integrate_reference_path() -> dict[str, np.ndarray]:
    time = np.arange(0.0, ROBOT_TIME_STOP + 0.5 * ROBOT_TIME_STEP, ROBOT_TIME_STEP)
    curvature = curvature_profile(time)
    yaw_rate = BODY_SPEED * curvature

    x = np.zeros_like(time)
    y = np.zeros_like(time)
    heading = np.zeros_like(time)

    for index in range(len(time) - 1):
        x[index + 1] = x[index] + ROBOT_TIME_STEP * BODY_SPEED * np.cos(heading[index])
        y[index + 1] = y[index] + ROBOT_TIME_STEP * BODY_SPEED * np.sin(heading[index])
        heading[index + 1] = heading[index] + ROBOT_TIME_STEP * yaw_rate[index]

    return {
        "time": time,
        "x": x,
        "y": y,
        "heading": heading,
        "curvature": curvature,
        "yaw_rate": yaw_rate,
    }


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def simulate_robot(reference: dict[str, np.ndarray], use_curvature_feedforward: bool) -> dict[str, np.ndarray]:
    time = reference["time"]
    x = reference["x"][0]
    y = reference["y"][0] - 0.24
    heading = reference["heading"][0] + 0.18

    x_history = np.zeros_like(time)
    y_history = np.zeros_like(time)
    heading_history = np.zeros_like(time)
    lateral_error = np.zeros_like(time)
    wheel_left = np.zeros_like(time)
    wheel_right = np.zeros_like(time)

    k_x = 0.75
    k_y = 2.45
    k_heading = 1.75

    for index, t in enumerate(time):
        ref_x = reference["x"][index]
        ref_y = reference["y"][index]
        ref_heading = reference["heading"][index]
        dx = ref_x - x
        dy = ref_y - y

        longitudinal_error = np.cos(ref_heading) * dx + np.sin(ref_heading) * dy
        lateral = -np.sin(ref_heading) * dx + np.cos(ref_heading) * dy
        heading_error = wrap_angle(ref_heading - heading)

        v_command = BODY_SPEED + k_x * longitudinal_error
        yaw_command = k_y * lateral + k_heading * heading_error
        if use_curvature_feedforward:
            yaw_command += BODY_SPEED * reference["curvature"][index]

        left_linear = v_command - 0.5 * TRACK_WIDTH * yaw_command
        right_linear = v_command + 0.5 * TRACK_WIDTH * yaw_command
        left_limited = float(
            np.clip(left_linear / WHEEL_RADIUS, -WHEEL_SPEED_LIMIT, WHEEL_SPEED_LIMIT)
        )
        right_limited = float(
            np.clip(right_linear / WHEEL_RADIUS, -WHEEL_SPEED_LIMIT, WHEEL_SPEED_LIMIT)
        )

        actual_left_linear = WHEEL_RADIUS * left_limited
        actual_right_linear = WHEEL_RADIUS * right_limited
        actual_v = 0.5 * (actual_left_linear + actual_right_linear)
        actual_yaw_rate = (actual_right_linear - actual_left_linear) / TRACK_WIDTH

        x_history[index] = x
        y_history[index] = y
        heading_history[index] = heading
        lateral_error[index] = lateral
        wheel_left[index] = left_limited
        wheel_right[index] = right_limited

        if index == len(time) - 1:
            break

        x += ROBOT_TIME_STEP * actual_v * np.cos(heading)
        y += ROBOT_TIME_STEP * actual_v * np.sin(heading)
        heading = wrap_angle(heading + ROBOT_TIME_STEP * actual_yaw_rate)

    return {
        "time": time,
        "x": x_history,
        "y": y_history,
        "heading": heading_history,
        "lateral_error": lateral_error,
        "wheel_left": wheel_left,
        "wheel_right": wheel_right,
    }


def reference_wheel_speeds(reference: dict[str, np.ndarray], scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    body_speed = scale * BODY_SPEED
    yaw_rate = body_speed * reference["curvature"]
    left = (body_speed - 0.5 * TRACK_WIDTH * yaw_rate) / WHEEL_RADIUS
    right = (body_speed + 0.5 * TRACK_WIDTH * yaw_rate) / WHEEL_RADIUS
    return left, right


def add_panel_label(axis, label: str) -> None:
    axis.text(
        0.015,
        0.965,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9.0,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.20",
            "facecolor": "white",
            "edgecolor": "#bbbbbb",
            "alpha": 0.94,
        },
    )


def style_axis(axis) -> None:
    axis.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.72)


def draw_arrow(axis, start: tuple[float, float], end: tuple[float, float], **kwargs) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.15,
        color=kwargs.pop("color", "#303030"),
        connectionstyle=kwargs.pop("connectionstyle", "arc3,rad=0.0"),
        **kwargs,
    )
    axis.add_patch(arrow)


def draw_box(
    axis,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.018",
        linewidth=1.05,
        edgecolor="#303030",
        facecolor=facecolor,
    )
    axis.add_patch(patch)
    axis.text(
        xy[0] + width / 2.0,
        xy[1] + height / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=8.1,
    )


def draw_signal_path(axis) -> None:
    axis.set_axis_off()
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)

    draw_box(axis, (0.04, 0.62), 0.18, 0.20, "既知目標\n$r,\\dot r,\\ddot r,\\kappa$", facecolor="#eef4ff")
    draw_box(axis, (0.30, 0.62), 0.19, 0.20, "目標 FF\n$u_r$", facecolor="#e5f4ff")
    draw_box(axis, (0.04, 0.28), 0.18, 0.18, "測定外乱\n$d_m$", facecolor="#eff8ed")
    draw_box(axis, (0.30, 0.28), 0.19, 0.18, "外乱 FF\n$u_d$", facecolor="#edf8ef")
    draw_box(axis, (0.59, 0.63), 0.13, 0.17, "和\n$u_c$", facecolor="#fff5df")
    draw_box(axis, (0.80, 0.63), 0.13, 0.17, "制限\n$u$", facecolor="#ffe8e5")
    draw_box(axis, (0.80, 0.27), 0.13, 0.17, "対象\n$y$", facecolor="#f1f1f1")
    draw_box(axis, (0.55, 0.20), 0.17, 0.16, "FB\n$u_{fb}$", facecolor="#f4ecff")

    draw_arrow(axis, (0.22, 0.72), (0.30, 0.72))
    draw_arrow(axis, (0.49, 0.72), (0.59, 0.72))
    draw_arrow(axis, (0.72, 0.72), (0.80, 0.72))
    draw_arrow(axis, (0.865, 0.63), (0.865, 0.44))
    draw_arrow(axis, (0.80, 0.35), (0.72, 0.30))
    draw_arrow(axis, (0.55, 0.28), (0.49, 0.28))
    draw_arrow(axis, (0.49, 0.28), (0.59, 0.66), connectionstyle="arc3,rad=0.25")
    draw_arrow(axis, (0.22, 0.37), (0.30, 0.37))
    draw_arrow(axis, (0.49, 0.37), (0.59, 0.68), connectionstyle="arc3,rad=-0.23")

    axis.text(0.025, 0.95, "(a) 信号経路", ha="left", va="top", fontsize=9.2, fontweight="bold")
    axis.text(0.39, 0.54, r"$u_r=P_n^{-1}r$", ha="center", va="center", fontsize=8.0, color="#1f4e79")
    axis.text(
        0.74,
        0.08,
        "先回りは既知成分だけを相殺し、\n残差は測定値から戻す。",
        ha="center",
        va="bottom",
        fontsize=8.4,
        color="#444444",
    )


def draw_motor_tracking(axis, fb_data, ff_data, disturbance_data) -> None:
    time = fb_data["time"]
    load = np.array([load_torque(float(t)) for t in time])

    axis.plot(time, fb_data["reference"], color=COLORS["reference"], linewidth=1.7, linestyle=":", label=r"目標 $\omega_r$")
    axis.plot(time, fb_data["output"], color=COLORS["feedback"], linewidth=1.7, label="FB のみ")
    axis.plot(time, ff_data["output"], color=COLORS["feedforward"], linewidth=1.8, label=r"$\dot\omega_r,\omega_r$ FF")
    axis.plot(time, disturbance_data["output"], color=COLORS["disturbance"], linewidth=1.8, linestyle="--", label="負荷測定も使用")
    axis.fill_between(time, 0.0, 1.0, where=load > 0.0, transform=axis.get_xaxis_transform(), color="#e0f1df", alpha=0.55)
    axis.text(2.25, 1.0, r"$\tau_L$ step", fontsize=8.2, color="#2f6b2f")
    axis.set_xlim(0.0, MOTOR_TIME_STOP)
    axis.set_ylim(-0.6, 17.2)
    axis.set_xlabel(r"時間 $t$ [s]")
    axis.set_ylabel(r"角速度 $\omega$ [rad/s]")
    axis.legend(loc="lower right", fontsize=7.8, framealpha=0.93)
    add_panel_label(axis, "(b) モータ速度追従")
    style_axis(axis)


def draw_trajectory(axis, reference, feedback_data, feedforward_data) -> None:
    rms_feedback = float(np.sqrt(np.mean(feedback_data["lateral_error"] ** 2)))
    rms_feedforward = float(np.sqrt(np.mean(feedforward_data["lateral_error"] ** 2)))

    axis.plot(reference["x"], reference["y"], color=COLORS["reference"], linewidth=1.8, linestyle=":", label="目標軌道")
    axis.plot(
        feedback_data["x"],
        feedback_data["y"],
        color=COLORS["feedback"],
        linewidth=1.7,
        label=rf"FB のみ, RMS={rms_feedback:.2f} m",
    )
    axis.plot(
        feedforward_data["x"],
        feedforward_data["y"],
        color=COLORS["feedforward"],
        linewidth=1.8,
        label=rf"曲率 FF, RMS={rms_feedforward:.2f} m",
    )
    axis.scatter(reference["x"][0], reference["y"][0], color="#303030", s=18, zorder=5)
    axis.scatter(reference["x"][-1], reference["y"][-1], color="#777777", s=18, zorder=5)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(r"$x$ [m]")
    axis.set_ylabel(r"$y$ [m]")
    axis.legend(loc="lower left", fontsize=7.5, framealpha=0.93)
    add_panel_label(axis, "(c) 差動二輪の軌道追従")
    style_axis(axis)


def draw_wheel_margin(axis, reference) -> None:
    scales = np.linspace(0.55, 1.55, 120)
    margins = []
    for scale in scales:
        left, right = reference_wheel_speeds(reference, scale)
        margins.append(1.0 - max(float(np.max(np.abs(left))), float(np.max(np.abs(right)))) / WHEEL_SPEED_LIMIT)
    margins = np.asarray(margins)

    left_nominal, right_nominal = reference_wheel_speeds(reference, 1.0)
    nominal_margin = 1.0 - max(float(np.max(np.abs(left_nominal))), float(np.max(np.abs(right_nominal)))) / WHEEL_SPEED_LIMIT

    axis.plot(scales, margins, color=COLORS["feedforward"], linewidth=2.0, label=r"$m_\omega$")
    axis.fill_between(scales, margins, 0.0, where=margins < 0.0, color="#ffe8e5", alpha=0.85)
    axis.axhline(0.0, color=COLORS["limit"], linestyle=":", linewidth=1.2)
    axis.axvline(1.0, color="#777777", linestyle="--", linewidth=1.0)
    axis.scatter([1.0], [nominal_margin], color=COLORS["feedforward"], edgecolor="#303030", zorder=5)
    axis.text(1.02, nominal_margin + 0.035, "本文の速度", fontsize=8.0, color="#303030")
    axis.set_xlim(0.55, 1.55)
    axis.set_ylim(-0.32, 0.55)
    axis.set_xlabel(r"軌道速度倍率 [-]")
    axis.set_ylabel(r"車輪速度余裕 $m_\omega$ [-]")
    axis.legend(loc="upper right", fontsize=8.0, framealpha=0.93)
    add_panel_label(axis, "(d) 曲率 FF と車輪速度限界")
    style_axis(axis)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "ja" / "figures" / "feedforward_tracking_examples.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    configure_fonts()
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 180,
            "savefig.dpi": 180,
        }
    )

    motor_feedback = simulate_motor("feedback")
    motor_feedforward = simulate_motor("feedforward")
    motor_disturbance = simulate_motor("disturbance")
    reference = integrate_reference_path()
    robot_feedback = simulate_robot(reference, use_curvature_feedforward=False)
    robot_feedforward = simulate_robot(reference, use_curvature_feedforward=True)

    figure = plt.figure(figsize=(7.4, 7.4), constrained_layout=False)
    grid = GridSpec(2, 2, figure=figure, height_ratios=[0.92, 1.05])

    signal_axis = figure.add_subplot(grid[0, 0])
    motor_axis = figure.add_subplot(grid[0, 1])
    trajectory_axis = figure.add_subplot(grid[1, 0])
    margin_axis = figure.add_subplot(grid[1, 1])

    draw_signal_path(signal_axis)
    draw_motor_tracking(motor_axis, motor_feedback, motor_feedforward, motor_disturbance)
    draw_trajectory(trajectory_axis, reference, robot_feedback, robot_feedforward)
    draw_wheel_margin(margin_axis, reference)

    figure.subplots_adjust(left=0.08, right=0.985, top=0.985, bottom=0.075, hspace=0.37, wspace=0.32)
    finalize_figure(figure, output_path, layout="none")
    plt.close(figure)
    print(output_path)


if __name__ == "__main__":
    main()
