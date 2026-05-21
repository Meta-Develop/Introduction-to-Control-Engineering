#!/usr/bin/env python3
"""Generate friction, dead-zone, and backlash compensation examples."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("matplotlib is required to generate the compensation figure") from exc

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

TORQUE_STATIC = 0.16  # N m
TORQUE_COULOMB = 0.105  # N m
VISCOUS_FRICTION = 0.018  # N m s/rad
DEAD_ZONE_VOLTAGE = 0.72  # V
SPEED_GAIN = 8.0  # rad/s/V after the dead-zone edge
BACKLASH_WIDTH = 0.18  # rad
POSITION_DEAD_ZONE = 0.055  # rad, equivalent small-command lost motion


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


def smooth_sign(value: np.ndarray | float, scale: float) -> np.ndarray:
    return np.tanh(np.asarray(value, dtype=float) / scale)


def measured_friction_torque(omega: np.ndarray) -> np.ndarray:
    ripple = 0.004 * np.sin(2.5 * omega) + 0.002 * np.sin(7.0 * omega)
    return (
        VISCOUS_FRICTION * omega
        + TORQUE_COULOMB * smooth_sign(omega, 0.22)
        + ripple
    )


def dead_zone_output(voltage: np.ndarray) -> np.ndarray:
    moving_voltage = np.sign(voltage) * np.maximum(np.abs(voltage) - DEAD_ZONE_VOLTAGE, 0.0)
    ripple = 0.10 * np.sin(4.8 * voltage) * (np.abs(moving_voltage) > 0.0)
    return SPEED_GAIN * moving_voltage + ripple


def compensated_voltage(linear_voltage: np.ndarray) -> np.ndarray:
    return linear_voltage + DEAD_ZONE_VOLTAGE * smooth_sign(linear_voltage, 0.05)


def apply_backlash(input_angle: np.ndarray, width: float) -> np.ndarray:
    output_angle = np.zeros_like(input_angle)
    output_angle[0] = input_angle[0]
    half_width = 0.5 * width

    for index in range(1, len(input_angle)):
        delta = input_angle[index] - output_angle[index - 1]
        if delta > half_width:
            output_angle[index] = input_angle[index] - half_width
        elif delta < -half_width:
            output_angle[index] = input_angle[index] + half_width
        else:
            output_angle[index] = output_angle[index - 1]

    return output_angle


def position_dead_zone(command: np.ndarray) -> np.ndarray:
    return np.sign(command) * np.maximum(np.abs(command) - POSITION_DEAD_ZONE, 0.0)


def reference_trace(time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = 0.44 * np.sin(2.0 * np.pi * time / 5.0)
    reference += 0.08 * np.sin(2.0 * np.pi * time / 10.0)
    reference_velocity = np.gradient(reference, time)
    return reference, reference_velocity


def tracking_traces() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time = np.linspace(0.0, 10.0, 800)
    reference, reference_velocity = reference_trace(time)

    uncompensated_motor = position_dead_zone(reference)
    uncompensated_output = apply_backlash(uncompensated_motor, BACKLASH_WIDTH)

    takeup = 0.5 * BACKLASH_WIDTH * smooth_sign(reference_velocity, 0.08)
    stiction_boost = POSITION_DEAD_ZONE * smooth_sign(reference + 0.08 * reference_velocity, 0.05)
    compensated_motor = position_dead_zone(reference + takeup + stiction_boost)
    compensated_output = apply_backlash(compensated_motor, BACKLASH_WIDTH)

    return time, reference, uncompensated_output, compensated_output


def add_panel_label(axis, label: str) -> None:
    axis.text(
        0.02,
        0.96,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.92},
    )


def style_axis(axis) -> None:
    axis.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.72)
    axis.axhline(0.0, color="#777777", linewidth=0.8)
    axis.axvline(0.0, color="#777777", linewidth=0.8)


def plot_friction(axis) -> None:
    omega = np.linspace(-8.0, 8.0, 400)
    torque = measured_friction_torque(omega)
    compensation = VISCOUS_FRICTION * omega + TORQUE_COULOMB * smooth_sign(omega, 0.22)

    axis.plot(omega, torque, color="#5a5a5a", linewidth=1.9, label="測定: 動摩擦")
    axis.plot(omega, compensation, color="#1f77b4", linewidth=1.8, linestyle="--", label="補償モデル")
    axis.plot(
        [0.0, 0.0],
        [-TORQUE_STATIC, TORQUE_STATIC],
        color="#d62728",
        linewidth=4.0,
        solid_capstyle="round",
        label="静止摩擦帯",
    )
    axis.fill_between(
        [-0.24, 0.24],
        -TORQUE_STATIC,
        TORQUE_STATIC,
        color="#d62728",
        alpha=0.12,
        linewidth=0.0,
    )
    axis.text(
        -7.7,
        0.19,
        r"$\tau_f=b\omega+\tau_k\,\sigma_\epsilon(\omega)$",
        fontsize=8.2,
        color="#1f4e79",
    )
    axis.set_xlabel(r"角速度 $\omega$ [rad/s]")
    axis.set_ylabel(r"必要トルク $\tau$ [N m]")
    axis.set_xlim(-8.2, 8.2)
    axis.set_ylim(-0.30, 0.30)
    style_axis(axis)
    add_panel_label(axis, "(a) Coulomb/static")
    axis.legend(loc="lower right", fontsize=8.0, framealpha=0.90)


def plot_dead_zone(axis) -> None:
    linear_command = np.linspace(-3.0, 3.0, 500)
    uncompensated_speed = dead_zone_output(linear_command)
    compensated_speed = dead_zone_output(compensated_voltage(linear_command))
    desired_line = SPEED_GAIN * linear_command

    axis.plot(linear_command, desired_line, color="#444444", linewidth=1.2, linestyle=":", label="理想線形")
    axis.plot(linear_command, uncompensated_speed, color="#d62728", linewidth=1.8, label="補償なし")
    axis.plot(linear_command, compensated_speed, color="#1f77b4", linewidth=1.8, label="不感帯補償")
    axis.axvspan(-DEAD_ZONE_VOLTAGE, DEAD_ZONE_VOLTAGE, color="#d62728", alpha=0.09)
    axis.text(
        -2.85,
        -19.0,
        r"$v_c=v_\ell+v_d\,\sigma_\epsilon(v_\ell)$",
        fontsize=8.2,
        color="#1f4e79",
    )
    axis.set_xlabel(r"線形入力 $v_\ell$ [V]")
    axis.set_ylabel(r"定常角速度 $\omega_{\rm ss}$ [rad/s]")
    axis.set_xlim(-3.05, 3.05)
    axis.set_ylim(-21.5, 21.5)
    style_axis(axis)
    add_panel_label(axis, "(b) Dead zone")
    axis.legend(loc="lower right", fontsize=8.0, framealpha=0.90)


def plot_backlash(axis) -> None:
    forward_motor = np.linspace(-0.85, 0.85, 260)
    reverse_motor = np.linspace(0.85, -0.85, 260)
    motor_angle = np.concatenate([forward_motor, reverse_motor])
    load_angle = apply_backlash(motor_angle, BACKLASH_WIDTH)

    reference = np.linspace(-0.72, 0.72, 250)
    positive_comp = reference + 0.5 * BACKLASH_WIDTH
    negative_comp = reference - 0.5 * BACKLASH_WIDTH

    axis.plot(motor_angle, load_angle, color="#5a5a5a", linewidth=1.8, label="測定: 往復掃引")
    axis.plot(reference, positive_comp, color="#1f77b4", linewidth=1.5, linestyle="--", label="正転補償")
    axis.plot(reference, negative_comp, color="#2ca02c", linewidth=1.5, linestyle="--", label="逆転補償")
    axis.fill_between(
        [-0.85, 0.85],
        np.asarray([-0.85, 0.85]) - 0.5 * BACKLASH_WIDTH,
        np.asarray([-0.85, 0.85]) + 0.5 * BACKLASH_WIDTH,
        color="#d62728",
        alpha=0.08,
        linewidth=0.0,
    )
    axis.text(
        -0.80,
        0.54,
        r"$\theta_{m,r}=\theta_{l,r}+\frac{\Delta_\theta}{2}\sigma_\epsilon(\dot{\theta}_{l,r})$",
        fontsize=7.4,
        color="#1f4e79",
    )
    axis.set_xlabel(r"モータ角 $\theta_m$ [rad]")
    axis.set_ylabel(r"負荷角 $\theta_l$ [rad]")
    axis.set_xlim(-0.90, 0.90)
    axis.set_ylim(-0.90, 0.90)
    style_axis(axis)
    add_panel_label(axis, "(c) Backlash")
    axis.legend(loc="lower right", fontsize=7.5, framealpha=0.90)


def plot_tracking(axis) -> None:
    time, reference, uncompensated, compensated = tracking_traces()
    residual_uncompensated = reference - uncompensated
    residual_compensated = reference - compensated

    axis.plot(time, reference, color="#444444", linewidth=1.6, linestyle=":", label="目標負荷角")
    axis.plot(time, uncompensated, color="#d62728", linewidth=1.7, label="補償なし")
    axis.plot(time, compensated, color="#1f77b4", linewidth=1.7, label="補償あり")
    axis2 = axis.twinx()
    axis2.plot(
        time,
        residual_uncompensated,
        color="#d62728",
        linewidth=1.0,
        linestyle="--",
        alpha=0.45,
        label="残差: 補償なし",
    )
    axis2.plot(
        time,
        residual_compensated,
        color="#1f77b4",
        linewidth=1.0,
        linestyle="--",
        alpha=0.45,
        label="残差: 補償あり",
    )
    axis.set_xlabel(r"時間 $t$ [s]")
    axis.set_ylabel(r"負荷角 $\theta_l$ [rad]")
    axis2.set_ylabel(r"追従残差 $e_\theta$ [rad]")
    axis.set_xlim(0.0, 10.0)
    axis.set_ylim(-0.62, 0.62)
    axis2.set_ylim(-0.30, 0.30)
    axis.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.72)
    axis.text(
        0.35,
        -0.54,
        r"反転直後は補償後も $|e_\theta|>0$",
        fontsize=8.0,
        color="#444444",
    )
    add_panel_label(axis, "(d) Before/after")
    lines1, labels1 = axis.get_legend_handles_labels()
    lines2, labels2 = axis2.get_legend_handles_labels()
    axis.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7.3, framealpha=0.90)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "ja" / "figures" / "friction_deadzone_backlash_examples.png"
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

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.8), constrained_layout=False)

    plot_friction(axes[0, 0])
    plot_dead_zone(axes[0, 1])
    plot_backlash(axes[1, 0])
    plot_tracking(axes[1, 1])

    finalize_figure(fig, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
