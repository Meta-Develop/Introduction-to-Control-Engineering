#!/usr/bin/env python3
"""Generate reference-shaping and rate-limit examples for a motor servo."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("matplotlib is required to generate the reference-shaping figure") from exc

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

TIME_STEP = 0.002
TIME_STOP = 5.0
STEP_TIME = 0.30
STEP_SIZE = 1.0

SHAPING_TIME = 0.60
RATE_LIMIT = 0.90

INERTIA = 1.0
DAMPING = 0.55
TORQUE_PER_VOLT = 1.0
VOLTAGE_LIMIT = 2.2
POSITION_KP = 10.0
VELOCITY_KD = 3.2

COLORS = {
    "raw": "#4a4a4a",
    "shaped": "#1f77b4",
    "limited": "#2ca02c",
    "limit": "#b22222",
}


def configure_fonts() -> None:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    selected_fonts = [
        font_name for font_name in JAPANESE_FONT_CANDIDATES if font_name in available_fonts
    ]
    if not selected_fonts:
        raise SystemExit(
            "A Japanese-capable Matplotlib font is required; install "
            "Noto Sans CJK JP, Harano Aji Gothic, or another listed font."
        )
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = selected_fonts + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def command_step(time: np.ndarray) -> np.ndarray:
    return np.where(time >= STEP_TIME, STEP_SIZE, 0.0)


def first_order_reference(time: np.ndarray, time_constant: float) -> tuple[np.ndarray, np.ndarray]:
    reference = np.zeros_like(time)
    speed = np.zeros_like(time)
    for index in range(len(time) - 1):
        command = STEP_SIZE if time[index] >= STEP_TIME else 0.0
        speed[index] = (command - reference[index]) / time_constant
        reference[index + 1] = reference[index] + TIME_STEP * speed[index]
    speed[-1] = speed[-2]
    return reference, speed


def rate_limited_reference(time: np.ndarray, max_rate: float) -> tuple[np.ndarray, np.ndarray]:
    reference = np.zeros_like(time)
    speed = np.zeros_like(time)
    max_increment = max_rate * TIME_STEP
    for index in range(len(time) - 1):
        command = STEP_SIZE if time[index] >= STEP_TIME else 0.0
        increment = float(np.clip(command - reference[index], -max_increment, max_increment))
        reference[index + 1] = reference[index] + increment
        speed[index] = increment / TIME_STEP
    speed[-1] = speed[-2]
    return reference, speed


def simulate_servo(reference: np.ndarray) -> dict[str, np.ndarray]:
    theta = 0.0
    omega = 0.0
    output = np.zeros_like(reference)
    speed = np.zeros_like(reference)
    command_voltage = np.zeros_like(reference)
    applied_voltage = np.zeros_like(reference)

    for index, target in enumerate(reference):
        voltage_command = POSITION_KP * (target - theta) - VELOCITY_KD * omega
        voltage = float(np.clip(voltage_command, -VOLTAGE_LIMIT, VOLTAGE_LIMIT))

        output[index] = theta
        speed[index] = omega
        command_voltage[index] = voltage_command
        applied_voltage[index] = voltage

        if index == len(reference) - 1:
            break

        theta += TIME_STEP * omega
        omega += TIME_STEP * (
            (TORQUE_PER_VOLT * voltage - DAMPING * omega) / INERTIA
        )

    margin = 1.0 - np.abs(command_voltage) / VOLTAGE_LIMIT
    return {
        "output": output,
        "speed": speed,
        "command_voltage": command_voltage,
        "applied_voltage": applied_voltage,
        "margin": margin,
    }


def add_panel_label(axis, label: str) -> None:
    axis.text(
        0.02,
        0.92,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": "#cccccc"},
    )


def style_axis(axis) -> None:
    axis.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)


def plot_reference_panel(axis, time, raw, shaped, limited) -> None:
    axis.plot(time, raw, color=COLORS["raw"], linestyle=":", linewidth=1.9, label="生ステップ")
    axis.plot(
        time,
        shaped,
        color=COLORS["shaped"],
        linewidth=2.0,
        label=rf"一次整形, $\tau_r={SHAPING_TIME:.2f}\,\mathrm{{s}}$",
    )
    axis.plot(
        time,
        limited,
        color=COLORS["limited"],
        linewidth=2.0,
        linestyle="--",
        label=rf"レート制限, $v_{{max}}={RATE_LIMIT:.2f}\,\mathrm{{rad/s}}$",
    )
    axis.axvline(STEP_TIME, color="#777777", linestyle="--", linewidth=1.0)
    axis.set_xlim(0.0, TIME_STOP)
    axis.set_ylim(-0.05, 1.08)
    axis.set_xlabel(r"時間 $t$ [s]")
    axis.set_ylabel(r"参照位置 $r_a$ [rad]")
    axis.legend(loc="lower right", fontsize=7.5)
    add_panel_label(axis, "(a) 参照値の作り方")
    style_axis(axis)


def plot_reference_speed_panel(axis, time, shaped_speed, limited_speed) -> None:
    axis.plot(
        time,
        shaped_speed,
        color=COLORS["shaped"],
        linewidth=2.0,
        label=rf"一次整形, 初期勾配 $1/\tau_r={1.0 / SHAPING_TIME:.2f}$",
    )
    axis.plot(
        time,
        limited_speed,
        color=COLORS["limited"],
        linewidth=2.0,
        linestyle="--",
        label=rf"レート制限, $v_{{max}}={RATE_LIMIT:.2f}$",
    )
    axis.axvline(STEP_TIME, color="#777777", linestyle="--", linewidth=1.0)
    axis.set_xlim(0.0, TIME_STOP)
    axis.set_ylim(-0.12, 1.85)
    axis.set_xlabel(r"時間 $t$ [s]")
    axis.set_ylabel(r"参照速度 $\dot{r}_a$ [rad/s]")
    axis.legend(loc="upper right", fontsize=7.5)
    add_panel_label(axis, "(b) 参照速度")
    style_axis(axis)


def plot_response_panel(axis, time, raw_result, shaped_result, limited_result) -> None:
    axis.plot(
        time,
        raw_result["output"],
        color=COLORS["raw"],
        linewidth=1.8,
        linestyle="-.",
        label="生ステップ応答",
    )
    axis.plot(
        time,
        shaped_result["output"],
        color=COLORS["shaped"],
        linewidth=2.0,
        label="一次整形応答",
    )
    axis.plot(
        time,
        limited_result["output"],
        color=COLORS["limited"],
        linewidth=2.0,
        linestyle="--",
        label="レート制限応答",
    )
    axis.axhline(STEP_SIZE, color="#777777", linestyle=":", linewidth=1.0)
    axis.set_xlim(0.0, TIME_STOP)
    axis.set_ylim(-0.07, 1.20)
    axis.set_xlabel(r"時間 $t$ [s]")
    axis.set_ylabel(r"位置応答 $\theta$ [rad]")
    axis.legend(loc="lower right", fontsize=7.4)
    add_panel_label(axis, "(c) 位置応答")
    style_axis(axis)


def plot_applied_voltage_panel(axis, time, raw_result, shaped_result, limited_result) -> None:
    axis.plot(
        time,
        raw_result["applied_voltage"],
        color=COLORS["raw"],
        linewidth=1.8,
        linestyle="-.",
        label="生ステップ",
    )
    axis.plot(
        time,
        shaped_result["applied_voltage"],
        color=COLORS["shaped"],
        linewidth=2.0,
        label="一次整形",
    )
    axis.plot(
        time,
        limited_result["applied_voltage"],
        color=COLORS["limited"],
        linewidth=2.0,
        linestyle="--",
        label="レート制限",
    )
    axis.axhline(VOLTAGE_LIMIT, color=COLORS["limit"], linestyle=":", linewidth=1.1)
    axis.axhline(-VOLTAGE_LIMIT, color=COLORS["limit"], linestyle=":", linewidth=1.1)
    axis.set_xlim(0.0, TIME_STOP)
    axis.set_ylim(-2.55, 2.55)
    axis.set_xlabel(r"時間 $t$ [s]")
    axis.set_ylabel(r"印加電圧 $u$ [V]")
    axis.legend(loc="upper right", fontsize=7.4)
    add_panel_label(axis, "(d) 印加入力")
    style_axis(axis)


def plot_margin_panel(axis, time, raw_result, shaped_result, limited_result) -> None:
    axis.plot(
        time,
        raw_result["margin"],
        color=COLORS["raw"],
        linewidth=1.8,
        linestyle="-.",
        label="生ステップ",
    )
    axis.plot(
        time,
        shaped_result["margin"],
        color=COLORS["shaped"],
        linewidth=2.0,
        label="一次整形",
    )
    axis.plot(
        time,
        limited_result["margin"],
        color=COLORS["limited"],
        linewidth=2.0,
        linestyle="--",
        label="レート制限",
    )
    axis.axhline(0.0, color=COLORS["limit"], linestyle=":", linewidth=1.1)
    axis.set_xlim(0.0, TIME_STOP)
    axis.set_ylim(-3.8, 1.05)
    axis.set_xlabel(r"時間 $t$ [s]")
    axis.set_ylabel(r"指令余裕 $\mu_c=1-|u_c|/u_{max}$ [-]")
    axis.legend(loc="lower right", fontsize=7.4)
    add_panel_label(axis, "(e) 飽和前の指令余裕")
    style_axis(axis)


def plot_parameter_panel(axis, time) -> None:
    speed_grid = np.linspace(0.50, 4.00, 16)
    shaped_margins = []
    limited_margins = []

    for maximum_speed in speed_grid:
        time_constant = STEP_SIZE / maximum_speed
        shaped, _ = first_order_reference(time, time_constant)
        limited, _ = rate_limited_reference(time, maximum_speed)
        shaped_margins.append(float(np.min(simulate_servo(shaped)["margin"])))
        limited_margins.append(float(np.min(simulate_servo(limited)["margin"])))

    axis.plot(
        speed_grid,
        shaped_margins,
        color=COLORS["shaped"],
        linewidth=2.0,
        marker="o",
        markersize=3.5,
        label=r"一次整形, $\Delta r/\tau_r$",
    )
    axis.plot(
        speed_grid,
        limited_margins,
        color=COLORS["limited"],
        linewidth=2.0,
        linestyle="--",
        marker="s",
        markersize=3.4,
        label=r"レート制限, $v_{max}$",
    )
    axis.axhline(0.0, color=COLORS["limit"], linestyle=":", linewidth=1.1)
    axis.set_xlim(0.45, 4.05)
    axis.set_ylim(-1.35, 0.85)
    axis.set_xlabel(r"最大参照速度 [rad/s]")
    axis.set_ylabel(r"最小指令余裕 $\min \mu_c$ [-]")
    axis.legend(loc="upper right", fontsize=7.2)
    add_panel_label(axis, "(f) パラメータ変化")
    style_axis(axis)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "ja" / "figures" / "reference_shaping_rate_limit_examples.png"
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

    time = np.arange(0.0, TIME_STOP + TIME_STEP, TIME_STEP)
    raw_reference = command_step(time)
    shaped_reference, shaped_speed = first_order_reference(time, SHAPING_TIME)
    limited_reference, limited_speed = rate_limited_reference(time, RATE_LIMIT)

    raw_result = simulate_servo(raw_reference)
    shaped_result = simulate_servo(shaped_reference)
    limited_result = simulate_servo(limited_reference)

    fig, axes = plt.subplots(3, 2, figsize=(7.2, 8.8), constrained_layout=False)

    plot_reference_panel(axes[0, 0], time, raw_reference, shaped_reference, limited_reference)
    plot_reference_speed_panel(axes[0, 1], time, shaped_speed, limited_speed)
    plot_response_panel(axes[1, 0], time, raw_result, shaped_result, limited_result)
    plot_applied_voltage_panel(axes[1, 1], time, raw_result, shaped_result, limited_result)
    plot_margin_panel(axes[2, 0], time, raw_result, shaped_result, limited_result)
    plot_parameter_panel(axes[2, 1], time)

    finalize_figure(fig, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
