#!/usr/bin/env python3
"""Generate accelerometer tilt and gravity-projection examples."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    from matplotlib.patches import Arc, FancyArrowPatch
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - figure creation requires it.
    raise SystemExit(
        "matplotlib is required to generate the accelerometer tilt figure"
    ) from exc

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

GRAVITY = 9.80665
TIME_STEP = 0.01
TIME_STOP = 12.0
ACCEL_NOISE_STD = 0.13
GYRO_BIAS = np.deg2rad(0.32)
GYRO_NOISE_STD = np.deg2rad(0.035)
CONTAMINATION_LEVELS = (0.0, 0.8, 1.6, 2.4)


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


def smooth_pulse(
    time: np.ndarray, start: float, stop: float, rise_time: float
) -> np.ndarray:
    leading = np.tanh((time - start) / rise_time)
    trailing = np.tanh((time - stop) / rise_time)
    return 0.5 * (leading - trailing)


def integrate_gyro(
    time: np.ndarray, theta_true: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    theta_rate = np.gradient(theta_true, time)
    gyro_measurement = theta_rate + GYRO_BIAS + rng.normal(
        0.0, GYRO_NOISE_STD, size=time.size
    )
    theta_gyro = np.empty_like(theta_true)
    theta_gyro[0] = theta_true[0]
    for index in range(1, time.size):
        theta_gyro[index] = theta_gyro[index - 1] + gyro_measurement[index - 1] * (
            time[index] - time[index - 1]
        )
    return theta_gyro


def simulate_signals() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(2606)
    time = np.arange(0.0, TIME_STOP + 0.5 * TIME_STEP, TIME_STEP)
    theta_true = np.deg2rad(6.0) * np.sin(2.0 * np.pi * time / 8.0)
    theta_true += np.deg2rad(2.2) * smooth_pulse(time, 2.2, 8.8, 0.55)

    gravity_x = -GRAVITY * np.sin(theta_true)
    gravity_z = GRAVITY * np.cos(theta_true)
    forward_acceleration = 1.6 * smooth_pulse(time, 4.0, 6.0, 0.18)
    forward_acceleration -= 0.7 * smooth_pulse(time, 8.2, 9.1, 0.16)

    noise_x = rng.normal(0.0, ACCEL_NOISE_STD, size=time.size)
    noise_z = rng.normal(0.0, ACCEL_NOISE_STD, size=time.size)
    theta_acc_low = np.arctan2(-(gravity_x + noise_x), gravity_z + noise_z)
    theta_acc_contaminated = np.arctan2(
        -(gravity_x + forward_acceleration + noise_x), gravity_z + noise_z
    )

    return {
        "time": time,
        "theta_true": theta_true,
        "theta_gyro": integrate_gyro(time, theta_true, rng),
        "theta_acc_low": theta_acc_low,
        "theta_acc_contaminated": theta_acc_contaminated,
        "gravity_x": gravity_x,
        "gravity_z": gravity_z,
        "forward_acceleration": forward_acceleration,
    }


def draw_arrow(
    axis,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    *,
    linewidth: float = 1.9,
    linestyle: str = "-",
    mutation_scale: float = 13.0,
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            mutation_scale=mutation_scale,
        )
    )


def draw_geometry(axis) -> None:
    theta = np.deg2rad(28.0)
    body_x = np.array([np.cos(theta), -np.sin(theta)])
    body_z = np.array([np.sin(theta), np.cos(theta)])
    origin = np.array([0.0, 0.0])

    axis.set_aspect("equal")
    axis.set_xlim(-1.25, 1.25)
    axis.set_ylim(-1.15, 1.30)
    axis.axis("off")

    draw_arrow(axis, (-1.05, 0.0), (1.05, 0.0), "#777777", linewidth=1.2)
    draw_arrow(axis, (0.0, -1.0), (0.0, 1.05), "#777777", linewidth=1.2)
    axis.text(1.08, -0.07, r"$x^I$", color="#555555", fontsize=10)
    axis.text(0.05, 1.06, r"$z^I$", color="#555555", fontsize=10)

    draw_arrow(axis, tuple(origin), tuple(0.98 * body_x), "#1f77b4", linewidth=2.4)
    draw_arrow(axis, tuple(origin), tuple(0.98 * body_z), "#2ca02c", linewidth=2.4)
    axis.text(*(1.03 * body_x + np.array([0.02, -0.04])), r"$x^B$", color="#1f77b4")
    axis.text(*(1.03 * body_z + np.array([0.02, 0.02])), r"$z^B$", color="#2ca02c")

    draw_arrow(axis, (0.62, 0.92), (0.62, 0.18), "#222222", linewidth=2.1)
    axis.text(0.66, 0.50, r"$g^I$ [m/s$^2$]", color="#222222", va="center")
    draw_arrow(axis, (-0.58, -0.65), (-0.58, 0.10), "#6a3d9a", linewidth=2.1)
    axis.text(-0.96, -0.32, r"$-g^I$: 静止時の比力", color="#6a3d9a")

    fx_vector = -0.38 * body_x
    fz_vector = 0.78 * body_z
    draw_arrow(axis, tuple(origin), tuple(fx_vector), "#d95f02", linewidth=2.3)
    draw_arrow(axis, tuple(origin), tuple(fz_vector), "#008b8b", linewidth=2.3)
    axis.text(
        *(fx_vector + np.array([-0.22, 0.03])),
        r"$a^B_{m,x}=-g\sin\theta$",
        color="#d95f02",
        fontsize=9,
    )
    axis.text(
        *(fz_vector + np.array([0.03, 0.03])),
        r"$a^B_{m,z}=g\cos\theta$",
        color="#008b8b",
        fontsize=9,
    )

    axis.add_patch(
        Arc((0.0, 0.0), 0.46, 0.46, theta1=90.0, theta2=90.0 + np.rad2deg(theta))
    )
    axis.text(0.05, 0.29, r"$\theta$", fontsize=10)
    axis.text(
        0.01,
        0.96,
        "(a) 重力投影と機体系成分",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )


def draw_tilt_traces(axis, signals: dict[str, np.ndarray]) -> None:
    time = signals["time"]
    axis.plot(time, np.rad2deg(signals["theta_true"]), color="#222222", linewidth=2.0, label="真の傾斜")
    axis.plot(time, np.rad2deg(signals["theta_gyro"]), color="#1f77b4", linewidth=1.8, label="ジャイロ積分")
    axis.plot(time, np.rad2deg(signals["theta_acc_low"]), color="#2ca02c", linewidth=1.2, alpha=0.85, label="加速度計: 低並進")
    axis.plot(
        time,
        np.rad2deg(signals["theta_acc_contaminated"]),
        color="#d62728",
        linewidth=1.2,
        alpha=0.88,
        label="加速度計: 並進を含む",
    )
    axis.axvspan(4.0, 6.0, color="#f7d7d7", alpha=0.55)
    axis.axvspan(8.2, 9.1, color="#f7d7d7", alpha=0.35)
    axis.set_xlim(0.0, TIME_STOP)
    axis.set_ylim(-13.5, 13.5)
    axis.set_xlabel(r"時刻 $t$ [s]")
    axis.set_ylabel(r"傾斜角 $\theta$ [deg]")
    axis.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.75)
    axis.legend(loc="upper left", fontsize=8)
    axis.text(
        0.99,
        0.04,
        "(b) 積分ドリフトと加速度計ノイズ",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )


def draw_contamination_sweep(axis, signals: dict[str, np.ndarray]) -> None:
    time = signals["time"]
    theta = signals["theta_true"]
    gravity_x = signals["gravity_x"]
    gravity_z = signals["gravity_z"]
    pulse = smooth_pulse(time, 4.0, 6.0, 0.18)
    colors = ("#444444", "#1f77b4", "#d95f02", "#d62728")

    for level, color in zip(CONTAMINATION_LEVELS, colors):
        theta_estimate = np.arctan2(-(gravity_x + level * pulse), gravity_z)
        error = np.rad2deg(theta_estimate - theta)
        axis.plot(
            time,
            error,
            color=color,
            linewidth=1.9,
            label=rf"$a_\parallel={level:.1f}$ m/s$^2$",
        )

    axis.axhline(0.0, color="#777777", linewidth=0.8)
    axis.axvspan(4.0, 6.0, color="#f7d7d7", alpha=0.45)
    axis.set_xlim(0.0, TIME_STOP)
    axis.set_ylim(-15.5, 1.8)
    axis.set_xlabel(r"時刻 $t$ [s]")
    axis.set_ylabel(r"傾斜推定誤差 [deg]")
    axis.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.75)
    axis.legend(loc="lower left", fontsize=8)
    axis.text(
        0.01,
        0.97,
        "(c) 並進加速度による偽の傾斜",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )


def draw_specific_force(axis, signals: dict[str, np.ndarray]) -> None:
    time = signals["time"]
    gravity_projection = signals["gravity_x"]
    forward_acceleration = signals["forward_acceleration"]
    measured_x = gravity_projection + forward_acceleration

    axis.plot(time, gravity_projection, color="#2ca02c", linewidth=1.8, label=r"重力投影 $-g\sin\theta$")
    axis.plot(time, forward_acceleration, color="#d95f02", linewidth=1.8, label=r"並進加速度 $a_\parallel$")
    axis.plot(time, measured_x, color="#d62728", linewidth=2.0, label=r"比力 $a^B_{m,x}$")
    axis.axhline(0.0, color="#777777", linewidth=0.8)
    axis.axvspan(4.0, 6.0, color="#f7d7d7", alpha=0.45)
    axis.axvspan(8.2, 9.1, color="#f7d7d7", alpha=0.30)
    axis.set_xlim(0.0, TIME_STOP)
    axis.set_ylim(-2.2, 2.4)
    axis.set_xlabel(r"時刻 $t$ [s]")
    axis.set_ylabel(r"加速度成分 [m/s$^2$]")
    axis.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.75)
    axis.legend(loc="upper left", fontsize=8)
    axis.text(
        0.99,
        0.04,
        "(d) 比力に混ざる重力投影と並進成分",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "ja" / "figures" / "flow6_accelerometer_tilt_examples.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    configure_fonts()
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 180,
            "savefig.dpi": 180,
        }
    )

    signals = simulate_signals()
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.0), constrained_layout=False)
    draw_geometry(axes[0, 0])
    draw_tilt_traces(axes[0, 1], signals)
    draw_contamination_sweep(axes[1, 0], signals)
    draw_specific_force(axes[1, 1], signals)
    finalize_figure(fig, output_path)
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
