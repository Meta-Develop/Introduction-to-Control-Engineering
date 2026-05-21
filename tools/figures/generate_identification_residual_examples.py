#!/usr/bin/env python3
"""Generate identification and residual robustness examples."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("matplotlib is required to generate the identification figure") from exc

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

TIME_STEP = 0.02
TIME_STOP = 15.0
TRAIN_END = 8.0
TRUE_GAIN = 9.4  # rad/s/V
TRUE_TIME_CONSTANT = 0.48  # s
TRUE_DEAD_VOLTAGE = 0.18  # V
SENSOR_NOISE_STD = 0.16  # rad/s
DISTURBANCE_START = 10.1
DISTURBANCE_END = 11.3
DISTURBANCE_ACCEL = -3.4  # rad/s^2
RANDOM_SEED = 20260520

COLORS = {
    "train": "#6f6f6f",
    "valid": "#fff3bf",
    "input": "#4c78a8",
    "measured": "#222222",
    "model": "#1f77b4",
    "residual": "#d62728",
    "band": "#f2b8b5",
    "zero": "#777777",
}


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


def smooth_dead_zone(voltage: np.ndarray | float) -> np.ndarray:
    voltage_array = np.asarray(voltage, dtype=float)
    return np.sign(voltage_array) * np.maximum(np.abs(voltage_array) - TRUE_DEAD_VOLTAGE, 0.0)


def voltage_command(time: np.ndarray) -> np.ndarray:
    voltage = np.zeros_like(time)
    segments = (
        (0.40, 1.70, 1.2),
        (1.70, 3.10, 2.0),
        (3.10, 4.40, 0.7),
        (4.40, 5.80, -1.3),
        (5.80, 7.20, 1.8),
        (7.20, 8.00, 0.4),
        (8.00, 9.20, 1.5),
        (9.20, 10.35, 2.2),
        (10.35, 11.70, -0.9),
        (11.70, 13.10, 1.0),
        (13.10, 15.00, 0.2),
    )
    for start, stop, level in segments:
        voltage[(time >= start) & (time < stop)] = level

    voltage += 0.10 * np.sin(2.0 * np.pi * time / 2.6)
    voltage += 0.05 * np.sin(2.0 * np.pi * time / 0.74)
    return voltage


def simulate_measurement() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RANDOM_SEED)
    time = np.arange(0.0, TIME_STOP + TIME_STEP, TIME_STEP)
    voltage = voltage_command(time)
    omega_true = np.zeros_like(time)

    for index in range(len(time) - 1):
        disturbance = (
            DISTURBANCE_ACCEL
            if DISTURBANCE_START <= time[index] <= DISTURBANCE_END
            else 0.0
        )
        nonlinear_gain = 1.0 - 0.065 * np.tanh(abs(omega_true[index]) / 13.0)
        effective_voltage = smooth_dead_zone(voltage[index])
        omega_dot = (
            -omega_true[index] / TRUE_TIME_CONSTANT
            + nonlinear_gain * TRUE_GAIN * effective_voltage / TRUE_TIME_CONSTANT
            + disturbance
        )
        omega_true[index + 1] = omega_true[index] + TIME_STEP * omega_dot

    ripple = 0.05 * np.sin(2.0 * np.pi * time * 5.5)
    omega_measured = omega_true + ripple + rng.normal(0.0, SENSOR_NOISE_STD, size=time.size)
    return time, voltage, omega_true, omega_measured


def fit_first_order_model(
    time: np.ndarray, voltage: np.ndarray, omega: np.ndarray
) -> tuple[float, float, float, float, float]:
    train = (time >= 0.4) & (time <= TRAIN_END)
    y_next = omega[1:][train[:-1]]
    phi = np.column_stack([omega[:-1][train[:-1]], voltage[:-1][train[:-1]], np.ones(np.count_nonzero(train[:-1]))])
    alpha, beta, bias = np.linalg.lstsq(phi, y_next, rcond=None)[0]
    alpha = float(np.clip(alpha, 0.05, 0.999))
    beta = float(beta)
    bias = float(bias)
    time_constant = -TIME_STEP / np.log(alpha)
    gain = beta / (1.0 - alpha)
    return alpha, beta, bias, gain, time_constant


def simulate_identified_model(
    voltage: np.ndarray, initial_speed: float, alpha: float, beta: float, bias: float
) -> np.ndarray:
    omega_model = np.zeros_like(voltage)
    omega_model[0] = initial_speed
    for index in range(len(voltage) - 1):
        omega_model[index + 1] = alpha * omega_model[index] + beta * voltage[index] + bias
    return omega_model


def residual_statistics(time: np.ndarray, residual: np.ndarray) -> tuple[float, float, float]:
    validation = time > TRAIN_END
    residual_validation = residual[validation]
    rms = float(np.sqrt(np.mean(residual_validation**2)))
    maximum = float(np.max(np.abs(residual_validation)))
    sigma = float(np.std(residual_validation))
    return rms, maximum, sigma


def add_panel_label(axis, label: str) -> None:
    axis.text(
        0.018,
        0.955,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.92},
    )


def shade_split(axis) -> None:
    axis.axvspan(0.0, TRAIN_END, color="#eeeeee", alpha=0.62, linewidth=0.0)
    axis.axvspan(TRAIN_END, TIME_STOP, color=COLORS["valid"], alpha=0.44, linewidth=0.0)
    axis.axvline(TRAIN_END, color="#7a5c00", linewidth=1.2, linestyle="--")


def style_time_axis(axis) -> None:
    axis.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.72)
    axis.set_xlim(0.0, TIME_STOP)


def plot_input(axis, time: np.ndarray, voltage: np.ndarray) -> None:
    shade_split(axis)
    axis.step(time, voltage, where="post", color=COLORS["input"], linewidth=1.7, label="入力電圧")
    axis.text(2.55, 2.55, "同定用データ", fontsize=8.5, color="#555555")
    axis.text(10.1, 2.55, "検証用データ", fontsize=8.5, color="#7a5c00")
    axis.set_ylabel(r"入力電圧 $v$ [V]")
    axis.set_ylim(-1.8, 2.9)
    axis.legend(loc="lower right", fontsize=8.0)
    style_time_axis(axis)
    add_panel_label(axis, "(a) Train/validation")


def plot_fit(
    axis,
    time: np.ndarray,
    omega_measured: np.ndarray,
    omega_model: np.ndarray,
    gain: float,
    time_constant: float,
) -> None:
    shade_split(axis)
    axis.plot(time, omega_measured, color=COLORS["measured"], linewidth=1.2, alpha=0.86, label="測定速度")
    axis.plot(time, omega_model, color=COLORS["model"], linewidth=2.0, label="一次モデル")
    axis.text(
        0.55,
        -9.7,
        rf"$\hat K={gain:.2f}\,\mathrm{{rad/s/V}},\quad \hat\tau={time_constant:.2f}\,\mathrm{{s}}$",
        fontsize=8.6,
        color="#1f4e79",
    )
    axis.set_ylabel(r"角速度 $\omega$ [rad/s]")
    axis.set_ylim(-12.4, 22.5)
    axis.legend(loc="upper right", fontsize=8.0)
    style_time_axis(axis)
    add_panel_label(axis, "(b) Model fit")


def plot_residual(
    axis,
    time: np.ndarray,
    residual: np.ndarray,
    rms: float,
    maximum: float,
    sigma: float,
) -> None:
    shade_split(axis)
    axis.axhline(0.0, color=COLORS["zero"], linewidth=0.9)
    axis.fill_between(time, -2.0 * sigma, 2.0 * sigma, color=COLORS["band"], alpha=0.32, label=r"$\pm2\sigma$ 残差帯")
    axis.plot(time, residual, color=COLORS["residual"], linewidth=1.35, label="測定 - モデル")
    axis.axvspan(DISTURBANCE_START, DISTURBANCE_END, color="#b5651d", alpha=0.14, linewidth=0.0)
    axis.text(
        DISTURBANCE_START + 0.06,
        -2.55,
        "未測定負荷",
        fontsize=8.0,
        color="#704214",
    )
    axis.text(
        8.25,
        3.35,
        rf"検証 RMS={rms:.2f} rad/s, 最大={maximum:.2f} rad/s",
        fontsize=8.2,
        color="#7f1d1d",
    )
    axis.set_xlabel(r"時間 $t$ [s]")
    axis.set_ylabel(r"残差 $e_v$ [rad/s]")
    axis.set_ylim(-4.4, 4.4)
    axis.legend(loc="lower right", fontsize=8.0)
    style_time_axis(axis)
    add_panel_label(axis, "(c) Validation residual")


def plot_residual_scatter(axis, time: np.ndarray, voltage: np.ndarray, residual: np.ndarray, sigma: float) -> None:
    train = time <= TRAIN_END
    validation = time > TRAIN_END
    axis.axhline(0.0, color=COLORS["zero"], linewidth=0.9)
    axis.axhspan(-2.0 * sigma, 2.0 * sigma, color=COLORS["band"], alpha=0.30)
    axis.scatter(voltage[train], residual[train], s=10, color="#808080", alpha=0.45, label="同定区間")
    axis.scatter(voltage[validation], residual[validation], s=11, color=COLORS["residual"], alpha=0.55, label="検証区間")
    axis.set_xlabel(r"入力電圧 $v$ [V]")
    axis.set_ylabel(r"残差 $e_v$ [rad/s]")
    axis.set_xlim(-1.7, 2.45)
    axis.set_ylim(-4.4, 4.4)
    axis.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.72)
    axis.legend(loc="lower right", fontsize=8.0)
    axis.text(
        -1.52,
        -3.65,
        "入力と残差が無関係なら白色に近い",
        fontsize=8.1,
        color="#555555",
    )
    add_panel_label(axis, "(d) Residual structure")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "ja" / "figures" / "identification_residual_examples.png"
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

    time, voltage, _omega_true, omega_measured = simulate_measurement()
    alpha, beta, bias, gain, time_constant = fit_first_order_model(time, voltage, omega_measured)
    omega_model = simulate_identified_model(voltage, omega_measured[0], alpha, beta, bias)
    residual = omega_measured - omega_model
    rms, maximum, sigma = residual_statistics(time, residual)

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.8), constrained_layout=False)
    plot_input(axes[0, 0], time, voltage)
    plot_fit(axes[0, 1], time, omega_measured, omega_model, gain, time_constant)
    plot_residual(axes[1, 0], time, residual, rms, maximum, sigma)
    plot_residual_scatter(axes[1, 1], time, voltage, residual, sigma)

    finalize_figure(fig, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
