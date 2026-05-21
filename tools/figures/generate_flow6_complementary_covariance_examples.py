#!/usr/bin/env python3
"""Generate complementary-filter and covariance-limit examples."""

from __future__ import annotations

from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit(
        "numpy is required to generate the complementary-filter figure"
    ) from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit(
        "matplotlib is required to generate the complementary-filter figure"
    ) from exc

from figure_style import finalize_figure


DT = 0.02
T_FINAL = 18.0
TAU = 0.85
ALPHA = TAU / (TAU + DT)
RNG_SEED = 20260520


def simulate_signals() -> dict[str, np.ndarray | float]:
    rng = np.random.default_rng(RNG_SEED)
    time = np.arange(0.0, T_FINAL + DT, DT)

    theta_true = 0.19 * np.sin(0.52 * time) + 0.075 * np.sin(0.17 * time + 0.8)
    omega_true = np.gradient(theta_true, DT)

    gyro_bias = 0.010 + 0.006 * (1.0 - np.exp(-time / 9.0)) + 0.0025 * time / T_FINAL
    gyro_noise = 0.009 * rng.normal(size=time.size)
    omega_measured = omega_true + gyro_bias + gyro_noise

    accel_contamination = (
        0.15 * np.exp(-0.5 * ((time - 5.6) / 0.55) ** 2)
        - 0.13 * np.exp(-0.5 * ((time - 12.2) / 0.72) ** 2)
    )
    accel_noise_sigma = (
        0.023
        + 0.045 * np.exp(-0.5 * ((time - 5.6) / 0.65) ** 2)
        + 0.038 * np.exp(-0.5 * ((time - 12.2) / 0.80) ** 2)
    )
    theta_acc = theta_true + accel_contamination + accel_noise_sigma * rng.normal(
        size=time.size
    )

    theta_gyro = np.zeros_like(time)
    theta_gyro[0] = theta_true[0]
    for index in range(1, time.size):
        theta_gyro[index] = theta_gyro[index - 1] + omega_measured[index] * DT

    theta_hat = np.zeros_like(time)
    theta_hat[0] = theta_acc[0]
    for index in range(1, time.size):
        prediction = theta_hat[index - 1] + omega_measured[index] * DT
        theta_hat[index] = ALPHA * prediction + (1.0 - ALPHA) * theta_acc[index]

    r_acc = accel_noise_sigma**2 + (0.55 * np.abs(accel_contamination) + 0.015) ** 2
    q_prediction = (0.0032) ** 2
    covariance = np.zeros_like(time)
    covariance[0] = 0.050**2
    for index in range(1, time.size):
        prior_covariance = covariance[index - 1] + q_prediction
        kalman_like_gain = prior_covariance / (prior_covariance + r_acc[index])
        covariance[index] = (1.0 - kalman_like_gain) * prior_covariance

    return {
        "time": time,
        "theta_true": theta_true,
        "theta_acc": theta_acc,
        "theta_gyro": theta_gyro,
        "theta_hat": theta_hat,
        "accel_contamination": accel_contamination,
        "covariance": covariance,
    }


def add_panel_label(
    axis: plt.Axes,
    label: str,
    *,
    x: float = 0.02,
    y: float = 0.96,
    ha: str = "left",
    va: str = "top",
) -> None:
    axis.text(
        x,
        y,
        label,
        transform=axis.transAxes,
        ha=ha,
        va=va,
        fontsize=10,
        fontweight="bold",
        clip_on=False,
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": "#bbbbbb",
            "alpha": 0.95,
        },
    )


def draw_box(axis: plt.Axes, xy: tuple[float, float], text: str, width: float) -> None:
    x_coord, y_coord = xy
    axis.add_patch(
        Rectangle(
            (x_coord, y_coord),
            width,
            0.16,
            facecolor="#f7fbff",
            edgecolor="#4477AA",
            linewidth=1.5,
            zorder=2,
        )
    )
    axis.text(
        x_coord + width / 2.0,
        y_coord + 0.08,
        text,
        ha="center",
        va="center",
        fontsize=8.8,
        zorder=3,
    )


def draw_arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str | None = None,
    *,
    color: str = "#333333",
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.5,
            color=color,
            shrinkA=3,
            shrinkB=3,
        )
    )
    if label is not None:
        axis.text(
            (start[0] + end[0]) / 2.0,
            (start[1] + end[1]) / 2.0 + 0.04,
            label,
            ha="center",
            va="bottom",
            fontsize=8.4,
            color=color,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.4},
        )


def draw_signal_path(axis: plt.Axes) -> None:
    axis.set_axis_off()
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    add_panel_label(axis, "(a) fixed complementary-filter signal path")

    draw_box(axis, (0.06, 0.66), r"gyro rate $\omega_m$", 0.23)
    draw_box(axis, (0.39, 0.66), r"integrate $\hat{\theta}^-_k$", 0.25)
    draw_box(axis, (0.72, 0.66), r"$\alpha\hat{\theta}^-_k$", 0.20)
    draw_box(axis, (0.06, 0.28), r"accel tilt $\theta_{\rm acc}$", 0.23)
    draw_box(axis, (0.39, 0.28), r"$(1-\alpha)\theta_{\rm acc}$", 0.25)
    draw_box(axis, (0.74, 0.44), r"sum $\hat{\theta}_k$", 0.20)

    draw_arrow(axis, (0.29, 0.74), (0.39, 0.74))
    draw_arrow(axis, (0.64, 0.74), (0.72, 0.74), r"slow drift")
    draw_arrow(axis, (0.29, 0.36), (0.39, 0.36))
    draw_arrow(axis, (0.64, 0.36), (0.74, 0.48), r"fast correction")
    draw_arrow(axis, (0.92, 0.66), (0.86, 0.60))

    axis.text(
        0.50,
        0.08,
        rf"$\alpha=\tau/(\tau+\Delta t)$,  $\tau={TAU:.2f}$ s,  $\Delta t={DT:.2f}$ s",
        ha="center",
        va="center",
        fontsize=9.0,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.95},
    )


def draw_traces(axis: plt.Axes, data: dict[str, np.ndarray | float]) -> None:
    time = data["time"]
    assert isinstance(time, np.ndarray)
    theta_true = data["theta_true"]
    theta_acc = data["theta_acc"]
    theta_gyro = data["theta_gyro"]
    theta_hat = data["theta_hat"]
    assert isinstance(theta_true, np.ndarray)
    assert isinstance(theta_acc, np.ndarray)
    assert isinstance(theta_gyro, np.ndarray)
    assert isinstance(theta_hat, np.ndarray)

    add_panel_label(axis, "(b) before/after filtering")
    axis.plot(time, theta_true, color="#222222", linewidth=2.1, label="true angle")
    axis.plot(time, theta_gyro, color="#D55E00", linewidth=1.5, linestyle="--", label="gyro integration")
    axis.plot(time, theta_acc, color="#56B4E9", linewidth=0.9, alpha=0.48, label="accelerometer tilt")
    axis.plot(time, theta_hat, color="#009E73", linewidth=2.0, label="complementary filter")
    axis.set_xlim(0.0, T_FINAL)
    axis.set_ylim(-0.35, 0.38)
    axis.set_ylabel(r"Angle $\theta$ [rad]")
    axis.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.75)
    axis.legend(loc="lower left", ncols=2, frameon=True)


def draw_uncertainty_band(axis: plt.Axes, data: dict[str, np.ndarray | float]) -> None:
    time = data["time"]
    theta_true = data["theta_true"]
    theta_hat = data["theta_hat"]
    contamination = data["accel_contamination"]
    covariance = data["covariance"]
    assert isinstance(time, np.ndarray)
    assert isinstance(theta_true, np.ndarray)
    assert isinstance(theta_hat, np.ndarray)
    assert isinstance(contamination, np.ndarray)
    assert isinstance(covariance, np.ndarray)

    error = theta_hat - theta_true
    sigma = np.sqrt(np.maximum(covariance, 0.0))

    add_panel_label(
        axis,
        "(c) residual error band",
        y=1.035,
        va="bottom",
    )
    axis.fill_between(
        time,
        -2.0 * sigma,
        2.0 * sigma,
        color="#CC79A7",
        alpha=0.24,
        label=r"covariance band $\pm2\sigma$",
    )
    axis.plot(time, error, color="#0072B2", linewidth=1.7, label=r"$\hat{\theta}-\theta$")
    for center in (5.6, 12.2):
        axis.axvspan(center - 0.8, center + 0.8, color="#F0E442", alpha=0.18)
    axis.plot(
        time,
        0.45 * contamination,
        color="#D55E00",
        linewidth=1.1,
        linestyle=":",
        label="scaled accel contamination",
    )
    axis.axhline(0.0, color="#666666", linewidth=0.8)
    axis.set_xlim(0.0, T_FINAL)
    axis.set_ylim(-0.17, 0.17)
    axis.set_xlabel(r"Time $t$ [s]")
    axis.set_ylabel(r"Angle error [rad]")
    axis.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.75)
    axis.legend(loc="upper right", frameon=True)


def covariance_ellipse(
    center: tuple[float, float],
    covariance: np.ndarray,
    *,
    scale: float,
    label: str,
    color: str,
) -> Ellipse:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2.0 * scale * np.sqrt(eigenvalues)
    return Ellipse(
        xy=center,
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        edgecolor=color,
        linewidth=2.0,
        alpha=0.24,
        label=label,
    )


def draw_covariance_ellipse(axis: plt.Axes) -> None:
    quiet_covariance = np.array([[0.018**2, -0.00010], [-0.00010, 0.010**2]])
    accelerating_covariance = np.array([[0.055**2, -0.00045], [-0.00045, 0.016**2]])

    add_panel_label(
        axis,
        "(d) angle-bias covariance",
        y=1.035,
        va="bottom",
    )
    axis.add_patch(
        covariance_ellipse(
            (0.0, 0.0),
            quiet_covariance,
            scale=2.0,
            label="nearly steady",
            color="#009E73",
        )
    )
    axis.add_patch(
        covariance_ellipse(
            (0.0, 0.0),
            accelerating_covariance,
            scale=2.0,
            label="during acceleration",
            color="#D55E00",
        )
    )
    axis.axhline(0.0, color="#777777", linewidth=0.8)
    axis.axvline(0.0, color="#777777", linewidth=0.8)
    axis.set_xlim(-0.15, 0.15)
    axis.set_ylim(-0.050, 0.050)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(r"Angle error [rad]")
    axis.set_ylabel(r"Gyro bias error [rad/s]")
    axis.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.75)
    axis.legend(loc="upper right", frameon=True)


def draw_figure() -> plt.Figure:
    data = simulate_signals()
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 7.4), constrained_layout=False)
    draw_signal_path(axes[0, 0])
    draw_traces(axes[0, 1], data)
    draw_uncertainty_band(axes[1, 0], data)
    draw_covariance_ellipse(axes[1, 1])
    return figure


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "ja" / "figures" / "flow6_complementary_covariance_examples.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 180,
            "savefig.dpi": 180,
        }
    )

    figure = draw_figure()
    finalize_figure(figure, output_path)
    plt.close(figure)
    print(output_path)


if __name__ == "__main__":
    main()
