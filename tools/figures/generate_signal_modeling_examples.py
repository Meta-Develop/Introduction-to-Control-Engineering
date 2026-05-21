#!/usr/bin/env python3
"""Generate linearization and Euler-step examples."""

from __future__ import annotations

from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("numpy is required to generate the signal-modeling figures") from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("matplotlib is required to generate the signal-modeling figures") from exc

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

T_AMBIENT = 20.0
R_TH = 2.0
C_TH = 10.0
ETA = 0.5
V_E = 2.0
T_E = T_AMBIENT + R_TH * ETA * V_E**2
VALID_V_DEVIATION = 0.4
TAU = 0.5
EULER_STEPS = (0.10, 0.60, 1.20)


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


def draw_linearization(axis) -> None:
    deviations = np.linspace(-1.6, 1.6, 401)
    exact = ETA * ((V_E + deviations) ** 2 - V_E**2) / C_TH
    linear = 2.0 * ETA * V_E * deviations / C_TH

    axis.axvspan(
        -VALID_V_DEVIATION,
        VALID_V_DEVIATION,
        color="#e8f4ff",
        alpha=0.85,
        label=rf"first-order-valid region $|\tilde{{v}}|\leq{VALID_V_DEVIATION:.1f}$ V",
    )
    axis.plot(deviations, exact, color="#1f77b4", linewidth=2.0, label="exact input contribution")
    axis.plot(deviations, linear, color="#d62728", linestyle="--", linewidth=2.0, label="first-order Taylor approximation")
    axis.axhline(0.0, color="#666666", linewidth=0.8)
    axis.axvline(0.0, color="#666666", linewidth=0.8)
    axis.set_title("Equilibrium, deviation variables, and linearization error", fontweight="bold")
    axis.set_xlabel(r"入力偏差 $\tilde{v}=v-v_e$ [V]")
    axis.set_ylabel(r"ヒータ入力寄与 $\Delta\dot{T}_{u}$ [K/s]")
    axis.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)
    axis.set_xlim(-1.6, 1.6)
    axis.set_ylim(-0.55, 0.85)
    axis.legend(loc="upper left", fontsize=8)
    axis.text(
        0.03,
        0.04,
        (
            rf"$v_e={V_E:.1f}$ V, $T_e={T_E:.1f}\,^{{\circ}}$C" "\n"
            rf"$\Delta\dot{{T}}_u\simeq(2\eta v_e/C_{{th}})\tilde{{v}}$"
            "\ncooling term $-\\tilde{T}/(R_{th}C_{th})$ omitted"
        ),
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#bbbbbb"},
    )


def euler_decay_points(step: float, stop: float) -> tuple[np.ndarray, np.ndarray]:
    times = np.arange(0.0, stop + 0.5 * step, step)
    values = (1.0 - step / TAU) ** np.arange(len(times))
    return times, values


def draw_euler_step(axis) -> None:
    stop = 3.0
    time = np.linspace(0.0, stop, 500)
    exact = np.exp(-time / TAU)
    colors = ("#1f77b4", "#ff7f0e", "#d62728")

    axis.plot(time, exact, color="#222222", linewidth=2.0, label=r"exact solution $e^{-t/\tau}$")
    for step, color in zip(EULER_STEPS, colors):
        times, values = euler_decay_points(step, stop)
        axis.step(
            times,
            values,
            where="post",
            color=color,
            linewidth=1.8,
            label=rf"Euler $h={step:.2f}$ s, $1-h/\tau={1-step/TAU:.1f}$",
        )
        axis.plot(times, values, "o", color=color, markersize=3.4)

    axis.axhline(0.0, color="#666666", linewidth=0.8)
    axis.axvline(2.0 * TAU, color="#999999", linestyle=":", linewidth=1.2)
    axis.text(
        2.0 * TAU + 0.03,
        1.05,
        r"stability condition $h<2\tau$",
        color="#555555",
        fontsize=8,
        ha="left",
        va="center",
    )
    axis.set_title("First-order ODE and step-size intuition", fontweight="bold")
    axis.set_xlabel(r"時間 $t$ [s]")
    axis.set_ylabel(r"偏差 $z(t)$ [V]")
    axis.set_xlim(0.0, stop)
    axis.set_ylim(-1.4, 1.35)
    axis.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)
    axis.legend(loc="lower left", fontsize=8)
    axis.text(
        0.98,
        0.94,
        rf"$\dot{{z}}=-z/\tau$, $\tau={TAU:.2f}$ s, $z_0=1.0$ V",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#bbbbbb"},
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = repo_root / "ja" / "figures"
    linearization_output_path = output_dir / "signal_modeling_linearization.png"
    euler_output_path = output_dir / "signal_modeling_euler_step.png"
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_fonts()
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "figure.dpi": 180,
            "savefig.dpi": 180,
        }
    )

    linearization_fig, linearization_axis = plt.subplots(
        figsize=(7.2, 4.2), constrained_layout=False
    )
    draw_linearization(linearization_axis)
    finalize_figure(linearization_fig, linearization_output_path)
    plt.close(linearization_fig)
    print(linearization_output_path)

    euler_fig, euler_axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=False)
    draw_euler_step(euler_axis)
    finalize_figure(euler_fig, euler_output_path)
    plt.close(euler_fig)
    print(euler_output_path)


if __name__ == "__main__":
    main()
