#!/usr/bin/env python3
"""Generate tank and pendulum modeling examples."""

from __future__ import annotations

from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("numpy is required to generate the tank-pendulum figure") from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc, Circle, FancyArrowPatch, Rectangle
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("matplotlib is required to generate the tank-pendulum figure") from exc

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

TANK_AREA = 0.50
TANK_HEIGHT_E = 0.64
TANK_INPUT_STEP = 0.008
TANK_K_VALUES = (0.06, 0.10)
TANK_TIME = np.linspace(0.0, 90.0, 500)

PENDULUM_DAMPING = 0.25
PENDULUM_TIME = np.linspace(0.0, 18.0, 900)


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


def add_arrow(
    axis,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str = "",
    color: str = "#333333",
    text_offset: tuple[float, float] = (0.0, 0.0),
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.6,
        color=color,
        shrinkA=4,
        shrinkB=4,
    )
    axis.add_patch(arrow)
    if label:
        axis.text(
            (start[0] + end[0]) / 2 + text_offset[0],
            (start[1] + end[1]) / 2 + text_offset[1],
            label,
            ha="center",
            va="center",
            fontsize=8.8,
            color=color,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none"},
        )


def draw_tank_schematic(axis) -> None:
    axis.set_axis_off()
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_title("単一タンクの保存則", loc="left", fontweight="bold")

    axis.add_patch(Rectangle((0.28, 0.20), 0.44, 0.56, facecolor="#f7f7f7", edgecolor="#333333", linewidth=1.5))
    axis.add_patch(Rectangle((0.30, 0.20), 0.40, 0.34, facecolor="#b9def5", edgecolor="none", alpha=0.95))
    axis.plot([0.30, 0.70], [0.54, 0.54], color="#1f77b4", linewidth=1.6)
    axis.plot([0.30, 0.30], [0.54, 0.20], color="#1f77b4", linewidth=0.8, linestyle=":")
    axis.text(0.73, 0.54, r"水位 $h$ [m]", ha="left", va="center", fontsize=9.5)
    axis.text(0.50, 0.13, r"断面積 $A_t$ [m$^2$]", ha="center", va="center", fontsize=9.5)
    axis.text(0.50, 0.38, r"状態 $x=h$", ha="center", va="center", fontsize=10, color="#0b4f7a")

    add_arrow(axis, (0.10, 0.86), (0.38, 0.76), r"流入 $q_{in}$ [m$^3$/s]", "#ff7f0e", (0.02, 0.03))
    add_arrow(axis, (0.89, 0.23), (0.72, 0.23), "", "#d62728")
    axis.text(0.78, 0.31, r"流出 $q_{out}$", ha="center", va="center", fontsize=8.8, color="#d62728")
    add_arrow(axis, (0.18, 0.47), (0.28, 0.47), r"漏れ・雨 $d_q$", "#9467bd", (0.00, 0.06))

    axis.text(
        0.05,
        0.03,
        r"$A_t\dot{h}=q_{in}+d_q-k_o\sqrt{h},\quad y=h$",
        ha="left",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#bbbbbb"},
    )


def draw_tank_linear_response(axis) -> None:
    colors = ("#1f77b4", "#d62728")
    for k_out, color in zip(TANK_K_VALUES, colors):
        alpha = k_out / (2.0 * TANK_AREA * np.sqrt(TANK_HEIGHT_E))
        final_delta = TANK_INPUT_STEP / (TANK_AREA * alpha)
        height = TANK_HEIGHT_E + final_delta * (1.0 - np.exp(-alpha * TANK_TIME))
        axis.plot(
            TANK_TIME,
            height,
            color=color,
            linewidth=2.0,
            label=rf"$k_o={k_out:.2f}$, $\tau={1.0/alpha:.1f}$ s",
        )
        axis.axhline(TANK_HEIGHT_E + final_delta, color=color, linewidth=0.9, linestyle=":")

    axis.axhline(TANK_HEIGHT_E, color="#666666", linewidth=0.8)
    axis.set_title("流出係数による水位応答の変化", fontweight="bold")
    axis.set_xlabel(r"時間 $t$ [s]")
    axis.set_ylabel(r"水位 $h(t)$ [m]")
    axis.set_xlim(0.0, 90.0)
    axis.set_ylim(0.60, 0.88)
    axis.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)
    axis.legend(loc="lower right", fontsize=8.2)
    axis.text(
        0.03,
        0.94,
        rf"$A_t={TANK_AREA:.2f}$ m$^2$, $h_e={TANK_HEIGHT_E:.2f}$ m, $\Delta q={TANK_INPUT_STEP:.3f}$ m$^3$/s",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": "#bbbbbb"},
    )


def draw_pendulum_schematic(axis) -> None:
    axis.set_axis_off()
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_title("トルク入力をもつ単振子", loc="left", fontweight="bold")

    pivot = np.array([0.50, 0.80])
    length = 0.46
    theta = np.deg2rad(28.0)
    bob = pivot + length * np.array([np.sin(theta), -np.cos(theta)])

    axis.plot([pivot[0], pivot[0]], [pivot[1], pivot[1] - length], color="#888888", linestyle=":", linewidth=1.2)
    axis.plot([pivot[0], bob[0]], [pivot[1], bob[1]], color="#333333", linewidth=2.0)
    axis.add_patch(Circle(tuple(pivot), 0.025, facecolor="#333333", edgecolor="#333333"))
    axis.add_patch(Circle(tuple(bob), 0.055, facecolor="#edf7ed", edgecolor="#333333", linewidth=1.4))
    axis.text(bob[0] + 0.07, bob[1], r"質量 $m$ [kg]", ha="left", va="center", fontsize=9.5)
    axis.text(0.43, 0.52, r"長さ $l$ [m]", ha="right", va="center", fontsize=9.5)

    arc = Arc(tuple(pivot), 0.22, 0.22, angle=0, theta1=242, theta2=270, color="#1f77b4", linewidth=1.5)
    axis.add_patch(arc)
    axis.text(0.57, 0.66, r"$\theta$ [rad]", ha="left", va="center", fontsize=9.5, color="#1f77b4")
    add_arrow(axis, (0.69, 0.73), (0.62, 0.80), r"入力 $u$ [N m]", "#ff7f0e", (0.03, 0.04))
    add_arrow(axis, (bob[0], bob[1] - 0.02), (bob[0], bob[1] - 0.20), r"$mg$", "#d62728", (0.05, 0.0))
    add_arrow(axis, (0.27, 0.35), (0.39, 0.43), r"減衰 $b\omega$", "#9467bd", (-0.03, 0.05))

    axis.text(
        0.05,
        0.06,
        r"$\dot{\theta}=\omega,\quad J\dot{\omega}=u+d_\tau-b\omega-mgl\,\sin\theta$",
        ha="left",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#bbbbbb"},
    )


def simulate_pendulum(theta0: float, nonlinear: bool) -> np.ndarray:
    dt = PENDULUM_TIME[1] - PENDULUM_TIME[0]
    state = np.array([theta0, 0.0], dtype=float)
    response = np.empty_like(PENDULUM_TIME)
    for index, _time in enumerate(PENDULUM_TIME):
        response[index] = state[0]
        theta, omega = state
        restoring = np.sin(theta) if nonlinear else theta
        derivative = np.array([omega, -restoring - PENDULUM_DAMPING * omega])
        state = state + dt * derivative
    return response


def draw_pendulum_response(axis) -> None:
    small_nonlinear = simulate_pendulum(0.25, nonlinear=True)
    small_linear = simulate_pendulum(0.25, nonlinear=False)
    large_nonlinear = simulate_pendulum(1.60, nonlinear=True)
    large_linear = simulate_pendulum(1.60, nonlinear=False)

    axis.plot(PENDULUM_TIME, small_nonlinear, color="#1f77b4", linewidth=2.0, label=r"非線形 $\theta_0=0.25$")
    axis.plot(PENDULUM_TIME, small_linear, color="#1f77b4", linewidth=1.7, linestyle="--", label=r"線形近似 $\theta_0=0.25$")
    axis.plot(PENDULUM_TIME, large_nonlinear, color="#d62728", linewidth=2.0, label=r"非線形 $\theta_0=1.60$")
    axis.plot(PENDULUM_TIME, large_linear, color="#d62728", linewidth=1.7, linestyle="--", label=r"線形近似 $\theta_0=1.60$")
    axis.axhline(0.0, color="#666666", linewidth=0.8)
    axis.set_title("単振子の局所線形近似の有効範囲", fontweight="bold")
    axis.set_xlabel(r"時間 $t$ [s]")
    axis.set_ylabel(r"角度 $\theta(t)$ [rad]")
    axis.set_xlim(0.0, 18.0)
    axis.set_ylim(-1.75, 1.75)
    axis.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)
    axis.legend(loc="upper right", fontsize=7.8)
    axis.text(
        0.03,
        0.06,
        rf"$J=1$, $mgl=1$, $b={PENDULUM_DAMPING:.2f}$; "
        r"小角度では $\sin\theta\simeq\theta$",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.8,
        bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": "#bbbbbb"},
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "ja" / "figures" / "tank_pendulum_modeling_examples.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    configure_fonts()
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "figure.dpi": 180,
            "savefig.dpi": 180,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.0), constrained_layout=False)
    draw_tank_schematic(axes[0, 0])
    draw_tank_linear_response(axes[0, 1])
    draw_pendulum_schematic(axes[1, 0])
    draw_pendulum_response(axes[1, 1])

    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.075, top=0.98, hspace=0.42, wspace=0.28)
    finalize_figure(fig, output_path, layout="none")
    print(output_path)


if __name__ == "__main__":
    main()
