#!/usr/bin/env python3
"""Generate Nyquist, root-locus, and sensitivity examples for frequency design."""

from __future__ import annotations

import itertools
import math
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("numpy is required to generate the frequency-design figure") from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("matplotlib is required to generate the frequency-design figure") from exc

from figure_style import finalize_figure


NYQUIST_K_VALUES = (0.6, 1.0, 2.0)
ROOT_K_LIMIT = 48.0
ROOT_K_MAX = 95.0
SENSITIVITY_K_VALUES = (0.3, 1.0, 3.0)


PANEL_BOX = {
    "boxstyle": "round,pad=0.25",
    "facecolor": "white",
    "alpha": 0.9,
    "edgecolor": "#bbbbbb",
}


def nyquist_omega() -> np.ndarray:
    negative = -np.geomspace(80.0, 0.01, 700)
    positive = np.geomspace(0.01, 80.0, 700)
    return np.concatenate([negative, np.array([0.0]), positive])


def unstable_first_order_loop(omega: np.ndarray, gain: float) -> np.ndarray:
    return gain / (1j * omega - 1.0)


def roots_for_gain(gain: float) -> np.ndarray:
    return np.roots([1.0, 6.0, 8.0, gain])


def sorted_root_locus(gains: np.ndarray) -> np.ndarray:
    roots = np.empty((len(gains), 3), dtype=complex)
    roots[0] = np.sort_complex(roots_for_gain(float(gains[0])))

    for index, gain in enumerate(gains[1:], start=1):
        candidates = roots_for_gain(float(gain))
        previous = roots[index - 1]
        best_order = min(
            itertools.permutations(range(3)),
            key=lambda order: sum(abs(candidates[order[i]] - previous[i]) for i in range(3)),
        )
        roots[index] = candidates[list(best_order)]

    return roots


def sensitivity_response(omega: np.ndarray, gain: float) -> tuple[np.ndarray, np.ndarray]:
    s = 1j * omega
    loop = gain / (s * (s + 1.0))
    sensitivity = 1.0 / (1.0 + loop)
    complementary = loop / (1.0 + loop)
    return sensitivity, complementary


def db(values: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(values), 1.0e-12))


def add_curve_arrow(ax, x_values, y_values, index: int, color: str) -> None:
    ax.annotate(
        "",
        xy=(x_values[index + 18], y_values[index + 18]),
        xytext=(x_values[index], y_values[index]),
        arrowprops={"arrowstyle": "->", "color": color, "linewidth": 1.2},
    )


def add_panel_label(ax, label: str, text: str) -> None:
    ax.text(
        0.03,
        0.97,
        f"{label} {text}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        weight="bold",
        bbox=PANEL_BOX,
        zorder=8,
    )


def plot_nyquist(ax) -> None:
    omega = nyquist_omega()
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c")
    labels = (
        r"$K=0.6$: 不安定",
        r"$K=1.0$: 限界",
        r"$K=2.0$: 安定",
    )

    for gain, color, label in zip(NYQUIST_K_VALUES, colors, labels):
        response = unstable_first_order_loop(omega, gain)
        ax.plot(response.real, response.imag, color=color, linewidth=1.8, label=label)
        start = response[0]
        ax.plot(start.real, start.imag, marker="o", color=color, markersize=3.0)
        arrow_index = 470 if gain < 1.0 else 430
        add_curve_arrow(ax, response.real, response.imag, arrow_index, color)

    response = unstable_first_order_loop(omega, 2.0)
    add_curve_arrow(ax, response.real, response.imag, 865, "#2ca02c")

    ax.scatter([-1.0], [0.0], marker="x", s=90, color="#d62728", linewidths=2.0, zorder=5)
    ax.text(-1.0, -0.14, r"$-1$", color="#d62728", ha="center", va="top", fontsize=9)
    ax.annotate(
        r"$\omega:-\infty\to+\infty$",
        xy=(-1.92, 0.34),
        xytext=(-2.28, 0.78),
        arrowprops={"arrowstyle": "->", "color": "#444444", "linewidth": 1.0},
        color="#444444",
        fontsize=8,
    )
    ax.axhline(0.0, color="#777777", linewidth=0.8)
    ax.axvline(0.0, color="#777777", linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-2.35, 0.45)
    ax.set_ylim(-1.25, 1.25)
    ax.set_xlabel(r"実部 $\Re L(j\omega)$ [-]")
    ax.set_ylabel(r"虚部 $\Im L(j\omega)$ [-]")
    ax.set_title(r"(a) ナイキスト曲線, $L_K(s)=K/(s-1)$")
    ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)
    ax.legend(loc="upper right", fontsize=7.2)
    add_panel_label(ax, "(a)", r"Nyquist: $K/(s-1)$")
    ax.text(
        0.03,
        0.05,
        r"$P=1$" "\n" r"$K>1$: $N=-1$, $Z=N+P=0$",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox=PANEL_BOX,
    )


def plot_root_locus(ax) -> None:
    gains = np.linspace(0.0, ROOT_K_MAX, 760)
    roots = sorted_root_locus(gains)
    stable = gains <= ROOT_K_LIMIT

    ax.axvspan(-5.2, 0.0, color="#e8f2fb", alpha=0.65, label="安定半平面")
    ax.axvspan(0.0, 1.4, color="#fdeaea", alpha=0.45, label="不安定半平面")
    ax.axvline(0.0, color="#777777", linewidth=0.9)
    ax.axhline(0.0, color="#777777", linewidth=0.8)

    for branch in range(roots.shape[1]):
        ax.plot(
            roots[stable, branch].real,
            roots[stable, branch].imag,
            color="#1f77b4",
            linewidth=1.9,
        )
        ax.plot(
            roots[~stable, branch].real,
            roots[~stable, branch].imag,
            color="#d62728",
            linewidth=1.7,
            linestyle="--",
        )

    open_loop_poles = np.array([0.0, -2.0, -4.0])
    ax.scatter(open_loop_poles, np.zeros_like(open_loop_poles), marker="x", s=70, color="#222222")
    crossing = math.sqrt(8.0)
    ax.scatter([0.0, 0.0], [crossing, -crossing], marker="o", s=45, color="#d62728", zorder=5)
    ax.text(0.10, crossing, r"$K=48$", color="#d62728", va="bottom", fontsize=8)
    ax.text(0.10, -crossing, r"$K=48$", color="#d62728", va="top", fontsize=8)
    ax.text(-4.55, 3.15, r"安定: $0<K<48$", color="#174a7c", fontsize=8)
    ax.text(0.48, 0.82, r"不安定: $K>48$", color="#b22222", fontsize=8, rotation=90)

    for gain in (8.0, 24.0):
        root_set = roots_for_gain(gain)
        root = root_set[np.argmax(root_set.imag)]
        ax.text(root.real + 0.08, root.imag + 0.08, rf"$K={gain:g}$", fontsize=8)

    ax.annotate(
        "",
        xy=(roots[245, 2].real, roots[245, 2].imag),
        xytext=(roots[190, 2].real, roots[190, 2].imag),
        arrowprops={"arrowstyle": "->", "color": "#1f77b4", "linewidth": 1.2},
    )
    ax.annotate(
        "",
        xy=(roots[170, 1].real, roots[170, 1].imag),
        xytext=(roots[115, 1].real, roots[115, 1].imag),
        arrowprops={"arrowstyle": "->", "color": "#1f77b4", "linewidth": 1.2},
    )
    ax.set_xlim(-4.8, 1.25)
    ax.set_ylim(-3.8, 3.8)
    ax.set_xlabel(r"実部 $\sigma$ [1/s]")
    ax.set_ylabel(r"虚部 [rad/s]")
    ax.set_title(r"(b) 根軌跡, $G(s)=1/\{s(s+2)(s+4)\}$")
    ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)
    add_panel_label(ax, "(b)", r"Root locus: $1/\{s(s+2)(s+4)\}$")
    ax.text(
        0.03,
        0.05,
        r"$s^3+6s^2+8s+K=0$" "\n" r"安定範囲 $0<K<48$",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox=PANEL_BOX,
    )


def plot_sensitivity(ax) -> None:
    omega = np.logspace(-2.0, 2.0, 1000)
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c")
    ms_lines = []
    peak_points = []

    for gain, color in zip(SENSITIVITY_K_VALUES, colors):
        sensitivity, complementary = sensitivity_response(omega, gain)
        sensitivity_db = db(sensitivity)
        complementary_db = db(complementary)
        ax.semilogx(
            omega,
            sensitivity_db,
            color=color,
            linewidth=1.9,
            label=rf"$|S|$, $K={gain:g}$",
        )
        ax.semilogx(
            omega,
            complementary_db,
            color=color,
            linewidth=1.6,
            linestyle="--",
            label=rf"$|T|$, $K={gain:g}$",
        )
        peak_index = int(np.argmax(sensitivity_db))
        peak_points.append((omega[peak_index], sensitivity_db[peak_index], color))
        ms_lines.append(rf"$K={gain:g}$: $M_s={sensitivity_db[peak_index]:.1f}$ dB")

    ax.axhline(0.0, color="#777777", linewidth=0.8)
    ax.axvspan(0.01, 0.18, color="#e8f2fb", alpha=0.55)
    ax.axvspan(6.0, 100.0, color="#f3f3f3", alpha=0.55)
    for peak_omega, peak_db, color in peak_points:
        ax.scatter([peak_omega], [peak_db], s=24, color=color, zorder=5)
    ax.annotate(
        r"$K\uparrow$: 低周波 $|S|\downarrow$",
        xy=(0.055, -30.0),
        xytext=(0.035, -18.0),
        arrowprops={"arrowstyle": "->", "color": "#174a7c", "linewidth": 1.0},
        color="#174a7c",
        fontsize=8,
    )
    ax.annotate(
        r"$K\uparrow$: $M_s\uparrow$, $|T|$帯域$\uparrow$",
        xy=(1.55, 5.0),
        xytext=(0.22, 4.7),
        arrowprops={"arrowstyle": "->", "color": "#444444", "linewidth": 1.0},
        color="#444444",
        fontsize=8,
        bbox=PANEL_BOX,
    )
    ax.text(0.014, -37.0, "外乱抑制\n低周波域", fontsize=8, color="#174a7c")
    ax.text(17.0, -13.0, "ノイズ・未モデル\n高周波域", fontsize=8, color="#555555")
    ax.set_xlim(0.01, 100.0)
    ax.set_ylim(-42.0, 16.0)
    ax.set_xlabel(r"角周波数 $\omega$ [rad/s]")
    ax.set_ylabel("ゲイン [dB]")
    ax.set_title(r"(c) 感度関数のトレードオフ, $L_K(s)=K/\{s(s+1)\}$")
    ax.grid(True, which="both", color="#d0d0d0", linewidth=0.7, alpha=0.7)
    ax.legend(loc="upper right", ncol=3, fontsize=7.1)
    add_panel_label(ax, "(c)", r"Sensitivity tradeoff: $K/\{s(s+1)\}$")
    ax.text(
        0.03,
        0.05,
        "\n".join(ms_lines) + "\n実線: |S|, 破線: |T|",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox=PANEL_BOX,
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "ja" / "figures" / "frequency_design_examples.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": ["Noto Sans CJK JP", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 7,
            "figure.dpi": 180,
            "savefig.dpi": 180,
        }
    )

    fig = plt.figure(figsize=(7.2, 6.8), constrained_layout=False)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05])

    nyquist_ax = fig.add_subplot(grid[0, 0])
    root_locus_ax = fig.add_subplot(grid[0, 1])
    sensitivity_ax = fig.add_subplot(grid[1, :])

    plot_nyquist(nyquist_ax)
    plot_root_locus(root_locus_ax)
    plot_sensitivity(sensitivity_ax)

    finalize_figure(fig, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
