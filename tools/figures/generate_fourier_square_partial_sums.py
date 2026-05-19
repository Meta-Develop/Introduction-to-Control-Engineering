#!/usr/bin/env python3
"""Generate square-wave Fourier partial sums used in the frequency bridge."""

from __future__ import annotations

import math
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised only on minimal systems.
    np = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("matplotlib is required to generate the Fourier figure") from exc

from figure_style import finalize_figure


CUTOFFS = (1, 3, 9, 25)
THETA_START = 0.0
THETA_STOP = 2.0 * math.pi
SAMPLE_COUNT = 1600


def make_linspace(start: float, stop: float, count: int):
    if np is not None:
        return np.linspace(start, stop, count)

    step = (stop - start) / (count - 1)
    return [start + step * i for i in range(count)]


def partial_sum(theta, cutoff: int):
    harmonics = range(1, cutoff + 1, 2)
    if np is not None:
        total = np.zeros_like(theta)
        for k in harmonics:
            total += np.sin(k * theta) / k
        return (4.0 / math.pi) * total

    values = []
    for angle in theta:
        total = sum(math.sin(k * angle) / k for k in harmonics)
        values.append((4.0 / math.pi) * total)
    return values


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "ja" / "figures" / "fourier_square_partial_sums.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    theta = make_linspace(THETA_START, THETA_STOP, SAMPLE_COUNT)

    plt.rcParams.update(
        {
            "font.family": ["Noto Sans CJK JP", "DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "figure.dpi": 180,
            "savefig.dpi": 180,
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=False)
    ax.set_title("方形波のフーリエ部分和")

    ax.step(
        [0.0, math.pi, 2.0 * math.pi],
        [1.0, -1.0, -1.0],
        where="post",
        color="#222222",
        linewidth=2.2,
        label="理想方形波",
    )

    styles = (
        ("#1f77b4", "-"),
        ("#ff7f0e", "--"),
        ("#2ca02c", "-."),
        ("#d62728", ":"),
    )
    for cutoff, (color, linestyle) in zip(CUTOFFS, styles):
        ax.plot(
            theta,
            partial_sum(theta, cutoff),
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            label=f"N={cutoff}",
        )

    ax.axhline(0.0, color="#666666", linewidth=0.8)
    for x_value in (0.0, math.pi, 2.0 * math.pi):
        ax.axvline(x_value, color="#bbbbbb", linestyle=":", linewidth=0.9)

    ax.annotate(
        "",
        xy=(0.0, -1.32),
        xytext=(2.0 * math.pi, -1.32),
        arrowprops={"arrowstyle": "<->", "color": "#555555", "linewidth": 1.0},
    )
    ax.text(math.pi, -1.38, r"周期 $2\pi$", ha="center", va="top", fontsize=8)
    ax.text(
        0.02,
        0.06,
        r"$q_N(\theta)=\frac{4}{\pi}\sum_{k=1,3,\ldots,N}\frac{\sin(k\theta)}{k}$"
        "\n奇数次高調波, 振幅 ±1",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "alpha": 0.88,
            "edgecolor": "#bbbbbb",
        },
    )

    ax.set_xlim(0.0, 2.0 * math.pi)
    ax.set_ylim(-1.45, 1.45)
    ax.set_xlabel(r"位相 $\theta$ [rad]")
    ax.set_ylabel("振幅 [-]")
    ax.set_xticks([0.0, math.pi, 2.0 * math.pi])
    ax.set_xticklabels(["0", r"$\pi$", r"$2\pi$"])
    ax.set_yticks([-1.0, 0.0, 1.0])
    ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)
    ax.legend(loc="upper right")

    finalize_figure(fig, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
