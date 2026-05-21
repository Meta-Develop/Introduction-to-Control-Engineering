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
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("matplotlib is required to generate the Fourier figure") from exc

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
CUTOFFS = (1, 3, 9, 25)
THETA_START = 0.0
THETA_STOP = 2.0 * math.pi
SAMPLE_COUNT = 1600


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


def harmonic_indices(cutoff: int):
    if np is not None:
        return np.arange(1, cutoff + 1, 2)
    return list(range(1, cutoff + 1, 2))


def harmonic_amplitudes(indices):
    if np is not None:
        return 4.0 / (math.pi * indices)
    return [4.0 / (math.pi * k) for k in indices]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "ja" / "figures" / "fourier_square_partial_sums.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    theta = make_linspace(THETA_START, THETA_STOP, SAMPLE_COUNT)

    configure_fonts()
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "figure.dpi": 180,
            "savefig.dpi": 180,
        }
    )

    fig, (wave_ax, spectrum_ax) = plt.subplots(
        2,
        1,
        figsize=(7.2, 4.8),
        constrained_layout=False,
        gridspec_kw={"height_ratios": [3.0, 1.25], "hspace": 0.32},
    )

    wave_ax.step(
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
        wave_ax.plot(
            theta,
            partial_sum(theta, cutoff),
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            label=f"N={cutoff}",
        )

    wave_ax.axhline(0.0, color="#666666", linewidth=0.8)
    for x_value in (0.0, math.pi, 2.0 * math.pi):
        wave_ax.axvline(x_value, color="#bbbbbb", linestyle=":", linewidth=0.9)

    wave_ax.annotate(
        "高い高調波を含めるほど立上りが鋭くなる",
        xy=(0.19, 1.10),
        xytext=(0.92, 1.31),
        arrowprops={"arrowstyle": "->", "color": "#555555", "linewidth": 1.0},
        ha="left",
        va="center",
        fontsize=8.5,
        color="#333333",
    )
    wave_ax.annotate(
        "帯域を切ると角は丸くなる",
        xy=(math.pi + 0.22, -1.05),
        xytext=(math.pi + 1.05, -0.62),
        arrowprops={"arrowstyle": "->", "color": "#555555", "linewidth": 1.0},
        ha="left",
        va="center",
        fontsize=8.5,
        color="#333333",
    )
    wave_ax.annotate(
        "",
        xy=(0.0, -1.34),
        xytext=(2.0 * math.pi, -1.34),
        arrowprops={"arrowstyle": "<->", "color": "#555555", "linewidth": 1.0},
    )
    wave_ax.text(math.pi, -1.26, r"周期 $2\pi$", ha="center", va="bottom", fontsize=8)
    wave_ax.text(
        0.015,
        0.07,
        r"$q_N(\theta)=\frac{4}{\pi}\sum_{k=1,3,\ldots,N}\frac{\sin(k\theta)}{k}$"
        "\n奇数次のみ、振幅は $1/k$ で減少",
        transform=wave_ax.transAxes,
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

    wave_ax.set_xlim(0.0, 2.0 * math.pi)
    wave_ax.set_ylim(-1.45, 1.45)
    wave_ax.set_xlabel("位相 θ [rad]")
    wave_ax.set_ylabel("振幅 [-]")
    wave_ax.set_xticks([0.0, math.pi, 2.0 * math.pi])
    wave_ax.set_xticklabels(["0", r"$\pi$", r"$2\pi$"])
    wave_ax.set_yticks([-1.0, 0.0, 1.0])
    wave_ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)
    wave_ax.legend(loc="upper right", ncol=2, framealpha=0.94)

    harmonic_k = harmonic_indices(max(CUTOFFS))
    amplitudes = harmonic_amplitudes(harmonic_k)
    spectrum_ax.bar(
        harmonic_k,
        amplitudes,
        width=0.72,
        color="#4c78a8",
        alpha=0.86,
        label=r"$4/(\pi k)$",
    )
    spectrum_ax.plot(
        harmonic_k,
        amplitudes,
        color="#222222",
        marker="o",
        markersize=3.2,
        linewidth=1.1,
    )
    for cutoff in CUTOFFS:
        spectrum_ax.axvline(cutoff, color="#999999", linestyle=":", linewidth=0.8)
    spectrum_ax.annotate(
        "低いN: 滑らか",
        xy=(3.0, 4.0 / (math.pi * 3.0)),
        xytext=(5.4, 0.74),
        arrowprops={"arrowstyle": "->", "color": "#555555", "linewidth": 0.9},
        ha="left",
        va="center",
        fontsize=8,
        color="#333333",
    )
    spectrum_ax.annotate(
        "高いN: 鋭い角まで近づく",
        xy=(25.0, 4.0 / (math.pi * 25.0)),
        xytext=(13.4, 0.35),
        arrowprops={"arrowstyle": "->", "color": "#555555", "linewidth": 0.9},
        ha="left",
        va="center",
        fontsize=8,
        color="#333333",
    )
    spectrum_ax.set_xlim(0.0, 26.0)
    spectrum_ax.set_ylim(0.0, 1.42)
    spectrum_ax.set_xlabel("高調波次数 k [-]")
    spectrum_ax.set_ylabel("正弦振幅 [-]")
    spectrum_ax.set_xticks([1.0, 3.0, 9.0, 17.0, 25.0])
    spectrum_ax.set_xticklabels(["1", "3", "9", "17", "25"])
    spectrum_ax.set_yticks([0.0, 0.5, 1.0])
    spectrum_ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7, axis="y")
    spectrum_ax.text(
        0.98,
        0.88,
        "棒: 部分和に入れる奇数次成分",
        transform=spectrum_ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "#bbbbbb"},
    )

    finalize_figure(fig, output_path)
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
