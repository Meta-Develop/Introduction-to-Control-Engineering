#!/usr/bin/env python3
"""Generate first- and second-order step-response examples."""

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
    raise SystemExit("matplotlib is required to generate the response figure") from exc

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
FIRST_ORDER_TAUS = (0.5, 1.0, 2.0)
SECOND_ORDER_ZETAS = (0.2, 0.5, 1.0, 1.8)
OMEGA_N = 2.0
FIRST_TIME_STOP = 6.0
SECOND_TIME_STOP = 8.0
SAMPLE_COUNT = 1000


def make_linspace(start: float, stop: float, count: int):
    if np is not None:
        return np.linspace(start, stop, count)

    step = (stop - start) / (count - 1)
    return [start + step * i for i in range(count)]


def first_order_step(time, tau: float):
    if np is not None:
        return 1.0 - np.exp(-time / tau)

    return [1.0 - math.exp(-t / tau) for t in time]


def second_order_step(time, zeta: float):
    if zeta < 1.0:
        omega_d = OMEGA_N * math.sqrt(1.0 - zeta * zeta)
        ratio = zeta / math.sqrt(1.0 - zeta * zeta)
        if np is not None:
            return 1.0 - np.exp(-zeta * OMEGA_N * time) * (
                np.cos(omega_d * time) + ratio * np.sin(omega_d * time)
            )

        return [
            1.0
            - math.exp(-zeta * OMEGA_N * t)
            * (math.cos(omega_d * t) + ratio * math.sin(omega_d * t))
            for t in time
        ]

    if zeta == 1.0:
        if np is not None:
            return 1.0 - (1.0 + OMEGA_N * time) * np.exp(-OMEGA_N * time)

        return [
            1.0 - (1.0 + OMEGA_N * t) * math.exp(-OMEGA_N * t) for t in time
        ]

    root = math.sqrt(zeta * zeta - 1.0)
    p1 = -OMEGA_N * (zeta - root)
    p2 = -OMEGA_N * (zeta + root)
    if np is not None:
        return 1.0 + (p2 / (p1 - p2)) * np.exp(p1 * time) - (
            p1 / (p1 - p2)
        ) * np.exp(p2 * time)

    return [
        1.0
        + (p2 / (p1 - p2)) * math.exp(p1 * t)
        - (p1 / (p1 - p2)) * math.exp(p2 * t)
        for t in time
    ]


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
    fallback_fonts = selected_fonts + ["DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = fallback_fonts
    plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = repo_root / "ja" / "figures"
    first_output_path = output_dir / "time_response_first_order.png"
    second_output_path = output_dir / "time_response_second_order.png"
    output_dir.mkdir(parents=True, exist_ok=True)

    first_time = make_linspace(0.0, FIRST_TIME_STOP, SAMPLE_COUNT)
    second_time = make_linspace(0.0, SECOND_TIME_STOP, SAMPLE_COUNT)

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

    colors = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")
    linestyles = ("-", "--", "-.", ":")

    first_fig, first_ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=False)
    for tau, color, linestyle in zip(FIRST_ORDER_TAUS, colors, linestyles):
        first_ax.plot(
            first_time,
            first_order_step(first_time, tau),
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            label=rf"$\tau={tau:.1f}\,\mathrm{{s}}$",
        )

    first_ax.axhline(1.0, color="#555555", linestyle=":", linewidth=1.0)
    first_ax.axhline(
        1.0 - math.exp(-1.0),
        color="#777777",
        linestyle="--",
        linewidth=1.0,
    )
    first_ax.set_title("First-order response")
    first_ax.set_xlabel(r"時間 $t$ [s]")
    first_ax.set_ylabel(r"正規化出力 $y/(K u_0)$ [-]")
    first_ax.set_xlim(0.0, FIRST_TIME_STOP)
    first_ax.set_ylim(0.0, 1.06)
    first_ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)
    first_ax.legend(loc="lower right")
    first_ax.text(
        0.03,
        0.62,
        r"$1-e^{-1}$",
        transform=first_ax.transAxes,
        ha="left",
        va="center",
        fontsize=8,
        color="#444444",
    )

    finalize_figure(first_fig, first_output_path)
    plt.close(first_fig)
    print(first_output_path)

    second_fig, second_ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=False)
    for zeta, color, linestyle in zip(SECOND_ORDER_ZETAS, colors, linestyles):
        second_ax.plot(
            second_time,
            second_order_step(second_time, zeta),
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            label=rf"$\zeta={zeta:.1f},\ \omega_n={OMEGA_N:.1f}\,\mathrm{{rad/s}}$",
        )

    second_ax.axhline(1.0, color="#555555", linestyle=":", linewidth=1.0)
    second_ax.set_title("Second-order response")
    second_ax.set_xlabel(r"時間 $t$ [s]")
    second_ax.set_ylabel(r"正規化出力 $y/(K u_0)$ [-]")
    second_ax.set_xlim(0.0, SECOND_TIME_STOP)
    second_ax.set_ylim(0.0, 1.6)
    second_ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)
    second_ax.legend(loc="lower right")

    finalize_figure(second_fig, second_output_path)
    plt.close(second_fig)
    print(second_output_path)


if __name__ == "__main__":
    main()
