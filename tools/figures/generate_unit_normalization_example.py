#!/usr/bin/env python3
"""Generate the unit-conversion and normalization comparison figure."""

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
    raise SystemExit("matplotlib is required to generate the normalization figure") from exc


CASES = (
    {"name": "条件 A", "theta0": 20.0, "tau": 60.0, "color": "#1f77b4", "linestyle": "-"},
    {"name": "条件 B", "theta0": 10.0, "tau": 120.0, "color": "#d62728", "linestyle": "--"},
)
TIME_STOP = 360.0
XI_STOP = 3.0
SAMPLE_COUNT = 900


def make_linspace(start: float, stop: float, count: int):
    if np is not None:
        return np.linspace(start, stop, count)

    step = (stop - start) / (count - 1)
    return [start + step * i for i in range(count)]


def dimensional_response(time, theta0: float, tau: float):
    if np is not None:
        return theta0 * np.exp(-time / tau)

    return [theta0 * math.exp(-t / tau) for t in time]


def normalized_response(xi):
    if np is not None:
        return np.exp(-xi)

    return [math.exp(-value) for value in xi]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "ja" / "figures" / "unit_normalization_example.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    time = make_linspace(0.0, TIME_STOP, SAMPLE_COUNT)
    xi = make_linspace(0.0, XI_STOP, SAMPLE_COUNT)

    plt.rcParams.update(
        {
            "font.family": ["Noto Sans CJK JP", "DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "figure.dpi": 180,
            "savefig.dpi": 180,
        }
    )

    fig, (dimensional_ax, normalized_ax) = plt.subplots(
        1, 2, figsize=(7.2, 4.2), constrained_layout=False
    )
    fig.suptitle("一次遅れ応答の単位付き表示と正規化表示")

    for case in CASES:
        label = (
            f"{case['name']}: "
            rf"$\theta_0={case['theta0']:.0f}$ K, $\tau={case['tau']:.0f}$ s"
        )
        dimensional_ax.plot(
            time,
            dimensional_response(time, case["theta0"], case["tau"]),
            color=case["color"],
            linestyle=case["linestyle"],
            linewidth=2.0,
            label=label,
        )
        normalized_ax.plot(
            xi,
            normalized_response(xi),
            color=case["color"],
            linestyle=case["linestyle"],
            linewidth=2.0,
            label=case["name"],
        )

    normalized_ax.plot(
        xi,
        normalized_response(xi),
        color="#222222",
        linestyle=":",
        linewidth=1.5,
        label=r"$\exp(-\xi)$",
    )

    dimensional_ax.set_title("単位付き応答")
    dimensional_ax.set_xlabel(r"時間 $t$ [s]")
    dimensional_ax.set_ylabel(r"温度偏差 $\theta$ [K]")
    dimensional_ax.set_xlim(0.0, TIME_STOP)
    dimensional_ax.set_ylim(0.0, 21.0)
    dimensional_ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)
    dimensional_ax.legend(loc="upper right")

    normalized_ax.set_title("正規化座標")
    normalized_ax.set_xlabel(r"正規化時間 $\xi=t/\tau$ [-]")
    normalized_ax.set_ylabel(r"正規化応答 $x=\theta/\theta_0$ [-]")
    normalized_ax.set_xlim(0.0, XI_STOP)
    normalized_ax.set_ylim(0.0, 1.05)
    normalized_ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)
    normalized_ax.legend(loc="upper right")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.savefig(output_path, facecolor="white")
    print(output_path)


if __name__ == "__main__":
    main()
