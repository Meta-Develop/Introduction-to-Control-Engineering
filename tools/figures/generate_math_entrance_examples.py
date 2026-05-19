#!/usr/bin/env python3
"""Generate visual examples for the mathematics entrance section."""

from __future__ import annotations

from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("numpy is required to generate the math entrance figure") from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("matplotlib is required to generate the math entrance figure") from exc

from figure_style import finalize_figure


OUTPUT_SIZE = (8.0, 6.0)
DPI = 180
PREFERRED_FONT_FAMILIES = (
    "Noto Sans CJK JP",
    "Noto Sans JP",
    "IPAexGothic",
    "IPAGothic",
    "TakaoGothic",
    "DejaVu Sans",
)


def preferred_font_families() -> list[str]:
    available = {font.name for font in font_manager.fontManager.ttflist}
    return [family for family in PREFERRED_FONT_FAMILIES if family in available] or ["DejaVu Sans"]


def q_curve(time: np.ndarray) -> np.ndarray:
    return 0.18 * (time - 1.55) ** 3 + 0.35 * (time - 1.55) ** 2 + 0.70 * time + 0.35


def dq_curve(time: float) -> float:
    return 0.54 * (time - 1.55) ** 2 + 0.70 * (time - 1.55) + 0.70


def taylor_function(x_value: np.ndarray) -> np.ndarray:
    return np.sin(x_value) + 0.18 * x_value**2


def dtaylor_function(x_value: float) -> float:
    return np.cos(x_value) + 0.36 * x_value


def cumulative_trapezoid(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    increments = 0.5 * (values[:-1] + values[1:]) * np.diff(grid)
    return np.concatenate(([0.0], np.cumsum(increments)))


def plot_secants(ax: plt.Axes) -> None:
    time = np.linspace(0.0, 2.8, 500)
    t0 = 1.0
    h_values = (1.20, 0.70, 0.35)
    colors = ("#7f7f7f", "#ff7f0e", "#2ca02c")

    ax.plot(time, q_curve(time), color="#1f77b4", linewidth=2.2, label=r"$q(t)$")
    ax.scatter([t0], [q_curve(np.array([t0]))[0]], color="#1f77b4", s=28, zorder=4)

    line_time = np.array([0.25, 2.65])
    q0 = q_curve(np.array([t0]))[0]
    for h_value, color in zip(h_values, colors):
        qh = q_curve(np.array([t0 + h_value]))[0]
        slope = (qh - q0) / h_value
        secant = q0 + slope * (line_time - t0)
        ax.plot(
            line_time,
            secant,
            color=color,
            linewidth=1.4,
            linestyle="--",
            label=rf"$h={h_value:.2f}$",
        )
        ax.scatter([t0 + h_value], [qh], color=color, s=18, zorder=4)

    tangent = q0 + dq_curve(t0) * (line_time - t0)
    ax.plot(line_time, tangent, color="#d62728", linewidth=2.0, label="接線")
    ax.axvline(t0, color="#555555", linewidth=0.8, linestyle=":")
    ax.set_title("(a) 割線から接線へ")
    ax.set_xlabel("時刻 t [s]")
    ax.set_ylabel("位置 q [m]")
    ax.set_xlim(0.0, 2.8)
    ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)
    ax.legend(loc="upper left", fontsize=7)


def plot_integral(ax: plt.Axes) -> None:
    time = np.linspace(0.0, 6.0, 800)
    rate = 0.82 * np.sin(1.15 * time) + 0.22 * np.cos(2.05 * time) - 0.10
    accumulation = cumulative_trapezoid(rate, time)

    ax.axhline(0.0, color="#555555", linewidth=0.9)
    ax.plot(time, rate, color="#1f77b4", linewidth=2.0, label=r"変化率 $r(t)$")
    ax.fill_between(
        time,
        0.0,
        rate,
        where=rate >= 0.0,
        color="#2ca02c",
        alpha=0.25,
        interpolate=True,
        label="正の面積",
    )
    ax.fill_between(
        time,
        0.0,
        rate,
        where=rate < 0.0,
        color="#d62728",
        alpha=0.25,
        interpolate=True,
        label="負の面積",
    )

    acc_ax = ax.twinx()
    acc_ax.plot(
        time,
        accumulation,
        color="#222222",
        linewidth=1.8,
        linestyle="-.",
        label=r"$\int_0^t r(\tau)d\tau$",
    )
    acc_ax.set_ylabel("累積量")
    acc_ax.tick_params(axis="y", labelsize=8)

    ax.set_title("(b) 符号付き面積と累積量")
    ax.set_xlabel("時刻 t [s]")
    ax.set_ylabel(r"変化率 $r(t)$")
    ax.set_xlim(0.0, 6.0)
    ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)

    lines, labels = ax.get_legend_handles_labels()
    acc_lines, acc_labels = acc_ax.get_legend_handles_labels()
    ax.legend(lines + acc_lines, labels + acc_labels, loc="lower left", fontsize=7)


def plot_taylor(ax: plt.Axes) -> None:
    x_value = np.linspace(-1.0, 2.2, 600)
    x0 = 0.45
    f0 = taylor_function(np.array([x0]))[0]
    slope = dtaylor_function(x0)
    approximation = f0 + slope * (x_value - x0)
    truth = taylor_function(x_value)

    ax.plot(x_value, truth, color="#1f77b4", linewidth=2.2, label=r"$f(x)$")
    ax.plot(
        x_value,
        approximation,
        color="#d62728",
        linewidth=1.8,
        linestyle="--",
        label="一次 Taylor 近似",
    )
    local = np.abs(x_value - x0) <= 0.75
    ax.fill_between(
        x_value,
        truth,
        approximation,
        where=local,
        color="#9467bd",
        alpha=0.22,
        interpolate=True,
        label="近傍誤差",
    )
    ax.scatter([x0], [f0], color="#222222", s=28, zorder=4)
    ax.axvline(x0, color="#555555", linewidth=0.8, linestyle=":")
    ax.text(x0 + 0.04, f0 - 0.35, r"$x_0$", fontsize=8, color="#222222")
    ax.set_title("(c) 一次 Taylor 近似の誤差")
    ax.set_xlabel("状態 x")
    ax.set_ylabel(r"出力 $f(x)$")
    ax.set_xlim(-1.0, 2.2)
    ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)
    ax.legend(loc="upper left", fontsize=7)


def quadratic_form(matrix: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
    return (
        matrix[0, 0] * x_grid**2
        + 2.0 * matrix[0, 1] * x_grid * y_grid
        + matrix[1, 1] * y_grid**2
    )


def plot_quadratic_forms(ax: plt.Axes) -> None:
    grid = np.linspace(-2.0, 2.0, 360)
    x1, x2 = np.meshgrid(grid, grid)
    p_balanced = np.array([[1.0, 0.0], [0.0, 1.0]])
    p_coupled = np.array([[3.0, 1.1], [1.1, 1.0]])
    levels = (1.0, 2.25)

    ax.contour(
        x1,
        x2,
        quadratic_form(p_balanced, x1, x2),
        levels=levels,
        colors="#1f77b4",
        linewidths=(2.0, 1.2),
    )
    ax.contour(
        x1,
        x2,
        quadratic_form(p_coupled, x1, x2),
        levels=levels,
        colors="#d62728",
        linewidths=(2.0, 1.2),
        linestyles="--",
    )

    ax.axhline(0.0, color="#777777", linewidth=0.8)
    ax.axvline(0.0, color="#777777", linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("(d) 二次形式の等高線")
    ax.set_xlabel(r"状態 $x_1$")
    ax.set_ylabel(r"状態 $x_2$")
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)
    ax.text(
        0.03,
        0.04,
        r"$x^\mathsf{T}Px=1,\ 2.25$",
        transform=ax.transAxes,
        fontsize=7.5,
        color="#333333",
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": "white",
            "edgecolor": "#cccccc",
            "alpha": 0.88,
        },
    )

    legend_items = [
        Line2D([0], [0], color="#1f77b4", linewidth=2.0, label=r"$P_1$=[[1,0],[0,1]]"),
        Line2D(
            [0],
            [0],
            color="#d62728",
            linewidth=2.0,
            linestyle="--",
            label=r"$P_2$=[[3,1.1],[1.1,1]]",
        ),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=7)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "ja" / "figures" / "math_entrance_examples.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 7,
            "font.family": preferred_font_families(),
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "axes.unicode_minus": False,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=OUTPUT_SIZE, constrained_layout=False)

    plot_secants(axes[0, 0])
    plot_integral(axes[1, 0])
    plot_taylor(axes[0, 1])
    plot_quadratic_forms(axes[1, 1])

    finalize_figure(fig, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
