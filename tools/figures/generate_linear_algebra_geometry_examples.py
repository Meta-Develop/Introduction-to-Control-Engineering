#!/usr/bin/env python3
"""Generate eigenvalue and quadratic-form geometry examples."""

from __future__ import annotations

from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("numpy is required to generate the linear algebra geometry figure") from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("matplotlib is required to generate the linear algebra geometry figure") from exc


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


def setup_2d_axes(ax: plt.Axes, limit: float = 2.4) -> None:
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.axvline(0, color="#777777", linewidth=0.8)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)
    ax.tick_params(labelsize=7)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")


def quadratic_form(matrix: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
    return (
        matrix[0, 0] * x_grid**2
        + 2.0 * matrix[0, 1] * x_grid * y_grid
        + matrix[1, 1] * y_grid**2
    )


def eigensystem(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(matrix)
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order]


def plot_linear_map(ax: plt.Axes) -> None:
    setup_2d_axes(ax, 2.3)
    matrix = np.array([[1.55, 0.55], [0.55, 0.75]])
    theta = np.linspace(0, 2 * np.pi, 500)
    unit_circle = np.vstack((np.cos(theta), np.sin(theta)))
    mapped = matrix @ unit_circle
    eigenvalues, eigenvectors = eigensystem(matrix)

    ax.plot(unit_circle[0], unit_circle[1], color="#999999", linestyle=":", linewidth=1.4, label="単位円")
    ax.plot(mapped[0], mapped[1], color="#4c78a8", linewidth=2.0, label=r"$Ax$")
    for index, (value, vector) in enumerate(zip(eigenvalues, eigenvectors.T)):
        color = "#e45756" if index == 0 else "#54a24b"
        start = -1.05 * vector
        end = 1.05 * vector
        mapped_end = value * vector
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "<|-|>", "color": color, "lw": 1.8})
        ax.annotate("", xy=mapped_end, xytext=(0, 0), arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2.2})
        ax.text(*(mapped_end + 0.08 * vector), rf"$\lambda_{index + 1}v_{index + 1}$", color=color, fontsize=7.5)
    ax.set_title("(a) 線形写像と固有方向")
    ax.legend(loc="lower right", fontsize=7)


def plot_quadratic_levels(ax: plt.Axes) -> None:
    setup_2d_axes(ax, 2.3)
    grid = np.linspace(-2.3, 2.3, 420)
    x1, x2 = np.meshgrid(grid, grid)
    matrix = np.array([[3.0, 1.05], [1.05, 1.05]])
    eigenvalues, eigenvectors = eigensystem(matrix)
    levels = (0.75, 1.5, 3.0)
    contour = ax.contour(
        x1,
        x2,
        quadratic_form(matrix, x1, x2),
        levels=levels,
        colors=("#4c78a8", "#4c78a8", "#4c78a8"),
        linewidths=(1.8, 1.3, 1.0),
    )
    ax.clabel(contour, inline=True, fontsize=7, fmt="%.2g")
    for index, (value, vector) in enumerate(zip(eigenvalues, eigenvectors.T)):
        color = "#e45756" if index == 0 else "#54a24b"
        scale = 1.45 / np.sqrt(value)
        ax.annotate(
            "",
            xy=scale * vector,
            xytext=-scale * vector,
            arrowprops={"arrowstyle": "<|-|>", "color": color, "lw": 1.8},
        )
        ax.text(*(scale * vector + 0.06), rf"$v_{index + 1}$", color=color, fontsize=8)
    ax.set_title("(b) 二次形式の楕円と固有軸")


def plot_quadratic_surface(ax: plt.Axes) -> None:
    matrix = np.array([[1.25, 0.40], [0.40, 2.10]])
    grid = np.linspace(-1.7, 1.7, 90)
    x1, x2 = np.meshgrid(grid, grid)
    z = quadratic_form(matrix, x1, x2)
    ax.plot_surface(x1, x2, z, cmap="viridis", linewidth=0, alpha=0.90, antialiased=True)
    ax.contour(x1, x2, z, zdir="z", offset=0.0, levels=(1.0, 2.0, 4.0), colors="#333333", linewidths=0.8)
    ax.set_proj_type("ortho")
    ax.view_init(elev=27, azim=-56)
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.7, 1.7)
    ax.set_zlim(0, 8.5)
    ax.set_box_aspect((1, 1, 0.75))
    ax.tick_params(labelsize=7)
    ax.set_xlabel(r"$x_1$", labelpad=-6)
    ax.set_ylabel(r"$x_2$", labelpad=-6)
    ax.set_zlabel(r"$x^\mathsf{T}Px$", labelpad=-4)
    ax.set_title("(c) 正定値二次形式の曲面")


def plot_parameter_variation(ax: plt.Axes) -> None:
    setup_2d_axes(ax, 2.3)
    grid = np.linspace(-2.3, 2.3, 420)
    x1, x2 = np.meshgrid(grid, grid)
    cases = (
        (0.0, "#4c78a8", "-"),
        (0.55, "#f58518", "--"),
        (0.90, "#e45756", "-."),
    )
    legend_items: list[Line2D] = []
    for rho, color, linestyle in cases:
        matrix = np.array([[1.0, rho], [rho, 1.0]])
        ax.contour(
            x1,
            x2,
            quadratic_form(matrix, x1, x2),
            levels=(1.0,),
            colors=color,
            linestyles=linestyle,
            linewidths=2.0,
        )
        legend_items.append(Line2D([0], [0], color=color, linestyle=linestyle, linewidth=2.0, label=rf"$\rho={rho:.2f}$"))
    ax.text(
        -2.17,
        2.02,
        r"$P(\rho)=[[1,\rho],[\rho,1]]$",
        fontsize=7.5,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.90},
    )
    ax.legend(handles=legend_items, loc="lower right", fontsize=7)
    ax.set_title("(d) 非対角成分による形状変化")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "ja" / "figures" / "linear_algebra_geometry_examples.png"
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

    fig = plt.figure(figsize=OUTPUT_SIZE, constrained_layout=False)
    axes = [
        fig.add_subplot(2, 2, 1),
        fig.add_subplot(2, 2, 2),
        fig.add_subplot(2, 2, 3, projection="3d"),
        fig.add_subplot(2, 2, 4),
    ]
    fig.suptitle("固有方向、二次形式、正定値性の幾何")

    plot_linear_map(axes[0])
    plot_quadratic_levels(axes[1])
    plot_quadratic_surface(axes[2])
    plot_parameter_variation(axes[3])

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(output_path, facecolor="white")
    print(output_path)


if __name__ == "__main__":
    main()
