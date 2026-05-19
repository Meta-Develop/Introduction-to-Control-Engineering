#!/usr/bin/env python3
"""Generate vector geometry examples for the mathematics entrance section."""

from __future__ import annotations

from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("numpy is required to generate the vector geometry figure") from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc, Circle, FancyArrowPatch
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("matplotlib is required to generate the vector geometry figure") from exc

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


def arrow2d(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    label: str | None = None,
    linewidth: float = 2.0,
    mutation_scale: float = 12.0,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(arrow)
    if label:
        end_array = np.asarray(end)
        ax.text(end_array[0] + 0.06, end_array[1] + 0.06, label, color=color, fontsize=8)


def setup_2d_axes(ax: plt.Axes, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.axvline(0, color="#777777", linewidth=0.8)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)
    ax.tick_params(labelsize=7)


def plot_projection(ax: plt.Axes) -> None:
    setup_2d_axes(ax, (-0.4, 3.6), (-0.4, 2.9))
    a = np.array([2.65, 2.05])
    b = np.array([3.0, 0.95])
    projection = (a @ b) / (b @ b) * b

    arrow2d(ax, (0, 0), tuple(b), color="#4c78a8", label=r"$b$")
    arrow2d(ax, (0, 0), tuple(a), color="#f58518", label=r"$a$")
    arrow2d(ax, (0, 0), tuple(projection), color="#54a24b", label=r"$\operatorname{proj}_b a$")
    ax.plot([a[0], projection[0]], [a[1], projection[1]], color="#666666", linestyle="--", linewidth=1.1)
    ax.scatter([projection[0]], [projection[1]], color="#54a24b", s=20, zorder=4)
    ax.add_patch(Arc((0, 0), 0.85, 0.85, angle=0, theta1=0, theta2=34, color="#555555", linewidth=1.0))
    ax.text(0.56, 0.18, r"$\theta$", fontsize=8, color="#333333")
    ax.set_title("(a) 内積と射影")
    ax.set_xlabel(r"$x$ 成分")
    ax.set_ylabel(r"$y$ 成分")


def plot_basis_change(ax: plt.Axes) -> None:
    setup_2d_axes(ax, (-0.7, 3.6), (-0.6, 3.1))
    b1 = np.array([1.25, 0.35])
    b2 = np.array([0.45, 1.15])
    coeffs = np.array([1.8, 1.3])
    point = coeffs[0] * b1 + coeffs[1] * b2

    arrow2d(ax, (0, 0), (1.0, 0.0), color="#999999", label=r"$e_x$", linewidth=1.4)
    arrow2d(ax, (0, 0), (0.0, 1.0), color="#999999", label=r"$e_y$", linewidth=1.4)
    arrow2d(ax, (0, 0), tuple(b1), color="#4c78a8", label=r"$b_1$")
    arrow2d(ax, (0, 0), tuple(b2), color="#f58518", label=r"$b_2$")
    ax.plot(
        [coeffs[0] * b1[0], point[0], coeffs[1] * b2[0]],
        [coeffs[0] * b1[1], point[1], coeffs[1] * b2[1]],
        color="#777777",
        linestyle="--",
        linewidth=1.0,
    )
    ax.plot([0, coeffs[0] * b1[0]], [0, coeffs[0] * b1[1]], color="#4c78a8", alpha=0.55)
    ax.plot([0, coeffs[1] * b2[0]], [0, coeffs[1] * b2[1]], color="#f58518", alpha=0.55)
    arrow2d(ax, (0, 0), tuple(point), color="#e45756", label=r"$p$")
    ax.text(point[0] - 0.20, point[1] + 0.18, r"$p=\xi_1b_1+\xi_2b_2$", color="#333333", fontsize=7.5)
    ax.set_title("(b) 基底と座標成分")
    ax.set_xlabel(r"$x$ 成分")
    ax.set_ylabel(r"$y$ 成分")


def plot_cross_product(ax: plt.Axes) -> None:
    ax.set_proj_type("ortho")
    ax.view_init(elev=22, azim=-56)
    ax.set_xlim(0.0, 1.25)
    ax.set_ylim(0.0, 1.25)
    ax.set_zlim(0.0, 1.25)
    ax.set_box_aspect((1, 1, 0.9))
    ax.set_xticks((0, 1))
    ax.set_yticks((0, 1))
    ax.set_zticks((0, 1))
    ax.tick_params(labelsize=7)
    ax.set_xlabel(r"$x$", labelpad=-6)
    ax.set_ylabel(r"$y$", labelpad=-6)
    ax.set_zlabel(r"$z$", labelpad=-6)

    a = np.array([0.88, 0.28, 0.0])
    b = np.array([0.18, 0.86, 0.0])
    cross = np.cross(a, b)
    cross = cross / np.linalg.norm(cross) * 0.78

    ax.quiver(0, 0, 0, *a, color="#4c78a8", linewidth=2.0, arrow_length_ratio=0.12)
    ax.quiver(0, 0, 0, *b, color="#f58518", linewidth=2.0, arrow_length_ratio=0.12)
    ax.quiver(0, 0, 0, *cross, color="#54a24b", linewidth=2.4, arrow_length_ratio=0.14)

    polygon = np.array([[0, 0, 0], a, a + b, b])
    ax.add_collection3d(
        matplotlib.collections.PolyCollection(
            [polygon[:, :2]],
            facecolors="#bbbbbb",
            alpha=0.20,
            edgecolors="#777777",
            linewidths=0.8,
        ),
        zs=0,
        zdir="z",
    )
    ax.text(*a, r"$a$", color="#4c78a8", fontsize=8)
    ax.text(*b, r"$b$", color="#f58518", fontsize=8)
    ax.text(*(cross + np.array([0.02, 0.02, 0.02])), r"$a\times b$", color="#54a24b", fontsize=8)
    ax.text(0.50, 0.50, 0.03, "面積", color="#555555", fontsize=7)
    ax.set_title("(c) 右手系と外積")


def plot_torque(ax: plt.Axes) -> None:
    setup_2d_axes(ax, (-0.5, 3.4), (-0.7, 2.7))
    r = np.array([2.25, 0.72])
    force = np.array([-0.40, 1.45])
    force_end = r + force
    moment_arm = abs(np.cross(np.append(r, 0.0), np.append(force / np.linalg.norm(force), 0.0))[2])

    arrow2d(ax, (0, 0), tuple(r), color="#4c78a8", label=r"$r$")
    arrow2d(ax, tuple(r), tuple(force_end), color="#f58518", label=r"$F$", linewidth=2.2)
    ax.plot([0, r[0]], [0, r[1]], color="#4c78a8", linewidth=1.0, alpha=0.35)
    ax.add_patch(Circle((0.45, 1.85), 0.18, edgecolor="#54a24b", facecolor="white", linewidth=1.8))
    ax.scatter([0.45], [1.85], color="#54a24b", s=18, zorder=5)
    ax.text(0.68, 1.82, r"$\tau=r\times F$", color="#54a24b", fontsize=8)
    ax.text(1.05, 0.30, rf"モーメント腕 $\simeq {moment_arm:.2f}$", color="#333333", fontsize=7.5)
    ax.text(-0.18, -0.18, r"$O$", fontsize=8)
    ax.set_title("(d) 力のモーメント")
    ax.set_xlabel(r"$x$ [m]")
    ax.set_ylabel(r"$y$ [m]")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "ja" / "figures" / "vector_geometry_examples.png"
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

    plot_projection(axes[0])
    plot_basis_change(axes[1])
    plot_cross_product(axes[2])
    plot_torque(axes[3])

    finalize_figure(fig, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
