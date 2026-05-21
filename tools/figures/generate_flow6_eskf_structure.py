#!/usr/bin/env python3
"""Generate the Flow 6 ESKF nominal/error-state structure diagram."""

from __future__ import annotations

from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("matplotlib is required to generate the ESKF figure") from exc

from figure_style import finalize_figure


BLUE = "#4477AA"
GREEN = "#228833"
ORANGE = "#CC6677"
GRAY = "#666666"
FILL_BLUE = "#edf4fb"
FILL_GREEN = "#edf8f0"
FILL_ORANGE = "#fbf0f2"


def draw_box(
    axis: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    edgecolor: str,
    facecolor: str,
    fontsize: float = 9.2,
) -> None:
    x_coord, y_coord = xy
    axis.add_patch(
        Rectangle(
            (x_coord, y_coord),
            width,
            height,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.55,
            zorder=2,
        )
    )
    axis.text(
        x_coord + width / 2.0,
        y_coord + height / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#222222",
        zorder=3,
    )


def draw_arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#333333",
    label: str | None = None,
    connectionstyle: str = "arc3,rad=0.0",
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.45,
            color=color,
            shrinkA=4,
            shrinkB=4,
            connectionstyle=connectionstyle,
            zorder=4,
        )
    )
    if label is not None:
        axis.text(
            (start[0] + end[0]) / 2.0,
            (start[1] + end[1]) / 2.0 + 0.035,
            label,
            ha="center",
            va="bottom",
            fontsize=8.2,
            color=color,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.86,
                "pad": 0.5,
            },
            zorder=5,
        )


def add_lane_label(axis: plt.Axes, y_coord: float, text: str, color: str) -> None:
    axis.text(
        0.018,
        y_coord,
        text,
        ha="left",
        va="center",
        fontsize=9.6,
        fontweight="bold",
        color=color,
        bbox={
            "facecolor": "white",
            "edgecolor": color,
            "linewidth": 1.0,
            "alpha": 0.95,
            "pad": 2.0,
        },
        zorder=6,
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "ja" / "figures" / "flow6_eskf_structure.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 9,
            "font.family": ["Noto Sans", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "figure.dpi": 180,
            "savefig.dpi": 180,
            "axes.unicode_minus": False,
        }
    )

    fig, axis = plt.subplots(figsize=(8.0, 3.9), constrained_layout=False)
    fig.subplots_adjust(left=0.025, right=0.985, bottom=0.05, top=0.985)
    axis.set_axis_off()
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)

    axis.axhspan(0.54, 0.95, facecolor="#f7fbff", edgecolor="none", zorder=0)
    axis.axhspan(0.08, 0.47, facecolor="#f7fff7", edgecolor="none", zorder=0)
    axis.axhline(0.505, color="#bbbbbb", linewidth=1.0, linestyle="--", zorder=1)

    add_lane_label(axis, 0.91, r"nominal state $\hat{x}$ on manifold", BLUE)
    add_lane_label(axis, 0.43, r"error state $\delta x$ in tangent space", GREEN)

    draw_box(
        axis,
        (0.05, 0.63),
        0.16,
        0.17,
        r"IMU measurement" "\n" r"$\omega_m,\ a_m$",
        edgecolor=GRAY,
        facecolor="white",
    )
    draw_box(
        axis,
        (0.27, 0.63),
        0.20,
        0.17,
        r"nominal propagation" "\n" r"$\hat{x}^{-}=f(\hat{x},u_m)$",
        edgecolor=BLUE,
        facecolor=FILL_BLUE,
    )
    draw_box(
        axis,
        (0.54, 0.63),
        0.17,
        0.17,
        r"predicted output" "\n" r"$\hat{y}=h(\hat{x}^{-})$",
        edgecolor=BLUE,
        facecolor=FILL_BLUE,
        fontsize=8.3,
    )
    draw_box(
        axis,
        (0.79, 0.63),
        0.17,
        0.17,
        r"error injection/reset" "\n" r"$\hat{x}^{+}=\hat{x}^{-}\oplus\widehat{\delta x}$" "\n" r"$\delta x\leftarrow0$",
        edgecolor=ORANGE,
        facecolor=FILL_ORANGE,
        fontsize=8.4,
    )

    draw_box(
        axis,
        (0.27, 0.18),
        0.20,
        0.17,
        r"linearized error model" "\n" r"$\delta x_{k+1}\simeq\Phi\delta x_k+w$",
        edgecolor=GREEN,
        facecolor=FILL_GREEN,
        fontsize=8.9,
    )
    draw_box(
        axis,
        (0.54, 0.18),
        0.17,
        0.17,
        r"covariance propagation" "\n" r"$P^{-}=\Phi P\Phi^\mathsf{T}+Q$",
        edgecolor=GREEN,
        facecolor=FILL_GREEN,
        fontsize=8.8,
    )
    draw_box(
        axis,
        (0.79, 0.18),
        0.17,
        0.17,
        r"Kalman error update" "\n" r"$\widehat{\delta x}=K\nu$" "\n" r"Joseph form $P^{+}$",
        edgecolor=GREEN,
        facecolor=FILL_GREEN,
        fontsize=8.4,
    )
    draw_box(
        axis,
        (0.54, 0.40),
        0.17,
        0.10,
        r"innovation $\nu=y-\hat{y}$" "\n" r"Jacobian $H_\delta$",
        edgecolor=ORANGE,
        facecolor=FILL_ORANGE,
        fontsize=8.5,
    )

    draw_arrow(axis, (0.21, 0.715), (0.27, 0.715))
    draw_arrow(axis, (0.47, 0.715), (0.54, 0.715))
    draw_arrow(axis, (0.71, 0.715), (0.79, 0.715), color=ORANGE)
    draw_arrow(axis, (0.21, 0.66), (0.27, 0.30), color=GRAY, connectionstyle="arc3,rad=-0.25")
    draw_arrow(axis, (0.47, 0.265), (0.54, 0.265), color=GREEN)
    draw_arrow(axis, (0.625, 0.63), (0.625, 0.50), color=ORANGE)
    draw_arrow(axis, (0.71, 0.45), (0.79, 0.30), color=ORANGE)
    draw_arrow(axis, (0.71, 0.265), (0.79, 0.265), color=GREEN)
    draw_arrow(axis, (0.875, 0.35), (0.875, 0.63), color=ORANGE, label=r"$\widehat{\delta x}$")
    draw_arrow(axis, (0.94, 0.18), (0.94, 0.10), color=GREEN)
    axis.text(
        0.905,
        0.085,
        r"$P\leftarrow G_rP^+G_r^\mathsf{T}$",
        ha="center",
        va="center",
        fontsize=8.0,
        color=GREEN,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 0.4},
        zorder=5,
    )

    finalize_figure(fig, output_path, layout="none")
    print(output_path)


if __name__ == "__main__":
    main()
