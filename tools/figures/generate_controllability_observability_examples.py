#!/usr/bin/env python3
"""Generate controllability, observability, and PBH geometry examples."""

from __future__ import annotations

from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit(
        "numpy is required to generate the controllability/observability figure"
    ) from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit(
        "matplotlib is required to generate the controllability/observability figure"
    ) from exc

from figure_style import finalize_figure


OUTPUT_SIZE = (8.4, 7.2)
DPI = 180
AXIS_LIMIT = 1.72
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
    "Meiryo",
    "TakaoGothic",
)


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


def normalized(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def setup_state_plane(axis: plt.Axes) -> None:
    axis.axhline(0.0, color="#777777", linewidth=0.8)
    axis.axvline(0.0, color="#777777", linewidth=0.8)
    axis.set_xlim(-AXIS_LIMIT, AXIS_LIMIT)
    axis.set_ylim(-AXIS_LIMIT, AXIS_LIMIT)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(r"正規化状態方向 $x_1$ [-]")
    axis.set_ylabel(r"正規化状態方向 $x_2$ [-]")
    axis.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.75)
    axis.tick_params(labelsize=8)


def add_panel_header(axis: plt.Axes, text: str) -> None:
    axis.text(
        0.02,
        0.98,
        text,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        fontweight="bold",
        color="#222222",
        bbox={
            "boxstyle": "round,pad=0.20",
            "facecolor": "white",
            "edgecolor": "#bbbbbb",
            "alpha": 0.94,
        },
    )


def add_note(axis: plt.Axes, text: str, loc: tuple[float, float]) -> None:
    axis.text(
        loc[0],
        loc[1],
        text,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.2,
        color="#222222",
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": "#d0d0d0",
            "alpha": 0.94,
        },
    )


def draw_direction(
    axis: plt.Axes,
    vector: np.ndarray,
    label: str,
    color: str,
    *,
    scale: float = 1.0,
    linestyle: str = "-",
    label_scale: float = 1.12,
    linewidth: float = 2.2,
) -> None:
    direction = normalized(vector) * scale
    axis.annotate(
        "",
        xy=direction,
        xytext=(0.0, 0.0),
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "lw": linewidth,
            "linestyle": linestyle,
            "mutation_scale": 13,
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )
    if label:
        axis.text(
            label_scale * direction[0],
            label_scale * direction[1],
            label,
            color=color,
            fontsize=8.4,
            ha="center",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.8},
        )


def draw_span_parallelogram(
    axis: plt.Axes, first: np.ndarray, second: np.ndarray, color: str
) -> None:
    first = 0.42 * normalized(first)
    second = 0.42 * normalized(second)
    vertices = np.array([first + second, first - second, -first - second, -first + second])
    axis.add_patch(
        Polygon(
            vertices,
            closed=True,
            facecolor=color,
            edgecolor=color,
            alpha=0.13,
            linewidth=1.2,
            zorder=0,
        )
    )


def draw_line_span(axis: plt.Axes, vector: np.ndarray, color: str) -> None:
    direction = normalized(vector)
    points = np.vstack((-AXIS_LIMIT * direction, AXIS_LIMIT * direction))
    axis.plot(points[:, 0], points[:, 1], color=color, linewidth=2.0, alpha=0.65)


def plot_controllable(axis: plt.Axes) -> None:
    setup_state_plane(axis)
    a_matrix = np.array([[0.0, 1.0], [-2.0, -0.4]])
    b_matrix = np.array([[0.0], [1.0]])
    controllability = np.column_stack((b_matrix[:, 0], a_matrix @ b_matrix[:, 0]))
    rank = int(np.linalg.matrix_rank(controllability))
    b_direction = controllability[:, 0]
    ab_direction = controllability[:, 1]

    draw_span_parallelogram(axis, b_direction, ab_direction, "#0072B2")
    draw_direction(axis, b_direction, r"$B=[0,1]^\mathsf{T}$", "#0072B2")
    draw_direction(axis, ab_direction, r"$AB=[1,-0.4]^\mathsf{T}$", "#D55E00")
    add_panel_header(axis, "(a) 可制御: 入力方向が平面を張る")
    add_note(
        axis,
        rf"$\mathcal{{C}}=[B\ AB]$, rank $={rank}=n$" "\n"
        r"$\operatorname{span}\{B,AB\}=\mathbb{R}^2$",
        (0.04, 0.04),
    )


def plot_uncontrollable(axis: plt.Axes) -> None:
    setup_state_plane(axis)
    a_matrix = np.array([[0.55, 0.0], [0.0, -0.2]])
    b_matrix = np.array([[1.0], [0.0]])
    controllability = np.column_stack((b_matrix[:, 0], a_matrix @ b_matrix[:, 0]))
    rank = int(np.linalg.matrix_rank(controllability))
    b_direction = controllability[:, 0]
    ab_direction = controllability[:, 1]

    draw_line_span(axis, b_direction, "#0072B2")
    draw_direction(axis, b_direction, "", "#0072B2", scale=0.95)
    draw_direction(axis, ab_direction, "", "#D55E00", scale=0.55)
    axis.text(
        0.25,
        -0.17,
        r"$B,\ AB\parallel x_1$",
        color="#D55E00",
        fontsize=8.4,
        ha="left",
        va="center",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.8},
    )
    draw_direction(
        axis,
        np.array([0.0, 1.0]),
        r"到達不能方向 $q$",
        "#CC3311",
        linestyle="--",
        linewidth=2.0,
    )
    add_panel_header(axis, "(b) 不可制御: 到達可能部分空間が直線")
    add_note(
        axis,
        rf"$\mathcal{{C}}=[B\ AB]$, rank $={rank}<n$" "\n"
        r"$q^\mathsf{T}B=q^\mathsf{T}AB=0$",
        (0.04, 0.04),
    )


def plot_observable(axis: plt.Axes) -> None:
    setup_state_plane(axis)
    a_matrix = np.array([[0.0, 1.0], [-2.0, -0.4]])
    c_matrix = np.array([[1.0, 0.0]])
    observability = np.vstack((c_matrix, c_matrix @ a_matrix))
    rank = int(np.linalg.matrix_rank(observability))
    c_direction = c_matrix.T[:, 0]
    atc_direction = a_matrix.T @ c_direction

    draw_span_parallelogram(axis, c_direction, atc_direction, "#009E73")
    draw_direction(axis, c_direction, r"$C^\mathsf{T}=[1,0]^\mathsf{T}$", "#009E73", label_scale=0.92)
    draw_direction(axis, atc_direction, r"$A^\mathsf{T}C^\mathsf{T}=[0,1]^\mathsf{T}$", "#D55E00")
    add_panel_header(axis, "(c) 可観測: 測定方向と出力微分方向")
    add_note(
        axis,
        rf"$\mathcal{{O}}=[C;\ CA]$, rank $={rank}=n$" "\n"
        r"$\operatorname{span}\{C^\mathsf{T},A^\mathsf{T}C^\mathsf{T}\}=\mathbb{R}^2$",
        (0.04, 0.04),
    )


def plot_pbh_rank_drop(axis: plt.Axes) -> None:
    setup_state_plane(axis)
    a_matrix = np.diag([1.0, -2.0])
    b_matrix = np.array([[0.0], [1.0]])
    c_matrix = np.array([[0.0, 1.0]])
    lambda_unstable = 1.0
    pbh_controllability = np.column_stack(
        (lambda_unstable * np.eye(2) - a_matrix, b_matrix)
    )
    pbh_observability = np.vstack(
        (lambda_unstable * np.eye(2) - a_matrix, c_matrix)
    )
    rank_c = int(np.linalg.matrix_rank(pbh_controllability))
    rank_o = int(np.linalg.matrix_rank(pbh_observability))

    axis.axvspan(-AXIS_LIMIT, AXIS_LIMIT, color="#CC3311", alpha=0.05, zorder=0)
    draw_direction(
        axis,
        np.array([1.0, 0.0]),
        "",
        "#CC3311",
        linestyle="--",
        linewidth=2.0,
    )
    draw_direction(axis, b_matrix[:, 0], r"$B=C^\mathsf{T}=e_2$", "#0072B2")
    axis.plot([0.0, 0.0], [-AXIS_LIMIT, AXIS_LIMIT], color="#0072B2", linewidth=2.0, alpha=0.45)
    axis.plot([-AXIS_LIMIT, AXIS_LIMIT], [0.0, 0.0], color="#CC3311", linestyle="--", linewidth=1.5, alpha=0.55)
    axis.text(
        0.50,
        0.07,
        r"不可制御・不可観測 $v_1$",
        color="#CC3311",
        fontsize=8.4,
        ha="left",
        va="center",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.8},
    )
    add_panel_header(axis, "(d) PBH: 不安定モードでランク低下")
    add_note(
        axis,
        rf"$A=\operatorname{{diag}}(1,-2),\ \lambda=1$" "\n"
        rf"$\operatorname{{rank}}[\lambda I-A\ B]={rank_c}<n$" "\n"
        rf"$\operatorname{{rank}}[\lambda I-A;\ C]={rank_o}<n$",
        (0.04, 0.04),
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = (
        repo_root / "ja" / "figures" / "controllability_observability_examples.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    configure_fonts()
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 7,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=OUTPUT_SIZE, constrained_layout=False)
    plot_controllable(axes[0, 0])
    plot_uncontrollable(axes[0, 1])
    plot_observable(axes[1, 0])
    plot_pbh_rank_drop(axes[1, 1])

    finalize_figure(fig, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
