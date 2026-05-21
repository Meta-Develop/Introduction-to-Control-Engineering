#!/usr/bin/env python3
"""Generate a closed-loop signal-path block diagram."""

from __future__ import annotations

from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("matplotlib is required to generate the signal-path figure") from exc

from figure_style import finalize_figure


SIGNAL = "#303030"
FEEDBACK = "#666666"
INPUT_DISTURBANCE = "#2ca02c"
OUTPUT_DISTURBANCE = "#ff7f0e"
NOISE = "#9467bd"
CONTROL = "#1f77b4"


def draw_arrow(
    axis,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = SIGNAL,
    rad: float = 0.0,
    linewidth: float = 1.55,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=linewidth,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    axis.add_patch(arrow)


def draw_box(
    axis,
    center: tuple[float, float],
    width: float,
    height: float,
    label: str,
    *,
    facecolor: str,
) -> None:
    x = center[0] - width / 2.0
    y = center[1] - height / 2.0
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.05",
        linewidth=1.25,
        edgecolor=SIGNAL,
        facecolor=facecolor,
    )
    axis.add_patch(patch)
    axis.text(center[0], center[1], label, ha="center", va="center", fontsize=13)


def draw_sum(
    axis,
    center: tuple[float, float],
    signs: list[tuple[str, tuple[float, float]]],
) -> None:
    circle = Circle(center, 0.25, edgecolor=SIGNAL, facecolor="white", linewidth=1.25)
    axis.add_patch(circle)
    for sign, offset in signs:
        axis.text(
            center[0] + offset[0],
            center[1] + offset[1],
            sign,
            ha="center",
            va="center",
            fontsize=12,
            color=SIGNAL,
        )


def label(
    axis,
    position: tuple[float, float],
    text: str,
    *,
    color: str = SIGNAL,
    size: float = 12.0,
    ha: str = "center",
    va: str = "center",
) -> None:
    axis.text(position[0], position[1], text, color=color, fontsize=size, ha=ha, va=va)


def draw_signal_paths() -> plt.Figure:
    figure, axis = plt.subplots(figsize=(8.4, 3.7))
    axis.set_axis_off()
    axis.set_xlim(0.0, 11.7)
    axis.set_ylim(0.0, 5.0)

    y_main = 3.10
    y_top = 4.25
    y_feedback = 1.05
    error = (1.25, y_main)
    controller = (2.85, y_main)
    input_sum = (4.45, y_main)
    plant = (5.95, y_main)
    output_sum = (7.45, y_main)
    output_dot = (8.35, y_main)
    measurement_sum = (8.35, y_feedback)

    draw_sum(axis, error, [("+", (-0.12, 0.11)), ("-", (0.0, -0.16))])
    draw_box(axis, controller, 1.15, 0.72, "$C(s)$", facecolor="#e8f3ff")
    draw_sum(axis, input_sum, [("+", (-0.12, 0.11)), ("+", (0.0, 0.17))])
    draw_box(axis, plant, 1.15, 0.72, "$G(s)$", facecolor="#eef8e8")
    draw_sum(axis, output_sum, [("+", (-0.12, 0.11)), ("+", (0.0, 0.17))])
    draw_sum(axis, measurement_sum, [("+", (-0.10, 0.12)), ("+", (0.14, 0.0))])

    axis.scatter([output_dot[0]], [output_dot[1]], s=30, color=SIGNAL, zorder=3)

    draw_arrow(axis, (0.30, y_main), (1.00, y_main))
    label(axis, (0.18, y_main), "$r$", ha="left")

    draw_arrow(axis, (1.50, y_main), (2.28, y_main))
    label(axis, (1.85, y_main + 0.28), "$e=r-y_m$")

    draw_arrow(axis, (3.43, y_main), (4.20, y_main), color=CONTROL)
    label(axis, (3.82, y_main + 0.28), "$u$", color=CONTROL)

    draw_arrow(axis, (4.45, y_top), (4.45, y_main + 0.25), color=INPUT_DISTURBANCE)
    label(axis, (4.45, y_top + 0.22), "$d_i$", color=INPUT_DISTURBANCE)

    draw_arrow(axis, (4.70, y_main), (5.38, y_main))
    label(axis, (5.03, y_main + 0.28), "$u+d_i$")

    draw_arrow(axis, (6.52, y_main), (7.20, y_main))

    draw_arrow(axis, (7.45, y_top), (7.45, y_main + 0.25), color=OUTPUT_DISTURBANCE)
    label(axis, (7.45, y_top + 0.22), "$d_o$", color=OUTPUT_DISTURBANCE)

    draw_arrow(axis, (7.70, y_main), (8.35, y_main))
    draw_arrow(axis, (8.35, y_main), (9.95, y_main))
    label(axis, (10.10, y_main), "$y$", ha="left")

    draw_arrow(axis, (8.35, y_main - 0.05), (8.35, y_feedback + 0.25), color=FEEDBACK)
    draw_arrow(axis, (9.75, y_feedback), (8.60, y_feedback), color=NOISE)
    label(axis, (9.95, y_feedback), "$n$", color=NOISE, ha="left")

    draw_arrow(axis, (8.10, y_feedback), (1.25, y_feedback), color=FEEDBACK)
    draw_arrow(axis, (1.25, y_feedback), (1.25, y_main - 0.25), color=FEEDBACK)
    label(axis, (4.75, y_feedback + 0.25), "$y_m=y+n$", color=FEEDBACK)

    axis.plot([4.45, 4.45], [y_top - 0.10, y_top - 0.32], color=INPUT_DISTURBANCE, linewidth=1.2)
    axis.plot([7.45, 7.45], [y_top - 0.10, y_top - 0.32], color=OUTPUT_DISTURBANCE, linewidth=1.2)

    label(axis, (2.85, 2.42), "controller", color="#4f6478", size=9.0)
    label(axis, (5.95, 2.42), "plant", color="#537053", size=9.0)
    label(axis, (8.35, 0.43), "measurement path", color="#777777", size=9.0)

    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.06, top=0.92)
    return figure


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "ja" / "figures" / "closed_loop_signal_paths.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "mathtext.fontset": "dejavusans",
            "figure.dpi": 180,
            "savefig.dpi": 180,
        }
    )

    figure = draw_signal_paths()
    finalize_figure(figure, output_path, layout="none")
    plt.close(figure)
    print(output_path)


if __name__ == "__main__":
    main()
