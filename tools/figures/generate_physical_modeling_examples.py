#!/usr/bin/env python3
"""Generate schematic examples for physical modeling."""

from __future__ import annotations

from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("matplotlib is required to generate the physical-modeling figure") from exc

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


def setup_panel(axis, title: str) -> None:
    axis.set_axis_off()
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_title(title, loc="left", fontweight="bold", pad=7)


def add_box(
    axis,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str = "#333333",
    fontsize: float = 9.5,
    linewidth: float = 1.2,
) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.015",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    axis.add_patch(box)
    axis.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        linespacing=1.25,
    )


def add_arrow(
    axis,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str = "",
    color: str = "#333333",
    text_offset: tuple[float, float] = (0.0, 0.0),
    linewidth: float = 1.5,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=linewidth,
        color=color,
        shrinkA=3,
        shrinkB=3,
    )
    axis.add_patch(arrow)
    if label:
        axis.text(
            (start[0] + end[0]) / 2 + text_offset[0],
            (start[1] + end[1]) / 2 + text_offset[1],
            label,
            ha="center",
            va="center",
            fontsize=8.6,
            color=color,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none"},
        )


def draw_spring(axis, start_x: float, end_x: float, y: float, coils: int = 6) -> None:
    width = end_x - start_x
    xs = [start_x, start_x + 0.06 * width]
    ys = [y, y]
    inner_start = start_x + 0.06 * width
    inner_end = end_x - 0.06 * width
    step = (inner_end - inner_start) / (2 * coils)
    for index in range(2 * coils + 1):
        xs.append(inner_start + index * step)
        ys.append(y + (0.035 if index % 2 else -0.035))
    xs.extend([end_x - 0.06 * width, end_x])
    ys.extend([y, y])
    axis.plot(xs, ys, color="#444444", linewidth=1.5)


def draw_thermal(axis) -> None:
    setup_panel(axis, "熱収支")
    add_box(axis, (0.36, 0.39), 0.26, 0.25, "物体\n$x=T$ [K]\n$C_{th}$ [J/K]", "#edf7ed", fontsize=9.0)
    add_box(axis, (0.06, 0.41), 0.18, 0.18, "ヒータ\n$u$ [W]", "#fff4e5", fontsize=9.0)
    add_box(axis, (0.72, 0.41), 0.20, 0.18, "周囲\n$T_a$ [K]", "#eef5ff")
    add_arrow(axis, (0.24, 0.50), (0.36, 0.50), "熱流 $u$", "#ff7f0e", (0.0, 0.07))
    add_arrow(axis, (0.62, 0.52), (0.72, 0.52), "流出\n$(T-T_a)/R_{th}$ [W]", "#d62728", (0.02, 0.10))
    add_arrow(axis, (0.48, 0.82), (0.48, 0.64), r"外乱 $d_q$ [W]", "#9467bd", (-0.08, 0.0))
    axis.text(
        0.05,
        0.16,
        r"$C_{th}\dot{T}=u+d_q-\dfrac{T-T_a}{R_{th}}$",
        ha="left",
        va="center",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#bbbbbb"},
    )


def draw_rc(axis) -> None:
    setup_panel(axis, "RC 回路")
    axis.plot([0.12, 0.22], [0.63, 0.63], color="#444444", linewidth=1.8)
    axis.plot([0.22, 0.28, 0.34, 0.40, 0.46], [0.63, 0.70, 0.56, 0.70, 0.63], color="#444444", linewidth=1.8)
    axis.plot([0.46, 0.64], [0.63, 0.63], color="#444444", linewidth=1.8)
    axis.plot([0.64, 0.64], [0.52, 0.74], color="#444444", linewidth=2.0)
    axis.plot([0.70, 0.70], [0.52, 0.74], color="#444444", linewidth=2.0)
    axis.plot([0.70, 0.84], [0.63, 0.63], color="#444444", linewidth=1.8)
    axis.plot([0.84, 0.84, 0.12, 0.12], [0.63, 0.28, 0.28, 0.63], color="#444444", linewidth=1.8)
    axis.text(0.34, 0.77, r"$R$ [$\Omega$]", ha="center", va="center", fontsize=9.2)
    axis.text(0.73, 0.78, r"$C_e$ [F]", ha="center", va="center", fontsize=9.2)
    axis.text(0.10, 0.74, r"$u=v_{in}$ [V]", ha="left", va="center", fontsize=9.2)
    axis.text(0.58, 0.46, r"状態 $x=v_C$ [V]", ha="left", va="center", fontsize=9.2)
    add_arrow(axis, (0.61, 0.86), (0.66, 0.74), r"外乱 $d_i$ [A]", "#9467bd", (-0.08, 0.0))
    axis.text(
        0.05,
        0.13,
        r"$C_e\dot{v}_C=\dfrac{u-v_C}{R}+d_i$",
        ha="left",
        va="center",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#bbbbbb"},
    )


def draw_mass_spring_damper(axis) -> None:
    setup_panel(axis, "質量ばねダンパ")
    axis.plot([0.08, 0.08], [0.30, 0.76], color="#444444", linewidth=2.2)
    for y_value in [0.34, 0.42, 0.50, 0.58, 0.66, 0.74]:
        axis.plot([0.04, 0.08], [y_value - 0.04, y_value], color="#777777", linewidth=1.1)
    draw_spring(axis, 0.10, 0.43, 0.62)
    axis.plot([0.12, 0.43], [0.42, 0.42], color="#444444", linewidth=1.5)
    axis.add_patch(Rectangle((0.24, 0.37), 0.12, 0.10, facecolor="#f7f7f7", edgecolor="#444444", linewidth=1.4))
    axis.plot([0.30, 0.30], [0.37, 0.31], color="#444444", linewidth=1.4)
    axis.plot([0.18, 0.42], [0.31, 0.31], color="#444444", linewidth=1.4)
    axis.add_patch(Rectangle((0.45, 0.47), 0.20, 0.25, facecolor="#edf7ed", edgecolor="#333333", linewidth=1.4))
    axis.text(0.55, 0.60, "質量 $M$ [kg]", ha="center", va="center", fontsize=9.5)
    axis.text(0.26, 0.73, "ばね $k$ [N/m]", ha="center", va="center", fontsize=8.8)
    axis.text(0.28, 0.24, "ダンパ $c$ [N s/m]", ha="center", va="center", fontsize=8.8)
    add_arrow(axis, (0.70, 0.60), (0.91, 0.60), r"入力 $u$ [N]", "#ff7f0e", (0.0, 0.06))
    add_arrow(axis, (0.64, 0.82), (0.83, 0.82), r"位置 $q$ [m]", "#1f77b4", (0.0, 0.05))
    add_arrow(axis, (0.67, 0.37), (0.82, 0.37), r"速度 $v=\dot{q}$ [m/s]", "#2ca02c", (0.02, -0.06))
    add_arrow(axis, (0.91, 0.44), (0.66, 0.52), r"外乱 $d_F$ [N]", "#9467bd", (0.03, -0.04))
    axis.text(
        0.05,
        0.12,
        r"$M\dot{v}=u+d_F-cv-kq,\quad \dot{q}=v$",
        ha="left",
        va="center",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#bbbbbb"},
    )


def draw_dc_motor(axis) -> None:
    setup_panel(axis, "DC モータ")
    add_box(axis, (0.06, 0.57), 0.18, 0.15, "端子電圧\n入力 $u=v_a$ [V]", "#fff4e5", fontsize=8.8)
    add_box(axis, (0.31, 0.57), 0.18, 0.15, "電機子\n電流 $i$ [A]\n$L,R$", "#eef5ff", fontsize=8.8)
    add_box(axis, (0.56, 0.57), 0.17, 0.15, "トルク\n$K_t i$ [N m]", "#f8f0ff", fontsize=8.8)
    add_box(axis, (0.77, 0.57), 0.17, 0.15, "軸\n$\\omega$ [rad/s]\n$\\theta$ [rad]", "#edf7ed", fontsize=8.8)
    add_arrow(axis, (0.24, 0.645), (0.31, 0.645), "", "#333333")
    add_arrow(axis, (0.49, 0.645), (0.56, 0.645), "", "#333333")
    add_arrow(axis, (0.73, 0.645), (0.77, 0.645), "", "#333333")
    add_arrow(axis, (0.80, 0.42), (0.55, 0.42), r"逆起電力 $K_e\omega$ [V]", "#d62728", (0.0, -0.07))
    add_arrow(axis, (0.95, 0.78), (0.89, 0.72), r"負荷 $\tau_L$ [N m]", "#9467bd", (-0.06, 0.0))
    axis.add_patch(Circle((0.855, 0.36), 0.065, facecolor="#f7f7f7", edgecolor="#444444", linewidth=1.3))
    axis.plot([0.855, 0.925], [0.36, 0.36], color="#444444", linewidth=1.5)
    add_arrow(axis, (0.79, 0.36), (0.82, 0.41), "", "#2ca02c", linewidth=1.2)
    axis.text(0.12, 0.30, r"$L\dot{i}=-Ri-K_e\omega+u$", ha="left", va="center", fontsize=9.5)
    axis.text(0.12, 0.18, r"$J\dot{\omega}=K_t i-b\omega-\tau_L,\quad \dot{\theta}=\omega$", ha="left", va="center", fontsize=9.5)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "ja" / "figures" / "physical_modeling_examples.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    configure_fonts()
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "figure.dpi": 180,
            "savefig.dpi": 180,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.0), constrained_layout=False)
    draw_thermal(axes[0, 0])
    draw_rc(axes[0, 1])
    draw_mass_spring_damper(axes[1, 0])
    draw_dc_motor(axes[1, 1])

    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.07, top=0.98, hspace=0.40, wspace=0.24)
    finalize_figure(fig, output_path, layout="none")
    print(output_path)


if __name__ == "__main__":
    main()
