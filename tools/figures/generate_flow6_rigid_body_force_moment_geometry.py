#!/usr/bin/env python3
"""Generate Flow 6 rigid-body force, moment, and inertia geometry."""

from __future__ import annotations

from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("numpy is required to generate the rigid-body geometry figure") from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc, Circle, Ellipse, FancyArrowPatch, Rectangle
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("matplotlib is required to generate the rigid-body geometry figure") from exc

from figure_style import finalize_figure


BLUE = "#4477AA"
GREEN = "#228833"
ORANGE = "#D55E00"
PURPLE = "#AA4499"
GRAY = "#555555"
LIGHT_BLUE = "#eef4fb"
LIGHT_GREEN = "#eef8f0"
LIGHT_ORANGE = "#fff3e8"


def rotation_2d(angle_rad: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ]
    )


def draw_arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#222222",
    linewidth: float = 1.8,
    linestyle: str = "-",
    mutation_scale: float = 12.0,
    zorder: int = 4,
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            mutation_scale=mutation_scale,
            shrinkA=0.0,
            shrinkB=0.0,
            zorder=zorder,
        )
    )


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        0.02,
        0.97,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9.0,
        fontweight="bold",
        color="#222222",
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": "#cccccc",
            "alpha": 0.96,
        },
        zorder=20,
    )


def setup_geometry_axis(axis: plt.Axes) -> None:
    axis.set_xlim(-1.18, 1.18)
    axis.set_ylim(-1.02, 1.08)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#dddddd")


def draw_frame(
    axis: plt.Axes,
    origin: np.ndarray,
    angle: float,
    *,
    length: float,
    label: str,
    alpha: float = 1.0,
    linestyle: str = "-",
    origin_label_offset: tuple[float, float] = (-0.045, -0.06),
    origin_label_ha: str = "right",
    origin_label_va: str = "top",
) -> None:
    basis = rotation_2d(angle)
    x_vec = basis @ np.array([length, 0.0])
    y_vec = basis @ np.array([0.0, length])
    draw_arrow(axis, tuple(origin), tuple(origin + x_vec), color=ORANGE, linestyle=linestyle, zorder=5)
    draw_arrow(axis, tuple(origin), tuple(origin + y_vec), color=GREEN, linestyle=linestyle, zorder=5)
    axis.add_patch(Circle(tuple(origin), 0.028, facecolor="white", edgecolor="#222222", linewidth=1.0, zorder=6, alpha=alpha))
    axis.text(*(origin + 1.11 * x_vec), rf"$x^{{{label}}}$", ha="center", va="center", fontsize=8.2, color=ORANGE)
    axis.text(*(origin + 1.11 * y_vec), rf"$y^{{{label}}}$", ha="center", va="center", fontsize=8.2, color=GREEN)
    axis.text(
        origin[0] + origin_label_offset[0],
        origin[1] + origin_label_offset[1],
        rf"$\{{{label}\}}$",
        ha=origin_label_ha,
        va=origin_label_va,
        fontsize=8.2,
    )


def draw_force_transform(axis: plt.Axes) -> None:
    add_panel_label(axis, "A  force transform")
    setup_geometry_axis(axis)
    origin = np.array([0.0, -0.03])
    yaw = np.deg2rad(30.0)
    draw_frame(axis, origin, 0.0, length=0.58, label="I", alpha=0.8, origin_label_offset=(-0.06, -0.07))
    draw_frame(
        axis,
        origin,
        yaw,
        length=0.70,
        label="B",
        linestyle=":",
        origin_label_offset=(0.07, -0.14),
        origin_label_ha="left",
    )

    force_body = rotation_2d(yaw) @ np.array([0.82, 0.0])
    draw_arrow(axis, tuple(origin), tuple(origin + force_body), color=BLUE, linewidth=2.6, mutation_scale=14)
    draw_arrow(axis, tuple(origin), tuple(origin + np.array([0.71, 0.41])), color=PURPLE, linewidth=1.8, linestyle="--")
    axis.text(0.64, 0.39, r"$f^I=R^I_B f^B$", color=PURPLE, ha="center", va="bottom", fontsize=8.4)
    axis.text(0.57, 0.17, r"$f^B$", color=BLUE, ha="center", va="bottom", fontsize=9.2)
    axis.add_patch(Arc(tuple(origin), 0.46, 0.46, theta1=0.0, theta2=30.0, color=GRAY, linewidth=1.2))
    axis.text(0.28, 0.04, r"$\psi=30^\circ$", ha="left", va="center", fontsize=8.2, color=GRAY)
    axis.text(
        -0.98,
        -0.77,
        r"translation uses $m\dot{v}^I=mg^I+R^I_B f^B$",
        fontsize=8.0,
        ha="left",
        va="center",
        bbox={"facecolor": LIGHT_BLUE, "edgecolor": BLUE, "alpha": 0.95, "pad": 2.0},
    )


def draw_moment_cross_product(axis: plt.Axes) -> None:
    add_panel_label(axis, "B  arm cross product")
    setup_geometry_axis(axis)
    origin = np.array([-0.04, -0.12])
    draw_frame(axis, origin, 0.0, length=0.62, label="B")
    contact = origin + np.array([0.0, 0.45])
    draw_arrow(axis, tuple(origin), tuple(contact), color=GREEN, linewidth=2.1)
    draw_arrow(axis, tuple(contact), tuple(contact + np.array([0.72, 0.0])), color=BLUE, linewidth=2.6, mutation_scale=14)
    axis.add_patch(Circle(tuple(contact), 0.035, facecolor="white", edgecolor=BLUE, linewidth=1.0, zorder=6))
    axis.text(contact[0] - 0.08, contact[1] + 0.03, r"$r^B$", color=GREEN, ha="right", va="center", fontsize=9.0)
    axis.text(contact[0] + 0.40, contact[1] + 0.08, r"$f^B$", color=BLUE, ha="center", va="bottom", fontsize=9.0)

    symbol_center = np.array([-0.54, 0.34])
    axis.add_patch(Circle(tuple(symbol_center), 0.105, facecolor="white", edgecolor=PURPLE, linewidth=1.5, zorder=5))
    axis.plot([symbol_center[0] - 0.055, symbol_center[0] + 0.055], [symbol_center[1] - 0.055, symbol_center[1] + 0.055], color=PURPLE, linewidth=1.5, zorder=6)
    axis.plot([symbol_center[0] - 0.055, symbol_center[0] + 0.055], [symbol_center[1] + 0.055, symbol_center[1] - 0.055], color=PURPLE, linewidth=1.5, zorder=6)
    axis.text(
        symbol_center[0],
        0.06,
        r"$\tau^B=r^B\times f^B$"
        "\n"
        r"$y^B\times x^B=-z^B$",
        ha="center",
        va="center",
        fontsize=8.3,
        color=PURPLE,
    )
    axis.text(
        -0.98,
        -0.77,
        r"order matters: $f^B\times r^B$ flips the sign",
        fontsize=8.0,
        ha="left",
        va="center",
        bbox={"facecolor": LIGHT_ORANGE, "edgecolor": ORANGE, "alpha": 0.95, "pad": 2.0},
    )


def draw_inertia_mapping(axis: plt.Axes) -> None:
    add_panel_label(axis, "C  torque to angular accel")
    axis.set_axis_off()
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)

    box_specs = [
        ((0.07, 0.62), 0.27, 0.18, r"$\tau^B$", "[N m]", PURPLE),
        ((0.38, 0.62), 0.27, 0.18, r"$(J^B)^{-1}$", "[1/(kg m^2)]", BLUE),
        ((0.69, 0.62), 0.27, 0.18, r"$\dot{\omega}^B$", "[rad/s^2]", ORANGE),
    ]
    for (xy, width, height, title, unit, color) in box_specs:
        axis.add_patch(Rectangle(xy, width, height, facecolor="white", edgecolor=color, linewidth=1.4))
        axis.text(xy[0] + width / 2.0, xy[1] + 0.115, title, ha="center", va="center", fontsize=10.5, color=color)
        axis.text(xy[0] + width / 2.0, xy[1] + 0.045, unit, ha="center", va="center", fontsize=7.7, color=GRAY)
    draw_arrow(axis, (0.34, 0.71), (0.38, 0.71), color=GRAY, linewidth=1.4)
    draw_arrow(axis, (0.65, 0.71), (0.69, 0.71), color=GRAY, linewidth=1.4)

    axis.text(0.50, 0.49, r"$J^B\dot{\omega}^B=\tau^B-\omega^B\times J^B\omega^B$", ha="center", va="center", fontsize=9.5)
    axis.text(0.20, 0.28, r"current state $\omega^B$ [rad/s]", ha="center", va="center", fontsize=8.2, color=BLUE)
    axis.add_patch(Arc((0.20, 0.19), 0.26, 0.20, theta1=20.0, theta2=315.0, color=BLUE, linewidth=1.6))
    draw_arrow(axis, (0.30, 0.14), (0.32, 0.19), color=BLUE, linewidth=1.6, mutation_scale=10)
    axis.text(0.73, 0.26, r"change rate $\dot{\omega}^B$ [rad/s$^2$]", ha="center", va="center", fontsize=8.2, color=ORANGE)
    draw_arrow(axis, (0.60, 0.14), (0.88, 0.14), color=ORANGE, linewidth=1.8)
    axis.text(0.50, 0.06, r"with $\omega^B=0$:  $\dot{\omega}^B=(J^B)^{-1}\tau^B$", ha="center", va="center", fontsize=8.4)


def draw_inertia_coupling(axis: plt.Axes) -> None:
    add_panel_label(axis, "D  inertia-axis coupling")
    setup_geometry_axis(axis)
    origin = np.array([0.0, -0.04])
    draw_frame(axis, origin, 0.0, length=0.64, label="B")
    axis.add_patch(Ellipse(tuple(origin), 1.20, 0.55, angle=34.0, facecolor=LIGHT_GREEN, edgecolor=GREEN, linewidth=1.4, alpha=0.96, zorder=1))
    principal = rotation_2d(np.deg2rad(34.0))
    p1 = principal @ np.array([0.72, 0.0])
    p2 = principal @ np.array([0.0, 0.36])
    draw_arrow(axis, tuple(origin), tuple(origin + p1), color=GREEN, linewidth=1.8)
    draw_arrow(axis, tuple(origin), tuple(origin + p2), color=GREEN, linewidth=1.5)
    axis.text(*(origin + 1.08 * p1), r"$e_1$", ha="center", va="center", fontsize=8.2, color=GREEN)
    axis.text(*(origin + 1.14 * p2), r"$e_2$", ha="center", va="center", fontsize=8.2, color=GREEN)
    draw_arrow(axis, (-0.68, -0.58), (-0.24, -0.58), color=PURPLE, linewidth=2.2)
    draw_arrow(axis, (-0.68, -0.58), (-0.39, -0.36), color=ORANGE, linewidth=2.2)
    axis.text(-0.22, -0.58, r"$\tau_x^B$", color=PURPLE, va="center", fontsize=8.6)
    axis.text(-0.40, -0.32, r"$\dot{\omega}^B$", color=ORANGE, ha="center", fontsize=8.6)
    axis.text(
        0.03,
        0.67,
        r"$J^B:\ J_{xy}=J_{yx}\ne0$"
        "\n"
        r"$J_{xy}\ne0$ couples axes",
        ha="center",
        va="center",
        fontsize=8.1,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.94, "pad": 1.7},
    )


def draw_bookkeeping(axis: plt.Axes) -> None:
    add_panel_label(axis, "E  frame bookkeeping")
    axis.set_axis_off()
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    rows = [
        (0.69, r"$f^B$", r"B", r"force input"),
        (0.53, r"$R^I_B f^B$", r"I", r"translational eq."),
        (0.37, r"$r^B\times f^B$", r"B", r"moment about CG"),
        (0.21, r"$J^B,\ \omega^B,\ \dot{\omega}^B$", r"B", r"Euler equation"),
    ]
    headers = ("quantity", "frame", "use")
    xs = (0.20, 0.50, 0.77)
    for x, header in zip(xs, headers):
        axis.text(x, 0.82, header, ha="center", va="center", fontsize=8.3, fontweight="bold", color="#222222")
    axis.plot([0.05, 0.95], [0.77, 0.77], color="#bbbbbb", linewidth=1.0)
    for y, quantity, frame, use in rows:
        axis.add_patch(Rectangle((0.05, y - 0.055), 0.90, 0.105, facecolor="#fafafa", edgecolor="#dddddd", linewidth=0.8))
        axis.text(xs[0], y, quantity, ha="center", va="center", fontsize=8.8)
        axis.text(xs[1], y, frame, ha="center", va="center", fontsize=8.5, color=BLUE if frame == "I" else GREEN)
        axis.text(xs[2], y, use, ha="center", va="center", fontsize=7.9)
    axis.text(
        0.50,
        0.055,
        "sum and cross product only after matching frames",
        ha="center",
        va="center",
        fontsize=8.1,
        bbox={"facecolor": LIGHT_BLUE, "edgecolor": BLUE, "alpha": 0.95, "pad": 2.0},
    )


def draw_numeric_check(axis: plt.Axes) -> None:
    add_panel_label(axis, "F  example numeric check")
    axis.set_axis_off()
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    checks = [
        (0.78, r"$m=2,\ \psi=30^\circ,\ f^B=[4,0,0]^\mathsf{T}$"),
        (0.62, r"$\dot{v}^I=R^I_B f^B/m=[\sqrt{3},1,0]^\mathsf{T}$"),
        (0.46, r"$r^B=[0,0.2,0]^\mathsf{T}\Rightarrow \tau^B=[0,0,-0.8]^\mathsf{T}$"),
        (0.30, r"$J_z=0.3\Rightarrow \dot{\omega}_z^B=-0.8/0.3\simeq -2.67$"),
    ]
    for y, text in checks:
        axis.add_patch(Rectangle((0.06, y - 0.055), 0.88, 0.11, facecolor="white", edgecolor="#dddddd", linewidth=0.9))
        axis.text(0.50, y, text, ha="center", va="center", fontsize=8.1)
    axis.text(
        0.50,
        0.105,
        r"dropping $R^I_B$, $r^B$, or $J^B$ changes the plant",
        ha="center",
        va="center",
        fontsize=7.9,
        bbox={"facecolor": LIGHT_ORANGE, "edgecolor": ORANGE, "alpha": 0.95, "pad": 2.0},
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "ja" / "figures" / "flow6_rigid_body_force_moment_geometry.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 8.4,
            "font.family": ["DejaVu Sans"],
            "axes.labelsize": 8.6,
            "legend.fontsize": 8.0,
            "mathtext.fontset": "dejavusans",
            "figure.dpi": 180,
            "savefig.dpi": 180,
            "axes.unicode_minus": False,
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(9.0, 6.1), constrained_layout=False)
    fig.subplots_adjust(left=0.045, right=0.985, bottom=0.055, top=0.985, wspace=0.16, hspace=0.20)

    draw_force_transform(axes[0, 0])
    draw_moment_cross_product(axes[0, 1])
    draw_inertia_mapping(axes[0, 2])
    draw_inertia_coupling(axes[1, 0])
    draw_bookkeeping(axes[1, 1])
    draw_numeric_check(axes[1, 2])

    finalize_figure(fig, output_path, layout="none")
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
