#!/usr/bin/env python3
"""Generate Flow 6 attitude and coordinate-transform geometry examples."""

from __future__ import annotations

from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("numpy is required to generate the attitude geometry figure") from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc, Circle, FancyArrowPatch, Polygon
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("matplotlib is required to generate the attitude geometry figure") from exc

from figure_style import finalize_figure


BLUE = "#4477AA"
GREEN = "#228833"
ORANGE = "#D55E00"
PURPLE = "#AA4499"
GRAY = "#666666"
LIGHT_BLUE = "#edf4fb"
LIGHT_GREEN = "#edf8f0"
LIGHT_ORANGE = "#fff3e8"


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        0.02,
        0.98,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=11.0,
        fontweight="bold",
        color="#222222",
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "#cccccc",
            "alpha": 0.95,
        },
        zorder=10,
    )


def draw_arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#333333",
    linestyle: str = "-",
    linewidth: float = 1.9,
    mutation_scale: float = 16.0,
    zorder: int = 4,
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=linewidth,
            color=color,
            linestyle=linestyle,
            shrinkA=0,
            shrinkB=0,
            zorder=zorder,
        )
    )


def rotation_2d(angle: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )


def draw_frame(
    axis: plt.Axes,
    origin: tuple[float, float],
    angle: float,
    *,
    scale: float,
    label: str,
    label_offset: tuple[float, float] = (-0.025, -0.045),
    label_ha: str = "right",
    label_va: str = "top",
    alpha: float = 1.0,
    linestyle: str = "-",
) -> None:
    origin_vec = np.asarray(origin, dtype=float)
    basis = rotation_2d(angle)
    x_axis = basis @ np.array([scale, 0.0])
    y_axis = basis @ np.array([0.0, scale])

    draw_arrow(
        axis,
        tuple(origin_vec),
        tuple(origin_vec + x_axis),
        color=ORANGE,
        linestyle=linestyle,
        linewidth=2.3,
        zorder=5,
    )
    draw_arrow(
        axis,
        tuple(origin_vec),
        tuple(origin_vec + y_axis),
        color=GREEN,
        linestyle=linestyle,
        linewidth=2.3,
        zorder=5,
    )
    axis.add_patch(
        Circle(
            tuple(origin_vec),
            0.016,
            facecolor=BLUE,
            edgecolor="white",
            linewidth=1.1,
            alpha=alpha,
            zorder=6,
        )
    )
    axis.text(
        *(origin_vec + np.asarray(label_offset, dtype=float)),
        label,
        ha=label_ha,
        va=label_va,
        fontsize=11.0,
        color="#222222",
        alpha=alpha,
        zorder=6,
    )
    axis.text(*(origin_vec + 1.08 * x_axis), "$x$", ha="center", va="center", fontsize=10.5)
    axis.text(*(origin_vec + 1.08 * y_axis), "$y$", ha="center", va="center", fontsize=10.5)


def draw_pitch_singularity(axis: plt.Axes) -> None:
    add_panel_label(axis, "(a) オイラー角特異性")
    theta_deg = np.linspace(-89.0, 89.0, 700)
    gain = 1.0 / np.abs(np.cos(np.deg2rad(theta_deg)))

    axis.plot(theta_deg, gain, color=BLUE, linewidth=3.0)
    axis.axvspan(84.0, 92.0, color=ORANGE, alpha=0.12)
    axis.axvspan(-92.0, -84.0, color=ORANGE, alpha=0.12)
    axis.axhline(1.0, color="#999999", linewidth=0.9, linestyle="--")

    marked_angles = np.array([0.0, 60.0, 80.0, 89.0])
    marked_gain = 1.0 / np.cos(np.deg2rad(marked_angles))
    axis.scatter(marked_angles, marked_gain, color=ORANGE, s=58, zorder=5)
    for angle, value in zip(marked_angles, marked_gain):
        axis.text(
            angle,
            value + (3.2 if angle > 0 else 1.8),
            rf"{angle:.0f}$^\circ$",
            ha="center",
            va="bottom",
            fontsize=10.0,
            color="#222222",
        )

    draw_arrow(axis, (-80.0, 48.0), (-55.0, 53.0), color=GRAY, linewidth=1.7)
    draw_arrow(axis, (-80.0, 48.0), (-55.0, 49.4), color=PURPLE, linewidth=1.7)
    axis.text(
        -78.5,
        42.5,
        "ロール軸とヨー軸が\nほぼ同じ方向になる",
        ha="left",
        va="top",
        fontsize=9.6,
        bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.92, "pad": 2.0},
    )

    axis.set_xlim(-92.0, 92.0)
    axis.text(
        0.68,
        0.81,
        r"$1/|\cos\theta|$ 増大",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=10.2,
        bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.92, "pad": 1.2},
    )

    axis.set_ylim(0.0, 68.0)
    axis.set_xlabel(r"ピッチ角 $\theta$ [deg]")
    axis.set_ylabel(r"角速度表示の増幅 $1/|\cos\theta|$")
    axis.grid(True, color="#d0d0d0", linewidth=0.8, alpha=0.78)


def draw_exp_log(axis: plt.Axes) -> None:
    add_panel_label(axis, "(b) SO(3) 局所ベクトル")
    axis.set_axis_off()
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_aspect("equal")

    plane = Polygon(
        [(0.48, 0.15), (0.94, 0.27), (0.82, 0.53), (0.36, 0.41)],
        closed=True,
        facecolor=LIGHT_BLUE,
        edgecolor=BLUE,
        linewidth=1.7,
        alpha=0.92,
        zorder=1,
    )
    axis.add_patch(plane)
    axis.text(0.76, 0.23, r"接空間 $T_{\hat{R}}\mathrm{SO}(3)$", fontsize=10.2, ha="center", va="center")

    origin = (0.27, 0.52)
    phi_angle = np.deg2rad(27.0)
    draw_frame(axis, origin, 0.0, scale=0.24, label=r"$\hat{R}$")
    draw_frame(
        axis,
        origin,
        phi_angle,
        scale=0.24,
        label=r"$R$",
        label_offset=(0.15, 0.06),
        label_ha="left",
        label_va="bottom",
        linestyle=":",
    )

    axis.add_patch(
        Arc(
            origin,
            0.39,
            0.39,
            theta1=0.0,
            theta2=np.rad2deg(phi_angle),
            color=PURPLE,
            linewidth=2.2,
            zorder=6,
        )
    )
    axis.text(0.48, 0.58, r"$\|\varphi\|$", fontsize=10.6, color=PURPLE, ha="center")

    draw_arrow(axis, (0.54, 0.30), (0.77, 0.43), color=PURPLE, linewidth=2.3)
    axis.text(0.71, 0.47, r"$\varphi=\theta a$", fontsize=11.0, color=PURPLE, ha="center")
    axis.text(0.20, 0.26, "回転軸 $a$ は\n紙面外方向", fontsize=9.7, ha="center")
    axis.add_patch(Circle((0.27, 0.52), 0.040, facecolor="white", edgecolor=BLUE, linewidth=1.5, zorder=4))
    axis.text(0.27, 0.52, r"$a$", fontsize=10.0, ha="center", va="center", color=BLUE, zorder=5)

    draw_arrow(axis, (0.72, 0.44), (0.50, 0.62), color=BLUE, linewidth=1.9)
    axis.text(0.66, 0.69, r"$\operatorname{Exp}$ で回転へ戻す", fontsize=10.0, color=BLUE, ha="center")
    axis.text(
        0.34,
        0.10,
        r"$R=\hat{R}\operatorname{Exp}(\widehat{\varphi})$",
        fontsize=10.0,
        ha="center",
        va="center",
        bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.94, "pad": 1.4},
    )


def draw_quaternion_double_cover(axis: plt.Axes) -> None:
    add_panel_label(axis, "(c) 二重被覆")
    axis.set_axis_off()
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_aspect("equal")

    center = np.array([0.50, 0.49])
    radius = 0.31
    axis.add_patch(Circle(tuple(center), radius, facecolor="white", edgecolor="#777777", linewidth=1.8))
    axis.text(0.73, 0.93, r"単位球面断面", ha="center", va="center", fontsize=9.5)

    angle = np.deg2rad(34.0)
    point_q = center + radius * np.array([np.cos(angle), np.sin(angle)])
    point_minus_q = center - radius * np.array([np.cos(angle), np.sin(angle)])
    axis.plot([point_q[0], point_minus_q[0]], [point_q[1], point_minus_q[1]], color="#bbbbbb", linestyle="--", linewidth=1.6)
    axis.scatter([point_q[0], point_minus_q[0]], [point_q[1], point_minus_q[1]], s=72, color=[BLUE, ORANGE], zorder=5)
    axis.text(point_q[0] + 0.040, point_q[1] + 0.025, r"$q$", fontsize=12.0, color=BLUE)
    axis.text(point_minus_q[0] - 0.052, point_minus_q[1] - 0.040, r"$-q$", fontsize=12.0, color=ORANGE)
    axis.text(0.50, 0.52, "同じ姿勢\n$R(q)=R(-q)$", ha="center", va="center", fontsize=10.5)

    previous = center + radius * np.array([np.cos(angle - 0.22), np.sin(angle - 0.22)])
    corrected = center + radius * np.array([np.cos(angle + 0.13), np.sin(angle + 0.13)])
    raw = center - radius * np.array([np.cos(angle + 0.13), np.sin(angle + 0.13)])
    axis.scatter([previous[0], corrected[0]], [previous[1], corrected[1]], s=48, color=GREEN, zorder=5)
    draw_arrow(axis, tuple(previous), tuple(corrected), color=GREEN, linewidth=2.0, mutation_scale=13)
    axis.scatter([raw[0]], [raw[1]], s=52, color=ORANGE, marker="x", zorder=6)
    draw_arrow(axis, tuple(raw), tuple(corrected), color=PURPLE, linestyle=":", linewidth=1.9, mutation_scale=13)
    axis.text(
        0.50,
        0.13,
        r"$q_{k-1}^{\mathsf{T}}q_k<0$ なら符号を反転して連続化",
        ha="center",
        va="center",
        fontsize=9.8,
        bbox={"facecolor": LIGHT_GREEN, "edgecolor": GREEN, "alpha": 0.96, "pad": 2.0},
    )


def draw_local_error(axis: plt.Axes) -> None:
    add_panel_label(axis, "(d) 局所姿勢誤差")
    axis.set_axis_off()
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_aspect("equal")

    origin = (0.30, 0.58)
    nominal_angle = np.deg2rad(8.0)
    true_angle = np.deg2rad(32.0)
    draw_frame(axis, origin, nominal_angle, scale=0.24, label=r"$\hat{R}$", label_offset=(-0.050, -0.060))
    draw_frame(
        axis,
        origin,
        true_angle,
        scale=0.24,
        label=r"$R$",
        label_offset=(0.14, 0.060),
        label_ha="left",
        label_va="bottom",
        linestyle=":",
    )

    axis.add_patch(
        Arc(
            origin,
            0.46,
            0.46,
            theta1=np.rad2deg(nominal_angle),
            theta2=np.rad2deg(true_angle),
            color=PURPLE,
            linewidth=2.4,
            zorder=5,
        )
    )
    axis.text(0.54, 0.72, r"$\tilde{R}=\hat{R}^{\mathsf{T}}R$", fontsize=10.6, color=PURPLE, ha="center")

    plane = Polygon(
        [(0.54, 0.25), (0.94, 0.34), (0.82, 0.57), (0.42, 0.48)],
        closed=True,
        facecolor=LIGHT_ORANGE,
        edgecolor=ORANGE,
        linewidth=1.7,
        alpha=0.94,
        zorder=1,
    )
    axis.add_patch(plane)
    draw_arrow(axis, (0.56, 0.34), (0.81, 0.45), color=ORANGE, linewidth=2.4, zorder=3)
    axis.text(0.78, 0.51, r"$\delta\theta$", fontsize=11.5, color=ORANGE, ha="center")
    axis.text(0.31, 0.31, "近傍 3 成分で\n線形化", fontsize=9.6, ha="center")

    axis.text(
        0.52,
        0.10,
        r"$\delta\theta=\operatorname{Log}(\hat{R}^{\mathsf{T}}R)^\vee$"
        "\n"
        r"$\hat{R}\leftarrow\hat{R}\operatorname{Exp}(\widehat{\delta\theta})$ 後に誤差平均を 0 へ戻す",
        ha="center",
        va="center",
        fontsize=9.3,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.96, "pad": 2.0},
        zorder=8,
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "ja" / "figures" / "flow6_attitude_geometry_examples.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 9,
            "font.family": ["Noto Sans CJK JP", "Noto Sans", "DejaVu Sans"],
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "mathtext.fontset": "dejavusans",
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.unicode_minus": False,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.9), constrained_layout=False)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.070, top=0.975, wspace=0.18, hspace=0.38)

    draw_pitch_singularity(axes[0, 0])
    draw_exp_log(axes[0, 1])
    draw_quaternion_double_cover(axes[1, 0])
    draw_local_error(axes[1, 1])

    finalize_figure(fig, output_path, layout="none")
    print(output_path)


if __name__ == "__main__":
    main()
