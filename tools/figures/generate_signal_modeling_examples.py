#!/usr/bin/env python3
"""Generate signal-role, linearization, and Euler-step examples."""

from __future__ import annotations

from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("numpy is required to generate the signal-modeling figure") from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("matplotlib is required to generate the signal-modeling figure") from exc


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

T_AMBIENT = 20.0
R_TH = 2.0
C_TH = 10.0
ETA = 0.5
V_E = 2.0
T_E = T_AMBIENT + R_TH * ETA * V_E**2
VALID_V_DEVIATION = 0.4
TAU = 0.5
EULER_STEPS = (0.10, 0.60, 1.20)


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


def add_box(
    axis,
    xy,
    width,
    height,
    text,
    facecolor,
    edgecolor="#333333",
    fontsize=10,
) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        linewidth=1.2,
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
        linespacing=1.35,
    )


def add_arrow(axis, start, end, label, color="#333333", text_offset=(0.0, 0.0)) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.5,
        color=color,
        shrinkA=4,
        shrinkB=4,
    )
    axis.add_patch(arrow)
    if label:
        axis.text(
            (start[0] + end[0]) / 2 + text_offset[0],
            (start[1] + end[1]) / 2 + text_offset[1],
            label,
            ha="center",
            va="center",
            fontsize=9,
            color=color,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none"},
        )


def draw_signal_path(axis) -> None:
    axis.set_axis_off()
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_title("信号の役割と入る位置", loc="left", fontweight="bold")

    add_box(axis, (0.04, 0.62), 0.13, 0.18, "参照信号\n$r=25\\,^{\\circ}$C", "#eef5ff")
    add_box(axis, (0.22, 0.62), 0.15, 0.18, "制御器\n$e=r-y_m$", "#f8f0ff")
    add_box(axis, (0.43, 0.62), 0.17, 0.18, "アクチュエータ\n$u_c\\to u$", "#fff5e8")
    add_box(
        axis,
        (0.66, 0.56),
        0.20,
        0.30,
        "対象\n状態 $x=T$\n出力 $y=T$",
        "#edf7ed",
    )
    add_box(axis, (0.69, 0.18), 0.17, 0.17, "センサ\n$y_m=y+n_T$", "#f7f7f7")

    add_arrow(axis, (0.17, 0.71), (0.22, 0.71), "", "#1f77b4")
    add_arrow(axis, (0.37, 0.71), (0.43, 0.71), "指令 $u_c$ [-]", "#9467bd", (0.0, -0.075))
    add_arrow(axis, (0.60, 0.71), (0.66, 0.71), "入力 $u$ [W]", "#ff7f0e", (0.0, -0.075))
    add_arrow(axis, (0.86, 0.68), (0.96, 0.68), "出力 $y$ [$^{\\circ}$C]", "#2ca02c", (0.0, 0.07))
    add_arrow(axis, (0.78, 0.56), (0.78, 0.35), "", "#2ca02c")
    add_arrow(axis, (0.69, 0.27), (0.37, 0.62), "測定 $y_m$ [$^{\\circ}$C]", "#555555", (-0.04, -0.05))
    add_arrow(axis, (0.74, 0.93), (0.74, 0.86), "外乱 $d=(T_a,d_q)$\n[$^{\\circ}$C, W]", "#d62728", (-0.02, 0.02))
    add_arrow(axis, (0.55, 0.24), (0.69, 0.27), "ノイズ $n_T$ [$^{\\circ}$C]", "#8c564b", (-0.01, -0.06))

    axis.text(
        0.04,
        0.19,
        (
            "例: $R_{th}=2.0$ K/W, $C_{th}=10$ J/K, "
            "$\\eta=0.5$ W/V$^2$, $T_a=20\\,^{\\circ}$C"
        ),
        ha="left",
        va="center",
        fontsize=9,
        color="#333333",
    )
    axis.text(
        0.04,
        0.08,
        r"対象式: $C_{th}\dot{T}=-(T-T_a)/R_{th}+u+d_q,\quad y_m=y+n_T$",
        ha="left",
        va="center",
        fontsize=9,
        color="#333333",
    )


def draw_linearization(axis) -> None:
    deviations = np.linspace(-1.6, 1.6, 401)
    exact = ETA * ((V_E + deviations) ** 2 - V_E**2) / C_TH
    linear = 2.0 * ETA * V_E * deviations / C_TH

    axis.axvspan(
        -VALID_V_DEVIATION,
        VALID_V_DEVIATION,
        color="#e8f4ff",
        alpha=0.85,
        label=rf"一次近似の有効域 $|\tilde{{v}}|\leq{VALID_V_DEVIATION:.1f}$ V",
    )
    axis.plot(deviations, exact, color="#1f77b4", linewidth=2.0, label="厳密な入力寄与")
    axis.plot(deviations, linear, color="#d62728", linestyle="--", linewidth=2.0, label="一次 Taylor 近似")
    axis.axhline(0.0, color="#666666", linewidth=0.8)
    axis.axvline(0.0, color="#666666", linewidth=0.8)
    axis.set_title("平衡点・偏差変数・線形化誤差", fontweight="bold")
    axis.set_xlabel(r"入力偏差 $\tilde{v}=v-v_e$ [V]")
    axis.set_ylabel(r"ヒータ入力寄与 $\Delta\dot{T}_{u}$ [K/s]")
    axis.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)
    axis.set_xlim(-1.6, 1.6)
    axis.set_ylim(-0.55, 0.85)
    axis.legend(loc="upper left", fontsize=8)
    axis.text(
        0.03,
        0.04,
        (
            rf"$v_e={V_E:.1f}$ V, $T_e={T_E:.1f}\,^{{\circ}}$C" "\n"
            rf"$\Delta\dot{{T}}_u\simeq(2\eta v_e/C_{{th}})\tilde{{v}}$"
            "\n冷却項 $-\\tilde{T}/(R_{th}C_{th})$ は除く"
        ),
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#bbbbbb"},
    )


def euler_decay_points(step: float, stop: float) -> tuple[np.ndarray, np.ndarray]:
    times = np.arange(0.0, stop + 0.5 * step, step)
    values = (1.0 - step / TAU) ** np.arange(len(times))
    return times, values


def draw_euler_step(axis) -> None:
    stop = 3.0
    time = np.linspace(0.0, stop, 500)
    exact = np.exp(-time / TAU)
    colors = ("#1f77b4", "#ff7f0e", "#d62728")

    axis.plot(time, exact, color="#222222", linewidth=2.0, label=r"厳密解 $e^{-t/\tau}$")
    for step, color in zip(EULER_STEPS, colors):
        times, values = euler_decay_points(step, stop)
        axis.step(
            times,
            values,
            where="post",
            color=color,
            linewidth=1.8,
            label=rf"Euler $h={step:.2f}$ s, $1-h/\tau={1-step/TAU:.1f}$",
        )
        axis.plot(times, values, "o", color=color, markersize=3.4)

    axis.axhline(0.0, color="#666666", linewidth=0.8)
    axis.axvline(2.0 * TAU, color="#999999", linestyle=":", linewidth=1.2)
    axis.text(
        2.0 * TAU + 0.03,
        1.05,
        r"安定条件 $h<2\tau$",
        color="#555555",
        fontsize=8,
        ha="left",
        va="center",
    )
    axis.set_title("一次 ODE と刻み幅の直観", fontweight="bold")
    axis.set_xlabel(r"時間 $t$ [s]")
    axis.set_ylabel(r"偏差 $z(t)$ [V]")
    axis.set_xlim(0.0, stop)
    axis.set_ylim(-1.4, 1.35)
    axis.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)
    axis.legend(loc="lower left", fontsize=8)
    axis.text(
        0.98,
        0.94,
        rf"$\dot{{z}}=-z/\tau$, $\tau={TAU:.2f}$ s, $z_0=1.0$ V",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#bbbbbb"},
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "ja" / "figures" / "signal_modeling_examples.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    configure_fonts()
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "figure.dpi": 180,
            "savefig.dpi": 180,
        }
    )

    fig = plt.figure(figsize=(8.0, 6.0), constrained_layout=False)
    grid = fig.add_gridspec(2, 2, height_ratios=(0.95, 1.05), hspace=0.42, wspace=0.28)
    signal_axis = fig.add_subplot(grid[0, :])
    linearization_axis = fig.add_subplot(grid[1, 0])
    euler_axis = fig.add_subplot(grid[1, 1])

    draw_signal_path(signal_axis)
    draw_linearization(linearization_axis)
    draw_euler_step(euler_axis)

    fig.suptitle("信号の役割、偏差変数、局所近似、刻み幅", fontsize=13, fontweight="bold")
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.08, top=0.90)
    fig.savefig(output_path, facecolor="white")
    print(output_path)


if __name__ == "__main__":
    main()
