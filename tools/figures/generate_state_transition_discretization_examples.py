#!/usr/bin/env python3
"""Generate state-transition and discretization examples for the textbook."""

from __future__ import annotations

import math
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("numpy is required to generate the state-transition figure") from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit(
        "matplotlib is required to generate the state-transition figure"
    ) from exc

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

A = np.array([[0.0, 1.0], [-2.0, -3.0]])
B = np.array([[0.0], [1.0]])
X0 = np.array([0.25, -0.35])
TIME_STOP = 6.0
SAMPLE_COUNT = 3001
TS_MAIN = 0.45
TS_VALUES = (0.15, 0.45, 0.90)
P = np.array([[1.0, 1.0], [0.45, -0.85]])
P_INV = np.linalg.inv(P)


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


def smooth_input(time: float | np.ndarray) -> float | np.ndarray:
    return 1.0 + 0.45 * np.sin(0.7 * math.pi * time)


def exp_a(delta: float) -> np.ndarray:
    a = math.exp(-delta)
    b = math.exp(-2.0 * delta)
    return np.array(
        [
            [2.0 * a - b, a - b],
            [-2.0 * a + 2.0 * b, -a + 2.0 * b],
        ]
    )


def bd(delta: float) -> np.ndarray:
    a = math.exp(-delta)
    b = math.exp(-2.0 * delta)
    return np.array([0.5 * (1.0 - 2.0 * a + b), a - b])


def derivative(time: float, state: np.ndarray) -> np.ndarray:
    return A @ state + B[:, 0] * float(smooth_input(time))


def reference_trajectory(times: np.ndarray) -> np.ndarray:
    trajectory = np.empty((len(times), 2))
    trajectory[0] = X0
    for index in range(len(times) - 1):
        time = float(times[index])
        step = float(times[index + 1] - times[index])
        state = trajectory[index]
        k1 = derivative(time, state)
        k2 = derivative(time + 0.5 * step, state + 0.5 * step * k1)
        k3 = derivative(time + 0.5 * step, state + 0.5 * step * k2)
        k4 = derivative(time + step, state + step * k3)
        trajectory[index + 1] = state + (step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return trajectory


def zoh_sampled_states(sample_period: float) -> tuple[np.ndarray, np.ndarray]:
    step_count = math.ceil(TIME_STOP / sample_period)
    sample_times = np.arange(step_count + 1, dtype=float) * sample_period
    states = np.empty((step_count + 1, 2))
    states[0] = X0
    ad = exp_a(sample_period)
    bd_step = bd(sample_period)
    for index in range(step_count):
        uk = float(smooth_input(sample_times[index]))
        states[index + 1] = ad @ states[index] + bd_step * uk
    return sample_times, states


def zoh_continuous_trajectory(
    sample_period: float, times: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_times, states = zoh_sampled_states(sample_period)
    trajectory = np.empty((len(times), 2))
    for index, time in enumerate(times):
        interval = min(int(math.floor(float(time) / sample_period)), len(states) - 2)
        tau = float(time - sample_times[interval])
        uk = float(smooth_input(sample_times[interval]))
        trajectory[index] = exp_a(tau) @ states[interval] + bd(tau) * uk
    return trajectory, sample_times, states


def add_curve_arrow(axis, curve: np.ndarray, color: str, index: int) -> None:
    start = curve[index]
    stop = curve[min(index + 16, len(curve) - 1)]
    delta = stop - start
    axis.arrow(
        start[0],
        start[1],
        delta[0],
        delta[1],
        shape="full",
        length_includes_head=True,
        head_width=0.035,
        head_length=0.055,
        linewidth=0.0,
        color=color,
    )


def draw_basis_vector(axis, vector: np.ndarray, label: str, color: str) -> None:
    axis.arrow(
        0.0,
        0.0,
        vector[0],
        vector[1],
        length_includes_head=True,
        head_width=0.04,
        head_length=0.06,
        linewidth=1.7,
        linestyle="--",
        color=color,
    )
    axis.text(
        1.08 * vector[0],
        1.08 * vector[1],
        label,
        color=color,
        fontsize=8,
        ha="center",
        va="center",
    )


def configure_grid(axis) -> None:
    axis.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.75)
    axis.tick_params(labelsize=8)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = (
        repo_root / "ja" / "figures" / "state_transition_discretization_examples.png"
    )
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

    times = np.linspace(0.0, TIME_STOP, SAMPLE_COUNT)
    reference = reference_trajectory(times)
    zoh_main, sample_times, sample_states = zoh_continuous_trajectory(TS_MAIN, times)
    visible = sample_times <= TIME_STOP + 1.0e-12
    visible_sample_times = sample_times[visible]
    visible_sample_states = sample_states[visible]

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.4), constrained_layout=False)

    time_axis = axes[0, 0]
    time_axis.plot(
        times,
        zoh_main[:, 0],
        color="#0072B2",
        linewidth=2.0,
        label=rf"ZOH 下の連続状態 $T_s={TS_MAIN:.2f}\,\mathrm{{s}}$",
    )
    hold_times = visible_sample_times
    hold_values = visible_sample_states[:, 0]
    if hold_times[-1] < TIME_STOP:
        hold_times = np.append(hold_times, TIME_STOP)
        hold_values = np.append(hold_values, hold_values[-1])
    time_axis.step(
        hold_times,
        hold_values,
        where="post",
        color="#D55E00",
        linestyle="--",
        linewidth=1.6,
        label="サンプル値の保持表示",
    )
    time_axis.plot(
        visible_sample_times,
        visible_sample_states[:, 0],
        marker="o",
        linestyle="none",
        markersize=4,
        color="#111111",
        label=r"離散状態 $x_k$",
    )
    time_axis.set_title("連続軌道とサンプル値")
    time_axis.set_xlabel(r"時間 $t$ [s]")
    time_axis.set_ylabel(r"変位状態 $x_1$ [-]")
    time_axis.set_xlim(0.0, TIME_STOP)
    configure_grid(time_axis)
    time_axis.legend(loc="lower right")

    error_axis = axes[0, 1]
    colors = ("#009E73", "#0072B2", "#CC79A7")
    for sample_period, color in zip(TS_VALUES, colors):
        zoh_traj, _, _ = zoh_continuous_trajectory(sample_period, times)
        error = np.linalg.norm(zoh_traj - reference, axis=1)
        error_axis.plot(
            times,
            error,
            color=color,
            linewidth=1.9,
            label=rf"$T_s={sample_period:.2f}\,\mathrm{{s}}$",
        )
    error_axis.set_title("サンプリング周期と近似誤差")
    error_axis.set_xlabel(r"時間 $t$ [s]")
    error_axis.set_ylabel(r"$\|x_{\mathrm{ZOH}}-x_{\mathrm{cont}}\|$ [-]")
    error_axis.set_xlim(0.0, TIME_STOP)
    error_axis.set_ylim(bottom=0.0)
    configure_grid(error_axis)
    error_axis.legend(loc="upper left")
    error_axis.text(
        0.98,
        0.96,
        r"$u(t)=1+0.45\sin(0.7\pi t)$",
        transform=error_axis.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#bbbbbb",
            "alpha": 0.88,
        },
    )

    phase_axis = axes[1, 0]
    phase_axis.plot(
        reference[:, 0],
        reference[:, 1],
        color="#777777",
        linestyle=":",
        linewidth=1.8,
        label="連続入力の基準軌道",
    )
    phase_axis.plot(
        zoh_main[:, 0],
        zoh_main[:, 1],
        color="#0072B2",
        linewidth=2.0,
        label="ZOH 下の連続軌道",
    )
    phase_axis.plot(
        visible_sample_states[:, 0],
        visible_sample_states[:, 1],
        color="#111111",
        marker="o",
        markersize=3.8,
        linewidth=1.0,
        label="サンプル点",
    )
    add_curve_arrow(phase_axis, zoh_main, "#0072B2", 820)
    draw_basis_vector(phase_axis, P[:, 0], r"$p_1$", "#D55E00")
    draw_basis_vector(phase_axis, P[:, 1], r"$p_2$", "#CC79A7")
    phase_axis.axhline(0.0, color="#777777", linewidth=0.8)
    phase_axis.axvline(0.0, color="#777777", linewidth=0.8)
    phase_axis.set_title("状態平面の軌道")
    phase_axis.set_xlabel(r"変位状態 $x_1$ [-]")
    phase_axis.set_ylabel(r"速度状態 $x_2$ [-]")
    phase_axis.set_aspect("equal", adjustable="box")
    configure_grid(phase_axis)
    phase_axis.legend(loc="lower right")

    transform_axis = axes[1, 1]
    zoh_z = (P_INV @ zoh_main.T).T
    transform_axis.plot(
        zoh_main[:, 0],
        zoh_main[:, 1],
        color="#0072B2",
        linewidth=2.0,
        label=r"元の座標 $x$",
    )
    transform_axis.plot(
        zoh_z[:, 0],
        zoh_z[:, 1],
        color="#D55E00",
        linestyle="--",
        linewidth=2.0,
        label=r"変換後 $z=P^{-1}x$",
    )
    transform_axis.plot(
        [zoh_main[0, 0], zoh_z[0, 0]],
        [zoh_main[0, 1], zoh_z[0, 1]],
        marker="s",
        linestyle="none",
        color="#111111",
        markersize=4,
        label="初期点",
    )
    transform_axis.plot(
        [zoh_main[-1, 0], zoh_z[-1, 0]],
        [zoh_main[-1, 1], zoh_z[-1, 1]],
        marker="x",
        linestyle="none",
        color="#555555",
        markersize=5,
        label="終端点",
    )
    add_curve_arrow(transform_axis, zoh_z, "#D55E00", 920)
    transform_axis.axhline(0.0, color="#777777", linewidth=0.8)
    transform_axis.axvline(0.0, color="#777777", linewidth=0.8)
    transform_axis.set_title("同じ軌道の座標表示")
    transform_axis.set_xlabel("第1成分 [-]")
    transform_axis.set_ylabel("第2成分 [-]")
    configure_grid(transform_axis)
    transform_axis.legend(loc="lower right")
    transform_axis.text(
        0.04,
        0.96,
        "$x=Pz$\n"
        "P = [[1, 1], [0.45, -0.85]]",
        transform=transform_axis.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#bbbbbb",
            "alpha": 0.88,
        },
    )

    finalize_figure(fig, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
