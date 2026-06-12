#!/usr/bin/env python3
"""Generate nonlinear-control visual verification examples for Flow 6."""

from __future__ import annotations

from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("numpy is required to generate the nonlinear-control figure") from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("matplotlib is required to generate the nonlinear-control figure") from exc

from figure_style import finalize_figure


G = 9.8
DAMPING = 0.2

LOCAL_K1 = 15.8
LOCAL_K2 = 4.8
LOCAL_INPUT_LIMIT = 6.0
LOCAL_VALID_PHI = 0.40
LOCAL_VALID_OMEGA = 1.20

HOLD_KP = 4.0
HOLD_KD = 2.2
HOLD_INPUT_LIMIT = 8.0
HOLD_TARGET = 0.85

BLUE = "#4477AA"
ORANGE = "#D55E00"
GREEN = "#228833"
PURPLE = "#AA4499"
GRAY = "#555555"
RED = "#CC6677"


def rk4_step(state: np.ndarray, step_size: float, rhs) -> np.ndarray:
    k1 = rhs(state)
    k2 = rhs(state + 0.5 * step_size * k1)
    k3 = rhs(state + 0.5 * step_size * k2)
    k4 = rhs(state + step_size * k3)
    return state + (step_size / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def local_feedback(phi: float, omega: float) -> tuple[float, float]:
    command = -LOCAL_K1 * phi - LOCAL_K2 * omega
    actual = float(np.clip(command, -LOCAL_INPUT_LIMIT, LOCAL_INPUT_LIMIT))
    return command, actual


def upright_rhs(state: np.ndarray) -> np.ndarray:
    phi, omega = state
    _, input_torque = local_feedback(float(phi), float(omega))
    return np.array([omega, G * np.sin(phi) - DAMPING * omega + input_torque])


def simulate_upright(initial_state: tuple[float, float]) -> dict[str, np.ndarray]:
    time = np.linspace(0.0, 5.0, 1001)
    step_size = float(time[1] - time[0])
    states = np.zeros((time.size, 2))
    commands = np.zeros(time.size)
    inputs = np.zeros(time.size)
    states[0] = np.array(initial_state, dtype=float)

    for index in range(time.size):
        commands[index], inputs[index] = local_feedback(states[index, 0], states[index, 1])
        if index + 1 < time.size:
            states[index + 1] = rk4_step(states[index], step_size, upright_rhs)

    saturated = np.abs(commands) > LOCAL_INPUT_LIMIT + 1.0e-9
    return {"time": time, "states": states, "commands": commands, "inputs": inputs, "saturated": saturated}


def hold_command(theta: float, omega: float, target: float, *, compensated: bool) -> tuple[float, float]:
    bias = G * np.sin(target) if compensated else 0.0
    command = bias - HOLD_KP * (theta - target) - HOLD_KD * omega
    actual = float(np.clip(command, -HOLD_INPUT_LIMIT, HOLD_INPUT_LIMIT))
    return command, actual


def holding_rhs(state: np.ndarray, target: float, *, compensated: bool) -> np.ndarray:
    theta, omega = state
    _, input_torque = hold_command(float(theta), float(omega), target, compensated=compensated)
    return np.array([omega, -G * np.sin(theta) - DAMPING * omega + input_torque])


def simulate_holding(*, compensated: bool) -> dict[str, np.ndarray]:
    time = np.linspace(0.0, 8.0, 1601)
    step_size = float(time[1] - time[0])
    states = np.zeros((time.size, 2))
    commands = np.zeros(time.size)
    inputs = np.zeros(time.size)

    for index in range(time.size):
        commands[index], inputs[index] = hold_command(
            states[index, 0],
            states[index, 1],
            HOLD_TARGET,
            compensated=compensated,
        )
        if index + 1 < time.size:
            states[index + 1] = rk4_step(
                states[index],
                step_size,
                lambda state: holding_rhs(state, HOLD_TARGET, compensated=compensated),
            )

    return {"time": time, "states": states, "commands": commands, "inputs": inputs}


def solve_naive_steady_angle(target: float) -> float:
    lower = 0.0
    upper = max(target, 1.0e-6)
    for _ in range(70):
        middle = 0.5 * (lower + upper)
        value = G * np.sin(middle) - HOLD_KP * (target - middle)
        if value >= 0.0:
            upper = middle
        else:
            lower = middle
    return 0.5 * (lower + upper)


def add_panel_label(axis: plt.Axes, label: str, *, y: float = 0.98) -> None:
    axis.text(
        0.02,
        y,
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


def draw_state_plane(axis: plt.Axes, small: dict[str, np.ndarray], large: dict[str, np.ndarray]) -> None:
    add_panel_label(axis, "A  局所領域と飽和境界")
    axis.add_patch(
        Rectangle(
            (-LOCAL_VALID_PHI, -LOCAL_VALID_OMEGA),
            2.0 * LOCAL_VALID_PHI,
            2.0 * LOCAL_VALID_OMEGA,
            facecolor="#eef4fb",
            edgecolor=BLUE,
            linewidth=1.4,
            alpha=0.95,
            label="局所有効領域",
        )
    )

    phi_line = np.linspace(-1.55, 1.55, 300)
    for sign, label in ((1.0, r"$|u_c|=u_{\max}$"), (-1.0, None)):
        omega_line = (sign * LOCAL_INPUT_LIMIT - LOCAL_K1 * phi_line) / LOCAL_K2
        axis.plot(phi_line, omega_line, color=RED, linestyle="--", linewidth=1.3, label=label)

    small_states = small["states"]
    large_states = large["states"]
    axis.plot(small_states[:, 0], small_states[:, 1], color=GREEN, linewidth=2.0, label=r"小さい初期偏差")
    axis.plot(large_states[:, 0], large_states[:, 1], color=ORANGE, linewidth=2.0, label=r"大きい初期偏差")
    axis.scatter([small_states[0, 0], large_states[0, 0]], [small_states[0, 1], large_states[0, 1]], color=[GREEN, ORANGE], s=28, zorder=5)
    axis.scatter([0.0], [0.0], color="#222222", marker="*", s=70, zorder=6, label="倒立平衡点")
    axis.set_xlim(-1.50, 1.50)
    axis.set_ylim(-4.2, 4.2)
    axis.set_xlabel(r"偏差角 $\phi=\theta-\pi$ [rad]")
    axis.set_ylabel(r"角速度 $\omega$ [rad/s]")
    axis.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.75)
    axis.legend(loc="lower left", fontsize=8.0, frameon=True)


def draw_local_time_trace(axis: plt.Axes, small: dict[str, np.ndarray], large: dict[str, np.ndarray]) -> None:
    add_panel_label(axis, "B  同じ局所ゲインの時系列")
    time = small["time"]
    small_states = small["states"]
    large_states = large["states"]
    small_inputs = small["inputs"]
    large_inputs = large["inputs"]
    axis.plot(time, small_states[:, 0], color=GREEN, linewidth=2.0, label=r"$\phi(t)$ 小")
    axis.plot(time, large_states[:, 0], color=ORANGE, linewidth=2.0, label=r"$\phi(t)$ 大")
    axis.axhspan(-LOCAL_VALID_PHI, LOCAL_VALID_PHI, color="#eef4fb", alpha=0.9, zorder=0)
    axis.axhline(0.0, color="#444444", linewidth=0.8)
    axis.set_xlim(0.0, 5.0)
    axis.set_ylim(-1.55, 1.55)
    axis.set_xlabel(r"時間 $t$ [s]")
    axis.set_ylabel(r"偏差角 $\phi$ [rad]")
    axis.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.75)

    right_axis = axis.twinx()
    right_axis.plot(
        time,
        small_inputs / LOCAL_INPUT_LIMIT,
        color=GREEN,
        linestyle=":",
        linewidth=1.6,
        label=r"$u/u_{\max}$ 小",
    )
    right_axis.plot(
        time,
        large_inputs / LOCAL_INPUT_LIMIT,
        color=ORANGE,
        linestyle=":",
        linewidth=1.6,
        label=r"$u/u_{\max}$ 大",
    )
    right_axis.axhline(1.0, color=RED, linestyle="--", linewidth=1.0)
    right_axis.axhline(-1.0, color=RED, linestyle="--", linewidth=1.0)
    right_axis.set_ylim(-1.18, 1.18)
    right_axis.set_ylabel(r"入力比 $u/u_{\max}$ [-]")

    handles, labels = axis.get_legend_handles_labels()
    right_handles, right_labels = right_axis.get_legend_handles_labels()
    axis.legend(handles + right_handles, labels + right_labels, loc="lower right", fontsize=7.7, frameon=True)


def draw_holding_response(axis: plt.Axes, naive: dict[str, np.ndarray], compensated: dict[str, np.ndarray]) -> None:
    add_panel_label(axis, "C  保持角ステップ")
    time = naive["time"]
    naive_theta = naive["states"][:, 0]
    compensated_theta = compensated["states"][:, 0]
    naive_error = HOLD_TARGET - naive_theta[-1]
    compensated_error = HOLD_TARGET - compensated_theta[-1]

    axis.plot(time, naive_theta, color=ORANGE, linewidth=2.0, label="重力バイアスなし")
    axis.plot(time, compensated_theta, color=BLUE, linewidth=2.0, label="重力補償あり")
    axis.axhline(HOLD_TARGET, color="#222222", linewidth=1.2, linestyle="--", label=rf"目標 ${HOLD_TARGET:.2f}$ rad")
    axis.set_xlim(0.0, 8.0)
    axis.set_ylim(-0.08, 1.03)
    axis.set_xlabel(r"時間 $t$ [s]")
    axis.set_ylabel(r"角度 $\theta$ [rad]")
    axis.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.75)
    axis.legend(loc="upper right", fontsize=7.8, frameon=True)
    axis.text(
        0.04,
        0.10,
        rf"定常偏差: なし {naive_error:.2f} rad / 補償 {abs(compensated_error):.2f} rad",
        transform=axis.transAxes,
        ha="left",
        va="center",
        fontsize=8.2,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.94, "pad": 2.0},
    )


def draw_target_sweep(axis: plt.Axes) -> None:
    add_panel_label(axis, "D  目標角と入力余裕")
    targets = np.linspace(0.0, 1.30, 180)
    naive_angles = np.array([solve_naive_steady_angle(target) for target in targets])
    naive_errors = targets - naive_angles
    gravity_torque = G * np.sin(targets)
    margin = HOLD_INPUT_LIMIT - np.abs(gravity_torque)

    axis.plot(targets, naive_errors, color=ORANGE, linewidth=2.0, label="バイアスなし定常偏差")
    axis.axhline(0.0, color="#333333", linewidth=0.8)
    axis.set_xlim(0.0, 1.30)
    axis.set_ylim(-0.02, 0.78)
    axis.set_xlabel(r"目標角 $\theta_d$ [rad]")
    axis.set_ylabel(r"定常偏差 $\theta_d-\theta_\infty$ [rad]")
    axis.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.75)

    right_axis = axis.twinx()
    right_axis.plot(targets, margin, color=PURPLE, linewidth=2.0, label="補償後の入力余裕")
    right_axis.axhline(0.0, color=RED, linestyle="--", linewidth=1.1)
    right_axis.set_ylim(-2.2, 8.5)
    right_axis.set_ylabel(r"余裕 $u_{\max}-g\sin\theta_d$ [N m]")

    limit_angle = float(np.arcsin(HOLD_INPUT_LIMIT / G))
    axis.axvline(limit_angle, color=RED, linestyle="--", linewidth=1.1)
    axis.text(
        limit_angle + 0.02,
        0.69,
        "余裕0",
        color=RED,
        fontsize=8.0,
        ha="left",
        va="center",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 0.4},
    )

    handles, labels = axis.get_legend_handles_labels()
    right_handles, right_labels = right_axis.get_legend_handles_labels()
    axis.legend(handles + right_handles, labels + right_labels, loc="lower left", fontsize=8.0, frameon=True)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "ja" / "figures" / "flow6_nonlinear_control_visual_verification.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 9,
            "font.family": ["Noto Sans CJK JP", "Noto Sans", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "figure.dpi": 180,
            "savefig.dpi": 180,
            "axes.unicode_minus": False,
        }
    )

    small = simulate_upright((0.22, 0.0))
    large = simulate_upright((1.20, 0.0))
    naive = simulate_holding(compensated=False)
    compensated = simulate_holding(compensated=True)

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.0), constrained_layout=False)
    draw_state_plane(axes[0, 0], small, large)
    draw_local_time_trace(axes[0, 1], small, large)
    draw_holding_response(axes[1, 0], naive, compensated)
    draw_target_sweep(axes[1, 1])

    finalize_figure(fig, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
