#!/usr/bin/env python3
"""Generate an NMPC mesh, receding-horizon, and solver-trace visual."""

from __future__ import annotations

from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("numpy is required to generate the W07 NMPC figure") from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("matplotlib is required to generate the W07 NMPC figure") from exc

from figure_style import finalize_figure


HORIZON_STEPS = 8
SAMPLE_TIME = 0.18
YAW_RATE_LIMIT = 1.0
OBSTACLE_CENTER = np.asarray([0.60, 0.50])
OBSTACLE_RADIUS = 0.12
SAFETY_RADIUS = 0.17


def unicycle_step(state: np.ndarray, control: np.ndarray, dt: float = SAMPLE_TIME) -> np.ndarray:
    """Integrate one unicycle interval with deterministic small substeps."""

    next_state = state.astype(float).copy()
    substeps = 12
    sub_dt = dt / substeps
    speed, yaw_rate = control
    for _ in range(substeps):
        next_state[0] += sub_dt * speed * np.cos(next_state[2])
        next_state[1] += sub_dt * speed * np.sin(next_state[2])
        next_state[2] += sub_dt * yaw_rate
    return next_state


def integrate_prediction(controls: np.ndarray) -> np.ndarray:
    """Return the accepted nonlinear prediction nodes x_{i|k}."""

    state = np.asarray([0.0, 0.0, 0.08])
    states = [state.copy()]
    for control in controls:
        state = unicycle_step(state, control)
        states.append(state.copy())
    return np.asarray(states)


def warm_start_nodes(accepted_states: np.ndarray) -> np.ndarray:
    """Build a shifted warm start with visible multiple-shooting defects."""

    warm = accepted_states.copy()
    node_index = np.arange(len(warm))
    warm[:, 0] += 0.028 * np.sin(0.9 * node_index)
    warm[:, 1] += 0.035 * np.cos(0.7 * node_index + 0.4)
    warm[:, 2] += 0.035 * np.sin(0.6 * node_index + 0.2)
    warm[0] = accepted_states[0]
    return warm


def interval_curves(states: np.ndarray, controls: np.ndarray) -> list[np.ndarray]:
    """Sample continuous interval curves from node states and controls."""

    curves: list[np.ndarray] = []
    for state, control in zip(states[:-1], controls):
        samples = [state.copy()]
        running = state.copy()
        for _ in range(18):
            running = unicycle_step(running, control, SAMPLE_TIME / 18.0)
            samples.append(running.copy())
        curves.append(np.asarray(samples))
    return curves


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "ja" / "figures" / "w07_nmpc_mesh_solver_horizon.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 8.8,
            "axes.titlesize": 9.0,
            "legend.fontsize": 7.3,
            "font.family": ["Noto Sans CJK JP", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
            "figure.dpi": 180,
            "savefig.dpi": 180,
        }
    )

    controls = np.asarray(
        [
            [0.95, 1.00],
            [0.94, 0.92],
            [0.91, 0.78],
            [0.86, 0.54],
            [0.80, 0.28],
            [0.76, 0.06],
            [0.74, -0.08],
            [0.74, -0.16],
        ]
    )
    accepted_states = integrate_prediction(controls)
    warm_states = warm_start_nodes(accepted_states)
    warm_endpoints = np.asarray([unicycle_step(warm_states[i], controls[i]) for i in range(HORIZON_STEPS)])
    accepted_curves = interval_curves(accepted_states, controls)
    warm_curves = interval_curves(warm_states, controls)

    node_time = SAMPLE_TIME * np.arange(HORIZON_STEPS + 1)
    input_time = SAMPLE_TIME * np.arange(HORIZON_STEPS)
    safety_margin = np.linalg.norm(accepted_states[:, :2] - OBSTACLE_CENTER, axis=1) - SAFETY_RADIUS
    min_margin_index = int(np.argmin(safety_margin))

    fig = plt.figure(figsize=(9.0, 4.5), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.55, 1.0),
        height_ratios=(1.0, 1.0),
        left=0.055,
        right=0.94,
        bottom=0.12,
        top=0.955,
        wspace=0.24,
        hspace=0.34,
    )
    mesh_ax = fig.add_subplot(grid[:, 0])
    trace_ax = fig.add_subplot(grid[0, 1])
    solver_ax = fig.add_subplot(grid[1, 1])

    mesh_ax.add_patch(
        Circle(OBSTACLE_CENTER, SAFETY_RADIUS, color="#f5c7c7", alpha=0.42, label="安全半径")
    )
    mesh_ax.add_patch(
        Circle(OBSTACLE_CENTER, OBSTACLE_RADIUS, color="#9b2226", alpha=0.18, label="障害物")
    )
    for curve in warm_curves:
        mesh_ax.plot(curve[:, 0], curve[:, 1], color="#9a9a9a", linewidth=1.0, alpha=0.65)
    for curve in accepted_curves:
        mesh_ax.plot(curve[:, 0], curve[:, 1], color="#1f77b4", linewidth=1.8)

    mesh_ax.plot(
        warm_states[:, 0],
        warm_states[:, 1],
        marker="o",
        linestyle="--",
        color="#777777",
        markersize=4.0,
        linewidth=1.0,
        label=r"シフト $\bar{x}_{i|k}$",
    )
    mesh_ax.plot(
        accepted_states[:, 0],
        accepted_states[:, 1],
        marker="o",
        color="#1f77b4",
        markersize=4.8,
        linewidth=1.4,
        label=r"更新 $x^+_{i|k}$",
    )
    mesh_ax.plot(
        warm_endpoints[:, 0],
        warm_endpoints[:, 1],
        linestyle="none",
        marker="x",
        color="#d95f02",
        markersize=4.5,
        label=r"$\Phi_i(\bar{x}_i,\bar{u}_i)$",
    )
    mesh_ax.text(
        0.38,
        0.62,
        "安全半径",
        color="#9b2226",
        fontsize=8.0,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.0},
    )
    mesh_ax.text(
        OBSTACLE_CENTER[0],
        OBSTACLE_CENTER[1],
        "障害物",
        color="#7f1d1d",
        fontsize=8.0,
        ha="center",
        va="center",
    )
    mesh_ax.text(
        0.76,
        0.44,
        r"シフト $\bar{x}_{i|k}$",
        color="#5f5f5f",
        fontsize=8.0,
        rotation=31,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.8},
    )
    mesh_ax.text(
        0.30,
        0.11,
        r"更新 $x^+_{i|k}$",
        color="#1f77b4",
        fontsize=8.0,
        rotation=24,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.8},
    )
    for i in range(HORIZON_STEPS):
        if i in (1, 3, 5):
            mesh_ax.annotate(
                "",
                xy=warm_states[i + 1, :2],
                xytext=warm_endpoints[i, :2],
                arrowprops={"arrowstyle": "->", "color": "#d95f02", "linewidth": 1.0},
            )
        if i in (2, 4, 6):
            mesh_ax.annotate(
                "",
                xy=accepted_states[i, :2],
                xytext=warm_states[i, :2],
                arrowprops={"arrowstyle": "->", "color": "#2ca02c", "linewidth": 1.0},
            )

    mesh_ax.annotate(
        r"$d_i=x_{i+1}-\Phi_i$",
        xy=warm_states[4, :2],
        xytext=(0.20, 0.57),
        arrowprops={"arrowstyle": "->", "color": "#d95f02", "linewidth": 1.0},
        color="#d95f02",
        fontsize=8.0,
    )
    mesh_ax.annotate(
        r"$\Delta z$",
        xy=accepted_states[5, :2],
        xytext=(0.81, 0.28),
        arrowprops={"arrowstyle": "->", "color": "#2ca02c", "linewidth": 1.0},
        color="#2ca02c",
        fontsize=8.0,
    )
    mesh_ax.text(accepted_states[0, 0] - 0.03, accepted_states[0, 1] - 0.045, r"$x_{0|k}$", fontsize=8.0)
    mesh_ax.text(accepted_states[-1, 0] + 0.02, accepted_states[-1, 1] + 0.02, r"$x_{N|k}$", fontsize=8.0)
    mesh_ax.annotate(
        r"$u_{0|k}$",
        xy=accepted_states[1, :2],
        xytext=(0.05, 0.20),
        arrowprops={"arrowstyle": "->", "color": "#1f77b4", "linewidth": 1.0},
        color="#1f77b4",
        fontsize=8.0,
    )
    mesh_ax.set_aspect("equal", adjustable="box")
    mesh_ax.set_xlabel(r"位置 $X$ [m]")
    mesh_ax.set_ylabel(r"位置 $Y$ [m]")
    mesh_ax.set_xlim(-0.04, 1.07)
    mesh_ax.set_ylim(-0.055, 0.665)
    mesh_ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)

    trace_ax.plot(
        node_time,
        safety_margin,
        color="#1f77b4",
        marker="o",
        markersize=4.0,
        linewidth=1.8,
    )
    trace_ax.axhline(0.0, color="#9b2226", linestyle=":", linewidth=1.1)
    trace_ax.fill_between(node_time, 0.0, np.minimum(safety_margin, 0.0), color="#f5c7c7", alpha=0.55)
    trace_ax.annotate(
        rf"$m_{{{min_margin_index}}}={safety_margin[min_margin_index]:.3f}$ m",
        xy=(node_time[min_margin_index], safety_margin[min_margin_index]),
        xytext=(0.49, 0.14),
        arrowprops={"arrowstyle": "->", "color": "#1f77b4", "linewidth": 1.0},
        color="#1f77b4",
        fontsize=8.0,
    )
    trace_ax.set_xlabel("予測時刻 [s]")
    trace_ax.set_ylabel(r"安全余裕 $m_i$ [m]")
    trace_ax.set_xlim(0.0, node_time[-1])
    trace_ax.set_ylim(-0.03, 0.64)
    trace_ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)
    trace_ax.text(
        0.03,
        0.91,
        r"安全余裕 $m_i$",
        transform=trace_ax.transAxes,
        color="#1f77b4",
        fontsize=8.0,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.0},
    )
    trace_ax.text(
        0.04,
        0.05,
        "ハード制約境界",
        transform=trace_ax.transAxes,
        color="#9b2226",
        fontsize=8.0,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.0},
    )

    input_ax = trace_ax.twinx()
    input_ax.step(
        np.r_[input_time, node_time[-1]],
        np.r_[controls[:, 1], controls[-1, 1]],
        where="post",
        color="#d62728",
        linewidth=1.7,
    )
    input_ax.axhline(YAW_RATE_LIMIT, color="#d62728", linestyle="--", linewidth=0.9, alpha=0.65)
    input_ax.axhline(-YAW_RATE_LIMIT, color="#d62728", linestyle="--", linewidth=0.9, alpha=0.65)
    input_ax.plot(
        input_time[0],
        controls[0, 1],
        marker="s",
        color="#d62728",
        markersize=5.2,
    )
    input_ax.annotate(
        r"$u_{0|k}: \omega=\omega_{\max}$",
        xy=(input_time[0], controls[0, 1]),
        xytext=(0.17, 0.57),
        arrowprops={"arrowstyle": "->", "color": "#d62728", "linewidth": 1.0},
        color="#d62728",
        fontsize=8.0,
    )
    input_ax.text(
        0.97,
        0.82,
        r"入力 $\omega_i$",
        transform=input_ax.transAxes,
        color="#d62728",
        fontsize=8.0,
        ha="right",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.0},
    )
    input_ax.set_ylabel(r"ヨーレート $\omega_i$ [rad/s]")
    input_ax.set_ylim(-1.25, 1.22)

    elapsed_ms = np.asarray([0.0, 2.2, 3.4, 4.8])
    kkt_residual = np.asarray([0.82, 0.11, 0.018, 0.006])
    hard_violation = np.asarray([0.075, 0.012, 0.0015, 0.0005])
    defect_residual = np.asarray([0.160, 0.009, 0.0030, 0.0010])
    deadline_ms = 3.0
    hard_tol = 0.020
    defect_tol = 0.015

    solver_ax.semilogy(
        elapsed_ms,
        kkt_residual,
        marker="o",
        color="#6a4c93",
        linewidth=1.8,
    )
    solver_ax.semilogy(
        elapsed_ms,
        hard_violation,
        marker="s",
        color="#d62728",
        linewidth=1.8,
    )
    solver_ax.semilogy(
        elapsed_ms,
        defect_residual,
        marker="^",
        color="#2ca02c",
        linewidth=1.8,
    )
    solver_ax.axhline(hard_tol, color="#d62728", linestyle=":", linewidth=1.0, alpha=0.78)
    solver_ax.axhline(defect_tol, color="#2ca02c", linestyle=":", linewidth=1.0, alpha=0.78)
    solver_ax.axvline(deadline_ms, color="#555555", linestyle="--", linewidth=1.0)
    solver_ax.axvline(elapsed_ms[1], color="#1f77b4", linestyle="-.", linewidth=1.0)
    solver_ax.annotate(
        r"$j=1$: 制約内で $u_{0|k}$ 採用",
        xy=(elapsed_ms[1], hard_violation[1]),
        xytext=(0.28, 0.040),
        arrowprops={"arrowstyle": "->", "color": "#1f77b4", "linewidth": 1.0},
        color="#1f77b4",
        fontsize=8.0,
    )
    solver_ax.text(
        0.05,
        0.06,
        rf"許容: $v_{{hard}}\leq{hard_tol:.2f}$ m, $r_{{def}}\leq{defect_tol:.3f}$ m",
        transform=solver_ax.transAxes,
        color="#555555",
        fontsize=8.0,
    )
    solver_ax.text(4.85, kkt_residual[-1] * 1.12, "KKT", color="#6a4c93", fontsize=8.0, ha="right")
    solver_ax.text(
        4.85,
        hard_violation[-1] * 1.85,
        r"$v_{\mathrm{hard}}$",
        color="#d62728",
        fontsize=8.0,
        ha="right",
    )
    solver_ax.text(
        4.85,
        defect_residual[-1] * 1.45,
        r"$r_{\mathrm{def}}$",
        color="#2ca02c",
        fontsize=8.0,
        ha="right",
    )
    solver_ax.text(
        deadline_ms + 0.05,
        0.42,
        r"$T_{\mathrm{dead}}=3.0$ ms",
        color="#555555",
        fontsize=8.0,
        rotation=90,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.8},
    )
    solver_ax.set_xlabel("経過時間 [ms]")
    solver_ax.set_ylabel("残差・違反量 [- または m]")
    solver_ax.set_xlim(-0.1, 5.0)
    solver_ax.set_ylim(5.0e-4, 1.1)
    solver_ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7, which="both")

    finalize_figure(fig, output_path, layout="none")
    print(output_path)


if __name__ == "__main__":
    main()
