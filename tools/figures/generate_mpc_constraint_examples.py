#!/usr/bin/env python3
"""Generate MPC constraint and horizon-sweep visual examples."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("numpy is required to generate the MPC constraint figure") from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("matplotlib is required to generate the MPC constraint figure") from exc

from figure_style import finalize_figure


A_PLANT = 0.8
B_PLANT = 0.5
STATE_WEIGHT = 1.0
INPUT_WEIGHT = 0.12
TERMINAL_WEIGHT = 1.4
PREDICTION_HORIZON = 8
SIMULATION_STEPS = 12
INITIAL_STATE = 1.8
INPUT_LIMITS = (0.25, 0.50, 1.00)


def feasible_vertices(a_matrix: np.ndarray, b_vector: np.ndarray) -> np.ndarray:
    """Return sorted vertices of a two-dimensional polytope A z <= b."""

    points = []
    for first, second in combinations(range(len(b_vector)), 2):
        active = a_matrix[[first, second], :]
        rhs = b_vector[[first, second]]
        if abs(np.linalg.det(active)) < 1.0e-10:
            continue
        point = np.linalg.solve(active, rhs)
        if np.all(a_matrix @ point <= b_vector + 1.0e-9):
            points.append(point)

    vertices = np.unique(np.round(np.asarray(points), 12), axis=0)
    center = vertices.mean(axis=0)
    order = np.argsort(np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0]))
    return vertices[order]


def solve_constrained_quadratic(
    hessian: np.ndarray,
    free_point: np.ndarray,
    a_matrix: np.ndarray,
    b_vector: np.ndarray,
) -> np.ndarray:
    """Solve min 0.5 (z-z0)' H (z-z0) subject to A z <= b in 2D."""

    best_point = None
    best_value = float("inf")
    constraint_count = len(b_vector)

    for active_size in range(3):
        for active_set in combinations(range(constraint_count), active_size):
            if active_set:
                active = a_matrix[list(active_set), :]
                rhs = b_vector[list(active_set)]
                kkt = np.block(
                    [
                        [hessian, active.T],
                        [active, np.zeros((active_size, active_size))],
                    ]
                )
                right = np.concatenate([hessian @ free_point, rhs])
                try:
                    solution = np.linalg.solve(kkt, right)[:2]
                except np.linalg.LinAlgError:
                    continue
            else:
                solution = free_point

            if not np.all(a_matrix @ solution <= b_vector + 1.0e-9):
                continue

            error = solution - free_point
            value = 0.5 * float(error.T @ hessian @ error)
            if value < best_value:
                best_point = solution
                best_value = value

    if best_point is None:  # pragma: no cover - guarded by fixed feasible data.
        raise RuntimeError("no feasible point found for the example QP")
    return best_point


def condensed_qp_matrices(
    state: float,
    horizon: int,
    a_plant: float = A_PLANT,
    b_plant: float = B_PLANT,
) -> tuple[np.ndarray, np.ndarray]:
    """Build 0.5 U' H U + f' U for a scalar finite-horizon MPC problem."""

    state_map = np.zeros((horizon, horizon))
    free_response = np.zeros(horizon)
    for prediction_step in range(1, horizon + 1):
        free_response[prediction_step - 1] = (a_plant**prediction_step) * state
        for control_step in range(prediction_step):
            state_map[prediction_step - 1, control_step] = (
                (a_plant ** (prediction_step - 1 - control_step)) * b_plant
            )

    state_weights = np.full(horizon, STATE_WEIGHT)
    state_weights[-1] = TERMINAL_WEIGHT
    weighted_map = state_weights[:, None] * state_map

    hessian = 2.0 * (state_map.T @ weighted_map + INPUT_WEIGHT * np.eye(horizon))
    gradient = 2.0 * (state_map.T @ (state_weights * free_response))
    return hessian, gradient


def solve_box_qp(
    hessian: np.ndarray,
    gradient: np.ndarray,
    limit: float,
    max_iterations: int = 2000,
    tolerance: float = 1.0e-10,
) -> np.ndarray:
    """Projected-gradient solve for a small positive-definite box QP."""

    unconstrained = -np.linalg.solve(hessian, gradient)
    control = np.clip(unconstrained, -limit, limit)
    lipschitz = float(np.linalg.eigvalsh(hessian).max())
    step = 1.0 / lipschitz

    for _ in range(max_iterations):
        next_control = np.clip(control - step * (hessian @ control + gradient), -limit, limit)
        if np.linalg.norm(next_control - control, ord=np.inf) < tolerance:
            return next_control
        control = next_control

    return control


def simulate_mpc(limit: float) -> tuple[np.ndarray, np.ndarray]:
    """Simulate receding-horizon MPC for a scalar system under an input limit."""

    states = [INITIAL_STATE]
    inputs = []
    state = INITIAL_STATE

    for _ in range(SIMULATION_STEPS):
        hessian, gradient = condensed_qp_matrices(state, PREDICTION_HORIZON)
        planned_inputs = solve_box_qp(hessian, gradient, limit)
        applied_input = float(planned_inputs[0])
        inputs.append(applied_input)
        state = A_PLANT * state + B_PLANT * applied_input
        states.append(state)

    return np.asarray(states), np.asarray(inputs)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "ja" / "figures" / "mpc_constraint_examples.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "font.family": ["Noto Sans CJK JP", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
            "figure.dpi": 180,
            "savefig.dpi": 180,
        }
    )

    fig = plt.figure(figsize=(7.2, 5.2), constrained_layout=False)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.08, 1.0), height_ratios=(1.0, 1.0))
    feasible_ax = fig.add_subplot(grid[:, 0])
    state_ax = fig.add_subplot(grid[0, 1])
    input_ax = fig.add_subplot(grid[1, 1], sharex=state_ax)

    a_matrix = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
            [1.0, 0.7],
        ]
    )
    b_vector = np.asarray([1.0, 1.1, 0.4, 0.2, 1.25])
    hessian = np.asarray([[1.0, 0.2], [0.2, 1.4]])
    free_point = np.asarray([2.4, 0.5])
    optimum = solve_constrained_quadratic(hessian, free_point, a_matrix, b_vector)
    active = np.isclose(a_matrix @ optimum, b_vector, atol=1.0e-7)
    vertices = feasible_vertices(a_matrix, b_vector)

    mesh_u0 = np.linspace(-0.55, 2.55, 260)
    mesh_u1 = np.linspace(-0.35, 1.35, 220)
    grid_u0, grid_u1 = np.meshgrid(mesh_u0, mesh_u1)
    delta_u0 = grid_u0 - free_point[0]
    delta_u1 = grid_u1 - free_point[1]
    objective = 0.5 * (
        hessian[0, 0] * delta_u0**2
        + 2.0 * hessian[0, 1] * delta_u0 * delta_u1
        + hessian[1, 1] * delta_u1**2
    )

    feasible_ax.contour(grid_u0, grid_u1, objective, levels=12, colors="#9a9a9a", linewidths=0.8)
    feasible_ax.fill(vertices[:, 0], vertices[:, 1], color="#d9ecff", alpha=0.85, label="可行集合")

    x_line = np.linspace(-0.6, 2.6, 320)
    constraint_labels = (
        r"$u_0 \leq 1$",
        r"$u_1 \leq 1.1$",
        r"$u_0 \geq -0.4$",
        r"$u_1 \geq -0.2$",
        r"$u_0+0.7u_1 \leq 1.25$",
    )
    for row, rhs, is_active, label in zip(a_matrix, b_vector, active, constraint_labels):
        color = "#d62728" if is_active else "#555555"
        linewidth = 2.0 if is_active else 1.0
        if abs(row[1]) > 1.0e-12:
            y_line = (rhs - row[0] * x_line) / row[1]
            feasible_ax.plot(x_line, y_line, color=color, linewidth=linewidth)
        else:
            feasible_ax.axvline(rhs / row[0], color=color, linewidth=linewidth)
        if is_active:
            feasible_ax.plot([], [], color=color, linewidth=linewidth, label=f"有効: {label}")

    feasible_ax.plot(
        free_point[0],
        free_point[1],
        marker="x",
        markersize=8,
        color="#111111",
        label="制約なし最小点",
    )
    feasible_ax.plot(optimum[0], optimum[1], marker="o", markersize=6, color="#d62728", label="QP 最適点")
    feasible_ax.annotate(
        r"$z^\star$",
        xy=optimum,
        xytext=(optimum[0] + 0.10, optimum[1] + 0.12),
        arrowprops={"arrowstyle": "->", "color": "#d62728", "linewidth": 1.0},
        color="#d62728",
    )
    feasible_ax.set_title("QP 可行集合と有効制約")
    feasible_ax.set_xlabel(r"決定変数 $u_0$ [-]")
    feasible_ax.set_ylabel(r"決定変数 $u_1$ [-]")
    feasible_ax.set_xlim(-0.55, 2.55)
    feasible_ax.set_ylim(-0.35, 1.35)
    feasible_ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)
    feasible_ax.legend(loc="upper right", framealpha=0.92)

    colors = ("#1f77b4", "#ff7f0e", "#2ca02c")
    markers = ("o", "s", "^")
    time = np.arange(SIMULATION_STEPS + 1)
    input_time = np.arange(SIMULATION_STEPS)

    for limit, color, marker in zip(INPUT_LIMITS, colors, markers):
        states, inputs = simulate_mpc(limit)
        label = rf"入力幅 $u_{{\max}}={limit:.2f}$"
        state_ax.plot(time, states, color=color, marker=marker, markersize=3.3, linewidth=1.8, label=label)
        input_ax.step(input_time, inputs, where="post", color=color, marker=marker, markersize=3.3, linewidth=1.8, label=label)
        input_ax.axhline(-limit, color=color, linestyle="--", linewidth=0.9, alpha=0.45)

    state_ax.axhline(2.0, color="#777777", linestyle=":", linewidth=1.0)
    state_ax.axhline(-2.0, color="#777777", linestyle=":", linewidth=1.0)
    state_ax.text(0.02, 0.91, r"$|x|\leq 2$", transform=state_ax.transAxes, color="#555555", fontsize=8)
    state_ax.set_title(r"閉ループ状態 $N=8$")
    state_ax.set_ylabel(r"状態 $x_k$ [-]")
    state_ax.set_xlim(0, SIMULATION_STEPS)
    state_ax.set_ylim(-0.1, 2.1)
    state_ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)
    state_ax.legend(loc="upper right")

    input_ax.axhline(0.0, color="#555555", linestyle=":", linewidth=0.8)
    input_ax.annotate(
        "負側下限に飽和",
        xy=(0.1, -INPUT_LIMITS[0]),
        xytext=(2.3, -0.18),
        arrowprops={"arrowstyle": "->", "color": "#d62728", "linewidth": 1.0},
        color="#d62728",
        fontsize=8,
    )
    input_ax.text(
        0.98,
        0.05,
        r"破線: 下限 $-u_{\max}$",
        transform=input_ax.transAxes,
        ha="right",
        va="bottom",
        color="#555555",
        fontsize=8,
    )
    input_ax.set_title("入力幅による適用入力")
    input_ax.set_xlabel(r"サンプル $k$ [-]")
    input_ax.set_ylabel(r"入力 $u_k$ [-]")
    input_ax.set_ylim(-1.08, 0.10)
    input_ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)

    finalize_figure(fig, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
