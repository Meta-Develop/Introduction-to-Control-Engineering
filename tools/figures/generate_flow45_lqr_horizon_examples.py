#!/usr/bin/env python3
"""Generate finite-horizon LQR horizon and Riccati examples."""

from __future__ import annotations

from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("numpy is required to generate the finite-horizon LQR figure") from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("matplotlib is required to generate the finite-horizon LQR figure") from exc

from figure_style import finalize_figure


SAMPLE_TIME = 0.1
SIMULATION_STEPS = 80
RECEDING_HORIZON = 28
PREDICTION_HORIZONS = (8, 16, 32)
CONVERGENCE_HORIZONS = np.arange(1, 81)
INPUT_LIMIT = 2.0

A_DISCRETE = np.array([[1.0, SAMPLE_TIME], [0.0, 1.0]])
B_DISCRETE = np.array([[0.5 * SAMPLE_TIME**2], [SAMPLE_TIME]])
INITIAL_STATE = np.array([1.0, 0.0])
BASE_STATE_WEIGHT = np.diag([1.0, 0.08])
BASE_INPUT_WEIGHT = 0.20
ZERO_TERMINAL_WEIGHT = np.zeros((2, 2))


def riccati_step(cost_to_go: np.ndarray, state_weight: np.ndarray, input_weight: float) -> np.ndarray:
    gain = gain_from_cost_to_go(cost_to_go, input_weight)
    return (
        state_weight
        + A_DISCRETE.T @ cost_to_go @ A_DISCRETE
        - A_DISCRETE.T @ cost_to_go @ B_DISCRETE @ gain
    )


def gain_from_cost_to_go(cost_to_go: np.ndarray, input_weight: float) -> np.ndarray:
    hessian = np.array([[input_weight]]) + B_DISCRETE.T @ cost_to_go @ B_DISCRETE
    return np.linalg.solve(hessian, B_DISCRETE.T @ cost_to_go @ A_DISCRETE)


def solve_dare(
    state_weight: np.ndarray,
    input_weight: float,
    tolerance: float = 1.0e-12,
    max_iterations: int = 10000,
) -> np.ndarray:
    cost_to_go = state_weight.copy()
    for _ in range(max_iterations):
        next_cost = riccati_step(cost_to_go, state_weight, input_weight)
        if np.linalg.norm(next_cost - cost_to_go, ord="fro") < tolerance:
            return next_cost
        cost_to_go = next_cost
    return cost_to_go


def backward_riccati(
    horizon: int,
    state_weight: np.ndarray,
    input_weight: float,
    terminal_weight: np.ndarray = ZERO_TERMINAL_WEIGHT,
) -> tuple[np.ndarray, np.ndarray]:
    costs = np.zeros((horizon + 1, 2, 2))
    gains = np.zeros((horizon, 1, 2))
    costs[horizon] = terminal_weight

    for step in range(horizon - 1, -1, -1):
        next_cost = costs[step + 1]
        gain = gain_from_cost_to_go(next_cost, input_weight)
        gains[step] = gain
        costs[step] = (
            state_weight
            + A_DISCRETE.T @ next_cost @ A_DISCRETE
            - A_DISCRETE.T @ next_cost @ B_DISCRETE @ gain
        )

    return costs, gains


def simulate_time_varying_plan(
    horizon: int,
    state_weight: np.ndarray,
    input_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    _, gains = backward_riccati(horizon, state_weight, input_weight)
    state = INITIAL_STATE.copy()
    states = [state.copy()]
    inputs = []

    for step in range(horizon):
        control_input = -float((gains[step] @ state).item())
        inputs.append(control_input)
        state = A_DISCRETE @ state + B_DISCRETE[:, 0] * control_input
        states.append(state.copy())

    return np.asarray(states), np.asarray(inputs)


def simulate_receding_horizon(
    steps: int,
    horizon: int,
    state_weight: np.ndarray,
    input_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    _, gains = backward_riccati(horizon, state_weight, input_weight)
    first_gain = gains[0]
    state = INITIAL_STATE.copy()
    states = [state.copy()]
    inputs = []

    for _ in range(steps):
        control_input = -float((first_gain @ state).item())
        inputs.append(control_input)
        state = A_DISCRETE @ state + B_DISCRETE[:, 0] * control_input
        states.append(state.copy())

    return np.asarray(states), np.asarray(inputs)


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        0.02,
        0.96,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        fontweight="bold",
        color="#222222",
        bbox={
            "boxstyle": "round,pad=0.20",
            "facecolor": "white",
            "edgecolor": "#bbbbbb",
            "alpha": 0.94,
        },
    )


def configure_axis(axis: plt.Axes) -> None:
    axis.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.72)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "ja" / "figures" / "flow45_lqr_horizon_examples.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 7.5,
            "font.family": ["DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
            "figure.dpi": 180,
            "savefig.dpi": 180,
        }
    )

    fig = plt.figure(figsize=(8.4, 6.4), constrained_layout=False)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.04, 1.0), height_ratios=(1.0, 1.0))
    prediction_axis = fig.add_subplot(grid[0, 0])
    convergence_axis = fig.add_subplot(grid[1, 0])
    state_axis = fig.add_subplot(grid[0, 1])
    input_axis = fig.add_subplot(grid[1, 1], sharex=state_axis)

    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    linestyles = ("-", "--", "-.", ":")

    for horizon, color, linestyle in zip(PREDICTION_HORIZONS, colors, linestyles):
        states, _ = simulate_time_varying_plan(horizon, BASE_STATE_WEIGHT, BASE_INPUT_WEIGHT)
        time = np.arange(horizon + 1) * SAMPLE_TIME
        prediction_axis.plot(
            time,
            states[:, 0],
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            marker="o",
            markersize=2.6,
            markevery=max(1, horizon // 8),
            label=rf"$N={horizon}$",
        )
        prediction_axis.plot(time[-1], states[-1, 0], marker="s", color=color, markersize=4.0)

    prediction_axis.axhline(0.0, color="#555555", linestyle=":", linewidth=0.9)
    prediction_axis.set_xlabel("Prediction time [s]")
    prediction_axis.set_ylabel(r"Position deviation $x_1$ [-]")
    prediction_axis.legend(loc="upper right")
    prediction_axis.set_ylim(-0.22, 1.08)
    configure_axis(prediction_axis)
    add_panel_label(prediction_axis, "(a) finite horizon")

    scenarios = (
        (r"$R_u=0.05$", BASE_STATE_WEIGHT, 0.05, colors[0], "-"),
        (r"$R_u=0.20$", BASE_STATE_WEIGHT, 0.20, colors[1], "--"),
        (r"$R_u=1.00$", BASE_STATE_WEIGHT, 1.00, colors[2], "-."),
        (r"$q_1=3,\ R_u=0.20$", np.diag([3.0, 0.08]), 0.20, colors[3], ":"),
    )

    time = np.arange(SIMULATION_STEPS + 1) * SAMPLE_TIME
    input_time = np.arange(SIMULATION_STEPS) * SAMPLE_TIME
    for label, state_weight, input_weight, color, linestyle in scenarios:
        states, inputs = simulate_receding_horizon(
            SIMULATION_STEPS,
            RECEDING_HORIZON,
            state_weight,
            input_weight,
        )
        state_axis.plot(
            time,
            states[:, 0],
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            label=label,
        )
        input_axis.step(
            input_time,
            inputs,
            where="post",
            color=color,
            linestyle=linestyle,
            linewidth=1.85,
            label=label,
        )

    state_axis.axhline(0.0, color="#555555", linestyle=":", linewidth=0.9)
    state_axis.set_ylabel(r"Position deviation $x_1$ [-]")
    state_axis.set_ylim(-0.24, 1.08)
    state_axis.legend(loc="upper right")
    configure_axis(state_axis)
    add_panel_label(state_axis, "(b) state trace")

    for limit in (-INPUT_LIMIT, INPUT_LIMIT):
        input_axis.axhline(limit, color="#555555", linestyle=":", linewidth=0.9)
    input_axis.set_xlabel("Time [s]")
    input_axis.set_ylabel(r"Input $u_k$ [-]")
    input_axis.set_ylim(-4.15, 2.45)
    configure_axis(input_axis)
    add_panel_label(input_axis, "(c) input trace")
    input_axis.text(
        0.98,
        0.90,
        r"limit $\pm 2$",
        transform=input_axis.transAxes,
        ha="right",
        va="top",
        fontsize=8.0,
        bbox={
            "boxstyle": "round,pad=0.20",
            "facecolor": "white",
            "edgecolor": "#bbbbbb",
            "alpha": 0.92,
        },
    )

    steady_cost = solve_dare(BASE_STATE_WEIGHT, BASE_INPUT_WEIGHT)
    steady_gain = gain_from_cost_to_go(steady_cost, BASE_INPUT_WEIGHT)
    cost_errors = []
    gain_errors = []
    for horizon in CONVERGENCE_HORIZONS:
        costs, gains = backward_riccati(int(horizon), BASE_STATE_WEIGHT, BASE_INPUT_WEIGHT)
        cost_errors.append(np.linalg.norm(costs[0] - steady_cost, ord="fro"))
        gain_errors.append(np.linalg.norm(gains[0] - steady_gain))

    convergence_axis.semilogy(
        CONVERGENCE_HORIZONS,
        cost_errors,
        color="#0072B2",
        linewidth=2.0,
        label=rf"$\|P_0(N)-P_\infty\|_F$",
    )
    convergence_axis.semilogy(
        CONVERGENCE_HORIZONS,
        gain_errors,
        color="#D55E00",
        linestyle="--",
        linewidth=2.0,
        label=rf"$\|K_0(N)-K_\infty\|_2$",
    )
    convergence_axis.set_xlabel("Horizon length N [samples]")
    convergence_axis.set_ylabel("Error to steady solution")
    convergence_axis.legend(loc="upper right")
    convergence_axis.grid(True, which="both", color="#d0d0d0", linewidth=0.7, alpha=0.72)
    add_panel_label(convergence_axis, "(d) Riccati convergence")

    finalize_figure(fig, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
