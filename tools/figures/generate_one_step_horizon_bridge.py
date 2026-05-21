#!/usr/bin/env python3
"""Generate a one-step horizon bridge figure for scalar performance scoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("numpy is required to generate the one-step horizon figure") from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("matplotlib is required to generate the one-step horizon figure") from exc

from figure_style import finalize_figure


SAMPLE_TIME = 0.10
INERTIA = 0.20
DAMPING = 0.25
TORQUE_COEFF = 1.0
REFERENCE = 1.0
INPUT_LIMIT = 6.0
PREVIOUS_INPUT = 1.5
CURRENT_STATE = np.array([0.70, 0.45])
CANDIDATE_INPUTS = np.array([-INPUT_LIMIT, 0.0, 0.5 * INPUT_LIMIT, INPUT_LIMIT])

Q_ERROR = 5.0
Q_SPEED = 0.20
R_INPUT = 0.35
R_DELTA = 0.15

COLORS = ("#2b6cb0", "#2f855a", "#d97706", "#7c3aed")
COMPONENT_COLORS = {
    "predicted error": "#2563eb",
    "predicted speed": "#059669",
    "input size": "#f59e0b",
    "input change": "#8b5cf6",
}


@dataclass(frozen=True)
class CandidateResult:
    candidate_input: float
    next_state: np.ndarray
    cost_components: dict[str, float]

    @property
    def total_cost(self) -> float:
        return float(sum(self.cost_components.values()))


def predict_next_state(state: np.ndarray, candidate_input: float) -> np.ndarray:
    position, velocity = state
    acceleration = (TORQUE_COEFF * candidate_input - DAMPING * velocity) / INERTIA
    next_position = position + SAMPLE_TIME * velocity + 0.5 * SAMPLE_TIME**2 * acceleration
    next_velocity = velocity + SAMPLE_TIME * acceleration
    return np.array([next_position, next_velocity])


def evaluate_candidate(candidate_input: float) -> CandidateResult:
    next_state = predict_next_state(CURRENT_STATE, candidate_input)
    position_error = REFERENCE - next_state[0]
    components = {
        "predicted error": Q_ERROR * position_error**2,
        "predicted speed": Q_SPEED * next_state[1] ** 2,
        "input size": R_INPUT * (candidate_input / INPUT_LIMIT) ** 2,
        "input change": R_DELTA * ((candidate_input - PREVIOUS_INPUT) / INPUT_LIMIT) ** 2,
    }
    return CandidateResult(
        candidate_input=float(candidate_input),
        next_state=next_state,
        cost_components={name: float(value) for name, value in components.items()},
    )


def input_label(candidate_input: float) -> str:
    normalized = candidate_input / INPUT_LIMIT
    if np.isclose(normalized, -1.0):
        return r"$-u_{\max}$"
    if np.isclose(normalized, 0.0):
        return r"$0$"
    if np.isclose(normalized, 1.0):
        return r"$+u_{\max}$"
    return rf"${normalized:.1f}u_{{\max}}$"


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        0.02,
        0.96,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9.4,
        fontweight="bold",
        color="#111827",
        bbox={
            "boxstyle": "round,pad=0.20",
            "facecolor": "white",
            "edgecolor": "#cbd5e1",
            "alpha": 0.96,
        },
    )


def draw_prediction_panel(axis: plt.Axes, results: list[CandidateResult]) -> None:
    axis.axvline(REFERENCE, color="#555555", linestyle=":", linewidth=1.0)
    axis.text(
        REFERENCE - 0.012,
        -2.95,
        r"reference $r_{k+1}$",
        ha="right",
        va="bottom",
        fontsize=8.2,
        color="#555555",
    )
    axis.scatter(
        [CURRENT_STATE[0]],
        [CURRENT_STATE[1]],
        s=100,
        marker="s",
        color="#111827",
        edgecolor="white",
        linewidth=1.0,
        zorder=5,
        label=r"current state $x_k$",
    )
    axis.annotate(
        r"$x_k$",
        xy=tuple(CURRENT_STATE),
        xytext=(-18, 14),
        textcoords="offset points",
        fontsize=10,
        color="#111827",
    )

    for result, color in zip(results, COLORS):
        start = tuple(CURRENT_STATE)
        end = tuple(result.next_state)
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            mutation_scale=13,
            linewidth=1.8,
            color=color,
            alpha=0.88,
        )
        axis.add_patch(arrow)
        axis.scatter(
            [result.next_state[0]],
            [result.next_state[1]],
            s=80,
            marker="o",
            color=color,
            edgecolor="white",
            linewidth=0.9,
            zorder=4,
        )
        axis.annotate(
            input_label(result.candidate_input),
            xy=tuple(result.next_state),
            xytext=(7, -2),
            textcoords="offset points",
            fontsize=8.2,
            color=color,
            ha="left",
            va="center",
        )

    axis.set_xlabel(r"Predicted angle $\theta_{k+1}$ [rad]")
    axis.set_ylabel(r"Predicted speed $\omega_{k+1}$ [rad/s]")
    axis.set_xlim(0.54, 1.03)
    axis.set_ylim(-3.25, 3.85)
    axis.grid(True, color="#d5d5d5", linewidth=0.7, alpha=0.75)
    add_panel_label(axis, r"(a) $x_k \rightarrow x_{k+1}=f_d(x_k,u_k)$")


def draw_constraint_panel(axis: plt.Axes, results: list[CandidateResult]) -> None:
    axis.axhline(0.0, color="#333333", linewidth=1.0)
    axis.axvspan(-1.0, 1.0, ymin=0.25, ymax=0.75, color="#dcfce7", alpha=0.72)
    axis.axvline(-1.0, color="#991b1b", linestyle=":", linewidth=1.2)
    axis.axvline(1.0, color="#991b1b", linestyle=":", linewidth=1.2)
    axis.text(
        0.0,
        0.27,
        r"feasible interval: $|u_k|\leq u_{\max}$",
        ha="center",
        va="center",
        fontsize=9.0,
        color="#166534",
    )
    for result, color in zip(results, COLORS):
        normalized = result.candidate_input / INPUT_LIMIT
        axis.scatter(
            [normalized],
            [0.0],
            s=86,
            color=color,
            edgecolor="white",
            linewidth=0.9,
            zorder=4,
        )
        axis.annotate(
            input_label(result.candidate_input),
            xy=(normalized, 0.0),
            xytext=(0, 13),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=color,
        )
    axis.set_xlim(-1.28, 1.28)
    axis.set_ylim(-0.52, 0.52)
    axis.set_xlabel(r"Candidate input $u_k/u_{\max}$ [-]")
    axis.set_yticks([])
    axis.grid(True, axis="x", color="#d5d5d5", linewidth=0.7, alpha=0.75)
    add_panel_label(axis, "(b) input constraint")


def draw_cost_panel(axis: plt.Axes, results: list[CandidateResult]) -> None:
    positions = np.arange(len(results))
    bottoms = np.zeros(len(results))
    for name, color in COMPONENT_COLORS.items():
        values = np.array([result.cost_components[name] for result in results])
        axis.bar(
            positions,
            values,
            bottom=bottoms,
            color=color,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.7,
            label=name,
        )
        bottoms += values

    best_index = int(np.argmin([result.total_cost for result in results]))
    axis.scatter(
        [best_index],
        [results[best_index].total_cost + 0.08],
        marker="v",
        color="#111827",
        s=70,
        zorder=5,
    )
    axis.text(
        best_index,
        results[best_index].total_cost + 0.16,
        r"smallest $J_1$",
        ha="center",
        va="bottom",
        fontsize=8.4,
        color="#111827",
    )
    axis.set_xticks(positions)
    axis.set_xticklabels([input_label(result.candidate_input) for result in results])
    axis.set_ylabel(r"One-step cost pieces [-]")
    axis.set_xlabel(r"Candidate input $u_k$")
    axis.set_ylim(0.0, max(result.total_cost for result in results) * 1.24)
    axis.grid(True, axis="y", color="#d5d5d5", linewidth=0.7, alpha=0.75)
    axis.legend(loc="upper right", ncols=1, fontsize=7.8, framealpha=0.92)
    add_panel_label(axis, r"(c) cost pieces")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "ja" / "figures" / "one_step_horizon_bridge.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "font.family": ["DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
            "figure.dpi": 180,
            "savefig.dpi": 180,
        }
    )

    results = [evaluate_candidate(candidate_input) for candidate_input in CANDIDATE_INPUTS]
    fig = plt.figure(figsize=(8.4, 5.2), constrained_layout=False)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.05), width_ratios=(1.10, 1.0))
    prediction_axis = fig.add_subplot(grid[:, 0])
    constraint_axis = fig.add_subplot(grid[0, 1])
    cost_axis = fig.add_subplot(grid[1, 1])

    draw_prediction_panel(prediction_axis, results)
    draw_constraint_panel(constraint_axis, results)
    draw_cost_panel(cost_axis, results)

    finalize_figure(fig, output_path)
    print(output_path)
    print("input,total_cost,theta_next,omega_next")
    for result in results:
        print(
            f"{result.candidate_input:.3f},"
            f"{result.total_cost:.6f},"
            f"{result.next_state[0]:.6f},"
            f"{result.next_state[1]:.6f}"
        )


if __name__ == "__main__":
    main()
