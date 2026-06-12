#!/usr/bin/env python3
"""Generate W05 innovation and normalized innovation squared trace examples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("numpy is required to generate the W05 innovation figure") from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("matplotlib is required to generate the W05 innovation figure") from exc

from figure_style import finalize_figure


SAMPLE_TIME = 0.1
SIMULATION_STEPS = 140
BURN_IN_STEPS = 20
NIS_WINDOW = 8
MEASUREMENT_DIMENSION = 2
CHI2_95_DF2 = 5.99

A_DISCRETE = np.array([[1.0, SAMPLE_TIME], [0.0, 1.0]])
B_DISCRETE = np.array([[0.5 * SAMPLE_TIME**2], [SAMPLE_TIME]])
C_MEASUREMENT = np.eye(2)
G_ACCELERATION = np.array([[0.5 * SAMPLE_TIME**2], [SAMPLE_TIME]])

ACCELERATION_STD = 0.22
BASE_PROCESS_COVARIANCE = (ACCELERATION_STD**2) * (G_ACCELERATION @ G_ACCELERATION.T)
BASE_MEASUREMENT_COVARIANCE = np.diag([0.035**2, 0.075**2])
INITIAL_TRUE_STATE = np.array([0.5, -0.25])
INITIAL_ESTIMATE = np.array([0.42, -0.18])
INITIAL_COVARIANCE = np.diag([0.12**2, 0.18**2])


@dataclass(frozen=True)
class Scenario:
    label: str
    color: str
    actual_process_scale: float = 1.0
    actual_measurement_scale: float = 1.0
    filter_process_scale: float = 1.0
    filter_measurement_scale: float = 1.0
    model_bias_amplitude: float = 0.0
    model_bias_offset: float = 0.0
    outlier_step: int | None = None
    outlier_vector: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class ScenarioResult:
    scenario: Scenario
    time: np.ndarray
    innovations: np.ndarray
    nis: np.ndarray
    nis_window_mean: np.ndarray


SCENARIOS = (
    Scenario("matched covariance", "#0072B2"),
    Scenario(
        "underestimated noise / model error",
        "#D55E00",
        actual_process_scale=1.6,
        actual_measurement_scale=1.8,
        filter_process_scale=0.45,
        filter_measurement_scale=0.45,
        model_bias_amplitude=0.08,
        model_bias_offset=0.04,
    ),
    Scenario(
        "overconservative covariance",
        "#009E73",
        filter_process_scale=7.0,
        filter_measurement_scale=7.0,
    ),
    Scenario(
        "single outlier",
        "#7C3AED",
        outlier_step=72,
        outlier_vector=(0.20, -0.40),
    ),
)


def known_input(step: int) -> float:
    """Known cart force command used by the plant and the estimator."""
    time = step * SAMPLE_TIME
    return 0.35 * np.sin(0.65 * time) - 0.18 * np.cos(0.18 * time)


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    padded = np.pad(values, (window - 1, 0), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


def simulate_scenario(scenario: Scenario, seed: int = 11) -> ScenarioResult:
    rng = np.random.default_rng(seed)
    true_state = INITIAL_TRUE_STATE.copy()
    estimate = INITIAL_ESTIMATE.copy()
    covariance = INITIAL_COVARIANCE.copy()

    actual_measurement_covariance = BASE_MEASUREMENT_COVARIANCE * scenario.actual_measurement_scale
    filter_measurement_covariance = BASE_MEASUREMENT_COVARIANCE * scenario.filter_measurement_scale
    filter_process_covariance = BASE_PROCESS_COVARIANCE * scenario.filter_process_scale

    innovations: list[np.ndarray] = []
    nis_values: list[float] = []

    for step in range(SIMULATION_STEPS):
        command = known_input(step)
        acceleration_noise = rng.normal(
            0.0,
            ACCELERATION_STD * np.sqrt(scenario.actual_process_scale),
        )
        model_bias = (
            scenario.model_bias_amplitude * np.sin(0.23 * step * SAMPLE_TIME)
            + scenario.model_bias_offset
        )
        true_state = (
            A_DISCRETE @ true_state
            + B_DISCRETE[:, 0] * command
            + G_ACCELERATION[:, 0] * (acceleration_noise + model_bias)
        )

        measurement = C_MEASUREMENT @ true_state
        measurement += rng.multivariate_normal(np.zeros(MEASUREMENT_DIMENSION), actual_measurement_covariance)
        if scenario.outlier_step is not None and step == scenario.outlier_step:
            measurement += np.array(scenario.outlier_vector)

        predicted_estimate = A_DISCRETE @ estimate + B_DISCRETE[:, 0] * command
        predicted_covariance = (
            A_DISCRETE @ covariance @ A_DISCRETE.T
            + filter_process_covariance
        )
        innovation = measurement - C_MEASUREMENT @ predicted_estimate
        innovation_covariance = (
            C_MEASUREMENT @ predicted_covariance @ C_MEASUREMENT.T
            + filter_measurement_covariance
        )

        nis = float(innovation.T @ np.linalg.solve(innovation_covariance, innovation))
        gain = np.linalg.solve(innovation_covariance, C_MEASUREMENT @ predicted_covariance).T
        estimate = predicted_estimate + gain @ innovation
        identity = np.eye(2)
        covariance = (
            (identity - gain @ C_MEASUREMENT)
            @ predicted_covariance
            @ (identity - gain @ C_MEASUREMENT).T
            + gain @ filter_measurement_covariance @ gain.T
        )
        covariance = 0.5 * (covariance + covariance.T)

        innovations.append(innovation)
        nis_values.append(nis)

    time = np.arange(SIMULATION_STEPS) * SAMPLE_TIME
    nis_array = np.asarray(nis_values)
    return ScenarioResult(
        scenario=scenario,
        time=time,
        innovations=np.asarray(innovations),
        nis=nis_array,
        nis_window_mean=rolling_mean(nis_array, NIS_WINDOW),
    )


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        0.015,
        0.94,
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


def configure_trace_axis(axis: plt.Axes) -> None:
    axis.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.75)
    axis.axhline(0.0, color="#555555", linestyle=":", linewidth=0.9)


def draw_nis_axis(axis: plt.Axes, results: list[ScenarioResult]) -> None:
    for result in results:
        axis.plot(
            result.time,
            result.nis,
            color=result.scenario.color,
            linewidth=0.85,
            alpha=0.25,
        )
        axis.plot(
            result.time,
            result.nis_window_mean,
            color=result.scenario.color,
            linewidth=2.0,
            label=result.scenario.label,
        )

    axis.axhline(
        MEASUREMENT_DIMENSION,
        color="#333333",
        linestyle="-",
        linewidth=1.0,
        label=rf"expected mean $m_y={MEASUREMENT_DIMENSION}$",
    )
    axis.axhline(
        CHI2_95_DF2,
        color="#777777",
        linestyle="--",
        linewidth=1.0,
        label=r"$\chi^2_2$ 95% = 5.99",
    )
    axis.set_xlabel(r"Time $t$ [s]")
    axis.set_ylabel(r"NIS $\epsilon_k$ [-]")
    axis.set_ylim(0.0, 52.0)
    axis.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.75)
    axis.legend(loc="upper right", ncol=2, frameon=True)
    add_panel_label(axis, r"(a) raw $\epsilon_k$ and 8-sample mean")


def draw_innovation_axis(
    axis: plt.Axes,
    results: list[ScenarioResult],
    *,
    component: int,
    ylabel: str,
    panel_label: str,
) -> None:
    for result in results:
        axis.plot(
            result.time,
            result.innovations[:, component],
            color=result.scenario.color,
            linewidth=1.45,
            alpha=0.88,
        )
    configure_trace_axis(axis)
    axis.set_xlabel(r"Time $t$ [s]")
    axis.set_ylabel(ylabel)
    add_panel_label(axis, panel_label)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "ja" / "figures" / "w05_innovation_trace_examples.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 7.4,
            "font.family": ["DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
            "figure.dpi": 180,
            "savefig.dpi": 180,
        }
    )

    results = [simulate_scenario(scenario) for scenario in SCENARIOS]

    fig = plt.figure(figsize=(8.2, 6.4), constrained_layout=False)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.23, 1.0))
    nis_axis = fig.add_subplot(grid[0, :])
    position_axis = fig.add_subplot(grid[1, 0], sharex=nis_axis)
    velocity_axis = fig.add_subplot(grid[1, 1], sharex=nis_axis)

    draw_nis_axis(nis_axis, results)
    draw_innovation_axis(
        position_axis,
        results,
        component=0,
        ylabel=r"Innovation $\nu_{p,k}$ [m]",
        panel_label=r"(b) position innovation",
    )
    draw_innovation_axis(
        velocity_axis,
        results,
        component=1,
        ylabel=r"Innovation $\nu_{v,k}$ [m/s]",
        panel_label=r"(c) velocity innovation",
    )

    finalize_figure(fig, output_path)

    print(output_path)
    print(f"measurement_dimension={MEASUREMENT_DIMENSION}, expected_nis_mean={MEASUREMENT_DIMENSION:.2f}")
    for result in results:
        steady_nis = result.nis[BURN_IN_STEPS:]
        print(
            f"{result.scenario.label}: "
            f"mean_nis_after_burn_in={steady_nis.mean():.2f}, "
            f"median={np.median(steady_nis):.2f}, "
            f"peak={result.nis.max():.2f}"
        )


if __name__ == "__main__":
    main()
