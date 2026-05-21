#!/usr/bin/env python3
"""Generate performance-index and Pareto examples for a motor position loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("matplotlib is required to generate performance figures") from exc

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

INERTIA = 0.20
DAMPING = 0.25
TORQUE_COEFF = 1.0
REFERENCE = 1.0
INPUT_LIMIT = 6.0
TIME_STEP = 0.001
TIME_STOP = 5.0
SETTLING_BAND = 0.02


@dataclass(frozen=True)
class Candidate:
    label: str
    kp: float
    kd: float
    color: str
    linestyle: str


@dataclass(frozen=True)
class Metrics:
    settling_time: float
    overshoot_percent: float
    iae: float
    ise: float
    itae: float
    rms_input: float
    peak_input: float
    constraint_margin: float


@dataclass(frozen=True)
class SimulationResult:
    time: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    command_input: np.ndarray
    applied_input: np.ndarray
    metrics: Metrics


CANDIDATES = (
    Candidate("穏やか", kp=2.0, kd=1.10, color="#2b6cb0", linestyle="-"),
    Candidate("低減衰", kp=4.0, kd=0.15, color="#c2410c", linestyle="--"),
    Candidate("釣合い", kp=4.0, kd=0.90, color="#2f855a", linestyle="-."),
    Candidate("強引", kp=7.5, kd=1.15, color="#6b46c1", linestyle=":"),
)


def configure_fonts() -> None:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    selected_fonts = [
        font_name for font_name in JAPANESE_FONT_CANDIDATES if font_name in available_fonts
    ]
    if not selected_fonts:
        raise SystemExit(
            "A Japanese-capable Matplotlib font is required; install "
            "Noto Sans CJK JP, Harano Aji Gothic, or another listed font."
        )
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = selected_fonts + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def command_input(candidate: Candidate, state: np.ndarray) -> float:
    position, velocity = state
    return candidate.kp * (REFERENCE - position) - candidate.kd * velocity


def motor_dynamics(candidate: Candidate, state: np.ndarray) -> np.ndarray:
    position, velocity = state
    limited_input = float(np.clip(command_input(candidate, state), -INPUT_LIMIT, INPUT_LIMIT))
    acceleration = (TORQUE_COEFF * limited_input - DAMPING * velocity) / INERTIA
    return np.array([velocity, acceleration])


def integrate(candidate: Candidate, *, time_step: float = TIME_STEP) -> SimulationResult:
    time = np.arange(0.0, TIME_STOP + 0.5 * time_step, time_step)
    state = np.zeros(2)
    states = np.zeros((len(time), 2))
    command = np.zeros_like(time)
    applied = np.zeros_like(time)

    for index, current_time in enumerate(time):
        states[index] = state
        command[index] = command_input(candidate, state)
        applied[index] = np.clip(command[index], -INPUT_LIMIT, INPUT_LIMIT)
        if index == len(time) - 1:
            break

        step = time[index + 1] - current_time
        k1 = motor_dynamics(candidate, state)
        k2 = motor_dynamics(candidate, state + 0.5 * step * k1)
        k3 = motor_dynamics(candidate, state + 0.5 * step * k2)
        k4 = motor_dynamics(candidate, state + step * k3)
        state = state + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    position = states[:, 0]
    velocity = states[:, 1]
    return SimulationResult(
        time=time,
        position=position,
        velocity=velocity,
        command_input=command,
        applied_input=applied,
        metrics=compute_metrics(time, position, command),
    )


def integrate_area(time: np.ndarray, signal: np.ndarray) -> float:
    return float(np.trapezoid(signal, time))


def compute_metrics(
    time: np.ndarray,
    position: np.ndarray,
    command: np.ndarray,
) -> Metrics:
    error = REFERENCE - position
    tolerance = SETTLING_BAND * abs(REFERENCE)
    within_band = np.abs(error) <= tolerance
    settling_time = float("inf")
    for index, is_inside in enumerate(within_band):
        if is_inside and np.all(within_band[index:]):
            settling_time = float(time[index])
            break

    overshoot_percent = max(0.0, (float(np.max(position)) - REFERENCE) / REFERENCE) * 100.0
    iae = integrate_area(time, np.abs(error))
    ise = integrate_area(time, error * error)
    itae = integrate_area(time, time * np.abs(error))
    rms_input = float(np.sqrt(integrate_area(time, command * command) / (time[-1] - time[0])))
    peak_input = float(np.max(np.abs(command)))
    constraint_margin = 1.0 - peak_input / INPUT_LIMIT
    return Metrics(
        settling_time=settling_time,
        overshoot_percent=overshoot_percent,
        iae=iae,
        ise=ise,
        itae=itae,
        rms_input=rms_input,
        peak_input=peak_input,
        constraint_margin=constraint_margin,
    )


def settling_time_text(value: float) -> str:
    if np.isfinite(value):
        return f"{value:.2f}"
    return f">{TIME_STOP:.1f}"


def gain_label(candidate: Candidate) -> str:
    return rf"{candidate.label} ($K_P={candidate.kp:g},\,K_D={candidate.kd:g}$)"


def candidate_results() -> dict[Candidate, SimulationResult]:
    return {candidate: integrate(candidate) for candidate in CANDIDATES}


def sweep_candidates() -> list[tuple[float, float, float, float, float]]:
    points: list[tuple[float, float, float, float, float]] = []
    for kp in np.linspace(1.2, 8.0, 16):
        for zeta in np.linspace(0.28, 1.25, 10):
            kd = max(0.0, 2.0 * zeta * np.sqrt(INERTIA * TORQUE_COEFF * kp) - DAMPING)
            candidate = Candidate("sweep", kp=float(kp), kd=float(kd), color="#666666", linestyle="-")
            result = integrate(candidate, time_step=0.004)
            metrics = result.metrics
            points.append(
                (
                    metrics.iae,
                    metrics.rms_input,
                    metrics.constraint_margin,
                    float(kp),
                    float(kd),
                )
            )
    return points


def pareto_front(points: list[tuple[float, float, float, float, float]]) -> list[tuple[float, float]]:
    feasible = [point for point in points if point[2] >= 0.0]
    front: list[tuple[float, float]] = []
    for iae, rms_input, _margin, _kp, _kd in feasible:
        dominated = False
        for other_iae, other_rms, _other_margin, _other_kp, _other_kd in feasible:
            no_worse = other_iae <= iae and other_rms <= rms_input
            strictly_better = other_iae < iae or other_rms < rms_input
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append((iae, rms_input))
    return sorted(front)


def draw_responses(axis, results: dict[Candidate, SimulationResult]) -> None:
    axis.axhline(
        REFERENCE,
        color="#333333",
        linestyle=":",
        linewidth=1.2,
        label=r"目標値 $r=1\,\mathrm{rad}$",
    )
    axis.axhspan(
        REFERENCE * (1.0 - SETTLING_BAND),
        REFERENCE * (1.0 + SETTLING_BAND),
        color="#94a3b8",
        alpha=0.18,
        linewidth=0.0,
    )
    for candidate, result in results.items():
        axis.plot(
            result.time,
            result.position,
            color=candidate.color,
            linestyle=candidate.linestyle,
            linewidth=2.0,
            label=gain_label(candidate),
        )
    axis.text(
        0.02,
        0.94,
        "(a) 応答と 2% 帯",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#cbd5e1"},
    )
    axis.set_xlabel(r"時間 $t$ [s]")
    axis.set_ylabel(r"角度 $\theta(t)$ [rad]")
    axis.set_xlim(0.0, TIME_STOP)
    axis.set_ylim(-0.12, 1.62)
    axis.grid(True, color="#d5d5d5", linewidth=0.7, alpha=0.75)
    axis.legend(loc="lower right", fontsize=7.0)


def draw_inputs(axis, results: dict[Candidate, SimulationResult]) -> None:
    axis.axhline(INPUT_LIMIT, color="#991b1b", linestyle=":", linewidth=1.2)
    axis.axhline(-INPUT_LIMIT, color="#991b1b", linestyle=":", linewidth=1.2)
    axis.fill_between(
        [0.0, TIME_STOP],
        [-INPUT_LIMIT, -INPUT_LIMIT],
        [INPUT_LIMIT, INPUT_LIMIT],
        color="#fee2e2",
        alpha=0.26,
        linewidth=0.0,
    )
    for candidate, result in results.items():
        axis.plot(
            result.time,
            result.command_input,
            color=candidate.color,
            linestyle=candidate.linestyle,
            linewidth=1.8,
            label=candidate.label,
        )
    axis.text(
        0.02,
        0.94,
        "(b) 飽和前入力",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#cbd5e1"},
    )
    axis.text(
        0.98,
        0.08,
        r"$|u_c|\leq6\,\mathrm{V}$",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.0,
        color="#7f1d1d",
    )
    axis.set_xlabel(r"時間 $t$ [s]")
    axis.set_ylabel(r"入力指令 $u_c$ [V]")
    axis.set_xlim(0.0, TIME_STOP)
    axis.set_ylim(-3.2, 8.1)
    axis.grid(True, color="#d5d5d5", linewidth=0.7, alpha=0.75)


def draw_pareto(axis, results: dict[Candidate, SimulationResult]) -> None:
    points = sweep_candidates()
    feasible = np.array([point for point in points if point[2] >= 0.0])
    infeasible = np.array([point for point in points if point[2] < 0.0])

    if len(feasible) > 0:
        axis.scatter(
            feasible[:, 0],
            feasible[:, 1],
            s=18,
            color="#94a3b8",
            alpha=0.55,
            edgecolor="none",
            label="制約内候補",
        )
    if len(infeasible) > 0:
        axis.scatter(
            infeasible[:, 0],
            infeasible[:, 1],
            s=18,
            facecolor="none",
            edgecolor="#ef4444",
            linewidth=0.8,
            alpha=0.65,
            label="余裕不足候補",
        )

    front = pareto_front(points)
    if front:
        front_array = np.array(front)
        axis.plot(
            front_array[:, 0],
            front_array[:, 1],
            color="#111827",
            linewidth=1.8,
            label="Pareto 前面",
        )

    for candidate, result in results.items():
        metrics = result.metrics
        marker = "X" if metrics.constraint_margin < 0.0 else "o"
        axis.scatter(
            metrics.iae,
            metrics.rms_input,
            s=78,
            marker=marker,
            color=candidate.color,
            edgecolor="white",
            linewidth=0.9,
            zorder=5,
        )
        axis.annotate(
            candidate.label,
            (metrics.iae, metrics.rms_input),
            textcoords="offset points",
            xytext=(7, 5),
            ha="left",
            va="bottom",
            fontsize=8.0,
            color="#111827",
        )

    axis.text(
        0.02,
        0.94,
        "(c) IAE-RMS Pareto 図",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#cbd5e1"},
    )
    axis.set_xlabel(r"IAE [rad s]")
    axis.set_ylabel(r"RMS 入力 [V]")
    axis.set_xlim(0.18, 1.16)
    axis.set_ylim(0.15, 1.35)
    axis.grid(True, color="#d5d5d5", linewidth=0.7, alpha=0.75)
    axis.legend(loc="upper right", fontsize=7.0)


def draw_metric_table(axis, results: dict[Candidate, SimulationResult]) -> None:
    axis.set_axis_off()
    axis.text(
        0.02,
        0.97,
        "(d) 指標の数値",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#cbd5e1"},
    )
    columns = ["候補", "$t_s$", "$M_p$", "IAE", "RMS", "$\\mu_{min}$"]
    rows = []
    for candidate, result in results.items():
        metrics = result.metrics
        rows.append(
            [
                candidate.label,
                settling_time_text(metrics.settling_time),
                f"{metrics.overshoot_percent:.1f}",
                f"{metrics.iae:.3f}",
                f"{metrics.rms_input:.3f}",
                f"{metrics.constraint_margin:.2f}",
            ]
        )

    table = axis.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        bbox=[0.02, 0.10, 0.96, 0.72],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor("#e2e8f0")
        else:
            cell.set_facecolor("#ffffff")
    axis.text(
        0.02,
        0.04,
        r"$t_s$ [s], $M_p$ [%], RMS [V], $\mu_{min}=1-\max|u_c|/6$",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        color="#334155",
    )


def print_metrics(results: dict[Candidate, SimulationResult]) -> None:
    print("candidate,Kp,Kd,settling_time,overshoot,IAE,ISE,ITAE,RMS_input,peak_input,margin")
    for candidate, result in results.items():
        metrics = result.metrics
        print(
            f"{candidate.label},{candidate.kp:.3f},{candidate.kd:.3f},"
            f"{settling_time_text(metrics.settling_time)},"
            f"{metrics.overshoot_percent:.3f},{metrics.iae:.6f},"
            f"{metrics.ise:.6f},{metrics.itae:.6f},"
            f"{metrics.rms_input:.6f},{metrics.peak_input:.6f},"
            f"{metrics.constraint_margin:.6f}"
        )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "ja" / "figures" / "performance_pareto_examples.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    configure_fonts()
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 180,
            "savefig.dpi": 180,
        }
    )

    results = candidate_results()
    fig = plt.figure(figsize=(7.6, 7.4), constrained_layout=False)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.05, 1.0), width_ratios=(1.05, 0.95))
    response_axis = fig.add_subplot(grid[0, 0])
    input_axis = fig.add_subplot(grid[0, 1])
    pareto_axis = fig.add_subplot(grid[1, 0])
    table_axis = fig.add_subplot(grid[1, 1])

    draw_responses(response_axis, results)
    draw_inputs(input_axis, results)
    draw_pareto(pareto_axis, results)
    draw_metric_table(table_axis, results)

    finalize_figure(fig, output_path)
    print(output_path)
    print_metrics(results)


if __name__ == "__main__":
    main()
