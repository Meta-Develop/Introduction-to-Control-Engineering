#!/usr/bin/env python3
"""Generate derivative-filter and two-degree-of-freedom PID examples."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("matplotlib is required to generate the PID figure") from exc

from figure_style import finalize_figure


RNG = np.random.default_rng(20260520)

FILTER_TIME_STEP = 0.002
FILTER_TIME_STOP = 3.0
FILTER_TAU_FAST = 0.02
FILTER_TAU_SLOW = 0.08

PID_TIME_STEP = 0.002
PID_TIME_STOP = 5.5
PID_PLANT_TAU = 0.6
PID_KP = 3.0
PID_KI = 1.8
PID_KD = 0.12
PID_FILTER_TAU = 0.04
PID_BETA_FULL = 1.0
PID_BETA_WEIGHTED = 0.45
PID_REFERENCE_STEP_TIME = 0.4
PID_DISTURBANCE_STEP_TIME = 2.0
PID_DISTURBANCE = -0.25


def filtered_derivative(values: np.ndarray, step: float, tau: float) -> np.ndarray:
    derivative = np.zeros_like(values)
    a = tau / (tau + step)
    for index in range(1, len(values)):
        difference = (values[index] - values[index - 1]) / step
        derivative[index] = a * derivative[index - 1] + (1.0 - a) * difference
    return derivative


def line_sensor_signals() -> dict[str, np.ndarray]:
    time = np.arange(0.0, FILTER_TIME_STOP + FILTER_TIME_STEP, FILTER_TIME_STEP)
    clean = (
        5.0 * np.exp(-0.55 * time) * np.sin(2.0 * np.pi * 1.05 * time)
        + 1.5 * np.sin(2.0 * np.pi * 0.35 * time)
    )
    periodic_noise = 0.25 * np.sin(2.0 * np.pi * 47.0 * time)
    random_noise = 0.28 * RNG.standard_normal(len(time))
    measured = 0.2 * np.round((clean + periodic_noise + random_noise) / 0.2)
    raw_derivative = np.zeros_like(time)
    raw_derivative[1:] = np.diff(measured) / FILTER_TIME_STEP
    true_derivative = np.gradient(clean, FILTER_TIME_STEP)
    fast_derivative = filtered_derivative(
        measured, FILTER_TIME_STEP, FILTER_TAU_FAST
    )
    slow_derivative = filtered_derivative(
        measured, FILTER_TIME_STEP, FILTER_TAU_SLOW
    )
    return {
        "time": time,
        "clean": clean,
        "measured": measured,
        "raw_derivative": raw_derivative,
        "true_derivative": true_derivative,
        "fast_derivative": fast_derivative,
        "slow_derivative": slow_derivative,
    }


def simulate_two_dof_pid(beta: float, scenario: str) -> dict[str, np.ndarray]:
    time = np.arange(0.0, PID_TIME_STOP + PID_TIME_STEP, PID_TIME_STEP)
    output = np.zeros_like(time)
    reference = np.zeros_like(time)
    disturbance = np.zeros_like(time)
    command = np.zeros_like(time)

    if scenario == "disturbance":
        y = 1.0
        eta = 1.0 - PID_KP * (beta - 1.0)
        previous_y = 1.0
    elif scenario == "command":
        y = 0.0
        eta = 0.0
        previous_y = 0.0
    else:
        raise ValueError(f"unknown scenario: {scenario}")

    derivative_y = 0.0
    a = PID_FILTER_TAU / (PID_FILTER_TAU + PID_TIME_STEP)

    for index, current_time in enumerate(time):
        if index > 0:
            difference = (y - previous_y) / PID_TIME_STEP
            derivative_y = a * derivative_y + (1.0 - a) * difference

        if scenario == "command":
            r = 1.0 if current_time >= PID_REFERENCE_STEP_TIME else 0.0
            d = 0.0
        else:
            r = 1.0
            d = PID_DISTURBANCE if current_time >= PID_DISTURBANCE_STEP_TIME else 0.0

        u = PID_KP * (beta * r - y) + eta - PID_KD * derivative_y

        output[index] = y
        reference[index] = r
        disturbance[index] = d
        command[index] = u

        if index == len(time) - 1:
            break

        error = r - y
        eta += PID_TIME_STEP * PID_KI * error
        previous_y = y
        y += PID_TIME_STEP * (-y + u + d) / PID_PLANT_TAU

    return {
        "time": time,
        "output": output,
        "reference": reference,
        "disturbance": disturbance,
        "command": command,
    }


def add_panel_label(axis, label: str) -> None:
    axis.text(
        0.02,
        0.93,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
    )


def style_axis(axis) -> None:
    axis.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.75)


def plot_filter_examples(signal_axis, derivative_axis) -> None:
    signals = line_sensor_signals()
    time = signals["time"]
    signal_axis.plot(
        time,
        signals["measured"],
        color="#777777",
        linewidth=0.9,
        alpha=0.65,
        label=r"measured $y_m$",
    )
    signal_axis.plot(
        time,
        signals["clean"],
        color="#1f77b4",
        linewidth=2.0,
        label="clean trend",
    )
    signal_axis.set_xlabel(r"Time $t$ [s]")
    signal_axis.set_ylabel(r"Line signal $y_m$ [mm]")
    signal_axis.set_xlim(0.0, FILTER_TIME_STOP)
    signal_axis.set_ylim(-6.4, 6.4)
    signal_axis.legend(loc="upper right", fontsize=8)
    style_axis(signal_axis)
    add_panel_label(signal_axis, "(a)")

    derivative_axis.plot(
        time,
        signals["raw_derivative"],
        color="#999999",
        linewidth=0.65,
        alpha=0.5,
        label="raw difference",
    )
    derivative_axis.plot(
        time,
        signals["true_derivative"],
        color="#1f77b4",
        linewidth=1.8,
        label="clean derivative",
    )
    derivative_axis.plot(
        time,
        signals["fast_derivative"],
        color="#ff7f0e",
        linewidth=1.7,
        label=rf"filtered $\tau_f={FILTER_TAU_FAST:.2f}$ s",
    )
    derivative_axis.plot(
        time,
        signals["slow_derivative"],
        color="#2ca02c",
        linewidth=1.7,
        label=rf"filtered $\tau_f={FILTER_TAU_SLOW:.2f}$ s",
    )
    derivative_axis.set_xlabel(r"Time $t$ [s]")
    derivative_axis.set_ylabel(r"Derivative estimate [mm/s]")
    derivative_axis.set_xlim(0.0, FILTER_TIME_STOP)
    derivative_axis.set_ylim(-260.0, 260.0)
    derivative_axis.legend(loc="upper right", fontsize=7.1)
    style_axis(derivative_axis)
    add_panel_label(derivative_axis, "(b)")


def plot_two_dof_examples(command_axis, disturbance_axis) -> None:
    command_full = simulate_two_dof_pid(PID_BETA_FULL, "command")
    command_weighted = simulate_two_dof_pid(PID_BETA_WEIGHTED, "command")
    disturbance_full = simulate_two_dof_pid(PID_BETA_FULL, "disturbance")
    disturbance_weighted = simulate_two_dof_pid(PID_BETA_WEIGHTED, "disturbance")
    time = command_full["time"]

    command_axis.plot(
        time,
        command_full["reference"],
        color="#444444",
        linestyle=":",
        linewidth=1.7,
        label="reference",
    )
    command_axis.plot(
        time,
        command_full["output"],
        color="#d62728",
        linewidth=1.9,
        label=rf"$\beta={PID_BETA_FULL:.2f}$",
    )
    command_axis.plot(
        time,
        command_weighted["output"],
        color="#1f77b4",
        linewidth=1.9,
        label=rf"$\beta={PID_BETA_WEIGHTED:.2f}$",
    )
    command_axis.axvline(
        PID_REFERENCE_STEP_TIME, color="#777777", linestyle="--", linewidth=1.0
    )
    command_axis.set_xlabel(r"Time $t$ [s]")
    command_axis.set_ylabel(r"Output $y$ [-]")
    command_axis.set_xlim(0.0, PID_TIME_STOP)
    command_axis.set_ylim(-0.05, 1.35)
    command_axis.legend(loc="lower right", fontsize=8)
    style_axis(command_axis)
    add_panel_label(command_axis, "(c)")

    disturbance_axis.plot(
        time,
        disturbance_full["output"] - disturbance_full["reference"],
        color="#d62728",
        linewidth=1.9,
        label=rf"$\beta={PID_BETA_FULL:.2f}$",
    )
    disturbance_axis.plot(
        time,
        disturbance_weighted["output"] - disturbance_weighted["reference"],
        color="#1f77b4",
        linewidth=1.9,
        linestyle="--",
        label=rf"$\beta={PID_BETA_WEIGHTED:.2f}$",
    )
    disturbance_axis.axvline(
        PID_DISTURBANCE_STEP_TIME,
        color="#777777",
        linestyle="--",
        linewidth=1.0,
        label=rf"$d={PID_DISTURBANCE:.2f}$",
    )
    disturbance_axis.set_xlabel(r"Time $t$ [s]")
    disturbance_axis.set_ylabel(r"Deviation $y-r$ [-]")
    disturbance_axis.set_xlim(0.0, PID_TIME_STOP)
    disturbance_axis.set_ylim(-0.09, 0.025)
    disturbance_axis.legend(loc="lower right", fontsize=8)
    style_axis(disturbance_axis)
    add_panel_label(disturbance_axis, "(d)")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "ja" / "figures" / "pid_filter_2dof_examples.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.4), constrained_layout=False)
    plot_filter_examples(axes[0, 0], axes[0, 1])
    plot_two_dof_examples(axes[1, 0], axes[1, 1])
    finalize_figure(fig, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
