#!/usr/bin/env python3
"""Generate ON-OFF, P, and PD line-following feedback traces."""

from __future__ import annotations

from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("numpy is required to generate the line-following figure") from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("matplotlib is required to generate the line-following figure") from exc

from figure_style import finalize_figure


V_FORWARD = 0.35  # m/s
WHEEL_BASE = 0.12  # m
INITIAL_ERROR = 0.06  # m
INITIAL_HEADING = 0.0  # rad, positive is left of the line direction
T_FINAL = 5.0  # s
DT = 0.002  # s

U_RELAY = 0.12  # m/s, left-minus-right wheel speed command
KP = 1.45  # 1/s
KD = 0.75  # dimensionless
SENSOR_RIPPLE = 0.006  # m
SENSOR_RIPPLE_HZ = 18.0  # Hz


def relay_command(error: float, time: float) -> float:
    measured_error = error + SENSOR_RIPPLE * np.sin(2.0 * np.pi * SENSOR_RIPPLE_HZ * time)
    if measured_error > 0.0:
        return U_RELAY
    if measured_error < 0.0:
        return -U_RELAY
    return 0.0


def simulate(controller: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.arange(0.0, T_FINAL + DT, DT)
    errors = np.zeros_like(times)
    headings = np.zeros_like(times)
    commands = np.zeros_like(times)
    errors[0] = INITIAL_ERROR
    headings[0] = INITIAL_HEADING

    for index, time in enumerate(times[:-1]):
        error = errors[index]
        heading = headings[index]
        error_rate = V_FORWARD * heading

        if controller == "relay":
            command = relay_command(error, time)
        elif controller == "p":
            command = KP * error
        elif controller == "pd":
            command = KP * error + KD * error_rate
        else:  # pragma: no cover - guarded by caller.
            raise ValueError(f"unknown controller: {controller}")

        commands[index] = command
        errors[index + 1] = error + DT * error_rate
        headings[index + 1] = heading - DT * command / WHEEL_BASE

    commands[-1] = commands[-2]
    return times, errors, commands


def draw_feedback_bridge() -> plt.Figure:
    traces = {
        "ON-OFF": simulate("relay"),
        "P": simulate("p"),
        "PD": simulate("pd"),
    }
    colors = {
        "ON-OFF": "#d62728",
        "P": "#1f77b4",
        "PD": "#2ca02c",
    }

    figure, axes = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True)
    error_axis, command_axis = axes

    for label, (times, errors, commands) in traces.items():
        error_axis.plot(times, errors, color=colors[label], linewidth=2.0, label=label)
        command_axis.plot(times, commands, color=colors[label], linewidth=1.8, label=label)

    for axis in axes:
        axis.axhline(0.0, color="#666666", linewidth=0.8)
        axis.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)
        axis.set_xlim(0.0, T_FINAL)
        axis.legend(loc="upper right", ncols=3, frameon=True)

    error_axis.set_ylabel(r"Lateral error $e_y$ [m]")
    error_axis.set_ylim(-0.075, 0.075)
    command_axis.set_xlabel(r"Time $t$ [s]")
    command_axis.set_ylabel(r"Wheel command $u_{\Delta}$ [m/s]")
    command_axis.set_ylim(-0.16, 0.16)

    return figure


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "ja" / "figures" / "line_following_feedback_bridge.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 180,
            "savefig.dpi": 180,
        }
    )

    figure = draw_feedback_bridge()
    finalize_figure(figure, output_path)
    plt.close(figure)
    print(output_path)


if __name__ == "__main__":
    main()
