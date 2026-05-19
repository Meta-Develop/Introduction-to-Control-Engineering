#!/usr/bin/env python3
"""Generate mass-spring-damper classical design comparison plots."""

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
    raise SystemExit("matplotlib is required to generate the mass-spring figure") from exc

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

MASS = 1.0
DAMPING = 1.0
SPRING = 1.0
REFERENCE = 1.0
TIME_STEP = 0.002
TIME_STOP = 14.0


@dataclass(frozen=True)
class Candidate:
    label: str
    kp: float
    kd: float
    ki: float
    color: str
    linestyle: str


CANDIDATES = (
    Candidate("P 制御", kp=5.0, kd=0.0, ki=0.0, color="#1f77b4", linestyle="-"),
    Candidate("PD 制御", kp=5.0, kd=3.0, ki=0.0, color="#ff7f0e", linestyle="--"),
    Candidate("PID 制御", kp=5.0, kd=3.0, ki=2.0, color="#2ca02c", linestyle="-."),
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


def control_force(candidate: Candidate, state: np.ndarray) -> float:
    position, velocity, integral_state = state
    error = REFERENCE - position
    return candidate.kp * error + integral_state - candidate.kd * velocity


def dynamics(candidate: Candidate, state: np.ndarray) -> np.ndarray:
    position, velocity, _integral_state = state
    error = REFERENCE - position
    force = control_force(candidate, state)
    acceleration = (force - DAMPING * velocity - SPRING * position) / MASS
    integral_derivative = candidate.ki * error
    return np.array([velocity, acceleration, integral_derivative])


def simulate(candidate: Candidate) -> dict[str, np.ndarray]:
    time = np.arange(0.0, TIME_STOP + TIME_STEP, TIME_STEP)
    state = np.zeros(3)
    states = np.zeros((len(time), 3))
    forces = np.zeros_like(time)

    for index, current_time in enumerate(time):
        states[index] = state
        forces[index] = control_force(candidate, state)
        if index == len(time) - 1:
            break

        step = time[index + 1] - current_time
        k1 = dynamics(candidate, state)
        k2 = dynamics(candidate, state + 0.5 * step * k1)
        k3 = dynamics(candidate, state + 0.5 * step * k2)
        k4 = dynamics(candidate, state + step * k3)
        state = state + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    return {
        "time": time,
        "position": states[:, 0],
        "velocity": states[:, 1],
        "integral": states[:, 2],
        "force": forces,
    }


def gain_label(candidate: Candidate) -> str:
    terms = [rf"K_P={candidate.kp:g}"]
    if candidate.kd > 0.0:
        terms.append(rf"K_D={candidate.kd:g}")
    if candidate.ki > 0.0:
        terms.append(rf"K_I={candidate.ki:g}")
    return candidate.label + r" ($" + r",\ ".join(terms) + r"$)"


def rms_force(time: np.ndarray, force: np.ndarray) -> float:
    force_square_area = np.sum(
        0.5 * (force[1:] * force[1:] + force[:-1] * force[:-1]) * np.diff(time)
    )
    return float(np.sqrt(force_square_area / (time[-1] - time[0])))


def plot_response(response_ax, force_ax, tradeoff_ax) -> None:
    simulations = {candidate: simulate(candidate) for candidate in CANDIDATES}

    response_ax.axhline(
        REFERENCE,
        color="#444444",
        linestyle=":",
        linewidth=1.4,
        label=r"目標値 $r=1.0\,\mathrm{m}$",
    )

    for candidate, result in simulations.items():
        time = result["time"]
        response_ax.plot(
            time,
            result["position"],
            color=candidate.color,
            linestyle=candidate.linestyle,
            linewidth=2.0,
            label=gain_label(candidate),
        )
        force_ax.plot(
            time,
            result["force"],
            color=candidate.color,
            linestyle=candidate.linestyle,
            linewidth=1.8,
            label=candidate.label,
        )

    response_ax.set_title(r"(a) 変位応答")
    response_ax.set_xlabel(r"時間 $t$ [s]")
    response_ax.set_ylabel(r"変位 $q(t)$ [m]")
    response_ax.set_xlim(0.0, TIME_STOP)
    response_ax.set_ylim(-0.12, 1.38)
    response_ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)
    response_ax.legend(loc="lower right", fontsize=7.6)

    force_ax.axhline(0.0, color="#777777", linestyle=":", linewidth=1.0)
    force_ax.set_title(r"(b) 操作力")
    force_ax.set_xlabel(r"時間 $t$ [s]")
    force_ax.set_ylabel(r"力 $u(t)$ [N]")
    force_ax.set_xlim(0.0, TIME_STOP)
    force_ax.set_ylim(-2.6, 5.4)
    force_ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)
    force_ax.legend(loc="upper right", fontsize=7.4)

    for candidate, result in simulations.items():
        time = result["time"]
        terminal_error = abs(REFERENCE - result["position"][-1])
        force_rms = rms_force(time, result["force"])
        tradeoff_ax.scatter(
            terminal_error,
            force_rms,
            s=70,
            color=candidate.color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        tradeoff_ax.annotate(
            candidate.label,
            (terminal_error, force_rms),
            textcoords="offset points",
            xytext=(7, 6),
            ha="left",
            va="bottom",
            fontsize=8,
            color="#222222",
        )

    tradeoff_ax.set_title(r"(c) 終端偏差と操作力")
    tradeoff_ax.set_xlabel(r"終端偏差 $|r-q(T)|$ [m]")
    tradeoff_ax.set_ylabel(r"RMS 操作力 [N]")
    tradeoff_ax.set_xlim(-0.01, 0.20)
    tradeoff_ax.set_ylim(0.7, 1.7)
    tradeoff_ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)
    tradeoff_ax.text(
        0.98,
        0.05,
        r"$m=1,\ c=1,\ k=1$",
        transform=tradeoff_ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#444444",
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "ja" / "figures" / "mass_spring_design_examples.png"
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

    fig = plt.figure(figsize=(7.2, 6.4), constrained_layout=False)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.12, 1.0))
    response_ax = fig.add_subplot(grid[0, :])
    force_ax = fig.add_subplot(grid[1, 0])
    tradeoff_ax = fig.add_subplot(grid[1, 1])

    plot_response(response_ax, force_ax, tradeoff_ax)

    finalize_figure(fig, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
