#!/usr/bin/env python3
"""Generate a closed-loop implementation validation log trace."""

from __future__ import annotations

from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("numpy is required to generate the validation log trace figure") from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("matplotlib is required to generate the validation log trace figure") from exc

from figure_style import finalize_figure


SAMPLE_TIME = 0.5
SAMPLES = 80
AMBIENT_TEMPERATURE = 24.0
THERMAL_TIME_CONSTANT = 8.5
HEATER_STRENGTH = 39.0
TEMPERATURE_LIMIT = 55.0
INPUT_LIMIT = 0.92
RATE_LIMIT = 0.08
SOLVER_DEADLINE_MS = 320.0
KKT_LIMIT = 1.0e-2
MONITOR_RESIDUAL_LIMIT = 0.40


def reference_profile(time: np.ndarray) -> np.ndarray:
    """Build a smooth reference profile for a constrained heater loop."""

    first_step = 32.0 + 18.0 / (1.0 + np.exp(-(time - 4.5) / 0.9))
    second_step = 3.2 / (1.0 + np.exp(-(time - 20.0) / 1.1))
    return first_step + second_step


def solve_time_trace(sample_index: int) -> float:
    """Deterministic solve-time trace with one slowdown cluster."""

    base = 142.0 + 16.0 * np.sin(0.27 * sample_index) + 7.0 * np.cos(0.11 * sample_index)
    if sample_index == 48:
        return 432.0
    if sample_index == 49:
        return 348.0
    if sample_index == 52:
        return 276.0
    return float(base)


def kkt_residual_trace(sample_index: int) -> float:
    """Deterministic KKT residual trace with a failed and an infeasible solve."""

    base = 2.0e-5 * (1.0 + 0.6 * np.sin(0.31 * sample_index) ** 2)
    if sample_index == 48:
        return 3.2e-2
    if sample_index == 49:
        return 1.4e-2
    if sample_index == 52:
        return 1.9e-1
    if sample_index in (47, 50, 51, 53):
        return 7.0e-4
    return float(base)


def simulate_log() -> dict[str, np.ndarray]:
    """Simulate a deterministic validation log for a SIL/HIL-like run."""

    time = SAMPLE_TIME * np.arange(SAMPLES)
    reference = reference_profile(time)
    y = np.zeros(SAMPLES)
    y[0] = 31.0
    u_command = np.zeros(SAMPLES)
    u_actual = np.zeros(SAMPLES)
    solver_time = np.zeros(SAMPLES)
    kkt_residual = np.zeros(SAMPLES)
    monitor_residual = np.zeros(SAMPLES)
    fallback = np.zeros(SAMPLES, dtype=bool)
    solver_status = np.zeros(SAMPLES, dtype=int)
    predicted_margin = np.zeros(SAMPLES)
    realized_margin = np.zeros(SAMPLES)

    integral_error = 0.0
    previous_input = 0.0
    previous_model_y = y[0]

    for index in range(SAMPLES):
        error = reference[index] - y[index]
        integral_error = float(np.clip(integral_error + SAMPLE_TIME * error, -18.0, 18.0))
        feedforward = (reference[index] - AMBIENT_TEMPERATURE) / HEATER_STRENGTH
        u_nominal = feedforward + 0.020 * error + 0.0026 * integral_error
        u_nominal += 0.012 * np.sin(0.22 * index)
        u_command[index] = float(np.clip(u_nominal, 0.0, 1.05))

        solver_time[index] = solve_time_trace(index)
        kkt_residual[index] = kkt_residual_trace(index)
        timeout = solver_time[index] > SOLVER_DEADLINE_MS
        infeasible = index == 52
        poor_optimality = kkt_residual[index] > KKT_LIMIT
        fallback[index] = timeout or infeasible or poor_optimality
        if timeout:
            solver_status[index] = 1
        if infeasible:
            solver_status[index] = 2

        if fallback[index]:
            fallback_target = min(previous_input - 0.075, 0.56)
            fallback_target = max(0.28, fallback_target)
            requested_input = fallback_target
        else:
            requested_input = u_command[index]

        rate_limited = previous_input + float(np.clip(requested_input - previous_input, -RATE_LIMIT, RATE_LIMIT))
        u_actual[index] = float(np.clip(rate_limited, 0.0, INPUT_LIMIT))

        predicted_peak = y[index] + 1.85 * (u_command[index] - 0.55) + 0.28 * max(reference[index] - y[index], 0.0)
        if index in (48, 49, 52):
            predicted_peak -= 0.35
        predicted_margin[index] = TEMPERATURE_LIMIT - predicted_peak
        realized_margin[index] = TEMPERATURE_LIMIT - y[index]

        monitor_residual[index] = abs(y[index] - previous_model_y)
        if index in (48, 49, 52):
            monitor_residual[index] += 0.20

        if index + 1 < SAMPLES:
            disturbance = 0.55 if 44 <= index <= 55 else 0.0
            y[index + 1] = y[index] + SAMPLE_TIME / THERMAL_TIME_CONSTANT * (
                AMBIENT_TEMPERATURE + HEATER_STRENGTH * u_actual[index] + disturbance - y[index]
            )
            previous_model_y = y[index] + SAMPLE_TIME / THERMAL_TIME_CONSTANT * (
                AMBIENT_TEMPERATURE + HEATER_STRENGTH * previous_input - y[index]
            )
            previous_input = u_actual[index]

    return {
        "time": time,
        "reference": reference,
        "y": y,
        "u_command": u_command,
        "u_actual": u_actual,
        "solver_time": solver_time,
        "kkt_residual": kkt_residual,
        "monitor_residual": monitor_residual,
        "fallback": fallback,
        "solver_status": solver_status,
        "predicted_margin": predicted_margin,
        "realized_margin": realized_margin,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "ja" / "figures" / "validation_log_trace_examples.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 8.7,
            "axes.labelsize": 8.7,
            "axes.titlesize": 9.2,
            "legend.fontsize": 7.4,
            "font.family": ["Noto Sans CJK JP", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
            "figure.dpi": 180,
            "savefig.dpi": 180,
        }
    )

    log = simulate_log()
    time = log["time"]
    fallback_time = time[log["fallback"]]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4), constrained_layout=False)
    response_ax, input_ax, margin_ax, solver_ax = axes.ravel()

    response_ax.plot(time, log["reference"], color="#555555", linestyle="--", linewidth=1.5, label=r"参照 $r_k$")
    response_ax.plot(time, log["y"], color="#1f77b4", linewidth=1.9, label=r"測定温度 $y_k$")
    response_ax.axhline(TEMPERATURE_LIMIT, color="#9b2226", linestyle=":", linewidth=1.2, label="上限制約")
    response_ax.fill_between(time, TEMPERATURE_LIMIT, TEMPERATURE_LIMIT + 1.8, color="#f5c7c7", alpha=0.38)
    response_ax.text(0.03, 0.91, "A 参照と測定値", transform=response_ax.transAxes, fontsize=8.4)
    response_ax.set_xlabel(r"時刻 $kT_s$ [s]")
    response_ax.set_ylabel("温度 [degC]")
    response_ax.set_xlim(time[0], time[-1])
    response_ax.set_ylim(30.0, 56.8)
    response_ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)
    response_ax.legend(loc="lower right", framealpha=0.94)

    input_ax.plot(time, log["u_command"], color="#6a4c93", linestyle="--", linewidth=1.4, label=r"計算入力 $u_k^c$")
    input_ax.step(time, log["u_actual"], where="post", color="#d62728", linewidth=1.9, label=r"実入力 $u_k^{act}$")
    input_ax.axhline(INPUT_LIMIT, color="#777777", linestyle=":", linewidth=1.0, label="入力上限")
    if fallback_time.size:
        input_ax.plot(
            fallback_time,
            log["u_actual"][log["fallback"]],
            linestyle="none",
            marker="v",
            markersize=5.5,
            color="#9b2226",
            label="フォールバック採用",
        )
    input_ax.text(0.03, 0.91, "B 計算入力と適用入力", transform=input_ax.transAxes, fontsize=8.4)
    input_ax.set_xlabel(r"時刻 $kT_s$ [s]")
    input_ax.set_ylabel("ヒータ入力 [-]")
    input_ax.set_xlim(time[0], time[-1])
    input_ax.set_ylim(-0.03, 1.10)
    input_ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)
    input_ax.legend(loc="lower right", framealpha=0.94)

    margin_ax.plot(time, log["predicted_margin"], color="#2ca02c", linewidth=1.7, label=r"予測余裕 $d_{h,k}^{pred}$")
    margin_ax.plot(time, log["realized_margin"], color="#ff7f0e", linewidth=1.8, label=r"実余裕 $d_{h,k}^{real}$")
    margin_ax.axhline(0.0, color="#9b2226", linestyle=":", linewidth=1.1, label="制約境界")
    margin_ax.axhline(1.0, color="#8c8c8c", linestyle="--", linewidth=0.9, label="監視しきい余裕 1 K")
    margin_ax.fill_between(time, log["realized_margin"], 0.0, where=log["realized_margin"] < 1.0, color="#f5c7c7", alpha=0.36)
    margin_ax.text(0.03, 0.91, "C 制約余裕", transform=margin_ax.transAxes, fontsize=8.4)
    margin_ax.set_xlabel(r"時刻 $kT_s$ [s]")
    margin_ax.set_ylabel("温度上限までの余裕 [K]")
    margin_ax.set_xlim(time[0], time[-1])
    margin_ax.set_ylim(0.0, 25.5)
    margin_ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)
    margin_ax.legend(loc="upper right", framealpha=0.94)

    solver_ax.plot(time, log["solver_time"], color="#1f77b4", linewidth=1.7, label=r"解時間 $T_{\mathrm{solve},k}$")
    solver_ax.axhline(SOLVER_DEADLINE_MS, color="#9b2226", linestyle="--", linewidth=1.1, label="締切 320 ms")
    solver_ax.set_xlabel(r"時刻 $kT_s$ [s]")
    solver_ax.set_ylabel("解時間 [ms]")
    solver_ax.set_xlim(time[0], time[-1])
    solver_ax.set_ylim(70.0, 470.0)
    solver_ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)
    solver_ax.text(0.03, 0.91, "D ソルバと監視", transform=solver_ax.transAxes, fontsize=8.4)

    residual_ax = solver_ax.twinx()
    residual_ax.semilogy(time, log["kkt_residual"], color="#d95f02", linewidth=1.4, label="KKT 残差")
    residual_ax.semilogy(time, log["monitor_residual"], color="#2ca02c", linestyle=":", linewidth=1.4, label="監視残差")
    residual_ax.axhline(KKT_LIMIT, color="#d95f02", linestyle="--", linewidth=0.8, alpha=0.72)
    residual_ax.axhline(MONITOR_RESIDUAL_LIMIT, color="#2ca02c", linestyle="--", linewidth=0.8, alpha=0.72)
    residual_ax.set_ylabel("残差 [-]")
    residual_ax.set_ylim(1.0e-5, 1.0)

    for event_time in (24.0, 24.5, 26.0):
        solver_ax.axvline(event_time, color="#9b2226", linestyle=":", linewidth=0.9, alpha=0.72)
    solver_ax.annotate(
        "timeout",
        xy=(24.0, 432.0),
        xytext=(20.8, 446.0),
        arrowprops={"arrowstyle": "->", "color": "#7f1d1d", "linewidth": 0.9},
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#c7a0a0", "alpha": 0.90},
        color="#7f1d1d",
        fontsize=7.2,
    )
    solver_ax.annotate(
        "infeasible",
        xy=(26.0, 276.0),
        xytext=(27.4, 392.0),
        arrowprops={"arrowstyle": "->", "color": "#7f1d1d", "linewidth": 0.9},
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#c7a0a0", "alpha": 0.90},
        color="#7f1d1d",
        fontsize=7.2,
    )

    solver_handles, solver_labels = solver_ax.get_legend_handles_labels()
    residual_handles, residual_labels = residual_ax.get_legend_handles_labels()
    solver_ax.legend(
        solver_handles + residual_handles,
        solver_labels + residual_labels,
        loc="lower left",
        framealpha=0.94,
    )

    finalize_figure(fig, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
