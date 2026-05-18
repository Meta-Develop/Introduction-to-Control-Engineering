#!/usr/bin/env python3
"""Generate visual examples of implementation effects in digital control."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("numpy is required to generate the implementation effects figure") from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("matplotlib is required to generate the implementation effects figure") from exc


TOTAL_TIME = 4.0
INTEGRATION_STEP = 0.001
PLANT_TIME_CONSTANT = 0.42
REFERENCE = 1.0
STATE_LIMIT = 1.15
PROPORTIONAL_K = 2.0
INTEGRAL_K = 1.25


@dataclass(frozen=True)
class ImplementationCase:
    name: str
    sample_time: float
    computation_delay: float
    quantization_step: float
    noise_sigma: float
    moving_average_length: int
    input_limit: float
    rate_limit_per_sample: float
    color: str
    seed: int = 7


def quantize(value: float, step: float) -> float:
    """Round value to the nearest quantization step."""

    if step <= 0.0:
        return value
    return step * round(value / step)


def simulate(case: ImplementationCase) -> dict[str, np.ndarray]:
    """Simulate a PI-controlled first-order plant with implementation effects."""

    rng = np.random.default_rng(case.seed)
    time = np.arange(0.0, TOTAL_TIME + INTEGRATION_STEP, INTEGRATION_STEP)
    delay_steps = int(round(case.computation_delay / INTEGRATION_STEP))
    delay_queue = [0.0] * (delay_steps + 1)
    measurement_buffer: list[float] = []

    state = 0.0
    integral_state = 0.0
    command = 0.0
    rate_limited = 0.0
    actual_input = 0.0
    filtered_measurement = 0.0
    next_sample_time = 0.0

    states = np.zeros_like(time)
    filtered = np.zeros_like(time)
    commands = np.zeros_like(time)
    rate_limited_inputs = np.zeros_like(time)
    actual_inputs = np.zeros_like(time)
    sample_times: list[float] = []
    measured_samples: list[float] = []
    filtered_samples: list[float] = []

    for index, current_time in enumerate(time):
        if current_time + 0.5 * INTEGRATION_STEP >= next_sample_time:
            measurement = state + case.noise_sigma * rng.normal()
            measurement = quantize(measurement, case.quantization_step)
            measurement_buffer.append(measurement)
            measurement_buffer = measurement_buffer[-case.moving_average_length :]
            filtered_measurement = float(np.mean(measurement_buffer))

            error = REFERENCE - filtered_measurement
            integral_state += case.sample_time * error
            command = REFERENCE + PROPORTIONAL_K * error + INTEGRAL_K * integral_state

            sample_times.append(current_time)
            measured_samples.append(measurement)
            filtered_samples.append(filtered_measurement)
            next_sample_time += case.sample_time

        delay_queue.append(command)
        delayed_command = delay_queue.pop(0)

        if isfinite(case.rate_limit_per_sample):
            rate_per_second = case.rate_limit_per_sample / case.sample_time
            max_delta = rate_per_second * INTEGRATION_STEP
            rate_limited += float(np.clip(delayed_command - rate_limited, -max_delta, max_delta))
        else:
            rate_limited = delayed_command

        actual_input = float(np.clip(rate_limited, -case.input_limit, case.input_limit))
        state += INTEGRATION_STEP * ((-state + actual_input) / PLANT_TIME_CONSTANT)

        states[index] = state
        filtered[index] = filtered_measurement
        commands[index] = command
        rate_limited_inputs[index] = rate_limited
        actual_inputs[index] = actual_input

    return {
        "time": time,
        "state": states,
        "filtered": filtered,
        "command": commands,
        "rate_limited": rate_limited_inputs,
        "actual_input": actual_inputs,
        "sample_time": np.asarray(sample_times),
        "measured_sample": np.asarray(measured_samples),
        "filtered_sample": np.asarray(filtered_samples),
        "margin": STATE_LIMIT - states,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "ja" / "figures" / "implementation_effects_examples.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 8.7,
            "axes.labelsize": 8.7,
            "axes.titlesize": 9.8,
            "legend.fontsize": 7.6,
            "font.family": ["Noto Sans CJK JP", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
            "figure.dpi": 180,
            "savefig.dpi": 180,
        }
    )

    nominal_case = ImplementationCase(
        name=r"理想実装: $T_s=0.04$ s, 遅れなし",
        sample_time=0.04,
        computation_delay=0.0,
        quantization_step=0.0,
        noise_sigma=0.0,
        moving_average_length=1,
        input_limit=float("inf"),
        rate_limit_per_sample=float("inf"),
        color="#1f77b4",
    )
    practical_case = ImplementationCase(
        name=r"実装条件: $T_s=0.08$ s, $T_c=0.06$ s, $M=5$",
        sample_time=0.08,
        computation_delay=0.06,
        quantization_step=0.025,
        noise_sigma=0.018,
        moving_average_length=5,
        input_limit=1.25,
        rate_limit_per_sample=0.18,
        color="#d62728",
    )

    nominal = simulate(nominal_case)
    practical = simulate(practical_case)
    time = nominal["time"]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=False)
    state_ax, measurement_ax, input_ax, margin_ax = axes.ravel()
    fig.suptitle("実装条件が閉ループ応答と制約余裕へ与える影響")

    state_ax.plot(time, nominal["state"], color=nominal_case.color, linewidth=1.8, label=nominal_case.name)
    state_ax.plot(time, practical["state"], color=practical_case.color, linewidth=1.8, label=practical_case.name)
    state_ax.axhline(REFERENCE, color="#555555", linestyle="--", linewidth=0.9, label=r"参照 $r=1.0$")
    state_ax.axhline(STATE_LIMIT, color="#9b2226", linestyle=":", linewidth=1.1, label=r"上限制約 $x_{\max}=1.15$")
    state_ax.fill_between(time, STATE_LIMIT, max(STATE_LIMIT, float(practical["state"].max())) + 0.02, color="#f5c7c7", alpha=0.35)
    state_ax.set_title("時間応答と上限制約")
    state_ax.set_xlabel("時刻 [s]")
    state_ax.set_ylabel(r"状態 $x(t)$ [-]")
    state_ax.set_xlim(0.0, TOTAL_TIME)
    state_ax.set_ylim(-0.02, 1.32)
    state_ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)
    state_ax.legend(loc="lower right", framealpha=0.94)

    measurement_ax.plot(time, nominal["state"], color=nominal_case.color, linewidth=1.2, label="理想: 真の状態")
    measurement_ax.step(
        time,
        nominal["filtered"],
        where="post",
        color=nominal_case.color,
        linestyle="--",
        linewidth=1.0,
        alpha=0.86,
        label="理想: 測定値",
    )
    measurement_ax.plot(time, practical["state"], color="#303030", linewidth=1.2, label="実装: 真の状態")
    measurement_ax.plot(
        practical["sample_time"],
        practical["measured_sample"],
        linestyle="none",
        marker="o",
        markersize=2.6,
        color="#ff7f0e",
        alpha=0.72,
        label=r"量子化測定 $\Delta y=0.025$",
    )
    measurement_ax.step(time, practical["filtered"], where="post", color="#2ca02c", linewidth=1.5, label=r"移動平均 $M=5$")
    measurement_ax.set_title("量子化、ノイズ、フィルタ遅れ")
    measurement_ax.set_xlabel("時刻 [s]")
    measurement_ax.set_ylabel(r"測定値 $y$ [-]")
    measurement_ax.set_xlim(0.0, TOTAL_TIME)
    measurement_ax.set_ylim(-0.05, 1.32)
    measurement_ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)
    measurement_ax.legend(loc="lower right", framealpha=0.94)

    input_ax.plot(time, nominal["actual_input"], color=nominal_case.color, linewidth=1.6, label=r"理想: $u_c=u^{act}$")
    input_ax.plot(time, practical["command"], color="#6a4c93", linestyle="--", linewidth=1.2, label=r"実装: 計算入力 $u_c$")
    input_ax.plot(time, practical["rate_limited"], color="#ff7f0e", linewidth=1.2, label=r"レート制限後")
    input_ax.plot(time, practical["actual_input"], color="#d62728", linewidth=1.8, label=r"実装: 実入力 $u^{act}$")
    input_ax.axhline(practical_case.input_limit, color="#8c8c8c", linestyle=":", linewidth=1.0)
    input_ax.axhline(-practical_case.input_limit, color="#8c8c8c", linestyle=":", linewidth=1.0)
    input_ax.text(
        0.03,
        0.08,
        r"$u\in[-1.25,1.25]$,  $\Delta u_{\max}=0.18$/周期",
        transform=input_ax.transAxes,
        color="#555555",
        fontsize=7.8,
    )
    input_ax.set_title("飽和とレート制限")
    input_ax.set_xlabel("時刻 [s]")
    input_ax.set_ylabel(r"入力 $u$ [-]")
    input_ax.set_xlim(0.0, TOTAL_TIME)
    input_ax.set_ylim(-0.15, 3.55)
    input_ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)
    input_ax.legend(loc="upper right", framealpha=0.94)

    margin_ax.plot(time, nominal["margin"], color=nominal_case.color, linewidth=1.8, label="理想実装")
    margin_ax.plot(time, practical["margin"], color=practical_case.color, linewidth=1.8, label="実装条件")
    margin_ax.axhline(0.0, color="#9b2226", linestyle=":", linewidth=1.1, label="余裕 0")
    margin_ax.fill_between(time, practical["margin"], 0.0, where=practical["margin"] < 0.0, color="#f5c7c7", alpha=0.55)
    practical_min_margin = float(practical["margin"].min())
    nominal_min_margin = float(nominal["margin"].min())
    margin_ax.text(
        0.03,
        0.08,
        "\n".join(
            [
                rf"最小余裕: 理想 {nominal_min_margin:.3f}",
                rf"最小余裕: 実装 {practical_min_margin:.3f}",
                r"負値は制約違反",
            ]
        ),
        transform=margin_ax.transAxes,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.92},
        fontsize=7.8,
    )
    margin_ax.set_title("状態制約余裕")
    margin_ax.set_xlabel("時刻 [s]")
    margin_ax.set_ylabel(r"$x_{\max}-x(t)$ [-]")
    margin_ax.set_xlim(0.0, TOTAL_TIME)
    margin_ax.set_ylim(-0.14, 1.20)
    margin_ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)
    margin_ax.legend(loc="upper right", framealpha=0.94)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(output_path, facecolor="white")
    print(output_path)


if __name__ == "__main__":
    main()
