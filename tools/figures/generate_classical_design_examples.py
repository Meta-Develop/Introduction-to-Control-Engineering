#!/usr/bin/env python3
"""Generate classical-control time-response examples."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("matplotlib is required to generate the classical-design figure") from exc


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

ERROR_GAIN = 4.0
ERROR_TIME_STOP = 10.0
ERROR_SAMPLE_COUNT = 1200

PID_TIME_STEP = 0.01
PID_TIME_STOP = 16.0
PID_PLANT_TAU = 1.5
PID_KP = 2.0
PID_KI = 1.5
PID_TRACKING_TIME = 0.6
PID_U_MIN = 0.0
PID_U_MAX = 1.0
PID_REFERENCE_HIGH = 1.4
PID_REFERENCE_LOW = 0.4
PID_REFERENCE_SWITCH = 6.0


def configure_fonts() -> None:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    selected_fonts = [
        font_name
        for font_name in JAPANESE_FONT_CANDIDATES
        if font_name in available_fonts
    ]
    if not selected_fonts:
        raise SystemExit(
            "A Japanese-capable Matplotlib font is required; install "
            "Noto Sans CJK JP, Harano Aji Gothic, or another listed font."
        )
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = selected_fonts + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def transfer_state_space(numerator: list[float], denominator: list[float]):
    denominator_array = np.asarray(denominator, dtype=float)
    numerator_array = np.asarray(numerator, dtype=float)
    denominator_array = denominator_array / denominator_array[0]
    numerator_array = numerator_array / denominator[0]

    order = len(denominator_array) - 1
    if len(numerator_array) > order:
        raise ValueError("Only strictly proper transfer functions are supported.")

    padded = np.zeros(order)
    padded[-len(numerator_array) :] = numerator_array
    numerator_ascending = padded[::-1]
    denominator_ascending = denominator_array[1:][::-1]

    state = np.zeros((order, order))
    if order > 1:
        state[:-1, 1:] = np.eye(order - 1)
    state[-1, :] = -denominator_ascending

    input_matrix = np.zeros(order)
    input_matrix[-1] = 1.0
    output_matrix = numerator_ascending
    return state, input_matrix, output_matrix


def simulate_transfer(
    numerator: list[float],
    denominator: list[float],
    time: np.ndarray,
    input_signal: np.ndarray,
) -> np.ndarray:
    state_matrix, input_vector, output_vector = transfer_state_space(
        numerator, denominator
    )
    state = np.zeros(state_matrix.shape[0])
    output = np.zeros_like(time)

    def dynamics(current_state: np.ndarray, current_input: float) -> np.ndarray:
        return state_matrix @ current_state + input_vector * current_input

    for index, current_time in enumerate(time):
        output[index] = output_vector @ state
        if index == len(time) - 1:
            break

        step = time[index + 1] - current_time
        u0 = input_signal[index]
        u1 = input_signal[index + 1]
        um = 0.5 * (u0 + u1)
        k1 = dynamics(state, u0)
        k2 = dynamics(state + 0.5 * step * k1, um)
        k3 = dynamics(state + 0.5 * step * k2, um)
        k4 = dynamics(state + step * k3, u1)
        state = state + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    return output


def system_type_errors() -> tuple[
    np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]
]:
    time = np.linspace(0.0, ERROR_TIME_STOP, ERROR_SAMPLE_COUNT)
    step = np.ones_like(time)
    ramp = time.copy()
    transfer_functions = {
        "0 型": ([ERROR_GAIN], [1.0, 1.0 + ERROR_GAIN]),
        "1 型": ([ERROR_GAIN], [1.0, 1.0, ERROR_GAIN]),
        "2 型": ([ERROR_GAIN, ERROR_GAIN], [1.0, 4.0, ERROR_GAIN, ERROR_GAIN]),
    }
    step_errors = {}
    ramp_errors = {}

    for label, (numerator, denominator) in transfer_functions.items():
        step_output = simulate_transfer(numerator, denominator, time, step)
        ramp_output = simulate_transfer(numerator, denominator, time, ramp)
        step_errors[label] = step - step_output
        ramp_errors[label] = ramp - ramp_output

    return time, step_errors, ramp_errors


def reference_signal(time: float) -> float:
    if time < PID_REFERENCE_SWITCH:
        return PID_REFERENCE_HIGH
    return PID_REFERENCE_LOW


def simulate_pi(use_back_calculation: bool) -> dict[str, np.ndarray]:
    time = np.arange(0.0, PID_TIME_STOP + PID_TIME_STEP, PID_TIME_STEP)
    reference = np.array([reference_signal(float(t)) for t in time])
    output = np.zeros_like(time)
    control = np.zeros_like(time)
    calculated = np.zeros_like(time)
    integral = np.zeros_like(time)

    y = 0.0
    eta = 0.0
    for index, current_time in enumerate(time):
        r = reference[index]
        error = r - y
        uc = PID_KP * error + eta
        u = float(np.clip(uc, PID_U_MIN, PID_U_MAX))

        output[index] = y
        control[index] = u
        calculated[index] = uc
        integral[index] = eta

        if index == len(time) - 1:
            break

        if use_back_calculation:
            eta += PID_TIME_STEP * (
                PID_KI * error + (u - uc) / PID_TRACKING_TIME
            )
        else:
            eta += PID_TIME_STEP * PID_KI * error
        y += PID_TIME_STEP * (-y + u) / PID_PLANT_TAU

    return {
        "time": time,
        "reference": reference,
        "output": output,
        "control": control,
        "calculated": calculated,
        "integral": integral,
    }


def plot_system_type_errors(step_ax, ramp_ax) -> None:
    time, step_errors, ramp_errors = system_type_errors()
    colors = {"0 型": "#1f77b4", "1 型": "#ff7f0e", "2 型": "#2ca02c"}
    linestyles = {"0 型": "-", "1 型": "--", "2 型": "-."}

    for label in ("0 型", "1 型", "2 型"):
        step_ax.plot(
            time,
            step_errors[label],
            color=colors[label],
            linestyle=linestyles[label],
            linewidth=1.9,
            label=label,
        )
        ramp_ax.plot(
            time,
            ramp_errors[label],
            color=colors[label],
            linestyle=linestyles[label],
            linewidth=1.9,
            label=label,
        )

    step_ax.axhline(
        1.0 / (1.0 + ERROR_GAIN),
        color="#777777",
        linestyle=":",
        linewidth=1.0,
    )
    step_ax.text(
        ERROR_TIME_STOP - 0.25,
        1.0 / (1.0 + ERROR_GAIN) + 0.015,
        r"$1/(1+K)=0.2$",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    ramp_ax.axhline(
        1.0 / ERROR_GAIN,
        color="#777777",
        linestyle=":",
        linewidth=1.0,
    )
    ramp_ax.text(
        ERROR_TIME_STOP - 0.25,
        1.0 / ERROR_GAIN + 0.04,
        r"$1/K=0.25$",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#444444",
    )

    step_ax.set_title(r"(a) ステップ入力の偏差, $K=4$")
    step_ax.set_xlabel(r"時間 $t$ [s]")
    step_ax.set_ylabel(r"偏差 $e(t)$ [-]")
    step_ax.set_xlim(0.0, ERROR_TIME_STOP)
    step_ax.set_ylim(-0.35, 1.08)
    step_ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)
    step_ax.legend(loc="upper right", fontsize=8)

    ramp_ax.set_title(r"(b) ランプ入力の偏差, $r(t)=t$")
    ramp_ax.set_xlabel(r"時間 $t$ [s]")
    ramp_ax.set_ylabel(r"偏差 $e(t)$ [-]")
    ramp_ax.set_xlim(0.0, ERROR_TIME_STOP)
    ramp_ax.set_ylim(-0.45, 2.55)
    ramp_ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)
    ramp_ax.legend(loc="upper left", fontsize=8)


def plot_pi_windup(response_ax, input_ax) -> None:
    without_aw = simulate_pi(False)
    with_aw = simulate_pi(True)
    time = without_aw["time"]

    response_ax.plot(
        time,
        without_aw["reference"],
        color="#444444",
        linestyle=":",
        linewidth=1.7,
        label="目標値",
    )
    response_ax.plot(
        time,
        without_aw["output"],
        color="#d62728",
        linewidth=1.9,
        label="アンチワインドアップなし",
    )
    response_ax.plot(
        time,
        with_aw["output"],
        color="#1f77b4",
        linewidth=1.9,
        label="バック計算あり",
    )
    response_ax.axvline(PID_REFERENCE_SWITCH, color="#777777", linestyle="--", linewidth=1.0)
    response_ax.set_title(r"(c) 飽和した PI 制御の出力")
    response_ax.set_xlabel(r"時間 $t$ [s]")
    response_ax.set_ylabel(r"出力 $y(t)$ [-]")
    response_ax.set_xlim(0.0, PID_TIME_STOP)
    response_ax.set_ylim(-0.08, 1.55)
    response_ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)
    response_ax.legend(loc="upper right", fontsize=7.4)

    input_ax.plot(
        time,
        without_aw["control"],
        color="#d62728",
        linewidth=1.8,
        label=r"$u$, なし",
    )
    input_ax.plot(
        time,
        with_aw["control"],
        color="#1f77b4",
        linewidth=1.8,
        label=r"$u$, バック計算",
    )
    input_ax.plot(
        time,
        without_aw["integral"],
        color="#d62728",
        linewidth=1.4,
        linestyle="--",
        label=r"$\eta$, なし",
    )
    input_ax.plot(
        time,
        with_aw["integral"],
        color="#1f77b4",
        linewidth=1.4,
        linestyle="--",
        label=r"$\eta$, バック計算",
    )
    input_ax.axhline(PID_U_MAX, color="#777777", linestyle=":", linewidth=1.0)
    input_ax.axhline(PID_U_MIN, color="#777777", linestyle=":", linewidth=1.0)
    input_ax.axvline(PID_REFERENCE_SWITCH, color="#777777", linestyle="--", linewidth=1.0)
    input_ax.set_title(r"(d) 飽和入力と積分状態")
    input_ax.set_xlabel(r"時間 $t$ [s]")
    input_ax.set_ylabel(r"入力・積分状態 [-]")
    input_ax.set_xlim(0.0, PID_TIME_STOP)
    input_ax.set_ylim(-0.55, 6.2)
    input_ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)
    input_ax.legend(loc="upper right", ncol=2, fontsize=7.0)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "ja" / "figures" / "classical_design_examples.png"
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

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.8), constrained_layout=False)
    fig.suptitle("古典制御における偏差・飽和・アンチワインドアップの時間波形")

    plot_system_type_errors(axes[0, 0], axes[0, 1])
    plot_pi_windup(axes[1, 0], axes[1, 1])

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    fig.savefig(output_path, facecolor="white")
    print(output_path)


if __name__ == "__main__":
    main()
