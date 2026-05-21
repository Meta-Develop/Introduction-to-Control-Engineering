#!/usr/bin/env python3
"""Generate the Bode plot used by the normalized factor example."""

from __future__ import annotations

import math
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised only on minimal systems.
    np = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("matplotlib is required to generate the Bode figure") from exc

from figure_style import finalize_figure


K = 5.0
OMEGA_0 = 1.0
ZERO_W = 2.0
POLE_W = 0.2
ZETA = 0.5
OMEGA_N = 20.0
BREAKS = (
    (POLE_W, r"実極 $\omega_p=0.2$"),
    (ZERO_W, r"実零点 $\omega_z=2$"),
    (OMEGA_N, r"二次極 $\omega_n=20$"),
)
PHASE_WINDOWS = (
    (0.1 * POLE_W, 10.0 * POLE_W, r"$\omega_p:-90^\circ$"),
    (0.1 * ZERO_W, 10.0 * ZERO_W, r"$\omega_z:+90^\circ$"),
    (0.1 * OMEGA_N, 10.0 * OMEGA_N, r"$\omega_n:-180^\circ$"),
)


def make_logspace(start_exp: float, stop_exp: float, count: int):
    if np is not None:
        return np.logspace(start_exp, stop_exp, count)

    step = (stop_exp - start_exp) / (count - 1)
    return [10.0 ** (start_exp + step * i) for i in range(count)]


def loop_response(omega):
    if np is not None:
        s = 1j * omega
        return K * (1.0 + s / ZERO_W) / (
            (s / OMEGA_0)
            * (1.0 + s / POLE_W)
            * (1.0 + 2.0 * ZETA * s / OMEGA_N + (s / OMEGA_N) ** 2)
        )

    values = []
    for w in omega:
        s = 1j * w
        denominator = (
            (s / OMEGA_0)
            * (1.0 + s / POLE_W)
            * (1.0 + 2.0 * ZETA * s / OMEGA_N + (s / OMEGA_N) ** 2)
        )
        values.append(K * (1.0 + s / ZERO_W) / denominator)
    return values


def magnitude_db(response):
    if np is not None:
        return 20.0 * np.log10(np.abs(response))

    return [20.0 * math.log10(abs(value)) for value in response]


def unwrapped_phase_deg(response):
    if np is not None:
        return np.unwrap(np.angle(response)) * 180.0 / math.pi

    raw = [math.atan2(value.imag, value.real) for value in response]
    unwrapped = [raw[0]]
    offset = 0.0
    previous = raw[0]
    for angle in raw[1:]:
        delta = angle - previous
        if delta > math.pi:
            offset -= 2.0 * math.pi
        elif delta < -math.pi:
            offset += 2.0 * math.pi
        unwrapped.append(angle + offset)
        previous = angle
    return [angle * 180.0 / math.pi for angle in unwrapped]


def asymptotic_gain_db(omega):
    if np is not None:
        return (
            20.0 * math.log10(K)
            - 20.0 * np.log10(omega / OMEGA_0)
            - 20.0 * np.maximum(0.0, np.log10(omega / POLE_W))
            + 20.0 * np.maximum(0.0, np.log10(omega / ZERO_W))
            - 40.0 * np.maximum(0.0, np.log10(omega / OMEGA_N))
        )

    values = []
    for w in omega:
        values.append(
            20.0 * math.log10(K)
            - 20.0 * math.log10(w / OMEGA_0)
            - 20.0 * max(0.0, math.log10(w / POLE_W))
            + 20.0 * max(0.0, math.log10(w / ZERO_W))
            - 40.0 * max(0.0, math.log10(w / OMEGA_N))
        )
    return values


def phase_transition(omega, break_frequency: float, total_degrees: float):
    low = 0.1 * break_frequency
    high = 10.0 * break_frequency
    log_low = math.log10(low)
    log_high = math.log10(high)

    values = []
    for w in omega:
        if w <= low:
            values.append(0.0)
        elif w >= high:
            values.append(total_degrees)
        else:
            ratio = (math.log10(float(w)) - log_low) / (log_high - log_low)
            values.append(total_degrees * ratio)

    if np is not None:
        return np.array(values)
    return values


def asymptotic_phase_deg(omega):
    pole = phase_transition(omega, POLE_W, -90.0)
    zero = phase_transition(omega, ZERO_W, 90.0)
    second_order_pole = phase_transition(omega, OMEGA_N, -180.0)

    if np is not None:
        return -90.0 + pole + zero + second_order_pole
    return [
        -90.0 + pole_i + zero_i + second_i
        for pole_i, zero_i, second_i in zip(pole, zero, second_order_pole)
    ]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "ja" / "figures" / "bode_factor_example.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    omega = make_logspace(-2.0, 3.0, 1200)
    response = loop_response(omega)
    gain = magnitude_db(response)
    phase = unwrapped_phase_deg(response)
    gain_asymptote = asymptotic_gain_db(omega)
    phase_asymptote = asymptotic_phase_deg(omega)
    exact_color = "#1f77b4"
    asymptote_color = "#d62728"
    break_color = "#4b4b4b"
    transition_color = "#5f6368"

    plt.rcParams.update(
        {
            "font.family": ["Noto Sans CJK JP", "DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "figure.dpi": 180,
            "savefig.dpi": 180,
            "axes.unicode_minus": False,
        }
    )

    fig, (gain_ax, phase_ax) = plt.subplots(
        2, 1, figsize=(7.2, 5.6), sharex=True, constrained_layout=False
    )

    gain_ax.semilogx(omega, gain, color=exact_color, linewidth=2.2, label="厳密")
    gain_ax.semilogx(
        omega,
        gain_asymptote,
        color=asymptote_color,
        linestyle="--",
        linewidth=1.8,
        label="漸近",
    )
    gain_ax.set_ylabel("ゲイン [dB]")
    gain_ax.set_ylim(-145.0, 65.0)
    gain_ax.set_axisbelow(True)
    gain_ax.grid(True, which="both", color="#d0d0d0", linewidth=0.7, alpha=0.7)
    gain_ax.legend(loc="upper right")
    gain_ax.text(
        0.015,
        0.06,
        (
            r"$L(s)=5\dfrac{1+s/\omega_z}{(s/\omega_0)(1+s/\omega_p)F_2(s)}$"
            "\n"
            r"$F_2(s)=1+2\zeta s/\omega_n+(s/\omega_n)^2$"
            "\n"
            r"$\omega_0=1,\ K=5,\ \omega_p=0.2,\ \omega_z=2,\ "
            r"\omega_n=20\ \mathrm{rad/s},\ \zeta=0.5$"
        ),
        transform=gain_ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=8.0,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "alpha": 0.85,
            "edgecolor": "#bbbbbb",
        },
    )

    phase_ax.semilogx(omega, phase, color=exact_color, linewidth=2.2, label="厳密")
    phase_ax.semilogx(
        omega,
        phase_asymptote,
        color=asymptote_color,
        linestyle="--",
        linewidth=1.8,
        label="漸近",
    )
    phase_ax.set_ylabel("位相 [deg]")
    phase_ax.set_xlabel(r"角周波数 $\omega$ [rad/s]")
    phase_ax.set_ylim(-292.0, -82.0)
    phase_ax.set_yticks([-90, -120, -150, -180, -210, -240, -270])
    phase_ax.set_axisbelow(True)
    phase_ax.grid(True, which="both", color="#d0d0d0", linewidth=0.7, alpha=0.7)
    phase_ax.legend(loc="upper right")

    for ax in (gain_ax, phase_ax):
        for frequency, label in BREAKS:
            ax.axvline(
                frequency,
                color=break_color,
                linestyle=":",
                linewidth=1.1,
                alpha=0.8,
                zorder=0,
            )

    for frequency, label in BREAKS:
        gain_ax.text(
            frequency,
            1.015,
            label,
            transform=gain_ax.get_xaxis_transform(),
            va="bottom",
            ha="center",
            color=break_color,
            fontsize=8,
            clip_on=False,
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "alpha": 0.9,
                "edgecolor": "none",
            },
        )

    for row, (start, stop, label) in enumerate(PHASE_WINDOWS):
        y = -286.0 + 6.0 * row
        phase_ax.hlines(
            y,
            start,
            stop,
            color=transition_color,
            linewidth=1.8,
            alpha=0.9,
            zorder=3,
        )
        phase_ax.vlines(
            (start, stop),
            y - 1.7,
            y + 1.7,
            color=transition_color,
            linewidth=1.2,
            alpha=0.9,
            zorder=3,
        )
        phase_ax.text(
            math.sqrt(start * stop),
            y + 2.0,
            label,
            va="bottom",
            ha="center",
            color=transition_color,
            fontsize=8,
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "alpha": 0.86,
                "edgecolor": "none",
            },
        )

    finalize_figure(fig, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
