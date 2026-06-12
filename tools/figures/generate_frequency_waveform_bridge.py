#!/usr/bin/env python3
"""Generate a waveform, spectrum, and bandwidth bridge for frequency response."""

from __future__ import annotations

import math
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("numpy is required to generate the waveform bridge figure") from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("matplotlib is required to generate the waveform bridge figure") from exc

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

PERIOD = 4.0
OMEGA_0 = 2.0 * math.pi / PERIOD
TAU_C = 0.35
OMEGA_C = 1.0 / TAU_C
HARMONICS = np.arange(1, 63, 2)
VISIBLE_HARMONICS = np.arange(1, 18, 2)

PANEL_BOX = {
    "boxstyle": "round,pad=0.25",
    "facecolor": "white",
    "alpha": 0.9,
    "edgecolor": "#bbbbbb",
}


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


def first_order_response(omega: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + 1j * omega * TAU_C)


def harmonic_amplitudes(harmonics: np.ndarray) -> np.ndarray:
    return 4.0 / (math.pi * harmonics)


def square_command(t: np.ndarray) -> np.ndarray:
    return np.where((t % PERIOD) < 0.5 * PERIOD, 1.0, -1.0)


def filtered_periodic_response(t: np.ndarray, harmonics: np.ndarray) -> np.ndarray:
    output = np.zeros_like(t)
    amplitudes = harmonic_amplitudes(harmonics)
    for harmonic, amplitude in zip(harmonics, amplitudes):
        omega = harmonic * OMEGA_0
        response = first_order_response(np.array([omega]))[0]
        output += amplitude * abs(response) * np.sin(omega * t + np.angle(response))
    return output


def db(values: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(values), 1.0e-12))


def add_panel_label(ax, label: str, text: str) -> None:
    ax.text(
        0.025,
        0.965,
        f"{label} {text}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        weight="bold",
        bbox=PANEL_BOX,
        zorder=8,
    )


def plot_time_bridge(ax) -> None:
    t = np.linspace(0.0, 2.0 * PERIOD, 2200)
    command = square_command(t)
    output = filtered_periodic_response(t, HARMONICS)

    ax.step(
        t,
        command,
        where="post",
        color="#222222",
        linewidth=1.7,
        label="非正弦波指令",
    )
    ax.plot(
        t,
        output,
        color="#1f77b4",
        linewidth=2.2,
        label=r"帯域制限後 $T_b(j\omega)$",
    )
    for switch_time in np.arange(0.0, 2.0 * PERIOD + 0.1, 0.5 * PERIOD):
        ax.axvline(switch_time, color="#dddddd", linestyle=":", linewidth=0.8, zorder=0)

    ax.annotate(
        "角の鋭さは高調波で作られる",
        xy=(2.02, 0.94),
        xytext=(2.78, 1.27),
        arrowprops={"arrowstyle": "->", "color": "#444444", "linewidth": 1.0},
        color="#333333",
        fontsize=8,
        ha="left",
    )
    ax.annotate(
        "高調波の減衰が丸い立上りになる",
        xy=(4.26, -0.28),
        xytext=(4.95, -0.92),
        arrowprops={"arrowstyle": "->", "color": "#1f77b4", "linewidth": 1.0},
        color="#174a7c",
        fontsize=8,
        ha="left",
    )
    ax.text(
        0.025,
        0.08,
        rf"$T_b(s)=1/(\tau_c s+1),\ \tau_c={TAU_C:.2f}\,\mathrm{{s}}$",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.2,
        bbox=PANEL_BOX,
    )
    ax.set_xlim(0.0, 2.0 * PERIOD)
    ax.set_ylim(-1.45, 1.45)
    ax.set_xlabel(r"時間 $t$ [s]")
    ax.set_ylabel("正規化振幅 [-]")
    ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.72)
    ax.legend(loc="upper right", framealpha=0.94)
    add_panel_label(ax, "(a)", "時間波形")


def plot_harmonic_spectrum(ax) -> None:
    input_amplitudes = harmonic_amplitudes(VISIBLE_HARMONICS)
    omega = VISIBLE_HARMONICS * OMEGA_0
    output_amplitudes = input_amplitudes * np.abs(first_order_response(omega))

    bar_width = 0.36
    ax.bar(
        VISIBLE_HARMONICS - bar_width / 2.0,
        input_amplitudes,
        width=bar_width,
        color="#4c78a8",
        alpha=0.85,
        label="入力成分",
    )
    ax.bar(
        VISIBLE_HARMONICS + bar_width / 2.0,
        output_amplitudes,
        width=bar_width,
        color="#f58518",
        alpha=0.86,
        label="通過後",
    )
    ax.plot(VISIBLE_HARMONICS, output_amplitudes, color="#9c4f03", linewidth=1.0)
    ax.text(
        0.42,
        0.94,
        r"$b_k=4/(\pi k)$" "\n" r"$|T_b(jk\omega_0)|b_k$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=PANEL_BOX,
    )
    ax.set_xlim(0.2, 17.8)
    ax.set_ylim(0.0, 1.62)
    ax.set_xticks(VISIBLE_HARMONICS)
    ax.set_xlabel("高調波次数 k [-]")
    ax.set_ylabel("振幅 [-]")
    ax.grid(True, axis="y", color="#d0d0d0", linewidth=0.7, alpha=0.72)
    ax.legend(
        loc="center right",
        bbox_to_anchor=(1.0, 0.58),
        fontsize=7.5,
        framealpha=0.94,
    )
    add_panel_label(ax, "(b)", "高調波振幅")


def plot_attenuation(ax) -> None:
    omega = np.logspace(-1.0, 2.0, 850)
    attenuation = db(first_order_response(omega))
    harmonic_omega = VISIBLE_HARMONICS * OMEGA_0
    harmonic_attenuation = db(first_order_response(harmonic_omega))

    ax.semilogx(omega, attenuation, color="#1f77b4", linewidth=2.2, label=r"$|T_b(j\omega)|$")
    ax.scatter(
        harmonic_omega,
        harmonic_attenuation,
        color="#f58518",
        edgecolor="#7a3b00",
        linewidth=0.6,
        s=24,
        zorder=5,
        label=r"$k\omega_0$",
    )
    ax.axvline(OMEGA_C, color="#d62728", linestyle="--", linewidth=1.1)
    ax.axhline(-3.0, color="#777777", linestyle=":", linewidth=1.0)
    ax.text(
        OMEGA_C * 0.92,
        -4.4,
        rf"$\omega_c=1/\tau_c={OMEGA_C:.2f}$ rad/s",
        color="#b22222",
        fontsize=8,
        ha="right",
        va="top",
    )
    ax.annotate(
        "次数が上がるほど\n同じ指令内で強く減衰",
        xy=(VISIBLE_HARMONICS[-2] * OMEGA_0, harmonic_attenuation[-2]),
        xytext=(12.0, -25.0),
        arrowprops={"arrowstyle": "->", "color": "#444444", "linewidth": 1.0},
        color="#333333",
        fontsize=8,
        ha="left",
    )
    ax.set_xlim(0.1, 100.0)
    ax.set_ylim(-36.0, 2.0)
    ax.set_xlabel(r"角周波数 $\omega$ [rad/s]")
    ax.set_ylabel("ゲイン [dB]")
    ax.grid(True, which="both", color="#d0d0d0", linewidth=0.7, alpha=0.72)
    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.94)
    add_panel_label(ax, "(c)", "Bode 型の減衰")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "ja" / "figures" / "frequency_waveform_bridge.png"
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

    fig = plt.figure(figsize=(7.2, 5.4), constrained_layout=False)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.18, 1.0], hspace=0.34, wspace=0.26)
    time_ax = fig.add_subplot(grid[0, :])
    spectrum_ax = fig.add_subplot(grid[1, 0])
    attenuation_ax = fig.add_subplot(grid[1, 1])

    plot_time_bridge(time_ax)
    plot_harmonic_spectrum(spectrum_ax)
    plot_attenuation(attenuation_ax)

    finalize_figure(fig, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
