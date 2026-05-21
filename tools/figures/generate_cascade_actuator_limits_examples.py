#!/usr/bin/env python3
"""Generate cascade-control and actuator-limit examples for a DC motor servo."""

from __future__ import annotations

from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("numpy is required to generate cascade actuator-limit figures") from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("matplotlib is required to generate cascade actuator-limit figures") from exc

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

R = 1.0
L = 0.050
K_T = 0.08
K_E = 0.08
J = 0.003
B = 0.002
V_MAX = 12.0
I_MAX = 8.0
OMEGA_CMD_MAX = 42.0
TAU_MAX = K_T * I_MAX


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


def smooth_position_reference(t: float) -> tuple[float, float]:
    start = 0.06
    duration = 0.55
    final_position = 18.0
    s = np.clip((t - start) / duration, 0.0, 1.0)
    position = final_position * (3.0 * s**2 - 2.0 * s**3)
    if 0.0 < s < 1.0:
        velocity = final_position * (6.0 * s - 6.0 * s**2) / duration
    else:
        velocity = 0.0
    return position, velocity


def saturation_would_worsen(command: float, limited: float, error: float) -> bool:
    return (command > limited and error > 0.0) or (command < limited and error < 0.0)


def simulate_cascade() -> dict[str, np.ndarray]:
    dt = 5.0e-4
    stop = 1.55
    time = np.arange(0.0, stop + 0.5 * dt, dt)

    theta = 0.0
    omega = 0.0
    current = 0.0
    velocity_integral = 0.0
    current_integral = 0.0

    k_theta = 12.0
    k_omega_p = 0.90
    k_omega_i = 8.0
    k_current_p = 5.0
    k_current_i = 300.0

    theta_ref = np.zeros_like(time)
    omega_ref = np.zeros_like(time)
    omega_cmd = np.zeros_like(time)
    omega_actual = np.zeros_like(time)
    current_cmd = np.zeros_like(time)
    current_ref = np.zeros_like(time)
    current_actual = np.zeros_like(time)
    voltage_cmd = np.zeros_like(time)
    voltage_actual = np.zeros_like(time)
    torque_cmd = np.zeros_like(time)
    torque_actual = np.zeros_like(time)
    current_margin_cmd = np.zeros_like(time)
    voltage_margin_cmd = np.zeros_like(time)

    for k, t in enumerate(time):
        ref_position, ref_velocity = smooth_position_reference(float(t))
        theta_ref[k] = ref_position
        omega_ref[k] = ref_velocity

        position_error = ref_position - theta
        omega_unlimited = ref_velocity + k_theta * position_error
        omega_limited = float(np.clip(omega_unlimited, -OMEGA_CMD_MAX, OMEGA_CMD_MAX))

        velocity_error = omega_limited - omega
        velocity_integral_trial = velocity_integral + velocity_error * dt
        current_unlimited_trial = (
            k_omega_p * velocity_error
            + k_omega_i * velocity_integral_trial
            + B * omega_limited / K_T
        )
        current_limited_trial = float(
            np.clip(current_unlimited_trial, -I_MAX, I_MAX)
        )
        if not saturation_would_worsen(
            current_unlimited_trial, current_limited_trial, velocity_error
        ):
            velocity_integral = velocity_integral_trial

        current_unlimited = (
            k_omega_p * velocity_error
            + k_omega_i * velocity_integral
            + B * omega_limited / K_T
        )
        current_limited = float(np.clip(current_unlimited, -I_MAX, I_MAX))

        current_error = current_limited - current
        current_integral_trial = current_integral + current_error * dt
        voltage_feedforward = R * current_limited + K_E * omega
        voltage_unlimited_trial = (
            voltage_feedforward
            + k_current_p * current_error
            + k_current_i * current_integral_trial
        )
        voltage_limited_trial = float(
            np.clip(voltage_unlimited_trial, -V_MAX, V_MAX)
        )
        if not saturation_would_worsen(
            voltage_unlimited_trial, voltage_limited_trial, current_error
        ):
            current_integral = current_integral_trial

        voltage_unlimited = (
            voltage_feedforward
            + k_current_p * current_error
            + k_current_i * current_integral
        )
        voltage_limited = float(np.clip(voltage_unlimited, -V_MAX, V_MAX))

        load_torque = 0.035 if t >= 0.82 else 0.0
        current_dot = (voltage_limited - R * current - K_E * omega) / L
        omega_dot = (K_T * current - B * omega - load_torque) / J

        theta += omega * dt
        omega += omega_dot * dt
        current += current_dot * dt

        omega_cmd[k] = omega_limited
        omega_actual[k] = omega
        current_cmd[k] = current_unlimited
        current_ref[k] = current_limited
        current_actual[k] = current
        voltage_cmd[k] = voltage_unlimited
        voltage_actual[k] = voltage_limited
        torque_cmd[k] = K_T * current_unlimited
        torque_actual[k] = K_T * current
        current_margin_cmd[k] = 1.0 - abs(current_unlimited) / I_MAX
        voltage_margin_cmd[k] = 1.0 - abs(voltage_unlimited) / V_MAX

    return {
        "time": time,
        "theta_ref": theta_ref,
        "omega_ref": omega_ref,
        "omega_cmd": omega_cmd,
        "omega_actual": omega_actual,
        "current_cmd": current_cmd,
        "current_ref": current_ref,
        "current_actual": current_actual,
        "voltage_cmd": voltage_cmd,
        "voltage_actual": voltage_actual,
        "torque_cmd": torque_cmd,
        "torque_actual": torque_actual,
        "current_margin_cmd": current_margin_cmd,
        "voltage_margin_cmd": voltage_margin_cmd,
    }


def draw_arrow(axis, start: tuple[float, float], end: tuple[float, float], **kwargs) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.25,
        color=kwargs.pop("color", "#303030"),
        connectionstyle=kwargs.pop("connectionstyle", "arc3,rad=0.0"),
        **kwargs,
    )
    axis.add_patch(arrow)


def draw_box(
    axis,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.2,
        edgecolor="#303030",
        facecolor=facecolor,
    )
    axis.add_patch(patch)
    axis.text(
        xy[0] + width / 2.0,
        xy[1] + height / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=8.6,
    )


def draw_block_diagram(axis) -> None:
    axis.set_axis_off()
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)

    boxes = [
        ((0.04, 0.54), 0.16, 0.24, "位置ループ $C_\\theta$\n$e_\\theta$ [rad]\n$\\omega_r$ [rad/s]", "#e8f3ff"),
        ((0.25, 0.54), 0.16, 0.24, "速度ループ $C_\\omega$\n$e_\\omega$ [rad/s]\n$i_r^c$ [A]", "#eef8e8"),
        ((0.46, 0.54), 0.13, 0.24, "電流制限\n$|i_r|\\leq8$ A\n$|\\tau|\\leq0.64$ N m", "#fff3df"),
        ((0.64, 0.54), 0.14, 0.24, "電流ループ $C_i$\n$e_i$ [A]\n$v_c$ [V]", "#f3eaff"),
        ((0.83, 0.54), 0.13, 0.24, "電圧制限\nDC モータ\n$v\\to i\\to\\tau\\to\\omega\\to\\theta$", "#ffecec"),
    ]
    for xy, width, height, text, facecolor in boxes:
        draw_box(axis, xy, width, height, text, facecolor=facecolor)

    y_mid = 0.66
    x_edges = [0.20, 0.25, 0.41, 0.46, 0.59, 0.64, 0.78, 0.83]
    for start_x, end_x in zip(x_edges[0::2], x_edges[1::2]):
        draw_arrow(axis, (start_x, y_mid), (end_x, y_mid))
    draw_arrow(axis, (0.96, y_mid), (0.99, y_mid))
    axis.text(0.005, y_mid, "$\\theta_r$", ha="left", va="center", fontsize=9)

    draw_arrow(axis, (0.90, 0.50), (0.72, 0.47), color="#707070", connectionstyle="arc3,rad=-0.08")
    draw_arrow(axis, (0.90, 0.46), (0.34, 0.38), color="#707070", connectionstyle="arc3,rad=-0.08")
    draw_arrow(axis, (0.90, 0.42), (0.13, 0.30), color="#707070", connectionstyle="arc3,rad=-0.08")
    axis.text(0.73, 0.43, "電流帰還 $i$ [A]", ha="center", va="top", fontsize=7.7, color="#505050")
    axis.text(0.36, 0.35, "速度帰還 $\\omega$ [rad/s]", ha="center", va="top", fontsize=7.7, color="#505050")
    axis.text(0.16, 0.27, "位置帰還 $\\theta$ [rad]", ha="center", va="top", fontsize=7.7, color="#505050")

    axis.plot([0.08, 0.90], [0.12, 0.12], color="#777777", linewidth=1.0)
    axis.scatter([0.17, 0.50, 0.83], [0.12, 0.12, 0.12], s=[55, 75, 95], color=["#1f77b4", "#2ca02c", "#d62728"])
    axis.text(0.17, 0.06, "$\\omega_{b,\\theta}=2$", ha="center", va="top", fontsize=8)
    axis.text(0.50, 0.06, "$\\omega_{b,\\omega}=12$", ha="center", va="top", fontsize=8)
    axis.text(0.83, 0.06, "$\\omega_{b,i}=80$ rad/s", ha="center", va="top", fontsize=8)
    axis.text(
        0.50,
        0.18,
        "帯域は外側から内側へ速くする: 位置 < 速度 < 電流",
        ha="center",
        va="bottom",
        fontsize=9,
    )


def draw_speed_trace(axis, data: dict[str, np.ndarray]) -> None:
    time = data["time"]
    axis.plot(time, data["omega_cmd"], color="#1f77b4", linewidth=2.0, label=r"指令 $\omega_r$")
    axis.plot(time, data["omega_actual"], color="#d62728", linewidth=1.9, label=r"実速度 $\omega$")
    axis.axhline(OMEGA_CMD_MAX, color="#777777", linestyle=":", linewidth=1.2)
    axis.axhline(-OMEGA_CMD_MAX, color="#777777", linestyle=":", linewidth=1.2)
    axis.text(0.03, 0.93, "速度指令と到達速度", transform=axis.transAxes, ha="left", va="top", fontsize=9)
    axis.set_xlabel(r"時間 $t$ [s]")
    axis.set_ylabel(r"角速度 $\omega$ [rad/s]")
    axis.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)
    axis.set_xlim(time[0], time[-1])
    axis.set_ylim(-22.0, 64.0)
    axis.legend(loc="lower right", fontsize=8)


def draw_current_torque_trace(axis, data: dict[str, np.ndarray]) -> None:
    time = data["time"]
    axis.plot(time, data["current_cmd"], color="#9467bd", linewidth=1.8, label=r"計算 $i_r^c$")
    axis.plot(time, data["current_ref"], color="#1f77b4", linewidth=2.0, label=r"制限後 $i_r$")
    axis.plot(time, data["current_actual"], color="#d62728", linewidth=1.9, label=r"実電流 $i$")
    axis.axhline(I_MAX, color="#777777", linestyle=":", linewidth=1.2)
    axis.axhline(-I_MAX, color="#777777", linestyle=":", linewidth=1.2)
    axis.text(0.03, 0.93, "電流制限とトルク制限", transform=axis.transAxes, ha="left", va="top", fontsize=9)
    axis.text(
        0.03,
        0.08,
        rf"$\tau=K_t i$, $\tau_{{max}}={TAU_MAX:.2f}$ N m",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.3,
    )
    axis.set_xlabel(r"時間 $t$ [s]")
    axis.set_ylabel(r"電流 $i$ [A]")
    axis.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)
    axis.set_xlim(time[0], time[-1])
    axis.set_ylim(-35.0, 28.0)
    axis.legend(loc="upper right", fontsize=8)

    secondary = axis.secondary_yaxis(
        "right",
        functions=(lambda current: K_T * current, lambda torque: torque / K_T),
    )
    secondary.set_ylabel(r"トルク $\tau$ [N m]")


def draw_voltage_trace(axis, data: dict[str, np.ndarray]) -> None:
    time = data["time"]
    axis.plot(time, data["voltage_cmd"], color="#9467bd", linewidth=1.8, label=r"計算 $v_c$")
    axis.plot(time, data["voltage_actual"], color="#d62728", linewidth=1.9, label=r"印加 $v$")
    axis.axhline(V_MAX, color="#777777", linestyle=":", linewidth=1.2)
    axis.axhline(-V_MAX, color="#777777", linestyle=":", linewidth=1.2)
    axis.text(0.03, 0.93, "電圧指令と印加電圧", transform=axis.transAxes, ha="left", va="top", fontsize=9)
    axis.set_xlabel(r"時間 $t$ [s]")
    axis.set_ylabel(r"電圧 $v$ [V]")
    axis.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)
    axis.set_xlim(time[0], time[-1])
    axis.set_ylim(-36.0, 34.0)
    axis.legend(loc="upper right", fontsize=8)


def draw_margin_trace(axis, data: dict[str, np.ndarray]) -> None:
    time = data["time"]
    axis.plot(time, data["current_margin_cmd"], color="#1f77b4", linewidth=1.9, label=r"$m_I^c=1-|i_r^c|/I_{max}$")
    axis.plot(time, data["voltage_margin_cmd"], color="#d62728", linewidth=1.9, label=r"$m_V^c=1-|v_c|/V_{max}$")
    axis.axhline(0.0, color="#333333", linewidth=1.1)
    axis.fill_between(time, -3.3, 0.0, color="#ffe8e8", alpha=0.8)
    axis.set_xlabel(r"時間 $t$ [s]")
    axis.set_ylabel(r"制約余裕 [-]")
    axis.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)
    axis.set_xlim(time[0], time[-1])
    axis.set_ylim(-3.3, 1.1)
    axis.legend(loc="lower right", fontsize=8)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = repo_root / "ja" / "figures"
    output_path = output_dir / "cascade_actuator_limits_examples.png"
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_fonts()
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "figure.dpi": 180,
            "savefig.dpi": 180,
        }
    )

    data = simulate_cascade()
    figure = plt.figure(figsize=(7.2, 8.4), constrained_layout=False)
    grid = GridSpec(3, 2, figure=figure, height_ratios=[1.28, 1.0, 1.0])

    block_axis = figure.add_subplot(grid[0, :])
    speed_axis = figure.add_subplot(grid[1, 0])
    current_axis = figure.add_subplot(grid[1, 1])
    voltage_axis = figure.add_subplot(grid[2, 0])
    margin_axis = figure.add_subplot(grid[2, 1])

    draw_block_diagram(block_axis)
    draw_speed_trace(speed_axis, data)
    draw_current_torque_trace(current_axis, data)
    draw_voltage_trace(voltage_axis, data)
    draw_margin_trace(margin_axis, data)

    figure.subplots_adjust(left=0.08, right=0.91, top=0.985, bottom=0.075, hspace=0.44, wspace=0.38)
    finalize_figure(figure, output_path, layout="none")
    plt.close(figure)
    print(output_path)


if __name__ == "__main__":
    main()
