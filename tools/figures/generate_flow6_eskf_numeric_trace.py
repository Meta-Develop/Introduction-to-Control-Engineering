#!/usr/bin/env python3
"""Generate a numeric ESKF attitude-update trace for Flow 6."""

from __future__ import annotations

from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("numpy is required to generate the ESKF numeric trace") from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("matplotlib is required to generate the ESKF numeric trace") from exc

from figure_style import finalize_figure


DT = 0.01
T_FINAL = 6.0
UPDATE_TIME = 4.0
RAD_TO_DEG = 180.0 / np.pi
DEG_TO_RAD = np.pi / 180.0

TRUST_CASES = (
    ("重力を強く信頼", 0.60),
    ("標準設定", 1.20),
    ("加速度を疑う", 3.50),
)


def build_trace() -> dict[str, np.ndarray | float]:
    time = np.arange(0.0, T_FINAL + DT, DT)
    update_index = int(round(UPDATE_TIME / DT))

    theta_true = DEG_TO_RAD * (
        3.7 * np.sin(0.68 * time) + 1.2 * np.sin(1.25 * time + 0.35)
    )
    omega_true = np.gradient(theta_true, DT)

    gyro_bias_true = DEG_TO_RAD * (0.40 + 0.09 * time)
    gyro_bias_hat = DEG_TO_RAD * 0.12
    omega_used = omega_true + gyro_bias_true - gyro_bias_hat

    theta_nominal = np.zeros_like(time)
    theta_nominal[0] = theta_true[0] - DEG_TO_RAD * 0.85
    for index in range(1, time.size):
        theta_nominal[index] = theta_nominal[index - 1] + omega_used[index - 1] * DT

    sigma0 = DEG_TO_RAD * 0.60
    gyro_random_walk = DEG_TO_RAD * 0.70
    covariance_prior = sigma0**2 + (gyro_random_walk**2) * time
    prior_update = covariance_prior[update_index]

    accel_tilt_error = DEG_TO_RAD * -0.75
    accel_tilt = theta_true[update_index] + accel_tilt_error
    innovation = accel_tilt - theta_nominal[update_index]

    nominal_measurement_sigma = DEG_TO_RAD * TRUST_CASES[1][1]
    nominal_gain = prior_update / (prior_update + nominal_measurement_sigma**2)
    nominal_delta = nominal_gain * innovation
    nominal_posterior = (1.0 - nominal_gain) ** 2 * prior_update
    nominal_posterior += nominal_gain**2 * nominal_measurement_sigma**2

    theta_injected = theta_nominal.copy()
    theta_injected[update_index:] += nominal_delta

    covariance_reset = covariance_prior.copy()
    covariance_reset[update_index:] = nominal_posterior
    covariance_reset[update_index:] += gyro_random_walk**2 * (
        time[update_index:] - UPDATE_TIME
    )

    return {
        "time": time,
        "update_index": float(update_index),
        "theta_true": theta_true,
        "theta_nominal": theta_nominal,
        "theta_injected": theta_injected,
        "covariance_prior": covariance_prior,
        "covariance_reset": covariance_reset,
        "prior_update": prior_update,
        "accel_tilt": accel_tilt,
        "accel_tilt_error": accel_tilt_error,
        "innovation": innovation,
        "nominal_gain": nominal_gain,
        "nominal_delta": nominal_delta,
        "nominal_posterior": nominal_posterior,
    }


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        0.02,
        0.96,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": "#bbbbbb",
            "alpha": 0.95,
        },
    )


def trust_values(data: dict[str, np.ndarray | float]) -> list[dict[str, float | str]]:
    update_index = int(data["update_index"])
    theta_true = data["theta_true"]
    theta_nominal = data["theta_nominal"]
    assert isinstance(theta_true, np.ndarray)
    assert isinstance(theta_nominal, np.ndarray)

    prior_update = float(data["prior_update"])
    innovation = float(data["innovation"])
    prior_error = theta_nominal[update_index] - theta_true[update_index]

    values: list[dict[str, float | str]] = []
    for label, sigma_deg in TRUST_CASES:
        measurement_variance = (DEG_TO_RAD * sigma_deg) ** 2
        gain = prior_update / (prior_update + measurement_variance)
        delta = gain * innovation
        posterior = (1.0 - gain) ** 2 * prior_update + gain**2 * measurement_variance
        values.append(
            {
                "label": label,
                "sigma_deg": sigma_deg,
                "gain": gain,
                "delta_deg": delta * RAD_TO_DEG,
                "post_error_deg": (prior_error + delta) * RAD_TO_DEG,
                "post_sigma_deg": np.sqrt(posterior) * RAD_TO_DEG,
            }
        )
    return values


def draw_attitude_trace(axis: plt.Axes, data: dict[str, np.ndarray | float]) -> None:
    time = data["time"]
    theta_true = data["theta_true"]
    theta_nominal = data["theta_nominal"]
    theta_injected = data["theta_injected"]
    accel_tilt = float(data["accel_tilt"])
    update_index = int(data["update_index"])
    assert isinstance(time, np.ndarray)
    assert isinstance(theta_true, np.ndarray)
    assert isinstance(theta_nominal, np.ndarray)
    assert isinstance(theta_injected, np.ndarray)

    add_panel_label(axis, "(a) IMU 予測と注入")
    axis.plot(time, theta_true * RAD_TO_DEG, color="#222222", linewidth=2.1, label="真値")
    axis.plot(
        time,
        theta_nominal * RAD_TO_DEG,
        color="#D55E00",
        linewidth=1.5,
        linestyle="--",
        label=r"名目予測 $\hat{\theta}^{-}$",
    )
    axis.plot(
        time,
        theta_injected * RAD_TO_DEG,
        color="#009E73",
        linewidth=2.0,
        label=r"注入後 $\hat{\theta}^{+}$",
    )
    axis.scatter(
        [UPDATE_TIME],
        [accel_tilt * RAD_TO_DEG],
        s=42,
        color="#0072B2",
        marker="o",
        zorder=5,
        label="加速度計の傾き測定",
    )
    axis.axvline(UPDATE_TIME, color="#777777", linestyle=":", linewidth=1.1)
    axis.annotate(
        r"$\widehat{\delta\theta}=K\nu$ を注入",
        xy=(UPDATE_TIME, theta_injected[update_index] * RAD_TO_DEG),
        xytext=(UPDATE_TIME + 0.28, theta_injected[update_index] * RAD_TO_DEG + 1.0),
        arrowprops={"arrowstyle": "->", "color": "#444444", "linewidth": 1.1},
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 0.4},
    )
    axis.set_xlim(0.0, T_FINAL)
    axis.set_ylabel("Pitch angle [deg]")
    axis.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.75)
    axis.legend(loc="lower left", frameon=True, ncols=2)


def draw_covariance_trace(axis: plt.Axes, data: dict[str, np.ndarray | float]) -> None:
    time = data["time"]
    covariance_prior = data["covariance_prior"]
    covariance_reset = data["covariance_reset"]
    assert isinstance(time, np.ndarray)
    assert isinstance(covariance_prior, np.ndarray)
    assert isinstance(covariance_reset, np.ndarray)

    add_panel_label(axis, "(b) 共分散の成長と縮小")
    axis.plot(
        time,
        np.sqrt(covariance_prior) * RAD_TO_DEG,
        color="#CC6677",
        linewidth=1.8,
        linestyle="--",
        label=r"予測 $\sigma^{-}$",
    )
    axis.plot(
        time,
        np.sqrt(covariance_reset) * RAD_TO_DEG,
        color="#4477AA",
        linewidth=2.0,
        label=r"更新・リセット後 $\sigma$",
    )
    axis.axvline(UPDATE_TIME, color="#777777", linestyle=":", linewidth=1.1)
    axis.fill_between(
        [UPDATE_TIME - 0.16, UPDATE_TIME + 0.16],
        0.0,
        3.0,
        color="#88CCEE",
        alpha=0.16,
        linewidth=0,
        label="測定更新",
    )
    axis.set_xlim(0.0, T_FINAL)
    axis.set_ylim(0.0, 2.6)
    axis.set_ylabel(r"$1\sigma$ attitude error [deg]")
    axis.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.75)
    axis.legend(loc="lower right", frameon=True)


def draw_gain_trace(axis: plt.Axes, data: dict[str, np.ndarray | float]) -> None:
    values = trust_values(data)
    labels = [str(item["label"]) for item in values]
    x_pos = np.arange(len(values))
    gains = np.array([float(item["gain"]) for item in values])
    deltas = np.array([float(item["delta_deg"]) for item in values])

    add_panel_label(axis, "(c) 革新と Kalman ゲイン")
    bars = axis.bar(x_pos, gains, width=0.58, color="#88CCEE", edgecolor="#336699")
    for bar, delta_deg in zip(bars, deltas):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.035,
            rf"$\delta\theta={delta_deg:.2f}^\circ$",
            ha="center",
            va="bottom",
            fontsize=8.4,
        )
    innovation_deg = float(data["innovation"]) * RAD_TO_DEG
    axis.text(
        0.5,
        0.16,
        rf"共通の革新 $\nu={innovation_deg:.2f}^\circ$",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=9.0,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.95, "pad": 2.0},
    )
    axis.set_xticks(x_pos)
    axis.set_xticklabels(labels, rotation=0)
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel(r"Kalman gain $K$ [-]")
    axis.grid(True, axis="y", color="#d0d0d0", linewidth=0.7, alpha=0.75)


def draw_trust_reset(axis: plt.Axes, data: dict[str, np.ndarray | float]) -> None:
    values = trust_values(data)
    labels = [str(item["label"]) for item in values]
    x_pos = np.arange(len(values))
    post_error = np.array([float(item["post_error_deg"]) for item in values])
    post_sigma = np.array([float(item["post_sigma_deg"]) for item in values])

    update_index = int(data["update_index"])
    theta_true = data["theta_true"]
    theta_nominal = data["theta_nominal"]
    assert isinstance(theta_true, np.ndarray)
    assert isinstance(theta_nominal, np.ndarray)

    prior_error = (theta_nominal[update_index] - theta_true[update_index]) * RAD_TO_DEG
    accel_error = float(data["accel_tilt_error"]) * RAD_TO_DEG

    add_panel_label(axis, "(d) 注入後の誤差とリセット")
    axis.axhline(0.0, color="#333333", linewidth=1.0)
    axis.axhline(
        prior_error,
        color="#D55E00",
        linestyle="--",
        linewidth=1.4,
        label=rf"更新前 {prior_error:.2f}$^\circ$",
    )
    axis.axhline(
        accel_error,
        color="#0072B2",
        linestyle=":",
        linewidth=1.4,
        label=rf"加速度外乱 {accel_error:.2f}$^\circ$",
    )
    axis.bar(
        x_pos,
        post_error,
        width=0.52,
        color="#DDCC77",
        edgecolor="#8A7A22",
        label="注入後誤差",
    )
    axis.errorbar(
        x_pos,
        post_error,
        yerr=post_sigma,
        fmt="none",
        ecolor="#228833",
        elinewidth=1.6,
        capsize=5,
        label=r"更新後 $\pm1\sigma$",
    )
    axis.scatter(
        x_pos,
        np.zeros_like(x_pos),
        marker="x",
        s=56,
        color="#882255",
        linewidths=1.8,
        label=r"平均リセット",
        zorder=5,
    )
    axis.text(
        0.03,
        0.06,
        "棒: 注入後誤差\n緑線: 更新後 $\\pm1\\sigma$\n×: 誤差平均を 0 に戻す",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.2,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.94, "pad": 2.0},
    )
    axis.set_xticks(x_pos)
    axis.set_xticklabels(labels)
    axis.set_ylabel("Angle error [deg]")
    axis.set_ylim(-2.2, 2.6)
    axis.grid(True, axis="y", color="#d0d0d0", linewidth=0.7, alpha=0.75)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "ja" / "figures" / "flow6_eskf_numeric_trace.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 9,
            "font.family": ["Noto Sans CJK JP", "Noto Sans", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "figure.dpi": 180,
            "savefig.dpi": 180,
            "axes.unicode_minus": False,
        }
    )

    data = build_trace()

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.2), constrained_layout=False)
    draw_attitude_trace(axes[0, 0], data)
    draw_covariance_trace(axes[0, 1], data)
    draw_gain_trace(axes[1, 0], data)
    draw_trust_reset(axes[1, 1], data)
    for axis in axes[1, :]:
        axis.set_xlabel("Measurement trust case")
    for axis in axes[0, :]:
        axis.set_xlabel("Time [s]")

    finalize_figure(fig, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
