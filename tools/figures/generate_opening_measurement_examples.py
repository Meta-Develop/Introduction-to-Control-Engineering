from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import Affine2D


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "ja" / "figures" / "opening_measurement_examples.png"

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


def add_robot(ax: plt.Axes, x: float, y: float, theta: float) -> None:
    body = Rectangle(
        (-0.08, -0.05),
        0.16,
        0.10,
        facecolor="#f8f6f0",
        edgecolor="#4a4a4a",
        linewidth=0.9,
        zorder=5,
    )
    transform = Affine2D().rotate(theta).translate(x, y) + ax.transData
    body.set_transform(transform)
    ax.add_patch(body)

    for wheel_y in (-0.064, 0.064):
        wheel = Rectangle(
            (-0.055, wheel_y - 0.012),
            0.11,
            0.024,
            facecolor="#3e3e3e",
            edgecolor="#222222",
            linewidth=0.6,
            zorder=6,
        )
        wheel.set_transform(transform)
        ax.add_patch(wheel)

    for sensor_y in np.linspace(-0.045, 0.045, 5):
        sensor = Circle((0.075, sensor_y), 0.006, facecolor="#d64b3c", edgecolor="none", zorder=7)
        sensor.set_transform(transform)
        ax.add_patch(sensor)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    configure_fonts()

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig = plt.figure(figsize=(7.2, 5.2), dpi=180, constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=(1.05, 1.0))

    ax_path = fig.add_subplot(gs[0, 0])
    ax_sensor = fig.add_subplot(gs[0, 1])
    ax_trace = fig.add_subplot(gs[1, :])

    x = np.linspace(0.0, 2.4, 240)
    y_des = 0.12 * np.sin(1.15 * np.pi * (x - 0.18) / 2.4)
    y_meas = y_des + 0.065 * np.exp(-0.55 * x) * np.sin(2.4 * np.pi * x + 0.45)
    y_meas += 0.020 * np.sin(0.7 * np.pi * x)

    ax_path.plot(x, y_des, color="#202020", linewidth=4.2, alpha=0.85, label="目標線")
    ax_path.plot(x, y_meas, color="#1f77b4", linewidth=2.0, label="測定軌跡")
    for idx in (52, 128, 205):
        theta = np.arctan2(np.gradient(y_meas, x)[idx], 1.0)
        add_robot(ax_path, float(x[idx]), float(y_meas[idx]), float(theta))
    ax_path.set_title("上面図：目標線と測定軌跡")
    ax_path.set_xlabel("進行方向の距離 [m]")
    ax_path.set_ylabel("横方向位置 [m]")
    ax_path.set_ylim(-0.16, 0.24)
    ax_path.set_aspect("auto")
    ax_path.grid(True, color="#dddddd", linewidth=0.6)
    ax_path.legend(loc="lower right", frameon=False)

    sensor_x = np.linspace(-45.0, 45.0, 300)
    line_center_mm = 12.0
    reflectance = 0.86 - 0.72 * np.exp(-0.5 * ((sensor_x - line_center_mm) / 8.0) ** 2)
    sensor_positions = np.array([-32.0, -16.0, 0.0, 16.0, 32.0])
    sensor_values = 0.86 - 0.72 * np.exp(-0.5 * ((sensor_positions - line_center_mm) / 8.0) ** 2)

    ax_sensor.plot(sensor_x, reflectance, color="#2f5597", linewidth=2.0, label="反射強度分布")
    ax_sensor.scatter(sensor_positions, sensor_values, s=34, color="#d64b3c", zorder=4, label="センサ測定点")
    ax_sensor.axvline(0.0, color="#666666", linestyle="--", linewidth=1.0, label="ロボット中心")
    ax_sensor.axvline(line_center_mm, color="#202020", linestyle="-", linewidth=1.1, label="ライン中心")
    ax_sensor.set_title("ラインセンサの反射強度分布")
    ax_sensor.set_xlabel("センサ横位置 [mm]")
    ax_sensor.set_ylabel("正規化強度 [-]")
    ax_sensor.set_ylim(0.05, 1.02)
    ax_sensor.grid(True, color="#dddddd", linewidth=0.6)
    ax_sensor.legend(
        loc="lower left",
        frameon=True,
        framealpha=0.94,
        edgecolor="#d0d0d0",
        ncol=1,
    )

    t = np.linspace(0.0, 4.0, 161)
    error_mm = 36.0 * np.exp(-0.42 * t) * np.sin(2.25 * np.pi * t + 0.35)
    error_mm += 8.0 * np.exp(-0.70 * t)
    wheel_delta_mm = -0.75 * error_mm + 4.0 * np.sin(1.1 * np.pi * t)

    ax_trace.plot(t, error_mm, color="#1f77b4", linewidth=2.0, label="横方向誤差")
    ax_trace.axhline(0.0, color="#202020", linewidth=1.0)
    ax_trace.set_title("時系列：誤差と左右輪の移動差")
    ax_trace.set_xlabel("時間 [s]")
    ax_trace.set_ylabel("横方向誤差 [mm]", color="#1f77b4")
    ax_trace.tick_params(axis="y", labelcolor="#1f77b4")
    ax_trace.grid(True, color="#dddddd", linewidth=0.6)

    ax_wheel = ax_trace.twinx()
    ax_wheel.plot(t, wheel_delta_mm, color="#d97706", linewidth=1.8, label="左右輪の移動差")
    ax_wheel.set_ylabel("左右輪の移動差 [mm]", color="#d97706")
    ax_wheel.tick_params(axis="y", labelcolor="#d97706")

    handles_1, labels_1 = ax_trace.get_legend_handles_labels()
    handles_2, labels_2 = ax_wheel.get_legend_handles_labels()
    ax_trace.legend(handles_1 + handles_2, labels_1 + labels_2, loc="upper right", frameon=False)

    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
