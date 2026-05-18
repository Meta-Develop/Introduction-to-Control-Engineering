#!/usr/bin/env python3
"""Generate nonlinear geometry and estimation examples for the Japanese text."""

from __future__ import annotations

from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("numpy is required to generate the nonlinear figure") from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("matplotlib is required to generate the nonlinear figure") from exc


DAMPING = 0.35
LINEAR_THETA_LIMIT = 0.55
LINEAR_OMEGA_LIMIT = 1.0
PENDULUM_TIME = np.linspace(0.0, 18.0, 520)

MEAN = np.array([0.55, 0.05])
COVARIANCE = np.array([[0.35**2, 0.018], [0.018, 0.25**2]])
MEASUREMENT_X2_COEFF = 0.35
MEASUREMENT_VALUE = 0.55
MEASUREMENT_STD = 0.08
SIGMA_ALPHA = 0.85
PARTICLE_COUNT = 600
RNG_SEED = 42


def pendulum_rhs(state: np.ndarray) -> np.ndarray:
    theta, omega = state
    return np.array([omega, -np.sin(theta) - DAMPING * omega])


def rk4_step(state: np.ndarray, step_size: float) -> np.ndarray:
    k1 = pendulum_rhs(state)
    k2 = pendulum_rhs(state + 0.5 * step_size * k1)
    k3 = pendulum_rhs(state + 0.5 * step_size * k2)
    k4 = pendulum_rhs(state + step_size * k3)
    return state + (step_size / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def wrap_angle(theta: float) -> float:
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def simulate_pendulum(initial_state: np.ndarray) -> np.ndarray:
    step_size = float(PENDULUM_TIME[1] - PENDULUM_TIME[0])
    state = np.array(initial_state, dtype=float)
    trajectory = []
    for _ in PENDULUM_TIME:
        trajectory.append([wrap_angle(state[0]), state[1]])
        state = rk4_step(state, step_size)
        state[0] = wrap_angle(float(state[0]))
    return np.asarray(trajectory)


def pendulum_energy(theta: np.ndarray, omega: np.ndarray) -> np.ndarray:
    return 0.5 * omega**2 + 1.0 - np.cos(theta)


def measurement_function(points: np.ndarray) -> np.ndarray:
    return points[..., 0] ** 2 + MEASUREMENT_X2_COEFF * points[..., 1]


def covariance_ellipse(
    mean: np.ndarray, covariance: np.ndarray, scale: float = 2.0
) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, 240)
    circle = np.vstack((np.cos(angles), np.sin(angles)))
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    ellipse = vectors @ np.diag(np.sqrt(np.maximum(values, 0.0))) @ circle
    return mean + scale * ellipse.T


def ekf_measurement_update() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h_mean = measurement_function(MEAN)
    jacobian = np.array([[2.0 * MEAN[0], MEASUREMENT_X2_COEFF]])
    innovation_variance = float(
        (jacobian @ COVARIANCE @ jacobian.T)[0, 0] + MEASUREMENT_STD**2
    )
    gain = COVARIANCE @ jacobian.T / innovation_variance
    posterior_mean = MEAN + gain[:, 0] * (MEASUREMENT_VALUE - h_mean)
    identity = np.eye(2)
    posterior_covariance = (
        (identity - gain @ jacobian)
        @ COVARIANCE
        @ (identity - gain @ jacobian).T
        + gain @ np.array([[MEASUREMENT_STD**2]]) @ gain.T
    )
    return jacobian[0], posterior_mean, posterior_covariance


def sigma_points() -> np.ndarray:
    dimension = MEAN.size
    lam = SIGMA_ALPHA**2 * dimension - dimension
    spread = np.linalg.cholesky((dimension + lam) * COVARIANCE)
    points = [MEAN]
    for column in range(dimension):
        points.append(MEAN + spread[:, column])
        points.append(MEAN - spread[:, column])
    return np.asarray(points)


def configure_state_axis(axis, title: str) -> None:
    axis.axhline(0.0, color="#777777", linewidth=0.8)
    axis.axvline(0.0, color="#777777", linewidth=0.8)
    axis.set_xlim(-1.25, 1.45)
    axis.set_ylim(-0.95, 1.10)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(r"状態 $x_1$ [-]")
    axis.set_ylabel(r"状態 $x_2$ [-]")
    axis.set_title(title)
    axis.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.75)


def add_panel_label(axis, label: str) -> None:
    axis.text(
        0.015,
        0.985,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="#222222",
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "#cccccc",
            "alpha": 0.9,
        },
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = (
        repo_root / "ja" / "figures" / "nonlinear_geometry_estimation_examples.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 9,
            "font.family": [
                "Noto Sans CJK JP",
                "Noto Sans",
                "DejaVu Sans",
            ],
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 7,
            "figure.dpi": 180,
            "savefig.dpi": 180,
            "axes.unicode_minus": False,
        }
    )

    theta_grid = np.linspace(-np.pi, np.pi, 180)
    omega_grid = np.linspace(-2.7, 2.7, 160)
    theta_mesh, omega_mesh = np.meshgrid(theta_grid, omega_grid)
    energy = pendulum_energy(theta_mesh, omega_mesh)
    energy_derivative = -DAMPING * omega_mesh**2

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.4), constrained_layout=False)
    fig.suptitle("非線形幾何と非線形推定近似", y=0.992)
    fig.text(
        0.5,
        0.958,
        (
            r"振子: $\dot{\theta}=\omega,\ \dot{\omega}=-\sin\theta-0.35\omega$, "
            r"$V=\omega^2/2+1-\cos\theta$; "
            r"推定: $h(x)=x_1^2+0.35x_2,\ y=0.55,\ R=0.08^2$"
        ),
        ha="center",
        va="top",
        fontsize=8,
    )

    phase_axis = axes[0, 0]
    phase_theta = np.linspace(-np.pi, np.pi, 28)
    phase_omega = np.linspace(-2.5, 2.5, 24)
    th_stream, om_stream = np.meshgrid(phase_theta, phase_omega)
    stream_u = om_stream
    stream_v = -np.sin(th_stream) - DAMPING * om_stream
    phase_axis.streamplot(
        phase_theta,
        phase_omega,
        stream_u,
        stream_v,
        color="#b9b9b9",
        density=0.92,
        linewidth=0.75,
        arrowsize=0.65,
    )
    phase_axis.contour(
        theta_mesh,
        omega_mesh,
        energy,
        levels=[0.35, 0.9, 1.5, 2.0],
        colors=["#777777", "#777777", "#777777", "#333333"],
        linewidths=[0.8, 0.9, 1.0, 1.4],
        linestyles=[":", "--", "-", "--"],
    )
    phase_axis.add_patch(
        Rectangle(
            (-LINEAR_THETA_LIMIT, -LINEAR_OMEGA_LIMIT),
            2.0 * LINEAR_THETA_LIMIT,
            2.0 * LINEAR_OMEGA_LIMIT,
            facecolor="#f4a261",
            edgecolor="#d95f02",
            alpha=0.22,
            linestyle="--",
            linewidth=1.3,
        )
    )
    initial_states = (
        np.array([2.1, 0.15]),
        np.array([-2.25, -0.25]),
        np.array([1.15, 1.45]),
        np.array([-1.0, -1.55]),
        np.array([0.45, 1.95]),
    )
    colors = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e")
    for initial_state, color in zip(initial_states, colors):
        trajectory = simulate_pendulum(initial_state)
        phase_axis.plot(trajectory[:, 0], trajectory[:, 1], color=color, linewidth=1.5)
        phase_axis.plot(
            trajectory[0, 0],
            trajectory[0, 1],
            marker="o",
            color=color,
            markersize=3,
        )
    phase_axis.plot(0.0, 0.0, marker="o", color="#111111", markersize=4)
    phase_axis.plot(
        [-np.pi, np.pi], [0.0, 0.0], marker="x", linestyle="none", color="#111111"
    )
    phase_axis.text(-0.52, 1.12, "局所線形化領域", color="#9a4f00", fontsize=8)
    phase_axis.set_title("位相図・準位線・線形化領域")
    phase_axis.set_xlabel(r"角度 $\theta$ [rad]")
    phase_axis.set_ylabel(r"角速度 $\omega$ [rad/s]")
    phase_axis.set_xlim(-np.pi, np.pi)
    phase_axis.set_ylim(-2.7, 2.7)
    phase_axis.set_aspect("equal", adjustable="box")
    phase_axis.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.75)
    add_panel_label(phase_axis, "A")

    invariant_axis = axes[0, 1]
    invariant_axis.contourf(
        theta_mesh,
        omega_mesh,
        energy_derivative,
        levels=[-2.6, -1.2, -0.45, -0.12, -0.02, 0.0],
        colors=["#225ea8", "#41b6c4", "#a1dab4", "#ffffcc", "#fff7d6"],
        alpha=0.72,
    )
    invariant_axis.contour(
        theta_mesh,
        omega_mesh,
        energy,
        levels=[0.55, 1.2, 1.8, 2.0],
        colors=["#444444", "#444444", "#d95f02", "#222222"],
        linewidths=[0.9, 1.0, 1.5, 1.3],
        linestyles=[":", "--", "-", "--"],
    )
    invariant_axis.axhline(0.0, color="#d62728", linewidth=1.7, label=r"$E:\dot{V}=0$")
    invariant_axis.plot(0.0, 0.0, marker="o", color="#111111", markersize=4)
    invariant_axis.plot(
        [-np.pi, np.pi],
        [0.0, 0.0],
        marker="x",
        linestyle="none",
        color="#111111",
        markersize=5,
    )
    invariant_axis.arrow(
        1.2,
        0.0,
        0.0,
        -0.38,
        color="#333333",
        width=0.012,
        head_width=0.12,
        head_length=0.12,
        length_includes_head=True,
    )
    invariant_axis.arrow(
        -1.2,
        0.0,
        0.0,
        0.38,
        color="#333333",
        width=0.012,
        head_width=0.12,
        head_length=0.12,
        length_includes_head=True,
    )
    invariant_axis.text(0.16, 0.18, r"$M=\{(0,0)\}$", fontsize=8, color="#111111")
    invariant_axis.text(-2.75, 1.55, r"$\Omega_{1.8}$", fontsize=8, color="#d95f02")
    invariant_axis.text(1.55, -0.58, r"$c=2$ は上向き平衡点を含む", fontsize=7)
    invariant_axis.set_title(r"Lyapunov 準位集合と不変集合")
    invariant_axis.set_xlabel(r"角度 $\theta$ [rad]")
    invariant_axis.set_ylabel(r"角速度 $\omega$ [rad/s]")
    invariant_axis.set_xlim(-np.pi, np.pi)
    invariant_axis.set_ylim(-2.7, 2.7)
    invariant_axis.set_aspect("equal", adjustable="box")
    invariant_axis.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.75)
    invariant_axis.legend(loc="upper right")
    add_panel_label(invariant_axis, "B")

    jacobian, posterior_mean, posterior_covariance = ekf_measurement_update()
    x1_line = np.linspace(-1.25, 1.45, 400)
    nonlinear_curve = (MEASUREMENT_VALUE - x1_line**2) / MEASUREMENT_X2_COEFF
    h_mean = measurement_function(MEAN)
    tangent_line = (
        MEAN[1]
        + (
            MEASUREMENT_VALUE
            - h_mean
            - jacobian[0] * (x1_line - MEAN[0])
        )
        / jacobian[1]
    )

    ekf_axis = axes[1, 0]
    configure_state_axis(ekf_axis, "EKF: 接線測定による楕円更新")
    ekf_axis.plot(
        x1_line,
        nonlinear_curve,
        color="#333333",
        linewidth=1.6,
        label=r"非線形測定 $h(x)=y$",
    )
    ekf_axis.plot(
        x1_line,
        tangent_line,
        color="#d62728",
        linestyle="--",
        linewidth=1.5,
        label=r"接線 $H=[1.10,\ 0.35]$",
    )
    prior_ellipse = covariance_ellipse(MEAN, COVARIANCE)
    posterior_ellipse = covariance_ellipse(posterior_mean, posterior_covariance)
    ekf_axis.plot(
        prior_ellipse[:, 0],
        prior_ellipse[:, 1],
        color="#777777",
        linestyle=":",
        linewidth=1.7,
        label="予測楕円",
    )
    ekf_axis.plot(
        posterior_ellipse[:, 0],
        posterior_ellipse[:, 1],
        color="#1f77b4",
        linewidth=1.9,
        label="EKF 更新楕円",
    )
    ekf_axis.plot(MEAN[0], MEAN[1], marker="o", color="#777777", markersize=4)
    ekf_axis.plot(
        posterior_mean[0],
        posterior_mean[1],
        marker="o",
        color="#1f77b4",
        markersize=4,
    )
    ekf_axis.text(
        0.03,
        0.05,
        r"$\bar{x}=(0.55,0.05)^\mathsf{T}$, $P_{11}=0.35^2,\ P_{22}=0.25^2,\ P_{12}=0.018$",
        transform=ekf_axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": "#cccccc",
            "alpha": 0.9,
        },
    )
    ekf_axis.legend(loc="upper right")
    add_panel_label(ekf_axis, "C")

    particle_axis = axes[1, 1]
    configure_state_axis(particle_axis, "UKF シグマ点と粒子雲")
    rng = np.random.default_rng(RNG_SEED)
    particles = rng.multivariate_normal(MEAN, COVARIANCE, size=PARTICLE_COUNT)
    residual = MEASUREMENT_VALUE - measurement_function(particles)
    weights = np.exp(-0.5 * (residual / MEASUREMENT_STD) ** 2)
    weights /= np.max(weights)
    effective_sample_size = float((np.sum(weights) ** 2) / np.sum(weights**2))
    particle_axis.scatter(
        particles[:, 0],
        particles[:, 1],
        s=7.0 + 20.0 * weights,
        c=weights,
        cmap="viridis",
        alpha=0.48,
        linewidths=0.0,
        label="粒子雲",
    )
    particle_axis.plot(
        x1_line,
        nonlinear_curve,
        color="#333333",
        linewidth=1.4,
        label=r"尤度が大きい曲線",
    )
    sigma = sigma_points()
    particle_axis.scatter(
        sigma[:, 0],
        sigma[:, 1],
        marker="D",
        s=42,
        color="#d62728",
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
        label=rf"UKF シグマ点 $\alpha={SIGMA_ALPHA:g}$",
    )
    particle_axis.plot(
        prior_ellipse[:, 0],
        prior_ellipse[:, 1],
        color="#777777",
        linestyle=":",
        linewidth=1.4,
        label="予測楕円",
    )
    particle_axis.text(
        0.03,
        0.05,
        rf"$N={PARTICLE_COUNT}$, $N_{{\mathrm{{eff}}}}\simeq{effective_sample_size:.0f}$",
        transform=particle_axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": "#cccccc",
            "alpha": 0.9,
        },
    )
    particle_axis.legend(loc="upper right")
    add_panel_label(particle_axis, "D")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(output_path, facecolor="white")
    print(output_path)


if __name__ == "__main__":
    main()
