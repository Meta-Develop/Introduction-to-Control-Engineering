#!/usr/bin/env python3
"""Generate LQR and Kalman filter visual examples for the Japanese text."""

from __future__ import annotations

from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("numpy is required to generate the LQR/Kalman figure") from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("matplotlib is required to generate the LQR/Kalman figure") from exc


SAMPLE_TIME = 0.1
SIMULATION_STEPS = 80
RICCATI_STEPS = 70
R_SWEEP = (0.03, 0.12, 0.5, 2.0)
SATURATION_LIMIT = 2.0

A = np.array([[1.0, SAMPLE_TIME], [0.0, 1.0]])
B = np.array([[0.5 * SAMPLE_TIME**2], [SAMPLE_TIME]])
C = np.array([[1.0, 0.0]])
Q_STATE = np.diag([1.0, 0.08])
INITIAL_STATE = np.array([1.0, 0.0])


def dlqr_gain_from_cost_to_go(cost_to_go: np.ndarray, r_weight: float) -> np.ndarray:
    control_weight = np.array([[r_weight]])
    innovation = control_weight + B.T @ cost_to_go @ B
    return np.linalg.solve(innovation, B.T @ cost_to_go @ A)


def dare_step(cost_to_go: np.ndarray, r_weight: float) -> np.ndarray:
    gain = dlqr_gain_from_cost_to_go(cost_to_go, r_weight)
    return Q_STATE + A.T @ cost_to_go @ (A - B @ gain)


def solve_dare(r_weight: float, tolerance: float = 1.0e-12) -> np.ndarray:
    cost_to_go = Q_STATE.copy()
    for _ in range(10000):
        next_cost = dare_step(cost_to_go, r_weight)
        if np.linalg.norm(next_cost - cost_to_go, ord="fro") < tolerance:
            return next_cost
        cost_to_go = next_cost
    return cost_to_go


def riccati_sequence(r_weight: float, steps: int) -> np.ndarray:
    cost_to_go = np.zeros_like(Q_STATE)
    sequence = [cost_to_go.copy()]
    for _ in range(steps):
        cost_to_go = dare_step(cost_to_go, r_weight)
        sequence.append(cost_to_go.copy())
    return np.array(sequence)


def simulate_lqr(r_weight: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cost_to_go = solve_dare(r_weight)
    gain = dlqr_gain_from_cost_to_go(cost_to_go, r_weight)
    state = INITIAL_STATE.copy()
    states = [state.copy()]
    inputs = []

    for _ in range(SIMULATION_STEPS):
        control_command = float(-(gain @ state.reshape(-1, 1))[0, 0])
        inputs.append(control_command)
        state = A @ state + B[:, 0] * control_command
        states.append(state.copy())

    return (
        np.arange(SIMULATION_STEPS + 1) * SAMPLE_TIME,
        np.array(states),
        np.array(inputs),
    )


def covariance_update(prior: np.ndarray, measurement_variance: float) -> np.ndarray:
    innovation = C @ prior @ C.T + np.array([[measurement_variance]])
    kalman_gain = prior @ C.T @ np.linalg.inv(innovation)
    identity = np.eye(prior.shape[0])
    posterior = (identity - kalman_gain @ C) @ prior @ (identity - kalman_gain @ C).T
    posterior += kalman_gain @ np.array([[measurement_variance]]) @ kalman_gain.T
    return posterior


def ellipse_points(covariance: np.ndarray, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.linspace(0.0, 2.0 * np.pi, 240)
    unit_circle = np.vstack((np.cos(angle), np.sin(angle)))
    ellipse = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0))) @ unit_circle
    return scale * ellipse[0, :], scale * ellipse[1, :]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "ja" / "figures" / "lqr_kalman_examples.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 9,
            "font.family": ["Noto Sans CJK JP", "Noto Sans", "DejaVu Sans"],
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 7,
            "figure.dpi": 180,
            "savefig.dpi": 180,
            "axes.unicode_minus": False,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.6, 7.2), constrained_layout=False)
    fig.suptitle("LQR とカルマンフィルタの調整例", y=0.985)
    lqr_cases = ",".join(f"{r_weight:g}" for r_weight in R_SWEEP)
    fig.text(
        0.5,
        0.948,
        (
            f"設定: h={SAMPLE_TIME:g} s, "
            f"Q_x=diag(1,0.08), R_u={{{lqr_cases}}}, "
            "R_v={0.02,0.20}"
        ),
        ha="center",
        va="top",
        fontsize=8,
    )
    response_ax, input_ax, riccati_ax, covariance_ax = axes.ravel()

    colors = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd")
    linestyles = ("-", "--", "-.", ":")

    input_time = np.arange(SIMULATION_STEPS) * SAMPLE_TIME
    for r_weight, color, linestyle in zip(R_SWEEP, colors, linestyles):
        time, states, inputs = simulate_lqr(r_weight)
        response_ax.plot(
            time,
            states[:, 0],
            color=color,
            linestyle=linestyle,
            linewidth=1.9,
            label=rf"$R_u={r_weight:g}$",
        )
        input_ax.plot(
            input_time,
            inputs,
            color=color,
            linestyle=linestyle,
            linewidth=1.9,
            label=rf"$R_u={r_weight:g}$",
        )

    response_ax.axhline(0.0, color="#555555", linestyle=":", linewidth=0.9)
    response_ax.set_title("LQR 重み掃引: 位置偏差")
    response_ax.set_xlabel("時刻 t [s]")
    response_ax.set_ylabel(r"位置偏差 $x_1$ [-]")
    response_ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)
    response_ax.legend(loc="upper right")
    response_ax.text(
        0.03,
        0.08,
        r"$x_0=(1,0)^\mathsf{T}$",
        transform=response_ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=7,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "alpha": 0.88,
            "edgecolor": "#bbbbbb",
        },
    )

    for limit in (-SATURATION_LIMIT, SATURATION_LIMIT):
        input_ax.axhline(limit, color="#555555", linestyle=":", linewidth=0.9)
    input_ax.set_title("LQR 重み掃引: 入力指令")
    input_ax.set_xlabel("時刻 t [s]")
    input_ax.set_ylabel(r"入力指令 $u_c$ [-]")
    input_ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)
    input_ax.legend(loc="lower right")
    input_ax.text(
        0.03,
        0.92,
        r"点線: 入力限界 $\pm 2$",
        transform=input_ax.transAxes,
        va="top",
        ha="left",
        fontsize=7,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "alpha": 0.88,
            "edgecolor": "#bbbbbb",
        },
    )

    riccati_r = 0.12
    sequence = riccati_sequence(riccati_r, RICCATI_STEPS)
    steady_cost = solve_dare(riccati_r)
    errors = np.linalg.norm(sequence - steady_cost, axis=(1, 2))
    riccati_ax.semilogy(
        np.arange(RICCATI_STEPS + 1),
        errors,
        color="#1f77b4",
        linewidth=2.0,
    )
    riccati_ax.set_title("後退リカッチ再帰の収束")
    riccati_ax.set_xlabel("終端からの後退ステップ数")
    riccati_ax.set_ylabel(r"$\|P_k-P_\infty\|_F$")
    riccati_ax.grid(True, which="both", color="#d0d0d0", linewidth=0.7, alpha=0.7)
    riccati_ax.text(
        0.04,
        0.08,
        rf"設定: $Q_x=\mathrm{{diag}}(1,0.08),\ R_u={riccati_r:g},\ P_N=0$",
        transform=riccati_ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=7,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "alpha": 0.88,
            "edgecolor": "#bbbbbb",
        },
    )

    previous_posterior = np.array([[0.24, 0.05], [0.05, 0.32]])
    process_noise = np.array(
        [
            [SAMPLE_TIME**3 / 3.0, SAMPLE_TIME**2 / 2.0],
            [SAMPLE_TIME**2 / 2.0, SAMPLE_TIME],
        ]
    ) * 0.18
    prior = A @ previous_posterior @ A.T + process_noise
    prior_x, prior_y = ellipse_points(prior)
    covariance_ax.plot(
        prior_x,
        prior_y,
        color="#333333",
        linestyle="--",
        linewidth=1.7,
        label=r"予測後 $P^-$",
    )

    covariance_cases = (
        (0.02, "#d62728", r"更新後 $R_v=0.02$"),
        (0.20, "#2ca02c", r"更新後 $R_v=0.20$"),
    )
    for measurement_variance, color, label in covariance_cases:
        posterior = covariance_update(prior, measurement_variance)
        ellipse_x, ellipse_y = ellipse_points(posterior)
        covariance_ax.plot(
            ellipse_x,
            ellipse_y,
            color=color,
            linewidth=2.0,
            label=label,
        )

    covariance_ax.axhline(0.0, color="#777777", linewidth=0.8)
    covariance_ax.axvline(0.0, color="#777777", linewidth=0.8)
    covariance_ax.set_title("カルマン共分散更新")
    covariance_ax.set_xlabel(r"位置誤差 $e_1$ [-]")
    covariance_ax.set_ylabel(r"速度誤差 $e_2$ [-/s]")
    covariance_ax.set_aspect("equal", adjustable="box")
    covariance_ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)
    covariance_ax.legend(loc="upper right")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.925))
    fig.savefig(output_path, facecolor="white")
    print(output_path)


if __name__ == "__main__":
    main()
