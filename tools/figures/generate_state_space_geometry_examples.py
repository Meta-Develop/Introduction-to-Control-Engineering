#!/usr/bin/env python3
"""Generate state-space geometry examples for the Japanese textbook."""

from __future__ import annotations

from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("numpy is required to generate the state-space figure") from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("matplotlib is required to generate the state-space figure") from exc

from figure_style import finalize_figure


A = np.array([[0.0, 1.0], [-2.0, -3.0]])
B = np.array([[0.0], [1.0]])
C = np.array([[1.0, 0.0]])
K = np.array([[18.0, 6.0]])
L = np.array([[10.0], [10.0]])
Q = np.eye(2)
TIME = np.linspace(0.0, 2.0, 220)


def matrix_exponential_trajectory(matrix: np.ndarray, initial: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eig(matrix)
    inverse_vectors = np.linalg.inv(vectors)
    coeff = inverse_vectors @ initial
    points = []
    for time in TIME:
        exp_diag = np.diag(np.exp(values * time))
        point = vectors @ exp_diag @ coeff
        points.append(np.real_if_close(point).real)
    return np.asarray(points)


def solve_lyapunov(a_matrix: np.ndarray, q_matrix: np.ndarray) -> np.ndarray:
    size = a_matrix.shape[0]
    lhs = np.kron(np.eye(size), a_matrix.T) + np.kron(a_matrix.T, np.eye(size))
    rhs = -q_matrix.reshape(size * size, order="F")
    solution = np.linalg.solve(lhs, rhs)
    return solution.reshape((size, size), order="F")


def ellipse_points(p_matrix: np.ndarray, level: float) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, 361)
    circle = np.vstack((np.cos(angles), np.sin(angles)))
    factor = np.linalg.cholesky(p_matrix)
    return np.linalg.solve(factor.T, np.sqrt(level) * circle).T


def add_arrow_on_curve(axis, curve: np.ndarray, color: str, index: int) -> None:
    start = curve[index]
    stop = curve[min(index + 4, len(curve) - 1)]
    delta = stop - start
    axis.arrow(
        start[0],
        start[1],
        delta[0],
        delta[1],
        shape="full",
        length_includes_head=True,
        head_width=0.06,
        head_length=0.09,
        color=color,
        linewidth=0.0,
    )


def draw_vector(
    axis,
    vector,
    label: str,
    color: str,
    linestyle: str = "-",
    label_offset=(0.0, 0.0),
) -> None:
    vector = np.asarray(vector, dtype=float)
    axis.arrow(
        0.0,
        0.0,
        vector[0],
        vector[1],
        length_includes_head=True,
        head_width=0.07,
        head_length=0.11,
        linewidth=2.0,
        linestyle=linestyle,
        color=color,
    )
    axis.text(
        1.08 * vector[0] + label_offset[0],
        1.08 * vector[1] + label_offset[1],
        label,
        color=color,
        fontsize=8,
        ha="center",
        va="center",
    )


def configure_phase_axis(axis, xlabel: str, ylabel: str) -> None:
    axis.axhline(0.0, color="#777777", linewidth=0.8)
    axis.axvline(0.0, color="#777777", linewidth=0.8)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.75)
    axis.set_aspect("equal", adjustable="box")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "ja" / "figures" / "state_space_geometry_examples.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    closed_loop = A - B @ K
    observer_error = A - L @ C
    p_matrix = solve_lyapunov(closed_loop, Q)

    plt.rcParams.update(
        {
            "font.family": ["Noto Sans CJK JP", "DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "figure.dpi": 180,
            "savefig.dpi": 180,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4), constrained_layout=False)

    colors = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd")
    initials = (
        np.array([1.4, 0.6]),
        np.array([1.0, -1.0]),
        np.array([-1.2, 0.8]),
        np.array([-1.0, -0.8]),
        np.array([0.6, 1.3]),
    )

    feedback_ax = axes[0, 0]
    for initial, color in zip(initials, colors):
        curve = matrix_exponential_trajectory(closed_loop, initial)
        feedback_ax.plot(curve[:, 0], curve[:, 1], color=color, linewidth=1.8)
        feedback_ax.plot(initial[0], initial[1], marker="o", color=color, markersize=3)
        add_arrow_on_curve(feedback_ax, curve, color, 30)
    configure_phase_axis(feedback_ax, r"状態 $x_1$ [-]", r"状態 $x_2$ [-]")
    feedback_ax.set_title(r"状態フィードバック $A-BK$, $K=[18\ 6]$")
    feedback_ax.set_xlim(-2.4, 2.4)
    feedback_ax.set_ylim(-2.4, 2.4)
    feedback_ax.text(
        0.03,
        0.05,
        r"閉ループ極: $-4,-5$",
        transform=feedback_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="#333333",
    )

    directions_ax = axes[0, 1]
    controllable_1 = B[:, 0]
    controllable_2 = (A @ B)[:, 0]
    observable_1 = C[0, :]
    observable_2 = (C @ A)[0, :]
    draw_vector(
        directions_ax, controllable_1, r"$B$", "#1f77b4", label_offset=(-0.16, 0.0)
    )
    draw_vector(
        directions_ax,
        controllable_2 / np.linalg.norm(controllable_2),
        r"$AB$",
        "#1f77b4",
        "--",
    )
    draw_vector(directions_ax, observable_1, r"$C^\mathsf{T}$", "#d62728")
    draw_vector(
        directions_ax,
        observable_2,
        r"$A^\mathsf{T}C^\mathsf{T}$",
        "#d62728",
        "--",
        label_offset=(0.18, 0.08),
    )
    directions_ax.plot(
        [0.65, 0.65],
        [-1.15, 1.15],
        color="#999999",
        linestyle=":",
        linewidth=1.2,
    )
    directions_ax.text(0.72, 1.0, r"同じ $y=Cx$", fontsize=8, color="#555555")
    configure_phase_axis(directions_ax, r"状態 $x_1$ [-]", r"状態 $x_2$ [-]")
    directions_ax.set_title("可制御・可観測な方向")
    directions_ax.set_xlim(-1.35, 1.35)
    directions_ax.set_ylim(-1.35, 1.35)

    observer_ax = axes[1, 0]
    error_initials = (
        np.array([1.2, 0.5]),
        np.array([0.8, -1.0]),
        np.array([-1.1, 0.8]),
        np.array([-0.7, -0.9]),
    )
    for initial, color in zip(error_initials, colors):
        curve = matrix_exponential_trajectory(observer_error, initial)
        observer_ax.plot(curve[:, 0], curve[:, 1], color=color, linewidth=1.8)
        observer_ax.plot(initial[0], initial[1], marker="o", color=color, markersize=3)
        add_arrow_on_curve(observer_ax, curve, color, 20)
    configure_phase_axis(observer_ax, r"誤差 $e_1$ [-]", r"誤差 $e_2$ [-]")
    observer_ax.set_title(r"オブザーバ誤差 $A-LC$, $L=[10\ 10]^\mathsf{T}$")
    observer_ax.set_xlim(-1.35, 1.35)
    observer_ax.set_ylim(-1.35, 1.35)
    observer_ax.text(
        0.03,
        0.05,
        r"オブザーバ極: $-6,-7$",
        transform=observer_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="#333333",
    )

    lyapunov_ax = axes[1, 1]
    for level, linestyle in zip((0.02, 0.08, 0.18), (":", "--", "-")):
        ellipse = ellipse_points(p_matrix, level)
        lyapunov_ax.plot(
            ellipse[:, 0],
            ellipse[:, 1],
            color="#444444",
            linestyle=linestyle,
            linewidth=1.4,
            label=rf"$V={level:g}$",
        )
    for initial, color in zip(initials[:3], colors[:3]):
        curve = matrix_exponential_trajectory(closed_loop, initial)
        lyapunov_ax.plot(curve[:, 0], curve[:, 1], color=color, linewidth=1.8)
        add_arrow_on_curve(lyapunov_ax, curve, color, 30)
    configure_phase_axis(lyapunov_ax, r"状態 $x_1$ [-]", r"状態 $x_2$ [-]")
    lyapunov_ax.set_title(r"リャプノフ準位集合 $V=x^\mathsf{T}Px$")
    lyapunov_ax.set_xlim(-2.4, 2.4)
    lyapunov_ax.set_ylim(-2.4, 2.4)
    lyapunov_ax.legend(loc="upper right")

    finalize_figure(fig, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
