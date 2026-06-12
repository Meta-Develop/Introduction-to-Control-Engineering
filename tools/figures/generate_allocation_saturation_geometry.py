#!/usr/bin/env python3
"""Generate control-allocation saturation geometry for the Japanese textbook."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("numpy is required to generate the allocation geometry figure") from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.font_manager as font_manager
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - figure generation requires it.
    raise SystemExit("matplotlib is required to generate the allocation geometry figure") from exc

from figure_style import finalize_figure


B_ALLOCATION = np.asarray([[1.0, 1.0], [1.0, -1.0]])
REQUESTED_COMMAND = np.asarray([2.4, 0.6])
ACTUATOR_LIMIT = 1.2
VARIATION_LIMITS = (1.5, 1.0)


def configure_matplotlib() -> None:
    """Use a Japanese-capable font when available, with deterministic styling."""

    preferred_fonts = (
        "Noto Sans CJK JP",
        "Noto Sans JP",
        "Noto Sans",
        "DejaVu Sans",
    )
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    family = [font for font in preferred_fonts if font in available_fonts]
    if "DejaVu Sans" not in family:
        family.append("DejaVu Sans")

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "font.family": family,
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
            "figure.dpi": 180,
            "savefig.dpi": 180,
        }
    )


def ordered_polygon(points: np.ndarray) -> np.ndarray:
    """Return polygon vertices ordered counterclockwise around their center."""

    center = points.mean(axis=0)
    order = np.argsort(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))
    return points[order]


def actuator_box_vertices(limit: float) -> np.ndarray:
    """Return ordered vertices of the two-actuator command box."""

    vertices = np.asarray(
        [
            [-limit, -limit],
            [limit, -limit],
            [limit, limit],
            [-limit, limit],
        ]
    )
    return ordered_polygon(vertices)


def generalized_vertices(limit: float) -> np.ndarray:
    """Map actuator box vertices into generalized force/moment space."""

    return ordered_polygon((B_ALLOCATION @ actuator_box_vertices(limit).T).T)


def solve_box_allocation(
    allocation_matrix: np.ndarray,
    requested_command: np.ndarray,
    limit: float,
) -> np.ndarray:
    """Solve min 0.5 ||B u - w||^2 subject to |u_i| <= limit for two inputs."""

    hessian = allocation_matrix.T @ allocation_matrix
    gradient = -(allocation_matrix.T @ requested_command)
    constraints = np.asarray(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ]
    )
    bounds = np.full(4, limit)

    best_u = None
    best_value = float("inf")
    for active_size in range(3):
        for active_set in combinations(range(len(bounds)), active_size):
            if active_set:
                active = constraints[list(active_set), :]
                rhs = bounds[list(active_set)]
                kkt = np.block(
                    [
                        [hessian, active.T],
                        [active, np.zeros((active_size, active_size))],
                    ]
                )
                right = np.concatenate([-gradient, rhs])
                try:
                    candidate = np.linalg.solve(kkt, right)[:2]
                except np.linalg.LinAlgError:
                    continue
            else:
                candidate = np.linalg.solve(hessian, -gradient)

            if not np.all(constraints @ candidate <= bounds + 1.0e-9):
                continue

            residual = allocation_matrix @ candidate - requested_command
            value = 0.5 * float(residual.T @ residual)
            if value < best_value:
                best_u = candidate
                best_value = value

    if best_u is None:  # pragma: no cover - fixed example data is feasible.
        raise RuntimeError("no feasible allocation found")
    return best_u


def draw_closed_polygon(axis, vertices: np.ndarray, **kwargs) -> None:
    """Draw a closed polygon line."""

    closed = np.vstack([vertices, vertices[0]])
    axis.plot(closed[:, 0], closed[:, 1], **kwargs)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "ja" / "figures" / "allocation_saturation_geometry.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    configure_matplotlib()

    unconstrained_u = np.linalg.pinv(B_ALLOCATION) @ REQUESTED_COMMAND
    constrained_u = solve_box_allocation(B_ALLOCATION, REQUESTED_COMMAND, ACTUATOR_LIMIT)
    realized_command = B_ALLOCATION @ constrained_u
    residual_command = REQUESTED_COMMAND - realized_command

    fig, (actuator_ax, command_ax) = plt.subplots(1, 2, figsize=(7.2, 3.7))

    box = actuator_box_vertices(ACTUATOR_LIMIT)
    actuator_ax.fill(box[:, 0], box[:, 1], color="#d9ecff", alpha=0.88)
    draw_closed_polygon(actuator_ax, box, color="#1f77b4", linewidth=1.7)
    actuator_ax.axvline(
        ACTUATOR_LIMIT,
        color="#d62728",
        linewidth=2.0,
    )
    actuator_ax.plot(
        unconstrained_u[0],
        unconstrained_u[1],
        marker="x",
        markersize=8,
        markeredgewidth=2.0,
        color="#111111",
    )
    actuator_ax.plot(
        constrained_u[0],
        constrained_u[1],
        marker="o",
        markersize=6,
        color="#d62728",
    )
    actuator_ax.annotate(
        r"$u^\dagger=(1.5,0.9)$",
        xy=unconstrained_u,
        xytext=(1.10, 0.43),
        arrowprops={"arrowstyle": "->", "color": "#111111", "linewidth": 0.9},
        color="#111111",
        fontsize=8,
    )
    actuator_ax.annotate(
        r"$u^*=(1.2,0.9)$",
        xy=constrained_u,
        xytext=(0.18, 1.05),
        arrowprops={"arrowstyle": "->", "color": "#d62728", "linewidth": 0.9},
        color="#d62728",
        fontsize=8,
    )
    actuator_ax.annotate(
        "",
        xy=unconstrained_u,
        xytext=constrained_u,
        arrowprops={"arrowstyle": "->", "color": "#d62728", "linewidth": 1.2, "linestyle": "--"},
    )
    actuator_ax.annotate(
        r"残り余裕 $1.2-u_2^*=0.3$",
        xy=(ACTUATOR_LIMIT, constrained_u[1]),
        xytext=(0.02, -1.02),
        arrowprops={"arrowstyle": "<->", "color": "#2ca02c", "linewidth": 1.0},
        color="#2ca02c",
        fontsize=8,
    )
    actuator_ax.plot([ACTUATOR_LIMIT, ACTUATOR_LIMIT], [constrained_u[1], ACTUATOR_LIMIT], color="#2ca02c", linewidth=1.3)
    actuator_ax.text(
        0.03,
        0.96,
        "アクチュエータ空間",
        transform=actuator_ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
    )
    actuator_ax.text(
        -1.18,
        -1.12,
        r"青: $|u_i|\leq1.2$",
        fontsize=8,
        color="#1f77b4",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
    )
    actuator_ax.text(
        1.24,
        -1.18,
        r"赤: 能動境界 $u_1=1.2$",
        rotation=90,
        ha="left",
        va="bottom",
        fontsize=8,
        color="#d62728",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
    )
    actuator_ax.set_xlabel(r"アクチュエータ $u_1$ [-]")
    actuator_ax.set_ylabel(r"アクチュエータ $u_2$ [-]")
    actuator_ax.set_xlim(-1.35, 1.65)
    actuator_ax.set_ylim(-1.35, 1.35)
    actuator_ax.set_aspect("equal", adjustable="box")
    actuator_ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)

    command_polygon = generalized_vertices(ACTUATOR_LIMIT)
    command_ax.fill(
        command_polygon[:, 0],
        command_polygon[:, 1],
        color="#d9ecff",
        alpha=0.88,
    )
    draw_closed_polygon(command_ax, command_polygon, color="#1f77b4", linewidth=1.7)

    for limit, color, linestyle in (
        (VARIATION_LIMITS[0], "#666666", "--"),
        (VARIATION_LIMITS[1], "#999999", ":"),
    ):
        variation_polygon = generalized_vertices(limit)
        draw_closed_polygon(command_ax, variation_polygon, color=color, linestyle=linestyle, linewidth=1.2)

    u2_edge = np.linspace(-ACTUATOR_LIMIT, ACTUATOR_LIMIT, 100)
    active_edge_u = np.column_stack([np.full_like(u2_edge, ACTUATOR_LIMIT), u2_edge])
    active_edge_w = (B_ALLOCATION @ active_edge_u.T).T
    command_ax.plot(
        active_edge_w[:, 0],
        active_edge_w[:, 1],
        color="#d62728",
        linewidth=2.1,
    )
    command_ax.plot(
        REQUESTED_COMMAND[0],
        REQUESTED_COMMAND[1],
        marker="x",
        markersize=8,
        markeredgewidth=2.0,
        color="#111111",
    )
    command_ax.plot(
        realized_command[0],
        realized_command[1],
        marker="o",
        markersize=6,
        color="#d62728",
    )
    command_ax.annotate(
        r"残差 $(0.3,0.3)$",
        xy=REQUESTED_COMMAND,
        xytext=(1.55, 0.55),
        arrowprops={"arrowstyle": "->", "color": "#d62728", "linewidth": 1.3},
        color="#d62728",
        fontsize=8,
    )
    command_ax.annotate(
        r"$w_{\mathrm{req}}=(2.4,0.6)$",
        xy=REQUESTED_COMMAND,
        xytext=(1.02, 1.62),
        arrowprops={"arrowstyle": "->", "color": "#111111", "linewidth": 0.9},
        color="#111111",
        fontsize=8,
    )
    command_ax.annotate(
        r"$Bu^*=(2.1,0.3)$",
        xy=realized_command,
        xytext=(0.05, -0.82),
        arrowprops={"arrowstyle": "->", "color": "#d62728", "linewidth": 0.9},
        color="#d62728",
        fontsize=8,
    )
    command_ax.annotate(
        "",
        xy=(2.4, 0.0),
        xytext=realized_command,
        arrowprops={"arrowstyle": "->", "color": "#2ca02c", "linewidth": 1.1, "linestyle": "--"},
    )
    command_ax.text(
        0.90,
        -0.18,
        r"$u_2$ 上限までの余裕",
        color="#2ca02c",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
    )
    command_ax.text(
        0.03,
        0.96,
        "一般化力・モーメント空間",
        transform=command_ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
    )
    command_ax.text(
        -2.52,
        -2.42,
        r"青: $u_{\max}=1.2$、破線: $1.5$、点線: $1.0$",
        fontsize=8,
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
    )
    command_ax.text(
        0.55,
        1.94,
        r"赤: $u_1=1.2$",
        rotation=-38,
        fontsize=8,
        color="#d62728",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
    )
    command_ax.set_xlabel(r"合力 $F=u_1+u_2$ [-]")
    command_ax.set_ylabel(r"モーメント $M=u_1-u_2$ [-]")
    command_ax.set_xlim(-2.65, 3.05)
    command_ax.set_ylim(-2.65, 2.65)
    command_ax.set_aspect("equal", adjustable="box")
    command_ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)

    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.20, top=0.985, wspace=0.24)
    finalize_figure(fig, output_path, layout="none")
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
