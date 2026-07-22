"""Generate the two illustrative PNGs used in the project README.

This is a presentation script, not a pipeline entry point: the data below is
hand-crafted (not loaded from ``data/``) and uses two generic, unnamed KPIs
(``kpi1``, ``kpi2``) purely to make the two core ideas of Nadir visually
obvious at a glance, without tying the illustration to any one domain:

1. ``docs/assets/hasse_diagram.png``
   A small Hasse diagram (order-theoretic view): the Pareto frontier
   (non-dominated subjects) sits on top, dominated subjects are layered
   below by how many others beat them on every KPI.

2. ``docs/assets/trajectory_plot.png``
   The same subjects on the KPI plane (geometric view): both KPIs are
   "higher is better" so the Pareto frontier is the visible upper-right
   staircase, and "Us" is shown drifting toward it over time.

Run with:

    conda activate nadir
    python scripts/generate_readme_assets.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from core.poset import KpiSpec, PosetEngine

ASSETS_DIR = Path(__file__).resolve().parent.parent / "docs" / "assets"

FOCUS = "Us"

FOCUS_COLOR = "#d62728"
FRONTIER_COLOR = "#2ca02c"
DOMINATED_COLOR = "#9a9a9a"
EDGE_COLOR = "#c7c7c7"
GRID_COLOR = "#e6e6e6"

X_KPI = "kpi1"
Y_KPI = "kpi2"
X_LABEL = "KPI 1  \u2192 better"
Y_LABEL = "KPI 2  \u2192 better"

KPI_SPECS = [
    KpiSpec(name=X_KPI, higher_is_better=True),
    KpiSpec(name=Y_KPI, higher_is_better=True),
]

# Final-period KPI state for every subject. Both axes are "higher is
# better", so the Pareto frontier is whoever isn't beaten on *both* axes
# at once: Atlas, Orion, Vega and Us (each wins on a different trade-off).
FINAL_POSITIONS: dict[str, tuple[float, float]] = {
    FOCUS: (8.5, 7.0),
    "Orion": (7.0, 8.5),
    "Vega": (9.2, 5.5),
    "Atlas": (5.0, 9.5),
    "Nimbus": (6.0, 6.0),
    "Draco": (4.0, 4.0),
    "Lyra": (3.0, 7.0),
    "Phoenix": (7.5, 4.0),
    "Titan": (2.0, 2.0),
    "Rigel": (5.5, 5.5),
    "Sirius": (4.5, 6.5),
    "Zephyr": (6.5, 3.0),
}

# Us starts deep in dominated territory and closes the gap to the
# frontier period by period (a stand-in for a budget-optimized trajectory).
FOCUS_TRAJECTORY: list[tuple[float, float]] = [
    (2.0, 2.0),
    (3.2, 3.0),
    (4.5, 4.2),
    (5.8, 5.0),
    (6.8, 5.8),
    (7.8, 6.5),
    (8.5, 7.0),
]


def build_engine() -> PosetEngine:
    return PosetEngine(kpi_specs=KPI_SPECS, subjects=dict(FINAL_POSITIONS))


def staircase_path(
    points: list[tuple[float, float]], x_min: float, y_min: float
) -> tuple[list[float], list[float]]:
    """Upper-right step envelope through Pareto-optimal points.

    ``points`` must be sorted by ascending x (and therefore descending y,
    since none of them dominates another). The line stays flat at a point's
    y-level up to that point's x, then drops straight down (at that same x)
    to the next point's y-level — going further right at the higher y would
    only reach dominated territory. After the last (rightmost) point it
    drops straight down to ``y_min`` instead of continuing horizontally,
    since no point in the set constrains that region either.
    """
    xs: list[float] = [x_min]
    ys: list[float] = [points[0][1]]
    for i, (px, py) in enumerate(points):
        xs.append(px)
        ys.append(ys[-1])
        next_y = points[i + 1][1] if i + 1 < len(points) else y_min
        xs.append(px)
        ys.append(next_y)
    return xs, ys


def plot_hasse(engine: PosetEngine) -> None:
    hasse = engine.compute_hasse_diagram()
    frontier = set(engine.get_maximal_elements())

    # Layer 0 = frontier, layer 1 = beaten by the frontier only, etc.
    layers = list(nx.topological_generations(hasse))
    pos: dict[str, tuple[float, float]] = {}
    for depth, nodes in enumerate(layers):
        ordered = sorted(nodes, key=lambda n: -sum(engine.subjects[n]))
        width = len(ordered)
        for i, node in enumerate(ordered):
            pos[node] = (i - (width - 1) / 2.0, -float(depth))

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for u, v in hasse.edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.plot([x0, x1], [y0, y1], color=EDGE_COLOR, linewidth=1.4, zorder=1)

    for node, (x, y) in pos.items():
        if node == FOCUS:
            color, size, fontsize, fontcolor = FOCUS_COLOR, 1900, 10, "white"
        elif node in frontier:
            color, size, fontsize, fontcolor = FRONTIER_COLOR, 1500, 9, "white"
        else:
            color, size, fontsize, fontcolor = DOMINATED_COLOR, 1300, 8.5, "white"
        ax.scatter([x], [y], s=size, color=color, zorder=2, edgecolors="white", linewidths=1.5)
        ax.text(x, y, node, ha="center", va="center", color=fontcolor,
                fontsize=fontsize, fontweight="bold", zorder=3)

    ax.set_title(
        "Poset of Pareto dominance \u2014 frontier on top, dominated below",
        fontsize=15, fontweight="bold", pad=18, color="#222222",
    )
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=FOCUS_COLOR,
                   markersize=14, label="Us (focus)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=FRONTIER_COLOR,
                   markersize=14, label="Pareto frontier"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=DOMINATED_COLOR,
                   markersize=14, label="Dominated"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=10)
    ax.set_xlim(min(x for x, _ in pos.values()) - 1, max(x for x, _ in pos.values()) + 1)
    ax.set_ylim(min(y for _, y in pos.values()) - 0.8, max(y for _, y in pos.values()) + 0.8)
    ax.axis("off")
    fig.tight_layout()
    out = ASSETS_DIR / "hasse_diagram.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def plot_trajectory(engine: PosetEngine) -> None:
    frontier_names = sorted(engine.get_maximal_elements(), key=lambda n: engine.subjects[n][0])
    frontier_points = [tuple(engine.subjects[n]) for n in frontier_names]

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(True, color=GRID_COLOR, linewidth=1, zorder=0)
    ax.set_axisbelow(True)

    x_min, x_max = 0.0, 10.5
    y_min, y_max = 0.0, 10.5
    xs, ys = staircase_path(frontier_points, x_min, y_min)
    ax.fill_between(xs, ys, y_min, step=None, color=FRONTIER_COLOR, alpha=0.08, zorder=1)
    ax.plot(xs, ys, color=FRONTIER_COLOR, linewidth=2.2, linestyle="--",
            zorder=2, label="Pareto frontier")

    for name, (x, y) in FINAL_POSITIONS.items():
        if name == FOCUS:
            continue
        is_frontier = name in frontier_names
        color = FRONTIER_COLOR if is_frontier else DOMINATED_COLOR
        size = 220 if is_frontier else 170
        ax.scatter([x], [y], s=size, color=color, edgecolors="white",
                   linewidths=1.8, zorder=5)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8.5,
                    color=FRONTIER_COLOR if is_frontier else "#444444",
                    fontweight="bold" if is_frontier else "normal")

    traj_x = [p[0] for p in FOCUS_TRAJECTORY]
    traj_y = [p[1] for p in FOCUS_TRAJECTORY]
    ax.plot(traj_x, traj_y, color=FOCUS_COLOR, linewidth=2.6, zorder=4,
            marker="o", markersize=5, markerfacecolor="white", markeredgewidth=1.6)
    ax.annotate(
        "", xy=(traj_x[-1], traj_y[-1]), xytext=(traj_x[-2], traj_y[-2]),
        arrowprops={"arrowstyle": "-|>", "color": FOCUS_COLOR, "lw": 2.6},
        zorder=4,
    )
    ax.scatter([traj_x[-1]], [traj_y[-1]], marker="*", s=650, color=FOCUS_COLOR,
               edgecolors="white", linewidths=1.2, zorder=5, label="Us (focus)")
    ax.annotate(f"{FOCUS}  (closing the gap)", (traj_x[-1], traj_y[-1]),
                textcoords="offset points", xytext=(12, 4), ha="left",
                fontsize=10.5, fontweight="bold", color=FOCUS_COLOR)
    ax.annotate("t = 0", (traj_x[0], traj_y[0]), textcoords="offset points",
                xytext=(-8, -14), ha="right", fontsize=8.5, color=FOCUS_COLOR)

    ax.set_title(
        "KPI plane \u2014 both axes improve upward/rightward: the frontier is the top-right edge",
        fontsize=13.5, fontweight="bold", pad=16, color="#222222",
    )
    ax.set_xlabel(X_LABEL, fontsize=11)
    ax.set_ylabel(Y_LABEL, fontsize=11)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    fig.tight_layout()
    out = ASSETS_DIR / "trajectory_plot.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    engine = build_engine()
    plot_hasse(engine)
    plot_trajectory(engine)


if __name__ == "__main__":
    main()
