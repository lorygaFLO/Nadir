"""Plotly visualisations for poset structures and KPI trajectories."""

from __future__ import annotations

from typing import Sequence

import networkx as nx
import plotly.graph_objects as go
import polars as pl

from core.dataset import SUBJECT_COL, TIME_COL
from core.poset import KpiSpec, PosetEngine

FOCUS_COLOR = "#d62728"
FRONTIER_COLOR = "#2ca02c"
NEUTRAL_COLOR = "#7f7f7f"


def plot_hasse(engine: PosetEngine, focus_subject: str, t: int) -> go.Figure:
    """Interactive layered Hasse diagram: frontier on top, dominated below."""
    hasse = engine.compute_hasse_diagram()
    frontier = set(engine.get_maximal_elements())

    pos: dict[str, tuple[float, float]] = {}
    for layer, nodes in enumerate(nx.topological_generations(hasse)):
        nodes = sorted(nodes)
        for i, node in enumerate(nodes):
            pos[node] = (i - (len(nodes) - 1) / 2.0, -float(layer))

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for u, v in hasse.edges:
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]

    node_names = list(hasse.nodes)
    hover = [
        "<b>{}</b><br>{}<br>{}".format(
            node,
            "Focus (our guy)" if node == focus_subject
            else "Pareto frontier" if node in frontier
            else "Dominated",
            "<br>".join(
                f"{spec.name}: {engine.subjects[node][k]:.2f}"
                for k, spec in enumerate(engine.kpi_specs)
            ),
        )
        for node in node_names
    ]
    node_colors = [
        FOCUS_COLOR if node == focus_subject
        else FRONTIER_COLOR if node in frontier
        else NEUTRAL_COLOR
        for node in node_names
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line={"color": "#bbbbbb", "width": 1.5},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[pos[n][0] for n in node_names],
            y=[pos[n][1] for n in node_names],
            mode="markers+text",
            text=node_names,
            textposition="middle center",
            textfont={"color": "white", "size": 10},
            marker={"size": 42, "color": node_colors},
            hovertext=hover,
            hoverinfo="text",
            showlegend=False,
        )
    )
    fig.update_layout(
        title=(
            f"Interactive Hasse diagram — period t={t} "
            f"(red = Focus '{focus_subject}', green = Pareto frontier, "
            "edges point downward: upper dominates lower)"
        ),
        xaxis={"visible": False},
        yaxis={"visible": False},
        plot_bgcolor="white",
        height=550,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    return fig


def plot_kpi_plane(
    history: pl.DataFrame,
    kpi_specs: Sequence[KpiSpec],
    x_kpi: str,
    y_kpi: str,
    focus_subject: str,
    t: int,
) -> go.Figure:
    """KPI-plane scatter with trajectory trails from TIME=0 up to ``t``."""
    x_spec = next(spec for spec in kpi_specs if spec.name == x_kpi)
    y_spec = next(spec for spec in kpi_specs if spec.name == y_kpi)
    x_col, y_col = x_kpi.upper(), y_kpi.upper()

    df = history.filter(pl.col(TIME_COL) <= t)
    fig = go.Figure()
    for name in sorted(df[SUBJECT_COL].unique().to_list()):
        sub = df.filter(pl.col(SUBJECT_COL) == name).sort(TIME_COL)
        xs, ys = sub[x_col].to_list(), sub[y_col].to_list()
        is_focus = name == focus_subject
        color = FOCUS_COLOR if is_focus else NEUTRAL_COLOR
        # Trail: path travelled so far.
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line={"color": color, "width": 2.5 if is_focus else 1.0},
                opacity=1.0 if is_focus else 0.35,
                hoverinfo="skip",
                showlegend=False,
            )
        )
        # Current position at period t.
        fig.add_trace(
            go.Scatter(
                x=[xs[-1]],
                y=[ys[-1]],
                mode="markers+text",
                text=[name],
                textposition="top center",
                textfont={"size": 9},
                marker={
                    "size": 16 if is_focus else 10,
                    "color": color,
                    "symbol": "star" if is_focus else "circle",
                },
                name=f"{name} (Focus)" if is_focus else name,
                hovertext=(
                    f"<b>{name}</b> @ t={t}<br>"
                    f"{x_spec.name}: {xs[-1]:.2f}<br>"
                    f"{y_spec.name}: {ys[-1]:.2f}"
                ),
                hoverinfo="text",
                showlegend=is_focus,
            )
        )

    x_dir = "→ better" if x_spec.higher_is_better else "← better"
    y_dir = "↑ better" if y_spec.higher_is_better else "↓ better"
    fig.update_layout(
        title=f"KPI plane — trajectories up to t={t} (red star = Focus '{focus_subject}')",
        xaxis_title=f"{x_spec.name} ({x_dir})",
        yaxis_title=f"{y_spec.name} ({y_dir})",
        plot_bgcolor="white",
        height=550,
        margin={"l": 60, "r": 20, "t": 60, "b": 60},
    )
    fig.update_xaxes(gridcolor="#eeeeee")
    fig.update_yaxes(gridcolor="#eeeeee")
    return fig


def plot_allocations(
    allocations: pl.DataFrame,
    kpi_specs: Sequence[KpiSpec],
    currency: str = "EUR",
) -> go.Figure:
    """Stacked bars of yearly budget deployment per KPI (planning stream)."""
    fig = go.Figure()
    periods = allocations[TIME_COL].to_list()
    for spec in kpi_specs:
        fig.add_trace(
            go.Bar(
                x=periods,
                y=allocations[spec.name.upper()].to_list(),
                name=spec.name,
            )
        )
    fig.update_layout(
        barmode="stack",
        title=f"Planned budget deployment ({currency}/year)",
        xaxis_title="Period t",
        yaxis_title=f"Budget ({currency})",
        plot_bgcolor="white",
        height=400,
        margin={"l": 60, "r": 20, "t": 60, "b": 40},
    )
    fig.update_yaxes(gridcolor="#eeeeee")
    return fig
