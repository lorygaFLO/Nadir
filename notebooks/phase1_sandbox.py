"""Phase 1 sandbox: PosetEngine validation on synthetic data.

Run with:  marimo edit notebooks/phase1_sandbox.py
"""

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import plotly.graph_objects as go
    import yaml

    project_root = Path(str(mo.notebook_dir())).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from core.poset import KpiSpec, PosetEngine

    return KpiSpec, PosetEngine, go, mo, np, nx, plt, project_root, yaml


@app.cell
def _(KpiSpec, project_root, yaml):
    with open(project_root / "config" / "cost_settings.yaml", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    sim_settings = config["simulation_settings"]
    kpi_specs = [
        KpiSpec.from_config(name, cfg)
        for name, cfg in config["kpi_definitions"].items()
    ]
    time_horizon = int(sim_settings["time_horizon_years"])
    random_seed = int(sim_settings["random_seed"])
    return kpi_specs, random_seed, time_horizon


@app.cell
def _(kpi_specs, mo):
    n_subjects_slider = mo.ui.slider(
        4, 20, value=8, step=1, label="Number of synthetic subjects"
    )
    # One improvement-rate slider per KPI: automatically scales beyond two KPIs.
    improvement_sliders = mo.ui.dictionary(
        {
            spec.name: mo.ui.slider(
                0.0,
                0.15,
                value=0.05,
                step=0.005,
                label=f"Annual improvement rate — {spec.name}",
            )
            for spec in kpi_specs
        }
    )
    jitter_slider = mo.ui.slider(
        0.0, 0.05, value=0.01, step=0.005, label="Competitor jitter σ"
    )
    mo.vstack(
        [
            mo.md("## Phase 1 Sandbox — Poset Engine Validation"),
            n_subjects_slider,
            improvement_sliders,
            jitter_slider,
        ]
    )
    return improvement_sliders, jitter_slider, n_subjects_slider


@app.cell
def _(PosetEngine, kpi_specs, n_subjects_slider, np, random_seed):
    # Reproducible synthetic population: uniform draws per KPI.
    gen_rng = np.random.default_rng(random_seed)
    n_subjects = int(n_subjects_slider.value)
    initial_states = {
        f"S{i + 1:02d}": gen_rng.uniform(50.0, 150.0, size=len(kpi_specs))
        for i in range(n_subjects)
    }

    baseline_engine = PosetEngine(kpi_specs=kpi_specs, subjects=dict(initial_states))
    focus_subject = baseline_engine.get_most_dominated_element()
    return focus_subject, initial_states


@app.cell
def _(
    PosetEngine,
    focus_subject,
    improvement_sliders,
    initial_states,
    jitter_slider,
    kpi_specs,
    np,
    random_seed,
    time_horizon,
):
    # Simulate T periods: Focus improves linearly, competitors get seeded jitter.
    sim_rng = np.random.default_rng(random_seed)
    jitter_specs = [
        type(spec)(
            name=spec.name,
            higher_is_better=spec.higher_is_better,
            drift_noise_std=float(jitter_slider.value) or None,
            passive_decay_rate=None,
        )
        for spec in kpi_specs
    ]

    sim_engine = PosetEngine(kpi_specs=jitter_specs, subjects=dict(initial_states))
    history: list[dict[str, np.ndarray]] = [
        {name: vec.copy() for name, vec in sim_engine.subjects.items()}
    ]

    for _t in range(time_horizon):
        for _name in sorted(sim_engine.subjects):
            if _name == focus_subject:
                _current = sim_engine.subjects[_name].copy()
                for _idx, _spec in enumerate(kpi_specs):
                    _rate = float(improvement_sliders.value[_spec.name])
                    _direction = 1.0 if _spec.higher_is_better else -1.0
                    _current[_idx] *= 1.0 + _direction * _rate
                sim_engine.set_subject(_name, _current)
            else:
                sim_engine.apply_drift(_name, sim_rng)
        history.append(
            {name: vec.copy() for name, vec in sim_engine.subjects.items()}
        )
    return (history,)


@app.cell
def _(mo, time_horizon):
    period_slider = mo.ui.slider(
        0, time_horizon, value=0, step=1, label="Simulation period t"
    )
    period_slider
    return (period_slider,)


@app.cell
def _(
    PosetEngine,
    focus_subject,
    go,
    history: "list[dict[str, np.ndarray]]",
    kpi_specs,
    mo,
    nx,
    period_slider,
):
    _t = int(period_slider.value)
    _snapshot_engine = PosetEngine(kpi_specs=kpi_specs, subjects=dict(history[_t]))
    _hasse = _snapshot_engine.compute_hasse_diagram()
    _frontier = set(_snapshot_engine.get_maximal_elements())

    # Layered Hasse layout: Pareto frontier on top, dominated layers below.
    _pos: dict[str, tuple[float, float]] = {}
    for _layer, _nodes in enumerate(nx.topological_generations(_hasse)):
        _nodes = sorted(_nodes)
        for _i, _node in enumerate(_nodes):
            _pos[_node] = (_i - (len(_nodes) - 1) / 2.0, -float(_layer))

    _edge_x: list[float | None] = []
    _edge_y: list[float | None] = []
    for _u, _v in _hasse.edges:
        _edge_x += [_pos[_u][0], _pos[_v][0], None]
        _edge_y += [_pos[_u][1], _pos[_v][1], None]

    _node_names = list(_hasse.nodes)
    _hover = [
        "<b>{}</b><br>{}<br>{}".format(
            _node,
            "Focus (our guy)" if _node == focus_subject
            else "Pareto frontier" if _node in _frontier
            else "Dominated",
            "<br>".join(
                f"{_spec.name}: {history[_t][_node][_k]:.2f}"
                for _k, _spec in enumerate(kpi_specs)
            ),
        )
        for _node in _node_names
    ]
    _node_colors = [
        "#d62728" if _node == focus_subject
        else "#2ca02c" if _node in _frontier
        else "#7f7f7f"
        for _node in _node_names
    ]

    _fig = go.Figure()
    _fig.add_trace(
        go.Scatter(
            x=_edge_x,
            y=_edge_y,
            mode="lines",
            line={"color": "#bbbbbb", "width": 1.5},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    _fig.add_trace(
        go.Scatter(
            x=[_pos[_n][0] for _n in _node_names],
            y=[_pos[_n][1] for _n in _node_names],
            mode="markers+text",
            text=_node_names,
            textposition="middle center",
            textfont={"color": "white", "size": 10},
            marker={"size": 42, "color": _node_colors},
            hovertext=_hover,
            hoverinfo="text",
            showlegend=False,
        )
    )
    _fig.update_layout(
        title=(
            f"Interactive Hasse diagram — period t={_t} "
            f"(red = Focus '{focus_subject}', green = Pareto frontier, "
            "edges point downward: upper dominates lower)"
        ),
        xaxis={"visible": False},
        yaxis={"visible": False},
        plot_bgcolor="white",
        height=550,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )

    _focus_on_frontier = focus_subject in _frontier
    mo.vstack(
        [
            mo.md(
                f"**Focus subject:** `{focus_subject}` — "
                f"{'on the Efficient Frontier ✅' if _focus_on_frontier else 'still dominated'}"
            ),
            mo.ui.plotly(_fig),
        ]
    )
    return


@app.cell
def _(kpi_specs, mo):
    # Axis selectors keep the scatter usable when more than two KPIs exist.
    _names = [spec.name for spec in kpi_specs]
    x_kpi_dropdown = mo.ui.dropdown(_names, value=_names[0], label="Scatter X-axis KPI")
    y_kpi_dropdown = mo.ui.dropdown(
        _names, value=_names[1] if len(_names) > 1 else _names[0], label="Scatter Y-axis KPI"
    )
    mo.hstack([x_kpi_dropdown, y_kpi_dropdown], justify="start")
    return x_kpi_dropdown, y_kpi_dropdown


@app.cell
def _(
    focus_subject,
    go,
    history: "list[dict[str, np.ndarray]]",
    kpi_specs,
    mo,
    period_slider,
    x_kpi_dropdown,
    y_kpi_dropdown,
):
    # KPI-plane scatter: trajectory trails from t=0 up to the selected period.
    _t = int(period_slider.value)
    _kpi_names = [spec.name for spec in kpi_specs]
    _xi = _kpi_names.index(str(x_kpi_dropdown.value))
    _yi = _kpi_names.index(str(y_kpi_dropdown.value))
    _x_spec, _y_spec = kpi_specs[_xi], kpi_specs[_yi]

    _fig3 = go.Figure()
    for _name in sorted(history[0]):
        _is_focus = _name == focus_subject
        _xs = [history[_p][_name][_xi] for _p in range(_t + 1)]
        _ys = [history[_p][_name][_yi] for _p in range(_t + 1)]
        _color = "#d62728" if _is_focus else "#7f7f7f"
        # Trail: path travelled so far.
        _fig3.add_trace(
            go.Scatter(
                x=_xs,
                y=_ys,
                mode="lines",
                line={"color": _color, "width": 2.5 if _is_focus else 1.0},
                opacity=1.0 if _is_focus else 0.35,
                hoverinfo="skip",
                showlegend=False,
            )
        )
        # Current position at period t.
        _fig3.add_trace(
            go.Scatter(
                x=[_xs[-1]],
                y=[_ys[-1]],
                mode="markers+text",
                text=[_name],
                textposition="top center",
                textfont={"size": 9},
                marker={
                    "size": 16 if _is_focus else 10,
                    "color": _color,
                    "symbol": "star" if _is_focus else "circle",
                },
                name=f"{_name} (Focus)" if _is_focus else _name,
                hovertext=(
                    f"<b>{_name}</b> @ t={_t}<br>"
                    f"{_x_spec.name}: {_xs[-1]:.2f}<br>"
                    f"{_y_spec.name}: {_ys[-1]:.2f}"
                ),
                hoverinfo="text",
                showlegend=_is_focus,
            )
        )

    _x_dir = "→ better" if _x_spec.higher_is_better else "← better"
    _y_dir = "↑ better" if _y_spec.higher_is_better else "↓ better"
    _fig3.update_layout(
        title=f"KPI plane — trajectories up to t={_t} (red star = Focus '{focus_subject}')",
        xaxis_title=f"{_x_spec.name} ({_x_dir})",
        yaxis_title=f"{_y_spec.name} ({_y_dir})",
        plot_bgcolor="white",
        height=550,
        margin={"l": 60, "r": 20, "t": 60, "b": 60},
    )
    _fig3.update_xaxes(gridcolor="#eeeeee")
    _fig3.update_yaxes(gridcolor="#eeeeee")

    mo.ui.plotly(_fig3)
    return


@app.cell
def _(focus_subject, history: "list[dict[str, np.ndarray]]", kpi_specs, plt):
    # KPI trajectory panel: raw state values for all subjects across periods.
    _n = len(kpi_specs)
    _fig2, _axes = plt.subplots(1, _n, figsize=(6 * _n, 4), squeeze=False)
    _periods = list(range(len(history)))

    for _idx, _spec in enumerate(kpi_specs):
        _ax2 = _axes[0][_idx]
        for _name in sorted(history[0]):
            _series = [history[_p][_name][_idx] for _p in _periods]
            _is_focus = _name == focus_subject
            _ax2.plot(
                _periods,
                _series,
                marker="o",
                linewidth=2.5 if _is_focus else 1.0,
                color="#d62728" if _is_focus else None,
                alpha=1.0 if _is_focus else 0.45,
                label=_name if _is_focus else None,
            )
        _direction = "↑ better" if _spec.higher_is_better else "↓ better"
        _ax2.set_title(f"{_spec.name} ({_direction})")
        _ax2.set_xlabel("Period t")
        _ax2.set_ylabel("State value")
        _ax2.legend(loc="best")

    _fig2.tight_layout()
    _fig2
    return


if __name__ == "__main__":
    app.run()
