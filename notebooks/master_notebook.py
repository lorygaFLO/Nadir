"""Phase 1 user data: PosetEngine simulation on a user-provided dataset.

Loads a CSV/Parquet file with columns SUBJECT | TIME | <KPI...> (uppercase,
matching the YAML kpi_definitions). The latest observed TIME becomes the
simulation starting point (t=0).

Run with:  marimo edit notebooks/phase1_user_data.py
"""

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import yaml

    project_root = Path(str(mo.notebook_dir())).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from core.dataset import extract_snapshot, load_subject_dataset
    from core.poset import KpiSpec, PosetEngine
    from core.simulation import simulate_history
    from core.viz import plot_hasse, plot_kpi_plane

    return (
        KpiSpec,
        PosetEngine,
        extract_snapshot,
        load_subject_dataset,
        mo,
        plot_hasse,
        plot_kpi_plane,
        project_root,
        simulate_history,
        yaml,
    )


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
def _(mo, project_root):
    dataset_path_input = mo.ui.text(
        value=str(project_root / "data" / "example_subjects.csv"),
        label="Dataset path (.csv or .parquet)",
        full_width=True,
    )
    mo.vstack(
        [
            mo.md("## Phase 1 — Simulation on User Data"),
            dataset_path_input,
        ]
    )
    return (dataset_path_input,)


@app.cell
def _(dataset_path_input, kpi_specs, load_subject_dataset, mo):
    _error = None
    dataset_df = None
    try:
        dataset_df = load_subject_dataset(str(dataset_path_input.value), kpi_specs)
    except (OSError, ValueError) as exc:
        _error = str(exc)

    mo.stop(
        dataset_df is None,
        mo.md(f"⚠️ **Cannot load dataset:** {_error}"),
    )
    mo.vstack(
        [
            mo.md(
                f"Loaded **{dataset_df['SUBJECT'].n_unique()} subjects** × "
                f"**{dataset_df['TIME'].n_unique()} periods** "
                f"(latest observed TIME = {dataset_df['TIME'].max()})."
            ),
            mo.ui.table(dataset_df, selection=None, page_size=8),
        ]
    )
    return (dataset_df,)


@app.cell
def _(PosetEngine, dataset_df, extract_snapshot, kpi_specs, mo):
    # Simulation starts from the latest observed snapshot of the user data.
    initial_states = extract_snapshot(dataset_df, kpi_specs)

    _baseline = PosetEngine(kpi_specs=kpi_specs, subjects=dict(initial_states))
    focus_dropdown = mo.ui.dropdown(
        sorted(initial_states),
        value=_baseline.get_most_dominated_element(),
        label="Focus subject (our guy; default = most dominated)",
    )
    focus_dropdown
    return focus_dropdown, initial_states


@app.cell
def _(focus_dropdown, kpi_specs, mo):
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
    # Axis selectors keep the scatter usable when more than two KPIs exist.
    _names = [spec.name for spec in kpi_specs]
    x_kpi_dropdown = mo.ui.dropdown(_names, value=_names[0], label="Scatter X-axis KPI")
    y_kpi_dropdown = mo.ui.dropdown(
        _names,
        value=_names[1] if len(_names) > 1 else _names[0],
        label="Scatter Y-axis KPI",
    )
    focus_subject = str(focus_dropdown.value)
    mo.vstack(
        [
            improvement_sliders,
            jitter_slider,
            mo.hstack([x_kpi_dropdown, y_kpi_dropdown], justify="start"),
        ]
    )
    return (
        focus_subject,
        improvement_sliders,
        jitter_slider,
        x_kpi_dropdown,
        y_kpi_dropdown,
    )


@app.cell
def _(
    focus_subject,
    improvement_sliders,
    initial_states,
    jitter_slider,
    kpi_specs,
    random_seed,
    simulate_history,
    time_horizon,
):
    # Simulate T periods: Focus improves linearly, competitors get seeded jitter.
    history_df = simulate_history(
        kpi_specs=kpi_specs,
        initial_states=initial_states,
        focus_subject=focus_subject,
        improvement_rates={k: float(v) for k, v in improvement_sliders.value.items()},
        jitter_std=float(jitter_slider.value),
        horizon=time_horizon,
        seed=random_seed,
    )
    return (history_df,)


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
    extract_snapshot,
    focus_subject,
    history_df,
    kpi_specs,
    mo,
    period_slider,
    plot_hasse,
):
    _t = int(period_slider.value)
    _engine = PosetEngine(
        kpi_specs=kpi_specs,
        subjects=extract_snapshot(history_df, kpi_specs, time=_t),
    )
    _on_frontier = focus_subject in set(_engine.get_maximal_elements())
    mo.vstack(
        [
            mo.md(
                f"**Focus subject:** `{focus_subject}` — "
                f"{'on the Efficient Frontier ✅' if _on_frontier else 'still dominated'}"
            ),
            mo.ui.plotly(plot_hasse(_engine, focus_subject, _t)),
        ]
    )
    return


@app.cell
def _(
    focus_subject,
    history_df,
    kpi_specs,
    mo,
    period_slider,
    plot_kpi_plane,
    x_kpi_dropdown,
    y_kpi_dropdown,
):
    mo.ui.plotly(
        plot_kpi_plane(
            history_df,
            kpi_specs,
            x_kpi=str(x_kpi_dropdown.value),
            y_kpi=str(y_kpi_dropdown.value),
            focus_subject=focus_subject,
            t=int(period_slider.value),
        )
    )
    return


if __name__ == "__main__":
    app.run()
