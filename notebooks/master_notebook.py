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

    from core.cost_functions import CostFunctionSpec
    from core.dataset import extract_snapshot, load_subject_dataset
    from core.poset import KpiSpec, PosetEngine
    from core.simulation import (
        resolve_budget_schedule,
        simulate_history,
        time_to_frontier,
    )
    from core.viz import plot_allocations, plot_hasse, plot_kpi_plane

    return (
        CostFunctionSpec,
        KpiSpec,
        PosetEngine,
        extract_snapshot,
        load_subject_dataset,
        mo,
        plot_allocations,
        plot_hasse,
        plot_kpi_plane,
        project_root,
        resolve_budget_schedule,
        simulate_history,
        time_to_frontier,
        yaml,
    )


@app.cell
def _(CostFunctionSpec, KpiSpec, project_root, yaml):
    with open(project_root / "config" / "cost_settings.yaml", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    sim_settings = config["simulation_settings"]
    kpi_specs = [
        KpiSpec.from_config(name, cfg)
        for name, cfg in config["kpi_definitions"].items()
    ]
    cost_specs = [
        CostFunctionSpec.from_config(name, cfg)
        for name, cfg in config["kpi_definitions"].items()
    ]
    time_horizon = int(sim_settings["time_horizon_years"])
    random_seed = int(sim_settings["random_seed"])
    currency = str(sim_settings.get("currency", "EUR"))
    default_budget = float(sim_settings.get("annual_budget", 50000.0))
    return (
        cost_specs,
        currency,
        default_budget,
        kpi_specs,
        random_seed,
        time_horizon,
    )


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
def _(currency, default_budget, focus_dropdown, kpi_specs, mo):
    # Budget-driven improvement: the yearly step is derived from the budget
    # through the inverse cost functions, not chosen directly.
    budget_input = mo.ui.text(
        value=f"{default_budget:g}",
        label=f"Annual budget ({currency}) — one value, or comma-separated per year",
        full_width=True,
    )
    # Allocation policy: adaptive re-optimizes the split every period on the
    # signed distance metric D; manual keeps the sliders' static shares.
    allocation_policy = mo.ui.dropdown(
        ["adaptive (optimizer)", "manual weights"],
        value="adaptive (optimizer)",
        label="Allocation policy",
    )
    # One allocation slider per KPI (manual policy only; normalized to sum to 1).
    allocation_sliders = mo.ui.dictionary(
        {
            spec.name: mo.ui.slider(
                0.0,
                1.0,
                value=round(1.0 / len(kpi_specs), 2),
                step=0.01,
                label=f"Budget share — {spec.name}",
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
            budget_input,
            allocation_policy,
            allocation_sliders,
            jitter_slider,
            mo.hstack([x_kpi_dropdown, y_kpi_dropdown], justify="start"),
        ]
    )
    return (
        allocation_policy,
        allocation_sliders,
        budget_input,
        focus_subject,
        jitter_slider,
        x_kpi_dropdown,
        y_kpi_dropdown,
    )


@app.cell
def _(
    allocation_policy,
    allocation_sliders,
    budget_input,
    cost_specs,
    focus_subject,
    initial_states,
    jitter_slider,
    kpi_specs,
    mo,
    random_seed,
    simulate_history,
    time_horizon,
):
    # Parse the budget: a single value or a comma-separated per-year schedule.
    try:
        _parts = [
            float(x)
            for x in str(budget_input.value).replace(";", ",").split(",")
            if x.strip()
        ]
    except ValueError:
        _parts = []
    mo.stop(
        not _parts,
        mo.md("⚠️ **Invalid budget** — enter a number or a comma-separated list."),
    )
    annual_budget = _parts[0] if len(_parts) == 1 else _parts

    _adaptive = str(allocation_policy.value).startswith("adaptive")
    _weights = {k: float(v) for k, v in allocation_sliders.value.items()}
    mo.stop(
        not _adaptive and sum(_weights.values()) <= 0,
        mo.md("⚠️ Allocate at least one positive budget share."),
    )

    # Simulate T periods: Focus invests the budget, competitors get seeded jitter.
    history_df, allocations_df = simulate_history(
        kpi_specs=kpi_specs,
        cost_specs=cost_specs,
        initial_states=initial_states,
        focus_subject=focus_subject,
        annual_budget=annual_budget,
        horizon=time_horizon,
        seed=random_seed,
        jitter_std=float(jitter_slider.value),
        allocation_weights=None if _adaptive else _weights,
        adaptive=_adaptive,
    )
    return allocations_df, annual_budget, history_df


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


@app.cell
def _(allocations_df, currency, kpi_specs, mo, plot_allocations):
    # Planning stream: how the budget is actually deployed year by year.
    mo.ui.plotly(plot_allocations(allocations_df, kpi_specs, currency))
    return


@app.cell
def _(
    annual_budget,
    currency,
    focus_subject,
    history_df,
    kpi_specs,
    mo,
    resolve_budget_schedule,
    time_horizon,
    time_to_frontier,
):
    _reach = time_to_frontier(history_df, kpi_specs, focus_subject)
    _budgets = resolve_budget_schedule(annual_budget, time_horizon)
    if _reach is None:
        _msg = (
            f"❌ **Goal not reached.** `{focus_subject}` never enters the Pareto "
            f"frontier within {time_horizon} years, despite a total investment of "
            f"{sum(_budgets):,.0f} {currency}. Increase the budget, rebalance the "
            "allocation, or extend the horizon."
        )
    elif _reach == 0:
        _msg = (
            f"✅ **Already efficient.** `{focus_subject}` sits on the Pareto "
            "frontier at t=0 — no investment is required for this goal."
        )
    else:
        _msg = (
            f"🎯 **Goal reached in {_reach} year{'s' if _reach > 1 else ''}.** "
            f"`{focus_subject}` enters the Pareto frontier at t={_reach}, after a "
            f"cumulative investment of {sum(_budgets[:_reach]):,.0f} {currency}."
        )
    mo.md(f"### 📋 Report — time to reach the Efficient Frontier\n\n{_msg}")
    return


if __name__ == "__main__":
    app.run()
