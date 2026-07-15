"""Cost-function gallery: inspect every registered cost function.

Each function's parameters are shown immediately above its plotted curve.

Run with:  marimo edit notebooks/cost_functions_sandbox.py
"""

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go

    project_root = Path(str(mo.notebook_dir())).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from core.cost_functions import COST_FUNCTION_FORMULAS, COST_FUNCTIONS

    return COST_FUNCTIONS, COST_FUNCTION_FORMULAS, go, mo, np


@app.cell
def _(mo):
    mo.md("""
    ## Cost-Function Gallery
    """)
    return


@app.cell
def _(COST_FUNCTIONS, mo):
    # One alpha slider per registered cost function, plus a shared Delta%
    # range slider controlling the plotted x-axis span.
    alpha_sliders = mo.ui.dictionary(
        {
            name: mo.ui.slider(
                100_000,
                3_000_000,
                value=800_000,
                step=50_000,
                label=f"alpha — {name}",
            )
            for name in COST_FUNCTIONS
        }
    )
    delta_range_slider = mo.ui.slider(
        0.05, 1.0, value=0.3, step=0.05, label="Delta% plot range (+/- around zero)"
    )
    return alpha_sliders, delta_range_slider


@app.cell
def _(
    COST_FUNCTIONS,
    COST_FUNCTION_FORMULAS,
    alpha_sliders,
    delta_range_slider,
    go,
    mo,
    np,
):
    # For every cost function: its parameters (alpha slider) rendered
    # immediately above its plotted curve.
    delta_max = float(delta_range_slider.value)
    delta_pct = np.linspace(-delta_max, delta_max, 400)

    sections = []
    for _name, _fn in COST_FUNCTIONS.items():
        _alpha = float(alpha_sliders.value[_name])
        _cost = np.array([_fn(float(d), _alpha) for d in delta_pct])

        _fig = go.Figure()
        _fig.add_trace(
            go.Scatter(x=delta_pct * 100, y=_cost, mode="lines", name=_name)
        )
        _fig.update_layout(
            title=f"{_name.capitalize()} cost function",
            xaxis_title="Delta% (percentage variance)",
            yaxis_title="Cost (currency/period)",
            height=350,
            margin={"t": 60, "b": 40},
        )

        sections.append(
            mo.vstack(
                [
                    mo.md(f"### {_name.capitalize()}"),
                    mo.md(f"$${COST_FUNCTION_FORMULAS[_name]}$$"),
                    alpha_sliders[_name],
                    mo.ui.plotly(_fig),
                ]
            )
        )

    mo.vstack(sections)
    return


if __name__ == "__main__":
    app.run()
