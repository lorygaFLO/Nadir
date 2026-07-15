"""Synthetic SUBJECT | TIME | <KPI...> dataset generator for Nadir notebooks.

Produces a wide CSV compatible with ``core.dataset.load_subject_dataset``
(and therefore with ``notebooks/master_notebook.py``'s dataset loader),
letting you pick how many competitor subjects to generate. Each competitor
is assigned one of a handful of trend archetypes (all KPIs up, all down,
stable, alternating, ...) cycled across the population, so the generated
data always contains a genuine mix of improving / declining / stable
competitors instead of pure random noise. Data is generated for every
period of the requested horizon (TIME = 0 .. periods), not just a handful
of periods.

KPI names and per-KPI noise (``drift_noise_std``) are read straight from
``config/cost_settings.yaml`` so the generator stays in sync with whatever
KPIs the project is currently configured for.

Examples
--------
Generate 12 competitors over the full configured horizon, using defaults
for everything else::

    python scripts/generate_dataset.py --subjects 12

Generate 5 competitors over 20 periods, with a custom output path::

    python scripts/generate_dataset.py -n 5 -T 20 -o data/small_scenario.csv

See ``python scripts/generate_dataset.py --help`` for all options.
"""

from __future__ import annotations

import argparse
import string
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import polars as pl
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "cost_settings.yaml"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "generated_subjects.csv"

# Number of competitor subjects to generate when --subjects/-n is omitted.
# Edit this directly to change the default without passing a CLI flag.
N_SUBJECTS = 45

# Each archetype maps (kpi_index, n_kpis) -> "up" | "down" | "stable", so it
# generalizes to any number of configured KPIs, not just Phase 1's two.
ARCHETYPES: dict[str, Callable[[int, int], str]] = {
    "all_up": lambda k, n: "up",
    "all_down": lambda k, n: "down",
    "stable": lambda k, n: "stable",
    "alternating_up_down": lambda k, n: "up" if k % 2 == 0 else "down",
    "alternating_down_up": lambda k, n: "down" if k % 2 == 0 else "up",
    "first_up_rest_stable": lambda k, n: "up" if k == 0 else "stable",
    "first_down_rest_stable": lambda k, n: "down" if k == 0 else "stable",
}
ARCHETYPE_NAMES = list(ARCHETYPES)

# Per-period multiplicative drift range for each raw trend direction.
TREND_RATE_RANGES: dict[str, tuple[float, float]] = {
    "up": (0.005, 0.03),
    "down": (-0.03, -0.005),
    "stable": (-0.003, 0.003),
}


def _subject_name(index: int, total: int) -> str:
    """``CompetitorA``, ``CompetitorB``, ... or zero-padded numbers past 26."""
    if total <= 26:
        return f"Competitor{string.ascii_uppercase[index]}"
    return f"Competitor{index + 1:03d}"


def _generate_subject_series(
    rng: np.random.Generator,
    kpi_names: Sequence[str],
    archetype: str,
    periods: int,
    value_range: tuple[float, float],
    noise_std: Mapping[str, float],
) -> np.ndarray:
    """Return a ``(periods + 1, n_kpis)`` array of raw KPI values."""
    n_kpis = len(kpi_names)
    trend_fn = ARCHETYPES[archetype]
    values = np.empty((periods + 1, n_kpis), dtype=float)
    values[0] = rng.uniform(value_range[0], value_range[1], size=n_kpis)

    for k, name in enumerate(kpi_names):
        trend = trend_fn(k, n_kpis)
        low, high = TREND_RATE_RANGES[trend]
        rate = rng.uniform(low, high)
        sigma = noise_std.get(name, 0.02)
        for t in range(1, periods + 1):
            shock = rng.normal(0.0, sigma)
            values[t, k] = max(values[t - 1, k] * (1.0 + rate + shock), 1e-6)

    return values


def generate_dataset(
    kpi_names: Sequence[str],
    noise_std: Mapping[str, float],
    n_subjects: int,
    periods: int,
    seed: int,
    value_range: tuple[float, float] = (50.0, 150.0),
    regions: Sequence[str] = ("EMEA", "APAC", "AMER"),
    focus_name: str = "Us",
    focus_archetype: str = "all_up",
) -> pl.DataFrame:
    """Build a ``SUBJECT | TIME | <KPI...> | REGION`` dataframe.

    ``n_subjects`` is the number of *competitors*; the focus subject
    (``focus_name``) is generated separately with ``focus_archetype``
    (defaults to improving on every KPI).
    """
    if n_subjects < 1:
        raise ValueError("n_subjects must be at least 1.")
    if not regions:
        regions = ("EMEA",)

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    def add_subject(name: str, archetype: str, region: str) -> None:
        series = _generate_subject_series(
            rng, kpi_names, archetype, periods, value_range, noise_std
        )
        for t in range(periods + 1):
            row: dict[str, object] = {"SUBJECT": name, "TIME": t}
            row.update(
                {kpi.upper(): round(float(v), 3) for kpi, v in zip(kpi_names, series[t])}
            )
            row["REGION"] = region
            rows.append(row)

    add_subject(focus_name, focus_archetype, regions[0])
    for i in range(n_subjects):
        archetype = ARCHETYPE_NAMES[i % len(ARCHETYPE_NAMES)]
        region = regions[i % len(regions)]
        add_subject(_subject_name(i, n_subjects), archetype, region)

    return pl.DataFrame(rows).sort(["TIME", "SUBJECT"])


def _load_kpi_config(config_path: Path) -> tuple[list[str], dict[str, float], int, int]:
    with open(config_path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    kpi_names = list(config["kpi_definitions"])
    noise_std = {
        name: float(cfg.get("drift_noise_std", 0.02))
        for name, cfg in config["kpi_definitions"].items()
    }
    sim_settings = config.get("simulation_settings", {})
    default_periods = int(sim_settings.get("time_horizon_periods", 100))
    default_seed = int(sim_settings.get("random_seed", 42))
    return kpi_names, noise_std, default_periods, default_seed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-n", "--subjects", type=int, default=N_SUBJECTS,
        help=f"Number of competitor subjects to generate (default: {N_SUBJECTS}).",
    )
    parser.add_argument(
        "-T", "--periods", type=int, default=None,
        help="Generate TIME = 0..periods for every subject "
        "(default: config's time_horizon_periods).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="RNG seed (default: config's random_seed).",
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH,
        help=f"Path to cost_settings.yaml (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=DEFAULT_OUTPUT_PATH,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--focus-name", type=str, default="Us",
        help="Name of the focus subject (default: Us).",
    )
    parser.add_argument(
        "--focus-archetype", type=str, default="all_up", choices=ARCHETYPE_NAMES,
        help="Trend archetype for the focus subject (default: all_up).",
    )
    parser.add_argument(
        "--min-value", type=float, default=50.0,
        help="Lower bound for each subject's initial KPI values (default: 50).",
    )
    parser.add_argument(
        "--max-value", type=float, default=150.0,
        help="Upper bound for each subject's initial KPI values (default: 150).",
    )
    parser.add_argument(
        "--regions", type=str, default="EMEA,APAC,AMER",
        help="Comma-separated region labels cycled across competitors "
        "(default: EMEA,APAC,AMER).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    kpi_names, noise_std, default_periods, default_seed = _load_kpi_config(args.config)
    periods = args.periods if args.periods is not None else default_periods
    seed = args.seed if args.seed is not None else default_seed
    regions = [r.strip() for r in args.regions.split(",") if r.strip()]

    df = generate_dataset(
        kpi_names=kpi_names,
        noise_std=noise_std,
        n_subjects=args.subjects,
        periods=periods,
        seed=seed,
        value_range=(args.min_value, args.max_value),
        regions=regions,
        focus_name=args.focus_name,
        focus_archetype=args.focus_archetype,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(args.output)

    print(
        f"Wrote {df.height} rows ({args.subjects + 1} subjects x {periods + 1} "
        f"periods, TIME 0..{periods}) to {args.output}"
    )
    print(f"Focus subject: {args.focus_name} ({args.focus_archetype})")
    print("Competitor archetypes (cycled):")
    for i in range(args.subjects):
        name = _subject_name(i, args.subjects)
        archetype = ARCHETYPE_NAMES[i % len(ARCHETYPE_NAMES)]
        print(f"  {name}: {archetype}")


if __name__ == "__main__":
    main()
