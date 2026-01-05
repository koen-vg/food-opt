# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Compare consumption and objective breakdown across consumer values scenarios.

Generates:
1. Food consumption comparison (stacked horizontal bars, g/person/day)
2. Objective composition comparison (stacked bars by cost component)
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
import seaborn as sns

try:
    from workflow.scripts.plotting.plot_food_consumption import (
        DAYS_PER_YEAR,
        GRAMS_PER_MEGATONNE,
        _aggregate_group_mass,
        _assign_colors,
        _select_snapshot,
    )
    from workflow.scripts.plotting.plot_objective_breakdown import (
        compute_system_costs,
    )
except ImportError:
    from plot_food_consumption import (  # type: ignore
        DAYS_PER_YEAR,
        GRAMS_PER_MEGATONNE,
        _aggregate_group_mass,
        _assign_colors,
        _select_snapshot,
    )
    from plot_objective_breakdown import (  # type: ignore
        compute_system_costs,
    )


logger = logging.getLogger(__name__)

SCENARIO_LABELS = {
    "baseline": "Baseline (fixed)",
    "cv": "CV only",
    "cv_H": "CV + Health",
    "cv_G": "CV + GHG",
    "cv_HG": "CV + Health + GHG",
    "no_cv": "No CV",
    "no_cv_H": "No CV + Health",
    "no_cv_G": "No CV + GHG",
    "no_cv_HG": "No CV + Health + GHG",
}

SCENARIO_PAIR_ORDER = [
    ("cv", "no_cv"),
    ("cv_H", "no_cv_H"),
    ("cv_G", "no_cv_G"),
    ("cv_HG", "no_cv_HG"),
]


def _load_population(population_path: Path) -> float:
    population_df = pd.read_csv(population_path)
    if "population" not in population_df.columns:
        raise ValueError("Population file must contain a 'population' column")
    total_population = float(population_df["population"].sum())
    if total_population <= 0.0:
        raise ValueError("Total population must be positive for per-capita conversion")
    return total_population


def _per_capita_consumption(
    n: pypsa.Network, food_to_group: dict[str, str], population_total: float
) -> pd.Series:
    """Get per-capita consumption by food group in g/person/day."""
    snapshot = _select_snapshot(n)
    mass_mt = _aggregate_group_mass(n, snapshot, food_to_group)
    mass_per_capita = mass_mt * GRAMS_PER_MEGATONNE / (population_total * DAYS_PER_YEAR)
    return mass_per_capita


def _plot_consumption_comparison(
    consumption_data: dict[str, pd.Series],
    colors: dict[str, str],
    output_path: Path,
) -> None:
    """Plot horizontal stacked bar chart of food group consumption."""
    if not consumption_data:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No consumption data", ha="center", va="center")
        ax.axis("off")
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        return

    df = pd.DataFrame(consumption_data).T.fillna(0.0)
    ordered_groups = df.sum(axis=0).sort_values(ascending=False).index.tolist()

    bar_height = 0.7
    fig_height = max(4.0, 1.0 + bar_height * len(consumption_data))
    fig, ax = plt.subplots(figsize=(10, fig_height))

    labels = list(consumption_data.keys())
    for idx, label in enumerate(labels):
        series = df.loc[label]
        left = 0.0
        for group in ordered_groups:
            value = float(series.get(group, 0.0))
            if value <= 0.0:
                continue
            ax.barh(
                idx,
                value,
                left=left,
                height=bar_height,
                color=colors.get(group, "#888888"),
            )
            left += value

    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.5, len(labels) - 0.5)
    ax.set_xlabel("g/person/day")
    ax.set_title("Food Consumption by Scenario")
    ax.grid(axis="x", alpha=0.3)

    legend_handles = [
        matplotlib.patches.Patch(color=colors.get(group, "#888888"), label=group)
        for group in ordered_groups
    ]
    if legend_handles:
        fig.legend(
            legend_handles[::-1],
            [patch.get_label() for patch in legend_handles][::-1],
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
        )
        fig.tight_layout(rect=(0, 0, 0.85, 1))
    else:
        fig.tight_layout()

    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def _plot_objective_comparison(
    objective_data: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    """Plot stacked bar chart of objective components by scenario."""
    if not objective_data:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No objective data", ha="center", va="center")
        ax.axis("off")
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        return

    # Collect all categories across scenarios
    all_categories = set()
    for costs_df in objective_data.values():
        all_categories.update(costs_df.index.tolist())

    # Build matrix
    labels = list(objective_data.keys())
    categories = sorted(all_categories)
    data = np.zeros((len(labels), len(categories)))

    for i, label in enumerate(labels):
        costs_df = objective_data[label]
        for j, cat in enumerate(categories):
            if cat in costs_df.index:
                data[i, j] = float(costs_df.loc[cat, "total"])

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))
    bar_width = 0.6

    # Use a colormap for categories
    cmap = plt.get_cmap("tab20")
    bottom_pos = np.zeros(len(labels))
    bottom_neg = np.zeros(len(labels))

    for j, cat in enumerate(categories):
        values = data[:, j]
        color = cmap(j % 20)
        # Handle positive and negative values separately for proper stacking
        pos_vals = np.maximum(values, 0)
        neg_vals = np.minimum(values, 0)
        if np.any(pos_vals > 0):
            ax.bar(x, pos_vals, bar_width, bottom=bottom_pos, label=cat, color=color)
            bottom_pos += pos_vals
        if np.any(neg_vals < 0):
            ax.bar(x, neg_vals, bar_width, bottom=bottom_neg, color=color)
            bottom_neg += neg_vals

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("bnUSD")
    ax.set_title("Objective Composition by Scenario")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)

    # Legend outside
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize="small")
    fig.tight_layout()

    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


# Unit conversion: bnUSD/Mt -> USD/kg
# 1 bnUSD = 1e9 USD, 1 Mt = 1e9 kg
# bnUSD/Mt = 1e9/1e9 USD/kg = 1
BNUSD_PER_MT_TO_USD_PER_KG = 1.0


def _prepare_consumer_values_distribution(cv_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare per-country consumer values in USD/kg for plotting."""
    prepared = cv_df.copy()
    prepared["value_usd_per_kg"] = (
        prepared["value_bnusd_per_mt"] * BNUSD_PER_MT_TO_USD_PER_KG
    )
    return prepared


def _plot_consumer_values_distribution(
    cv_df: pd.DataFrame,
    colors: dict[str, str],
    output_path: Path,
) -> None:
    """Plot letter-value distribution of consumer values by food group."""
    if cv_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No consumer values data", ha="center", va="center")
        ax.axis("off")
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        return

    grouped = cv_df.groupby("group")["value_usd_per_kg"]
    group_order = (
        grouped.median().sort_values(ascending=False).index.astype(str).tolist()
    )

    fig_height = max(6.0, 0.4 * len(group_order) + 3.0)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    # sns.boxenplot(
    sns.stripplot(
        data=cv_df,
        x="group",
        y="value_usd_per_kg",
        order=group_order,
        hue="group",
        palette=colors,
        legend=False,
        linewidth=0.8,
        ax=ax,
    )

    ax.set_xticks(np.arange(len(group_order)))
    ax.set_xticklabels(group_order, rotation=45, ha="right")
    ax.set_ylabel("USD / kg")
    ax.set_title("Consumer Values by Food Group (Across Countries)")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def _write_consumer_values_csv(cv_df: pd.DataFrame, output_path: Path) -> None:
    """Write per-country consumer values data to CSV."""
    # Per-country values
    records = []
    for _, row in cv_df.iterrows():
        records.append(
            {
                "country": row["country"],
                "group": row["group"],
                "value_bnusd_per_mt": float(row["value_bnusd_per_mt"]),
                "value_usd_per_kg": float(row["value_bnusd_per_mt"])
                * BNUSD_PER_MT_TO_USD_PER_KG,
            }
        )

    pd.DataFrame.from_records(records).to_csv(output_path, index=False)


def _write_consumption_csv(
    consumption_data: dict[str, pd.Series], output_path: Path
) -> None:
    """Write consumption data to CSV."""
    records = []
    for label, series in consumption_data.items():
        for group, value in series.items():
            records.append(
                {
                    "scenario": label,
                    "group": group,
                    "g_per_person_per_day": float(value),
                }
            )
    pd.DataFrame.from_records(records).to_csv(output_path, index=False)


def _write_objective_csv(
    objective_data: dict[str, pd.DataFrame], output_path: Path
) -> None:
    """Write objective breakdown data to CSV."""
    records = []
    for label, costs_df in objective_data.items():
        for cat in costs_df.index:
            records.append(
                {
                    "scenario": label,
                    "category": cat,
                    "capex_bnusd": float(costs_df.loc[cat, "capex"]),
                    "opex_bnusd": float(costs_df.loc[cat, "opex"]),
                    "total_bnusd": float(costs_df.loc[cat, "total"]),
                }
            )
    pd.DataFrame.from_records(records).to_csv(output_path, index=False)


def main() -> None:
    try:
        snakemake  # type: ignore[name-defined]
    except NameError as exc:
        raise RuntimeError("This script must be run from Snakemake") from exc

    logging.basicConfig(level=logging.INFO)

    scenarios = list(snakemake.params.scenarios)
    population_path = Path(snakemake.input.population)
    food_groups_path = Path(snakemake.input.food_groups)

    # Create output directories
    Path(snakemake.output.consumption_pdf).parent.mkdir(parents=True, exist_ok=True)

    # Load food group mapping
    food_groups_df = pd.read_csv(food_groups_path)
    food_to_group = food_groups_df.set_index("food")["group"].to_dict()

    # Load population
    population_total = _load_population(population_path)
    logger.info("Total population: %.3e", population_total)

    # Load networks and compute metrics
    consumption_data = {}
    objective_data = {}

    ordered_scenarios: list[str] = []
    if "baseline" in scenarios:
        ordered_scenarios.append("baseline")
    for cv_scen, no_cv_scen in SCENARIO_PAIR_ORDER:
        if cv_scen in scenarios:
            ordered_scenarios.append(cv_scen)
        if no_cv_scen in scenarios:
            ordered_scenarios.append(no_cv_scen)
    for scen in scenarios:
        if scen not in ordered_scenarios:
            ordered_scenarios.append(scen)

    for scen in ordered_scenarios:
        input_key = f"network_{scen}"
        network_path = getattr(snakemake.input, input_key, None)
        if network_path is None:
            logger.warning("Network input not found for scenario: %s", scen)
            continue

        logger.info("Loading network for scenario: %s", scen)
        n = pypsa.Network(network_path)

        label = SCENARIO_LABELS.get(scen, scen)

        # Compute consumption
        consumption = _per_capita_consumption(n, food_to_group, population_total)
        consumption_data[label] = consumption

        # Compute objective breakdown
        costs_df = compute_system_costs(n)
        objective_data[label] = costs_df

    # Assign colors to food groups
    all_groups = set()
    for series in consumption_data.values():
        all_groups.update(series.index.tolist())
    group_colors = _assign_colors(
        sorted(all_groups),
        overrides=snakemake.params.group_colors or None,
    )

    # Generate scenario comparison plots
    _plot_consumption_comparison(
        consumption_data,
        group_colors,
        Path(snakemake.output.consumption_pdf),
    )
    _plot_objective_comparison(
        objective_data,
        Path(snakemake.output.objective_pdf),
    )

    # Write scenario comparison CSVs
    _write_consumption_csv(
        consumption_data,
        Path(snakemake.output.consumption_csv),
    )
    _write_objective_csv(
        objective_data,
        Path(snakemake.output.objective_csv),
    )

    # Load and visualize consumer values
    cv_df = pd.read_csv(snakemake.input.consumer_values)
    cv_df["country"] = cv_df["country"].astype(str).str.upper()
    cv_prepared = _prepare_consumer_values_distribution(cv_df)
    logger.info(
        "Prepared consumer values distribution for %d food groups",
        cv_prepared["group"].nunique(),
    )

    # Include food groups from consumer values in colors
    for group in cv_prepared["group"].unique():
        if group not in group_colors:
            group_colors[group] = "#888888"

    _plot_consumer_values_distribution(
        cv_prepared,
        group_colors,
        Path(snakemake.output.cv_pdf),
    )
    _write_consumer_values_csv(cv_df, Path(snakemake.output.cv_csv))

    logger.info("Consumer values comparison plots saved")


if __name__ == "__main__":
    main()
