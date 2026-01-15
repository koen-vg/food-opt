# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Plot global food consumption by food group per person per day."""

import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pypsa

matplotlib.use("pdf")
import matplotlib.pyplot as plt

from workflow.scripts.constants import DAYS_PER_YEAR, GRAMS_PER_MEGATONNE, PJ_TO_KCAL
from workflow.scripts.plotting.color_utils import categorical_colors
from workflow.scripts.population import get_total_population

logger = logging.getLogger(__name__)

# Enable new PyPSA components API
pypsa.options.api.new_components_api = True

# Alias for backwards compatibility with modules that import from here
KCAL_PER_PJ = PJ_TO_KCAL


def _select_snapshot(network: pypsa.Network) -> pd.Index | str:
    if "now" in network.snapshots:
        return "now"
    if len(network.snapshots) == 1:
        return network.snapshots[0]
    raise ValueError("Expected snapshot 'now' or single snapshot in solved network")


def _bus_column_to_leg(column: str) -> int | None:
    if not column.startswith("bus"):
        return None
    suffix = column[len("bus") :]
    if not suffix:
        return 0
    if suffix.isdigit():
        return int(suffix)
    return None


def _link_dispatch_at_snapshot(
    network: pypsa.Network, snapshot
) -> dict[int, pd.Series]:
    dispatch: dict[int, pd.Series] = {}
    links_dynamic = network.links.dynamic
    for attr in dir(links_dynamic):
        if not attr.startswith("p"):
            continue
        suffix = attr[1:]
        if not suffix.isdigit():
            continue
        series = getattr(links_dynamic, attr)
        if snapshot not in series.index:
            continue
        dispatch[int(suffix)] = series.loc[snapshot]
    return dispatch


def _aggregate_group_mass(
    network: pypsa.Network, snapshot, food_to_group: dict[str, str]
) -> pd.Series:
    """Aggregate consumption by food group using link attributes."""
    links = network.links.static
    consume_links = links[links["carrier"].str.startswith("consume_")]
    if consume_links.empty:
        return pd.Series(dtype=float)

    dispatch_lookup = _link_dispatch_at_snapshot(network, snapshot)
    if not dispatch_lookup:
        return pd.Series(dtype=float)

    totals: dict[str, float] = {}
    for link_name in consume_links.index:
        food = str(consume_links.at[link_name, "food"])
        group = food_to_group.get(food)
        if group is None:
            continue

        # Find which leg outputs to the group bus and get its flow
        for leg, dispatch in dispatch_lookup.items():
            value = float(dispatch.get(link_name, 0.0))
            if value == 0.0 or not np.isfinite(value):
                continue
            # Check if this leg goes to a group bus
            bus_col = f"bus{leg}" if leg > 0 else "bus0"
            bus_value = consume_links.at[link_name, bus_col]
            if isinstance(bus_value, str) and bus_value.startswith("group:"):
                totals[group] = totals.get(group, 0.0) + abs(value)
                break

    return pd.Series(totals, dtype=float)


def _available_legs(links: pd.DataFrame) -> list[int]:
    legs: set[int] = set()
    for column in links.columns:
        if not column.startswith("bus"):
            continue
        if column == "bus0":
            continue
        suffix = column[3:]
        if not suffix:
            continue
        try:
            legs.add(int(suffix))
        except ValueError:
            continue
    return sorted(legs)


def _aggregate_group_calories(
    network: pypsa.Network, snapshot, food_to_group: dict[str, str]
) -> pd.Series:
    """Aggregate calorie consumption by food group using link attributes."""
    links = network.links.static
    legs = _available_legs(links)
    if not legs:
        return pd.Series(dtype=float)

    links_dynamic = network.links.dynamic
    time_series_lookup: dict[int, pd.Series] = {}
    for leg in legs:
        attr = f"p{leg}"
        series = getattr(links_dynamic, attr, None)
        if series is None or snapshot not in series.index:
            continue
        time_series_lookup[leg] = series.loc[snapshot]

    if not time_series_lookup:
        return pd.Series(dtype=float)

    totals: dict[str, float] = {}
    consume_links = links[links["carrier"].str.startswith("consume_")]
    for link_name in consume_links.index:
        food = str(consume_links.at[link_name, "food"])
        group_name = food_to_group.get(food)
        if group_name is None:
            continue

        # Find kcal leg
        kcal_leg: int | None = None
        for leg in legs:
            column = f"bus{leg}"
            bus_value = consume_links.at[link_name, column]
            if pd.notna(bus_value) and str(bus_value).startswith("nutrient:cal:"):
                kcal_leg = leg
                break

        if kcal_leg is None:
            continue

        series = time_series_lookup.get(kcal_leg)
        if series is None:
            continue

        value = abs(float(series.get(link_name, 0.0)))
        if value > 0.0:
            totals[group_name] = totals.get(group_name, 0.0) + value

    return pd.Series(totals, dtype=float)


def _assign_colors(
    groups: list[str], overrides: dict[str, str] | None = None
) -> dict[str, str]:
    return categorical_colors(groups, overrides)


def _plot(
    mass_g_per_person_day: pd.Series,
    calories_kcal_per_person_day: pd.Series,
    output_pdf: Path,
) -> None:
    mass_g_per_person_day = mass_g_per_person_day[mass_g_per_person_day > 0]
    calories_kcal_per_person_day = calories_kcal_per_person_day[
        calories_kcal_per_person_day > 0
    ]

    ordered_groups: list[str] = []
    ordered_groups.extend(
        mass_g_per_person_day.sort_values(ascending=False).index.tolist()
    )
    for group in calories_kcal_per_person_day.sort_values(ascending=False).index:
        if group not in ordered_groups:
            ordered_groups.append(group)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    if not ordered_groups:
        for ax in axes:
            ax.text(
                0.5, 0.5, "No food group consumption data", ha="center", va="center"
            )
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_pdf, bbox_inches="tight", dpi=300)
        plt.close(fig)
        return

    group_colors = getattr(snakemake.params, "group_colors", {}) or {}
    colors = _assign_colors(ordered_groups, group_colors)

    # Mass subplot
    ax_mass = axes[0]
    bottom = 0.0
    for group in ordered_groups:
        value = float(mass_g_per_person_day.get(group, 0.0))
        if value <= 0.0:
            continue
        ax_mass.bar(0, value, bottom=bottom, color=colors[group], label=group)
        bottom += value

    ax_mass.set_xticks([0])
    ax_mass.set_xticklabels(["Mass"])
    ax_mass.set_ylabel("g/person/day")
    ax_mass.set_title("Global Food Consumption (Mass)")
    ax_mass.grid(axis="y", alpha=0.3)

    # Calories subplot
    ax_cal = axes[1]
    bottom = 0.0
    for group in ordered_groups:
        value = float(calories_kcal_per_person_day.get(group, 0.0))
        if value <= 0.0:
            continue
        ax_cal.bar(0, value, bottom=bottom, color=colors[group])
        bottom += value

    ax_cal.set_xticks([0])
    ax_cal.set_xticklabels(["Calories"])
    ax_cal.set_ylabel("kcal/person/day")
    ax_cal.set_title("Global Food Consumption (Calories)")
    ax_cal.grid(axis="y", alpha=0.3)

    handles, labels = ax_mass.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles[::-1],
            labels[::-1],
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
        )
        fig.tight_layout(rect=(0, 0, 0.85, 1))
    else:
        fig.tight_layout()

    fig.savefig(output_pdf, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    try:
        snakemake  # type: ignore[name-defined]
    except NameError as exc:  # pragma: no cover - Snakemake injects the variable
        raise RuntimeError("This script must be run from Snakemake") from exc

    network_path = snakemake.input.network  # type: ignore[attr-defined]
    food_groups_path = snakemake.input.food_groups  # type: ignore[attr-defined]
    output_pdf = Path(snakemake.output.pdf)  # type: ignore[attr-defined]
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading solved network from %s", network_path)
    network = pypsa.Network(network_path)

    # Load food->group mapping
    food_groups_df = pd.read_csv(food_groups_path)
    food_to_group = food_groups_df.set_index("food")["group"].to_dict()

    snapshot = _select_snapshot(network)
    logger.info("Using snapshot '%s' for aggregation", snapshot)

    mass = _aggregate_group_mass(network, snapshot, food_to_group)
    calories_pj = _aggregate_group_calories(network, snapshot, food_to_group)

    total_population = get_total_population(network)

    mass_per_capita = mass * GRAMS_PER_MEGATONNE / (total_population * DAYS_PER_YEAR)
    calories_per_capita = calories_pj * KCAL_PER_PJ / (total_population * DAYS_PER_YEAR)

    logger.info(
        "Found %d food groups with mass data and %d with calorie data",
        mass_per_capita[mass_per_capita > 0].shape[0],
        calories_per_capita[calories_per_capita > 0].shape[0],
    )

    _plot(mass_per_capita, calories_per_capita, output_pdf)

    logger.info("Food consumption plot saved to %s", output_pdf)


if __name__ == "__main__":
    main()
