# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Nutrition components for the food systems model.

This module handles food groups, macronutrients, and the links that
convert foods into nutritional outputs for human consumption.
"""

from collections.abc import Iterable
import logging

import numpy as np
import pandas as pd
import pypsa

from .. import constants
from .utils import _nutrition_efficiency_factor

logger = logging.getLogger(__name__)

_LOW_DEFAULT_MARGINAL_COST = (
    0.01 * constants.USD_TO_BNUSD / constants.TONNE_TO_MEGATONNE
)


def _build_food_group_equals_from_baseline(
    diet_df: pd.DataFrame,
    countries: Iterable[str],
    groups: Iterable[str],
    *,
    baseline_age: str,
    reference_year: int | None,
) -> dict[str, dict[str, float]]:
    """Map baseline diet table to per-country equality targets for food groups."""

    df = diet_df.copy()
    df["country"] = df["country"].str.upper()
    if baseline_age:
        df = df[df["age"] == baseline_age]
    if reference_year is not None and "year" in df.columns:
        sel = df[df["year"] == reference_year]
        if sel.empty:
            raise ValueError(
                f"No baseline diet records for year {reference_year} and age '{baseline_age}'"
            )
        df = sel

    filtered = df[df["country"].isin(countries) & df["item"].isin(groups)]
    if filtered.empty:
        raise ValueError(
            "Baseline diet table is empty after filtering by countries/groups"
        )

    pivot = (
        filtered.groupby(["country", "item"])["value"].mean().unstack(fill_value=np.nan)
    )

    result: dict[str, dict[str, float]] = {}
    missing_entries: list[str] = []
    for group in groups:
        values = {}
        for country in countries:
            value = pivot.get(group, pd.Series(dtype=float)).get(country)
            if pd.isna(value):
                missing_entries.append(f"{country}:{group}")
                continue
            # Floor at 1g/person/day to avoid numerical issues with consumer
            # values when baseline intake is very small or zero.
            values[country] = max(1.0, float(value))
        if values:
            result[str(group)] = values

    if missing_entries:
        logger.warning(
            "Missing baseline diet values for %d country/group pairs (examples: %s)",
            len(missing_entries),
            ", ".join(sorted(missing_entries)[:5]),
        )

    return result


def add_food_group_buses_and_loads(
    n: pypsa.Network,
    food_group_list: list,
    countries: list,
    population: pd.Series,
    *,
    max_per_capita: dict[str, float] | None = None,
    add_slack_for_fixed_consumption: bool = False,
    slack_marginal_cost: float | None = None,
) -> None:
    """Add carriers, buses, and stores for food groups.

    Parameters
    ----------
    n
        The PyPSA network.
    food_group_list
        List of food groups to add.
    countries
        List of country ISO3 codes.
    population
        Population per country (indexed by ISO3).
    max_per_capita
        Optional per-group consumption caps in g/person/day. Applied as e_nom_max
        on stores after converting to Mt/year using country population.
    add_slack_for_fixed_consumption
        Whether to add slack generators for baseline consumption enforcement.
    slack_marginal_cost
        Marginal cost for slack generators.
    """

    countries_index = pd.Index(countries, dtype="object")

    logger.info("Adding food group stores for nutrition requirements...")
    for group in food_group_list:
        names = f"{group}_" + countries_index
        buses = f"group_{group}_" + countries_index
        carriers = f"group_{group}"

        # Compute e_nom_max from per-capita cap if specified
        # Convert g/person/day -> Mt/year: cap_g * pop * 365 / 1e12
        if max_per_capita and group in max_per_capita:
            cap_g = max_per_capita[group]
            e_nom_max_values = cap_g * population.loc[countries].values * 365 / 1e12
        else:
            e_nom_max_values = np.inf

        store_names = "store_" + names
        n.stores.add(
            store_names,
            bus=buses,
            carrier=carriers,
            e_nom_extendable=True,
            e_nom_max=e_nom_max_values,
            country=countries,
            food_group=group,
        )

        if add_slack_for_fixed_consumption:
            n.carriers.add("slack_positive_group_" + group, unit="Mt")
            n.carriers.add("slack_negative_group_" + group, unit="Mt")
            n.generators.add(
                f"slack_positive_{group}_" + countries_index,
                bus=buses,
                carrier=f"slack_positive_group_{group}",
                p_nom_extendable=True,
                marginal_cost=slack_marginal_cost,
            )
            n.generators.add(
                f"slack_negative_{group}_" + countries_index,
                bus=buses,
                carrier=f"slack_negative_group_{group}",
                p_nom_extendable=True,
                p_min_pu=-1.0,
                p_max_pu=0.0,
                marginal_cost=-slack_marginal_cost,
            )


def add_macronutrient_loads(
    n: pypsa.Network,
    all_nutrients: list,
    macronutrients_config: dict,
    countries: list,
    population: pd.Series,
    nutrient_units: dict[str, str],
) -> None:
    """Add per-country stores for macronutrient tracking.

    Each macronutrient gets an extendable Store per country; the actual
    nutritional bounds are enforced later in ``solve_model`` via explicit
    linopy constraints on the storage level. This keeps the network
    structure simple while making the constraint logic easier to follow.
    """

    logger.info("Adding macronutrient stores and constraints per country...")

    for nutrient in all_nutrients:
        names = [f"{nutrient}_{c}" for c in countries]
        carriers = nutrient

        store_names = [f"store_{nutrient}_{c}" for c in countries]

        n.stores.add(
            store_names,
            bus=names,
            carrier=carriers,
            e_nom_extendable=True,
            e_cyclic=False,
            country=countries,
            nutrient=nutrient,
        )


def add_food_nutrition_links(
    n: pypsa.Network,
    food_list: list,
    foods: pd.DataFrame,
    food_groups: pd.DataFrame,
    nutrition: pd.DataFrame,
    nutrient_units: dict[str, str],
    countries: list,
    byproduct_list: list,
) -> None:
    """Add multilinks per country for converting foods to groups and macronutrients.

    Byproduct foods (from config) are excluded from human consumption.
    """
    # Pre-index food_groups for lookup
    food_to_group = food_groups.set_index("food")["group"].to_dict()

    # Filter out byproducts from human consumption (using config list)
    byproduct_foods = set(byproduct_list)
    consumable_foods = [f for f in food_list if f not in byproduct_foods]

    if byproduct_foods:
        logger.info(
            "Excluding %d byproduct foods from human consumption: %s",
            len(byproduct_foods),
            ", ".join(sorted(byproduct_foods)),
        )

    nutrients = list(nutrition.index.get_level_values("nutrient").unique())
    for food in consumable_foods:
        group_val = food_to_group.get(food, None)
        names = [f"consume_{food}_{c}" for c in countries]
        bus0 = [f"food_{food}_{c}" for c in countries]

        # macronutrient outputs
        out_bus_lists = []
        eff_values = []
        for _i, nutrient in enumerate(nutrients, start=1):
            unit = nutrient_units[nutrient]
            factor = _nutrition_efficiency_factor(unit)
            out_bus_lists.append([f"{nutrient}_{c}" for c in countries])
            eff_val = (
                float(nutrition.loc[(food, nutrient), "value"])
                if (food, nutrient) in nutrition.index
                else 0.0
            )
            eff_values.append(eff_val * factor)

        # Food bus flows are Mt/year, so efficiencies below represent nutrient fractions.
        params = {"bus0": bus0, "marginal_cost": _LOW_DEFAULT_MARGINAL_COST}
        for i, buses in enumerate(out_bus_lists, start=1):
            params[f"bus{i}"] = buses
            eff_key = "efficiency" if i == 1 else f"efficiency{i}"
            params[eff_key] = eff_values[i - 1]

        # optional food group output as last leg
        if group_val is not None and pd.notna(group_val):
            idx = len(nutrients) + 1
            params[f"bus{idx}"] = [f"group_{group_val}_{c}" for c in countries]
            params[f"efficiency{idx}"] = 1.0

        # Add metadata attributes
        params["food"] = food
        params["country"] = countries

        n.links.add(names, p_nom_extendable=True, **params)
