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

from . import constants
from .utils import (
    _log_food_group_target_summary,
    _nutrition_efficiency_factor,
    _per_capita_mass_to_mt_per_year,
    _per_capita_to_bus_units,
)

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
            values[country] = float(value)
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
    food_groups: pd.DataFrame,
    food_group_constraints: dict,
    countries: list,
    population: pd.Series,
    *,
    per_country_equal: dict[str, dict[str, float]] | None = None,
    add_slack_for_fixed_consumption: bool = False,
    slack_marginal_cost: float | None = None,
) -> None:
    """Add carriers, buses, and loads for food groups defined in the CSV.

    Supports min/max/equal per-person-per-day targets per food group. Country-level
    equality overrides can be supplied via ``per_country_equal``. When
    ``add_slack_for_fixed_consumption`` is set, groups that are fully fixed via
    equality constraints receive high-cost slack generators (cost set by
    ``slack_marginal_cost``) so validation runs stay feasible while reporting
    shortages explicitly.
    """

    per_country_equal = per_country_equal or {}
    countries_index = pd.Index(countries, dtype="object")
    population = population.astype(float)

    logger.info("Adding food group loads based on nutrition requirements...")
    for group in food_group_list:
        group_config = food_group_constraints.get(group, {}) or {}
        min_value = group_config.get("min")
        max_value = group_config.get("max")
        equal_value = group_config.get("equal")
        equal_overrides = per_country_equal.get(group, {})

        names = f"{group}_" + countries_index
        buses = f"group_{group}_" + countries_index
        carriers = [f"group_{group}"] * len(countries)

        # Build per-country equality targets (if any)
        equal_values: list[float] = []
        if equal_overrides:
            # Per-country baseline: all countries must have values
            equal_values = [equal_overrides[country] for country in countries]
        elif equal_value is not None:
            # Blanket equality: same value for all countries
            equal_values = [equal_value] * len(countries)

        # If we have equality constraints, use them and skip min/max stores
        if equal_values:
            _log_food_group_target_summary(group, equal_values)
            equal_totals = [
                _per_capita_mass_to_mt_per_year(value, float(population[country]))
                for value, country in zip(equal_values, countries)
            ]  # demand in Mt/year because group buses use Mt
            n.loads.add(names, bus=buses, carrier=carriers, p_set=equal_totals)

            if add_slack_for_fixed_consumption:
                n.carriers.add("slack_positive_group_" + group, unit="Mt")
                n.carriers.add("slack_negative_group_" + group, unit="Mt")
                n.generators.add(
                    f"slack_positive_{group}_" + countries_index,
                    bus=buses,
                    carrier=f"slack_negative_group_{group}",
                    p_nom_extendable=True,
                    marginal_cost=slack_marginal_cost,
                )
                n.stores.add(
                    f"slack_negative_{group}_" + countries_index,
                    bus=buses,
                    carrier=f"slack_negative_group_{group}",
                    e_nom_extendable=True,
                    marginal_cost=slack_marginal_cost,
                )
            # Equality constraint fixes consumption; no additional stores required
            continue

        # No equality constraints: use min/max bounds with stores
        min_totals: list[float] | None = None
        if min_value is not None and min_value > 0.0:
            min_totals = [
                _per_capita_mass_to_mt_per_year(min_value, float(population[country]))
                for country in countries
            ]
            n.loads.add(names, bus=buses, carrier=carriers, p_set=min_totals)

        max_totals: list[float] | None = None
        if max_value is not None:
            max_totals = [
                _per_capita_mass_to_mt_per_year(max_value, float(population[country]))
                for country in countries
            ]

        store_names = "store_" + names
        store_kwargs: dict[str, Iterable[float]] = {}
        if max_totals is not None:
            if min_totals is not None:
                e_nom_max = [
                    max(max_total - min_total, 0.0)
                    for max_total, min_total in zip(max_totals, min_totals)
                ]
            else:
                e_nom_max = max_totals
            store_kwargs["e_nom_max"] = e_nom_max

        n.stores.add(
            store_names,
            bus=buses,
            carrier=carriers,
            e_nom_extendable=True,
            **store_kwargs,
        )


def add_macronutrient_loads(
    n: pypsa.Network,
    all_nutrients: list,
    macronutrients_config: dict,
    countries: list,
    population: pd.Series,
    nutrient_units: dict[str, str],
) -> None:
    """Add per-country loads and stores for macronutrient tracking and bounds.

    All nutrients get extendable Stores (to absorb flows from consumption links).
    Only configured nutrients get Loads (min/equal constraints) and e_nom_max (max constraints).
    """

    logger.info("Adding macronutrient stores and constraints per country...")

    for nutrient in all_nutrients:
        unit = nutrient_units[nutrient]
        names = [f"{nutrient}_{c}" for c in countries]
        carriers = [nutrient] * len(countries)

        # Get configuration for this nutrient (if any)
        nutrient_config = macronutrients_config.get(nutrient, {}) or {}
        equal_value = nutrient_config.get("equal")
        min_value = nutrient_config.get("min")
        max_value = nutrient_config.get("max")

        # Handle equality constraint
        if equal_value is not None:
            p_set = [
                _per_capita_to_bus_units(equal_value, float(population[c]), unit)
                for c in countries
            ]
            n.loads.add(names, bus=names, carrier=carriers, p_set=p_set)
            # For equality constraints, we don't need a Store (Load fixes the flow)
            continue

        # Handle min constraint with Load
        min_totals = None
        if min_value is not None:
            min_totals = [
                _per_capita_to_bus_units(min_value, float(population[c]), unit)
                for c in countries
            ]
            n.loads.add(names, bus=names, carrier=carriers, p_set=min_totals)

        # Always add Store (to absorb consumption flows)
        # Only set e_nom_max if max constraint is configured
        store_names = [f"store_{nutrient}_{c}" for c in countries]

        e_nom_max = None
        if max_value is not None:
            max_totals = [
                _per_capita_to_bus_units(max_value, float(population[c]), unit)
                for c in countries
            ]
            if min_totals is not None:
                e_nom_max = [
                    max(max_t - min_t, 0.0)
                    for max_t, min_t in zip(max_totals, min_totals)
                ]
            else:
                e_nom_max = max_totals

        n.stores.add(
            store_names,
            bus=names,
            carrier=carriers,
            e_nom_extendable=True,
            e_cyclic=False,
            **({"e_nom_max": e_nom_max} if e_nom_max is not None else {}),
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
        eff_lists = []
        for _i, nutrient in enumerate(nutrients, start=1):
            unit = nutrient_units[nutrient]
            factor = _nutrition_efficiency_factor(unit)
            out_bus_lists.append([f"{nutrient}_{c}" for c in countries])
            eff_val = (
                float(nutrition.loc[(food, nutrient), "value"])
                if (food, nutrient) in nutrition.index
                else 0.0
            )
            eff_lists.append([eff_val * factor] * len(countries))

        # Food bus flows are Mt/year, so efficiencies below represent nutrient fractions.
        params = {"bus0": bus0, "marginal_cost": _LOW_DEFAULT_MARGINAL_COST}
        for i, (buses, effs) in enumerate(zip(out_bus_lists, eff_lists), start=1):
            params[f"bus{i}"] = buses
            params["efficiency" if i == 1 else f"efficiency{i}"] = effs

        # optional food group output as last leg
        if group_val is not None and pd.notna(group_val):
            idx = len(nutrients) + 1
            params[f"bus{idx}"] = [f"group_{group_val}_{c}" for c in countries]
            params[f"efficiency{idx}"] = 1.0

        # Add metadata attributes
        params["food"] = food
        params["country"] = countries

        n.links.add(names, p_nom_extendable=True, **params)
