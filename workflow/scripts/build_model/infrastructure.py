# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Infrastructure setup for the food systems model.

This module handles the creation of carriers and buses that form the
foundation of the PyPSA network model.
"""

import pypsa

from .. import constants
from .utils import _carrier_unit_for_nutrient, _nutrient_kind


def add_carriers_and_buses(
    n: pypsa.Network,
    crop_list: list,
    food_list: list,
    residue_feed_items: list,
    food_group_list: list,
    nutrient_list: list,
    nutrient_units: dict[str, str],
    countries: list,
    regions: list,
    water_regions: list,
) -> None:
    """Add all carriers and their corresponding buses to the network.

    - Regional land buses remain per-region.
    - Crops, residues, foods, food groups, and macronutrients are created per-country.
    - Primary resources (water) and emissions (co2, ch4, n2o) use global buses.
    - Fertilizer has a global supply bus with per-country delivery buses.

    Bus names use ':' as delimiter: {type}:{specifier}:{scope}
    All buses have 'country' and 'region' columns (NaN when not applicable).
    """
    # Land carrier (class-level buses are added later)
    n.carriers.add("land", unit="Mha")

    # Crops per country
    crop_buses = [
        f"crop:{crop}:{country}" for country in countries for crop in crop_list
    ]
    crop_carriers = [f"crop_{crop}" for country in countries for crop in crop_list]
    crop_countries = [country for country in countries for _ in crop_list]
    if crop_buses:
        n.carriers.add(sorted({f"crop_{crop}" for crop in crop_list}), unit="Mt")
        n.buses.add(crop_buses, carrier=crop_carriers, country=crop_countries)

    # Residues per country
    residue_items_sorted = sorted(dict.fromkeys(residue_feed_items))
    if residue_items_sorted:
        residue_buses = [
            f"residue:{item}:{country}"
            for country in countries
            for item in residue_items_sorted
        ]
        residue_carriers = [
            f"residue_{item}" for country in countries for item in residue_items_sorted
        ]
        residue_countries = [
            country for country in countries for _ in residue_items_sorted
        ]
        n.carriers.add(sorted(set(residue_carriers)), unit="Mt")
        n.buses.add(residue_buses, carrier=residue_carriers, country=residue_countries)

    # Foods per country
    food_buses = [
        f"food:{food}:{country}" for country in countries for food in food_list
    ]
    food_carriers = [f"food_{food}" for country in countries for food in food_list]
    food_countries = [country for country in countries for _ in food_list]
    if food_buses:
        n.carriers.add(sorted({f"food_{food}" for food in food_list}), unit="Mt")
        n.buses.add(food_buses, carrier=food_carriers, country=food_countries)

    # Food groups per country
    group_buses = [
        f"group:{group}:{country}" for country in countries for group in food_group_list
    ]
    group_carriers = [
        f"group_{group}" for country in countries for group in food_group_list
    ]
    group_countries = [country for country in countries for _ in food_group_list]
    if group_buses:
        n.carriers.add(
            sorted({f"group_{group}" for group in food_group_list}),
            unit="Mt",
        )
        n.buses.add(group_buses, carrier=group_carriers, country=group_countries)

    # Macronutrients per country
    nutrient_list_sorted = sorted(dict.fromkeys(nutrient_list))
    for nutrient in nutrient_list_sorted:
        unit = nutrient_units[nutrient]
        carrier_unit = _carrier_unit_for_nutrient(unit)
        if nutrient not in n.carriers.static.index:
            n.carriers.add(nutrient, unit=carrier_unit)

    if nutrient_list_sorted:
        nutrient_buses = [
            f"nutrient:{nut}:{country}"
            for country in countries
            for nut in nutrient_list_sorted
        ]
        nutrient_carriers = [
            nut for country in countries for nut in nutrient_list_sorted
        ]
        nutrient_countries = [
            country for country in countries for _ in nutrient_list_sorted
        ]
        n.buses.add(
            nutrient_buses, carrier=nutrient_carriers, country=nutrient_countries
        )

        scale_meta = n.meta.setdefault("carrier_unit_scale", {})
        if any(
            _nutrient_kind(nutrient_units[nut]) == "energy"
            for nut in nutrient_list_sorted
        ):
            scale_meta["macronutrient_kcal_to_PJ"] = constants.KCAL_TO_PJ

    # Feed carriers per country (9 pools: 5 ruminant + 4 monogastric quality classes)
    feed_categories = [
        "ruminant_grassland",
        "ruminant_roughage",
        "ruminant_forage",
        "ruminant_grain",
        "ruminant_protein",
        "monogastric_low_quality",
        "monogastric_grain",
        "monogastric_energy",
        "monogastric_protein",
    ]
    feed_buses = [
        f"feed:{fc}:{country}" for country in countries for fc in feed_categories
    ]
    feed_carriers = [f"feed_{fc}" for country in countries for fc in feed_categories]
    feed_countries = [country for country in countries for _ in feed_categories]
    if feed_buses:
        n.carriers.add(sorted(set(feed_carriers)), unit="Mt")
        n.buses.add(feed_buses, carrier=feed_carriers, country=feed_countries)

    n.carriers.add("convert_to_feed", unit="Mt")

    # Water carrier (buses added per region below)
    n.carriers.add("water", unit="Mm^3")

    # Global emission and resource carriers with buses
    for carrier, unit in [
        ("fertilizer", "Mt"),
        ("co2", "MtCO2"),
        ("ch4", "tCH4"),
        ("n2o", "tN2O"),
        ("ghg", "MtCO2e"),
    ]:
        n.carriers.add(carrier, unit=unit)
    # Add global emission buses (no country)
    n.buses.add("emission:co2", carrier="co2")
    n.buses.add("emission:ch4", carrier="ch4")
    n.buses.add("emission:n2o", carrier="n2o")
    n.buses.add("emission:ghg", carrier="ghg")
    # Global fertilizer supply bus
    n.buses.add("fertilizer:supply", carrier="fertilizer")

    # Per-country fertilizer buses
    fert_country_buses = [f"fertilizer:{country}" for country in countries]
    n.buses.add(
        fert_country_buses,
        carrier="fertilizer",
        country=countries,
    )

    scale_meta = n.meta.setdefault("carrier_unit_scale", {})
    scale_meta["co2_t_to_Mt"] = constants.TONNE_TO_MEGATONNE
    scale_meta["ch4_t_to_Mt"] = constants.TONNE_TO_MEGATONNE
    scale_meta["ghg_t_to_Mt"] = constants.TONNE_TO_MEGATONNE
    scale_meta["n2o_t_to_Mt"] = constants.TONNE_TO_MEGATONNE
    scale_meta["fertilizer_t_to_Mt"] = constants.TONNE_TO_MEGATONNE
    scale_meta["water_mm3_per_m3"] = constants.MM3_PER_M3

    for region in water_regions:
        bus_name = f"water:{region}"
        n.buses.add(bus_name, carrier="water", region=region)
