# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Infrastructure setup for the food systems model.

This module handles the creation of carriers and buses that form the
foundation of the PyPSA network model.
"""

import pypsa

from . import constants
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
    """
    # Land carrier (class-level buses are added later)
    n.carriers.add("land", unit="Mha")

    # Crops per country
    crop_buses = [
        f"crop_{crop}_{country}" for country in countries for crop in crop_list
    ]
    crop_carriers = [f"crop_{crop}" for country in countries for crop in crop_list]
    if crop_buses:
        n.carriers.add(sorted({f"crop_{crop}" for crop in crop_list}), unit="Mt")
        n.buses.add(crop_buses, carrier=crop_carriers)

    # Residues per country
    residue_items_sorted = sorted(dict.fromkeys(residue_feed_items))
    if residue_items_sorted:
        residue_buses = [
            f"residue_{item}_{country}"
            for country in countries
            for item in residue_items_sorted
        ]
        residue_carriers = [
            f"residue_{item}" for country in countries for item in residue_items_sorted
        ]
        n.carriers.add(sorted(set(residue_carriers)), unit="Mt")
        n.buses.add(residue_buses, carrier=residue_carriers)

    # Foods per country
    food_buses = [
        f"food_{food}_{country}" for country in countries for food in food_list
    ]
    food_carriers = [f"food_{food}" for country in countries for food in food_list]
    if food_buses:
        n.carriers.add(sorted({f"food_{food}" for food in food_list}), unit="Mt")
        n.buses.add(food_buses, carrier=food_carriers)

    # Food groups per country
    group_buses = [
        f"group_{group}_{country}" for country in countries for group in food_group_list
    ]
    group_carriers = [
        f"group_{group}" for country in countries for group in food_group_list
    ]
    if group_buses:
        n.carriers.add(
            sorted({f"group_{group}" for group in food_group_list}),
            unit="Mt",
        )
        n.buses.add(group_buses, carrier=group_carriers)

    # Macronutrients per country
    nutrient_list_sorted = sorted(dict.fromkeys(nutrient_list))
    for nutrient in nutrient_list_sorted:
        unit = nutrient_units[nutrient]
        carrier_unit = _carrier_unit_for_nutrient(unit)
        if nutrient not in n.carriers.static.index:
            n.carriers.add(nutrient, unit=carrier_unit)

    if nutrient_list_sorted:
        nutrient_buses = [
            f"{nut}_{country}" for country in countries for nut in nutrient_list_sorted
        ]
        nutrient_carriers = [
            nut for country in countries for nut in nutrient_list_sorted
        ]
        n.buses.add(nutrient_buses, carrier=nutrient_carriers)

        scale_meta = n.meta.setdefault("carrier_unit_scale", {})
        if any(
            _nutrient_kind(nutrient_units[nut]) == "energy"
            for nut in nutrient_list_sorted
        ):
            scale_meta["macronutrient_kcal_to_Mcal"] = constants.KCAL_TO_MCAL

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
        f"feed_{fc}_{country}" for country in countries for fc in feed_categories
    ]
    feed_carriers = [f"feed_{fc}" for country in countries for fc in feed_categories]
    if feed_buses:
        n.carriers.add(sorted(set(feed_carriers)), unit="Mt")
        n.buses.add(feed_buses, carrier=feed_carriers)

    n.carriers.add("convert_to_feed", unit="Mt")

    # Water carrier (buses added per region below)
    n.carriers.add("water", unit="km^3")

    # Global emission and resource carriers with buses
    for carrier, unit in [
        ("fertilizer", "Mt"),
        ("co2", "MtCO2"),
        ("ch4", "MtCH4"),
        ("n2o", "MtN2O"),
        ("ghg", "MtCO2e"),
    ]:
        n.carriers.add(carrier, unit=unit)
        n.buses.add(carrier, carrier=carrier)

    fert_country_buses = [f"fertilizer_{country}" for country in countries]
    n.buses.add(
        fert_country_buses,
        carrier="fertilizer",
    )

    scale_meta = n.meta.setdefault("carrier_unit_scale", {})
    scale_meta["co2_t_to_Mt"] = constants.TONNE_TO_MEGATONNE
    scale_meta["ch4_t_to_Mt"] = constants.TONNE_TO_MEGATONNE
    scale_meta["ghg_t_to_Mt"] = constants.TONNE_TO_MEGATONNE
    scale_meta["n2o_t_to_Mt"] = constants.TONNE_TO_MEGATONNE
    scale_meta["fertilizer_kg_to_Mt"] = constants.KG_TO_MEGATONNE

    for region in water_regions:
        bus_name = f"water_{region}"
        n.buses.add(bus_name, carrier="water")
