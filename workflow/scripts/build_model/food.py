# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Food conversion and feed supply for the food systems model.

This module handles the conversion of crops to food items through
processing pathways, and the routing of crops and foods to animal
feed categories.
"""

import logging

import pandas as pd
import pypsa

from .. import constants

logger = logging.getLogger(__name__)

_LOW_PROCESSING_COST = 0.01 * constants.USD_TO_BNUSD / constants.TONNE_TO_MEGATONNE


def add_food_conversion_links(
    n: pypsa.Network,
    food_list: list,
    foods: pd.DataFrame,
    countries: list,
    crop_to_fresh_factor: dict[str, float],
    food_to_group: dict[str, str],
    loss_waste: pd.DataFrame,
    crop_list: list,
    byproduct_list: list,
) -> None:
    """Add links for converting crops to foods via processing pathways.

    Pathways can have multiple outputs (e.g., wheat → white flour + bran).
    Each pathway creates one multi-output Link per country.
    Only processes crops that are in the configured crop_list.
    Foods flagged as byproducts are ignored when checking for food-group mappings.
    """

    # Filter foods DataFrame to only include configured crops
    foods = foods[foods["crop"].isin(crop_list)].copy()

    # Add pathway carriers for all unique pathways
    unique_pathways = foods["pathway"].dropna().unique()
    pathway_carriers = sorted({f"pathway_{str(p).strip()}" for p in unique_pathways})
    if pathway_carriers:
        n.carriers.add(pathway_carriers, unit="Mt")

    # Load loss/waste data (already validated by prepare_food_loss_waste.py)
    loss_waste_pairs: dict[tuple[str, str], tuple[float, float]] = {}
    for _, row in loss_waste.iterrows():
        key = (str(row["country"]), str(row["food_group"]))
        loss_waste_pairs[key] = (
            float(row["loss_fraction"]),
            float(row["waste_fraction"]),
        )

    missing_group_foods: set[str] = set()
    byproduct_foods: set[str] = set(byproduct_list or [])
    excessive_losses: set[tuple[str, str]] = set()
    invalid_pathways: list[str] = []

    normalized_countries = [str(c).upper() for c in countries]

    # Group foods by pathway and crop
    pathway_groups = foods.groupby(["pathway", "crop"])

    for (pathway, crop), pathway_df in pathway_groups:
        pathway = str(pathway).strip()
        crop = str(crop).strip()

        # Filter to foods that are in the food_list
        pathway_df = pathway_df[pathway_df["food"].isin(food_list)].copy()
        if pathway_df.empty:
            continue

        # Get output foods and factors
        output_foods = []
        output_factors = []
        for _, row in pathway_df.iterrows():
            output_foods.append(str(row["food"]))
            output_factors.append(float(row["factor"]))

        # Verify mass balance (sum of factors should be ≤ 1.0)
        total_factor = sum(output_factors)
        if total_factor > 1.01:  # Allow small rounding tolerance
            invalid_pathways.append(f"{pathway} ({crop}): sum={total_factor:.3f}")

        # Get conversion factor (dry matter → fresh edible)
        conversion_factor = crop_to_fresh_factor[crop]

        # Create multi-output link names (one per country)
        names = [f"pathway:{pathway}:{c}" for c in normalized_countries]
        bus0 = [f"crop:{crop}:{c}" for c in normalized_countries]

        # All crop/food buses are in Mt, so the efficiency coefficients are
        # simple mass fractions after accounting for losses.
        link_params = {
            "bus0": bus0,
            "carrier": f"pathway_{pathway}",
            "marginal_cost": _LOW_PROCESSING_COST,
            "p_nom_extendable": True,
            "country": normalized_countries,
            "crop": crop,
        }

        # Add each output food as a separate bus with its efficiency
        for output_idx, (food, factor) in enumerate(
            zip(output_foods, output_factors), start=1
        ):
            bus_key = f"bus{output_idx}"
            eff_key = "efficiency" if output_idx == 1 else f"efficiency{output_idx}"

            link_params[bus_key] = [f"food:{food}:{c}" for c in normalized_countries]

            # Calculate efficiencies per country (including loss/waste adjustments)
            efficiencies: list[float] = []
            group = food_to_group.get(food)
            for country in normalized_countries:
                multiplier = 1.0
                if group is None:
                    # Food has no group mapping - no loss/waste adjustment
                    if food not in byproduct_foods:
                        missing_group_foods.add(food)
                else:
                    # Look up loss/waste fractions (guaranteed to exist by preprocessing)
                    raw_loss, raw_waste = loss_waste_pairs[(country, group)]
                    loss_fraction = max(0.0, min(1.0, float(raw_loss)))
                    waste_fraction = max(0.0, min(1.0, float(raw_waste)))

                    if loss_fraction > 0.99 or waste_fraction > 0.99:
                        excessive_losses.add((country, group))

                    multiplier = (1.0 - loss_fraction) * (1.0 - waste_fraction)
                    if multiplier <= 0.0:
                        excessive_losses.add((country, group))
                        multiplier = 0.01  # Small positive to avoid division issues

                efficiencies.append(factor * conversion_factor * multiplier)

            link_params[eff_key] = efficiencies

        n.links.add(names, **link_params)

    # Warnings
    if invalid_pathways:
        logger.warning(
            "Pathways with mass balance issues (sum of factors > 1.0): %s",
            "; ".join(invalid_pathways[:5]),
        )

    if missing_group_foods:
        logger.warning(
            "Food items without food-group mapping (loss/waste ignored): %s",
            ", ".join(sorted(missing_group_foods)),
        )

    if excessive_losses:
        sample = ", ".join(
            f"{country}:{group}" for country, group in sorted(excessive_losses)[:10]
        )
        logger.warning(
            "Extreme food loss/waste values for %d country-group pairs (efficiency clamped to feasible range). Examples: %s",
            len(excessive_losses),
            sample,
        )


def add_feed_supply_links(
    n: pypsa.Network,
    ruminant_categories: pd.DataFrame,
    ruminant_mapping: pd.DataFrame,
    monogastric_categories: pd.DataFrame,
    monogastric_mapping: pd.DataFrame,
    crop_list: list,
    food_list: list,
    residue_items: list,
    countries: list,
) -> None:
    """Add links converting crops and foods into categorized feed pools.

    Uses pre-computed feed categories and mappings to route items to appropriate
    feed pools (4 ruminant + 4 monogastric quality classes).
    """
    # Process ruminant feeds
    ruminant_feeds = ruminant_mapping[
        (
            (ruminant_mapping["source_type"] == "crop")
            & ruminant_mapping["feed_item"].isin(crop_list)
        )
        | (
            (ruminant_mapping["source_type"] == "food")
            & ruminant_mapping["feed_item"].isin(food_list)
        )
        | (
            (ruminant_mapping["source_type"] == "residue")
            & ruminant_mapping["feed_item"].isin(residue_items)
        )
    ].copy()

    # Process monogastric feeds
    monogastric_feeds = monogastric_mapping[
        (
            (monogastric_mapping["source_type"] == "crop")
            & monogastric_mapping["feed_item"].isin(crop_list)
        )
        | (
            (monogastric_mapping["source_type"] == "food")
            & monogastric_mapping["feed_item"].isin(food_list)
        )
        | (
            (monogastric_mapping["source_type"] == "residue")
            & monogastric_mapping["feed_item"].isin(residue_items)
        )
    ].copy()

    # Feed buses are expressed in tonnes of dry matter intake (tDM).
    # Conversion links therefore use efficiency=1.0; digestibility is accounted
    # for downstream in feed-to-animal efficiencies and emissions.

    # Build ruminant links
    all_names = []
    all_bus0 = []
    all_bus1 = []
    all_countries = []
    all_feed_categories = []

    for _, row in ruminant_feeds.iterrows():
        item = row["feed_item"]
        category = row["category"]
        source_type = row["source_type"]

        if source_type == "crop":
            bus_prefix = "crop"
            link_prefix = "convert"
        elif source_type == "food":
            bus_prefix = "food"
            link_prefix = "convert_food"
        else:
            bus_prefix = "residue"
            link_prefix = "convert_residue"

        for country in countries:
            all_names.append(f"{link_prefix}:{item}_to_ruminant_{category}:{country}")
            all_bus0.append(f"{bus_prefix}:{item}:{country}")
            all_bus1.append(f"feed:ruminant_{category}:{country}")
            all_countries.append(country)
            all_feed_categories.append(f"ruminant_{category}")

    # Build monogastric links
    for _, row in monogastric_feeds.iterrows():
        item = row["feed_item"]
        category = row["category"]
        source_type = row["source_type"]

        if source_type == "crop":
            bus_prefix = "crop"
            link_prefix = "convert"
        elif source_type == "food":
            bus_prefix = "food"
            link_prefix = "convert_food"
        else:
            bus_prefix = "residue"
            link_prefix = "convert_residue"

        for country in countries:
            all_names.append(
                f"{link_prefix}:{item}_to_monogastric_{category}:{country}"
            )
            all_bus0.append(f"{bus_prefix}:{item}:{country}")
            all_bus1.append(f"feed:monogastric_{category}:{country}")
            all_countries.append(country)
            all_feed_categories.append(f"monogastric_{category}")

    if not all_names:
        logger.info("No feed supply links to create; check crop/food lists")
        return

    n.links.add(
        all_names,
        bus0=all_bus0,
        bus1=all_bus1,
        carrier="convert_to_feed",
        marginal_cost=_LOW_PROCESSING_COST,
        p_nom_extendable=True,
        country=all_countries,
        feed_category=all_feed_categories,
    )

    logger.info(
        "Created %d feed supply links (%d ruminant, %d monogastric)",
        len(all_names),
        len(ruminant_feeds) * len(countries),
        len(monogastric_feeds) * len(countries),
    )
