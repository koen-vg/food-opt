# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Crop production components for the food systems model.

This module handles all crop-related production links including regional
crop production, multi-cropping systems, grassland feed production, and
spared land allocation with carbon sequestration.
"""

from collections.abc import Callable, Mapping
import logging

import numpy as np
import pandas as pd
import pypsa

from . import constants

logger = logging.getLogger(__name__)


def add_regional_crop_production_links(
    n: pypsa.Network,
    crop_list: list,
    yields_data: dict,
    region_to_country: pd.Series,
    allowed_countries: set,
    crop_costs_per_year: pd.Series,
    crop_costs_per_planting: pd.Series,
    fertilizer_n_rates: Mapping[str, float],
    luc_lef_lookup: Mapping[tuple[str, int, str, str], float] | None = None,
    residue_lookup: Mapping[tuple[str, str, str, int], dict[str, float]] | None = None,
    harvested_area_data: Mapping[str, pd.DataFrame] | None = None,
    use_actual_production: bool = False,
) -> None:
    """Add crop production links per region/resource class and water supply.

    Rainfed yields must be present for every crop; irrigated yields are used when
    provided by the preprocessing pipeline. Output links produce into the same
    crop bus per country; link names encode supply type (i/r) and resource class.
    """
    luc_lef_lookup = luc_lef_lookup or {}
    residue_lookup = residue_lookup or {}
    harvested_area_data = harvested_area_data or {}

    for crop in crop_list:
        # Get fertilizer N application rate (kg N/ha/year) for this crop
        # If crop not in fertilizer data, default to 0 (no fertilizer requirement)
        fert_n_rate_kg_per_ha = float(fertilizer_n_rates.get(crop, 0.0))

        available_supplies = [
            ws for ws in ("r", "i") if f"{crop}_yield_{ws}" in yields_data
        ]

        # Process available water supplies (rainfed always first for stability)
        for ws in available_supplies:
            key = f"{crop}_yield_{ws}"
            crop_yields = yields_data[key].copy()

            if use_actual_production:
                harvest_table = harvested_area_data[f"{crop}_harvested_{ws}"]
                crop_yields = crop_yields.join(
                    harvest_table["harvested_area"].rename("harvested_area"),
                    how="left",
                )

            # Add a unique name per link including water supply and class
            crop_yields["name"] = crop_yields.index.map(
                lambda x,
                crop=crop,
                ws=ws: f"produce_{crop}_{'irrigated' if ws == 'i' else 'rainfed'}_{x[0]}_class{x[1]}"
            )

            # Make index levels columns
            df = crop_yields.reset_index()

            # Set index to "name"
            df.set_index("name", inplace=True)
            df.index.name = None

            # Filter out rows with zero suitable area or zero yield
            df = df[(df["suitable_area"] > 0) & (df["yield"] > 0)]

            if use_actual_production:
                df["harvested_area"] = pd.to_numeric(
                    df.get("harvested_area"), errors="coerce"
                )
                df = df[df["harvested_area"] > 0]

            # Map regions to countries and filter to allowed countries
            df["country"] = df["region"].map(region_to_country)
            df = df[df["country"].isin(allowed_countries)]

            if df.empty:
                continue

            # Cost for this crop: per-year + per-planting costs (USD/ha); if missing, use 0
            cost_year = float(crop_costs_per_year.get(crop, float("nan")))
            cost_planting = float(crop_costs_per_planting.get(crop, float("nan")))

            if not np.isfinite(cost_year):
                cost_year = 0.0
            if not np.isfinite(cost_planting):
                cost_planting = 0.0

            if cost_year == 0.0 and cost_planting == 0.0:
                logger.info(
                    "No USDA cost for crop '%s'; defaulting marginal_cost to 0",
                    crop,
                )

            # For single crops, total cost = per-year + per-planting
            cost_per_ha = cost_year + cost_planting

            # Add links
            # Connect to class-level land bus per region/resource class and water supply
            # Land is now tracked in Mha, so scale yields and areas accordingly
            resource_classes = df["resource_class"].astype(int).to_numpy()
            regions = df["region"].astype(str).to_numpy()
            water_code = "i" if ws == "i" else "r"
            luc_lefs = np.array(
                [
                    luc_lef_lookup.get(
                        (region, int(resource_class), water_code, "cropland"), 0.0
                    )
                    for region, resource_class in zip(regions, resource_classes)
                ],
                dtype=float,
            )  # tCO2/ha/yr
            # Cost is per hectare; convert to per Mha (USD/Mha = USD/ha * 1e6)
            base_cost = cost_per_ha * 1e6

            link_params = {
                "name": df.index,
                # Use the crop's own carrier so no extra carrier is needed
                "carrier": f"crop_{crop}",
                "bus0": df.apply(
                    lambda r,
                    ws=ws: f"land_{r['region']}_class{int(r['resource_class'])}_{'i' if ws == 'i' else 'r'}",
                    axis=1,
                ).tolist(),
                "bus1": df["country"]
                .apply(lambda c, crop=crop: f"crop_{crop}_{c}")
                .tolist(),
                "efficiency": df["yield"] * 1e6,  # t/ha → t/Mha
                "bus3": df["country"].apply(lambda c: f"fertilizer_{c}").tolist(),
                "efficiency3": -fert_n_rate_kg_per_ha
                * 1e6
                * constants.KG_TO_MEGATONNE,  # kg N/ha → Mt N/Mha
                # Link marginal_cost is per unit of bus0 flow (now Mha).
                "marginal_cost": base_cost,
                "p_nom_max": df["suitable_area"] / 1e6,  # ha → Mha
                "p_nom_extendable": not use_actual_production,
            }

            if use_actual_production:
                fixed_area_mha = df["harvested_area"] / 1e6
                link_params["p_nom"] = fixed_area_mha
                link_params["p_nom_max"] = fixed_area_mha
                link_params["p_nom_min"] = fixed_area_mha
                link_params["p_min_pu"] = 1.0

            if ws == "i":
                water_requirement = pd.to_numeric(
                    df["water_requirement_m3_per_ha"], errors="coerce"
                )

                link_params["bus2"] = df["region"].apply(lambda r: f"water_{r}")
                # Convert m³/ha to km³/Mha for compatibility with scaled water units
                link_params["efficiency2"] = -water_requirement * 1e-3

            next_bus_idx = 4
            if residue_lookup:
                residue_feed_items = sorted(
                    {
                        feed_item
                        for region, resource_class in zip(regions, resource_classes)
                        for feed_item in residue_lookup.get(
                            (crop, water_code, region, int(resource_class)), {}
                        )
                    }
                )
                if residue_feed_items:
                    countries_for_rows = df["country"].astype(str).tolist()
                    for feed_item in residue_feed_items:
                        efficiencies = np.zeros(len(df), dtype=float)
                        for idx_row, (region, resource_class) in enumerate(
                            zip(regions, resource_classes)
                        ):
                            residue_dict = residue_lookup.get(
                                (crop, water_code, region, int(resource_class))
                            )
                            if not residue_dict:
                                continue
                            residue_yield = residue_dict.get(feed_item)
                            if residue_yield is None:
                                continue
                            efficiencies[idx_row] = residue_yield * 1e6  # t/ha → t/Mha
                        if np.allclose(efficiencies, 0.0):
                            continue
                        bus_key = f"bus{next_bus_idx}"
                        eff_key = f"efficiency{next_bus_idx}"
                        link_params[bus_key] = [
                            f"residue_{feed_item}_{country}"
                            for country in countries_for_rows
                        ]
                        link_params[eff_key] = efficiencies
                        next_bus_idx += 1

            emission_outputs: dict[str, np.ndarray] = {}

            # Note: Methane emissions from rice cultivation will be added in a separate module

            luc_emissions = (
                luc_lefs * 1e6 * constants.TONNE_TO_MEGATONNE
            )  # tCO2/ha/yr → MtCO2/Mha/yr
            if not np.allclose(luc_emissions, 0.0):
                emission_outputs["co2"] = emission_outputs.get(
                    "co2", np.zeros(len(luc_emissions), dtype=float)
                )
                emission_outputs["co2"] += luc_emissions

            for bus_name in sorted(emission_outputs.keys()):
                values = emission_outputs[bus_name]
                key_bus = f"bus{next_bus_idx}"
                key_eff = f"efficiency{next_bus_idx}"
                link_params[key_bus] = [bus_name] * len(values)
                link_params[key_eff] = values
                next_bus_idx += 1

            n.links.add(**link_params)


def add_multi_cropping_links(
    n: pypsa.Network,
    eligible_area: pd.DataFrame,
    cycle_yields: pd.DataFrame,
    region_to_country: pd.Series,
    allowed_countries: set[str],
    crop_costs_per_year: Mapping[str, float],
    crop_costs_per_planting: Mapping[str, float],
    fertilizer_n_rates: Mapping[str, float],
    residue_lookup: Mapping[tuple[str, str, str, int], dict[str, float]] | None = None,
    luc_lef_lookup: Mapping[tuple[str, int, str, str], float] | None = None,
) -> None:
    """Add multi-cropping production links with a vectorised workflow."""

    if eligible_area.empty or cycle_yields.empty:
        logger.info("No multi-cropping combinations with positive area; skipping")
        return

    residue_lookup = residue_lookup or {}
    luc_lef_lookup = luc_lef_lookup or {}

    key_cols = ["combination", "region", "resource_class", "water_supply"]

    area_df = eligible_area.copy()
    area_df["resource_class"] = area_df["resource_class"].astype(int)
    area_df["water_supply"] = area_df["water_supply"].astype(str)
    area_df["eligible_area_ha"] = pd.to_numeric(
        area_df["eligible_area_ha"], errors="coerce"
    )
    area_df["water_requirement_m3_per_ha"] = pd.to_numeric(
        area_df.get("water_requirement_m3_per_ha", 0.0), errors="coerce"
    )

    region_to_country = region_to_country.astype(str)
    area_df["country"] = area_df["region"].map(region_to_country)
    area_df = area_df.dropna(subset=["eligible_area_ha", "country"])
    area_df = area_df[area_df["eligible_area_ha"] > 0]
    if allowed_countries:
        area_df = area_df[area_df["country"].isin(allowed_countries)]

    if area_df.empty:
        logger.info("No eligible multi-cropping areas after filtering; skipping")
        return

    cycle_df = cycle_yields.copy()
    cycle_df["resource_class"] = cycle_df["resource_class"].astype(int)
    cycle_df["water_supply"] = cycle_df["water_supply"].astype(str)
    cycle_df["yield_t_per_ha"] = pd.to_numeric(
        cycle_df["yield_t_per_ha"], errors="coerce"
    )
    cycle_df = cycle_df.dropna(subset=["yield_t_per_ha", "crop"])
    cycle_df = cycle_df[cycle_df["yield_t_per_ha"] > 0]
    if cycle_df.empty:
        logger.info("No positive multi-cropping yields; skipping")
        return

    merged = cycle_df.merge(area_df, on=key_cols, how="inner")
    if merged.empty:
        logger.info(
            "No overlapping multi-cropping combinations between area and yield tables"
        )
        return

    merged = merged.sort_values([*key_cols, "cycle_index", "crop"])
    merged["crop"] = merged["crop"].astype(str).str.strip()
    merged["country"] = merged["country"].astype(str).str.strip()
    merged["crop_bus"] = "crop_" + merged["crop"] + "_" + merged["country"]
    merged["yield_efficiency"] = merged["yield_t_per_ha"] * 1e6
    merged["output_idx"] = merged.groupby(key_cols).cumcount()

    base = (
        merged.loc[
            :,
            [
                *key_cols,
                "eligible_area_ha",
                "water_requirement_m3_per_ha",
                "country",
            ],
        ]
        .drop_duplicates()
        .set_index(key_cols)
    )

    crop_counts = merged.groupby(key_cols)["crop"].size().rename("crop_count")
    base = base.join(crop_counts)
    base = base[base["crop_count"] > 0]
    if base.empty:
        logger.info(
            "Multi-cropping combinations have no positive-yield crops; skipping"
        )
        return

    cost_year_series = pd.Series(
        {str(k): float(v) for k, v in crop_costs_per_year.items()}
    )
    cost_planting_series = pd.Series(
        {str(k): float(v) for k, v in crop_costs_per_planting.items()}
    )
    merged["cost_per_year"] = merged["crop"].map(cost_year_series).fillna(0.0)
    merged["cost_per_planting"] = merged["crop"].map(cost_planting_series).fillna(0.0)

    costs = merged.groupby(key_cols).agg(
        total_cost_per_year=("cost_per_year", "sum"),
        total_cost_per_planting=("cost_per_planting", "sum"),
    )
    base = base.join(costs)

    fert_series = pd.Series({str(k): float(v) for k, v in fertilizer_n_rates.items()})
    merged["fertilizer_rate"] = merged["crop"].map(fert_series).fillna(0.0)
    fertilizer_totals = (
        merged.groupby(key_cols)["fertilizer_rate"].sum().rename("fertilizer_total")
    )
    base = base.join(fertilizer_totals)

    base[["total_cost_per_year", "total_cost_per_planting", "fertilizer_total"]] = base[
        ["total_cost_per_year", "total_cost_per_planting", "fertilizer_total"]
    ].fillna(0.0)

    base["avg_cost_per_year"] = base["total_cost_per_year"] / base["crop_count"]
    base["marginal_cost"] = (
        base["avg_cost_per_year"] + base["total_cost_per_planting"]
    ) * 1e6
    base["p_nom_extendable"] = True
    base["p_nom_max"] = base["eligible_area_ha"] / 1e6

    residue_records: list[dict[str, object]] = []
    for (crop, water, region, res_class), feed_dict in residue_lookup.items():
        if not isinstance(feed_dict, Mapping):
            continue
        for feed_item, value in feed_dict.items():
            residue_records.append(
                {
                    "crop": str(crop),
                    "water_supply": str(water),
                    "region": str(region),
                    "resource_class": int(res_class),
                    "feed_item": str(feed_item),
                    "residue_yield": float(value),
                }
            )

    if residue_records:
        residue_df = pd.DataFrame(residue_records)
        residue_join = merged.merge(
            residue_df,
            on=["crop", "region", "resource_class", "water_supply"],
            how="left",
        )
        residue_join = residue_join.dropna(subset=["feed_item", "residue_yield"])
        residue_join = residue_join[residue_join["residue_yield"] > 0]
        if residue_join.empty:
            residue_agg = pd.DataFrame(
                columns=[*key_cols, "feed_item", "country", "residue_total"],
            )
        else:
            residue_agg = (
                residue_join.groupby([*key_cols, "feed_item", "country"])[
                    "residue_yield"
                ]
                .sum()
                .rename("residue_total")
                .reset_index()
            )
    else:
        residue_agg = pd.DataFrame(
            columns=[*key_cols, "feed_item", "country", "residue_total"],
        )

    residue_counts = (
        residue_agg.groupby(key_cols).size().rename("residue_count")
        if not residue_agg.empty
        else pd.Series(dtype=int)
    )
    base["residue_count"] = 0
    if not residue_counts.empty:
        base.loc[residue_counts.index, "residue_count"] = residue_counts

    index_df = base.reset_index()
    index_df["resource_class"] = index_df["resource_class"].astype(int)
    index_df["carrier"] = "multi_crop_" + index_df["combination"].astype(str)
    index_df["bus0"] = (
        "land_"
        + index_df["region"].astype(str)
        + "_class"
        + index_df["resource_class"].astype(str)
        + "_"
        + index_df["water_supply"].astype(str)
    )
    index_df["link_name"] = (
        "produce_multi_"
        + index_df["combination"].astype(str)
        + "_"
        + index_df["water_supply"].astype(str)
        + "_"
        + index_df["region"].astype(str)
        + "_class"
        + index_df["resource_class"].astype(str)
    )

    missing_land = index_df[~index_df["bus0"].isin(n.buses.static.index)]
    if not missing_land.empty:
        missing_count = missing_land.shape[0]
        missing_preview = ", ".join(missing_land["bus0"].unique()[:5])
        logger.warning(
            "Skipping %d multi-cropping links due to missing land buses (examples: %s)",
            missing_count,
            missing_preview,
        )
        index_df = index_df[index_df["bus0"].isin(n.buses.static.index)]

    if index_df.empty:
        return

    carriers = sorted(index_df["carrier"].unique())
    if carriers:
        n.carriers.add(carriers, unit="Mha")

    water_req = index_df["water_requirement_m3_per_ha"].astype(float)
    water_valid = (
        index_df["water_supply"].eq("i") & np.isfinite(water_req) & (water_req > 0)
    )
    water_invalid = index_df["water_supply"].eq("i") & ~np.isfinite(water_req)
    if water_invalid.any():
        logger.warning(
            "Ignoring invalid irrigation requirements for %d multi-cropping links",
            int(water_invalid.sum()),
        )

    index_df["water_efficiency"] = np.where(water_valid, -water_req * 1e-3, 0.0)
    index_df["has_water"] = water_valid.astype(int)

    fert_total = index_df["fertilizer_total"].astype(float)
    fert_valid = fert_total > 0
    index_df["fert_efficiency"] = np.where(
        fert_valid, -fert_total * 1e6 * constants.KG_TO_MEGATONNE, 0.0
    )
    index_df["has_fertilizer"] = fert_valid.astype(int)

    luc_keys = list(
        zip(
            index_df["region"].astype(str),
            index_df["resource_class"].astype(int),
            index_df["water_supply"].astype(str),
            ["cropland"] * len(index_df),
        )
    )
    luc_values = np.array([float(luc_lef_lookup.get(key, 0.0)) for key in luc_keys])
    luc_valid = ~np.isclose(luc_values, 0.0)
    index_df["luc_efficiency"] = luc_values * 1e6 * constants.TONNE_TO_MEGATONNE
    index_df["has_luc"] = luc_valid.astype(int)

    outputs = merged.merge(index_df[[*key_cols, "link_name"]], on=key_cols, how="left")
    outputs["offset"] = outputs["output_idx"] + 1
    offset_str = outputs["offset"].astype(int).astype(str)
    outputs["bus_col"] = "bus" + offset_str
    outputs["eff_col"] = np.where(
        outputs["offset"].eq(1),
        "efficiency",
        "efficiency" + offset_str,
    )
    outputs_entries = outputs[
        [
            "link_name",
            "bus_col",
            "crop_bus",
            "eff_col",
            "yield_efficiency",
        ]
    ].rename(columns={"crop_bus": "bus_value", "yield_efficiency": "eff_value"})

    entry_frames = [outputs_entries]

    water_columns = [*key_cols, "link_name", "water_efficiency", "crop_count"]
    water_entries = index_df.loc[index_df["has_water"] == 1, water_columns].copy()
    if not water_entries.empty:
        water_entries["offset"] = water_entries["crop_count"] + 1
        offset_str = water_entries["offset"].astype(int).astype(str)
        water_entries["bus_col"] = "bus" + offset_str
        water_entries["eff_col"] = "efficiency" + offset_str
        water_entries.loc[water_entries["offset"].eq(1), "eff_col"] = "efficiency"
        water_entries["bus_value"] = "water_" + water_entries["region"].astype(str)
        water_entries = water_entries[
            [
                "link_name",
                "bus_col",
                "bus_value",
                "eff_col",
                "water_efficiency",
            ]
        ].rename(columns={"water_efficiency": "eff_value"})
        entry_frames.append(water_entries)

    fert_entries = index_df[index_df["has_fertilizer"] == 1][
        [
            *key_cols,
            "link_name",
            "country",
            "fert_efficiency",
            "crop_count",
            "has_water",
        ]
    ].copy()
    if not fert_entries.empty:
        fert_entries["offset"] = (
            fert_entries["crop_count"] + fert_entries["has_water"] + 1
        )
        offset_str = fert_entries["offset"].astype(int).astype(str)
        fert_entries["bus_col"] = "bus" + offset_str
        fert_entries["eff_col"] = "efficiency" + offset_str
        fert_entries.loc[fert_entries["offset"].eq(1), "eff_col"] = "efficiency"
        fert_entries["bus_value"] = "fertilizer_" + fert_entries["country"].astype(str)
        fert_entries = fert_entries[
            [
                "link_name",
                "bus_col",
                "bus_value",
                "eff_col",
                "fert_efficiency",
            ]
        ].rename(columns={"fert_efficiency": "eff_value"})
        entry_frames.append(fert_entries)

    if not residue_agg.empty:
        residue_entries = residue_agg.merge(
            index_df[
                [
                    *key_cols,
                    "link_name",
                    "crop_count",
                    "has_water",
                    "has_fertilizer",
                ]
            ],
            on=key_cols,
            how="left",
        )
        residue_entries = residue_entries.sort_values([*key_cols, "feed_item"])
        residue_entries["entry_order"] = residue_entries.groupby(key_cols).cumcount()
        residue_entries["offset"] = (
            residue_entries["crop_count"]
            + residue_entries["has_water"]
            + residue_entries["has_fertilizer"]
            + residue_entries["entry_order"]
            + 1
        )
        offset_str = residue_entries["offset"].astype(int).astype(str)
        residue_entries["bus_col"] = "bus" + offset_str
        residue_entries["eff_col"] = "efficiency" + offset_str
        residue_entries.loc[residue_entries["offset"].eq(1), "eff_col"] = "efficiency"
        residue_entries["bus_value"] = (
            "residue_"
            + residue_entries["feed_item"].astype(str)
            + "_"
            + residue_entries["country"].astype(str)
        )
        residue_entries["eff_value"] = residue_entries["residue_total"] * 1e6
        entry_frames.append(
            residue_entries[
                [
                    "link_name",
                    "bus_col",
                    "bus_value",
                    "eff_col",
                    "eff_value",
                ]
            ]
        )

    luc_entries = index_df[index_df["has_luc"] == 1][
        [
            *key_cols,
            "link_name",
            "luc_efficiency",
            "crop_count",
            "has_water",
            "has_fertilizer",
            "residue_count",
        ]
    ].copy()
    if not luc_entries.empty:
        luc_entries["offset"] = (
            luc_entries["crop_count"]
            + luc_entries["has_water"]
            + luc_entries["has_fertilizer"]
            + luc_entries["residue_count"]
            + 1
        )
        offset_str = luc_entries["offset"].astype(int).astype(str)
        luc_entries["bus_col"] = "bus" + offset_str
        luc_entries["eff_col"] = "efficiency" + offset_str
        luc_entries.loc[luc_entries["offset"].eq(1), "eff_col"] = "efficiency"
        luc_entries["bus_value"] = "co2"
        luc_entries = luc_entries[
            [
                "link_name",
                "bus_col",
                "bus_value",
                "eff_col",
                "luc_efficiency",
            ]
        ].rename(columns={"luc_efficiency": "eff_value"})
        entry_frames.append(luc_entries)

    entries = pd.concat(entry_frames, ignore_index=True)
    bus_wide = entries.pivot_table(
        index="link_name", columns="bus_col", values="bus_value", aggfunc="first"
    )
    eff_wide = entries.pivot_table(
        index="link_name", columns="eff_col", values="eff_value", aggfunc="first"
    )

    link_df = index_df.set_index("link_name")
    component_cols = [
        "carrier",
        "bus0",
        "p_nom_extendable",
        "p_nom_max",
        "marginal_cost",
    ]
    link_df = link_df[component_cols]
    link_df = link_df.join(bus_wide, how="left").join(eff_wide, how="left")

    bus_cols = sorted(
        [c for c in link_df.columns if c.startswith("bus") and c != "bus0"],
        key=lambda name: int(name[3:]),
    )
    eff_cols = [
        "efficiency",
        *sorted(
            [
                c
                for c in link_df.columns
                if c.startswith("efficiency") and c != "efficiency"
            ],
            key=lambda name: int(name[len("efficiency") :]),
        ),
    ]

    missing_outputs = link_df["bus1"].isna() | link_df["efficiency"].isna()
    if missing_outputs.any():
        logger.warning(
            "Dropping %d multi-cropping links without valid crop outputs",
            int(missing_outputs.sum()),
        )
        link_df = link_df[~missing_outputs]

    if link_df.empty:
        return

    for col in bus_cols:
        link_df[col] = link_df[col].where(link_df[col].notna(), None)
    for col in eff_cols:
        link_df[col] = link_df[col].fillna(0.0)

    link_names = link_df.index.tolist()
    kwargs = {
        col: link_df[col].tolist() for col in component_cols + bus_cols + eff_cols
    }
    n.links.add(link_names, **kwargs)


def add_grassland_feed_links(
    n: pypsa.Network,
    grassland: pd.DataFrame,
    land_rainfed: pd.DataFrame,
    region_to_country: pd.Series,
    allowed_countries: set,
    luc_lef_lookup: Mapping[tuple[str, int, str, str], float] | None = None,
    current_grassland_area: pd.DataFrame | None = None,
    pasture_land_area: pd.Series | None = None,
    use_actual_production: bool = False,
) -> None:
    """Add links supplying ruminant feed directly from rainfed land."""

    luc_lef_lookup = luc_lef_lookup or {}

    grass_df = grassland.copy()
    grass_df = grass_df[np.isfinite(grass_df["yield"]) & (grass_df["yield"] > 0)]
    if grass_df.empty:
        logger.info("Grassland yields contain no positive entries; skipping")
        return

    grass_df = grass_df.reset_index()
    grass_df["resource_class"] = grass_df["resource_class"].astype(int)
    grass_df = grass_df.set_index(["region", "resource_class"])

    base_frame = grass_df.join(
        land_rainfed[["area_ha"]].rename(columns={"area_ha": "land_area"}),
        how="inner",
    )
    if use_actual_production:
        observed_area = (
            current_grassland_area.set_index(["region", "resource_class"])["area_ha"]
            .astype(float)
            .rename("observed_area")
        )
        base_frame = base_frame.join(observed_area, how="left")

    candidate_area = base_frame["suitable_area"].fillna(base_frame["land_area"])
    land_cap = np.minimum(candidate_area.to_numpy(), base_frame["land_area"].to_numpy())
    base_index = base_frame.index
    land_cap_series = pd.Series(land_cap, index=base_index, dtype=float)

    cropland_frame = base_frame.copy()
    marginal_frame: pd.DataFrame | None = None

    if use_actual_production:
        # Under validation the observed harvested/grazed area is split so that
        # marginal hectares are satisfied first (subject to the derived
        # land_marginal potential) and only the remainder pulls from the shared
        # cropland pool.
        observed_series = (
            pd.to_numeric(base_frame.get("observed_area"), errors="coerce")
            .fillna(0.0)
            .astype(float)
        )
        base_frame = base_frame.drop(columns=["observed_area"])
        if pasture_land_area is not None and not pasture_land_area.empty:
            marginal_cap_series = pasture_land_area.reindex(base_index, fill_value=0.0)
        else:
            marginal_cap_series = pd.Series(0.0, index=base_index, dtype=float)
        observed_aligned = observed_series.reindex(base_index)
        marginal_alloc = np.minimum(
            observed_aligned.to_numpy(), marginal_cap_series.to_numpy()
        )
        cropland_observed = np.maximum(
            observed_aligned.to_numpy() - marginal_alloc, 0.0
        )
        cropland_available = np.minimum(land_cap_series.to_numpy(), cropland_observed)
        cropland_frame["available_area"] = cropland_available
        cropland_frame = cropland_frame[cropland_frame["available_area"] > 0]

        if np.any(marginal_alloc > 0.0):
            marginal_series = pd.Series(
                marginal_alloc,
                index=base_index,
                name="available_area",
            )
            marginal_frame = grass_df.join(marginal_series, how="inner")
            marginal_frame = marginal_frame[marginal_frame["available_area"] > 0]
    else:
        cropland_frame["available_area"] = land_cap_series.reindex(
            cropland_frame.index
        ).to_numpy()
        cropland_frame = cropland_frame[cropland_frame["available_area"] > 0]
        if pasture_land_area is not None and not pasture_land_area.empty:
            marginal_frame = grass_df.join(
                pasture_land_area.rename("available_area"), how="inner"
            )
            marginal_frame = marginal_frame[marginal_frame["available_area"] > 0]

    # Helper to convert a per-region/class frame into Link components. The caller
    # passes a name prefix so we can distinguish cropland-competing vs.
    # marginal-only grassland in the network outputs.
    def _add_links_for_frame(
        frame: pd.DataFrame,
        name_prefix: str,
        bus0_builder: Callable[[pd.Series], str],
    ) -> bool:
        if frame is None or frame.empty:
            return False
        work = frame.reset_index()
        work["country"] = work["region"].map(region_to_country)
        work = work[work["country"].isin(allowed_countries)]
        work = work.dropna(subset=["country"])
        if work.empty:
            return False
        work["name"] = work.apply(
            lambda r: f"{name_prefix}_{r['region']}_class{int(r['resource_class'])}",
            axis=1,
        )
        work["bus0"] = work.apply(bus0_builder, axis=1)
        work["bus1"] = work["country"].apply(lambda c: f"feed_ruminant_grassland_{c}")

        luc_emissions = (
            np.array(
                [
                    luc_lef_lookup.get(
                        (row["region"], int(row["resource_class"]), "r", "pasture"),
                        0.0,
                    )
                    for _, row in work.iterrows()
                ],
                dtype=float,
            )
            * 1e6
            * constants.TONNE_TO_MEGATONNE
        )

        available_mha = work["available_area"].to_numpy() / 1e6
        params = {
            "carrier": "feed_ruminant_grassland",
            "bus0": work["bus0"].tolist(),
            "bus1": work["bus1"].tolist(),
            "efficiency": work["yield"].to_numpy() * 1e6,
            "p_nom_max": available_mha,
            "p_nom_extendable": not use_actual_production,
            "marginal_cost": 0.0,
        }
        if use_actual_production:
            params["p_nom"] = available_mha
            params["p_nom_min"] = available_mha
            params["p_min_pu"] = 1.0
        if not np.allclose(luc_emissions, 0.0):
            params["bus2"] = "co2"
            params["efficiency2"] = luc_emissions

        n.links.add(work["name"].tolist(), **params)
        return True

    link_added = False

    # Standard grassland links consume land from the same rainfed cropland pool
    # that crops use, so they continue to compete for those hectares when
    # optimisation is unconstrained.
    link_added |= _add_links_for_frame(
        cropland_frame,
        "grassland",
        lambda r: f"land_{r['region']}_class{int(r['resource_class'])}_r",
    )

    if marginal_frame is not None and not marginal_frame.empty:
        # Marginal grassland links tap into the exclusive land_marginal buses so
        # grazing can expand without reducing cropland-suitable land.
        link_added |= _add_links_for_frame(
            marginal_frame,
            "grassland_marginal",
            lambda r: f"land_marginal_{r['region']}_class{int(r['resource_class'])}",
        )

    if not link_added:
        logger.info("Grassland entries have zero available area; skipping")


def add_spared_land_links(
    n: pypsa.Network,
    land_class_df: pd.DataFrame,
    luc_lef_lookup: Mapping[tuple[str, int, str, str], float],
    grazing_only_area: pd.Series | None = None,
) -> None:
    """Add optional links to allocate spared land and credit CO2 sinks.

    The AGB threshold filtering is now applied directly in the LEF calculation,
    so this function simply uses the LEF values as provided.

    Parameters
    ----------
    n : pypsa.Network
        The network to add links to.
    land_class_df : pd.DataFrame
        Land area by region/water_supply/resource_class.
    luc_lef_lookup : Mapping
        Land-use change emission factors (tCO2/ha/yr) by (region, class, water, use).
        The spared land LEFs already incorporate AGB threshold filtering.
    """

    if not luc_lef_lookup:
        logger.info("No LUC LEF entries available for spared land; skipping")
        return

    frames: list[pd.DataFrame] = []

    base_df = land_class_df.reset_index()
    base_df["resource_class"] = base_df["resource_class"].astype(int)
    base_df["water_supply"] = base_df["water_supply"].astype(str)
    base_df["lookup_ws"] = base_df["water_supply"]
    frames.append(base_df)

    if grazing_only_area is not None and not grazing_only_area.empty:
        marginal_df = (
            grazing_only_area.rename("area_ha")
            .reset_index()
            .astype({"resource_class": int})
        )
        marginal_df["water_supply"] = "m"
        marginal_df["lookup_ws"] = "r"
        frames.append(marginal_df)

    df = pd.concat(frames, ignore_index=True)
    df["lef"] = df.apply(
        lambda r: luc_lef_lookup.get(
            (r["region"], int(r["resource_class"]), r["lookup_ws"], "spared"), 0.0
        ),
        axis=1,
    )

    filtered_count = (df["lef"] == 0.0).sum()
    df = df[(df["lef"] != 0.0) & (df["area_ha"] > 0)].copy()

    if filtered_count > 0:
        logger.debug("Filtered %d spared land entries with zero LEF", filtered_count)

    if df.empty:
        logger.info("No eligible spared land entries; skipping spared links")
        return

    def _bus0(row: pd.Series) -> str:
        if row["water_supply"] == "m":
            return f"land_marginal_{row['region']}_class{int(row['resource_class'])}"
        return f"land_{row['region']}_class{int(row['resource_class'])}_{row['water_supply']}"

    def _sink_bus(row: pd.Series) -> str:
        if row["water_supply"] == "m":
            return f"land_spared_marginal_{row['region']}_class{int(row['resource_class'])}"
        return f"land_spared_{row['region']}_class{int(row['resource_class'])}_{row['water_supply']}"

    def _link_name(row: pd.Series) -> str:
        if row["water_supply"] == "m":
            return f"spare_marginal_{row['region']}_class{int(row['resource_class'])}"
        return f"spare_{row['region']}_class{int(row['resource_class'])}_{row['water_supply']}"

    df["bus0"] = df.apply(_bus0, axis=1)
    df["sink_bus"] = df.apply(_sink_bus, axis=1)
    df["link_name"] = df.apply(_link_name, axis=1)
    df["area_mha"] = df["area_ha"] / 1e6

    # Add carrier and sink buses
    n.carriers.add("spared_land", unit="Mha")
    n.buses.add(df["sink_bus"].tolist(), carrier="spared_land")

    # Add stores for sink buses
    store_names = [f"{bus}_store" for bus in df["sink_bus"]]
    n.stores.add(
        store_names,
        bus=df["sink_bus"].tolist(),
        carrier="spared_land",
        e_nom_extendable=True,
    )

    # Add spared land links
    n.links.add(
        df["link_name"].tolist(),
        carrier="spared_land",
        bus0=df["bus0"].tolist(),
        bus1=df["sink_bus"].tolist(),
        efficiency=1.0,
        bus2="co2",
        efficiency2=(
            df["lef"] * 1e6 * constants.TONNE_TO_MEGATONNE
        ).to_numpy(),  # tCO2/ha/yr → MtCO2/Mha/yr
        p_nom_extendable=True,
        p_nom_max=df["area_mha"].to_numpy(),
    )


def add_residue_soil_incorporation_links(
    n: pypsa.Network,
    residue_feed_items: list[str],
    ruminant_feed_mapping: pd.DataFrame,
    ruminant_feed_categories: pd.DataFrame,
    monogastric_feed_mapping: pd.DataFrame,
    monogastric_feed_categories: pd.DataFrame,
    countries: list[str],
    incorporation_n2o_factor: float,
) -> None:
    """Add links for crop residue incorporation into soil with N₂O emissions.

    Residues left on the field decompose and release N₂O. This function creates
    links that consume residues and produce N₂O emissions based on their N content
    and the IPCC emission factor.

    This processes ALL residues in the model, regardless of whether they're used
    for ruminant or monogastric feed. N content is looked up from whichever feed
    category dataset contains the residue.

    Parameters
    ----------
    n : pypsa.Network
        The network to add links to.
    residue_feed_items : list[str]
        Complete list of all residue items in the model.
    ruminant_feed_mapping : pd.DataFrame
        Ruminant feed mapping (columns: feed_item, category).
    ruminant_feed_categories : pd.DataFrame
        Ruminant feed category properties (column: N_g_per_kg_DM).
    monogastric_feed_mapping : pd.DataFrame
        Monogastric feed mapping (columns: feed_item, category).
    monogastric_feed_categories : pd.DataFrame
        Monogastric feed category properties (column: N_g_per_kg_DM).
    countries : list[str]
        List of country ISO codes.
    incorporation_n2o_factor : float
        IPCC EF1 emission factor (kg N₂O-N per kg N input).
    """

    if not residue_feed_items:
        logger.info("No residue items found; skipping soil incorporation links")
        return

    # Build lookup for N content from both ruminant and monogastric feed data
    n_content_lookup = {}

    # First, try ruminant feed categories
    for _, row in ruminant_feed_mapping[
        ruminant_feed_mapping["source_type"] == "residue"
    ].iterrows():
        item = row["feed_item"]
        category = row["category"]
        # Look up N content for this category
        cat_data = ruminant_feed_categories[
            ruminant_feed_categories["category"] == category
        ]
        if not cat_data.empty and "N_g_per_kg_DM" in cat_data.columns:
            n_content = cat_data.iloc[0]["N_g_per_kg_DM"]
            if pd.notna(n_content):
                n_content_lookup[item] = float(n_content)

    # Then try monogastric feed categories (may override or add new entries)
    for _, row in monogastric_feed_mapping[
        monogastric_feed_mapping["source_type"] == "residue"
    ].iterrows():
        item = row["feed_item"]
        if item in n_content_lookup:
            continue  # Already have N content from ruminant data
        category = row["category"]
        cat_data = monogastric_feed_categories[
            monogastric_feed_categories["category"] == category
        ]
        if not cat_data.empty and "N_g_per_kg_DM" in cat_data.columns:
            n_content = cat_data.iloc[0]["N_g_per_kg_DM"]
            if pd.notna(n_content):
                n_content_lookup[item] = float(n_content)

    if not n_content_lookup:
        logger.info(
            "No residue items with N content data; skipping soil incorporation links"
        )
        return

    # Build links for all residue x country combinations
    all_names = []
    all_bus0 = []
    all_efficiency = []

    # Fallback N content for residues without data (g N/kg DM)
    # Conservative estimate based on typical crop straw/stover N content
    fallback_n_content = 8.0

    for item in residue_feed_items:
        # Use fallback if we don't have N content data for this residue
        if item not in n_content_lookup:
            logger.info(
                "No N content data for residue %s; using fallback value %.1f g N/kg DM",
                item,
                fallback_n_content,
            )
            n_content_g_per_kg = fallback_n_content
        else:
            n_content_g_per_kg = n_content_lookup[item]

        # Calculate N₂O emission efficiency
        # N₂O emissions (Mt N₂O) = residue_DM (Mt DM) x N_content (kg N / kg DM)
        #                          x EF1 (kg N₂O-N / kg N) x (44/28) (kg N₂O / kg N₂O-N)
        # Convert g/kg to kg/kg: n_content_g_per_kg / 1000
        # Convert to Mt: x 1e-6
        n2o_efficiency = (
            (n_content_g_per_kg / 1000.0)  # g N/kg DM → kg N/kg DM
            * incorporation_n2o_factor  # kg N₂O-N / kg N
            * (44.0 / 28.0)  # kg N₂O / kg N₂O-N
            * 1e-6  # kg → Mt
        )

        for country in countries:
            bus_name = f"residue_{item}_{country}"

            # Only add link if the residue bus exists in the network
            if bus_name not in n.buses.static.index:
                continue

            all_names.append(f"incorporate_{item}_{country}")
            all_bus0.append(bus_name)
            all_efficiency.append(n2o_efficiency)

    if not all_names:
        logger.info("No valid residue buses found; skipping soil incorporation links")
        return

    # Add the carrier
    carrier = "residue_incorporation"
    if carrier not in n.carriers.static.index:
        n.carriers.add(carrier, unit="tDM")

    # Add the links
    n.links.add(
        all_names,
        bus0=all_bus0,
        bus1="n2o",
        carrier=carrier,
        efficiency=all_efficiency,
        marginal_cost=0.0,  # No cost to incorporate residues
        p_nom_extendable=True,
    )

    logger.info(
        "Created %d residue soil incorporation links for %d residue types",
        len(all_names),
        len(n_content_lookup),
    )
