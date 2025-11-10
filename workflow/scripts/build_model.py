# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

from collections.abc import Callable, Iterable, Mapping
import functools
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
from sklearn.cluster import KMeans

KM3_PER_M3 = 1e-9  # convert cubic metres to cubic kilometres
TONNE_TO_MEGATONNE = 1e-6  # convert tonnes to megatonnes
KG_TO_MEGATONNE = 1e-9  # convert kilograms to megatonnes
KCAL_TO_MCAL = 1e-6  # convert kilocalories to megacalories
KCAL_PER_100G_TO_MCAL_PER_TONNE = 1e-2  # kcal/100g to Mcal per tonne of food
DAYS_PER_YEAR = 365
N2O_N_TO_N2O = 44.0 / 28.0  # molecular weight ratio to convert N2O-N to N2O

SUPPORTED_NUTRITION_UNITS = {
    "g/100g": {"kind": "mass", "efficiency_factor": TONNE_TO_MEGATONNE},
    "kcal/100g": {
        "kind": "energy",
        "efficiency_factor": KCAL_PER_100G_TO_MCAL_PER_TONNE,
    },
}


def _nutrient_kind(unit: str) -> str:
    try:
        return SUPPORTED_NUTRITION_UNITS[unit]["kind"]
    except KeyError as exc:
        raise ValueError(f"Unsupported nutrition unit '{unit}'") from exc


def _nutrition_efficiency_factor(unit: str) -> float:
    try:
        return SUPPORTED_NUTRITION_UNITS[unit]["efficiency_factor"]
    except KeyError as exc:
        raise ValueError(f"Unsupported nutrition unit '{unit}'") from exc


def _per_capita_to_bus_units(
    value_per_person_per_day: float,
    population: float,
    unit: str,
) -> float:
    kind = _nutrient_kind(unit)
    if kind == "mass":
        # g/person/day → Mt/year (1e-12 = 1e-6 g→t x 1e-6 t→Mt)
        return value_per_person_per_day * population * DAYS_PER_YEAR * 1e-12
    if kind == "energy":
        return value_per_person_per_day * population * DAYS_PER_YEAR * KCAL_TO_MCAL
    raise ValueError(f"Unsupported nutrient kind '{kind}' for unit '{unit}'")


def _per_capita_food_group_to_mt(
    value_per_person_per_day: float, population: float
) -> float:
    """Convert g/person/day to Mt/year for food group buses."""

    return value_per_person_per_day * population * DAYS_PER_YEAR * 1e-12


def _carrier_unit_for_nutrient(unit: str) -> str:
    kind = _nutrient_kind(unit)
    if kind == "mass":
        return "Mt"
    if kind == "energy":
        return "Mcal"
    raise ValueError(f"Unsupported nutrient kind '{kind}'")


def _load_crop_yield_table(path: str) -> tuple[pd.DataFrame, dict[str, str | float]]:
    df = pd.read_csv(path)

    grouped_units = (
        df.groupby("variable")["unit"].agg(lambda s: s.dropna().unique()).to_dict()
    )
    units: dict[str, str | float] = {}
    for var, vals in grouped_units.items():
        if len(vals) == 1:
            units[var] = vals[0]
        else:
            units[var] = np.nan

    pivot = (
        df.pivot(index=["region", "resource_class"], columns="variable", values="value")
        .rename_axis(index=("region", "resource_class"), columns=None)
        .sort_index()
    )

    # Ensure resource_class level is integer
    pivot.index = pivot.index.set_levels(
        pivot.index.levels[1].astype(int), level="resource_class"
    )

    # Ensure numeric columns
    for column in pivot.columns:
        pivot[column] = pd.to_numeric(pivot[column], errors="coerce")

    return pivot, units


def _gaez_code_to_crop_map(mapping_df: pd.DataFrame) -> dict[str, str]:
    code_columns = [c for c in mapping_df.columns if c.endswith("_code")]
    mapping: dict[str, str] = {}
    for _, row in mapping_df.iterrows():
        crop_name = str(row.get("crop_name", "")).strip()
        if not crop_name:
            continue
        for col in code_columns:
            code = row.get(col)
            if pd.isna(code):
                continue
            code_str = str(code).strip().lower()
            if not code_str:
                continue
            mapping[code_str] = crop_name
    return mapping


def _fresh_mass_conversion_factors(
    edible_portion_df: pd.DataFrame,
    moisture_df: pd.DataFrame,
    crops: set[str],
) -> dict[str, float]:
    """Compute fresh mass conversion factors from edible portion and moisture data."""
    df = edible_portion_df.copy()
    df["crop"] = df["crop"].astype(str).str.strip()

    df = df.set_index("crop")
    df["edible_portion_coefficient"] = pd.to_numeric(
        df["edible_portion_coefficient"], errors="coerce"
    )

    moisture = moisture_df.copy()
    moisture["crop"] = moisture["crop"].astype(str).str.strip()
    moisture = moisture.set_index("crop")
    moisture["moisture_fraction"] = pd.to_numeric(
        moisture["moisture_fraction"], errors="coerce"
    )

    factors: dict[str, float] = {}
    missing_edible: list[str] = []
    missing_moisture: list[str] = []
    for crop in sorted(crops):
        if crop not in df.index:
            missing_edible.append(crop)
            continue
        if crop not in moisture.index:
            missing_moisture.append(crop)
            continue
        edible_coeff = df.at[crop, "edible_portion_coefficient"]
        moisture_fraction = moisture.at[crop, "moisture_fraction"]
        if pd.isna(edible_coeff):
            missing_edible.append(crop)
            continue
        if pd.isna(moisture_fraction):
            missing_moisture.append(crop)
            continue
        if not (0 < edible_coeff <= 1):
            raise ValueError(
                f"Invalid edible portion coefficient {edible_coeff} for crop '{crop}'"
            )
        if moisture_fraction < 0 or moisture_fraction >= 1:
            raise ValueError(
                f"Moisture fraction for crop '{crop}' must be in [0, 1); found {moisture_fraction}"
            )

        dry_fraction = 1 - moisture_fraction
        if dry_fraction <= 0:
            raise ValueError(
                f"Dry matter fraction for crop '{crop}' must be positive; moisture={moisture_fraction}"
            )
        factor = edible_coeff / dry_fraction

        if not np.isfinite(factor) or factor <= 0:
            raise ValueError(
                f"Computed non-positive fresh mass factor {factor} for crop '{crop}'"
            )
        factors[crop] = factor

    if missing_edible:
        raise ValueError(
            "Missing edible portion data for crops: "
            + ", ".join(sorted(missing_edible))
        )
    if missing_moisture:
        raise ValueError(
            "Missing moisture fraction data for crops: "
            + ", ".join(sorted(missing_moisture))
        )

    return factors


def _build_luc_lef_lookup(
    df: pd.DataFrame,
) -> dict[tuple[str, int, str, str], float]:
    """Return LEF (tCO2/ha/yr) lookup keyed by (region, class, water, use)."""

    if df.empty:
        return {}

    lookup: dict[tuple[str, int, str, str], float] = {}
    for row in df.itertuples(index=False):
        lef = getattr(row, "LEF_tCO2_per_ha_yr", np.nan)
        if not np.isfinite(lef):
            continue
        key = (
            str(row.region),
            int(row.resource_class),
            str(row.water),
            str(row.use),
        )
        lookup[key] = float(lef)
    return lookup


logger = logging.getLogger(__name__)


def _calculate_manure_n_outputs(
    product: str,
    feed_category: str,
    efficiency: float,
    ruminant_categories: pd.DataFrame,
    monogastric_categories: pd.DataFrame,
    nutrition: pd.DataFrame,
    manure_n_to_fertilizer: float,
    manure_n2o_factor: float,
) -> tuple[float, float]:
    """Calculate manure N fertilizer and N₂O outputs per tonne feed intake.

    Parameters
    ----------
    product : str
        Animal product name
    feed_category : str
        Feed category (e.g., "ruminant_forage", "monogastric_grain")
    efficiency : float
        Feed conversion efficiency (t product / t feed DM)
    ruminant_categories : pd.DataFrame
        Ruminant feed categories with N_g_per_kg_DM
    monogastric_categories : pd.DataFrame
        Monogastric feed categories with N_g_per_kg_DM
    nutrition : pd.DataFrame
        Nutrition data indexed by (food, nutrient)
    manure_n_to_fertilizer : float
        Fraction of excreted N available as fertilizer
    manure_n2o_factor : float
        kg N2O-N per kg manure N applied

    Returns
    -------
    tuple[float, float]
        (N fertilizer t/t feed, N2O emissions t/t feed)
    """
    # Get feed N content (g N/kg DM)
    category_name = feed_category.split("_", 1)[
        1
    ]  # Extract category from "ruminant_forage" etc.

    if feed_category.startswith("ruminant_"):
        feed_n_g_per_kg = ruminant_categories.loc[
            ruminant_categories["category"] == category_name, "N_g_per_kg_DM"
        ].values[0]
    else:
        feed_n_g_per_kg = monogastric_categories.loc[
            monogastric_categories["category"] == category_name, "N_g_per_kg_DM"
        ].values[0]

    # Get product protein content (g protein/100g product)
    try:
        protein_g_per_100g = nutrition.loc[(product, "protein"), "value"]
    except KeyError:
        logger.warning(f"No protein data for {product}, assuming 0 N in product")
        protein_g_per_100g = 0.0

    # Convert protein to N using factor 6.25 (protein = N * 6.25)
    # N (g/kg product) = protein (g/100g) * 10 / 6.25
    product_n_g_per_kg = (protein_g_per_100g * 10) / 6.25

    # Calculate N flows per tonne feed
    feed_n_t_per_t_feed = feed_n_g_per_kg / 1000  # t N/t feed
    product_output_t_per_t_feed = efficiency  # t product/t feed
    product_n_t_per_t_feed = (product_n_g_per_kg / 1000) * product_output_t_per_t_feed

    # N excreted = N in feed - N in product
    n_excreted_t_per_t_feed = feed_n_t_per_t_feed - product_n_t_per_t_feed

    # Special handling for grassland: manure deposited on pasture, not collected
    if feed_category.endswith("_grassland"):
        # No N available as fertilizer (deposited on pasture)
        n_fertilizer_t_per_t_feed = 0.0
        # But still produce N2O from pasture deposition
        # Use the same N2O factor as for applied manure
        n2o_n_t_per_t_feed = n_excreted_t_per_t_feed * manure_n2o_factor
        n2o_t_per_t_feed = n2o_n_t_per_t_feed * (44.0 / 28.0)
    else:
        # N available as fertilizer (after collection losses)
        n_fertilizer_t_per_t_feed = n_excreted_t_per_t_feed * manure_n_to_fertilizer

        # N2O emissions from applied manure N
        # N2O-N = manure_n * n2o_factor
        # N2O = N2O-N * 44/28 (molecular weight conversion)
        n2o_n_t_per_t_feed = n_fertilizer_t_per_t_feed * manure_n2o_factor
        n2o_t_per_t_feed = n2o_n_t_per_t_feed * (44.0 / 28.0)

    return n_fertilizer_t_per_t_feed, n2o_t_per_t_feed


def _calculate_ch4_per_feed_intake(
    product: str,
    feed_category: str,
    country: str,
    enteric_my_lookup: dict[str, float],
    manure_emissions: pd.DataFrame,
) -> float:
    """Calculate total CH4 emissions (tCH4/t feed DM) from enteric + manure sources.

    Note: This is calculated per tonne of feed intake (bus0), not per product output.

    Parameters
    ----------
    product : str
        Animal product name (e.g., "meat-cattle", "dairy", "meat-pig")
    feed_category : str
        Feed category name (e.g., "ruminant_roughage", "monogastric_grain")
    country : str
        Country code (ISO3)
    enteric_my_lookup : dict[str, float]
        Enteric methane yields by ruminant feed category (g CH4 / kg DMI)
    manure_emissions : pd.DataFrame
        Manure CH4 emission factors with columns: country, product, feed_category,
        manure_ch4_kg_per_kg_DMI

    Returns
    -------
    float
        Total CH4 emissions in tCH4/t feed DM (enteric + manure)
    """
    # Initialize total CH4 per tonne feed
    total_ch4_per_t_feed = 0.0

    # Add enteric fermentation CH4 (ruminants only)
    if feed_category.startswith("ruminant_"):
        category = feed_category.split("_", 1)[1]
        if category in enteric_my_lookup:
            # Convert from g CH4/kg DM to t CH4/t DM
            enteric_t_per_t = enteric_my_lookup[category] / 1000.0
            total_ch4_per_t_feed += enteric_t_per_t

    # Add manure CH4 (all animal products)
    manure_row = manure_emissions[
        (manure_emissions["country"] == country)
        & (manure_emissions["product"] == product)
        & (manure_emissions["feed_category"] == feed_category)
    ]

    if not manure_row.empty:
        # Convert from kg CH4/kg DM to t CH4/t DM
        manure_kg_per_kg = manure_row["manure_ch4_kg_per_kg_DMI"].values[0]
        manure_t_per_t = manure_kg_per_kg / 1000.0
        total_ch4_per_t_feed += manure_t_per_t

    return total_ch4_per_t_feed  # t CH4 / t feed DM


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
    n.add("Carrier", "land", unit="Mha")

    # Crops per country
    crop_buses = [
        f"crop_{crop}_{country}" for country in countries for crop in crop_list
    ]
    crop_carriers = [f"crop_{crop}" for country in countries for crop in crop_list]
    if crop_buses:
        n.add("Carrier", sorted({f"crop_{crop}" for crop in crop_list}), unit="t")
        n.add("Bus", crop_buses, carrier=crop_carriers)

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
        n.add("Carrier", sorted(set(residue_carriers)), unit="t")
        n.add("Bus", residue_buses, carrier=residue_carriers)

    # Foods per country
    food_buses = [
        f"food_{food}_{country}" for country in countries for food in food_list
    ]
    food_carriers = [f"food_{food}" for country in countries for food in food_list]
    if food_buses:
        n.add("Carrier", sorted({f"food_{food}" for food in food_list}), unit="t")
        n.add("Bus", food_buses, carrier=food_carriers)

    # Food groups per country
    group_buses = [
        f"group_{group}_{country}" for country in countries for group in food_group_list
    ]
    group_carriers = [
        f"group_{group}" for country in countries for group in food_group_list
    ]
    if group_buses:
        n.add(
            "Carrier",
            sorted({f"group_{group}" for group in food_group_list}),
            unit="Mt",
        )
        n.add("Bus", group_buses, carrier=group_carriers)
        scale_meta = n.meta.setdefault("carrier_unit_scale", {})
        scale_meta["food_group_t_to_Mt"] = TONNE_TO_MEGATONNE

    # Macronutrients per country
    nutrient_list_sorted = sorted(dict.fromkeys(nutrient_list))
    for nutrient in nutrient_list_sorted:
        unit = nutrient_units[nutrient]
        carrier_unit = _carrier_unit_for_nutrient(unit)
        if nutrient not in n.carriers.index:
            n.add("Carrier", nutrient, unit=carrier_unit)

    if nutrient_list_sorted:
        nutrient_buses = [
            f"{nut}_{country}" for country in countries for nut in nutrient_list_sorted
        ]
        nutrient_carriers = [
            nut for country in countries for nut in nutrient_list_sorted
        ]
        n.add("Bus", nutrient_buses, carrier=nutrient_carriers)

        scale_meta = n.meta.setdefault("carrier_unit_scale", {})
        if any(
            _nutrient_kind(nutrient_units[nut]) == "mass"
            for nut in nutrient_list_sorted
        ):
            scale_meta["macronutrient_t_to_Mt"] = TONNE_TO_MEGATONNE
        if any(
            _nutrient_kind(nutrient_units[nut]) == "energy"
            for nut in nutrient_list_sorted
        ):
            scale_meta["macronutrient_kcal_to_Mcal"] = KCAL_TO_MCAL

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
        n.add("Carrier", sorted(set(feed_carriers)), unit="t")
        n.add("Bus", feed_buses, carrier=feed_carriers)

    n.add("Carrier", "convert_to_feed", unit="t")

    # Water carrier (buses added per region below)
    n.add("Carrier", "water", unit="km^3")

    # Global emission and resource carriers with buses
    for carrier, unit in [
        ("fertilizer", "Mt"),
        ("co2", "MtCO2"),
        ("ch4", "MtCH4"),
        ("n2o", "MtN2O"),
        ("ghg", "MtCO2e"),
    ]:
        n.add("Carrier", carrier, unit=unit)
        n.add("Bus", carrier, carrier=carrier)

    fert_country_buses = [f"fertilizer_{country}" for country in countries]
    n.add(
        "Bus",
        fert_country_buses,
        carrier="fertilizer",
    )

    scale_meta = n.meta.setdefault("carrier_unit_scale", {})
    scale_meta["co2_t_to_Mt"] = TONNE_TO_MEGATONNE
    scale_meta["ch4_t_to_Mt"] = TONNE_TO_MEGATONNE
    scale_meta["ghg_t_to_Mt"] = TONNE_TO_MEGATONNE
    scale_meta["n2o_t_to_Mt"] = TONNE_TO_MEGATONNE
    scale_meta["fertilizer_kg_to_Mt"] = KG_TO_MEGATONNE

    for region in water_regions:
        bus_name = f"water_{region}"
        n.add("Bus", bus_name, carrier="water")


def _add_land_slack_generators(
    n: pypsa.Network, bus_names: list[str], marginal_cost: float
) -> None:
    """Attach slack generators to the provided land buses."""

    if not bus_names:
        return

    n.add(
        "Generator",
        [f"{bus}_slack" for bus in bus_names],
        bus=bus_names,
        carrier="land_slack",
        p_nom_extendable=True,
        marginal_cost=marginal_cost,
    )


def add_primary_resources(
    n: pypsa.Network,
    primary_config: dict,
    region_water_limits: pd.Series,
    co2_price: float,
    ch4_to_co2_factor: float,
    n2o_to_co2_factor: float,
    use_actual_production: bool,
) -> None:
    """Add primary resource components and emissions bookkeeping."""
    # Water limits
    water_limits = region_water_limits * KM3_PER_M3
    n.add(
        "Store",
        "water_store_" + water_limits.index,
        bus="water_" + water_limits.index,
        carrier="water",
        e_nom=water_limits.values,
        e_initial=water_limits.values,
        e_nom_extendable=False,
        e_cyclic=False,
    )

    # Slack in water limits when using actual (current) production
    if use_actual_production:
        n.add(
            "Generator",
            "water_slack_" + water_limits.index,
            bus="water_" + water_limits.index,
            carrier="water",
            marginal_cost=1e-6,
            p_nom_extendable=True,
        )

    scale_meta = n.meta.setdefault("carrier_unit_scale", {})
    scale_meta["water_km3_per_m3"] = KM3_PER_M3

    co2_price_per_mt = co2_price / TONNE_TO_MEGATONNE

    # Fertilizer remains global (no regionalization yet)
    n.add(
        "Generator",
        "fertilizer",
        bus="fertilizer",
        carrier="fertilizer",
        p_nom_extendable=True,
        p_nom_max=float(primary_config["fertilizer"]["limit"]) * KG_TO_MEGATONNE,
    )

    # Add GHG aggregation store and links from individual gases
    n.add(
        "Store",
        "ghg",
        bus="ghg",
        carrier="ghg",
        e_nom_extendable=True,
        e_nom_min=-np.inf,
        e_min_pu=-1.0,
        marginal_cost_storage=co2_price_per_mt,
    )
    n.add(
        "Link",
        "convert_co2_to_ghg",
        bus0="co2",
        bus1="ghg",
        carrier="co2",
        efficiency=1.0,
        p_min_pu=-1.0,  # allow negative emissions flow
        p_nom_extendable=True,
    )
    n.add(
        "Link",
        "convert_ch4_to_ghg",
        bus0="ch4",
        bus1="ghg",
        carrier="ch4",
        efficiency=ch4_to_co2_factor,
        p_nom_extendable=True,
    )
    n.add(
        "Link",
        "convert_n2o_to_ghg",
        bus0="n2o",
        bus1="ghg",
        carrier="n2o",
        efficiency=n2o_to_co2_factor,
        p_nom_extendable=True,
    )


def add_fertilizer_distribution_links(
    n: pypsa.Network,
    countries: Iterable[str],
    synthetic_n2o_factor: float,
) -> None:
    """Connect the global fertilizer supply bus to country-level fertilizer buses."""

    country_list = list(countries)
    if not country_list:
        return

    names = [f"distribute_synthetic_fertilizer_{country}" for country in country_list]
    params: dict[str, object] = {
        "bus0": ["fertilizer"] * len(country_list),
        "bus1": [f"fertilizer_{country}" for country in country_list],
        "carrier": "fertilizer",
        "efficiency": [1.0] * len(country_list),
        "p_nom_extendable": True,
    }

    emission_mt_per_mt = max(0.0, float(synthetic_n2o_factor)) * N2O_N_TO_N2O
    if emission_mt_per_mt > 0.0:
        params["bus2"] = ["n2o"] * len(country_list)
        params["efficiency2"] = [emission_mt_per_mt] * len(country_list)

    n.add("Link", names, **params)


def add_regional_crop_production_links(
    n: pypsa.Network,
    crop_list: list,
    yields_data: dict,
    region_to_country: pd.Series,
    allowed_countries: set,
    crop_costs_per_year: pd.Series,
    crop_costs_per_planting: pd.Series,
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

        if "r" not in available_supplies:
            raise ValueError(
                f"Rainfed yield data missing for crop '{crop}'; ensure build_crop_yields ran"
            )

        # Process available water supplies (rainfed always first for stability)
        for ws in available_supplies:
            key = f"{crop}_yield_{ws}"
            crop_yields = yields_data[key].copy()

            if use_actual_production:
                harvest_key = f"{crop}_harvested_{ws}"
                try:
                    harvest_table = harvested_area_data[harvest_key]
                except KeyError as exc:
                    raise ValueError(
                        f"Missing harvested area data for crop '{crop}' ({'irrigated' if ws == 'i' else 'rainfed'})"
                    ) from exc
                if "harvested_area" not in harvest_table.columns:
                    raise ValueError(
                        f"Harvested area table for crop '{crop}' ({ws}) missing 'harvested_area' column"
                    )
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
                * KG_TO_MEGATONNE,  # kg N/ha → Mt N/Mha
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
                if "water_requirement_m3_per_ha" not in df.columns:
                    raise ValueError(
                        "Missing GAEZ water requirement column for irrigated crop "
                        f"'{crop}'"
                    )

                water_requirement = pd.to_numeric(
                    df["water_requirement_m3_per_ha"], errors="coerce"
                )
                invalid = (~np.isfinite(water_requirement)) | (water_requirement < 0)
                if invalid.any():
                    invalid_rows = df.loc[invalid, ["region", "resource_class"]]
                    sample = (
                        invalid_rows.head(5)
                        .apply(
                            lambda r: f"{r['region']}#class{int(r['resource_class'])}",
                            axis=1,
                        )
                        .tolist()
                    )
                    raise ValueError(
                        "Invalid irrigated water requirement for crop "
                        f"'{crop}' in {invalid_rows.shape[0]} rows (examples: "
                        + ", ".join(sample)
                        + ")"
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
                luc_lefs * 1e6 * TONNE_TO_MEGATONNE
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

            n.add("Link", **link_params)


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

    missing_land = index_df[~index_df["bus0"].isin(n.buses.index)]
    if not missing_land.empty:
        missing_count = missing_land.shape[0]
        missing_preview = ", ".join(missing_land["bus0"].unique()[:5])
        logger.warning(
            "Skipping %d multi-cropping links due to missing land buses (examples: %s)",
            missing_count,
            missing_preview,
        )
        index_df = index_df[index_df["bus0"].isin(n.buses.index)]

    if index_df.empty:
        return

    carriers = sorted(index_df["carrier"].unique())
    if carriers:
        n.add("Carrier", carriers, unit="Mha")

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
        fert_valid, -fert_total * 1e6 * KG_TO_MEGATONNE, 0.0
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
    index_df["luc_efficiency"] = luc_values * 1e6 * TONNE_TO_MEGATONNE
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
    n.add("Link", link_names, **kwargs)


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
                        (row["region"], int(row["resource_class"]), "r", "pasture"), 0.0
                    )
                    for _, row in work.iterrows()
                ],
                dtype=float,
            )
            * 1e6
            * TONNE_TO_MEGATONNE
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

        n.add("Link", work["name"].tolist(), **params)
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
    n.add("Carrier", "spared_land", unit="Mha")
    n.add("Bus", df["sink_bus"].tolist(), carrier="spared_land")

    # Add stores for sink buses
    store_names = [f"{bus}_store" for bus in df["sink_bus"]]
    n.add(
        "Store",
        store_names,
        bus=df["sink_bus"].tolist(),
        carrier="spared_land",
        e_nom_extendable=True,
    )

    # Add spared land links
    n.add(
        "Link",
        df["link_name"].tolist(),
        carrier="spared_land",
        bus0=df["bus0"].tolist(),
        bus1=df["sink_bus"].tolist(),
        efficiency=1.0,
        bus2="co2",
        efficiency2=(
            df["lef"] * 1e6 * TONNE_TO_MEGATONNE
        ).to_numpy(),  # tCO2/ha/yr → MtCO2/Mha/yr
        p_nom_extendable=True,
        p_nom_max=df["area_mha"].to_numpy(),
    )


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

    # Validate that foods.csv has the new pathway column
    if "pathway" not in foods.columns:
        raise ValueError(
            "foods.csv must contain a 'pathway' column. "
            "See data/foods.csv for the expected format with pathway-based structure."
        )

    # Filter foods DataFrame to only include configured crops
    foods = foods[foods["crop"].isin(crop_list)].copy()

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
            food = str(row["food"]).strip()
            if pd.isna(row["factor"]):
                raise ValueError(
                    f"Missing conversion factor in pathway '{pathway}' for crop '{crop}' to food '{food}'"
                )
            factor = float(row["factor"])
            if not np.isfinite(factor) or factor <= 0:
                raise ValueError(
                    f"Invalid conversion factor {factor} in pathway '{pathway}' for crop '{crop}' to food '{food}'"
                )
            output_foods.append(food)
            output_factors.append(factor)

        # Verify mass balance (sum of factors should be ≤ 1.0)
        total_factor = sum(output_factors)
        if total_factor > 1.01:  # Allow small rounding tolerance
            invalid_pathways.append(f"{pathway} ({crop}): sum={total_factor:.3f}")

        # Get conversion factor (dry matter → fresh edible)
        try:
            conversion_factor = crop_to_fresh_factor[crop]
        except KeyError as exc:
            raise ValueError(
                f"Missing moisture/edible conversion data for crop '{crop}' in pathway '{pathway}'"
            ) from exc

        # Create multi-output link names (one per country)
        safe_pathway_name = pathway.replace(" ", "_").replace("(", "").replace(")", "")
        names = [f"pathway_{safe_pathway_name}_{c}" for c in normalized_countries]
        bus0 = [f"crop_{crop}_{c}" for c in normalized_countries]

        # Build parameters for multi-output link
        link_params = {
            "bus0": bus0,
            "marginal_cost": 0.01,
            "p_nom_extendable": True,
        }

        # Add each output food as a separate bus with its efficiency
        for output_idx, (food, factor) in enumerate(
            zip(output_foods, output_factors), start=1
        ):
            bus_key = f"bus{output_idx}"
            eff_key = "efficiency" if output_idx == 1 else f"efficiency{output_idx}"

            link_params[bus_key] = [f"food_{food}_{c}" for c in normalized_countries]

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

        n.add("Link", names, **link_params)

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

    # Merge with category digestibility
    ruminant_feeds = ruminant_feeds.merge(
        ruminant_categories[["category", "digestibility"]],
        on="category",
        how="left",
    )

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

    # Merge with category digestibility
    monogastric_feeds = monogastric_feeds.merge(
        monogastric_categories[["category", "digestibility"]],
        on="category",
        how="left",
    )

    # Build ruminant links
    all_names = []
    all_bus0 = []
    all_bus1 = []
    all_efficiency = []

    for _, row in ruminant_feeds.iterrows():
        item = row["feed_item"]
        category = row["category"]
        source_type = row["source_type"]
        digestibility = row["digestibility"]

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
            all_names.append(f"{link_prefix}_{item}_to_ruminant_{category}_{country}")
            all_bus0.append(f"{bus_prefix}_{item}_{country}")
            all_bus1.append(f"feed_ruminant_{category}_{country}")
            all_efficiency.append(digestibility)

    # Build monogastric links
    for _, row in monogastric_feeds.iterrows():
        item = row["feed_item"]
        category = row["category"]
        source_type = row["source_type"]
        digestibility = row["digestibility"]

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
                f"{link_prefix}_{item}_to_monogastric_{category}_{country}"
            )
            all_bus0.append(f"{bus_prefix}_{item}_{country}")
            all_bus1.append(f"feed_monogastric_{category}_{country}")
            all_efficiency.append(digestibility)

    if not all_names:
        logger.info("No feed supply links to create; check crop/food lists")
        return

    n.add(
        "Link",
        all_names,
        bus0=all_bus0,
        bus1=all_bus1,
        carrier="convert_to_feed",
        efficiency=all_efficiency,
        marginal_cost=0.01,
        p_nom_extendable=True,
    )

    logger.info(
        "Created %d feed supply links (%d ruminant, %d monogastric)",
        len(all_names),
        len(ruminant_feeds) * len(countries),
        len(monogastric_feeds) * len(countries),
    )


def add_feed_to_animal_product_links(
    n: pypsa.Network,
    animal_products: list,
    feed_requirements: pd.DataFrame,
    ruminant_feed_categories: pd.DataFrame,
    monogastric_feed_categories: pd.DataFrame,
    manure_emissions: pd.DataFrame,
    nutrition: pd.DataFrame,
    fertilizer_config: dict,
    countries: list,
) -> None:
    """Add links that convert feed pools into animal products with emissions and manure N.

    UNITS:

    - Input (bus0): Feed in DRY MATTER (tonnes DM)
    - Output (bus1): Animal products in FRESH WEIGHT, RETAIL MEAT (tonnes fresh)

      - For meats: retail/edible meat weight (boneless, trimmed)
      - For dairy: whole milk (fresh weight)
      - For eggs: whole eggs (fresh weight)

    - Efficiency: tonnes retail product per tonne feed DM

      - Incorporates carcass-to-retail conversion for meat products
      - Generated from Wirsenius (2000) + GLEAM feed energy values

    Outputs per link:

    - bus1: Animal product (fresh weight, retail meat)
    - bus2: CH₄ emissions (enteric + manure)
    - bus3: Manure N available as fertilizer
    - bus4: N₂O emissions from manure N application

    Parameters
    ----------
    n : pypsa.Network
        The network to add links to
    animal_products : list
        List of animal product names
    feed_requirements : pd.DataFrame
        Feed requirements with columns: product, feed_category, efficiency
        Efficiency in tonnes RETAIL PRODUCT per tonne FEED DM
    ruminant_feed_categories : pd.DataFrame
        Ruminant feed categories with enteric CH4 yields and N content
    monogastric_feed_categories : pd.DataFrame
        Monogastric feed categories with N content
    manure_emissions : pd.DataFrame
        Manure CH4 emission factors by country, product, and feed_category
    nutrition : pd.DataFrame
        Nutrition data (indexed by food, nutrient) with protein content
    fertilizer_config : dict
        Fertilizer configuration with manure_n_to_fertilizer and manure_n2o_factor
    countries : list
        List of country codes
    """

    produce_carriers = sorted({f"produce_{product!s}" for product in animal_products})
    if produce_carriers:
        n.add("Carrier", produce_carriers, unit="t")

    if not animal_products:
        logger.info("No animal products configured; skipping feed→animal links")
        return

    # Build enteric methane yield lookup from ruminant feed categories
    enteric_my_lookup = (
        ruminant_feed_categories.set_index("category")["MY_g_CH4_per_kg_DMI"]
        .astype(float)
        .to_dict()
    )

    df = feed_requirements.copy()
    df = df[df["product"].isin(animal_products)]

    if df.empty:
        return

    df["efficiency"] = df["efficiency"].astype(float)

    # Get config parameters
    manure_n_to_fert = fertilizer_config.get("manure_n_to_fertilizer", 0.75)
    manure_n2o_factor = fertilizer_config.get("manure_n2o_factor", 0.01)

    # Build all link names and buses (expand each row for all countries)
    all_names = []
    all_bus0 = []
    all_bus1 = []
    all_bus3 = []
    all_carrier = []
    all_efficiency = []
    all_ch4 = []
    all_n_fert = []
    all_n2o = []

    skipped_count = 0
    for _, row in df.iterrows():
        for country in countries:
            # Check if required buses exist
            feed_bus = f"feed_{row['feed_category']}_{country}"
            food_bus = f"food_{row['product']}_{country}"
            if feed_bus not in n.buses.index or food_bus not in n.buses.index:
                skipped_count += 1
                continue

            # Calculate total CH4 (enteric + manure) per tonne feed intake
            # This is relative to bus0 (feed), so it can be used directly as efficiency2
            ch4_per_t_feed = _calculate_ch4_per_feed_intake(
                product=row["product"],
                feed_category=row["feed_category"],
                country=country,
                enteric_my_lookup=enteric_my_lookup,
                manure_emissions=manure_emissions,
            )

            # Calculate manure N fertilizer and N2O outputs per tonne feed intake
            n_fert_per_t_feed, n2o_per_t_feed = _calculate_manure_n_outputs(
                product=row["product"],
                feed_category=row["feed_category"],
                efficiency=row["efficiency"],
                ruminant_categories=ruminant_feed_categories,
                monogastric_categories=monogastric_feed_categories,
                nutrition=nutrition,
                manure_n_to_fertilizer=manure_n_to_fert,
                manure_n2o_factor=manure_n2o_factor,
            )

            all_names.append(
                f"produce_{row['product']}_from_{row['feed_category']}_{country}"
            )
            all_bus0.append(feed_bus)
            all_bus1.append(food_bus)
            all_bus3.append(f"fertilizer_{country}")
            all_carrier.append(f"produce_{row['product']}")
            all_efficiency.append(row["efficiency"])
            all_ch4.append(ch4_per_t_feed * TONNE_TO_MEGATONNE)
            all_n_fert.append(n_fert_per_t_feed * TONNE_TO_MEGATONNE)
            all_n2o.append(n2o_per_t_feed * TONNE_TO_MEGATONNE)

    # All animal production links now have multiple outputs:
    # bus1: animal product, bus2: CH4, bus3: manure N fertilizer (country-specific), bus4: N2O
    n.add(
        "Link",
        all_names,
        bus0=all_bus0,
        bus1=all_bus1,
        carrier=all_carrier,
        efficiency=all_efficiency,
        marginal_cost=0.0,
        p_nom_extendable=True,
        bus2="ch4",
        efficiency2=all_ch4,
        bus3=all_bus3,
        efficiency3=all_n_fert,
        bus4="n2o",
        efficiency4=all_n2o,
    )

    logger.info(
        "Added %d feed→animal product links with outputs: product, CH4 (enteric+manure), manure N fertilizer, N2O",
        len(all_names),
    )
    if skipped_count > 0:
        logger.info("Skipped %d links due to missing buses", skipped_count)


def add_food_group_buses_and_loads(
    n: pypsa.Network,
    food_group_list: list,
    food_groups: pd.DataFrame,
    food_groups_config: dict,
    countries: list,
    population: pd.Series,
    *,
    per_country_equal: dict[str, dict[str, float]] | None = None,
) -> None:
    """Add carriers, buses, and loads for food groups defined in the CSV.

    Supports min/max/equal per-person-per-day targets per food group. Country-level
    equality overrides can be supplied via ``per_country_equal``.
    """

    if not food_groups_config:
        return

    per_country_equal = per_country_equal or {}

    logger.info("Adding food group loads based on nutrition requirements...")
    for group in food_group_list:
        group_config = food_groups_config.get(group, {}) or {}
        min_value = group_config.get("min")
        max_value = group_config.get("max")
        equal_value = group_config.get("equal")
        equal_overrides = per_country_equal.get(group, {})

        names = [f"{group}_{c}" for c in countries]
        buses = [f"group_{group}_{c}" for c in countries]
        carriers = [f"group_{group}"] * len(countries)

        def _value_list(
            base_value: float | None,
            overrides: dict[str, float] | None,
        ) -> list[float | None]:
            override_map = overrides or {}
            values: list[float | None] = []
            for country in countries:
                if country in override_map:
                    values.append(float(override_map[country]))
                elif base_value is not None:
                    values.append(float(base_value))
                else:
                    values.append(None)
            return values

        equal_values = _value_list(equal_value, equal_overrides)
        if all(v is not None for v in equal_values):
            equal_totals = [
                _per_capita_food_group_to_mt(value, float(population[country]))
                for value, country in zip(equal_values, countries)
            ]
            n.add("Load", names, bus=buses, carrier=carriers, p_set=equal_totals)
            # Equality constraint fixes consumption; no additional stores required
            continue

        min_values = _value_list(min_value, None)
        max_values = _value_list(max_value, None)

        min_totals: list[float] | None = None
        if any(v is not None and v > 0.0 for v in min_values):
            min_totals = [
                _per_capita_food_group_to_mt(v or 0.0, float(population[country]))
                for v, country in zip(min_values, countries)
            ]
            n.add("Load", names, bus=buses, carrier=carriers, p_set=min_totals)

        max_totals: list[float] | None = None
        if any(v is not None for v in max_values):
            max_totals = [
                _per_capita_food_group_to_mt(v or 0.0, float(population[country]))
                for v, country in zip(max_values, countries)
            ]

        store_names = [f"store_{group}_{c}" for c in countries]
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

        n.add(
            "Store",
            store_names,
            bus=buses,
            carrier=carriers,
            e_nom_extendable=True,
            **store_kwargs,
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
        nutrient_config = (
            macronutrients_config.get(nutrient, {}) if macronutrients_config else {}
        )
        equal_value = nutrient_config.get("equal")
        min_value = nutrient_config.get("min")
        max_value = nutrient_config.get("max")

        # Handle equality constraint
        if equal_value is not None:
            p_set = [
                _per_capita_to_bus_units(equal_value, float(population[c]), unit)
                for c in countries
            ]
            n.add("Load", names, bus=names, carrier=carriers, p_set=p_set)
            # For equality constraints, we don't need a Store (Load fixes the flow)
            continue

        # Handle min constraint with Load
        min_totals = None
        if min_value is not None:
            min_totals = [
                _per_capita_to_bus_units(min_value, float(population[c]), unit)
                for c in countries
            ]
            n.add("Load", names, bus=names, carrier=carriers, p_set=min_totals)

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

        n.add(
            "Store",
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
        names = [
            f"consume_{food.replace(' ', '_').replace('(', '').replace(')', '')}_{c}"
            for c in countries
        ]
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

        params = {"bus0": bus0, "marginal_cost": 0.01}
        for i, (buses, effs) in enumerate(zip(out_bus_lists, eff_lists), start=1):
            params[f"bus{i}"] = buses
            params["efficiency" if i == 1 else f"efficiency{i}"] = effs

        # optional food group output as last leg
        if group_val is not None and pd.notna(group_val):
            idx = len(nutrients) + 1
            params[f"bus{idx}"] = [f"group_{group_val}_{c}" for c in countries]
            params[f"efficiency{idx}"] = TONNE_TO_MEGATONNE

        n.add("Link", names, p_nom_extendable=True, **params)


def _resolve_trade_costs(
    trade_config: dict,
    items: list,
    *,
    categories_key: str | None,
    default_cost_key: str | None,
    fallback_cost_key: str,
    category_item_key: str,
) -> tuple[dict[str, float], float]:
    """Map each item to its configured trade cost per kilometre."""

    # Get default cost from config hierarchy
    if default_cost_key is not None:
        default_cost = float(trade_config[default_cost_key])
    else:
        default_cost = float(trade_config[fallback_cost_key])

    item_costs = {str(item): default_cost for item in items}

    if categories_key is None:
        return item_costs, default_cost

    # Override with category-specific costs
    categories = trade_config.get(categories_key, {})
    for _category, cfg in categories.items():
        category_cost = float(cfg.get("cost_per_km", default_cost))
        configured_items = cfg.get(category_item_key, [])

        for item in configured_items:
            item_label = str(item)
            if item_label in item_costs:
                item_costs[item_label] = category_cost

    return item_costs, default_cost


def _add_trade_hubs_and_links(
    n: pypsa.Network,
    trade_config: dict,
    regions_gdf: gpd.GeoDataFrame,
    countries: list,
    items: list,
    *,
    hub_count_key: str,
    marginal_cost_key: str,
    cost_categories_key: str | None,
    default_cost_key: str | None,
    category_item_key: str,
    non_tradable_key: str,
    bus_prefix: str,
    carrier_prefix: str,
    hub_name_prefix: str,
    link_name_prefix: str,
    log_label: str,
) -> None:
    """Shared implementation for adding trade hubs and links for a set of items."""

    n_hubs = int(trade_config[hub_count_key])
    item_costs, default_cost = _resolve_trade_costs(
        trade_config,
        items,
        categories_key=cost_categories_key,
        default_cost_key=default_cost_key,
        fallback_cost_key=marginal_cost_key,
        category_item_key=category_item_key,
    )

    if len(regions_gdf) == 0 or len(countries) == 0:
        logger.info("Skipping %s trade hubs: no regions/countries available", log_label)
        return

    items = list(dict.fromkeys(items))
    if len(items) == 0:
        logger.info("Skipping %s trade hubs: no items configured", log_label)
        return

    non_tradable = {
        str(item) for item in trade_config[non_tradable_key] if item in items
    }
    tradable_items = [item for item in items if item not in non_tradable]
    if non_tradable:
        logger.info(
            "Skipping %s trade network for configured non-tradable items: %s",
            log_label,
            ", ".join(sorted(non_tradable)),
        )

    if not tradable_items:
        logger.info("Skipping %s trade hubs: no tradable items available", log_label)
        return

    gdf = regions_gdf.copy()
    gdf_ee = gdf.to_crs(6933)

    cent = gdf_ee.geometry.centroid
    X = np.column_stack([cent.x.values, cent.y.values])
    k = min(max(1, n_hubs), len(X))
    if k < n_hubs:
        logger.info(
            "Reducing %s hub count from %d to %d (regions=%d)",
            log_label,
            n_hubs,
            k,
            len(X),
        )
        n_hubs = k

    km = KMeans(n_clusters=n_hubs, n_init=10, random_state=0)
    km.fit_predict(X)
    centers = km.cluster_centers_

    hub_ids = list(range(n_hubs))
    hub_bus_names: list[str] = []
    hub_bus_carriers: list[str] = []
    for item in tradable_items:
        item_label = str(item)
        for h in hub_ids:
            hub_bus_names.append(f"{hub_name_prefix}_{h}_{item_label}")
            hub_bus_carriers.append(f"{carrier_prefix}{item_label}")

    if hub_bus_names:
        n.add("Bus", hub_bus_names, carrier=hub_bus_carriers)

    gdf_countries = gdf_ee[gdf_ee["country"].isin(countries)].dissolve(
        by="country", as_index=True
    )
    ccent = gdf_countries.geometry.centroid
    C = np.column_stack([ccent.x.values, ccent.y.values])
    dch = ((C[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2) ** 0.5
    nearest_hub_idx = dch.argmin(axis=1)
    nearest_hub_dist_km = dch[np.arange(len(C)), nearest_hub_idx] / 1000.0

    country_index = gdf_countries.index.to_list()
    country_to_hub = {c: int(h) for c, h in zip(country_index, nearest_hub_idx)}
    country_to_dist_km = {
        c: float(d) for c, d in zip(country_index, nearest_hub_dist_km)
    }

    valid_countries = [c for c in countries if c in country_to_hub]

    link_names: list[str] = []
    link_bus0: list[str] = []
    link_bus1: list[str] = []
    link_costs: list[float] = []

    if valid_countries:
        for item in tradable_items:
            item_label = str(item)
            item_cost = item_costs.get(item_label, default_cost)
            for c in valid_countries:
                hub_idx = country_to_hub[c]
                cost = country_to_dist_km[c] * item_cost

                country_bus = f"{bus_prefix}{item_label}_{c}"
                hub_bus = f"{hub_name_prefix}_{hub_idx}_{item_label}"

                link_names.append(f"{link_name_prefix}_{item_label}_{c}_hub{hub_idx}")
                link_bus0.append(country_bus)
                link_bus1.append(hub_bus)
                link_costs.append(cost)

                link_names.append(f"{link_name_prefix}_{item_label}_hub{hub_idx}_{c}")
                link_bus0.append(hub_bus)
                link_bus1.append(country_bus)
                link_costs.append(cost)

    if link_names:
        n.add(
            "Link",
            link_names,
            bus0=link_bus0,
            bus1=link_bus1,
            marginal_cost=link_costs,
            p_nom_extendable=[True] * len(link_names),
        )

    if n_hubs >= 2:
        H = centers
        D = np.sqrt(((H[:, None, :] - H[None, :, :]) ** 2).sum(axis=2)) / 1000.0
        ii, jj = np.where(~np.eye(n_hubs, dtype=bool))

        hub_link_names: list[str] = []
        hub_link_bus0: list[str] = []
        hub_link_bus1: list[str] = []
        hub_link_costs: list[float] = []

        if len(ii) > 0:
            dists_km = D[ii, jj]
            for item in tradable_items:
                item_label = str(item)
                item_cost = item_costs.get(item_label, default_cost)
                for i, j, dist in zip(ii, jj, dists_km):
                    hub_link_names.append(
                        f"{link_name_prefix}_{item_label}_hub{i}_to_hub{j}"
                    )
                    hub_link_bus0.append(f"{hub_name_prefix}_{i}_{item_label}")
                    hub_link_bus1.append(f"{hub_name_prefix}_{j}_{item_label}")
                    hub_link_costs.append(float(dist) * item_cost)

        if hub_link_names:
            n.add(
                "Link",
                hub_link_names,
                bus0=hub_link_bus0,
                bus1=hub_link_bus1,
                marginal_cost=hub_link_costs,
                p_nom_extendable=[True] * len(hub_link_names),
            )


def add_crop_trade_hubs_and_links(
    n: pypsa.Network,
    trade_config: dict,
    regions_gdf: gpd.GeoDataFrame,
    countries: list,
    crop_list: list,
) -> None:
    """Add crop trading hubs and connect crop buses via hubs."""

    _add_trade_hubs_and_links(
        n,
        trade_config,
        regions_gdf,
        countries,
        crop_list,
        hub_count_key="crop_hubs",
        marginal_cost_key="crop_trade_marginal_cost_per_km",
        cost_categories_key="crop_trade_cost_categories",
        default_cost_key="crop_default_trade_cost_per_km",
        category_item_key="crops",
        non_tradable_key="non_tradable_crops",
        bus_prefix="crop_",
        carrier_prefix="crop_",
        hub_name_prefix="hub",
        link_name_prefix="trade",
        log_label="crop",
    )


def add_animal_product_trade_hubs_and_links(
    n: pypsa.Network,
    trade_config: dict,
    regions_gdf: gpd.GeoDataFrame,
    countries: list,
    animal_product_list: list,
) -> None:
    """Add trading hubs and links for animal products."""

    _add_trade_hubs_and_links(
        n,
        trade_config,
        regions_gdf,
        countries,
        animal_product_list,
        hub_count_key="animal_product_hubs",
        marginal_cost_key="animal_product_trade_marginal_cost_per_km",
        cost_categories_key="animal_product_trade_cost_categories",
        default_cost_key="animal_product_default_trade_cost_per_km",
        category_item_key="products",
        non_tradable_key="non_tradable_animal_products",
        bus_prefix="food_",
        carrier_prefix="food_",
        hub_name_prefix="hub_food",
        link_name_prefix="trade_food",
        log_label="animal product",
    )


if __name__ == "__main__":
    read_csv = functools.partial(pd.read_csv, comment="#")

    validation_cfg = snakemake.config.get("validation", {})  # type: ignore[attr-defined]
    use_actual_production = bool(validation_cfg.get("use_actual_production", False))

    # Read fertilizer N application rates (kg N/ha/year for high-input agriculture)
    fertilizer_n_rates = read_csv(snakemake.input.fertilizer_n_rates, index_col="crop")[
        "n_rate_kg_per_ha"
    ].to_dict()

    # Read food conversion data
    foods = read_csv(snakemake.input.foods)
    if not foods.empty:
        foods["food"] = foods["food"].astype(str).str.strip()
        foods["crop"] = foods["crop"].astype(str).str.strip()
        foods["factor"] = pd.to_numeric(foods["factor"], errors="coerce")
    edible_portion_df = read_csv(snakemake.input.edible_portion)
    moisture_df = read_csv(snakemake.input.moisture_content)

    # Read food groups data
    food_groups = read_csv(snakemake.input.food_groups)

    # Read nutrition data
    nutrition = read_csv(snakemake.input.nutrition, index_col=["food", "nutrient"])

    # Read categorized feed data
    ruminant_feed_categories = read_csv(snakemake.input.ruminant_feed_categories)
    ruminant_feed_mapping = read_csv(snakemake.input.ruminant_feed_mapping)
    monogastric_feed_categories = read_csv(snakemake.input.monogastric_feed_categories)
    monogastric_feed_mapping = read_csv(snakemake.input.monogastric_feed_mapping)

    # Read crop residue yields (may be empty if no residues available)
    residue_tables = {
        str(key): path
        for key, path in snakemake.input.items()
        if str(key).startswith("residue_")
    }
    residue_frames: list[pd.DataFrame] = []
    for path in residue_tables.values():
        df = read_csv(path)
        if not df.empty:
            residue_frames.append(df)

    if residue_frames:
        residue_yields = pd.concat(residue_frames, ignore_index=True)
        residue_feed_items = (
            residue_yields["feed_item"]
            .dropna()
            .astype(str)
            .sort_values()
            .unique()
            .tolist()
        )
        residue_lookup = {}
        for row in residue_yields.itertuples(index=False):
            feed_item = getattr(row, "feed_item", "")
            if not isinstance(feed_item, str) or not feed_item:
                continue
            key = (
                str(row.crop),
                str(row.water_supply),
                str(row.region),
                int(row.resource_class),
            )
            residue_lookup.setdefault(key, {})[feed_item] = float(
                getattr(row, "residue_yield_t_per_ha", 0.0)
            )
    else:
        residue_feed_items = []
        residue_lookup = {}

    # Read feed requirements for animal products (feed pools -> foods)
    feed_to_products = read_csv(snakemake.input.feed_to_products)

    # Read manure CH4 emission factors
    manure_ch4_emissions = read_csv(snakemake.input.manure_ch4_emissions)

    # Read food loss & waste fractions per country and food group
    food_loss_waste = read_csv(snakemake.input.food_loss_waste)
    if not food_loss_waste.empty:
        food_loss_waste["country"] = food_loss_waste["country"].astype(str).str.upper()
        food_loss_waste["food_group"] = food_loss_waste["food_group"].astype(str)

    irrigation_cfg = snakemake.config["irrigation"]["irrigated_crops"]  # type: ignore[index]
    if irrigation_cfg == "all":
        expected_irrigated_crops = set(snakemake.params.crops)
    else:
        expected_irrigated_crops = set(map(str, irrigation_cfg))

    # Read yields data for each configured crop and water supply
    yields_data: dict[str, pd.DataFrame] = {}
    for crop in snakemake.params.crops:
        expected_supplies = ["r"]
        if crop in expected_irrigated_crops:
            expected_supplies.append("i")

        for ws in expected_supplies:
            yields_key = f"{crop}_yield_{ws}"
            try:
                path = snakemake.input[yields_key]
            except AttributeError as exc:
                supply_label = "irrigated" if ws == "i" else "rainfed"
                raise ValueError(
                    f"Missing {supply_label} yield input for crop '{crop}'. Ensure the crop yield preprocessing "
                    f"step produced '{yields_key}'."
                ) from exc

            yields_df, var_units = _load_crop_yield_table(path)
            yield_unit = var_units.get("yield")
            if yield_unit != "t/ha (DM)":
                raise ValueError(
                    f"Unexpected unit for 'yield' in '{path}': expected 't/ha (DM)', found '{yield_unit}'"
                )
            area_unit = var_units.get("suitable_area")
            if area_unit != "ha":
                raise ValueError(
                    f"Unexpected unit for 'suitable_area' in '{path}': expected 'ha', found '{area_unit}'"
                )
            if ws == "i":
                water_unit = var_units.get("water_requirement_m3_per_ha")
                if water_unit not in {None, np.nan, "m^3/ha"}:
                    raise ValueError(
                        f"Unexpected unit for 'water_requirement_m3_per_ha' in '{path}': "
                        f"expected 'm^3/ha', found '{water_unit}'"
                    )
            yields_data[yields_key] = yields_df
            logger.info(
                "Loaded yields for %s (%s): %d rows",
                crop,
                "irrigated" if ws == "i" else "rainfed",
                len(yields_df),
            )

    harvested_area_data: dict[str, pd.DataFrame] = {}
    if use_actual_production:
        for crop in snakemake.params.crops:
            expected_supplies = ["r"]
            if crop in expected_irrigated_crops:
                expected_supplies.append("i")
            for ws in expected_supplies:
                harvest_key = f"{crop}_harvested_{ws}"
                path = getattr(snakemake.input, harvest_key, None)
                if path is None:
                    raise ValueError(
                        f"Missing harvested area input for crop '{crop}' ({'irrigated' if ws == 'i' else 'rainfed'}). "
                        "Ensure the harvested area preprocessing step produced the expected files."
                    )
                harvest_df, harvest_units = _load_crop_yield_table(path)
                area_unit = harvest_units.get("harvested_area")
                if area_unit != "ha":
                    raise ValueError(
                        f"Unexpected unit for 'harvested_area' in '{path}': expected 'ha', found '{area_unit}'"
                    )
                harvested_area_data[harvest_key] = harvest_df
                logger.info(
                    "Loaded harvested area for %s (%s): %d rows",
                    crop,
                    "irrigated" if ws == "i" else "rainfed",
                    len(harvest_df),
                )

    # Read regions
    regions_df = gpd.read_file(snakemake.input.regions)

    # Load class-level land areas
    land_class_df = read_csv(snakemake.input.land_area_by_class)
    # Expect columns: region, water_supply, resource_class, area_ha
    land_class_df = land_class_df.set_index(
        ["region", "water_supply", "resource_class"]
    ).sort_index()

    multi_cropping_area_df = read_csv(snakemake.input.multi_cropping_area)
    multi_cropping_cycle_df = read_csv(snakemake.input.multi_cropping_yields)

    luc_lef_lookup: dict[tuple[str, int, str, str], float] = {}
    carbon_price = float(snakemake.params.emissions["ghg_price"])
    ch4_to_co2_factor = float(snakemake.params.emissions["ch4_to_co2_factor"])
    n2o_to_co2_factor = float(snakemake.params.emissions["n2o_to_co2_factor"])
    if ch4_to_co2_factor <= 0.0:
        raise ValueError("`emissions.ch4_to_co2_factor` must be positive.")
    try:
        luc_coefficients_path = snakemake.input.luc_carbon_coefficients
        luc_coeff_df = read_csv(luc_coefficients_path)
        if not luc_coeff_df.empty:
            luc_lef_lookup = _build_luc_lef_lookup(luc_coeff_df)
            logger.info(
                "Loaded LUC LEFs for %d (region, class, water, use) combinations",
                len(luc_lef_lookup),
            )
        else:
            logger.warning(
                "LUC carbon coefficients file is empty; skipping LUC emission adjustments"
            )
    except (AttributeError, FileNotFoundError) as e:
        logger.info(
            "LUC carbon coefficients not available (%s); skipping LUC emission adjustments",
            type(e).__name__,
        )

    land_rainfed_df = land_class_df.xs("r", level="water_supply").copy()
    grassland_df = pd.DataFrame()
    current_grassland_area_df: pd.DataFrame | None = None
    grazing_only_area_series: pd.Series | None = None
    if snakemake.params.grazing["enabled"]:
        grassland_df = read_csv(
            snakemake.input.grassland_yields, index_col=["region", "resource_class"]
        ).sort_index()
        if use_actual_production:
            current_grassland_area_df = read_csv(snakemake.input.current_grassland_area)
        grazing_only_area_df = read_csv(snakemake.input.grazing_only_land)
        if not grazing_only_area_df.empty:
            grazing_only_area_series = (
                grazing_only_area_df.set_index(["region", "resource_class"])["area_ha"]
                .astype(float)
                .sort_index()
            )

    blue_water_availability_df = read_csv(snakemake.input.blue_water_availability)
    monthly_region_water_df = read_csv(snakemake.input.monthly_region_water)
    region_growing_water_df = read_csv(snakemake.input.growing_season_water)

    logger.info(
        "Loaded blue water availability data: %d basin-month pairs",
        len(blue_water_availability_df),
    )
    logger.info(
        "Loaded monthly region water availability: %d rows",
        len(monthly_region_water_df),
    )
    logger.info(
        "Loaded region growing-season water availability: %d regions",
        region_growing_water_df.shape[0],
    )

    # Load population per country for planning horizon
    population_df = read_csv(snakemake.input.population)
    # Expect columns: iso3, country, year, population
    # Select only configured countries and validate coverage
    cfg_countries = list(snakemake.params.countries)
    pop_map = population_df.set_index("iso3")["population"].reindex(cfg_countries)
    missing = pop_map[pop_map.isna()].index.tolist()
    if missing:
        raise ValueError("Missing population for countries: " + ", ".join(missing))
    # population series indexed by country code (ISO3)
    population = pop_map.astype(float)

    diet_cfg = snakemake.params.get("diet", {})
    health_reference_year = int(snakemake.params.health_reference_year)
    enforce_baseline = bool(diet_cfg.get("enforce_gdd_baseline", False))
    baseline_equals: dict[str, dict[str, float]] = {}
    if enforce_baseline:
        baseline_age = str(diet_cfg.get("baseline_age", "All ages"))
        baseline_year = diet_cfg.get("baseline_reference_year", health_reference_year)
        if baseline_year is not None:
            baseline_year = int(baseline_year)
        diet_table_path = snakemake.input.get("baseline_diet")
        if diet_table_path is None:
            raise ValueError(
                "Baseline diet enforcement requested but no GDD intake table provided"
            )
        diet_table = read_csv(diet_table_path)
        baseline_equals = _build_food_group_equals_from_baseline(
            diet_table,
            cfg_countries,
            food_groups["group"].unique(),
            baseline_age=baseline_age,
            reference_year=baseline_year,
        )
        logger.info(
            "Enforcing baseline diet using GDD data (age=%s, year=%s)",
            baseline_age,
            baseline_year,
        )

    region_to_country = regions_df.set_index("region")["country"]
    # Warn if any configured countries are missing from regions
    present_countries = set(region_to_country.unique())
    missing_in_regions = [c for c in cfg_countries if c not in present_countries]
    if missing_in_regions:
        logger.warning(
            "Configured countries missing from regions and may be disconnected: %s",
            ", ".join(sorted(missing_in_regions)),
        )
    # Keep only regions whose country is in configured countries
    region_to_country = region_to_country[region_to_country.isin(cfg_countries)]

    regions = sorted(region_to_country.index.unique())

    region_water_limits = (
        region_growing_water_df.set_index("region")["growing_season_water_available_m3"]
        .reindex(regions)
        .fillna(0.0)
    )

    irrigated_regions: set[str] = set()
    for key, df in yields_data.items():
        if key.endswith("_yield_i"):
            irrigated_regions.update(df.index.get_level_values("region"))

    land_regions = set(land_class_df.index.get_level_values("region"))
    water_bus_regions = sorted(
        set(region_water_limits.index)
        .union(irrigated_regions)
        .intersection(land_regions)
    )

    logger.debug("Foods data:\n%s", foods.head())
    logger.debug("Food groups data:\n%s", food_groups.head())
    logger.debug("Nutrition data:\n%s", nutrition.head())

    # Read USDA production costs (USD/ha in base year dollars)
    # Note: These costs are averaged over 2015-2024 and inflation-adjusted.
    # Costs are per hectare of planted area, independent of yields.
    # Costs are split into per-year (annual fixed) and per-planting (variable per crop).
    costs_df = read_csv(snakemake.input.costs)
    base_year = int(snakemake.config["currency_base_year"])
    cost_per_year_column = f"cost_per_year_usd_{base_year}_per_ha"
    cost_per_planting_column = f"cost_per_planting_usd_{base_year}_per_ha"

    crop_costs_per_year = costs_df.set_index("crop")[cost_per_year_column].astype(float)
    crop_costs_per_planting = costs_df.set_index("crop")[
        cost_per_planting_column
    ].astype(float)

    # For crops with missing cost data, default to 0 (will be flagged in logs)
    # Fallbacks are handled in the retrieval script via crop_cost_fallbacks.yaml

    # Build the network (inlined)
    n = pypsa.Network()
    n.set_snapshots(["now"])
    n.name = "food-opt"

    crop_list = snakemake.params.crops
    animal_products_cfg = snakemake.params.animal_products
    animal_product_list = list(animal_products_cfg["include"])

    # Validate foods.csv structure
    if "pathway" not in foods.columns:
        raise ValueError(
            "foods.csv must contain a 'pathway' column. "
            "Update data/foods.csv to use the pathway-based format."
        )

    food_crops = set(foods.loc[foods["crop"].isin(crop_list), "crop"])
    crop_to_fresh_factor = _fresh_mass_conversion_factors(
        edible_portion_df, moisture_df, food_crops
    )

    base_food_list = foods.loc[foods["crop"].isin(crop_list), "food"].unique().tolist()
    food_list = sorted(set(base_food_list).union(animal_product_list))
    food_groups_clean = food_groups.dropna(subset=["food", "group"]).copy()
    food_groups_clean["food"] = food_groups_clean["food"].astype(str).str.strip()
    food_groups_clean["group"] = food_groups_clean["group"].astype(str).str.strip()
    duplicate_groups = (
        food_groups_clean.groupby("food")["group"].nunique().loc[lambda s: s > 1]
    )
    if not duplicate_groups.empty:
        raise ValueError(
            "Each food must map to a single food group. Conflicts for: "
            + ", ".join(duplicate_groups.index.tolist())
        )
    food_to_group = (
        food_groups_clean.drop_duplicates(subset=["food"])
        .set_index("food")["group"]
        .to_dict()
    )
    food_group_list = food_groups_clean.loc[
        food_groups_clean["food"].isin(food_list), "group"
    ].unique()

    if enforce_baseline:
        missing_pairs = [
            f"{country}:{group}"
            for group in food_group_list
            for country in cfg_countries
            if baseline_equals.get(str(group), {}).get(country) is None
        ]
        if missing_pairs:
            raise ValueError(
                "Baseline diet enforcement missing values for: "
                + ", ".join(sorted(missing_pairs)[:10])
            )

    macronutrient_cfg = snakemake.params.macronutrients
    nutrient_units = (
        nutrition.reset_index()
        .drop_duplicates(subset=["nutrient"])
        .set_index("nutrient")["unit"]
        .to_dict()
    )
    # All nutrients from nutrition data get buses (tracked but not necessarily constrained)
    all_nutrient_names = list(nutrient_units.keys())
    # Only configured macronutrients get constraints applied
    macronutrient_names = list(macronutrient_cfg.keys()) if macronutrient_cfg else []

    add_carriers_and_buses(
        n,
        crop_list,
        food_list,
        residue_feed_items,
        food_group_list,
        all_nutrient_names,
        nutrient_units,
        cfg_countries,
        regions,
        water_bus_regions,
    )
    add_primary_resources(
        n,
        snakemake.params.primary,
        region_water_limits,
        carbon_price,
        ch4_to_co2_factor,
        n2o_to_co2_factor,
        use_actual_production=use_actual_production,
    )
    synthetic_n2o_factor = float(
        snakemake.params.primary["fertilizer"].get("synthetic_n2o_factor", 0.010)
    )
    add_fertilizer_distribution_links(n, cfg_countries, synthetic_n2o_factor)

    # Add class-level land buses and generators (shared pools), replacing region-level caps
    # Apply same regional_limit factor per class pool
    land_cfg = snakemake.params.primary["land"]
    reg_limit = float(land_cfg["regional_limit"])
    land_slack_cost = float(land_cfg.get("slack_marginal_cost", 5e9))
    # Build all unique class buses
    bus_names = [f"land_{r}_class{int(k)}_{ws}" for (r, ws, k) in land_class_df.index]
    n.add("Bus", bus_names, carrier=["land"] * len(bus_names))
    n.add(
        "Generator",
        bus_names,
        bus=bus_names,
        carrier=["land"] * len(bus_names),
        p_nom_extendable=[True] * len(bus_names),
        p_nom_max=(reg_limit * land_class_df["area_ha"] / 1e6).values,  # ha → Mha
    )
    if use_actual_production:
        _add_land_slack_generators(n, bus_names, land_slack_cost)

    # Land that is unsuitable for crop production but usable for grazing-only
    # expansion. Derived from the ESA/GAEZ overlay prepared by
    # build_grazing_only_land.
    marginal_bus_names: list[str] = []
    if grazing_only_area_series is not None and not grazing_only_area_series.empty:
        marginal_bus_names = [
            f"land_marginal_{region}_class{int(cls)}"
            for region, cls in grazing_only_area_series.index
        ]
        n.add("Bus", marginal_bus_names, carrier=["land"] * len(marginal_bus_names))
        n.add(
            "Generator",
            marginal_bus_names,
            bus=marginal_bus_names,
            carrier=["land"] * len(marginal_bus_names),
            p_nom_extendable=[True] * len(marginal_bus_names),
            p_nom_max=(reg_limit * grazing_only_area_series.values / 1e6),
        )
        if use_actual_production:
            _add_land_slack_generators(n, marginal_bus_names, land_slack_cost)

    add_spared_land_links(
        n, land_class_df, luc_lef_lookup, grazing_only_area=grazing_only_area_series
    )
    add_regional_crop_production_links(
        n,
        crop_list,
        yields_data,
        region_to_country,
        set(cfg_countries),
        crop_costs_per_year,
        crop_costs_per_planting,
        luc_lef_lookup,
        residue_lookup,
        harvested_area_data=harvested_area_data if use_actual_production else None,
        use_actual_production=use_actual_production,
    )
    enable_multiple_cropping = (
        bool(snakemake.params.multiple_cropping) and not use_actual_production
    )
    if enable_multiple_cropping:
        add_multi_cropping_links(
            n,
            multi_cropping_area_df,
            multi_cropping_cycle_df,
            region_to_country,
            set(cfg_countries),
            crop_costs_per_year,
            crop_costs_per_planting,
            fertilizer_n_rates,
            residue_lookup,
            luc_lef_lookup,
        )
    elif use_actual_production:
        logger.info("Skipping multiple cropping links under actual production mode")
    if snakemake.params.grazing["enabled"]:
        add_grassland_feed_links(
            n,
            grassland_df,
            land_rainfed_df,
            region_to_country,
            set(cfg_countries),
            luc_lef_lookup,
            current_grassland_area=current_grassland_area_df,
            pasture_land_area=grazing_only_area_series,
            use_actual_production=use_actual_production,
        )
    add_food_conversion_links(
        n,
        food_list,
        foods,
        cfg_countries,
        crop_to_fresh_factor,
        food_to_group,
        food_loss_waste,
        snakemake.params.crops,
        snakemake.params.byproducts,
    )
    add_feed_supply_links(
        n,
        ruminant_feed_categories,
        ruminant_feed_mapping,
        monogastric_feed_categories,
        monogastric_feed_mapping,
        crop_list,
        food_list,
        residue_feed_items,
        cfg_countries,
    )
    add_feed_to_animal_product_links(
        n,
        animal_product_list,
        feed_to_products,
        ruminant_feed_categories,
        monogastric_feed_categories,
        manure_ch4_emissions,
        nutrition,
        snakemake.params.primary["fertilizer"],
        cfg_countries,
    )
    add_food_group_buses_and_loads(
        n,
        food_group_list,
        food_groups,
        snakemake.params.food_groups,
        cfg_countries,
        population,
        per_country_equal=baseline_equals,
    )
    add_macronutrient_loads(
        n,
        all_nutrient_names,
        macronutrient_cfg,
        cfg_countries,
        population,
        nutrient_units,
    )
    add_food_nutrition_links(
        n,
        food_list,
        foods,
        food_groups,
        nutrition,
        nutrient_units,
        cfg_countries,
        snakemake.params.byproducts,
    )

    # Add crop trading hubs and links (hierarchical trade network)
    add_crop_trade_hubs_and_links(
        n, snakemake.params.trade, regions_df, cfg_countries, list(crop_list)
    )
    add_animal_product_trade_hubs_and_links(
        n,
        snakemake.params.trade,
        regions_df,
        cfg_countries,
        animal_product_list,
    )

    logger.info("Network summary:")
    logger.info("Carriers: %d", len(n.carriers))
    logger.info("Buses: %d", len(n.buses))
    logger.info("Stores: %d", len(n.stores))
    logger.info("Links: %d", len(n.links))

    n.export_to_netcdf(snakemake.output.network)
