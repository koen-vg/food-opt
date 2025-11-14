# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Utility functions and helpers for food systems model building.

This module contains data loading helpers, unit conversion functions,
and other utility functions used across the model building process.
"""

from collections.abc import Sequence
import logging

import numpy as np
import pandas as pd

from . import constants

logger = logging.getLogger(__name__)


def _nutrient_kind(unit: str) -> str:
    try:
        return constants.SUPPORTED_NUTRITION_UNITS[unit]["kind"]
    except KeyError as exc:
        raise ValueError(f"Unsupported nutrition unit '{unit}'") from exc


def _nutrition_efficiency_factor(unit: str) -> float:
    try:
        return constants.SUPPORTED_NUTRITION_UNITS[unit]["efficiency_factor"]
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
        return value_per_person_per_day * population * constants.DAYS_PER_YEAR * 1e-12
    if kind == "energy":
        return (
            value_per_person_per_day
            * population
            * constants.DAYS_PER_YEAR
            * constants.KCAL_TO_MCAL
        )
    raise ValueError(f"Unsupported nutrient kind '{kind}' for unit '{unit}'")


def _per_capita_food_group_to_mt(
    value_per_person_per_day: float, population: float
) -> float:
    """Convert g/person/day to Mt/year for food group buses."""

    return value_per_person_per_day * population * constants.DAYS_PER_YEAR * 1e-12


def _log_food_group_target_summary(group: str, values: Sequence[float]) -> None:
    """Emit a concise log message for the enforced equality targets."""
    arr = np.array(values, dtype=float)
    logger.info(
        "Food group '%s': equality constraint for %d countries "
        "(median %.1f g/person/day, min %.1f, max %.1f)",
        group,
        len(values),
        float(np.median(arr)),
        float(np.min(arr)),
        float(np.max(arr)),
    )


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

        dry_fraction = 1 - moisture_fraction
        factor = edible_coeff / dry_fraction
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
