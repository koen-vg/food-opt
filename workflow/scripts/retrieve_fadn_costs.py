# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Retrieve FADN (Farm Accountancy Data Network) production cost data for EU crops.

This script processes the FADN Public Database from Zenodo (LAMASUS dataset) to extract
production costs per hectare for European agriculture, averaged across EU countries and
years. Costs are allocated to crops proportionally based on output value shares and split
into per-year (annual fixed) and per-planting (variable per crop) categories.

Inputs
- snakemake.input.data: CSV with FADN standard results at NUTS0 (country) level
- snakemake.input.mapping: YAML mapping FADN crop categories (SE codes) to model crops
- snakemake.input.hicp: CSV with EU HICP inflation indices (year, hicp)
- snakemake.input.ppp: CSV with PPP conversion rate (EUR per international $)
- snakemake.params.crops: list of crop names from config
- snakemake.params.base_year: base year for inflation adjustment (default: 2024)

Output
- snakemake.output.costs: CSV with columns:
    crop,fadn_category,n_years,n_countries,cost_per_year_usd_{base_year}_per_ha,cost_per_planting_usd_{base_year}_per_ha

Notes
- Per-year costs (annual fixed): Machinery, energy, contract work, depreciation, wages, interest
- Per-planting costs (variable): Seeds, crop protection, other crop-specific costs
- Costs explicitly EXCLUDED: Fertilizer (SE295), rent (SE375 - land opportunity cost)
- Costs are allocated proportionally to crops by output value share within each FADN category
- Inflation-adjusted using EU HICP, then converted to international $ using PPP rates
- PPP (Purchasing Power Parity) conversion accounts for price level differences between EUR and USD
- Averaged over 2015-2020 period (FADN data ends 2020)
"""

import logging

from logging_config import setup_script_logging
import pandas as pd
import yaml

# Logger will be configured in __main__ block
logger = logging.getLogger(__name__)

# Years to average over (FADN data available 2004-2020)
YEARS = list(range(2015, 2021))  # [2015, 2020] inclusive

# Conversion factor: hectares (data is already in ha)
HA_TO_HA = 1.0

# FADN cost variable definitions
# Per-year costs: Fixed annual costs that don't multiply with number of plantings
PER_YEAR_COSTS = {
    "SE340": "Machinery & building current costs",
    "SE345": "Energy",
    "SE350": "Contract work",
    "SE360": "Depreciation",
    "SE370": "Wages paid",
    "SE380": "Interest paid",  # Interest on operating capital
}

# Per-planting costs: Variable costs incurred for each crop planted
PER_PLANTING_COSTS = {
    "SE285": "Seeds and plants",
    "SE300": "Crop protection",
    "SE305": "Other crop specific costs",
}

# Explicitly EXCLUDE these (modeled endogenously)
EXCLUDE_COSTS = {
    "SE295": "Fertilisers",  # Modeled endogenously
    "SE375": "Rent paid",  # Land opportunity cost, modeled endogenously
}

# FADN crop output value variables (SE1xx codes)
CROP_OUTPUT_VARS = {
    "SE140": "Cereals",
    "SE145": "Protein crops",
    "SE150": "Potatoes",
    "SE155": "Sugar beet",
    "SE160": "Oil-seed crops",
    "SE170": "Vegetables & flowers",
    "SE175": "Fruit trees and berries",
    "SE180": "Citrus fruit",
    "SE185": "Wine and grapes",
    "SE190": "Olives & olive oil",
}


def load_hicp_data(hicp_path: str) -> dict[int, float]:
    """Load EU HICP annual averages from CSV.

    Returns dict mapping year -> HICP value.
    """
    df = pd.read_csv(hicp_path)
    if not {"year", "hicp"}.issubset(df.columns):
        raise ValueError("HICP CSV must have columns: year, hicp")

    hicp_values = df.set_index("year")["hicp"].to_dict()
    logger.info(
        f"Loaded HICP data for years {min(hicp_values.keys())}-{max(hicp_values.keys())}"
    )
    return hicp_values


def load_ppp_rate(ppp_path: str) -> float:
    """Load PPP conversion rate from CSV.

    Returns PPP rate (EUR per international $).
    """
    df = pd.read_csv(ppp_path)
    if "avg_ppp_eur_per_intl_dollar" not in df.columns:
        raise ValueError("PPP CSV must have column: avg_ppp_eur_per_intl_dollar")

    ppp_rate = float(df["avg_ppp_eur_per_intl_dollar"].iloc[0])
    logger.info(f"Loaded PPP rate: {ppp_rate:.4f} EUR per international $")
    return ppp_rate


def inflate_eur_to_base_year(
    value: float, year: int, base_year: int, hicp_values: dict[int, float]
) -> float:
    """Adjust nominal EUR value to base year EUR using HICP."""
    if year not in hicp_values:
        raise ValueError(f"No HICP data for year {year}")
    if base_year not in hicp_values:
        raise ValueError(f"No HICP data for base year {base_year}")
    return value * (hicp_values[base_year] / hicp_values[year])


def convert_eur_to_intl_dollar(value_eur: float, ppp_rate: float) -> float:
    """Convert EUR to international $ using PPP rate.

    PPP rate is EUR per international $, so to convert EUR to international $:
    international $ = EUR / (EUR per international $)
    """
    return value_eur / ppp_rate


def process_fadn_costs(
    fadn_df: pd.DataFrame,
    crop_mapping: dict,
    base_year: int,
    ppp_rate: float,
    hicp_values: dict[int, float],
) -> pd.DataFrame:
    """
    Extract and process FADN cost data, allocating to model crops.

    Returns DataFrame with columns: crop, fadn_category, n_years, n_countries,
    cost_per_year_usd_{base_year}_per_ha, cost_per_planting_usd_{base_year}_per_ha
    """
    # Filter to relevant years
    fadn_df = fadn_df[fadn_df["year"].isin(YEARS)].copy()

    # Remove rows with missing data
    fadn_df = fadn_df[fadn_df["SE025"].notna()].copy()  # SE025 = Total UAA

    if fadn_df.empty:
        logger.warning(f"No FADN data available for years {YEARS}")
        return pd.DataFrame()

    logger.info(
        f"Processing FADN data: {len(fadn_df)} country-year observations "
        f"from {fadn_df['year'].min()}-{fadn_df['year'].max()}"
    )

    # Calculate cost allocation for each row (country-year)
    results = []

    for _, row in fadn_df.iterrows():
        year = int(row["year"])
        country = row["NUTS"]
        total_uaa = row["SE025"]  # Total Utilized Agricultural Area (ha)

        # Calculate total crop output value and output shares by category
        crop_outputs = {}
        total_crop_output = 0.0
        for se_code in CROP_OUTPUT_VARS:
            if se_code in row.index and pd.notna(row[se_code]):
                output_value = float(row[se_code])
                if output_value > 0:
                    crop_outputs[se_code] = output_value
                    total_crop_output += output_value

        if total_crop_output == 0:
            logger.debug(f"No crop output for {country} {year}, skipping")
            continue

        # Calculate output shares
        output_shares = {
            se_code: value / total_crop_output
            for se_code, value in crop_outputs.items()
        }

        # Sum per-year and per-planting costs
        total_per_year = sum(
            float(row.get(se_code, 0) or 0) for se_code in PER_YEAR_COSTS
        )
        total_per_planting = sum(
            float(row.get(se_code, 0) or 0) for se_code in PER_PLANTING_COSTS
        )

        # Allocate costs to each crop category proportionally by output share
        for se_code, share in output_shares.items():
            if se_code not in crop_mapping:
                logger.debug(f"No mapping for FADN category {se_code}, skipping")
                continue

            # Costs allocated to this category (EUR)
            category_per_year_eur = total_per_year * share
            category_per_planting_eur = total_per_planting * share

            # Inflate to base year EUR
            category_per_year_eur_base = inflate_eur_to_base_year(
                category_per_year_eur, year, base_year, hicp_values
            )
            category_per_planting_eur_base = inflate_eur_to_base_year(
                category_per_planting_eur, year, base_year, hicp_values
            )

            # Convert to international $ using PPP
            category_per_year_usd = convert_eur_to_intl_dollar(
                category_per_year_eur_base, ppp_rate
            )
            category_per_planting_usd = convert_eur_to_intl_dollar(
                category_per_planting_eur_base, ppp_rate
            )

            # Divide by the category's share of UAA to get per-hectare costs.
            # Use output share as a proxy for area share so the allocation shares
            # cancel out instead of shrinking the result.
            category_uaa = total_uaa * share
            if category_uaa > 0:
                cost_per_year_per_ha = category_per_year_usd / category_uaa
                cost_per_planting_per_ha = category_per_planting_usd / category_uaa
            else:
                cost_per_year_per_ha = 0.0
                cost_per_planting_per_ha = 0.0

            results.append(
                {
                    "fadn_category": se_code,
                    "country": country,
                    "year": year,
                    "output_share": share,
                    "cost_per_year_usd_per_ha": cost_per_year_per_ha,
                    "cost_per_planting_usd_per_ha": cost_per_planting_per_ha,
                }
            )

    if not results:
        logger.warning("No valid FADN cost data after processing")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    # Average across countries and years for each FADN category
    category_costs = (
        results_df.groupby("fadn_category")
        .agg(
            {
                "cost_per_year_usd_per_ha": "mean",
                "cost_per_planting_usd_per_ha": "mean",
                "year": "count",
                "country": "nunique",
            }
        )
        .rename(columns={"year": "n_observations", "country": "n_countries"})
    )

    logger.info(f"Averaged costs across {len(results_df)} observations")
    for cat, row in category_costs.iterrows():
        logger.info(
            f"  {CROP_OUTPUT_VARS.get(cat, cat)}: "
            f"per-year=${row['cost_per_year_usd_per_ha']:.2f}/ha, "
            f"per-planting=${row['cost_per_planting_usd_per_ha']:.2f}/ha "
            f"({row['n_countries']} countries, {row['n_observations']} obs)"
        )

    # Map to individual crops
    crop_costs = []
    cost_per_year_column = f"cost_per_year_usd_{base_year}_per_ha"
    cost_per_planting_column = f"cost_per_planting_usd_{base_year}_per_ha"

    for se_code, mapping in crop_mapping.items():
        if se_code not in category_costs.index:
            logger.debug(f"No cost data for FADN category {se_code}")
            continue

        cat_data = category_costs.loc[se_code]
        crops = mapping["crops"]

        for crop in crops:
            crop_costs.append(
                {
                    "crop": crop,
                    "fadn_category": se_code,
                    "n_years": len(YEARS),
                    "n_countries": int(cat_data["n_countries"]),
                    cost_per_year_column: cat_data["cost_per_year_usd_per_ha"],
                    cost_per_planting_column: cat_data["cost_per_planting_usd_per_ha"],
                }
            )

    return pd.DataFrame(crop_costs)


def main():
    fadn_data_path: str = snakemake.input.data  # type: ignore[name-defined]
    mapping_path: str = snakemake.input.mapping  # type: ignore[name-defined]
    hicp_path: str = snakemake.input.hicp  # type: ignore[name-defined]
    ppp_path: str = snakemake.input.ppp  # type: ignore[name-defined]
    base_year: int = int(snakemake.params.base_year)  # type: ignore[name-defined]
    out_path: str = snakemake.output.costs  # type: ignore[name-defined]

    # Load inflation and PPP data
    hicp_values = load_hicp_data(hicp_path)
    ppp_rate = load_ppp_rate(ppp_path)

    # Load FADN data
    fadn_df = pd.read_csv(fadn_data_path)
    logger.info(f"Loaded FADN data: {len(fadn_df)} rows")

    # Load crop mapping
    with open(mapping_path) as f:
        crop_mapping = yaml.safe_load(f)
    logger.info(f"Loaded mapping for {len(crop_mapping)} FADN categories")

    # Process costs
    costs_df = process_fadn_costs(
        fadn_df, crop_mapping, base_year, ppp_rate, hicp_values
    )

    if costs_df.empty:
        logger.warning("No FADN cost data produced")
        # Create empty output with correct columns for downstream compatibility
        cost_per_year_column = f"cost_per_year_usd_{base_year}_per_ha"
        cost_per_planting_column = f"cost_per_planting_usd_{base_year}_per_ha"
        costs_df = pd.DataFrame(
            columns=[
                "crop",
                "fadn_category",
                "n_years",
                "n_countries",
                cost_per_year_column,
                cost_per_planting_column,
            ]
        )

    # Write output
    costs_df.to_csv(out_path, index=False)
    logger.info(f"Wrote FADN cost data for {len(costs_df)} crops to {out_path}")


if __name__ == "__main__":
    # Configure logging
    logger = setup_script_logging(log_file=snakemake.log[0] if snakemake.log else None)  # type: ignore[name-defined]

    main()
