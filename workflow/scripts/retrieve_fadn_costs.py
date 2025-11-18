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
- Costs are allocated to crop groups using specific area variables (SE codes) to capture intensity differences.
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
    years: list[int],
    per_year_costs: dict[str, str],
    per_planting_costs: dict[str, str],
    crop_groups: dict,
) -> pd.DataFrame:
    """
    Extract and process FADN cost data, allocating to model crops using specific area variables.
    """
    # Filter to relevant years
    fadn_df = fadn_df[fadn_df["year"].isin(years)].copy()

    # Remove rows with missing data
    fadn_df = fadn_df[fadn_df["SE025"].notna()].copy()  # SE025 = Total UAA

    if fadn_df.empty:
        logger.warning(f"No FADN data available for years {years}")
        return pd.DataFrame()

    logger.info(
        f"Processing FADN data: {len(fadn_df)} country-year observations "
        f"from {fadn_df['year'].min()}-{fadn_df['year'].max()}"
    )

    results = []

    for _, row in fadn_df.iterrows():
        year = int(row["year"])
        country = row["NUTS"]

        # 1. Calculate Total Farm Output (SE131) and Total Costs
        if "SE131" not in row or pd.isna(row["SE131"]) or float(row["SE131"]) == 0:
            continue

        total_output = float(row["SE131"])

        total_per_year = sum(
            float(row.get(se_code, 0) or 0) for se_code in per_year_costs
        )
        total_per_planting = sum(
            float(row.get(se_code, 0) or 0) for se_code in per_planting_costs
        )

        # 2. Process each Crop Group
        for group_name, config in crop_groups.items():
            area_code = config["area"]

            # Check if we have area data for this group
            if (
                area_code not in row
                or pd.isna(row[area_code])
                or float(row[area_code]) <= 0
            ):
                continue

            group_area_ha = float(row[area_code])

            # Sum output value for this group
            group_output_value = 0.0
            for out_code in config["outputs"]:
                if out_code in row and pd.notna(row[out_code]):
                    group_output_value += float(row[out_code])

            if group_output_value <= 0:
                continue

            # Calculate Output Share for this group
            group_share = group_output_value / total_output

            # Allocate Costs to Group (EUR)
            group_per_year_eur = total_per_year * group_share
            group_per_planting_eur = total_per_planting * group_share

            # Calculate Cost per Hectare (EUR/ha)
            # This correctly captures the intensity: High Value Share / Low Area -> High Cost/ha
            per_year_eur_ha = group_per_year_eur / group_area_ha
            per_planting_eur_ha = group_per_planting_eur / group_area_ha

            # Inflate and Convert to USD
            per_year_usd_ha = convert_eur_to_intl_dollar(
                inflate_eur_to_base_year(per_year_eur_ha, year, base_year, hicp_values),
                ppp_rate,
            )
            per_planting_usd_ha = convert_eur_to_intl_dollar(
                inflate_eur_to_base_year(
                    per_planting_eur_ha, year, base_year, hicp_values
                ),
                ppp_rate,
            )

            # Assign this cost to all FADN categories (SE codes) in this group
            for se_code in config["crops"]:
                # Only add if this specific SE code exists in our mapping
                if se_code in crop_mapping:
                    results.append(
                        {
                            "fadn_category": se_code,
                            "country": country,
                            "year": year,
                            "group": group_name,
                            "cost_per_year_usd_per_ha": per_year_usd_ha,
                            "cost_per_planting_usd_per_ha": per_planting_usd_ha,
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
            f"  {cat}: "
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
                    "n_years": len(years),
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
    cost_params: dict = snakemake.params.cost_params  # type: ignore[name-defined]
    averaging_period: dict = snakemake.params.averaging_period  # type: ignore[name-defined]
    out_path: str = snakemake.output.costs  # type: ignore[name-defined]

    # Extract params
    years = list(
        range(averaging_period["start_year"], averaging_period["end_year"] + 1)
    )
    per_year_costs = cost_params["per_year_costs"]
    per_planting_costs = cost_params["per_planting_costs"]
    crop_groups = cost_params["crop_groups"]

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
        fadn_df,
        crop_mapping,
        base_year,
        ppp_rate,
        hicp_values,
        years,
        per_year_costs,
        per_planting_costs,
        crop_groups,
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
