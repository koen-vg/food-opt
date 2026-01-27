#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Retrieve raw food supply data from FAOSTAT Food Balance Sheets (FBS).

Fetches item-level supply data (kg/capita/year) for all items mapped in
the food item mapping file. This raw data is used for calculating
within-group food consumption ratios.

Input:
    - data/faostat_food_item_map.csv: Mapping from model foods to FBS items
    - FAOSTAT API (via faostat package)

Output:
    - CSV with columns: item_code, item_name, country, supply_kg_per_capita_year
      Raw per-capita food supply from FBS (not adjusted for waste)
"""

import logging
import sys

import faostat
import pandas as pd

from workflow.scripts.logging_config import setup_script_logging

logger = logging.getLogger(__name__)

# Proxy mapping for missing countries (same as retrieve_faostat_gdd_supplements.py)
FALLBACK_MAPPING = {
    "ASM": ["WSM", "USA"],  # American Samoa -> Samoa / USA
    "BEN": ["TGO", "BFA", "NGA"],  # Benin -> Togo / Burkina Faso / Nigeria
    "BRN": ["MYS", "SGP"],  # Brunei -> Malaysia
    "BTN": ["NPL", "IND"],  # Bhutan -> Nepal
    "CAF": ["TCD", "CMR", "COG"],  # Central African Republic -> Chad / Cameroon / Congo
    "ERI": ["ETH"],  # Eritrea -> Ethiopia
    "GNQ": ["GAB", "CMR"],  # Eq. Guinea -> Gabon
    "GUF": ["GUY", "SUR", "FRA"],  # Fr. Guiana
    "PRI": ["USA", "DOM"],  # Puerto Rico
    "PSE": ["JOR", "ISR"],  # Palestine
    "SDN": ["EGY", "ETH"],  # Sudan -> Egypt / Ethiopia
    "SSD": ["SDN", "ETH"],  # South Sudan
    "SOM": ["ETH"],  # Somalia
    "TWN": ["CHN"],  # Taiwan
    "XKX": ["SRB", "ALB"],  # Kosovo
    "ESH": ["MAR", "MRT"],  # Western Sahara
    "JPN": ["KOR", "CHN"],  # Japan -> South Korea / China
    "MLI": ["SEN", "BFA", "NER"],  # Mali -> Senegal / Burkina Faso / Niger
    "BDI": ["RWA", "TZA"],  # Burundi
    "COD": ["COG", "AGO"],  # DR Congo
    "SYR": ["JOR", "LBN"],  # Syria
    "TCD": ["SDN", "NER", "CMR"],  # Chad -> Sudan / Niger / Cameroon
    "TGO": ["GHA", "BFA"],  # Togo -> Ghana / Burkina Faso
    "VEN": ["COL", "BRA"],  # Venezuela
    "YEM": ["OMN", "SAU"],  # Yemen
}


def fetch_faostat_fbs_data(
    item_codes: list[int], countries: list[str], reference_year: int
) -> pd.DataFrame:
    """Fetch FBS supply data for specified item codes."""
    dataset = "FBS"

    # Get element code for food supply quantity
    try:
        elem_map = faostat.get_par(dataset, "Element")
        if "Food supply quantity (kg/capita/yr)" in elem_map:
            elem_code = elem_map["Food supply quantity (kg/capita/yr)"]
        else:
            logger.warning(
                "Element 'Food supply quantity (kg/capita/yr)' not found. Using 645."
            )
            elem_code = "645"
    except Exception:
        elem_code = "645"

    pars = {
        "element": elem_code,
        "year": str(reference_year),
        "item": ",".join(str(c) for c in item_codes),
    }

    logger.info(
        "Fetching FAOSTAT FBS data for %d items, %d countries, year %d",
        len(item_codes),
        len(countries),
        reference_year,
    )
    try:
        df = faostat.get_data_df(
            dataset, pars=pars, coding={"area": "ISO3"}, strval=True
        )
    except Exception as e:
        logger.error("Failed to fetch FAOSTAT data: %s", e)
        sys.exit(1)

    return df


def main():
    countries = [str(c).upper() for c in snakemake.params.countries]
    reference_year = int(snakemake.params.reference_year)
    food_item_map_path = snakemake.input.food_item_map
    output_file = snakemake.output.fbs_items

    # Load food-to-FBS-item mapping
    food_map_df = pd.read_csv(food_item_map_path, comment="#")
    unique_item_codes = food_map_df["item_code"].dropna().astype(int).unique().tolist()

    if not unique_item_codes:
        raise ValueError("No item codes found in food item mapping file")

    logger.info("Found %d unique FBS item codes to fetch", len(unique_item_codes))

    # Fetch FBS data
    fao_df = fetch_faostat_fbs_data(unique_item_codes, countries, reference_year)

    if fao_df.empty:
        raise ValueError("FAOSTAT returned no data")

    # Identify columns
    iso_col = "Area Code"
    for col in fao_df.columns:
        if (
            ("iso" in col.lower() or "area" in col.lower())
            and not fao_df[col].empty
            and len(str(fao_df[col].iloc[0])) == 3
            and str(fao_df[col].iloc[0]).isalpha()
        ):
            iso_col = col
            break

    item_col = "Item Code"
    item_name_col = "Item"
    val_col = "Value"

    fao_df["country"] = fao_df[iso_col].astype(str).str.upper()
    fao_df = fao_df[fao_df["country"].isin(countries)]
    fao_df["supply_kg_per_capita_year"] = pd.to_numeric(
        fao_df[val_col], errors="coerce"
    ).fillna(0.0)
    fao_df["item_code"] = (
        pd.to_numeric(fao_df[item_col], errors="coerce").fillna(0).astype(int)
    )
    fao_df["item_name"] = fao_df[item_name_col].astype(str)

    # Build result DataFrame
    results = fao_df[
        ["item_code", "item_name", "country", "supply_kg_per_capita_year"]
    ].copy()

    # Handle missing countries via proxies
    present_countries = set(results["country"].unique())
    missing = set(countries) - present_countries

    if missing:
        logger.info(
            "Attempting to fill %d missing countries via proxies...", len(missing)
        )

        proxy_rows = []
        for iso in missing:
            proxies = FALLBACK_MAPPING.get(iso, [])
            filled = False
            for proxy in proxies:
                if proxy in present_countries:
                    logger.info("Filling %s using proxy %s", iso, proxy)
                    proxy_data = results[results["country"] == proxy].copy()
                    proxy_data["country"] = iso
                    proxy_rows.append(proxy_data)
                    filled = True
                    break
            if not filled:
                available_proxies = ", ".join(FALLBACK_MAPPING.get(iso, []))
                raise ValueError(
                    f"Missing FAOSTAT FBS data for country {iso}. "
                    f"Attempted proxies ({available_proxies}) had no data. "
                    f"Please add valid proxy countries to FALLBACK_MAPPING."
                )

        if proxy_rows:
            results = pd.concat([results, *proxy_rows], ignore_index=True)

    results.to_csv(output_file, index=False)
    logger.info(
        "Wrote %d rows (%d countries, %d items) to %s",
        len(results),
        results["country"].nunique(),
        results["item_code"].nunique(),
        output_file,
    )


if __name__ == "__main__":
    logger = setup_script_logging(log_file=snakemake.log[0] if snakemake.log else None)
    main()
