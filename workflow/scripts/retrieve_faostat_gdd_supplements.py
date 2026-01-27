#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Retrieve FAOSTAT supply data to supplement GDD dietary intake.

The Global Dietary Database (GDD) lacks data for certain food groups. This
script fetches supply data from FAOSTAT Food Balance Sheets (FBS) for:
- Dairy (milk, butter, cream - converted to milk equivalents)
- Poultry meat
- Vegetable oils

Values are converted to g/day per capita. These supplement GDD data in
merge_dietary_sources.py to create complete baseline dietary intake estimates.

Input:
    - FAOSTAT API (via faostat package)

Output:
    - CSV with columns: unit, item, country, age, year, value
      Values are raw food supply in g/day (not adjusted for waste)
"""

import logging
import sys

import faostat
import pandas as pd

from workflow.scripts.logging_config import setup_script_logging

# Logger will be configured in __main__ block
logger = logging.getLogger(__name__)

# FAOSTAT Item Codes (FBS)
FAO_ITEMS = {
    "poultry": [2734],  # Poultry Meat
    "oil": [2586],  # Vegetable Oils
    "dairy": [2848, 2740, 2743],  # Milk (excl butter), Butter/Ghee, Cream
    # 2848: Milk - Excluding Butter (Aggregated Fluid + Cheese + Yoghurt origin milk in FBS)
    # 2740: Butter, Ghee
    # 2743: Cream
}

# Milk→product extraction rates from FAO dairy; see
# https://www.fao.org/fileadmin/templates/ess/documents/methodology/tcf.pdf,
# commodity tree 57. Percent ranges below are extraction rates;
# milk-equivalent factors are their reciprocals. Butter/ghee: 4.7%
# yield → ~21.3 kg milk / kg butter Cream fresh: 15% yield → ~6.7 kg
# milk / kg cream
DAIRY_MILK_EQUIV_FACTORS = {
    2848: 1.0,  # Milk - Excluding Butter (already milk-equivalent)
    2740: 21.3,  # Butter/Ghee (cow milk commodity tree No. 57)
    2743: 6.7,  # Cream (fresh) milk-equivalent
}

# Standard Age Groups
AGE_GROUPS = [
    "0-1 years",
    "1-2 years",
    "2-5 years",
    "6-10 years",
    "11-74 years",
    "75+ years",
    "All ages",
]

# Proxy mapping for missing countries
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


def fetch_faostat_data(countries, reference_year):
    dataset = "FBS"
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

    item_codes = []
    for items in FAO_ITEMS.values():
        item_codes.extend(items)

    pars = {
        "element": elem_code,
        "year": str(reference_year),
        "item": ",".join(str(c) for c in item_codes),
    }

    logger.info(
        f"Fetching FAOSTAT FBS data for {len(countries)} countries, year {reference_year}"
    )
    try:
        df = faostat.get_data_df(
            dataset, pars=pars, coding={"area": "ISO3"}, strval=True
        )
    except Exception as e:
        logger.error(f"Failed to fetch FAOSTAT data: {e}")
        sys.exit(1)
    return df


def main():
    countries = snakemake.params.countries
    reference_year = snakemake.params.reference_year
    output_file = snakemake.output.supply

    fao_df = fetch_faostat_data(countries, reference_year)

    if fao_df.empty:
        logger.warning("FAOSTAT returned no data.")

    # Identify columns
    iso_col = "Area Code"  # Default
    for col in fao_df.columns:
        # Heuristic: check if column name suggests ISO codes and first value is 3-letter code
        if (
            ("iso" in col.lower() or "area" in col.lower())
            and not fao_df[col].empty
            and len(str(fao_df[col].iloc[0])) == 3
            and str(fao_df[col].iloc[0]).isalpha()
        ):
            iso_col = col
            break

    item_col = "Item Code"
    val_col = "Value"

    fao_df["iso3"] = fao_df[iso_col].astype(str).str.upper()
    fao_df = fao_df[fao_df["iso3"].isin(countries)]
    fao_df[val_col] = pd.to_numeric(fao_df[val_col], errors="coerce").fillna(0.0)
    fao_df[item_col] = (
        pd.to_numeric(fao_df[item_col], errors="coerce").fillna(0).astype(int)
    )

    results = []

    # Process present countries
    present_countries = fao_df["iso3"].unique()
    logger.info(f"Processing data for {len(present_countries)} countries...")

    for country, group_df in fao_df.groupby("iso3"):
        supplies = {}

        # Poultry
        poultry_rows = group_df[group_df[item_col].isin(FAO_ITEMS["poultry"])]
        supplies["poultry"] = poultry_rows[val_col].sum()

        # Oil
        oil_rows = group_df[group_df[item_col].isin(FAO_ITEMS["oil"])]
        supplies["oil"] = oil_rows[val_col].sum()

        # Dairy: convert butter/ghee and cream to milk equivalents using FAO commodity tree
        dairy_sum = 0.0
        for item_code in FAO_ITEMS["dairy"]:
            val = group_df[group_df[item_col] == item_code][val_col].sum()
            factor = DAIRY_MILK_EQUIV_FACTORS.get(item_code, 1.0)
            dairy_sum += val * factor
        supplies["dairy"] = dairy_sum

        for item, supply_kg in supplies.items():
            supply_g = (supply_kg * 1000.0) / 365.0
            unit = "g/day (milk equiv)" if item == "dairy" else "g/day (fresh wt)"

            for age in AGE_GROUPS:
                results.append(
                    {
                        "unit": unit,
                        "item": item,
                        "country": country,
                        "age": age,
                        "year": reference_year,
                        "value": supply_g,
                    }
                )

    # Handle missing countries
    missing = set(countries) - set(present_countries)
    if missing:
        logger.info(
            f"Attempting to fill {len(missing)} missing countries via proxies..."
        )

        # Build lookup
        data_by_country = {}
        for r in results:
            data_by_country.setdefault(r["country"], []).append(r)

        for iso in missing:
            proxies = FALLBACK_MAPPING.get(iso, [])
            filled = False
            for proxy in proxies:
                if proxy in data_by_country:
                    logger.info(f"Filling {iso} using proxy {proxy}")
                    for row in data_by_country[proxy]:
                        new_row = row.copy()
                        new_row["country"] = iso
                        results.append(new_row)
                    filled = True
                    break
            if not filled:
                logger.error(f"Could not fill {iso} - no proxy data available.")
                available_proxies = ", ".join(FALLBACK_MAPPING.get(iso, []))
                raise ValueError(
                    f"Missing FAOSTAT data for country {iso}. "
                    f"Attempted proxies ({available_proxies}) had no data. "
                    f"Please add valid proxy countries to FALLBACK_MAPPING or obtain data for {iso}."
                )

    pd.DataFrame(results).to_csv(output_file, index=False)
    logger.info(f"Wrote {len(results)} rows to {output_file}")


if __name__ == "__main__":
    logger = setup_script_logging(log_file=snakemake.log[0] if snakemake.log else None)
    main()
