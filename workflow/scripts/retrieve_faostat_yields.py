# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Retrieve FAOSTAT animal production and stock data to calculate yields.
Used to convert FADN per-head costs to per-tonne costs.
"""

import logging
from pathlib import Path

import faostat
from logging_config import setup_script_logging
import pandas as pd
import yaml

# Logger will be configured in __main__ block
logger = logging.getLogger(__name__)


def map_items_to_codes(item_names: list[str]) -> dict[str, str]:
    """Map item names to FAOSTAT item codes."""
    item_map = faostat.get_par("QCL", "Item")
    item_name_to_code = {}

    for name in item_names:
        # Try exact match first
        if name in item_map:
            item_name_to_code[name] = str(item_map[name])
        else:
            # Try case-insensitive match
            for k, v in item_map.items():
                if str(k).lower() == name.lower():
                    item_name_to_code[name] = str(v)
                    break
            else:
                logger.warning(f"Item '{name}' not found in FAOSTAT")

    if not item_name_to_code:
        raise RuntimeError("No valid items found")

    return item_name_to_code


def calculate_yields(
    df: pd.DataFrame,
    product_to_items: dict,
    element_codes: dict[str, list[str]],
    aggregate_limit: int,
) -> list[dict]:
    """Calculate yields from production and stock data."""
    # Normalize column names
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # Filter aggregates (Area Code < aggregate_limit)
    if "area_code" in df.columns:
        df = df[pd.to_numeric(df["area_code"], errors="coerce") < aggregate_limit]

    records = []

    # Group by Country, Year
    for (country, year), group in df.groupby(["area", "year"]):
        for product, info in product_to_items.items():
            prod_item_name = info.get("production_item")
            stock_item_name = info.get("stock_item")
            use_producing = info.get("use_element_producing_animals", False)

            production = 0.0
            stocks = 0.0

            # Get Production
            if prod_item_name:
                p_rows = group[
                    (group["item"].str.lower() == prod_item_name.lower())
                    & (
                        group["element_code"]
                        .astype(str)
                        .isin(element_codes["production"])
                    )
                ]
                if not p_rows.empty:
                    production = pd.to_numeric(p_rows["value"], errors="coerce").sum()

            # Get Stocks or Producing Animals
            s_rows = pd.DataFrame()
            if use_producing:
                s_rows = group[
                    (group["item"].str.lower() == prod_item_name.lower())
                    & (
                        group["element_code"]
                        .astype(str)
                        .isin(element_codes["producing_animals"])
                    )
                ]
            elif stock_item_name:
                s_rows = group[
                    (group["item"].str.lower() == stock_item_name.lower())
                    & (group["element_code"].astype(str).isin(element_codes["stocks"]))
                ]

            if not s_rows.empty:
                val = pd.to_numeric(s_rows["value"], errors="coerce").sum()
                unit = str(s_rows.iloc[0]["unit"]).lower()

                # Handle "1000 Head" units
                if "1000" in unit:
                    val *= 1000.0

                stocks = val

            # Calculate yield if we have both production and stocks
            if production > 0 and stocks > 0:
                records.append(
                    {
                        "country": country,
                        "year": year,
                        "product": product,
                        "yield_t_per_head": production / stocks,
                        "production_t": production,
                        "stocks_head": stocks,
                    }
                )

    return records


if __name__ == "__main__":
    setup_script_logging(log_file=snakemake.log[0] if snakemake.log else None)

    output_path = Path(snakemake.output[0])
    mapping_path = snakemake.input.mapping
    cost_params = snakemake.params.cost_params
    averaging_period = snakemake.params.averaging_period

    # Extract params
    years = [
        str(y)
        for y in range(averaging_period["start_year"], averaging_period["end_year"] + 1)
    ]
    element_codes = cost_params["element_codes"]
    # Convert all codes to strings to ensure matching works
    for key in element_codes:
        element_codes[key] = [str(c) for c in element_codes[key]]

    aggregate_limit = int(cost_params["aggregate_area_code_limit"])

    # Load mapping configuration
    with open(mapping_path) as f:
        product_to_items = yaml.safe_load(f)

    # Collect all unique items needed
    all_items = set()
    for info in product_to_items.values():
        if prod_item := info.get("production_item"):
            all_items.add(prod_item)
        if stock_item := info.get("stock_item"):
            all_items.add(stock_item)

    # Map item names to FAOSTAT codes
    item_name_to_code = map_items_to_codes(list(all_items))

    # Query FAOSTAT for all elements and items
    all_elem_codes = (
        element_codes["production"]
        + element_codes["stocks"]
        + element_codes["producing_animals"]
    )

    logger.info("Querying FAOSTAT...")
    df = faostat.get_data_df(
        "QCL",
        pars={
            "element": all_elem_codes,
            "item": list(item_name_to_code.values()),
            "year": years,
        },
        strval=True,
    )

    if df.empty:
        raise RuntimeError("FAOSTAT returned empty dataframe")

    # Calculate yields
    records = calculate_yields(df, product_to_items, element_codes, aggregate_limit)

    if not records:
        raise RuntimeError("No valid yield records calculated from FAOSTAT data")

    # Save results
    result_df = pd.DataFrame(records)
    result_df.to_csv(output_path, index=False)
    logger.info(f"Saved yields for {len(result_df)} records")
