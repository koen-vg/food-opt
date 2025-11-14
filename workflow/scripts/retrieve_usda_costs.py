# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Retrieve USDA production cost data (USD/acre) for configured crops and
compute a 10-year inflation-adjusted average (2015-2024) in base year dollars.

This script downloads CSV files from USDA ERS Cost and Returns data and
extracts production costs per acre, excluding costs that are modeled
endogenously (fertilizer, land, irrigation water). Costs are split into
per-year (annual fixed) and per-planting (variable per crop) categories
to accurately model multiple cropping economies of scale.

For detailed documentation on cost modeling, data sources, and workflow integration,
see docs/crop_production.rst (Production Costs section) and docs/data_sources.rst
(USDA Cost and Returns Data section).

Inputs
- snakemake.input.sources: CSV with columns (crop, url) mapping to USDA data URLs
- snakemake.input.cpi: CSV with CPI-U annual averages (year, cpi_u)
- snakemake.params.base_year: base year for inflation adjustment

Output
- snakemake.output.costs: CSV with columns:
    crop,n_years,cost_per_year_usd_{base_year}_per_ha,cost_per_planting_usd_{base_year}_per_ha

Notes
- Per-year costs (annual fixed): Machinery depreciation, farm overhead, taxes/insurance
- Per-planting costs (variable): Seed, chemicals, labor, fuel, repairs, custom services
- Costs explicitly EXCLUDED: Fertilizer, land opportunity costs, irrigation water
- Inflation-adjusted using US CPI-U from BLS
- Converted from USD/acre to USD/ha (factor: 2.47105)
- Only U.S. total (not regional) data is used
- Only crops with actual USDA data are included (no fallbacks)
"""

from io import StringIO
import logging

from logging_config import setup_script_logging
import pandas as pd
import requests

# Logger will be configured in __main__ block
logger = logging.getLogger(__name__)

# Years to average over
YEARS = list(range(2015, 2025))  # inclusive [2015, 2024]

# Conversion factor: acres to hectares
ACRES_TO_HA = 0.404686
HA_PER_ACRE = 1.0 / ACRES_TO_HA  # 2.47105

# Cost categories split into per-year and per-planting
# Per-year costs: Fixed annual costs that don't multiply with number of plantings
PER_YEAR_COSTS = {
    "Capital recovery of machinery and equipment",  # Annual depreciation
    "General farm overhead",  # Fixed annual overhead
    "Taxes and insurance",  # Annual fixed costs
}

# Per-planting costs: Variable costs incurred for each crop planted
PER_PLANTING_COSTS = {
    # Operating costs
    "Chemicals",  # Each crop needs its own applications
    "Custom services",  # Hired for each planting/harvest operation
    "Fuel, lube, and electricity",  # Each crop requires field operations
    "Interest on operating capital",  # Borrowed for each crop cycle
    "Repairs",  # More plantings = more wear and tear
    "Seed",  # Each crop needs its own seed
    # Labor costs
    "Hired labor",  # Labor for each crop operation
    "Opportunity cost of unpaid labor",  # Time spent on each crop
}

# Explicitly EXCLUDE these
EXCLUDE_ITEMS = {
    "Fertilizer",  # Modeled endogenously
    "Opportunity cost of land",  # Modeled endogenously
    "Purchased irrigation water",  # Water modeled endogenously
}


def load_cpi_data(cpi_path: str) -> dict[int, float]:
    """Load CPI-U annual averages from CSV.

    Returns dict mapping year -> CPI value.
    """
    df = pd.read_csv(cpi_path)
    if not {"year", "cpi_u"}.issubset(df.columns):
        raise ValueError("CPI CSV must have columns: year, cpi_u")

    cpi_values = df.set_index("year")["cpi_u"].to_dict()
    logger.info(
        f"Loaded CPI data for years {min(cpi_values.keys())}-{max(cpi_values.keys())}"
    )
    return cpi_values


def inflate_to_base_year(
    value: float, year: int, base_year: int, cpi_values: dict[int, float]
) -> float:
    """Adjust nominal USD value to base year dollars using CPI."""
    if year not in cpi_values:
        raise ValueError(f"No CPI data for year {year}")
    if base_year not in cpi_values:
        raise ValueError(f"No CPI data for base year {base_year}")
    return value * (cpi_values[base_year] / cpi_values[year])


def fetch_usda_csv(url: str) -> pd.DataFrame:
    """Download USDA CSV from URL and return as DataFrame."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    # Use StringIO to parse the CSV content
    return pd.read_csv(StringIO(response.text))


def process_usda_crop_costs(
    df: pd.DataFrame, usda_crop: str, base_year: int, cpi_values: dict[int, float]
) -> dict:
    """
    Extract relevant cost data from USDA DataFrame for a single crop.

    Returns dict with:
        - n_years: number of years with data
        - cost_usd_base_year_per_ha: inflation-adjusted cost in base year USD per hectare
    """
    # Filter to U.S. total only
    df = df[df["Region"] == "U.S. total"].copy()

    # Filter to relevant years
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df[df["Year"].isin(YEARS)]

    if df.empty:
        logger.warning(f"No data for {usda_crop} in years {YEARS}")
        return {
            "n_years": 0,
            f"cost_per_year_usd_{base_year}_per_ha": float("nan"),
            f"cost_per_planting_usd_{base_year}_per_ha": float("nan"),
        }

    # Filter to cost categories we want to include
    all_costs = PER_YEAR_COSTS | PER_PLANTING_COSTS
    df = df[df["Item"].isin(all_costs)].copy()

    # Explicitly exclude certain items
    df = df[~df["Item"].isin(EXCLUDE_ITEMS)]

    # Extract numeric values
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Value"])

    if df.empty:
        logger.warning(f"No valid cost data for {usda_crop} after filtering")
        return {
            "n_years": 0,
            f"cost_per_year_usd_{base_year}_per_ha": float("nan"),
            f"cost_per_planting_usd_{base_year}_per_ha": float("nan"),
        }

    # Split into per-year and per-planting costs
    df_per_year = df[df["Item"].isin(PER_YEAR_COSTS)].copy()
    df_per_planting = df[df["Item"].isin(PER_PLANTING_COSTS)].copy()

    # Sum costs within each year
    yearly_per_year_costs = df_per_year.groupby("Year")["Value"].sum()
    yearly_per_planting_costs = df_per_planting.groupby("Year")["Value"].sum()

    # Inflate each year to base year dollars
    yearly_per_year_costs_base = pd.Series(
        {
            year: inflate_to_base_year(cost, int(year), base_year, cpi_values)
            for year, cost in yearly_per_year_costs.items()
        }
    )
    yearly_per_planting_costs_base = pd.Series(
        {
            year: inflate_to_base_year(cost, int(year), base_year, cpi_values)
            for year, cost in yearly_per_planting_costs.items()
        }
    )

    # Compute averages across years
    avg_per_year_per_acre = (
        yearly_per_year_costs_base.mean()
        if not yearly_per_year_costs_base.empty
        else 0.0
    )
    avg_per_planting_per_acre = (
        yearly_per_planting_costs_base.mean()
        if not yearly_per_planting_costs_base.empty
        else 0.0
    )

    # Convert from $/acre to $/ha
    avg_per_year_per_ha = avg_per_year_per_acre * HA_PER_ACRE
    avg_per_planting_per_ha = avg_per_planting_per_acre * HA_PER_ACRE

    n_years = len(df["Year"].unique())

    logger.info(
        f"{usda_crop}: {n_years} years, "
        f"per-year = ${avg_per_year_per_ha:.2f}/ha, "
        f"per-planting = ${avg_per_planting_per_ha:.2f}/ha ({base_year} USD)"
    )

    return {
        "n_years": n_years,
        f"cost_per_year_usd_{base_year}_per_ha": avg_per_year_per_ha,
        f"cost_per_planting_usd_{base_year}_per_ha": avg_per_planting_per_ha,
    }


def main():
    sources_path: str = snakemake.input.sources  # type: ignore[name-defined]
    cpi_path: str = snakemake.input.cpi  # type: ignore[name-defined]
    base_year: int = int(snakemake.params.base_year)  # type: ignore[name-defined]
    out_path: str = snakemake.output.costs  # type: ignore[name-defined]

    # Load CPI data for inflation adjustment
    cpi_values = load_cpi_data(cpi_path)

    # Load USDA source URLs (crop name maps directly to USDA crop)
    sources_df = pd.read_csv(sources_path, comment="#")
    if not {"crop", "url"}.issubset(sources_df.columns):
        raise ValueError("Sources file must have columns: crop, url")

    # Process only crops with USDA data
    results = []

    for _, row in sources_df.iterrows():
        crop = row["crop"]
        url = row["url"]

        logger.info(f"Fetching USDA data for {crop} from {url}")

        try:
            df = fetch_usda_csv(url)
            cost_data = process_usda_crop_costs(df, crop, base_year, cpi_values)

            # Only include if we got valid data
            if cost_data["n_years"] > 0:
                results.append(
                    {
                        "crop": crop,
                        **cost_data,
                    }
                )
            else:
                logger.warning(f"No valid data for {crop}, skipping")
        except Exception as e:
            logger.error(f"Failed to process {crop}: {e}, skipping")

    # Write output
    out_df = pd.DataFrame(results)
    out_df.to_csv(out_path, index=False)
    logger.info(f"Wrote cost data to {out_path}")


if __name__ == "__main__":
    # Configure logging
    logger = setup_script_logging(log_file=snakemake.log[0] if snakemake.log else None)

    main()
