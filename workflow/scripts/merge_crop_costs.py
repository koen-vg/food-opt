# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Merge crop production costs from multiple sources (USDA, FADN, etc.) into a single
unified cost dataset.

This script combines cost data from different sources, averages costs for crops with
multiple data points, and applies fallback mappings for crops without direct cost data.
The merging approach is designed to be extensible to additional cost sources in the future.

Inputs
- snakemake.input.cost_sources: List of cost CSV files from different sources (USDA, FADN, etc.)
- snakemake.input.fallbacks: YAML file mapping crops without data to similar crops with data
- snakemake.params.crops: List of all model crops from config
- snakemake.params.base_year: Base year for cost values (for column naming)

Output
- snakemake.output.costs: CSV with columns:
    crop,n_sources,cost_per_year_usd_{base_year}_per_ha,cost_per_planting_usd_{base_year}_per_ha

Notes
- For crops with data from multiple sources, costs are averaged
- For crops without data, fallback mappings are applied (e.g., rye -> wheat)
- Crops without data or fallback mappings receive zero costs (logged as warnings)
- The merging logic is source-agnostic and can accommodate additional sources
"""

import logging
from pathlib import Path

import pandas as pd
import yaml

from workflow.scripts.logging_config import setup_script_logging

# Logger will be configured in __main__ block
logger = logging.getLogger(__name__)


def load_cost_sources(source_paths: list[str], base_year: int) -> pd.DataFrame:
    """
    Load and concatenate cost data from multiple sources.

    Returns DataFrame with columns: crop, source, cost_per_year_usd_{base_year}_per_ha,
    cost_per_planting_usd_{base_year}_per_ha
    """
    cost_per_year_column = f"cost_per_year_usd_{base_year}_per_ha"
    cost_per_planting_column = f"cost_per_planting_usd_{base_year}_per_ha"

    all_costs = []

    for source_path in source_paths:
        source_name = Path(source_path).stem  # e.g., "usda_costs" or "fadn_costs"
        logger.info(f"Loading cost data from {source_name}")

        df = pd.read_csv(source_path)

        # Check for required columns
        if "crop" not in df.columns:
            logger.warning(f"No 'crop' column in {source_name}, skipping")
            continue

        if (
            cost_per_year_column not in df.columns
            or cost_per_planting_column not in df.columns
        ):
            logger.warning(f"Missing cost columns in {source_name}, skipping")
            continue

        # Extract relevant columns
        df_subset = df[["crop", cost_per_year_column, cost_per_planting_column]].copy()
        df_subset["source"] = source_name
        df_subset = df_subset.dropna(
            subset=[cost_per_year_column, cost_per_planting_column]
        )

        logger.info(f"  Loaded {len(df_subset)} cost entries from {source_name}")
        all_costs.append(df_subset)

    if not all_costs:
        logger.warning("No cost data loaded from any source")
        return pd.DataFrame()

    combined = pd.concat(all_costs, ignore_index=True)
    logger.info(f"Total cost entries across all sources: {len(combined)}")

    return combined


def merge_costs(
    costs_df: pd.DataFrame,
    all_crops: list[str],
    fallback_mapping: dict,
    base_year: int,
) -> pd.DataFrame:
    """
    Merge costs from multiple sources, apply fallbacks, and ensure all crops have cost data.

    Returns DataFrame with columns: crop, n_sources, cost_per_year_usd_{base_year}_per_ha,
    cost_per_planting_usd_{base_year}_per_ha
    """
    cost_per_year_column = f"cost_per_year_usd_{base_year}_per_ha"
    cost_per_planting_column = f"cost_per_planting_usd_{base_year}_per_ha"

    # Step 1: Average costs for crops with multiple sources
    averaged_costs = (
        costs_df.groupby("crop")
        .agg(
            {
                cost_per_year_column: "mean",
                cost_per_planting_column: "mean",
                "source": "count",  # Count number of sources
            }
        )
        .rename(columns={"source": "n_sources"})
        .reset_index()
    )

    logger.info(f"Averaged costs for {len(averaged_costs)} crops with direct data")

    for _, row in averaged_costs.iterrows():
        if row["n_sources"] > 1:
            logger.info(
                f"  {row['crop']}: averaged from {row['n_sources']} sources "
                f"(per-year=${row[cost_per_year_column]:.2f}/ha, "
                f"per-planting=${row[cost_per_planting_column]:.2f}/ha)"
            )

    # Step 2: Create cost dictionary for easy lookup
    cost_dict = averaged_costs.set_index("crop")[
        [cost_per_year_column, cost_per_planting_column, "n_sources"]
    ].to_dict("index")

    # Step 3: Process all crops, applying fallbacks where needed
    results = []

    for crop in all_crops:
        if crop in cost_dict:
            # Crop has direct data
            results.append(
                {
                    "crop": crop,
                    "n_sources": cost_dict[crop]["n_sources"],
                    cost_per_year_column: cost_dict[crop][cost_per_year_column],
                    cost_per_planting_column: cost_dict[crop][cost_per_planting_column],
                }
            )
        elif crop in fallback_mapping:
            # Apply fallback mapping
            fallback_crop = fallback_mapping[crop].get("usda_crop")

            if fallback_crop and fallback_crop in cost_dict:
                logger.info(f"Applying fallback for {crop} -> {fallback_crop}")
                results.append(
                    {
                        "crop": crop,
                        "n_sources": 0,  # Indicate this is a fallback
                        cost_per_year_column: cost_dict[fallback_crop][
                            cost_per_year_column
                        ],
                        cost_per_planting_column: cost_dict[fallback_crop][
                            cost_per_planting_column
                        ],
                    }
                )
            else:
                # Fallback crop also has no data
                logger.warning(
                    f"No cost data for {crop} or its fallback ({fallback_crop}), using zero costs"
                )
                results.append(
                    {
                        "crop": crop,
                        "n_sources": 0,
                        cost_per_year_column: 0.0,
                        cost_per_planting_column: 0.0,
                    }
                )
        else:
            # No data and no fallback
            logger.warning(f"No cost data or fallback for {crop}, using zero costs")
            results.append(
                {
                    "crop": crop,
                    "n_sources": 0,
                    cost_per_year_column: 0.0,
                    cost_per_planting_column: 0.0,
                }
            )

    return pd.DataFrame(results)


def main():
    cost_source_paths: list[str] = list(snakemake.input.cost_sources)  # type: ignore[name-defined]
    fallbacks_path: str = snakemake.input.fallbacks  # type: ignore[name-defined]
    all_crops: list[str] = list(snakemake.params.crops)  # type: ignore[name-defined]
    base_year: int = int(snakemake.params.base_year)  # type: ignore[name-defined]
    out_path: str = snakemake.output.costs  # type: ignore[name-defined]

    logger.info(
        f"Merging crop costs from {len(cost_source_paths)} sources for {len(all_crops)} crops"
    )

    # Load all cost sources
    costs_df = load_cost_sources(cost_source_paths, base_year)

    # Load fallback mappings
    with open(fallbacks_path) as f:
        fallback_mapping = yaml.safe_load(f)
    logger.info(f"Loaded fallback mappings for {len(fallback_mapping)} crops")

    # Merge and apply fallbacks
    merged_costs = merge_costs(costs_df, all_crops, fallback_mapping, base_year)

    # Write output
    merged_costs.to_csv(out_path, index=False)
    logger.info(f"Wrote merged cost data for {len(merged_costs)} crops to {out_path}")

    # Summary statistics
    with_direct_data = (merged_costs["n_sources"] > 0).sum()
    with_fallback = (
        (merged_costs["n_sources"] == 0)
        & (merged_costs[f"cost_per_year_usd_{base_year}_per_ha"] > 0)
    ).sum()
    with_zero = (merged_costs[f"cost_per_year_usd_{base_year}_per_ha"] == 0).sum()

    logger.info(
        f"Summary: {with_direct_data} crops with direct data, "
        f"{with_fallback} with fallback, {with_zero} with zero costs"
    )


if __name__ == "__main__":
    # Configure logging
    logger = setup_script_logging(log_file=snakemake.log[0] if snakemake.log else None)  # type: ignore[name-defined]

    main()
