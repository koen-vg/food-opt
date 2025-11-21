#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Merge dietary intake data from multiple sources (GDD and FAOSTAT).

Input:
    - GDD dietary intake CSV
    - FAOSTAT dietary intake CSV

Output:
    - Combined dietary intake CSV
"""

import logging

from logging_config import setup_script_logging
import pandas as pd

# Logger will be configured in __main__ block
logger = logging.getLogger(__name__)


def main():
    gdd_file = snakemake.input.gdd
    fao_file = snakemake.input.faostat
    output_file = snakemake.output.diet

    logger.info(f"Reading GDD data from {gdd_file}")
    gdd_df = pd.read_csv(gdd_file)

    logger.info(f"Reading FAOSTAT data from {fao_file}")
    fao_df = pd.read_csv(fao_file)

    # Identify overlapping items
    gdd_items = set(gdd_df["item"].unique())
    fao_items = set(fao_df["item"].unique())

    overlap = gdd_items.intersection(fao_items)

    if overlap:
        logger.warning(
            f"Overlapping items found in both sources: {overlap}. FAOSTAT will take precedence."
        )
        # Remove overlapping items from GDD
        gdd_df = gdd_df[~gdd_df["item"].isin(overlap)]

    combined = pd.concat([gdd_df, fao_df], ignore_index=True)

    # Sort for consistency
    combined = combined.sort_values(["country", "item", "age"]).reset_index(drop=True)

    combined.to_csv(output_file, index=False)
    logger.info(f"Wrote {len(combined)} rows to {output_file}")


if __name__ == "__main__":
    logger = setup_script_logging(log_file=snakemake.log[0] if snakemake.log else None)
    main()
