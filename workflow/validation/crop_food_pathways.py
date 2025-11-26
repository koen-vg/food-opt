# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Validation for crop-to-food processing pathways against config crops."""

from pathlib import Path

import pandas as pd
from pandera.pandas import Column, DataFrameSchema
from snakemake.logging import logger

FOODS_SCHEMA = DataFrameSchema(
    {
        "pathway": Column(str, nullable=False, coerce=True),
        "crop": Column(str, nullable=False, coerce=True),
        "food": Column(str, nullable=False, coerce=True),
        "factor": Column(float, nullable=False, coerce=True),
        "description": Column(str, nullable=False, coerce=True),
    },
    strict=True,
    coerce=True,
    # Each (pathway, food) combination should be unique
    unique=["pathway", "food"],
)


def validate_crop_food_pathways(config: dict, project_root: Path) -> None:
    """Validate that foods.csv references only crops defined in config.

    Also validates CSV structure and warns about crops without pathways.

    Note: Basic config structure is validated by JSON Schema.
    """
    csv_path = project_root / "data" / "foods.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected data file at {csv_path}")

    df = FOODS_SCHEMA.validate(pd.read_csv(csv_path, comment="#"))

    # Check all crops referenced in foods.csv exist in config
    config_crops = set(config["crops"])
    csv_crops = set(df["crop"].unique())

    missing = sorted(csv_crops - config_crops)
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            f"foods.csv references crops not in config.crops: {missing_text}"
        )

    # Warn about crops in config without food pathways
    # (This could be intentional for feed-only crops like alfalfa)
    unused = sorted(config_crops - csv_crops)
    if unused:
        unused_text = ", ".join(unused)
        logger.warning(
            f"Crops in config without food pathways (feed-only?): {unused_text}"
        )

    # Check that byproducts listed in config appear as foods
    byproducts_cfg = set(config.get("byproducts", []))
    foods_in_csv = set(df["food"].unique())

    missing_byproducts = sorted(byproducts_cfg - foods_in_csv)
    if missing_byproducts:
        missing_text = ", ".join(missing_byproducts)
        raise ValueError(f"Byproducts in config not found in foods.csv: {missing_text}")
