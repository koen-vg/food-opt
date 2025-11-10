# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Common configuration variables and helper functions shared across Snakemake rules.

This file should be included first in the main Snakefile, before any other rule files.
"""

import csv

# Extract configuration name and relevant config sections
name = config["name"]
gaez_cfg = config["data"]["gaez"]
grazing_cfg = config.get("grazing", {})

# Load GAEZ crop code mapping from CSV
with open("data/gaez_crop_code_mapping.csv", newline="") as _gaez_mapping_file:
    _GAEZ_CODE_MAPPING = {
        row["crop_name"]: {
            "res02": row["res02_code"],
            "res05": row["res05_code"],
            "res06": row["res06_code"],
        }
        for row in csv.DictReader(_gaez_mapping_file)
    }


def get_gaez_code(crop_name: str, module: str) -> str:
    """Look up the GAEZ RES code for a given crop and module."""

    module_key = module.lower()
    if module_key not in {"res02", "res05", "res06"}:
        raise ValueError(f"Unknown GAEZ module '{module}'")

    try:
        code = _GAEZ_CODE_MAPPING[crop_name][module_key]
    except KeyError as exc:
        raise ValueError(f"Crop '{crop_name}' not found in mapping") from exc

    if not code:
        raise ValueError(f"Crop '{crop_name}' has no {module_key} code")

    return code.strip().upper()


def gaez_path(kind: str, water_supply: str, crop: str) -> str:
    """Return GAEZ v5 raster path for a given kind and water supply.

    kind: one of {"yield", "suitability", "water_requirement", "growing_season_start", "growing_season_length", "actual_yield", "harvested_area"}
    water_supply: "i" (irrigated) or "r" (rainfed)
    crop: crop name (e.g., "wheat")
    """
    ws = water_supply.lower()
    if ws not in {"i", "r"}:
        raise ValueError(f"Unsupported water supply '{water_supply}'")

    if kind == "actual_yield":
        return f"data/downloads/gaez_actual_yield_{ws}_{crop}.tif"
    if kind == "harvested_area":
        return f"data/downloads/gaez_harvested_area_{ws}_{crop}.tif"

    climate = gaez_cfg["climate_model"]
    period = gaez_cfg["period"]
    scenario = gaez_cfg["scenario"]
    input_level = gaez_cfg["input_level"]

    prefix_by_kind = {
        "yield": "data/downloads/gaez_yield",
        "water_requirement": "data/downloads/gaez_water",
        "suitability": "data/downloads/gaez_suitability",
        "multiple_cropping_zone": "data/downloads/gaez_multiple_cropping",
        "growing_season_start": "data/downloads/gaez_growing_season_start",
        "growing_season_length": "data/downloads/gaez_growing_season_length",
    }

    try:
        prefix = prefix_by_kind[kind]
    except KeyError as exc:
        raise ValueError(f"Unknown kind for gaez_path: {kind}") from exc

    if kind == "multiple_cropping_zone":
        return f"{prefix}_{climate}_{period}_{scenario}_{input_level}_{ws}.tif"

    return f"{prefix}_{climate}_{period}_{scenario}_{input_level}_{ws}_{crop}.tif"
