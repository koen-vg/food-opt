"""
SPDX-FileCopyrightText: 2025 Koen van Greevenbroek

SPDX-License-Identifier: GPL-3.0-or-later

Retrieve FAO animal production statistics for validation constraints.

This script queries FAOSTAT for global production totals of major animal
products (dairy, beef, pork, poultry, eggs) to establish baseline ratios
for validation mode.
"""

import logging
from pathlib import Path

import faostat
from logging_config import setup_script_logging
import numpy as np
import pandas as pd

# Logger will be configured in __main__ block
logger = logging.getLogger(__name__)

# Mapping from model product names to FAOSTAT item names
ANIMAL_PRODUCT_MAPPING = {
    "dairy": "Raw milk of cattle",
    "meat-cattle": "Meat of cattle with the bone, fresh or chilled",
    "meat-pig": "Meat of pig with the bone, fresh or chilled",
    "meat-chicken": "Meat of chickens, fresh or chilled",
    "eggs": "Hen eggs in shell, fresh",
}


def _normalize(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _find_column(df: pd.DataFrame, candidates: set[str]) -> str:
    for column in df.columns:
        if _normalize(column) in candidates:
            return column
    raise KeyError(
        f"Could not find column matching {sorted(candidates)} in {df.columns.tolist()}"
    )


def main() -> None:
    output_path = Path(snakemake.output[0])  # type: ignore[name-defined]
    production_year = int(snakemake.params.production_year)  # type: ignore[name-defined]

    dataset = "QCL"  # Crops and livestock products

    logger.info("Loading FAOSTAT parameter metadata for dataset '%s'", dataset)
    try:
        element_map = faostat.get_par(dataset, "Element")
        item_map = faostat.get_par(dataset, "Item")
    except Exception as exc:  # pragma: no cover - network error
        raise RuntimeError(f"Failed to retrieve FAOSTAT metadata: {exc}") from exc

    # Find production element
    element_code = None
    element_label_used = None
    for label, code in element_map.items():
        normalized = str(label).strip().lower()
        if normalized.startswith("production"):
            element_code = code
            element_label_used = label
            break
    if element_code is None:
        raise RuntimeError(
            f"Failed to locate a production element in FAOSTAT dataset '{dataset}'"
        )
    logger.info(
        "Using FAOSTAT element '%s' (code %s)", element_label_used, element_code
    )

    # Map FAOSTAT items to codes
    faostat_to_model: dict[str, str] = {}  # faostat_item -> model_product
    item_codes: list[str] = []

    for model_product, faostat_item in ANIMAL_PRODUCT_MAPPING.items():
        if faostat_item not in item_map:
            logger.warning(
                "FAOSTAT item '%s' not found for product '%s'; skipping",
                faostat_item,
                model_product,
            )
            continue
        item_code = item_map[faostat_item]
        item_codes.append(str(item_code))
        faostat_to_model[str(item_code)] = model_product
        logger.info(
            "Mapped '%s' -> FAOSTAT item '%s' (code %s)",
            model_product,
            faostat_item,
            item_code,
        )

    if not item_codes:
        raise RuntimeError("No FAOSTAT items could be mapped")

    query_pars = {
        "element": element_code,
        "item": item_codes,
        "year": str(production_year),
    }

    logger.info(
        "Requesting FAOSTAT global production data for year %s (%d animal products)",
        production_year,
        len(item_codes),
    )
    try:
        df = faostat.get_data_df(
            dataset,
            pars=query_pars,
            strval=True,
        )
    except Exception as exc:  # pragma: no cover - network error
        raise RuntimeError(
            f"Failed to retrieve FAOSTAT production data: {exc}"
        ) from exc

    if df.empty:
        raise RuntimeError(
            "FAOSTAT returned no production data for the requested selection"
        )

    # Find required columns
    item_code_candidates = {"item_code", "item_code_(faostat)"}
    value_candidates = {"value"}
    year_candidates = {"year"}

    item_code_col = _find_column(df, item_code_candidates)
    value_col = _find_column(df, value_candidates)
    year_col = _find_column(df, year_candidates)

    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col])
    if df.empty:
        raise RuntimeError("FAOSTAT production data contains no numeric values")

    # Check and filter by unit (should be tonnes for most, number for eggs)
    if "Unit" in df.columns:
        units_series = df["Unit"].astype(str).str.lower()
        logger.info(
            "FAOSTAT returned units: %s",
            ", ".join(sorted(units_series.unique())),
        )

    # Exclude regional aggregates (area codes >= 1000)
    area_code_candidates = {"area_code"}
    area_code_col = _find_column(df, area_code_candidates)
    df[area_code_col] = pd.to_numeric(df[area_code_col], errors="coerce")

    before_filter = len(df)
    df = df[df[area_code_col] < 1000]
    after_filter = len(df)

    if before_filter > after_filter:
        logger.info(
            "Excluded %d regional aggregate rows (keeping %d country records)",
            before_filter - after_filter,
            after_filter,
        )

    if df.empty:
        raise RuntimeError(
            "No country-level production data after filtering aggregates"
        )

    # Aggregate by product (sum across all countries)
    records: list[dict[str, object]] = []
    for _, row in df.iterrows():
        item_code = str(row[item_code_col]).strip()
        product = faostat_to_model.get(item_code)
        if not product:
            continue

        value = float(row[value_col])
        if not np.isfinite(value) or value <= 0:
            continue

        year = int(float(row[year_col]))
        records.append(
            {
                "product": product,
                "year": year,
                "production_tonnes": value,
            }
        )

    if not records:
        raise RuntimeError(
            "No FAOSTAT production records matched the configured animal products"
        )

    result = pd.DataFrame(records)

    # Aggregate to global totals
    result = (
        result.groupby(["product", "year"], as_index=False)["production_tonnes"]
        .sum()
        .sort_values(["product"])
    )

    # Convert eggs from number to tonnes (approximate: 60g per egg)
    # FAOSTAT reports eggs in "1000 no" units (thousands of eggs)
    egg_mask = result["product"] == "eggs"
    if egg_mask.any():
        egg_thousands = result.loc[egg_mask, "production_tonnes"].iloc[0]
        # egg_thousands * 1000 eggs/thousand * 60g/egg / 1000 g/kg / 1000 kg/tonne
        # = egg_thousands * 0.06 tonnes
        egg_tonnes = egg_thousands * 0.06
        result.loc[egg_mask, "production_tonnes"] = egg_tonnes
        logger.info(
            "Converted eggs from %.2e thousand to %.0f tonnes (assuming 60g/egg)",
            egg_thousands,
            egg_tonnes,
        )

    logger.info("Retrieved production data:")
    for _, row in result.iterrows():
        logger.info(
            "  %s (%d): %.0f tonnes",
            row["product"],
            row["year"],
            row["production_tonnes"],
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    logger.info("Saved animal production data to %s", output_path)


if __name__ == "__main__":
    # Configure logging
    logger = setup_script_logging(log_file=snakemake.log[0] if snakemake.log else None)  # type: ignore[name-defined]

    main()
