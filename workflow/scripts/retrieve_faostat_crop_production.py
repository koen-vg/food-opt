"""
SPDX-FileCopyrightText: 2025 Koen van Greevenbroek

SPDX-License-Identifier: GPL-3.0-or-later
"""

import logging
from pathlib import Path
import sys

import faostat
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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
    mapping_path = Path(snakemake.input.mapping)  # type: ignore[name-defined]
    output_path = Path(snakemake.output[0])  # type: ignore[name-defined]
    countries = [str(c).upper() for c in snakemake.params.countries]  # type: ignore[name-defined]
    production_year = int(snakemake.params.production_year)  # type: ignore[name-defined]

    mapping_df = pd.read_csv(mapping_path)
    if mapping_df.empty:
        raise RuntimeError("FAOSTAT item mapping table is empty")

    mapping_df["crop"] = mapping_df["crop"].astype(str).str.strip()
    mapping_df["faostat_item"] = mapping_df["faostat_item"].astype(str).str.strip()

    missing_item_mask = mapping_df["faostat_item"].eq("") | mapping_df[
        "faostat_item"
    ].str.lower().eq("nan")
    if missing_item_mask.any():
        skipped = mapping_df.loc[missing_item_mask, "crop"].tolist()
        logger.warning(
            "Skipping %d crops without FAOSTAT item mapping: %s",
            len(skipped),
            ", ".join(skipped[:5]) + ("..." if len(skipped) > 5 else ""),
        )
        mapping_df = mapping_df.loc[~missing_item_mask].copy()

    if mapping_df.empty:
        raise RuntimeError(
            "All FAOSTAT item mappings are empty after filtering missing entries"
        )

    dataset = "QCL"

    logger.info("Loading FAOSTAT parameter metadata for dataset '%s'", dataset)
    try:
        element_map = faostat.get_par(dataset, "Element")
        item_map = faostat.get_par(dataset, "Item")
    except Exception as exc:  # pragma: no cover - network error is unrecoverable here
        raise RuntimeError(f"Failed to retrieve FAOSTAT metadata: {exc}") from exc

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

    missing_items = sorted(
        {item for item in mapping_df["faostat_item"].unique() if item not in item_map}
    )
    if missing_items:
        raise RuntimeError(
            "FAOSTAT item(s) missing from parameter table: " + ", ".join(missing_items)
        )

    mapping_df["item_code"] = mapping_df["faostat_item"].map(item_map).astype(str)
    item_code_to_crops: dict[str, list[str]] = {
        str(code): sorted(set(rows))
        for code, rows in mapping_df.groupby("item_code")["crop"]
    }
    item_label_to_crops: dict[str, list[str]] = {
        str(label).strip().lower(): sorted(set(rows))
        for label, rows in mapping_df.groupby("faostat_item")["crop"]
    }

    query_pars = {
        "element": element_code,
        "item": sorted(item_code_to_crops.keys()),
        "year": str(production_year),
    }

    logger.info(
        "Requesting FAOSTAT production data for year %s (%d items, %d countries)",
        production_year,
        len(item_code_to_crops),
        len(countries),
    )
    try:
        df = faostat.get_data_df(
            dataset,
            pars=query_pars,
            strval=True,
            coding={"area": "ISO3"},
        )
    except Exception as exc:  # pragma: no cover - network error
        raise RuntimeError(
            f"Failed to retrieve FAOSTAT production data: {exc}"
        ) from exc

    if df.empty:
        raise RuntimeError(
            "FAOSTAT returned no production data for the requested selection"
        )

    iso_candidates = {"area_code_iso3", "area_code_(iso3)", "area_code"}
    item_code_candidates = {"item_code", "item_code_(faostat)"}
    item_label_candidates = {"item", "item_description"}
    value_candidates = {"value"}
    year_candidates = {"year"}

    iso_col = _find_column(df, iso_candidates)
    item_code_col = _find_column(df, item_code_candidates)
    item_label_col = _find_column(df, item_label_candidates)
    value_col = _find_column(df, value_candidates)
    year_col = _find_column(df, year_candidates)

    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col])
    if df.empty:
        raise RuntimeError("FAOSTAT production data contains no numeric values")

    if "Unit" in df.columns:
        units_series = df["Unit"].astype(str).str.lower()
        unit_mask = (
            units_series.str.contains("tonne")
            | units_series.str.fullmatch("t")
            | units_series.str.contains("tons")
        )
        if unit_mask.any():
            df = df[unit_mask]
        else:
            logger.warning(
                "FAOSTAT production data returned unexpected units: %s",
                ", ".join(sorted(units_series.unique())[:5]),
            )

    df["country"] = df[iso_col].astype(str).str.upper()
    if countries:
        df = df[df["country"].isin(countries)]
        if df.empty:
            raise RuntimeError(
                "FAOSTAT returned no records for the requested ISO3 countries"
            )

    records: list[dict[str, object]] = []
    for _, row in df.iterrows():
        item_code = str(row[item_code_col]).strip()
        crops = item_code_to_crops.get(item_code)
        if not crops:
            # As a fallback, match on the item label
            item_label = str(row[item_label_col]).strip()
            crops = item_label_to_crops.get(item_label.lower(), [])
        if not crops:
            continue
        value = float(row[value_col])
        if not np.isfinite(value):
            continue

        share_value = value / len(crops) if crops else 0.0
        year = int(float(row[year_col]))
        for crop in crops:
            records.append(
                {
                    "country": row["country"],
                    "crop": crop,
                    "year": year,
                    "production_tonnes": share_value,
                }
            )

    if not records:
        raise RuntimeError(
            "No FAOSTAT production records matched the configured crop list"
        )

    result = pd.DataFrame(records)
    result = (
        result.groupby(["country", "crop", "year"], as_index=False)["production_tonnes"]
        .sum()
        .sort_values(["country", "crop"])
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    main()
