# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Retrieve FAOSTAT emissions data (Domain GT) for global comparison."""

from pathlib import Path

import faostat
import pandas as pd

from workflow.scripts.logging_config import setup_script_logging

if __name__ == "__main__":
    logger = setup_script_logging(log_file=snakemake.log[0] if snakemake.log else None)

    output_path = Path(snakemake.output[0])
    year = int(snakemake.params.year)

    # FAOSTAT items to fetch (mapped to model categories)
    # Note: We fetch individual gases to apply model GWPs later.
    items = {
        "Crop Residues": 5064,
        "Rice Cultivation": 5060,
        "Burning - Crop residues": 5066,
        "Synthetic Fertilizers": 5061,
        "Drained organic soils": 6729,
        "Enteric Fermentation": 5058,
        "Manure Management": 5059,
        "Manure applied to Soils": 5062,
        "Manure left on Pasture": 5063,
        "Net Forest conversion": 6750,
        # "Forestland": 6751, # Total global forest sinks; not aligned with model scope
        # "Food Processing": 6507, # Energy-related emissions not modelled
        # "Food Transport": 6815, # Energy-related emissions not modelled
        # "On-farm energy use": 6994, # Energy-related emissions not modelled
    }

    # Elements: Emissions in kt (CH4, N2O, CO2)
    elements = {
        "Emissions (CH4)": 7225,  # kt
        "Emissions (N2O)": 7230,  # kt
        "Emissions (CO2)": 7273,  # kt
    }

    dataset = "GT"

    pars = {
        "area": 5000,  # World
        "item": list(items.values()),
        "element": list(elements.values()),
        "year": year,
    }

    logger.info("Fetching FAOSTAT GT data for World, Year %s", year)

    try:
        df = faostat.get_data_df(dataset, pars=pars, strval=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to retrieve FAOSTAT GT data: {exc}") from exc

    if df.empty:
        raise RuntimeError(
            "FAOSTAT returned no emissions data for the requested selection"
        )

    # Dynamically find columns (standard faostat structure: Area Code, Item Code, Element Code, Year Code, Flag, Value, Unit...)
    # But get_data_df with strval=True might return slightly different names or standard names.
    # Let's check columns by normalizing.

    def _normalize(name: str) -> str:
        return name.strip().lower().replace(" ", "_")

    def _find_column(df: pd.DataFrame, candidates: set[str]) -> str:
        for column in df.columns:
            if _normalize(column) in candidates:
                return column
        raise KeyError(
            f"Could not find column matching {sorted(candidates)} in {df.columns.tolist()}"
        )

    item_col = _find_column(df, {"item", "item_description"})
    element_col = _find_column(df, {"element", "element_description"})
    value_col = _find_column(df, {"value"})
    unit_col = _find_column(df, {"unit"})

    # Filter and clean
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col])

    # Create standardized output
    records = []

    # Reverse maps for codes if needed, but we have names in the DF usually.
    # We'll trust the returned Item/Element names or map back from codes if names are missing?
    # faostat usually returns names.

    for _, row in df.iterrows():
        item = str(row[item_col]).strip()
        element = str(row[element_col]).strip()
        value = float(row[value_col])
        unit = str(row[unit_col]).strip().lower()

        # Normalize unit to kilotonnes (kt)
        # FAOSTAT emissions are usually in kt.
        # If unit is tonnes, divide by 1000.
        factor = 1.0
        if unit in ["tonnes", "t"]:
            factor = 1e-3
        elif unit in ["kilotonnes", "kt"] or unit in ["gigagrams", "gg"]:
            factor = 1.0
        else:
            logger.warning(
                "Unknown unit '%s' for %s - %s. Assuming kt.", unit, item, element
            )

        records.append(
            {"item": item, "element": element, "value_kt": value * factor, "year": year}
        )

    result = pd.DataFrame(records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    logger.info("Saved %d emission records to %s", len(result), output_path)
