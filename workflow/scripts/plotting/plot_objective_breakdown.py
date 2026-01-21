#! /usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Generate an objective breakdown plot with updated emissions accounting."""

from collections.abc import Iterable
import logging
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa

logger = logging.getLogger(__name__)

# Global mass unit conversion: tonne to megatonne
TONNE_TO_MEGATONNE = 1e-6


def objective_category(n: pypsa.Network, component: str, **_: object) -> pd.Series:
    """Group assets into high-level categories for system cost aggregation."""

    static = n.components[component].static
    if static.empty:
        return pd.Series(dtype="object")

    index = static.index
    if component == "Generator":
        # Separate biomass exports so they don't get netted out inside the
        # generic "Generator" bucket.
        carriers = static.get("carrier", pd.Series(dtype=str))
        categories = []
        for name in index:
            carrier = str(carriers.get(name, "")) if not carriers.empty else ""
            if carrier == "biomass_for_energy":
                categories.append("Biomass exports")
            elif carrier.startswith("slack"):
                categories.append("Slack penalties")
            elif carrier == "fertilizer":
                categories.append("Fertilizer (synthetic)")
            else:
                categories.append("Other")
        return pd.Series(categories, index=index, name="category")

    if component == "Link":
        mapping = {
            "crop": "Crop production",
            "produce": "Crop production",
            "trade": "Trade",
            "convert": "Processing",
            "consume": "Consumption",
        }
        carriers = static.get("carrier", pd.Series(dtype=str))
        categories = []
        for name in index:
            carrier = str(carriers.get(name, "")) if not carriers.empty else ""
            if carrier in ("crop_to_biomass", "byproduct_to_biomass"):
                categories.append("Biomass routing")
                continue

            # Extract category prefix from carrier (e.g., "crop_wheat_rainfed" -> "crop")
            carrier_prefix = carrier.split("_", 1)[0] if carrier else ""
            categories.append(mapping.get(carrier_prefix, "Other"))
        return pd.Series(categories, index=index, name="category")

    if component == "Store":
        carriers = static["carrier"].astype(str)
        nutrients = static.get("nutrient", pd.Series(index=index, dtype=object))
        food_groups = static.get("food_group", pd.Series(index=index, dtype=object))

        def _has_value(value: object) -> bool:
            return bool(str(value).strip())

        categories = []
        for name, carrier, nutrient, food_group in zip(
            index, carriers, nutrients, food_groups
        ):
            name_str = str(name)
            if carrier == "ghg" or name_str == "ghg":
                categories.append("GHG storage")
            elif carrier.startswith("yll_"):
                categories.append(f"Health ({carrier.removeprefix('yll_')})")
            elif _has_value(food_group) or carrier.startswith("group_"):
                categories.append("Consumer values (food groups)")
            elif _has_value(nutrient):
                categories.append("Macronutrient stores")
            elif carrier == "water":
                categories.append("Water storage")
            elif carrier == "fertilizer":
                categories.append("Fertilizer storage")
            elif carrier == "spared_land":
                categories.append("Spared land")
            else:
                categories.append("Store")
        return pd.Series(categories, index=index, name="category")

    return pd.Series(component, index=index, name="category")


def compute_system_costs(n: pypsa.Network) -> pd.Series:
    """Aggregate system costs by the objective categories defined above."""
    capex = n.statistics.capex(groupby=objective_category)
    opex = n.statistics.opex(groupby=objective_category)

    def _to_series(df_or_series: pd.Series | pd.DataFrame) -> pd.Series:
        if isinstance(df_or_series, pd.DataFrame):
            df_or_series = df_or_series.iloc[:, 0]
        if df_or_series.empty:
            return pd.Series(dtype=float)
        idx = df_or_series.index
        if "category" not in idx.names:
            idx = idx.set_names([*list(idx.names[:-1]), "category"])
            df_or_series.index = idx
        return df_or_series.groupby("category").sum()

    capex_series = _to_series(capex)
    opex_series = _to_series(opex)

    total = capex_series.add(opex_series, fill_value=0.0)
    combined = pd.DataFrame(
        {
            "capex": capex_series,
            "opex": opex_series,
            "total": total,
        }
    ).fillna(0.0)

    return combined.sort_values("total", key=np.abs, ascending=False)


def compute_ghg_cost_breakdown(n: pypsa.Network, ghg_price: float) -> dict[str, float]:
    """Return the objective contribution from the priced GHG store.

    Assumes:
    - GHG store level in MtCO2-eq
    - `ghg_price` in USD per tCO2-eq (config currency_year)
    - Objective units in bnUSD (model-wide)
    """

    if len(n.snapshots) == 0:
        return {}

    snapshot = n.snapshots[-1]
    if snapshot not in n.stores_t.e.index:
        return {}

    store_levels = n.stores_t.e.loc[snapshot]
    if "ghg" not in store_levels.index:
        return {}

    level_mt = float(store_levels["ghg"])
    if level_mt == 0.0:
        return {}

    # Convert: [USD/tCO2] * [MtCO2] * [1e6 t/Mt] * [1e-9 bnUSD/USD] => bnUSD
    # = USD/tCO2 * MtCO2 * 1e-3 => bnUSD
    contribution = ghg_price * level_mt / TONNE_TO_MEGATONNE / 1e9
    label = "GHG pricing (CO₂-eq)"
    logger.info(
        "Computed %s contribution %.3e bnUSD (level %.3e MtCO2-eq, price %.2f USD/tCO2-eq)",
        label,
        contribution,
        level_mt,
        ghg_price,
    )
    return {label: contribution}


def choose_scale(values: Iterable[float]) -> tuple[float, str]:
    """Choose appropriate scale for cost values in billion USD (bnUSD)."""
    max_val = max((abs(v) for v in values), default=1.0)
    if max_val >= 1e9:
        return 1e9, "quintillion USD"
    if max_val >= 1e6:
        return 1e6, "quadrillion USD"
    if max_val >= 1e3:
        return 1e3, "trillion USD"
    return 1.0, "billion USD"


def plot_cost_breakdown(series: pd.Series, output_path: Path) -> None:
    if series.empty:
        logger.warning("No cost data available for plotting")
        return

    scale, label = choose_scale(series.values)
    values = series / scale

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    bars = ax.bar(series.index, values, color="#4e79a7")

    ax.set_ylabel(f"Cost ({label})")
    ax.set_title("Objective Breakdown by Category")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, raw in zip(bars, series.values):
        height = bar.get_height()
        ax.annotate(
            f"{raw / scale:,.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    n = pypsa.Network(snakemake.input.network)  # type: ignore[name-defined]
    logger.info("Loaded network with objective %.3e", n.objective)

    system_costs = compute_system_costs(n)
    ghg_store_cost = 0.0
    if "GHG storage" in system_costs.index:
        ghg_store_cost = float(system_costs.loc["GHG storage", "total"])
        system_costs = system_costs.drop(index="GHG storage")
        if ghg_store_cost != 0.0:
            logger.info(
                "Removed 'GHG storage' category contribution %.3e USD to avoid double counting priced emissions",
                ghg_store_cost,
            )

    total_series = system_costs["total"].copy()

    # Add priced greenhouse gas emissions
    ghg_price = float(snakemake.params.ghg_price)
    ghg_terms = compute_ghg_cost_breakdown(n, ghg_price)
    if ghg_terms:
        for label, value in ghg_terms.items():
            total_series.loc[label] = value
            system_costs.loc[label, ["capex", "opex", "total"]] = [0.0, value, value]

    total_series = total_series.sort_values(key=np.abs, ascending=False)
    system_costs = system_costs.loc[total_series.index]

    breakdown_csv = Path(snakemake.output.breakdown_csv)  # type: ignore[attr-defined]
    breakdown_pdf = Path(snakemake.output.breakdown_pdf)
    breakdown_csv.parent.mkdir(parents=True, exist_ok=True)

    out_df = system_costs.rename(
        columns={"capex": "capex_bnusd", "opex": "opex_bnusd", "total": "total_bnusd"}
    )
    out_df.to_csv(breakdown_csv)
    logger.info("Wrote objective breakdown table to %s", breakdown_csv)
    plot_cost_breakdown(total_series, breakdown_pdf)
    logger.info("Wrote objective breakdown plot to %s", breakdown_pdf)


if __name__ == "__main__":
    main()
