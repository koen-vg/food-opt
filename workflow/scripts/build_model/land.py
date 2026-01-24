"""
SPDX-FileCopyrightText: 2025 Koen van Greevenbroek

SPDX-License-Identifier: GPL-3.0-or-later
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd
import pypsa

from .. import constants
from . import primary_resources


def add_land_components(
    n: pypsa.Network,
    total_land_area: pd.DataFrame,
    baseline_land_area: pd.DataFrame,
    luc_lef_lookup: Mapping[tuple[str, int, str, str], float],
    *,
    reg_limit: float,
    land_slack_cost: float,
    enable_land_slack: bool,
    min_area_ha: float,
) -> None:
    """Add land buses/generators that distinguish existing vs. new cropland.

    Parameters
    ----------
    n : pypsa.Network
        Target network.
    total_land_area : pd.DataFrame
        Total suitable land indexed by (region, water_supply, resource_class).
    baseline_land_area : pd.DataFrame
        Currently managed cropland indexed the same way.
    luc_lef_lookup : Mapping
        Lookup for cropland LUC coefficients.
    reg_limit : float
        Maximum fraction of total potential cropland that can be utilized.
        Applies to both existing and new cropland combined.
    land_slack_cost : float
        Marginal cost (bnUSD/Mha) for slack generators.
    enable_land_slack : bool
        Whether to add slack generators that allow exceeding regional land limits.
    min_area_ha : float, optional
        Minimum area threshold (ha). Entries with area below this are filtered out.
    """

    if total_land_area.empty:
        return

    # Ensure carriers exist before adding components
    for carrier_name in (
        "land",
        "land_existing",
        "land_new",
        "land_use",
        "land_conversion",
    ):
        if carrier_name not in n.carriers.static.index:
            n.carriers.add(carrier_name, unit="Mha")

    baseline_series = (
        baseline_land_area.reindex(total_land_area.index, fill_value=0.0)["area_ha"]
        .astype(float)
        .rename("area_ha")
    )
    total_area = total_land_area["area_ha"].astype(float)
    total_area = np.maximum(total_area, baseline_series)
    total_land_area["area_ha"] = total_area
    expansion_series = (total_area - baseline_series).clip(lower=0.0)

    land_index_df = total_land_area.reset_index()
    land_index_df["resource_class"] = land_index_df["resource_class"].astype(int)
    land_index_df["baseline_area_ha"] = baseline_series.to_numpy()
    land_index_df["expansion_area_ha"] = expansion_series.to_numpy()

    # Apply reg_limit to total potential area, then split between existing and new
    # total_available = total_area * reg_limit
    # existing_available = min(baseline, total_available)
    # new_available = max(0, total_available - baseline)
    total_available = land_index_df["area_ha"] * reg_limit
    land_index_df["existing_available_ha"] = np.minimum(
        land_index_df["baseline_area_ha"], total_available
    )
    land_index_df["new_available_ha"] = np.maximum(
        0.0, total_available - land_index_df["baseline_area_ha"]
    )

    # Build bus names using ':' delimiter
    land_index_df["pool_bus"] = (
        "land:pool:"
        + land_index_df["region"].astype(str)
        + "_c"
        + land_index_df["resource_class"].astype(str)
        + "_"
        + land_index_df["water_supply"].astype(str)
    )
    land_index_df["existing_bus"] = (
        "land:existing:"
        + land_index_df["region"].astype(str)
        + "_c"
        + land_index_df["resource_class"].astype(str)
        + "_"
        + land_index_df["water_supply"].astype(str)
    )
    land_index_df["new_bus"] = (
        "land:new:"
        + land_index_df["region"].astype(str)
        + "_c"
        + land_index_df["resource_class"].astype(str)
        + "_"
        + land_index_df["water_supply"].astype(str)
    )

    active_mask = (
        (land_index_df["area_ha"] > 0)
        | (land_index_df["baseline_area_ha"] > 0)
        | (land_index_df["expansion_area_ha"] > 0)
    )
    land_index_df = land_index_df[active_mask].copy()
    if land_index_df.empty:
        return

    # Filter small areas for numerical stability
    if min_area_ha > 0:
        small_area_mask = land_index_df["area_ha"] < min_area_ha
        land_index_df = land_index_df[~small_area_mask].copy()
        if land_index_df.empty:
            return

    pool_bus_names = land_index_df["pool_bus"].tolist()
    pool_regions = land_index_df["region"].tolist()
    n.buses.add(pool_bus_names, carrier="land", region=pool_regions)

    # Filter to rows where existing land is available (after applying reg_limit)
    baseline_rows = land_index_df[land_index_df["existing_available_ha"] > 0].copy()
    if not baseline_rows.empty:
        n.buses.add(
            baseline_rows["existing_bus"].tolist(),
            carrier="land_existing",
            region=baseline_rows["region"].tolist(),
        )

    if not baseline_rows.empty:
        existing_gen_names = [
            f"supply:land_existing:{row.region}_c{int(row.resource_class)}_{row.water_supply}"
            for row in baseline_rows.itertuples(index=False)
        ]
        # Use existing_available_ha (constrained by reg_limit) instead of baseline
        existing_available_mha = baseline_rows["existing_available_ha"].to_numpy() / 1e6
        n.generators.add(
            existing_gen_names,
            bus=baseline_rows["existing_bus"].tolist(),
            carrier="land_existing",
            p_nom=existing_available_mha,
            p_nom_extendable=False,
            marginal_cost=0.0,
            region=baseline_rows["region"].tolist(),
            resource_class=baseline_rows["resource_class"].tolist(),
            water_supply=baseline_rows["water_supply"].tolist(),
        )
        existing_link_names = [
            f"use:existing_land:{row.region}_c{int(row.resource_class)}_{row.water_supply}"
            for row in baseline_rows.itertuples(index=False)
        ]
        n.links.add(
            existing_link_names,
            carrier="land_use",
            bus0=baseline_rows["existing_bus"].tolist(),
            bus1=baseline_rows["pool_bus"].tolist(),
            efficiency=1.0,
            p_nom=existing_available_mha,
            p_nom_extendable=False,
            region=baseline_rows["region"].tolist(),
            resource_class=baseline_rows["resource_class"].tolist(),
            water_supply=baseline_rows["water_supply"].tolist(),
        )

    if luc_lef_lookup:
        land_index_df["luc_efficiency"] = land_index_df.apply(
            lambda r: float(
                luc_lef_lookup.get(
                    (
                        r["region"],
                        int(r["resource_class"]),
                        r["water_supply"],
                        "cropland",
                    ),
                    0.0,
                )
            ),
            axis=1,
        )
    else:
        land_index_df["luc_efficiency"] = 0.0

    # Filter to rows where new land is available (after applying reg_limit)
    expansion_rows = land_index_df[land_index_df["new_available_ha"] > 0].copy()
    if not expansion_rows.empty:
        n.buses.add(
            expansion_rows["new_bus"].tolist(),
            carrier="land_new",
            region=expansion_rows["region"].tolist(),
        )

    if not expansion_rows.empty:
        new_gen_names = [
            f"supply:land_new:{row.region}_c{int(row.resource_class)}_{row.water_supply}"
            for row in expansion_rows.itertuples(index=False)
        ]
        # Use new_available_ha (already accounts for reg_limit)
        new_available_mha = expansion_rows["new_available_ha"].to_numpy() / 1e6
        n.generators.add(
            new_gen_names,
            bus=expansion_rows["new_bus"].tolist(),
            carrier="land_new",
            p_nom_extendable=True,
            p_nom_max=new_available_mha,
            marginal_cost=0.0,
            region=expansion_rows["region"].tolist(),
            resource_class=expansion_rows["resource_class"].tolist(),
            water_supply=expansion_rows["water_supply"].tolist(),
        )
        new_link_names = [
            f"convert:new_land:{row.region}_c{int(row.resource_class)}_{row.water_supply}"
            for row in expansion_rows.itertuples(index=False)
        ]
        n.links.add(
            new_link_names,
            carrier="land_conversion",
            bus0=expansion_rows["new_bus"].tolist(),
            bus1=expansion_rows["pool_bus"].tolist(),
            efficiency=1.0,
            bus2="emission:co2",
            efficiency2=(
                expansion_rows["luc_efficiency"].to_numpy()
                * 1e6
                * constants.TONNE_TO_MEGATONNE
            ),
            p_nom_extendable=True,
            p_nom_max=new_available_mha,
            region=expansion_rows["region"].tolist(),
            resource_class=expansion_rows["resource_class"].tolist(),
            water_supply=expansion_rows["water_supply"].tolist(),
        )

    if enable_land_slack:
        primary_resources._add_land_slack_generators(n, pool_bus_names, land_slack_cost)
