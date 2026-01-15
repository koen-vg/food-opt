# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Grassland feed production components for the food systems model.

This module handles the creation of links that produce ruminant feed from
grassland (both rainfed cropland and dedicated marginal pasture).
"""

from collections.abc import Callable
import logging

import numpy as np
import pandas as pd
import pypsa

from workflow.scripts.constants import (
    HA_PER_MHA,
    MEGATONNE_TO_TONNE,
    USD_TO_BNUSD,
)

logger = logging.getLogger(__name__)


def calculate_grazing_cost_per_tonne_dm(
    animal_costs_df: pd.DataFrame,
    feed_to_products_df: pd.DataFrame,
    base_year: int,
) -> float:
    """
    Calculate global average grazing cost per tonne of dry matter.

    Logic:
    1. Get grazing cost per tonne of animal product (e.g. beef, milk) from animal_costs_df.
    2. Get feed efficiency (tonne product / tonne feed DM) from feed_to_products_df.
    3. Calculate implied feed cost: Cost_Feed = Cost_Product * Efficiency
    4. Average across all relevant entries.

    Parameters
    ----------
    animal_costs_df : pd.DataFrame
        Animal cost data with columns: product, grazing_cost_per_mt_usd_{base_year}
    feed_to_products_df : pd.DataFrame
        Feed efficiency data with columns: product, feed_category, region, efficiency
    base_year : int
        Base year for cost data

    Returns
    -------
    float
        Average grazing cost per tonne of dry matter in USD/t
    """
    grazing_col = f"grazing_cost_per_mt_usd_{base_year}"

    # Filter for products with grazing costs
    grazing_costs = animal_costs_df[animal_costs_df[grazing_col] > 0][
        ["product", grazing_col]
    ].copy()

    # Filter feed_to_products for grass-based feed categories
    # The costs are allocated from "Grazed feed" item in USDA/FADN,
    # which corresponds to the "ruminant_grassland" feed category.
    grass_feeds = feed_to_products_df[
        feed_to_products_df["feed_category"] == "ruminant_grassland"
    ].copy()

    # Merge costs and efficiencies
    merged = pd.merge(grazing_costs, grass_feeds, on="product", how="inner")

    # Cost_Feed ($/tDM) = Cost_Product ($/tProduct) * Efficiency (tProduct/tFeedDM)
    merged["implied_feed_cost"] = merged[grazing_col] * merged["efficiency"]

    # Calculate average
    avg_cost = merged["implied_feed_cost"].mean()

    logger.info(
        f"Calculated average grazing cost: ${avg_cost:.2f}/tDM "
        f"(from {len(merged)} product-feed combinations)"
    )

    return float(avg_cost)


def add_grassland_feed_links(
    n: pypsa.Network,
    grassland: pd.DataFrame,
    land_rainfed: pd.DataFrame,
    region_to_country: pd.Series,
    allowed_countries: set,
    marginal_cost: float = 0.0,
    current_grassland_area: pd.DataFrame | None = None,
    pasture_land_area: pd.Series | None = None,
    use_actual_production: bool = False,
    pasture_utilization_rate: float = 1.0,
    *,
    min_yield_t_per_ha: float,
) -> None:
    """Add links supplying ruminant feed directly from rainfed land.

    Parameters
    ----------
    n : pypsa.Network
        The network to add links to.
    grassland : pd.DataFrame
        Grassland yield data.
    land_rainfed : pd.DataFrame
        Rainfed land area availability.
    region_to_country : pd.Series
        Mapping from region to country code.
    allowed_countries : set
        Set of allowed country codes.
    marginal_cost : float, optional
        Marginal cost of grassland feed in USD per tonne DM, by default 0.0.
        Converted internally to bnUSD per Mha based on yield.
    current_grassland_area : pd.DataFrame | None, optional
        Observed grassland area for validation, by default None.
    pasture_land_area : pd.Series | None, optional
        Available pasture land area, by default None.
    use_actual_production : bool, optional
        Whether to cap production at observed values, by default False.
    pasture_utilization_rate : float, optional
        Fraction of grassland biomass actually consumed by animals, by default 1.0.
    """

    grass_df = grassland.copy()
    grass_df = grass_df[np.isfinite(grass_df["yield"]) & (grass_df["yield"] > 0)]

    # Filter low yields for numerical stability
    if min_yield_t_per_ha > 0:
        low_yield_mask = grass_df["yield"] < min_yield_t_per_ha
        grass_df = grass_df[~low_yield_mask]

    if grass_df.empty:
        logger.warning("No valid grassland yield data available; skipping")
        return

    grass_df = grass_df.reset_index()
    grass_df["resource_class"] = grass_df["resource_class"].astype(int)
    grass_df = grass_df.set_index(["region", "resource_class"])

    base_frame = grass_df.join(
        land_rainfed[["area_ha"]].rename(columns={"area_ha": "land_area"}),
        how="inner",
    )
    if use_actual_production:
        observed_area = (
            current_grassland_area.set_index(["region", "resource_class"])["area_ha"]
            .astype(float)
            .rename("observed_area")
        )
        base_frame = base_frame.join(observed_area, how="left")

    candidate_area = base_frame["suitable_area"].fillna(base_frame["land_area"])
    land_cap = np.minimum(candidate_area.to_numpy(), base_frame["land_area"].to_numpy())
    base_index = base_frame.index
    land_cap_series = pd.Series(land_cap, index=base_index, dtype=float)

    cropland_frame = base_frame.copy()
    marginal_frame: pd.DataFrame | None = None

    if use_actual_production:
        # Under validation the observed harvested/grazed area is split so that
        # marginal hectares are satisfied first (subject to the derived
        # land_marginal potential) and only the remainder pulls from the shared
        # cropland pool.
        observed_series = (
            pd.to_numeric(base_frame.get("observed_area"), errors="coerce")
            .fillna(0.0)
            .astype(float)
        )
        base_frame = base_frame.drop(columns=["observed_area"])
        if pasture_land_area is not None and not pasture_land_area.empty:
            marginal_cap_series = pasture_land_area.reindex(base_index, fill_value=0.0)
        else:
            marginal_cap_series = pd.Series(0.0, index=base_index, dtype=float)
        observed_aligned = observed_series.reindex(base_index)
        marginal_alloc = np.minimum(
            observed_aligned.to_numpy(), marginal_cap_series.to_numpy()
        )
        cropland_observed = np.maximum(
            observed_aligned.to_numpy() - marginal_alloc, 0.0
        )
        cropland_available = np.minimum(land_cap_series.to_numpy(), cropland_observed)
        cropland_frame["available_area"] = cropland_available
        cropland_frame = cropland_frame[cropland_frame["available_area"] > 0]

        if np.any(marginal_alloc > 0.0):
            marginal_series = pd.Series(
                marginal_alloc,
                index=base_index,
                name="available_area",
            )
            marginal_frame = grass_df.join(marginal_series, how="inner")
            marginal_frame = marginal_frame[marginal_frame["available_area"] > 0]
    else:
        cropland_frame["available_area"] = land_cap_series.reindex(
            cropland_frame.index
        ).to_numpy()
        cropland_frame = cropland_frame[cropland_frame["available_area"] > 0]
        if pasture_land_area is not None and not pasture_land_area.empty:
            marginal_frame = grass_df.join(
                pasture_land_area.rename("available_area"), how="inner"
            )
            marginal_frame = marginal_frame[marginal_frame["available_area"] > 0]

    # Helper to convert a per-region/class frame into Link components. The caller
    # passes a name prefix so we can distinguish cropland-competing vs.
    # marginal-only grassland in the network outputs.
    def _add_links_for_frame(
        frame: pd.DataFrame,
        name_prefix: str,
        bus0_builder: Callable[[pd.Series], str],
    ) -> bool:
        if frame is None or frame.empty:
            return False
        work = frame.reset_index()
        work["country"] = work["region"].map(region_to_country)
        work = work[work["country"].isin(allowed_countries)]
        work = work.dropna(subset=["country"])
        if work.empty:
            return False
        work["name"] = work.apply(
            lambda r: f"{name_prefix}:{r['region']}_c{int(r['resource_class'])}",
            axis=1,
        )
        work["bus0"] = work.apply(bus0_builder, axis=1)
        work["bus1"] = work["country"].apply(lambda c: f"feed:ruminant_grassland:{c}")

        available_mha = work["available_area"].to_numpy() / HA_PER_MHA

        # Calculate efficiency (Mt/Mha) applying pasture utilization rate.
        # Yields are in t/ha, which equals Mt/Mha numerically.
        yields = work["yield"].to_numpy()  # t/ha = Mt/Mha numerically
        efficiencies = yields * pasture_utilization_rate  # Mt/Mha

        # Calculate marginal cost per Mha (bnUSD/Mha).
        # In PyPSA, marginal_cost is per unit of bus0 (land in Mha).
        # To get cost per unit output (feed in Mt), we need:
        #   cost_per_output = marginal_cost_pypsa / efficiency
        # We want: cost_per_output = marginal_cost (USD/t) * conversion to bnUSD/Mt
        # Therefore: marginal_cost_pypsa = marginal_cost * conversion * efficiency
        cost_per_mha_bnusd = (
            marginal_cost * efficiencies * MEGATONNE_TO_TONNE * USD_TO_BNUSD
        )

        # Index by name for proper alignment with PyPSA component names
        work_indexed = work.set_index("name")
        params = {
            "carrier": "produce_grassland",
            "bus0": work_indexed["bus0"],
            "bus1": work_indexed["bus1"],
            "efficiency": efficiencies,
            "p_nom_max": available_mha,
            "p_nom_extendable": not use_actual_production,
            "marginal_cost": cost_per_mha_bnusd,
            "region": work_indexed["region"],
            "resource_class": work_indexed["resource_class"],
            "country": work_indexed["country"],
        }
        if use_actual_production:
            params["p_nom"] = available_mha

        n.links.add(work_indexed.index, **params)
        return True

    link_added = False

    # Standard grassland links consume land from the same rainfed cropland pool
    # that crops use, so they continue to compete for those hectares when
    # optimisation is unconstrained.
    link_added |= _add_links_for_frame(
        cropland_frame,
        "produce:grassland",
        lambda r: f"land:pool:{r['region']}_c{int(r['resource_class'])}_r",
    )

    if marginal_frame is not None and not marginal_frame.empty:
        # Marginal grassland links tap into the exclusive land_marginal buses so
        # grazing can expand without reducing cropland-suitable land.
        link_added |= _add_links_for_frame(
            marginal_frame,
            "produce:grassland_marginal",
            lambda r: f"land:marginal:{r['region']}_c{int(r['resource_class'])}",
        )

    if not link_added:
        logger.info("Grassland entries have zero available area; skipping")
