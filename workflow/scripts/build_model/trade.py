# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Trade network components for the food systems model.

This module handles the creation of hierarchical trade networks for crops
and foods, using clustering-based hub systems for efficient trade routing.
"""

import logging

import geopandas as gpd
import numpy as np
import pypsa
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


def _resolve_trade_costs(
    trade_config: dict,
    items: list,
    *,
    categories_key: str | None,
    default_cost_key: str | None,
    fallback_cost_key: str,
    category_item_key: str,
) -> tuple[dict[str, float], float]:
    """Map each item to its configured trade cost per kilometre."""

    # Get default cost from config hierarchy
    if default_cost_key is not None:
        default_cost = float(trade_config[default_cost_key])
    else:
        default_cost = float(trade_config[fallback_cost_key])

    item_costs = {str(item): default_cost for item in items}

    if categories_key is None:
        return item_costs, default_cost

    # Override with category-specific costs
    categories = trade_config[categories_key]
    for _category, cfg in categories.items():
        category_cost = float(cfg["cost_per_km"])
        configured_items = cfg[category_item_key]

        for item in configured_items:
            item_label = str(item)
            if item_label in item_costs:
                item_costs[item_label] = category_cost

    return item_costs, default_cost


def _add_trade_hubs_and_links(
    n: pypsa.Network,
    trade_config: dict,
    regions_gdf: gpd.GeoDataFrame,
    countries: list,
    items: list,
    *,
    hub_count_key: str,
    marginal_cost_key: str,
    cost_categories_key: str | None,
    default_cost_key: str | None,
    category_item_key: str,
    non_tradable_key: str,
    bus_prefix: str,
    carrier_prefix: str,
    hub_name_prefix: str,
    link_name_prefix: str,
    log_label: str,
) -> None:
    """Shared implementation for adding trade hubs and links for a set of items."""

    n_hubs = int(trade_config[hub_count_key])
    item_costs, _default_cost = _resolve_trade_costs(
        trade_config,
        items,
        categories_key=cost_categories_key,
        default_cost_key=default_cost_key,
        fallback_cost_key=marginal_cost_key,
        category_item_key=category_item_key,
    )

    if len(regions_gdf) == 0 or len(countries) == 0:
        logger.info("Skipping %s trade hubs: no regions/countries available", log_label)
        return

    items = list(dict.fromkeys(items))
    if len(items) == 0:
        logger.info("Skipping %s trade hubs: no items configured", log_label)
        return

    non_tradable = {
        str(item) for item in trade_config[non_tradable_key] if item in items
    }
    tradable_items = [item for item in items if item not in non_tradable]
    if non_tradable:
        logger.info(
            "Skipping %s trade network for configured non-tradable items: %s",
            log_label,
            ", ".join(sorted(non_tradable)),
        )

    if not tradable_items:
        logger.info("Skipping %s trade hubs: no tradable items available", log_label)
        return

    gdf = regions_gdf.copy()
    gdf_ee = gdf.to_crs(6933)

    cent = gdf_ee.geometry.centroid
    region_coords = np.column_stack([cent.x.values, cent.y.values])
    k = min(max(1, n_hubs), len(region_coords))
    if k < n_hubs:
        logger.info(
            "Reducing %s hub count from %d to %d (regions=%d)",
            log_label,
            n_hubs,
            k,
            len(region_coords),
        )
        n_hubs = k

    km = KMeans(n_clusters=n_hubs, n_init=10, random_state=0)
    km.fit_predict(region_coords)
    centers = km.cluster_centers_

    hub_ids = list(range(n_hubs))
    hub_bus_names: list[str] = []
    hub_bus_carriers: list[str] = []
    for item in tradable_items:
        item_label = str(item)
        for h in hub_ids:
            hub_bus_names.append(f"{hub_name_prefix}_{h}_{item_label}")
            hub_bus_carriers.append(f"{carrier_prefix}{item_label}")

    if hub_bus_names:
        n.buses.add(hub_bus_names, carrier=hub_bus_carriers)

    gdf_countries = gdf_ee[gdf_ee["country"].isin(countries)].dissolve(
        by="country", as_index=True
    )
    ccent = gdf_countries.geometry.centroid
    country_coords = np.column_stack([ccent.x.values, ccent.y.values])
    dch = ((country_coords[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2) ** 0.5
    nearest_hub_idx = dch.argmin(axis=1)
    nearest_hub_dist_km = dch[np.arange(len(country_coords)), nearest_hub_idx] / 1000.0

    country_index = gdf_countries.index.to_list()
    country_to_hub = {c: int(h) for c, h in zip(country_index, nearest_hub_idx)}
    country_to_dist_km = {
        c: float(d) for c, d in zip(country_index, nearest_hub_dist_km)
    }

    valid_countries = [c for c in countries if c in country_to_hub]

    link_names: list[str] = []
    link_bus0: list[str] = []
    link_bus1: list[str] = []
    link_costs: list[float] = []

    if valid_countries:
        for item in tradable_items:
            item_label = str(item)
            item_cost = item_costs[item_label]
            for c in valid_countries:
                hub_idx = country_to_hub[c]
                cost = country_to_dist_km[c] * item_cost

                country_bus = f"{bus_prefix}{item_label}_{c}"
                hub_bus = f"{hub_name_prefix}_{hub_idx}_{item_label}"

                link_names.append(f"{link_name_prefix}_{item_label}_{c}_hub{hub_idx}")
                link_bus0.append(country_bus)
                link_bus1.append(hub_bus)
                link_costs.append(cost)

                link_names.append(f"{link_name_prefix}_{item_label}_hub{hub_idx}_{c}")
                link_bus0.append(hub_bus)
                link_bus1.append(country_bus)
                link_costs.append(cost)

    if link_names:
        n.links.add(
            link_names,
            bus0=link_bus0,
            bus1=link_bus1,
            marginal_cost=link_costs,
            p_nom_extendable=[True] * len(link_names),
        )

    if n_hubs >= 2:
        hub_distances = (
            np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
            / 1000.0
        )
        ii, jj = np.where(~np.eye(n_hubs, dtype=bool))

        hub_link_names: list[str] = []
        hub_link_bus0: list[str] = []
        hub_link_bus1: list[str] = []
        hub_link_costs: list[float] = []

        if len(ii) > 0:
            dists_km = hub_distances[ii, jj]
            for item in tradable_items:
                item_label = str(item)
                item_cost = item_costs[item_label]
                for i, j, dist in zip(ii, jj, dists_km):
                    hub_link_names.append(
                        f"{link_name_prefix}_{item_label}_hub{i}_to_hub{j}"
                    )
                    hub_link_bus0.append(f"{hub_name_prefix}_{i}_{item_label}")
                    hub_link_bus1.append(f"{hub_name_prefix}_{j}_{item_label}")
                    hub_link_costs.append(float(dist) * item_cost)

        if hub_link_names:
            n.links.add(
                hub_link_names,
                bus0=hub_link_bus0,
                bus1=hub_link_bus1,
                marginal_cost=hub_link_costs,
                p_nom_extendable=[True] * len(hub_link_names),
            )


def add_crop_trade_hubs_and_links(
    n: pypsa.Network,
    trade_config: dict,
    regions_gdf: gpd.GeoDataFrame,
    countries: list,
    crop_list: list,
) -> None:
    """Add crop trading hubs and connect crop buses via hubs."""

    _add_trade_hubs_and_links(
        n,
        trade_config,
        regions_gdf,
        countries,
        crop_list,
        hub_count_key="crop_hubs",
        marginal_cost_key="crop_trade_marginal_cost_per_km",
        cost_categories_key="crop_trade_cost_categories",
        default_cost_key="crop_default_trade_cost_per_km",
        category_item_key="crops",
        non_tradable_key="non_tradable_crops",
        bus_prefix="crop_",
        carrier_prefix="crop_",
        hub_name_prefix="hub",
        link_name_prefix="trade",
        log_label="crop",
    )


def add_food_trade_hubs_and_links(
    n: pypsa.Network,
    trade_config: dict,
    regions_gdf: gpd.GeoDataFrame,
    countries: list,
    food_list: list,
) -> None:
    """Add trading hubs and links for foods (including byproducts)."""

    _add_trade_hubs_and_links(
        n,
        trade_config,
        regions_gdf,
        countries,
        food_list,
        hub_count_key="food_hubs",
        marginal_cost_key="food_trade_marginal_cost_per_km",
        cost_categories_key="food_trade_cost_categories",
        default_cost_key="food_default_trade_cost_per_km",
        category_item_key="foods",
        non_tradable_key="non_tradable_foods",
        bus_prefix="food_",
        carrier_prefix="food_",
        hub_name_prefix="hub_food",
        link_name_prefix="trade_food",
        log_label="food",
    )
