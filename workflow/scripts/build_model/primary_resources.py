# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Primary resources management for the food systems model.

This module handles land, water, and fertilizer resources, including
emissions bookkeeping for GHG, CO2, CH4, and N2O.
"""

from collections.abc import Iterable

import numpy as np
import pandas as pd
import pypsa

from . import constants


def _add_land_slack_generators(
    n: pypsa.Network, bus_names: list[str], marginal_cost: float
) -> None:
    """Attach slack generators to the provided land buses."""

    if "land_slack" not in n.carriers.static.index:
        n.carriers.add("land_slack", unit="Mha")
    n.generators.add(
        [f"{bus}_slack" for bus in bus_names],
        bus=bus_names,
        carrier="land_slack",
        p_nom_extendable=True,
        marginal_cost=marginal_cost,
    )


def add_primary_resources(
    n: pypsa.Network,
    primary_config: dict,
    region_water_limits: pd.Series,
    co2_price: float,
    ch4_to_co2_factor: float,
    n2o_to_co2_factor: float,
    use_actual_production: bool,
) -> None:
    """Add primary resource components and emissions bookkeeping."""
    # Water stores use km^3, so convert m^3 limits accordingly.
    water_limits = region_water_limits * constants.KM3_PER_M3
    n.stores.add(
        "water_store_" + water_limits.index,
        bus="water_" + water_limits.index,
        carrier="water",
        e_nom=water_limits.values,
        e_initial=water_limits.values,
        e_nom_extendable=False,
        e_cyclic=False,
    )

    # Slack in water limits when using actual (current) production
    if use_actual_production:
        slack_cost = 1e-6 * constants.USD_TO_BNUSD
        n.generators.add(
            "water_slack_" + water_limits.index,
            bus="water_" + water_limits.index,
            carrier="water",
            marginal_cost=slack_cost,
            p_nom_extendable=True,
        )

    scale_meta = n.meta.setdefault("carrier_unit_scale", {})
    scale_meta["water_km3_per_m3"] = constants.KM3_PER_M3

    co2_price_per_mt = (
        co2_price / constants.TONNE_TO_MEGATONNE * constants.USD_TO_BNUSD
    )  # convert USD/tCO2 to bnUSD/MtCO2

    # Fertilizer remains global (no regionalization yet)
    fertilizer_cfg = primary_config["fertilizer"]
    limit_mt = float(fertilizer_cfg["limit"]) * constants.KG_TO_MEGATONNE
    marginal_cost_bnusd_per_mt = (
        float(fertilizer_cfg["marginal_cost_usd_per_tonne"])
        * constants.MEGATONNE_TO_TONNE
        * constants.USD_TO_BNUSD
    )
    n.generators.add(
        "fertilizer",
        bus="fertilizer",
        carrier="fertilizer",
        p_nom_extendable=True,
        p_nom_max=limit_mt,
        marginal_cost=marginal_cost_bnusd_per_mt,
    )

    # Add GHG aggregation store and links from individual gases
    n.stores.add(
        "ghg",
        bus="ghg",
        carrier="ghg",
        e_nom_extendable=True,
        e_nom_min=-np.inf,
        e_min_pu=-1.0,
        marginal_cost_storage=co2_price_per_mt,
    )
    n.links.add(
        "convert_co2_to_ghg",
        bus0="co2",
        bus1="ghg",
        carrier="co2",
        efficiency=1.0,
        p_min_pu=-1.0,  # allow negative emissions flow
        p_nom_extendable=True,
    )
    n.links.add(
        "convert_ch4_to_ghg",
        bus0="ch4",
        bus1="ghg",
        carrier="ch4",
        efficiency=ch4_to_co2_factor,
        p_nom_extendable=True,
    )
    n.links.add(
        "convert_n2o_to_ghg",
        bus0="n2o",
        bus1="ghg",
        carrier="n2o",
        efficiency=n2o_to_co2_factor,
        p_nom_extendable=True,
    )


def add_fertilizer_distribution_links(
    n: pypsa.Network,
    countries: Iterable[str],
    synthetic_n2o_factor: float,
) -> None:
    """Connect the global fertilizer supply bus to country-level fertilizer buses."""

    country_list = list(countries)
    if not country_list:
        return

    names = [f"distribute_synthetic_fertilizer_{country}" for country in country_list]
    params: dict[str, object] = {
        "bus0": ["fertilizer"] * len(country_list),
        "bus1": [f"fertilizer_{country}" for country in country_list],
        "carrier": "fertilizer",
        "efficiency": [1.0] * len(country_list),
        "p_nom_extendable": True,
    }

    emission_mt_per_mt = max(0.0, float(synthetic_n2o_factor)) * constants.N2O_N_TO_N2O
    if emission_mt_per_mt > 0.0:
        params["bus2"] = ["n2o"] * len(country_list)
        params["efficiency2"] = [emission_mt_per_mt] * len(country_list)

    n.links.add(names, **params)
