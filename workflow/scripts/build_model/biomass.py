# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Biomass infrastructure and routing for the food systems model.

This module handles optional biomass exports to the energy sector,
including infrastructure setup and routing from crops and byproducts.
"""

from collections.abc import Iterable, Mapping

import pandas as pd
import pypsa

from .. import constants


def add_biomass_infrastructure(
    n: pypsa.Network, countries: Iterable[str], biomass_cfg: Mapping[str, object]
) -> bool:
    """Create biomass buses and sinks for optional exports to the energy sector."""

    marginal_cost = float(biomass_cfg["marginal_cost"])
    marginal_cost *= constants.USD_TO_BNUSD / constants.TONNE_TO_MEGATONNE
    # Biomass quantities are in Mt DM throughout this module.
    biomass_carrier = "biomass"
    n.carriers.add(biomass_carrier, unit="MtDM")

    biomass_buses = [f"biomass_{country}" for country in countries]
    n.buses.add(biomass_buses, carrier=biomass_carrier)

    n.generators.add(
        [f"biomass_for_energy_{country}" for country in countries],
        bus=biomass_buses,
        carrier=biomass_carrier,
        p_nom_extendable=True,
        marginal_cost=marginal_cost,
        p_min_pu=-1,  # Allow consumption, not generation of biomass
        p_max_pu=0,
    )


def add_biomass_byproduct_links(
    n: pypsa.Network, countries: Iterable[str], byproducts: Iterable[str]
) -> None:
    """Allow food byproducts to be routed to biomass buses."""
    combos = pd.MultiIndex.from_product(
        [byproducts, countries], names=["item", "country"]
    ).to_frame(index=False)
    combos["bus0"] = "food_" + combos["item"] + "_" + combos["country"]
    combos["bus1"] = "biomass_" + combos["country"]
    bus_index = n.buses.static.index
    combos = combos[combos["bus0"].isin(bus_index) & combos["bus1"].isin(bus_index)]
    if combos.empty:
        return

    combos["name"] = "byproduct_to_biomass_" + combos["item"] + "_" + combos["country"]
    combos = combos.set_index("name")

    carrier = "byproduct_to_biomass"
    n.carriers.add(carrier, unit="MtDM")

    n.links.add(
        combos.index,
        bus0=combos["bus0"],
        bus1=combos["bus1"],
        carrier=carrier,
        p_nom_extendable=True,
    )


def add_biomass_crop_links(
    n: pypsa.Network, countries: Iterable[str], crops: Iterable[str]
) -> None:
    """Route configured crops to biomass buses (dry-matter accounting)."""
    combos = pd.MultiIndex.from_product(
        [crops, countries], names=["crop", "country"]
    ).to_frame(index=False)
    combos["bus0"] = "crop_" + combos["crop"] + "_" + combos["country"]
    combos["bus1"] = "biomass_" + combos["country"]
    bus_index = n.buses.static.index
    combos = combos[combos["bus0"].isin(bus_index) & combos["bus1"].isin(bus_index)]
    if combos.empty:
        return

    combos["name"] = "crop_to_biomass_" + combos["crop"] + "_" + combos["country"]
    combos = combos.set_index("name")

    carrier = "crop_to_biomass"
    n.carriers.add(carrier, unit="MtDM")
    n.links.add(
        combos.index,
        bus0=combos["bus0"],
        bus1=combos["bus1"],
        carrier=carrier,
        p_nom_extendable=True,
    )
