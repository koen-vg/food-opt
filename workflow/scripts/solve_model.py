# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

from collections import defaultdict
import itertools
import math

from linopy.constraints import print_single_constraint
import numpy as np
import pandas as pd
import pypsa
import xarray as xr

from workflow.scripts import constants
from workflow.scripts.build_model.nutrition import (
    _build_food_group_equals_from_baseline,
)
from workflow.scripts.build_model.utils import _per_capita_mass_to_mt_per_year
from workflow.scripts.logging_config import setup_script_logging
from workflow.scripts.snakemake_utils import apply_scenario_config

try:  # Used for type annotations / documentation; fallback when unavailable
    import linopy  # type: ignore
except Exception:  # pragma: no cover - documentation build without linopy

    class linopy:  # type: ignore
        class Model:  # Minimal stub to satisfy type checkers and autodoc
            pass


# Enable new PyPSA components API
pypsa.options.api.new_components_api = True

# Helpers and state for health objective construction
HEALTH_AUX_MAP: dict[int, set[str]] = {}
_SOS2_COUNTER = [0]  # Use list for mutable counter
_LAMBDA_GROUP_COUNTER = itertools.count()
_TOTAL_GROUP_COUNTER = itertools.count()


def _register_health_variable(m: "linopy.Model", name: str) -> None:
    aux = HEALTH_AUX_MAP.setdefault(id(m), set())
    aux.add(name)


def add_macronutrient_constraints(
    n: pypsa.Network, macronutrient_cfg: dict | None, population: dict[str, float]
) -> None:
    """Add per-country macronutrient bounds directly to the linopy model.

    The bounds are expressed on the storage level of each macronutrient store.
    RHS values are converted from per-person-per-day units using stored
    population and nutrient unit metadata.
    """

    if not macronutrient_cfg:
        return

    m = n.model
    store_e = m.variables["Store-e"].sel(snapshot="now")
    stores_df = n.stores.static

    for nutrient, bounds in macronutrient_cfg.items():
        if not bounds:
            continue

        carrier_unit = n.carriers.static.at[nutrient, "unit"]
        nutrient_stores = stores_df[stores_df["carrier"] == nutrient]
        countries = nutrient_stores["country"].astype(str)

        lhs = store_e.sel(name=nutrient_stores.index)

        def rhs_from(
            value: float,
            carrier_unit=carrier_unit,
            countries=countries,
            nutrient_stores=nutrient_stores,
        ) -> xr.DataArray:
            # Carrier unit encodes the nutrient type: "Mt" for mass, "PJ" for energy (kcal)
            if carrier_unit == "Mt":
                rhs_vals = [
                    _per_capita_mass_to_mt_per_year(
                        float(value), float(population[country])
                    )
                    for country in countries
                ]
            else:
                rhs_vals = [
                    float(value)
                    * float(population[country])
                    * constants.DAYS_PER_YEAR
                    * constants.KCAL_TO_PJ
                    for country in countries
                ]
            return xr.DataArray(
                rhs_vals, coords={"name": nutrient_stores.index}, dims="name"
            )

        for key, operator, label in (
            ("equal", "==", "equal"),
            ("min", ">=", "min"),
            ("max", "<=", "max"),
        ):
            if bounds.get(key) is None:
                continue
            rhs = rhs_from(bounds[key])
            constr_name = f"macronutrient_{label}_{nutrient}"

            if operator == "==":
                m.add_constraints(lhs == rhs, name=f"GlobalConstraint-{constr_name}")
                n.global_constraints.add(
                    f"{constr_name}_" + nutrient_stores.index,
                    sense="==",
                    constant=rhs.values,
                    type="nutrition",
                )
                break

            if operator == ">=":
                m.add_constraints(lhs >= rhs, name=f"GlobalConstraint-{constr_name}")
            else:
                m.add_constraints(lhs <= rhs, name=f"GlobalConstraint-{constr_name}")

            n.global_constraints.add(
                f"{constr_name}_" + nutrient_stores.index,
                sense=operator,
                constant=rhs.values,
                type="nutrition",
            )


def add_food_group_constraints(
    n: pypsa.Network,
    food_group_cfg: dict | None,
    population: dict[str, float],
    per_country_equal: dict[str, dict[str, float]] | None = None,
) -> None:
    """Add per-country food group bounds on store levels."""

    if not food_group_cfg and not per_country_equal:
        return

    food_group_cfg = food_group_cfg or {}
    per_country_equal = per_country_equal or {}

    m = n.model
    store_e = m.variables["Store-e"].sel(snapshot="now")
    stores_df = n.stores.static

    groups = set(food_group_cfg) | set(per_country_equal)
    for group in groups:
        bounds = food_group_cfg.get(group, {})
        if not bounds and group not in per_country_equal:
            continue

        group_stores = stores_df[stores_df["carrier"] == f"group_{group}"]
        countries = group_stores["country"].astype(str)
        lhs = store_e.sel(name=group_stores.index)

        def rhs_from(
            value: float, countries=countries, group_stores=group_stores
        ) -> xr.DataArray:
            rhs_vals = [
                _per_capita_mass_to_mt_per_year(
                    float(value), float(population[country])
                )
                for country in countries
            ]
            return xr.DataArray(
                rhs_vals, coords={"name": group_stores.index}, dims="name"
            )

        def rhs_from_equal(
            group=group, countries=countries, group_stores=group_stores, bounds=bounds
        ) -> xr.DataArray | None:
            overrides = per_country_equal.get(group)
            if overrides:
                rhs_vals = [
                    _per_capita_mass_to_mt_per_year(
                        float(overrides[country]), float(population[country])
                    )
                    for country in countries
                ]
                return xr.DataArray(
                    rhs_vals, coords={"name": group_stores.index}, dims="name"
                )
            if bounds.get("equal") is None:
                return None
            return rhs_from(bounds["equal"])

        # Apply at most one equality; otherwise allow independent min/max bounds
        for key, operator, label in (
            ("equal", "==", "equal"),
            ("min", ">=", "min"),
            ("max", "<=", "max"),
        ):
            if key == "equal":
                rhs = rhs_from_equal()
                if rhs is None:
                    continue
            else:
                if bounds.get(key) is None:
                    continue
                rhs = rhs_from(bounds[key])

            constr_name = f"food_group_{label}_{group}"

            if operator == "==":
                m.add_constraints(lhs == rhs, name=f"GlobalConstraint-{constr_name}")
                n.global_constraints.add(
                    f"{constr_name}_" + group_stores.index,
                    sense="==",
                    constant=rhs.values,
                    type="nutrition",
                )
                break

            if operator == ">=":
                m.add_constraints(lhs >= rhs, name=f"GlobalConstraint-{constr_name}")
            else:
                m.add_constraints(lhs <= rhs, name=f"GlobalConstraint-{constr_name}")

            n.global_constraints.add(
                f"{constr_name}_" + group_stores.index,
                sense=operator,
                constant=rhs.values,
                type="nutrition",
            )


def _add_sos2_with_fallback(m, variable, sos_dim: str, solver_name: str) -> list[str]:
    """Add SOS2 or binary fallback depending on solver support."""

    if solver_name.lower() != "highs":
        m.add_sos_constraints(variable, sos_type=2, sos_dim=sos_dim)
        return []

    coords = variable.coords[sos_dim]
    n_points = len(coords)
    if n_points <= 1:
        return []

    other_dims = [dim for dim in variable.dims if dim != sos_dim]

    interval_dim = f"{sos_dim}_interval"
    suffix = 1
    while interval_dim in variable.dims:
        interval_dim = f"{sos_dim}_interval{suffix}"
        suffix += 1

    interval_index = pd.Index(range(n_points - 1), name=interval_dim)
    binary_coords = [variable.coords[d] for d in other_dims] + [interval_index]

    # Use counter instead of checking existing variables
    _SOS2_COUNTER[0] += 1
    base_name = f"{variable.name}_segment" if variable.name else "health_segment"
    binary_name = f"{base_name}_{_SOS2_COUNTER[0]}"

    binaries = m.add_variables(coords=binary_coords, binary=True, name=binary_name)

    m.add_constraints(binaries.sum(interval_dim) == 1)

    # Vectorize SOS2 constraints: variable[i] <= binary[i-1] + binary[i]
    if n_points >= 2:
        adjacency_data = np.zeros((n_points, n_points - 1))
        indices = np.arange(n_points - 1)
        adjacency_data[indices, indices] = 1
        adjacency_data[indices + 1, indices] = 1

        adjacency = xr.DataArray(
            adjacency_data,
            coords={sos_dim: coords, interval_dim: range(n_points - 1)},
            dims=[sos_dim, interval_dim],
        )

        rhs = (adjacency * binaries).sum(interval_dim)
        m.add_constraints(variable <= rhs)

    return [binary_name]


def _apply_solver_threads_option(
    solver_options: dict, solver_name: str, threads: int
) -> dict:
    """Ensure the solver options include a threads override when configured."""

    solver_key = solver_name.lower()
    if solver_key == "gurobi":
        solver_options["Threads"] = threads
    elif solver_key == "highs":
        solver_options["threads"] = threads

    return solver_options


def add_ghg_pricing_to_objective(n: pypsa.Network, ghg_price_usd_per_t: float) -> None:
    """Add GHG emissions pricing to the objective function.

    Adds the cost of GHG emissions (stored in the 'ghg' store) to the
    objective function at solve time.

    Parameters
    ----------
    n : pypsa.Network
        The network containing the model.
    ghg_price_usd_per_t : float
        Price per tonne of CO2-equivalent in USD (config currency_year).
    """
    # Convert USD/tCO2 to bnUSD/MtCO2 (matching model units)
    ghg_price_bnusd_per_mt = (
        ghg_price_usd_per_t / constants.TONNE_TO_MEGATONNE * constants.USD_TO_BNUSD
    )

    # Add marginal storage cost to store
    n.stores.static.at["ghg", "marginal_cost_storage"] = ghg_price_bnusd_per_mt


def add_food_group_incentives_to_objective(
    n: pypsa.Network, incentives_paths: list[str]
) -> None:
    """Add food-group incentives/penalties to the objective function.

    Incentives are applied as adjustments to marginal storage costs of
    food group stores. Positive values penalize consumption; negative
    values subsidize consumption.

    Parameters
    ----------
    n : pypsa.Network
        The network containing the model.
    incentives_paths : list[str]
        Paths to CSVs with columns: group, country, adjustment_bnusd_per_mt
    """
    if not incentives_paths:
        raise ValueError("food_group_incentives enabled but no sources are configured")

    combined = []
    for path in incentives_paths:
        incentives_df = pd.read_csv(path)
        required = {"group", "country", "adjustment_bnusd_per_mt"}
        missing = required - set(incentives_df.columns)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(
                f"Missing required columns in incentives file {path}: {missing_text}"
            )

        incentives_df["country"] = incentives_df["country"].astype(str).str.upper()
        incentives_df["store_name"] = (
            "store_" + incentives_df["group"] + "_" + incentives_df["country"]
        )
        combined.append(incentives_df[["store_name", "adjustment_bnusd_per_mt"]].copy())

    all_incentives = pd.concat(combined, ignore_index=True)
    summed = (
        all_incentives.groupby("store_name")["adjustment_bnusd_per_mt"]
        .sum()
        .reset_index()
    )

    if "marginal_cost_storage" not in n.stores.static.columns:
        n.stores.static["marginal_cost_storage"] = 0.0

    store_index = n.stores.static.index
    missing_stores = summed[~summed["store_name"].isin(store_index)]
    if not missing_stores.empty:
        sample = ", ".join(missing_stores["store_name"].head(5))
        logger.warning(
            "Missing %d food group stores for incentives (examples: %s)",
            len(missing_stores),
            sample,
        )

    applicable = summed[summed["store_name"].isin(store_index)]
    if applicable.empty:
        logger.info(
            "No applicable food group incentives found in %d sources",
            len(incentives_paths),
        )
        return

    n.stores.static.loc[applicable["store_name"], "marginal_cost_storage"] = (
        n.stores.static.loc[applicable["store_name"], "marginal_cost_storage"].astype(
            float
        )
        + applicable["adjustment_bnusd_per_mt"].astype(float).values
    )

    logger.info(
        "Applied food-group incentives to %d stores from %d sources",
        len(applicable),
        len(incentives_paths),
    )


def build_residue_feed_fraction_by_country(
    config: dict, m49_path: str
) -> dict[str, float]:
    """Build per-country residue feed fraction overrides from config."""
    overrides = config["residues"]["max_feed_fraction_by_region"]
    if not overrides:
        return {}

    countries = [str(country).upper() for country in config["countries"]]

    m49_df = pd.read_csv(m49_path, sep=";", encoding="utf-8-sig", comment="#")
    m49_df = m49_df[m49_df["ISO-alpha3 Code"].notna()]
    m49_df["iso3"] = m49_df["ISO-alpha3 Code"].astype(str).str.upper()
    m49_df = m49_df[m49_df["iso3"].isin(countries)]

    region_to_countries = m49_df.groupby("Region Name")["iso3"].apply(list).to_dict()
    subregion_to_countries = (
        m49_df.groupby("Sub-region Name")["iso3"].apply(list).to_dict()
    )

    region_overrides = {
        key: overrides[key] for key in overrides if key in region_to_countries
    }
    subregion_overrides = {
        key: overrides[key] for key in overrides if key in subregion_to_countries
    }
    country_overrides = {key: overrides[key] for key in overrides if key in countries}

    unknown = (
        set(overrides)
        - set(region_overrides)
        - set(subregion_overrides)
        - set(country_overrides)
    )
    if unknown:
        unknown_text = ", ".join(sorted(unknown))
        raise ValueError(
            f"Unknown residues.max_feed_fraction_by_region keys: {unknown_text}"
        )

    per_country: dict[str, float] = {}
    for region, value in region_overrides.items():
        for country in region_to_countries[region]:
            per_country[country] = float(value)
    for subregion, value in subregion_overrides.items():
        for country in subregion_to_countries[subregion]:
            per_country[country] = float(value)
    for country, value in country_overrides.items():
        per_country[country] = float(value)

    return per_country


def _residue_bus_country(residue_bus: str) -> str:
    return residue_bus.rsplit("_", 1)[-1].upper()


def add_residue_feed_constraints(
    n: pypsa.Network,
    max_feed_fraction: float,
    max_feed_fraction_by_country: dict[str, float],
) -> None:
    """Add constraints limiting residue removal for animal feed.

    Constrains the fraction of residues that can be removed for feed vs.
    incorporated into soil. The constraint is formulated as::

        feed_use ≤ (max_feed_fraction / (1 - max_feed_fraction)) x incorporation

    This ensures that if a total amount R of residue is generated::

        R = feed_use + incorporation
        feed_use ≤ max_feed_fraction x R

    Parameters
    ----------
    n : pypsa.Network
        The network containing the model.
    max_feed_fraction : float
        Maximum fraction of residues that can be used for feed (e.g., 0.30 for 30%).
    max_feed_fraction_by_country : dict[str, float]
        Overrides keyed by ISO3 country code.
    """

    m = n.model

    # Get link flow variables and link data
    link_p = m.variables["Link-p"].sel(snapshot="now")
    links_df = n.links.static

    # Find residue feed links (carrier="convert_to_feed", bus0 starts with "residue_")
    feed_mask = (links_df["carrier"] == "convert_to_feed") & (
        links_df["bus0"].str.startswith("residue_")
    )
    feed_links_df = links_df[feed_mask]

    # Find incorporation links (carrier="residue_incorporation")
    incorp_mask = links_df["carrier"] == "residue_incorporation"
    incorp_links_df = links_df[incorp_mask]

    if feed_links_df.empty or incorp_links_df.empty:
        logger.info(
            "No residue feed limit constraints added (missing feed or incorporation links)"
        )
        return

    # Identify common residue buses
    feed_buses = set(feed_links_df["bus0"].unique())
    incorp_buses = set(incorp_links_df["bus0"].unique())
    common_buses = sorted(feed_buses.intersection(incorp_buses))

    if not common_buses:
        logger.info(
            "No residue feed limit constraints added (no matching residue flows found)"
        )
        return

    # Filter DataFrames to common buses
    feed_links_df = feed_links_df[feed_links_df["bus0"].isin(common_buses)]
    incorp_links_df = incorp_links_df[incorp_links_df["bus0"].isin(common_buses)]

    # Prepare mapping DataArrays for groupby
    # Map feed link names to their residue bus
    feed_bus_map = xr.DataArray(
        feed_links_df["bus0"],
        coords={"name": feed_links_df.index},
        dims="name",
        name="residue_bus",
    )

    # Map incorp link names to their residue bus
    incorp_bus_map = xr.DataArray(
        incorp_links_df["bus0"],
        coords={"name": incorp_links_df.index},
        dims="name",
        name="residue_bus",
    )

    # Get variables
    feed_vars = link_p.sel(name=feed_links_df.index)
    incorp_vars = link_p.sel(name=incorp_links_df.index)

    # Sum/Group
    # Group feed vars by residue bus and sum
    feed_sum = feed_vars.groupby(feed_bus_map).sum()

    # Group incorp vars by residue bus and sum (handles alignment)
    incorp_flow = incorp_vars.groupby(incorp_bus_map).sum()

    ratios = []
    for bus in common_buses:
        country = _residue_bus_country(bus)
        max_fraction = max_feed_fraction_by_country.get(country, max_feed_fraction)
        ratios.append(max_fraction / (1.0 - max_fraction))

    ratio = xr.DataArray(
        ratios, coords={"residue_bus": common_buses}, dims="residue_bus"
    )

    # Add constraints
    constr_name = "residue_feed_limit"
    m.add_constraints(
        feed_sum <= ratio * incorp_flow,
        name=f"GlobalConstraint-{constr_name}",
    )

    # Add GlobalConstraints for shadow price tracking
    gc_names = [f"{constr_name}_{bus}" for bus in common_buses]
    n.global_constraints.add(
        gc_names,
        sense="<=",
        constant=0.0,  # RHS is dynamic (depends on incorp_flow), use 0 as placeholder
        type="residue_feed",
    )

    if max_feed_fraction_by_country:
        logger.info(
            "Applied residue feed fraction overrides for %d countries",
            len(max_feed_fraction_by_country),
        )

    logger.info(
        "Added %d residue feed limit constraints (max %.0f%% for feed)",
        len(common_buses),
        max_feed_fraction * 100,
    )


def add_animal_production_constraints(
    n: pypsa.Network,
    fao_production: pd.DataFrame,
    food_to_group: dict[str, str],
    loss_waste: pd.DataFrame,
) -> None:
    """Add constraints to fix animal production at FAO levels per country.

    For each (country, product) combination in the FAO data, adds a constraint
    that total production from all feed categories equals the FAO target,
    adjusted for food loss and waste. Since the model applies FLW to the
    feed→product efficiency, the constraint target must also be adjusted
    to net production (gross FAO production * (1-loss) * (1-waste)).

    Parameters
    ----------
    n : pypsa.Network
        The network containing the model.
    fao_production : pd.DataFrame
        FAO production data with columns: country, product, production_mt.
    food_to_group : dict[str, str]
        Mapping from product names to food group names for FLW lookup.
    loss_waste : pd.DataFrame
        Food loss and waste fractions with columns: country, food_group,
        loss_fraction, waste_fraction.
    """
    if fao_production.empty:
        logger.warning(
            "No FAO animal production data available; skipping production constraints"
        )
        return

    # Build FLW lookup: (country, food_group) -> (1-loss)*(1-waste)
    flw_multipliers: dict[tuple[str, str], float] = {}
    for _, row in loss_waste.iterrows():
        key = (str(row["country"]), str(row["food_group"]))
        loss_frac = float(row["loss_fraction"])
        waste_frac = float(row["waste_fraction"])
        flw_multipliers[key] = (1.0 - loss_frac) * (1.0 - waste_frac)

    m = n.model
    link_p = m.variables["Link-p"].sel(snapshot="now")
    links_df = n.links.static

    # Filter to animal production links using carrier
    # Animal production links have carriers starting with "produce_"
    prod_mask = links_df["carrier"].str.startswith("produce_")
    prod_links = links_df[prod_mask]

    if prod_links.empty:
        logger.info("No animal production links found.")
        return

    products = prod_links["product"].astype(str)
    countries = prod_links["country"].astype(str)

    # Prepare DataArrays aligned with the filtered links
    link_names = prod_links.index

    # Efficiencies
    efficiencies = xr.DataArray(
        prod_links["efficiency"].values, coords={"name": link_names}, dims="name"
    )

    # Production = p * efficiency
    # Group by (product, country) and sum
    production_vars = link_p.sel(name=link_names)

    grouper = pd.MultiIndex.from_arrays(
        [products.values, countries.values], names=["product", "country"]
    )
    da_grouper = xr.DataArray(grouper, coords={"name": link_names}, dims="name")

    total_production = (production_vars * efficiencies).groupby(da_grouper).sum()

    target_series = fao_production.set_index(["product", "country"])[
        "production_mt"
    ].astype(float)

    # Adjust targets by FLW: net_target = gross_target * (1-loss) * (1-waste)
    adjusted_targets = []
    for product, country in target_series.index:
        gross_value = target_series.loc[(product, country)]
        group = food_to_group[product]
        multiplier = flw_multipliers[(country, group)]
        adjusted_targets.append(gross_value * multiplier)
    target_series = pd.Series(adjusted_targets, index=target_series.index)

    model_index = pd.Index(total_production.coords["group"].values, name="group")
    common_index = model_index.intersection(target_series.index)

    if common_index.empty:
        logger.warning(
            "No matching animal production targets found for model structure."
        )
        return

    lhs = total_production.sel(group=common_index)
    rhs = xr.DataArray(
        target_series.loc[common_index].values,
        coords={"group": common_index},
        dims="group",
    )

    constr_name = "animal_production_target"
    m.add_constraints(lhs == rhs, name=f"GlobalConstraint-{constr_name}")

    # Add GlobalConstraints for shadow price tracking
    gc_names = [f"{constr_name}_{prod}_{country}" for prod, country in common_index]
    n.global_constraints.add(
        gc_names,
        sense="==",
        constant=rhs.values,
        type="production_target",
    )

    logger.info(
        "Added %d country-level animal production constraints (FLW-adjusted)",
        len(common_index),
    )


def add_production_stability_constraints(
    n: pypsa.Network,
    crop_baseline: pd.DataFrame | None,
    crop_to_fao_item: dict[str, str],
    animal_baseline: pd.DataFrame | None,
    stability_cfg: dict,
    food_to_group: dict[str, str],
    loss_waste: pd.DataFrame,
) -> None:
    """Add constraints limiting production deviation from baseline levels.

    For crops and animal products, applies per-(product, country) bounds:
    ``(1 - delta) * baseline <= production <= (1 + delta) * baseline``

    Products with zero baseline are constrained to zero production.

    Note: Multi-cropping is disabled when production stability is enabled.

    Parameters
    ----------
    n : pypsa.Network
        The network containing the model.
    crop_baseline : pd.DataFrame | None
        FAO crop production with columns: country, crop, production_tonnes.
    crop_to_fao_item : dict[str, str]
        Mapping from crop names to FAO item names; used to aggregate crops
        that share an FAO item (e.g., dryland-rice and wetland-rice both
        map to "Rice").
    animal_baseline : pd.DataFrame | None
        FAO animal production with columns: country, product, production_mt.
    stability_cfg : dict
        Configuration with enabled, crops.max_relative_deviation, etc.
    food_to_group : dict[str, str]
        Mapping from product names to food group names for FLW lookup.
    loss_waste : pd.DataFrame
        Food loss and waste fractions with columns: country, food_group,
        loss_fraction, waste_fraction.
    """
    if not stability_cfg["enabled"]:
        return

    m = n.model
    link_p = m.variables["Link-p"].sel(snapshot="now")
    links_df = n.links.static

    # --- CROP PRODUCTION BOUNDS ---
    crops_cfg = stability_cfg["crops"]
    if crops_cfg["enabled"] and crop_baseline is not None:
        _add_crop_stability_constraints(
            n, link_p, links_df, crop_baseline, crop_to_fao_item, crops_cfg
        )

    # --- ANIMAL PRODUCTION BOUNDS ---
    animals_cfg = stability_cfg["animals"]
    if animals_cfg["enabled"] and animal_baseline is not None:
        _add_animal_stability_constraints(
            n, link_p, links_df, animal_baseline, animals_cfg, food_to_group, loss_waste
        )


def _add_crop_stability_constraints(
    n: pypsa.Network,
    link_p,
    links_df: pd.DataFrame,
    crop_baseline: pd.DataFrame,
    crop_to_fao_item: dict[str, str],
    crops_cfg: dict,
) -> None:
    """Add crop production stability bounds.

    Crops that share a FAO item (e.g., dryland-rice and wetland-rice both map
    to "Rice") are aggregated together for the constraint.
    """
    m = n.model
    delta = crops_cfg["max_relative_deviation"]

    # Filter to crop production links using the crop column
    # Note: some links have empty string instead of NaN, so check for both
    crop_mask = links_df["crop"].notna() & (links_df["crop"] != "")
    crop_links = links_df[crop_mask].copy()

    if crop_links.empty:
        logger.info(
            "No crop production links found; skipping crop stability constraints"
        )
        return

    crops = crop_links["crop"].astype(str)
    countries = crop_links["country"].astype(str)
    link_names = crop_links.index

    # Map crops to FAO items; use crop name as fallback for unmapped crops
    fao_items = crops.map(lambda c: crop_to_fao_item.get(c, c))
    # Filter out crops with empty/nan FAO item (e.g., alfalfa, biomass-sorghum)
    valid_mask = (
        fao_items.notna() & (fao_items != "") & (fao_items.str.lower() != "nan")
    )

    if not valid_mask.any():
        logger.info(
            "No crops with FAO item mappings; skipping crop stability constraints"
        )
        return

    fao_items = fao_items[valid_mask]
    countries_filtered = countries[valid_mask]
    link_names_filtered = link_names[valid_mask]
    efficiencies_filtered = crop_links.loc[valid_mask, "efficiency"].values

    # Efficiencies (yield: Mt/Mha)
    efficiencies = xr.DataArray(
        efficiencies_filtered, coords={"name": link_names_filtered}, dims="name"
    )

    # Production = p * efficiency (p is land in Mha)
    production_vars = link_p.sel(name=link_names_filtered)

    # Group by (fao_item, country) to aggregate related crops
    grouper = pd.MultiIndex.from_arrays(
        [fao_items.values, countries_filtered.values], names=["fao_item", "country"]
    )
    da_grouper = xr.DataArray(
        grouper, coords={"name": link_names_filtered}, dims="name"
    )

    total_production = (production_vars * efficiencies).groupby(da_grouper).sum()

    # Convert baseline to Mt and aggregate by FAO item
    baseline_df = crop_baseline.copy()
    baseline_df["production_mt"] = baseline_df["production_tonnes"] * 1e-6
    # Map baseline crops to FAO items
    baseline_df["fao_item"] = baseline_df["crop"].map(
        lambda c: crop_to_fao_item.get(c, c)
    )
    # Aggregate baseline by (fao_item, country) - this sums the split values back
    baseline_agg = (
        baseline_df.groupby(["fao_item", "country"])["production_mt"]
        .sum()
        .reset_index()
    )
    target_series = baseline_agg.set_index(["fao_item", "country"])["production_mt"]

    # Match to model index
    model_index = pd.Index(total_production.coords["group"].values, name="group")
    common_index = model_index.intersection(target_series.index)

    if common_index.empty:
        logger.warning("No matching crop production targets for stability bounds")
        return

    # Build RHS bounds
    baselines = target_series.loc[common_index].values
    lower_bounds = np.maximum(0.0, (1.0 - delta) * baselines)
    upper_bounds = (1.0 + delta) * baselines

    rhs_lower = xr.DataArray(lower_bounds, coords={"group": common_index}, dims="group")
    rhs_upper = xr.DataArray(upper_bounds, coords={"group": common_index}, dims="group")

    # Handle zero baselines: force production to zero
    zero_mask = baselines == 0
    nonzero_mask = ~zero_mask

    if zero_mask.any():
        zero_index = common_index[zero_mask]
        lhs_zero = total_production.sel(group=zero_index)
        constr_name = "crop_production_zero"
        m.add_constraints(lhs_zero == 0, name=f"GlobalConstraint-{constr_name}")
        gc_names = [
            f"{constr_name}_{fao_item}_{country}" for fao_item, country in zero_index
        ]
        n.global_constraints.add(
            gc_names,
            sense="==",
            constant=0.0,
            type="production_stability",
        )
        logger.info(
            "Added %d crop production constraints for zero-baseline (fao_item, country) pairs",
            int(zero_mask.sum()),
        )

    if nonzero_mask.any():
        nonzero_index = common_index[nonzero_mask]
        lhs_nonzero = total_production.sel(group=nonzero_index)
        lower_nonzero = rhs_lower.sel(group=nonzero_index)
        upper_nonzero = rhs_upper.sel(group=nonzero_index)

        constr_name_min = "crop_production_min"
        constr_name_max = "crop_production_max"
        m.add_constraints(
            lhs_nonzero >= lower_nonzero, name=f"GlobalConstraint-{constr_name_min}"
        )
        m.add_constraints(
            lhs_nonzero <= upper_nonzero, name=f"GlobalConstraint-{constr_name_max}"
        )

        gc_names_min = [
            f"{constr_name_min}_{fao_item}_{country}"
            for fao_item, country in nonzero_index
        ]
        gc_names_max = [
            f"{constr_name_max}_{fao_item}_{country}"
            for fao_item, country in nonzero_index
        ]
        n.global_constraints.add(
            gc_names_min,
            sense=">=",
            constant=lower_nonzero.values,
            type="production_stability",
        )
        n.global_constraints.add(
            gc_names_max,
            sense="<=",
            constant=upper_nonzero.values,
            type="production_stability",
        )

        logger.info(
            "Added %d crop production stability constraints (delta=%.0f%%)",
            2 * int(nonzero_mask.sum()),
            delta * 100,
        )

    # Log missing baselines (at FAO item level)
    missing = model_index.difference(target_series.index)
    if len(missing) > 0:
        examples = [f"{item}/{country}" for item, country in list(missing)[:5]]
        logger.warning(
            "Missing crop baseline data for %d (fao_item, country) pairs; examples: %s",
            len(missing),
            ", ".join(examples),
        )


def _add_animal_stability_constraints(
    n: pypsa.Network,
    link_p,
    links_df: pd.DataFrame,
    animal_baseline: pd.DataFrame,
    animals_cfg: dict,
    food_to_group: dict[str, str],
    loss_waste: pd.DataFrame,
) -> None:
    """Add animal production stability bounds.

    Reuses the aggregation logic from add_animal_production_constraints()
    but applies inequality bounds instead of equality.
    """
    m = n.model
    delta = animals_cfg["max_relative_deviation"]

    # Build FLW lookup (same as add_animal_production_constraints)
    flw_multipliers: dict[tuple[str, str], float] = {}
    for _, row in loss_waste.iterrows():
        key = (str(row["country"]), str(row["food_group"]))
        loss_frac = float(row["loss_fraction"])
        waste_frac = float(row["waste_fraction"])
        flw_multipliers[key] = (1.0 - loss_frac) * (1.0 - waste_frac)

    # Filter to animal production links using product column
    # Note: some links have empty string instead of NaN, so check for both
    prod_mask = links_df["product"].notna() & (links_df["product"] != "")
    prod_links = links_df[prod_mask]

    if prod_links.empty:
        logger.info(
            "No animal production links found; skipping animal stability constraints"
        )
        return

    products = prod_links["product"].astype(str)
    countries = prod_links["country"].astype(str)
    link_names = prod_links.index

    efficiencies = xr.DataArray(
        prod_links["efficiency"].values, coords={"name": link_names}, dims="name"
    )

    production_vars = link_p.sel(name=link_names)

    grouper = pd.MultiIndex.from_arrays(
        [products.values, countries.values], names=["product", "country"]
    )
    da_grouper = xr.DataArray(grouper, coords={"name": link_names}, dims="name")

    total_production = (production_vars * efficiencies).groupby(da_grouper).sum()

    # Build FLW-adjusted targets (same logic as add_animal_production_constraints)
    target_series = animal_baseline.set_index(["product", "country"])[
        "production_mt"
    ].astype(float)

    adjusted_targets = []
    for product, country in target_series.index:
        gross_value = target_series.loc[(product, country)]
        group = food_to_group.get(product, product)
        multiplier = flw_multipliers.get((country, group), 1.0)
        adjusted_targets.append(gross_value * multiplier)
    target_series = pd.Series(adjusted_targets, index=target_series.index)

    model_index = pd.Index(total_production.coords["group"].values, name="group")
    common_index = model_index.intersection(target_series.index)

    if common_index.empty:
        logger.warning("No matching animal production targets for stability bounds")
        return

    # Build bounds
    baselines = target_series.loc[common_index].values
    lower_bounds = np.maximum(0.0, (1.0 - delta) * baselines)
    upper_bounds = (1.0 + delta) * baselines

    rhs_lower = xr.DataArray(lower_bounds, coords={"group": common_index}, dims="group")
    rhs_upper = xr.DataArray(upper_bounds, coords={"group": common_index}, dims="group")

    # Handle zero baselines: force production to zero
    zero_mask = baselines == 0
    nonzero_mask = ~zero_mask

    if zero_mask.any():
        zero_index = common_index[zero_mask]
        lhs_zero = total_production.sel(group=zero_index)
        constr_name = "animal_production_zero"
        m.add_constraints(lhs_zero == 0, name=f"GlobalConstraint-{constr_name}")
        gc_names = [f"{constr_name}_{prod}_{country}" for prod, country in zero_index]
        n.global_constraints.add(
            gc_names,
            sense="==",
            constant=0.0,
            type="production_stability",
        )
        logger.info(
            "Added %d animal production constraints for zero-baseline (product, country) pairs",
            int(zero_mask.sum()),
        )

    if nonzero_mask.any():
        nonzero_index = common_index[nonzero_mask]
        lhs_nonzero = total_production.sel(group=nonzero_index)
        lower_nonzero = rhs_lower.sel(group=nonzero_index)
        upper_nonzero = rhs_upper.sel(group=nonzero_index)

        constr_name_min = "animal_production_min"
        constr_name_max = "animal_production_max"
        m.add_constraints(
            lhs_nonzero >= lower_nonzero, name=f"GlobalConstraint-{constr_name_min}"
        )
        m.add_constraints(
            lhs_nonzero <= upper_nonzero, name=f"GlobalConstraint-{constr_name_max}"
        )

        gc_names_min = [
            f"{constr_name_min}_{prod}_{country}" for prod, country in nonzero_index
        ]
        gc_names_max = [
            f"{constr_name_max}_{prod}_{country}" for prod, country in nonzero_index
        ]
        n.global_constraints.add(
            gc_names_min,
            sense=">=",
            constant=lower_nonzero.values,
            type="production_stability",
        )
        n.global_constraints.add(
            gc_names_max,
            sense="<=",
            constant=upper_nonzero.values,
            type="production_stability",
        )

        logger.info(
            "Added %d animal production stability constraints (delta=%.0f%%)",
            2 * int(nonzero_mask.sum()),
            delta * 100,
        )

    # Log missing baselines
    missing = model_index.difference(target_series.index)
    if len(missing) > 0:
        examples = [f"{p}/{c}" for p, c in list(missing)[:5]]
        logger.warning(
            "Missing animal baseline data for %d (product, country) pairs; examples: %s",
            len(missing),
            ", ".join(examples),
        )


def _get_consumption_link_map(
    p_names: pd.Index,
    links_df: pd.DataFrame,
    food_map: pd.DataFrame,
    cluster_lookup: dict[str, int],
    cluster_population: dict[int, float],
) -> pd.DataFrame:
    """Map consumption links to health clusters and risk factors."""
    # Filter for consumption links
    consume_mask = links_df.index.isin(p_names) & links_df.index.str.startswith(
        "consume_"
    )
    consume_links = links_df[consume_mask]

    if consume_links.empty:
        return pd.DataFrame()

    # Extract food and country from link attributes (set during model building)
    df = pd.DataFrame(
        {
            "link_name": consume_links.index,
            "food": consume_links["food"],
            "country": consume_links["country"],
        }
    )

    # Merge with food_map
    df = df.merge(food_map, left_on="food", right_index=True)

    # Map to cluster
    df["cluster"] = df["country"].map(cluster_lookup)
    df = df.dropna(subset=["cluster"])
    df["cluster"] = df["cluster"].astype(int)

    # Map to population
    df["population"] = df["cluster"].map(cluster_population)
    df = df[df["population"] > 0]

    # Calculate coefficient: share * grams per megatonne / (365 * population)
    df["coeff"] = (
        df["share"] * constants.GRAMS_PER_MEGATONNE / (365.0 * df["population"])
    )

    return df


def add_health_objective(
    n: pypsa.Network,
    risk_breakpoints_path: str,
    cluster_cause_path: str,
    cause_log_path: str,
    cluster_summary_path: str,
    clusters_path: str,
    population_totals_path: str,
    risk_factors: list[str],
    risk_cause_map: dict[str, list[str]],
    solver_name: str,
    enforce_baseline: bool,
) -> None:
    """Link SOS2-based health costs to explicit PyPSA stores.

    The function builds the same intake→relative-risk logic as before, but
    instead of writing directly to the linopy objective it constrains the
    level of per-cluster, per-cause YLL stores. The monetary contribution is
    then handled by the store ``marginal_cost_storage`` configured during network
    construction.

    Health costs are formulated relative to TMREL (Theoretical Minimum Risk
    Exposure Level) intake, such that cost is zero when RR = RR_ref. Store
    energy levels represent (RR - RR_ref) * (YLL_base / RR_ref) and are
    non-negative since TMREL is the theoretical minimum risk level.
    """

    m = n.model

    risk_breakpoints = pd.read_csv(risk_breakpoints_path)
    cluster_cause = pd.read_csv(cluster_cause_path)
    cause_log_breakpoints = pd.read_csv(cause_log_path)
    cluster_summary = pd.read_csv(cluster_summary_path)
    if "health_cluster" in cluster_summary.columns:
        cluster_summary["health_cluster"] = cluster_summary["health_cluster"].astype(
            int
        )
    cluster_map = pd.read_csv(clusters_path)
    population_totals = pd.read_csv(population_totals_path)

    # Load food→risk factor mapping from food_groups.csv (only GBD risk factors)
    food_groups_df = pd.read_csv(snakemake.input.food_groups)
    food_map = food_groups_df[food_groups_df["group"].isin(risk_factors)].copy()
    food_map = food_map.rename(columns={"group": "risk_factor"})
    food_map["share"] = 1.0
    food_map = food_map.set_index("food")[["risk_factor", "share"]]

    cluster_lookup = cluster_map.set_index("country_iso3")["health_cluster"].to_dict()
    cluster_population_baseline = cluster_summary.set_index("health_cluster")[
        "population_persons"
    ].to_dict()

    cluster_cause_metadata = cluster_cause.set_index(["health_cluster", "cause"])

    population_totals = population_totals.dropna(subset=["iso3", "population"]).copy()
    population_totals["iso3"] = population_totals["iso3"].astype(str).str.upper()
    population_map = population_totals.set_index("iso3")["population"].to_dict()

    cluster_population_planning: dict[int, float] = defaultdict(float)
    for iso3, cluster in cluster_lookup.items():
        value = float(population_map.get(iso3, 0.0))
        if value <= 0:
            continue
        cluster_population_planning[int(cluster)] += value

    cluster_population: dict[int, float] = {}
    for cluster, baseline_pop in cluster_population_baseline.items():
        planning_pop = cluster_population_planning.get(int(cluster), 0.0)
        if planning_pop > 0:
            cluster_population[int(cluster)] = planning_pop
        else:
            cluster_population[int(cluster)] = float(baseline_pop)

    risk_breakpoints = risk_breakpoints.sort_values(
        ["risk_factor", "intake_g_per_day", "cause"]
    )
    cause_log_breakpoints = cause_log_breakpoints.sort_values(["cause", "log_rr_total"])

    logger.info(
        "Health data: %d risk breakpoints across %d risks / %d causes; %d cause breakpoints",
        len(risk_breakpoints),
        risk_breakpoints["risk_factor"].nunique(),
        risk_breakpoints["cause"].nunique(),
        len(cause_log_breakpoints),
    )

    # Filter risk_cause_map to only include risk factors in the breakpoints
    available_risks = set(risk_breakpoints["risk_factor"].unique())
    risk_cause_map = {
        r: causes for r, causes in risk_cause_map.items() if r in available_risks
    }

    # Restrict to configured risk-cause pairs to avoid silent zeros
    allowed_pairs = {(r, c) for r, causes in risk_cause_map.items() for c in causes}
    rb_pairs = set(zip(risk_breakpoints["risk_factor"], risk_breakpoints["cause"]))
    missing_pairs = sorted(allowed_pairs - rb_pairs)
    if missing_pairs:
        text = ", ".join([f"{r}:{c}" for r, c in missing_pairs])
        raise ValueError(f"Risk breakpoints missing required pairs: {text}")

    p = m.variables["Link-p"].sel(snapshot="now")

    if enforce_baseline:
        # Fixed dietary intakes: skip interpolation and pin store levels to the
        # baseline RR reported in the precomputed health tables.
        store_e = m.variables["Store-e"].sel(snapshot="now")
        health_stores = (
            n.stores.static[
                n.stores.static["carrier"].notna()
                & n.stores.static["carrier"].str.startswith("yll_")
            ]
            .reset_index()
            .set_index(["health_cluster", "cause"])
        )
        constraints_added = 0
        if "log_rr_total_baseline" not in cluster_cause.columns:
            raise ValueError("cluster_cause must contain log_rr_total_baseline")
        for (cluster, cause), row in cluster_cause_metadata.iterrows():
            rr_ref = math.exp(float(row["log_rr_total_ref"]))
            rr_base = math.exp(float(row["log_rr_total_baseline"]))
            yll_base = float(row["yll_base"])
            delta = max(0.0, rr_base - rr_ref)
            store_level = delta * (yll_base / rr_ref) * constants.YLL_TO_MILLION_YLL
            store_name = health_stores.loc[(int(cluster), str(cause)), "name"]
            m.add_constraints(
                store_e.sel(name=store_name) == store_level,
                name=f"health_store_fixed_c{cluster}_cause{cause}",
            )
            constraints_added += 1
        logger.info(
            "Health stores fixed to baseline intake: added %d equality constraints",
            constraints_added,
        )
        return

    # --- Stage 1: Intake to Log-RR ---

    # Vectorized Link Mapping
    link_map = _get_consumption_link_map(
        pd.Index(p.coords["name"].values),
        n.links.static,
        food_map,
        cluster_lookup,
        cluster_population,
    )

    if link_map.empty:
        logger.info("No consumption links map to health risk factors; skipping")
        return

    logger.info(
        "Health intake mapping: %d links -> %d cluster-risk pairs across %d clusters",
        len(link_map),
        len(link_map[["cluster", "risk_factor"]].drop_duplicates()),
        link_map["cluster"].nunique(),
    )

    risk_data = {}
    for risk, grp in risk_breakpoints.groupby("risk_factor"):
        intakes = pd.Index(sorted(grp["intake_g_per_day"].unique()), name="intake")
        if intakes.empty:
            continue
        pivot = (
            grp.pivot_table(
                index="intake_g_per_day",
                columns="cause",
                values="log_rr",
                aggfunc="first",
            )
            .reindex(intakes, axis=0)
            .sort_index()
        )
        intake_steps = pd.Index(range(len(intakes)), name="intake_step")
        pivot.index = intake_steps
        risk_data[risk] = {
            "intake_steps": intake_steps,
            "intake_values": xr.DataArray(
                intakes.values, coords={"intake_step": intake_steps}, dims="intake_step"
            ),
            "log_rr": pivot,
        }

    cause_breakpoint_data = {
        cause: df.sort_values("log_rr_total")
        for cause, df in cause_log_breakpoints.groupby("cause")
    }

    # Group (cluster, risk) pairs by intake coordinate patterns
    # Identify unique (cluster, risk) pairs from the map
    unique_pairs = link_map[["cluster", "risk_factor"]].drop_duplicates()

    intake_groups: dict[tuple[float, ...], list[tuple[int, str]]] = defaultdict(list)
    for _, row in unique_pairs.iterrows():
        cluster = int(row["cluster"])
        risk = row["risk_factor"]

        risk_table = risk_data.get(risk)
        if risk_table is None:
            continue

        coords_key = tuple(risk_table["intake_values"].values)
        intake_groups[coords_key].append((cluster, risk))

    log_rr_totals_dict = {}

    for coords_key, group_pairs in intake_groups.items():
        intake_steps = risk_table["intake_steps"]

        # Identify group labels
        cluster_risk_labels = [f"c{cluster}_r{risk}" for cluster, risk in group_pairs]
        cluster_risk_index = pd.Index(cluster_risk_labels, name="cluster_risk")

        # Create lambdas (vectorized)
        risk_label = str(group_pairs[0][1])
        lambdas_group = m.add_variables(
            lower=0,
            upper=1,
            coords=[cluster_risk_index, intake_steps],
            name=f"health_lambda_group_{next(_LAMBDA_GROUP_COUNTER)}_{risk_label}",
        )

        # Register all variables
        _register_health_variable(m, lambdas_group.name)

        # Single SOS2 constraint call for entire group
        aux_names = _add_sos2_with_fallback(
            m, lambdas_group, sos_dim="intake_step", solver_name=solver_name
        )
        for aux_name in aux_names:
            _register_health_variable(m, aux_name)

        # Vectorized convexity constraints
        m.add_constraints(lambdas_group.sum("intake_step") == 1)

        # --- Intake Balance ---
        # LHS: Sum of link flows * coeffs
        # Filter link_map for this group
        group_map_df = pd.DataFrame(group_pairs, columns=["cluster", "risk_factor"])
        group_map_df["cluster_risk"] = cluster_risk_labels

        # Join link_map with group_map_df
        merged_links = link_map.merge(
            group_map_df, on=["cluster", "risk_factor"], how="inner"
        )

        if not merged_links.empty:
            # Create sparse aggregation
            relevant_links = merged_links["link_name"].values
            # Construct DataArrays with "name" coordinate to align with p
            coeffs = xr.DataArray(
                merged_links["coeff"].values,
                coords={"name": relevant_links},
                dims="name",
            )
            grouper = xr.DataArray(
                merged_links["cluster_risk"].values,
                coords={"name": relevant_links},
                dims="name",
            )

            # Groupby sum on DataArray of LinearExpressions (p) works in linopy
            flow_expr = (p.sel(name=relevant_links) * coeffs).groupby(grouper).sum()

            # RHS: Intake interpolation
            coeff_intake = risk_table["intake_values"]
            intake_expr = (lambdas_group * coeff_intake).sum("intake_step")

            # Add constraints vectorized
            m.add_constraints(
                flow_expr == intake_expr,
                name=f"health_intake_balance_group_{hash(coords_key)}",
            )

        # --- Log RR Calculation ---
        # Collect log_rr matrices
        log_rr_frames = []
        for _cluster, risk in group_pairs:
            df = risk_data[risk]["log_rr"]  # index=intake, cols=causes
            log_rr_frames.append(df)

        if not log_rr_frames:
            continue

        # Concat along cluster_risk dimension
        combined_log_rr = pd.concat(
            log_rr_frames,
            keys=cluster_risk_index,
            names=["cluster_risk", "intake_step"],
        ).fillna(0.0)

        # Convert to DataArray: (cluster_risk, intake_step, cause)
        # Use stack() to flatten columns (cause) into index
        s_log = combined_log_rr.stack()
        s_log.index.names = ["cluster_risk", "intake_step", "cause"]
        da_log = xr.DataArray.from_series(s_log).fillna(0.0)

        # Calculate contribution: sum(lambda * log_rr) over intake
        # lambdas_group: (cluster_risk, intake_step)
        # da_log: (cluster_risk, intake_step, cause)
        # Result: (cause, cluster_risk) of LinearExpressions
        contrib = (lambdas_group * da_log).sum("intake_step")

        # Accumulate into totals by grouping cluster_risk -> cluster
        c_map = group_map_df.set_index("cluster_risk")["cluster"]
        # Ensure we only map coords present in contrib
        present_cr = contrib.coords["cluster_risk"].values
        cluster_grouper = xr.DataArray(
            c_map.loc[present_cr].values,
            coords={"cluster_risk": present_cr},
            dims="cluster_risk",
            name="cluster",
        )

        # Group sum over cluster_risk -> yields (cause, cluster)
        group_total = contrib.groupby(cluster_grouper).sum()

        # Accumulate into dictionary for Stage 2
        # group_total is a LinearExpression with dims (cause, cluster)
        # We iterate over coordinates to extract scalar expressions

        causes = group_total.coords["cause"].values
        clusters = group_total.coords["cluster"].values

        for c in clusters:
            for cause in causes:
                # Extract scalar expression for this (cluster, cause)
                expr = group_total.sel(cluster=c, cause=cause)

                key = (c, cause)
                if key in log_rr_totals_dict:
                    log_rr_totals_dict[key] = log_rr_totals_dict[key] + expr
                else:
                    log_rr_totals_dict[key] = expr
    # Group (cluster, cause) pairs by their log_total coordinate patterns
    # so we can reuse a single SOS2 variable for all pairs that share the same
    # breakpoint grid (one grid per cause).
    log_total_groups: dict[tuple[float, ...], list[tuple[int, str]]] = defaultdict(list)
    cluster_cause_data: dict[tuple[int, str], dict] = {}

    for (cluster, cause), row in cluster_cause_metadata.iterrows():
        cluster = int(cluster)
        cause = str(cause)
        yll_base = float(row["yll_base"])

        cause_bp = cause_breakpoint_data[cause]
        coords_key = tuple(cause_bp["log_rr_total"].values)
        if len(coords_key) == 1:
            raise ValueError(
                "Need at least two breakpoints for piecewise linear approximation"
            )

        log_total_groups[coords_key].append((cluster, cause))

        # Store data for later use
        log_rr_total_ref = float(row["log_rr_total_ref"])
        cluster_cause_data[(cluster, cause)] = {
            "yll_base": yll_base,
            "log_rr_total_ref": log_rr_total_ref,
            "rr_ref": math.exp(log_rr_total_ref),
            "cause_bp": cause_bp,
        }

    logger.info(
        "Health risk aggregation: %d (cluster, cause) pairs grouped into %d log-RR grids",
        len(cluster_cause_data),
        len(log_total_groups),
    )

    store_e = m.variables["Store-e"].sel(snapshot="now")
    health_stores = (
        n.stores.static[
            n.stores.static["carrier"].notna()
            & n.stores.static["carrier"].str.startswith("yll_")
        ]
        .reset_index()
        .set_index(["health_cluster", "cause"])
    )

    constraints_added = 0

    # Process each group with vectorized operations. The outer loop handles one
    # shared grid (coords_key) at a time; the inner loop attaches each
    # (cluster, cause) pair using that grid to its own balance and store.
    for coords_key, cluster_cause_pairs in log_total_groups.items():
        log_total_vals = np.asarray(coords_key, dtype=float)
        log_total_steps = pd.Index(range(len(log_total_vals)), name="log_total_step")

        # Create flattened index for this group
        cluster_cause_labels = [
            f"c{cluster}_cause{cause}" for cluster, cause in cluster_cause_pairs
        ]
        cluster_cause_index = pd.Index(cluster_cause_labels, name="cluster_cause")

        # Single vectorized variable creation
        cause_label = str(cluster_cause_pairs[0][1])
        lambda_total_group = m.add_variables(
            lower=0,
            upper=1,
            coords=[cluster_cause_index, log_total_steps],
            name=f"health_lambda_total_group_{next(_TOTAL_GROUP_COUNTER)}_{cause_label}",
        )

        # Register all variables
        _register_health_variable(m, lambda_total_group.name)

        # Single SOS2 constraint call for entire group
        aux_names = _add_sos2_with_fallback(
            m, lambda_total_group, sos_dim="log_total_step", solver_name=solver_name
        )
        for aux_name in aux_names:
            _register_health_variable(m, aux_name)
        # Vectorized convexity constraints
        m.add_constraints(lambda_total_group.sum("log_total_step") == 1)

        # Process each (cluster, cause) for balance constraints and objective
        coeff_log_total = xr.DataArray(
            log_total_vals,
            coords={"log_total_step": log_total_steps},
            dims=["log_total_step"],
        )

        for (cluster, cause), label in zip(cluster_cause_pairs, cluster_cause_labels):
            data = cluster_cause_data[(cluster, cause)]
            lambda_total = lambda_total_group.sel(cluster_cause=label)

            total_expr = log_rr_totals_dict[(cluster, cause)]
            cause_bp = data["cause_bp"]

            log_interp = m.linexpr((coeff_log_total, lambda_total)).sum(
                "log_total_step"
            )
            coeff_rr = xr.DataArray(
                cause_bp["rr_total"].values,
                coords={"log_total_step": log_total_steps},
                dims=["log_total_step"],
            )
            rr_interp = m.linexpr((coeff_rr, lambda_total)).sum("log_total_step")
            m.add_constraints(
                log_interp == total_expr,
                name=f"health_total_balance_c{cluster}_cause{cause}",
            )

            store_name = health_stores.loc[(cluster, cause), "name"]

            if data["yll_base"] <= 0:
                logger.warning(
                    "Health store has non-positive yll_base (cluster=%d, cause=%s); constraint will be non-binding",
                    cluster,
                    cause,
                )

            # Health cost is zero at TMREL (where RR = RR_ref) and increases with
            # deviation from optimal intake. Since TMREL is the theoretical minimum
            # risk level, RR >= RR_ref always, so store levels are non-negative.
            yll_expr_myll = (
                (rr_interp - data["rr_ref"])
                * (data["yll_base"] / data["rr_ref"])
                * constants.YLL_TO_MILLION_YLL
            )

            m.add_constraints(
                store_e.sel(name=store_name) >= yll_expr_myll,
                name=f"health_store_level_c{cluster}_cause{cause}",
            )
            constraints_added += 1

    logger.info("Added %d health store level constraints", constraints_added)


if __name__ == "__main__":
    # Configure logging to write to Snakemake log file
    logger = setup_script_logging(log_file=snakemake.log[0] if snakemake.log else None)

    # Apply scenario config overrides based on wildcard
    apply_scenario_config(snakemake.config, snakemake.wildcards.scenario)

    n = pypsa.Network(snakemake.input.network)

    # Add GHG pricing to the objective if enabled
    if snakemake.config["emissions"]["ghg_pricing_enabled"]:
        ghg_price = float(snakemake.params.ghg_price)
        add_ghg_pricing_to_objective(n, ghg_price)

    # Add food-group incentives to the objective if enabled
    if snakemake.config["food_group_incentives"]["enabled"]:
        incentives_paths = list(snakemake.input.food_group_incentives)
        add_food_group_incentives_to_objective(n, incentives_paths)

    # Create the linopy model
    logger.info("Creating linopy model...")
    n.optimize.create_model()
    logger.info("Linopy model created.")

    solver_name = snakemake.params.solver
    solver_threads = snakemake.params.solver_threads
    solver_options = _apply_solver_threads_option(
        dict(snakemake.params.solver_options),
        solver_name,
        solver_threads,
    )
    io_api = snakemake.params.io_api
    netcdf_compression = snakemake.params.netcdf_compression

    # Configure Gurobi to write detailed logs to the same file
    if solver_name.lower() == "gurobi" and snakemake.log:
        if "LogFile" not in solver_options:
            solver_options["LogFile"] = snakemake.log[0]
        if "LogToConsole" not in solver_options:
            solver_options["LogToConsole"] = 1  # Also print to console

    # Add macronutrient intake bounds
    population_df = pd.read_csv(snakemake.input.population)
    population_df["iso3"] = population_df["iso3"].astype(str).str.upper()
    population_map = (
        population_df.set_index("iso3")["population"].astype(float).to_dict()
    )

    # Food group baseline equals (optional)
    per_country_equal: dict[str, dict[str, float]] | None = None
    equal_source = snakemake.config["food_groups"]["equal_by_country_source"]
    if bool(snakemake.params.enforce_baseline) and equal_source:
        raise ValueError(
            "Cannot combine enforce_gdd_baseline with food_groups.equal_by_country_source"
        )
    if bool(snakemake.params.enforce_baseline):
        baseline_df = pd.read_csv(snakemake.input.baseline_diet)
        per_country_equal = _build_food_group_equals_from_baseline(
            baseline_df,
            list(population_map.keys()),
            pd.read_csv(snakemake.input.food_groups)["group"].unique(),
            baseline_age=str(snakemake.params.diet["baseline_age"]),
            reference_year=int(snakemake.params.diet["baseline_reference_year"]),
        )
    elif equal_source:
        equal_df = pd.read_csv(snakemake.input.food_group_equal)
        required = {"group", "country", "consumption_g_per_day"}
        missing = required - set(equal_df.columns)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(
                f"Missing required columns in food group equality file: {missing_text}"
            )
        equal_df["country"] = equal_df["country"].astype(str).str.upper()
        per_country_equal = {}
        all_countries = set(population_map.keys())
        for group, group_df in equal_df.groupby("group"):
            values = dict.fromkeys(all_countries, 0.0)
            for _, row in group_df.iterrows():
                country = str(row["country"]).upper()
                if country not in values:
                    logger.warning(
                        "Unknown country '%s' in food group equality file", country
                    )
                    continue
                values[country] = float(row["consumption_g_per_day"])
            missing_countries = sorted(all_countries - set(group_df["country"]))
            if missing_countries:
                preview = ", ".join(missing_countries[:5])
                logger.warning(
                    "Food group '%s' missing %d countries in equality file; "
                    "setting them to 0 (examples: %s)",
                    group,
                    len(missing_countries),
                    preview,
                )
            per_country_equal[str(group)] = values

    add_macronutrient_constraints(n, snakemake.params.macronutrients, population_map)
    add_food_group_constraints(
        n,
        snakemake.params.food_group_constraints,
        population_map,
        per_country_equal,
    )

    # Add residue feed limit constraints
    max_feed_fraction = float(snakemake.config["residues"]["max_feed_fraction"])
    max_feed_fraction_by_country = build_residue_feed_fraction_by_country(
        snakemake.config, snakemake.input.m49
    )
    add_residue_feed_constraints(n, max_feed_fraction, max_feed_fraction_by_country)

    # Add animal production constraints in validation mode
    use_actual_production = bool(
        snakemake.config["validation"]["use_actual_production"]
    )
    if use_actual_production:
        fao_animal_production = pd.read_csv(snakemake.input.animal_production)
        food_groups_df = pd.read_csv(snakemake.input.food_groups)
        food_to_group = food_groups_df.set_index("food")["group"].to_dict()
        food_loss_waste = pd.read_csv(snakemake.input.food_loss_waste)
        add_animal_production_constraints(
            n, fao_animal_production, food_to_group, food_loss_waste
        )

    # Add production stability constraints
    stability_cfg = snakemake.params.production_stability
    if stability_cfg["enabled"]:
        # Load food_to_group if not already loaded
        if "food_to_group" not in dir():
            food_groups_df = pd.read_csv(snakemake.input.food_groups)
            food_to_group = food_groups_df.set_index("food")["group"].to_dict()

        crop_baseline = None
        crop_to_fao_item: dict[str, str] = {}
        animal_baseline = None
        food_loss_waste_df = pd.DataFrame()

        if stability_cfg["crops"]["enabled"]:
            crop_baseline = pd.read_csv(snakemake.input.crop_production_baseline)
            # Load FAO item mapping to aggregate crops sharing an FAO item
            fao_map_df = pd.read_csv(snakemake.input.faostat_item_map)
            crop_to_fao_item = dict(
                zip(
                    fao_map_df["crop"].astype(str),
                    fao_map_df["faostat_item"].astype(str),
                )
            )

        if stability_cfg["animals"]["enabled"]:
            animal_baseline = pd.read_csv(snakemake.input.animal_production_baseline)
            food_loss_waste_df = pd.read_csv(snakemake.input.food_loss_waste)

        add_production_stability_constraints(
            n,
            crop_baseline,
            crop_to_fao_item,
            animal_baseline,
            stability_cfg,
            food_to_group,
            food_loss_waste_df,
        )

    # Add health impacts / store levels if enabled or baseline intake is enforced
    health_enabled = bool(snakemake.config["health"]["enabled"])
    enforce_baseline = bool(snakemake.params.enforce_baseline)
    if health_enabled and enforce_baseline:
        raise ValueError(
            "health.enabled and validation.enforce_gdd_baseline cannot both be true; "
            "disable one of them in the config."
        )
    if health_enabled or enforce_baseline:
        add_health_objective(
            n,
            snakemake.input.health_risk_breakpoints,
            snakemake.input.health_cluster_cause,
            snakemake.input.health_cause_log,
            snakemake.input.health_cluster_summary,
            snakemake.input.health_clusters,
            snakemake.input.population,
            snakemake.params.health_risk_factors,
            snakemake.params.health_risk_cause_map,
            solver_name,
            enforce_baseline,
        )

    status, condition = n.model.solve(
        solver_name=solver_name,
        io_api=io_api,
        calculate_fixed_duals=snakemake.params.calculate_fixed_duals,
        **solver_options,
    )
    result = (status, condition)

    # Temporary debug export of the raw solved linopy model
    # linopy_debug_path = Path(snakemake.output.network).with_name("linopy_model.nc")
    # linopy_debug_path.parent.mkdir(parents=True, exist_ok=True)
    # n.model.to_netcdf(linopy_debug_path)
    # logger.info("Wrote linopy model snapshot to %s", linopy_debug_path)

    if status == "ok":
        aux_names = HEALTH_AUX_MAP.pop(id(n.model), set())
        variables_container = n.model.variables
        removed = {}
        for name in aux_names:
            if name in variables_container.data:
                removed[name] = variables_container.data.pop(name)

        try:
            n.optimize.assign_solution()
            n.optimize.assign_duals(False)
            n.optimize.post_processing()
        finally:
            if removed:
                variables_container.data.update(removed)

        n.export_to_netcdf(
            snakemake.output.network,
            compression=netcdf_compression,
        )
    elif condition in {"infeasible", "infeasible_or_unbounded"}:
        logger.error("Model is infeasible or unbounded!")
        if solver_name.lower() == "gurobi":
            try:
                logger.error("Computing IIS (Irreducible Inconsistent Subsystem)...")

                # Get infeasible constraint labels
                infeasible_labels = n.model.compute_infeasibilities()

                if not infeasible_labels:
                    logger.error("No infeasible constraints found in IIS")
                else:
                    logger.error(
                        "Found %d infeasible constraints:", len(infeasible_labels)
                    )

                    constraint_details = []
                    for label in infeasible_labels:
                        try:
                            detail = print_single_constraint(n.model, label)
                            constraint_details.append(detail)
                        except Exception as e:
                            constraint_details.append(
                                f"Label {label}: <error formatting: {e}>"
                            )

                    # Log all infeasible constraints
                    iis_output = "\n".join(constraint_details)
                    logger.error("IIS constraints:\n%s", iis_output)

            except Exception as exc:
                logger.error("Could not compute infeasibilities: %s", exc)
        else:
            logger.error("Infeasibility diagnosis only available with Gurobi solver")
    else:
        logger.error("Optimization unsuccessful: %s", result)
