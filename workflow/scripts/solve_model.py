# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

from collections import defaultdict
import math
from pathlib import Path
import sys

from linopy.constraints import print_single_constraint
from logging_config import setup_script_logging
import numpy as np
import pandas as pd
import pypsa
import xarray as xr

# Ensure project root on sys.path for Snakemake temp copies
_script_path = Path(__file__).resolve()
try:
    _project_root = _script_path.parents[2]
except IndexError:
    _project_root = _script_path.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from workflow.scripts.build_model import constants  # noqa: E402
from workflow.scripts.build_model.nutrition import (  # noqa: E402
    _build_food_group_equals_from_baseline,
)
from workflow.scripts.build_model.utils import (  # noqa: E402
    _per_capita_mass_to_mt_per_year,
)
from workflow.scripts.snakemake_utils import apply_objective_config  # noqa: E402

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
                m.add_constraints(lhs == rhs, name=constr_name)
                n.global_constraints.add(
                    f"{constr_name}_" + nutrient_stores.index,
                    sense="==",
                    constant=rhs.values,
                    type="nutrition",
                )
                break

            if operator == ">=":
                m.add_constraints(lhs >= rhs, name=constr_name)
            else:
                m.add_constraints(lhs <= rhs, name=constr_name)

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

    if not food_group_cfg:
        return

    per_country_equal = per_country_equal or {}

    m = n.model
    store_e = m.variables["Store-e"].sel(snapshot="now")
    stores_df = n.stores.static

    for group, bounds in food_group_cfg.items():
        if not bounds:
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
                m.add_constraints(lhs == rhs, name=constr_name)
                n.global_constraints.add(
                    f"{constr_name}_" + group_stores.index,
                    sense="==",
                    constant=rhs.values,
                    type="nutrition",
                )
                break

            if operator == ">=":
                m.add_constraints(lhs >= rhs, name=constr_name)
            else:
                m.add_constraints(lhs <= rhs, name=constr_name)

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
    n.stores.static.at["ghg", "marginal_storage_cost"] = ghg_price_bnusd_per_mt


def add_residue_feed_constraints(n: pypsa.Network, max_feed_fraction: float) -> None:
    """Add constraints limiting residue removal for animal feed.

    Constrains the fraction of residues that can be removed for feed vs.
    incorporated into soil. The constraint is formulated as:
        feed_use ≤ (max_feed_fraction / (1 - max_feed_fraction)) x incorporation

    This ensures that if a total amount R of residue is generated:
        R = feed_use + incorporation
        feed_use ≤ max_feed_fraction x R

    Parameters
    ----------
    n : pypsa.Network
        The network containing the model.
    max_feed_fraction : float
        Maximum fraction of residues that can be used for feed (e.g., 0.30 for 30%).
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

    # Add constraints for each residue bus
    ratio = max_feed_fraction / (1.0 - max_feed_fraction)

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

    # 6. Add constraints
    m.add_constraints(
        feed_sum <= ratio * incorp_flow,
        name="residue_feed_limit",
    )

    logger.info(
        "Added %d residue feed limit constraints (max %.0f%% for feed)",
        len(common_buses),
        max_feed_fraction * 100,
    )


def add_animal_production_constraints(
    n: pypsa.Network, fao_production: pd.DataFrame
) -> None:
    """Add constraints to fix animal production at FAO levels per country.

    For each (country, product) combination in the FAO data, adds a constraint
    that total production from all feed categories equals the FAO target.

    Parameters
    ----------
    n : pypsa.Network
        The network containing the model.
    fao_production : pd.DataFrame
        FAO production data with columns: country, product, production_mt.
    """
    if fao_production.empty:
        logger.warning(
            "No FAO animal production data available; skipping production constraints"
        )
        return

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

    m.add_constraints(lhs == rhs, name="animal_production_target")

    logger.info(
        "Added %d country-level animal production constraints",
        len(common_index),
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
) -> None:
    """Link SOS2-based health costs to explicit PyPSA stores.

    The function builds the same intake→relative-risk logic as before, but
    instead of writing directly to the linopy objective it constrains the
    level of per-cluster, per-cause YLL stores. The monetary contribution is
    then handled by the store ``capital_cost`` configured during network
    construction.
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

    # Restrict to configured risk-cause pairs to avoid silent zeros
    allowed_pairs = {(r, c) for r, causes in risk_cause_map.items() for c in causes}
    rb_pairs = set(zip(risk_breakpoints["risk_factor"], risk_breakpoints["cause"]))
    missing_pairs = sorted(allowed_pairs - rb_pairs)
    if missing_pairs:
        text = ", ".join([f"{r}:{c}" for r, c in missing_pairs])
        raise ValueError(f"Risk breakpoints missing required pairs: {text}")

    p = m.variables["Link-p"].sel(snapshot="now")

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
        risk_data[risk] = {"intakes": intakes, "log_rr": pivot}

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

        coords_key = tuple(risk_table["intakes"].values)
        intake_groups[coords_key].append((cluster, risk))

    log_rr_totals_dict = {}

    for coords_key, group_pairs in intake_groups.items():
        coords = pd.Index(coords_key, name="intake")

        # Identify group labels
        cluster_risk_labels = [f"c{cluster}_r{risk}" for cluster, risk in group_pairs]
        cluster_risk_index = pd.Index(cluster_risk_labels, name="cluster_risk")

        # Create lambdas (vectorized)
        lambdas_group = m.add_variables(
            lower=0,
            upper=1,
            coords=[cluster_risk_index, coords],
            name=f"health_lambda_group_{hash(coords_key)}",
        )

        # Register all variables
        _register_health_variable(m, lambdas_group.name)

        # Single SOS2 constraint call for entire group
        aux_names = _add_sos2_with_fallback(
            m, lambdas_group, sos_dim="intake", solver_name=solver_name
        )
        for aux_name in aux_names:
            _register_health_variable(m, aux_name)

        # Vectorized convexity constraints
        m.add_constraints(lambdas_group.sum("intake") == 1)

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
            coeff_intake = xr.DataArray(
                coords.values, coords={"intake": coords.values}, dims="intake"
            )
            intake_expr = (lambdas_group * coeff_intake).sum("intake")

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
            names=["cluster_risk", "intake"],
        ).fillna(0.0)

        # Convert to DataArray: (cluster_risk, intake, cause)
        # Use stack() to flatten columns (cause) into index
        s_log = combined_log_rr.stack()
        s_log.index.names = ["cluster_risk", "intake", "cause"]
        da_log = xr.DataArray.from_series(s_log).fillna(0.0)

        # Calculate contribution: sum(lambda * log_rr) over intake
        # lambdas_group: (cluster_risk, intake)
        # da_log: (cluster_risk, intake, cause)
        # Result: (cause, cluster_risk) of LinearExpressions
        contrib = (lambdas_group * da_log).sum("intake")

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
    log_total_groups: dict[tuple[float, ...], list[tuple[int, str]]] = defaultdict(list)
    cluster_cause_data: dict[tuple[int, str], dict] = {}

    for (cluster, cause), row in cluster_cause_metadata.iterrows():
        cluster = int(cluster)
        cause = str(cause)
        yll_base = float(row.get("yll_base", 0.0))
        if yll_base == 0 or not math.isfinite(yll_base):
            continue

        cause_bp = cause_breakpoint_data[cause]
        coords_key = tuple(cause_bp["log_rr_total"].values)
        if len(coords_key) == 1:
            raise ValueError(
                "Need at least two breakpoints for piecewise linear approximation"
            )

        log_total_groups[coords_key].append((cluster, cause))

        # Store data for later use
        log_rr_total_ref = float(row.get("log_rr_total_ref", 0.0))
        cluster_cause_data[(cluster, cause)] = {
            "yll_base": yll_base,
            "log_rr_total_ref": log_rr_total_ref,
            "rr_ref": math.exp(log_rr_total_ref),
            "cause_bp": cause_bp,
        }

    store_e = m.variables["Store-e"].sel(snapshot="now")
    health_stores = (
        n.stores.static[
            n.stores.static["carrier"].notna()
            & n.stores.static["carrier"].str.startswith("yll_")
        ]
        .reset_index()
        .set_index(["health_cluster", "cause"])
    )

    # Process each group with vectorized operations
    for coords_key, cluster_cause_pairs in log_total_groups.items():
        coords = pd.Index(coords_key, name="log_total")

        # Create flattened index for this group
        cluster_cause_labels = [
            f"c{cluster}_cause{cause}" for cluster, cause in cluster_cause_pairs
        ]
        cluster_cause_index = pd.Index(cluster_cause_labels, name="cluster_cause")

        # Single vectorized variable creation
        lambda_total_group = m.add_variables(
            lower=0,
            upper=1,
            coords=[cluster_cause_index, coords],
            name=f"health_lambda_total_group_{hash(coords_key)}",
        )

        # Register all variables
        _register_health_variable(m, lambda_total_group.name)

        # Single SOS2 constraint call for entire group
        aux_names = _add_sos2_with_fallback(
            m, lambda_total_group, sos_dim="log_total", solver_name=solver_name
        )
        for aux_name in aux_names:
            _register_health_variable(m, aux_name)
        # Vectorized convexity constraints
        m.add_constraints(lambda_total_group.sum("log_total") == 1)

        # Process each (cluster, cause) for balance constraints and objective
        coeff_log_total = xr.DataArray(
            coords.values,
            coords={"log_total": coords.values},
            dims=["log_total"],
        )

        for (cluster, cause), label in zip(cluster_cause_pairs, cluster_cause_labels):
            data = cluster_cause_data[(cluster, cause)]
            lambda_total = lambda_total_group.sel(cluster_cause=label)

            total_expr = log_rr_totals_dict[(cluster, cause)]
            cause_bp = data["cause_bp"]

            log_interp = m.linexpr((coeff_log_total, lambda_total)).sum("log_total")
            coeff_rr = xr.DataArray(
                cause_bp["rr_total"].values,
                coords={"log_total": coords.values},
                dims=["log_total"],
            )
            rr_interp = m.linexpr((coeff_rr, lambda_total)).sum("log_total")
            m.add_constraints(
                log_interp == total_expr,
                name=f"health_total_balance_c{cluster}_cause{cause}",
            )

            store_name = health_stores.loc[(cluster, cause), "name"]

            yll_expr_myll = (
                rr_interp
                * (data["yll_base"] / data["rr_ref"])
                * constants.YLL_TO_MILLION_YLL
            )

            m.add_constraints(
                store_e.sel(name=store_name) == yll_expr_myll,
                name=f"health_store_level_c{cluster}_cause{cause}",
            )

    logger.info("Linked health stores to risk model (no direct objective edits)")


if __name__ == "__main__":
    # Configure logging to write to Snakemake log file
    logger = setup_script_logging(log_file=snakemake.log[0] if snakemake.log else None)

    # Apply objective config overrides based on wildcard
    apply_objective_config(snakemake.config, snakemake.wildcards.objective)

    n = pypsa.Network(snakemake.input.network)

    # Add GHG pricing to the objective if enabled
    if snakemake.config["emissions"]["ghg_pricing_enabled"]:
        ghg_price = float(snakemake.params.ghg_price)
        add_ghg_pricing_to_objective(n, ghg_price)

    # Create the linopy model
    logger.info("Creating linopy model...")
    n.optimize.create_model()
    logger.info("Linopy model created.")

    solver_name = snakemake.params.solver
    solver_options = snakemake.params.solver_options
    io_api = snakemake.params.io_api
    netcdf_compression = snakemake.params.netcdf_compression

    # Configure Gurobi to write detailed logs to the same file
    if solver_name.lower() == "gurobi" and snakemake.log:
        solver_options = dict(solver_options)  # Make a copy to avoid modifying config
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
    if bool(snakemake.params.enforce_baseline):
        baseline_df = pd.read_csv(snakemake.input.baseline_diet)
        per_country_equal = _build_food_group_equals_from_baseline(
            baseline_df,
            list(population_map.keys()),
            pd.read_csv(snakemake.input.food_groups)["group"].unique(),
            baseline_age=str(snakemake.params.diet["baseline_age"]),
            reference_year=int(snakemake.params.diet["baseline_reference_year"]),
        )

    add_macronutrient_constraints(n, snakemake.params.macronutrients, population_map)
    add_food_group_constraints(
        n,
        snakemake.params.food_group_constraints,
        population_map,
        per_country_equal,
    )

    # Add residue feed limit constraints
    max_feed_fraction = float(snakemake.config["residues"]["max_feed_fraction"])
    add_residue_feed_constraints(n, max_feed_fraction)

    # Add animal production constraints in validation mode
    use_actual_production = bool(
        snakemake.config["validation"]["use_actual_production"]
    )
    if use_actual_production:
        fao_animal_production = pd.read_csv(snakemake.input.animal_production)
        add_animal_production_constraints(n, fao_animal_production)

    # Add health impacts to the objective if enabled
    health_enabled = bool(snakemake.config["health"]["enabled"])
    if health_enabled:
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
        )

    # Temporary debug export of the raw linopy model for coefficient inspection.
    # linopy_debug_path = Path(snakemake.output.network).with_name("linopy_model.nc")
    # linopy_debug_path.parent.mkdir(parents=True, exist_ok=True)
    # n.model.to_netcdf(linopy_debug_path)
    # logger.info("Wrote linopy model snapshot to %s", linopy_debug_path)

    status, condition = n.model.solve(
        solver_name=solver_name,
        io_api=io_api,
        **solver_options,
    )
    result = (status, condition)

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
