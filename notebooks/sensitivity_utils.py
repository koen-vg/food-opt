# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared utilities for sensitivity analysis notebooks.

This module contains common functions used by yll_sensitivity, ghg_sensitivity,
and combined_sensitivity notebooks.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
from scipy import sparse
from scipy.sparse.linalg import spsolve

# Constants for unit conversion
GRAMS_PER_MEGATONNE = 1e12
DAYS_PER_YEAR = 365
KCAL_TO_PJ = 4.184e-12
PJ_TO_KCAL = 1.0 / KCAL_TO_PJ
USD_TO_BNUSD = 1e-9

# GWP values (AR5 100-year)
CH4_GWP = 28.0
N2O_GWP = 265.0

# Pretty names for food groups (including aggregated groups)
PRETTY_NAMES = {
    "grain": "Refined grains",
    "whole_grains": "Whole grains",
    "dairy": "Dairy",
    "eggs": "Eggs",
    "fruits": "Fruits",
    "legumes": "Legumes",
    "nuts_seeds": "Nuts & seeds",
    "oil": "Oil",
    "poultry": "Poultry",
    "red_meat": "Red meat",
    "starchy_vegetable": "Starchy veg.",
    "sugar": "Sugar",
    "vegetables": "Vegetables",
    "fruits_vegetables": "Fruits & veg.",
    "eggs_poultry": "Eggs & poultry",
}

# Health-specific labels: "Diet low in X" for protective, "Diet high in X" for harmful
PRETTY_NAMES_HEALTH = {
    "fruits": "Diet low in\nfruits",
    "vegetables": "Diet low in\nvegetables",
    "whole_grains": "Diet low in\nwhole grains",
    "legumes": "Diet low in\nlegumes",
    "nuts_seeds": "Diet low in\nnuts & seeds",
    "red_meat": "Diet high in\nred meat",
    "fruits_vegetables": "Diet low in\nfruits & veg.",
}

# Pretty names for objective categories
PRETTY_NAMES_OBJ = {
    "Crop production": "Crop production",
    "Trade": "Trade",
    "Health burden": "Health burden",
    "GHG cost": "GHG cost",
    "GHG cost (positive)": "GHG cost",
    "GHG cost (negative)": "GHG cost",
    "Fertilizer (synthetic)": "Fertilizer",
    "Consumer values": "Consumer values",
    "Biomass exports": "Biomass exports",
}


# -----------------------------------------------------------------------------
# Data loading utilities
# -----------------------------------------------------------------------------


def is_cache_valid(cache_path: Path, network_files: list[Path]) -> bool:
    """Check if cache file is newer than all network files."""
    if not cache_path.exists():
        return False
    cache_mtime = cache_path.stat().st_mtime
    return all(nf.stat().st_mtime <= cache_mtime for nf in network_files)


def load_population_from_network(network_path: Path) -> float:
    """Load total global population from network metadata."""
    n = pypsa.Network(network_path)
    pop_meta = n.meta.get("population")
    if pop_meta is None:
        raise KeyError(
            "Population data not found in network metadata. "
            "Ensure the model was built with population embedding enabled."
        )
    return sum(pop_meta["country"].values())


def extract_param_value(scenario_name: str, prefix: str) -> float | None:
    """Extract a numeric parameter value from a scenario name.

    Args:
        scenario_name: e.g. 'yll_5000' or 'ghg_100'
        prefix: e.g. 'yll' or 'ghg'

    Returns:
        The numeric value, or None if pattern doesn't match
    """
    if scenario_name == "baseline":
        return None
    match = re.match(rf"{prefix}_(\d+)", scenario_name)
    if match:
        return float(match.group(1))
    return None


def extract_combined_param_value(scenario_name: str) -> tuple[float, float] | None:
    """Extract GHG price and YLL value from combined scenario name.

    Args:
        scenario_name: e.g. 'ghg_yll_100' (ghg=100, yll=10000)

    Returns:
        Tuple of (ghg_price, yll_value), or None if pattern doesn't match
    """
    if scenario_name == "baseline":
        return None
    match = re.match(r"ghg_yll_(\d+)", scenario_name)
    if match:
        ghg_price = float(match.group(1))
        yll_value = ghg_price * 100  # Fixed ratio
        return (ghg_price, yll_value)
    return None


def load_food_to_group(project_root: Path) -> dict[str, str]:
    """Load food to group mapping from CSV."""
    food_groups_df = pd.read_csv(project_root / "data" / "food_groups.csv")
    return dict(zip(food_groups_df["food"], food_groups_df["group"]))


# -----------------------------------------------------------------------------
# Worker functions for parallel data extraction
# -----------------------------------------------------------------------------


def _extract_consumption_calories_worker(args):
    """Worker function for parallel extraction of caloric consumption.

    Takes all necessary data as args since module-level constants
    aren't available in subprocess.
    """
    network_path, population, food_to_group, pj_to_kcal, days_per_year = args

    n = pypsa.Network(network_path)
    links = n.links

    snapshot = "now" if "now" in n.snapshots else n.snapshots[-1]

    legs = []
    for col in links.columns:
        if col.startswith("bus") and col[3:].isdigit():
            leg = int(col[3:])
            if leg > 0:
                legs.append(leg)
    legs = sorted(legs)

    time_series_lookup = {}
    for leg in legs:
        attr = f"p{leg}"
        series = getattr(n.links_t, attr, None)
        if series is not None and snapshot in series.index:
            time_series_lookup[leg] = series.loc[snapshot]

    totals = {}
    for link_name in links.index:
        if not link_name.startswith("consume_"):
            continue

        food = str(links.at[link_name, "food"])
        group_name = food_to_group.get(food)
        if group_name is None:
            continue

        kcal_leg = None
        for leg in legs:
            column = f"bus{leg}"
            bus_value = links.at[link_name, column]
            if pd.notna(bus_value) and str(bus_value).startswith("cal_"):
                kcal_leg = leg
                break

        if kcal_leg is None:
            continue

        series = time_series_lookup.get(kcal_leg)
        if series is None:
            continue

        value_pj = abs(float(series.get(link_name, 0.0)))
        if value_pj > 0.0:
            totals[group_name] = totals.get(group_name, 0.0) + value_pj

    calories_series = pd.Series(totals, dtype=float)
    calories_series = calories_series * pj_to_kcal / (population * days_per_year)

    return calories_series


def _objective_category(n: pypsa.Network, component: str, **_) -> pd.Series:
    """Group assets into high-level categories for system cost aggregation."""
    static = n.components[component].static
    if static.empty:
        return pd.Series(dtype="object")

    index = static.index

    if component == "Generator":
        carriers = static.get("carrier", pd.Series(dtype=str))
        categories = []
        for name in index:
            name_str = str(name)
            carrier = str(carriers.get(name, "")) if not carriers.empty else ""
            if name_str.startswith("biomass_for_energy_"):
                categories.append("Biomass exports")
            elif "slack" in name_str or "slack" in carrier:
                categories.append("Slack penalties")
            elif carrier == "fertilizer":
                categories.append("Fertilizer (synthetic)")
            else:
                categories.append("Other generators")
        return pd.Series(categories, index=index, name="category")

    if component == "Link":
        mapping = {
            "produce": "Crop production",
            "trade": "Trade",
            "convert": "Processing",
            "consume": "Consumption",
        }
        categories = []
        for name in index:
            name_str = str(name)
            if name_str.startswith(("crop_to_biomass_", "byproduct_to_biomass_")):
                categories.append("Biomass routing")
                continue
            categories.append(mapping.get(name_str.split("_", 1)[0], "Other links"))
        return pd.Series(categories, index=index, name="category")

    if component == "Store":
        carriers = static["carrier"].astype(str)
        categories = []
        for name, carrier in zip(index, carriers):
            if carrier == "ghg" or str(name) == "ghg":
                categories.append("GHG storage")
            elif carrier.startswith("yll_"):
                categories.append("Health burden")
            elif carrier.startswith("group_"):
                categories.append("Consumer values")
            else:
                categories.append("Other stores")
        return pd.Series(categories, index=index, name="category")

    return pd.Series(component, index=index, name="category")


def _extract_objective_breakdown_worker(args):
    """Worker function for parallel objective extraction."""
    network_path, constant_health_value, constant_ghg_price, usd_to_bnusd = args

    n = pypsa.Network(network_path)

    capex = n.statistics.capex(groupby=_objective_category)
    opex = n.statistics.opex(groupby=_objective_category)

    def _to_series(df_or_series):
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

    if "Health burden" in total.index:
        total = total.drop("Health burden")
    if "GHG storage" in total.index:
        total = total.drop("GHG storage")

    snapshot = n.snapshots[-1] if len(n.snapshots) > 0 else None

    if snapshot is not None and snapshot in n.stores_t.e.index:
        health_stores = n.stores[n.stores.carrier.str.startswith("yll_")]
        if not health_stores.empty:
            health_levels = n.stores_t.e.loc[snapshot, health_stores.index]
            total_myll = health_levels.sum()
            health_cost_bnusd = constant_health_value * total_myll * 1e6 * usd_to_bnusd
            total["Health burden"] = health_cost_bnusd

    if snapshot is not None and snapshot in n.stores_t.e.index:
        ghg_stores = n.stores[n.stores.carrier == "ghg"]
        if not ghg_stores.empty:
            ghg_levels = n.stores_t.e.loc[snapshot, ghg_stores.index]
            total_mtco2eq = ghg_levels.sum()
            ghg_cost_bnusd = constant_ghg_price * total_mtco2eq * 1e6 * usd_to_bnusd
            total["GHG cost"] = ghg_cost_bnusd

    total = total[total.abs() > 1e-6]
    return total.sort_values(ascending=False)


# -----------------------------------------------------------------------------
# GHG emission attribution functions
# -----------------------------------------------------------------------------


def build_ghg_links_dataframe(n, p0, ch4_gwp, n2o_gwp):
    """Build DataFrame of links with flows and GHG emissions."""
    gwp = {"co2": 1.0, "ch4": ch4_gwp * 1e-6, "n2o": n2o_gwp * 1e-6}

    links = n.links.copy()
    links["link_name"] = links.index
    links["flow"] = p0.reindex(links.index).fillna(0.0)
    links = links[links["flow"] > 1e-12].copy()

    if links.empty:
        return pd.DataFrame()

    links["efficiency"] = links["efficiency"].fillna(1.0)
    links["emissions_co2e"] = 0.0

    for bus_col, eff_col in [
        ("bus2", "efficiency2"),
        ("bus3", "efficiency3"),
        ("bus4", "efficiency4"),
    ]:
        if bus_col not in links.columns:
            continue
        emission_bus = links[bus_col].fillna("")
        eff = links[eff_col].fillna(0.0) if eff_col in links.columns else 0.0
        for gas, gwp_factor in gwp.items():
            mask = (emission_bus == gas) & (eff > 0)
            links.loc[mask, "emissions_co2e"] += eff[mask] * gwp_factor

    return links[["link_name", "bus0", "bus1", "flow", "efficiency", "emissions_co2e"]]


def solve_emission_intensities(links_df):
    """Solve for emission intensity at each bus using sparse matrix."""
    all_buses = pd.concat([links_df["bus0"], links_df["bus1"]]).unique()
    bus_to_idx = {bus: i for i, bus in enumerate(all_buses)}
    n_buses = len(all_buses)

    links_df = links_df.copy()
    links_df["idx0"] = links_df["bus0"].map(bus_to_idx)
    links_df["idx1"] = links_df["bus1"].map(bus_to_idx)

    links_df["outflow"] = links_df["flow"] * links_df["efficiency"]
    total_outflow = links_df.groupby("idx1")["outflow"].transform("sum")
    links_df["weight"] = links_df["flow"] / total_outflow
    links_df["emission_contrib"] = (
        links_df["flow"] * links_df["emissions_co2e"] / total_outflow
    )

    row = links_df["idx1"].values
    col = links_df["idx0"].values
    data = links_df["weight"].values

    adj_matrix = sparse.coo_matrix((data, (row, col)), shape=(n_buses, n_buses)).tocsr()

    e = np.zeros(n_buses)
    np.add.at(e, links_df["idx1"].values, links_df["emission_contrib"].values)

    identity = sparse.eye(n_buses, format="csr")
    system_matrix = identity - adj_matrix

    rho = spsolve(system_matrix, e)

    idx_to_bus = {i: bus for bus, i in bus_to_idx.items()}
    return {idx_to_bus[i]: float(rho[i]) for i in range(n_buses)}


def _extract_ghg_by_food_group_worker(args):
    """Worker function to extract GHG emissions by food group."""
    network_path, food_to_group, ch4_gwp, n2o_gwp = args

    n = pypsa.Network(network_path)
    snapshot = n.snapshots[-1]
    p0 = (
        n.links_t.p0.loc[snapshot]
        if snapshot in n.links_t.p0.index
        else pd.Series(dtype=float)
    )

    links_df = build_ghg_links_dataframe(n, p0, ch4_gwp, n2o_gwp)

    if links_df.empty:
        return pd.Series(dtype=float)

    bus_intensities = solve_emission_intensities(links_df)

    consume_mask = links_df["link_name"].str.startswith("consume_")
    consume_df = links_df.loc[consume_mask, ["link_name", "bus0", "flow"]].copy()
    consume_df["intensity"] = consume_df["bus0"].map(bus_intensities).fillna(0.0)
    consume_df["ghg_mtco2e"] = consume_df["flow"] * consume_df["intensity"]

    totals = {}
    for _, row in consume_df.iterrows():
        link_name = row["link_name"]
        parts = str(link_name).split("_")
        if len(parts) < 3:
            continue
        food = "_".join(parts[1:-1])
        group = food_to_group.get(food)
        if group:
            totals[group] = totals.get(group, 0.0) + row["ghg_mtco2e"]

    return pd.Series(totals, dtype=float)


# -----------------------------------------------------------------------------
# Data extraction orchestrators
# -----------------------------------------------------------------------------


def extract_consumption_data(
    scenarios: list[tuple[float, str, Path]],
    food_to_group: dict,
    cache_path: Path,
    param_name: str = "param_value",
    n_workers: int = 8,
) -> pd.DataFrame:
    """Extract caloric consumption data for all scenarios.

    Args:
        scenarios: List of (param_value, scenario_name, network_path) tuples
        food_to_group: Mapping from food to food group
        cache_path: Path to cache file
        param_name: Name for the parameter (used as index name)
        n_workers: Number of parallel workers

    Returns:
        DataFrame with param_value as index and food groups as columns
    """
    network_paths = [f for _, _, f in scenarios]

    if is_cache_valid(cache_path, network_paths):
        print(f"Loading consumption data from cache: {cache_path}")
        return pd.read_csv(cache_path, index_col=param_name)

    # Load population from first network's embedded metadata
    population = load_population_from_network(network_paths[0])
    print(f"Total population: {population:,.0f}")
    print(f"Extracting consumption data using {n_workers} workers...")

    worker_args = [
        (network_path, population, food_to_group, PJ_TO_KCAL, DAYS_PER_YEAR)
        for _, _, network_path in scenarios
    ]
    param_values = [pv for pv, _, _ in scenarios]

    consumption_data = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_extract_consumption_calories_worker, args): pv
            for args, pv in zip(worker_args, param_values)
        }

        for future in as_completed(futures):
            param_value = futures[future]
            consumption_data[param_value] = future.result()
            print(f"  Loaded {param_name}={int(param_value)}")

    df = pd.DataFrame(consumption_data).T.fillna(0)
    df.index.name = param_name
    df = df.sort_index()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path)
    print(f"Saved consumption data to cache: {cache_path}")

    return df


def extract_objective_data(
    scenarios: list[tuple[float, str, Path]],
    cache_path: Path,
    param_name: str = "param_value",
    constant_health_value: float = 10000,
    constant_ghg_price: float = 100,
    n_workers: int = 8,
) -> pd.DataFrame:
    """Extract objective breakdown data for all scenarios.

    Args:
        scenarios: List of (param_value, scenario_name, network_path) tuples
        cache_path: Path to cache file
        param_name: Name for the parameter (used as index name)
        constant_health_value: USD/YLL for health burden calculation
        constant_ghg_price: USD/tCO2eq for GHG cost calculation
        n_workers: Number of parallel workers

    Returns:
        DataFrame with param_value as index and cost categories as columns
    """
    network_paths = [f for _, _, f in scenarios]

    if is_cache_valid(cache_path, network_paths):
        print(f"Loading objective data from cache: {cache_path}")
        return pd.read_csv(cache_path, index_col=param_name)

    print(f"Extracting objective data using {n_workers} workers...")

    worker_args = [
        (network_path, constant_health_value, constant_ghg_price, USD_TO_BNUSD)
        for _, _, network_path in scenarios
    ]
    param_values = [pv for pv, _, _ in scenarios]

    objective_data = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_extract_objective_breakdown_worker, args): pv
            for args, pv in zip(worker_args, param_values)
        }

        for future in as_completed(futures):
            param_value = futures[future]
            objective_data[param_value] = future.result()
            print(f"  Loaded {param_name}={int(param_value)}")

    df = pd.DataFrame(objective_data).T.fillna(0)
    df.index.name = param_name
    df = df.sort_index()

    df.to_csv(cache_path)
    print(f"Saved objective data to cache: {cache_path}")

    return df


def extract_ghg_data(
    scenarios: list[tuple[float, str, Path]],
    food_to_group: dict,
    cache_path: Path,
    param_name: str = "param_value",
    n_workers: int = 8,
) -> pd.DataFrame:
    """Extract GHG emissions by food group for all scenarios.

    Args:
        scenarios: List of (param_value, scenario_name, network_path) tuples
        food_to_group: Mapping from food to food group
        cache_path: Path to cache file
        param_name: Name for the parameter (used as index name)
        n_workers: Number of parallel workers

    Returns:
        DataFrame with param_value as index and food groups as columns (in GtCO2eq)
    """
    network_paths = [f for _, _, f in scenarios]

    if is_cache_valid(cache_path, network_paths):
        print(f"Loading GHG data from cache: {cache_path}")
        return pd.read_csv(cache_path, index_col=param_name)

    print(f"Extracting GHG data using {n_workers} workers...")

    worker_args = [
        (network_path, food_to_group, CH4_GWP, N2O_GWP)
        for _, _, network_path in scenarios
    ]
    param_values = [pv for pv, _, _ in scenarios]

    ghg_data = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_extract_ghg_by_food_group_worker, args): pv
            for args, pv in zip(worker_args, param_values)
        }

        for future in as_completed(futures):
            param_value = futures[future]
            ghg_data[param_value] = future.result()
            print(f"  Loaded {param_name}={int(param_value)}")

    df = pd.DataFrame(ghg_data).T.fillna(0)
    df.index.name = param_name
    df = df.sort_index()

    # Convert MtCO2e to GtCO2e
    df = df / 1000

    df.to_csv(cache_path)
    print(f"Saved GHG data to cache: {cache_path}")

    return df


# -----------------------------------------------------------------------------
# Health cost attribution functions
# -----------------------------------------------------------------------------


def _load_health_tables(
    processing_dir: Path,
    network: pypsa.Network | None = None,
) -> tuple[dict, dict, pd.DataFrame, pd.DataFrame, dict]:
    """Load health data tables from processing directory.

    Args:
        processing_dir: Path to processing directory
        network: Optional network to get cluster population from embedded metadata

    Returns:
        Tuple of (cluster_lookup, cluster_population, risk_breakpoints,
                  cluster_cause_baseline, tmrel_g_per_day)
    """
    health_dir = processing_dir / "health"

    # Country to cluster mapping
    country_clusters = pd.read_csv(health_dir / "country_clusters.csv")
    cluster_lookup = dict(
        zip(country_clusters["country_iso3"], country_clusters["health_cluster"])
    )

    # Get cluster population from network metadata if available
    if network is not None:
        pop_meta = network.meta.get("population")
        if pop_meta is not None and "health_cluster" in pop_meta:
            # Convert string keys to int (JSON serialization)
            cluster_population = {
                int(k): float(v) for k, v in pop_meta["health_cluster"].items()
            }
        else:
            cluster_population = _load_cluster_population_fallback(
                health_dir, cluster_lookup
            )
    else:
        cluster_population = _load_cluster_population_fallback(
            health_dir, cluster_lookup
        )

    # Risk breakpoints for log(RR) lookup
    risk_breakpoints = pd.read_csv(health_dir / "risk_breakpoints.csv")

    # Baseline YLL and RR data per (cluster, cause)
    cluster_cause_baseline = pd.read_csv(health_dir / "cluster_cause_baseline.csv")

    # TMREL values per risk factor
    tmrel_df = pd.read_csv(health_dir / "derived_tmrel.csv")
    tmrel_g_per_day = dict(zip(tmrel_df["risk_factor"], tmrel_df["tmrel_g_per_day"]))

    return (
        cluster_lookup,
        cluster_population,
        risk_breakpoints,
        cluster_cause_baseline,
        tmrel_g_per_day,
    )


def _load_cluster_population_fallback(
    health_dir: Path, cluster_lookup: dict
) -> dict[int, float]:
    """Fallback: load cluster population from CSV files."""
    cluster_summary = pd.read_csv(health_dir / "cluster_summary.csv")
    return {
        int(k): float(v)
        for k, v in zip(
            cluster_summary["health_cluster"], cluster_summary["population_persons"]
        )
    }


def _interpolate_log_rr(
    intake: float, breakpoints: pd.DataFrame, risk_factor: str, cause: str
) -> float:
    """Interpolate log(RR) from breakpoints for given intake.

    Args:
        intake: Intake in g/day
        breakpoints: DataFrame with risk_factor, cause, intake_g_per_day, log_rr
        risk_factor: Risk factor name
        cause: Disease cause name

    Returns:
        Interpolated log(RR) value
    """
    mask = (breakpoints["risk_factor"] == risk_factor) & (breakpoints["cause"] == cause)
    bp = breakpoints.loc[mask].sort_values("intake_g_per_day")

    if bp.empty:
        return 0.0

    return float(np.interp(intake, bp["intake_g_per_day"], bp["log_rr"]))


def _extract_health_by_risk_factor_worker(args):
    """Worker function to extract health costs attributed to each risk factor.

    Steps:
    1. Load network and get food group store levels
    2. Aggregate by cluster to get per-capita intakes (g/day)
    3. Look up log(RR) for each (cluster, risk_factor, cause)
    4. Compute excess log(RR) relative to TMREL for proper attribution
    5. Proportionally allocate YLL based on excess log(RR)
    6. Sum across clusters and causes, return by risk factor
    """
    (
        network_path,
        processing_dir,
        grams_per_mt,
        days_per_year,
    ) = args

    # Load network first to access embedded population
    n = pypsa.Network(network_path)

    # Load health data tables (uses embedded cluster population from network)
    (
        cluster_lookup,
        cluster_population,
        risk_breakpoints,
        cluster_cause_baseline,
        tmrel_g_per_day,
    ) = _load_health_tables(processing_dir, network=n)

    # Get unique risk factors from breakpoints
    risk_factors = risk_breakpoints["risk_factor"].unique().tolist()

    # Build risk to causes mapping
    risk_cause_map = {}
    for rf in risk_factors:
        rf_causes = (
            risk_breakpoints.loc[risk_breakpoints["risk_factor"] == rf, "cause"]
            .unique()
            .tolist()
        )
        risk_cause_map[rf] = rf_causes

    # Precompute log(RR) at TMREL for each (risk_factor, cause) pair
    # This is the reference point for computing "excess" risk
    log_rr_at_tmrel = {}  # (rf, cause) -> log_rr
    for rf in risk_factors:
        tmrel_intake = tmrel_g_per_day.get(rf, 0.0)
        for cause in risk_cause_map.get(rf, []):
            log_rr_tmrel = _interpolate_log_rr(
                tmrel_intake, risk_breakpoints, rf, cause
            )
            log_rr_at_tmrel[(rf, cause)] = log_rr_tmrel

    # Get snapshot (network already loaded above)
    snapshot = n.snapshots[-1]

    # Get store levels at the snapshot
    if snapshot in n.stores_t.e.index:
        store_levels = n.stores_t.e.loc[snapshot]
    else:
        store_levels = pd.Series(dtype=float)

    # 1. Compute cluster intakes from food group stores
    # Stores with carrier group_{risk_factor} hold consumption
    cluster_intakes = {}  # (cluster, risk_factor) -> g/day per capita

    for rf in risk_factors:
        carrier = f"group_{rf}"
        fg_stores = n.stores[n.stores["carrier"] == carrier]

        if fg_stores.empty:
            continue

        for store_name in fg_stores.index:
            level_mt = store_levels.get(store_name, 0.0)
            if level_mt <= 0:
                continue

            # Get country from store (format: group_{rf}_{country})
            parts = store_name.split("_")
            if len(parts) >= 3:
                country = parts[-1]
            else:
                continue

            cluster = cluster_lookup.get(country)
            if cluster is None:
                continue

            # Accumulate total grams per cluster
            key = (cluster, rf)
            if key not in cluster_intakes:
                cluster_intakes[key] = 0.0
            cluster_intakes[key] += level_mt * grams_per_mt

    # Convert to g/day per capita
    for (cluster, rf), total_grams in list(cluster_intakes.items()):
        pop = cluster_population.get(cluster, 0)
        if pop > 0:
            cluster_intakes[(cluster, rf)] = total_grams / (days_per_year * pop)
        else:
            cluster_intakes[(cluster, rf)] = 0.0

    # 2. Look up log(RR) for each (cluster, risk_factor, cause)
    log_rr_values = {}  # (cluster, rf, cause) -> log_rr

    for (cluster, rf), intake in cluster_intakes.items():
        for cause in risk_cause_map.get(rf, []):
            log_rr = _interpolate_log_rr(intake, risk_breakpoints, rf, cause)
            log_rr_values[(cluster, rf, cause)] = log_rr

    # 3. Compute attributed YLL using proportional allocation based on EXCESS log(RR)
    attributed_yll = {}  # rf -> MYLL

    for cluster in cluster_population:
        cluster_rows = cluster_cause_baseline[
            cluster_cause_baseline["health_cluster"] == cluster
        ]

        for _, row in cluster_rows.iterrows():
            cause = row["cause"]

            # Compute EXCESS log(RR) for each risk factor relative to TMREL
            # excess = log(RR(x)) - log(RR(tmrel))
            # For protective foods at TMREL: excess ≈ 0 (no contribution)
            # For protective foods below TMREL: excess > 0 (contributing to burden)
            # For harmful foods (TMREL=0): excess = log(RR(x)) - 0 = log(RR(x))
            excess_contributions = {}
            for rf in risk_factors:
                key = (cluster, rf, cause)
                log_rr_current = log_rr_values.get(key, 0.0)
                log_rr_tmrel = log_rr_at_tmrel.get((rf, cause), 0.0)
                # Excess is how much worse than optimal; should be >= 0
                excess = max(0.0, log_rr_current - log_rr_tmrel)
                excess_contributions[rf] = excess

            # Total excess log(RR) for weighting
            total_excess = sum(excess_contributions.values())

            if total_excess <= 0:
                continue

            # Get actual YLL from the health store (instead of recomputing)
            # Store naming: yll_{cause}_cluster{cluster:03d}
            store_name = f"yll_{cause}_cluster{cluster:03d}"
            yll_myll = store_levels.get(store_name, 0.0)

            # Skip if negligible
            if yll_myll <= 1e-9:
                continue

            # Proportionally allocate to risk factors based on excess log(RR)
            for rf, excess in excess_contributions.items():
                if excess > 0:
                    weight = excess / total_excess
                    if rf not in attributed_yll:
                        attributed_yll[rf] = 0.0
                    attributed_yll[rf] += weight * yll_myll

    return pd.Series(attributed_yll, dtype=float)


def extract_health_data(
    scenarios: list[tuple[float, str, Path]],
    processing_dir: Path,
    cache_path: Path,
    param_name: str = "param_value",
    n_workers: int = 8,
) -> pd.DataFrame:
    """Extract health cost attribution by risk factor for all scenarios.

    Args:
        scenarios: List of (param_value, scenario_name, network_path) tuples
        processing_dir: Path to processing directory (for health data)
        cache_path: Path to cache file
        param_name: Name for the parameter (used as index name)
        n_workers: Number of parallel workers

    Returns:
        DataFrame with param_value as index and risk_factors as columns (in MYLL)
    """
    network_paths = [f for _, _, f in scenarios]

    if is_cache_valid(cache_path, network_paths):
        print(f"Loading health data from cache: {cache_path}")
        return pd.read_csv(cache_path, index_col=param_name)

    print(f"Extracting health data using {n_workers} workers...")

    worker_args = [
        (network_path, processing_dir, GRAMS_PER_MEGATONNE, DAYS_PER_YEAR)
        for _, _, network_path in scenarios
    ]
    param_values = [pv for pv, _, _ in scenarios]

    health_data = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_extract_health_by_risk_factor_worker, args): pv
            for args, pv in zip(worker_args, param_values)
        }

        for future in as_completed(futures):
            param_value = futures[future]
            health_data[param_value] = future.result()
            print(f"  Loaded {param_name}={int(param_value)}")

    df = pd.DataFrame(health_data).T.fillna(0)
    df.index.name = param_name
    df = df.sort_index()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path)
    print(f"Saved health data to cache: {cache_path}")

    return df


# -----------------------------------------------------------------------------
# Data preparation utilities
# -----------------------------------------------------------------------------


def aggregate_food_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate food groups: combine fruits+vegetables and eggs+poultry."""
    df_plot = df.copy()

    if "fruits" in df_plot.columns and "vegetables" in df_plot.columns:
        df_plot["fruits_vegetables"] = df_plot["fruits"] + df_plot["vegetables"]
        df_plot = df_plot.drop(columns=["fruits", "vegetables"])

    if "eggs" in df_plot.columns and "poultry" in df_plot.columns:
        df_plot["eggs_poultry"] = df_plot["eggs"] + df_plot["poultry"]
        df_plot = df_plot.drop(columns=["eggs", "poultry"])

    return df_plot


def prepare_objective_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare objective data: aggregate fertilizer into crop production and order categories."""
    df_obj = df.copy()

    if (
        "Fertilizer (synthetic)" in df_obj.columns
        and "Crop production" in df_obj.columns
    ):
        df_obj["Crop production"] = (
            df_obj["Crop production"] + df_obj["Fertilizer (synthetic)"]
        )
        df_obj = df_obj.drop(columns=["Fertilizer (synthetic)"])

    priority_order = ["Crop production", "Trade"]
    other_cats = [c for c in df_obj.columns if c not in priority_order]
    other_cats_sorted = (
        df_obj[other_cats].mean().sort_values(ascending=False).index.tolist()
    )
    cat_order = [c for c in priority_order if c in df_obj.columns] + other_cats_sorted

    return df_obj[cat_order]


def assign_food_colors(df: pd.DataFrame) -> dict:
    """Assign tab20 colors to food groups based on consumption at minimum x-value."""
    cmap = plt.colormaps["tab20"]
    min_val = df.index.min()
    group_order = df.loc[min_val].sort_values(ascending=False).index.tolist()
    return {group: cmap(i) for i, group in enumerate(group_order)}


# -----------------------------------------------------------------------------
# Plotting functions
# -----------------------------------------------------------------------------


def set_dual_xaxis_labels(
    ax: plt.Axes,
    x_ticks: list[float],
    ghg_values: list[float],
    yll_values: list[float],
    ghg_color: str = "darkgreen",
    yll_color: str = "darkblue",
    fontsize: int = 7,
):
    """Set up dual-colored x-axis tick labels showing both GHG price and YLL value.

    Args:
        ax: Matplotlib axes
        x_ticks: X-axis tick positions
        ghg_values: GHG price values for each tick
        yll_values: YLL values for each tick
        ghg_color: Color for GHG labels
        yll_color: Color for YLL labels
        fontsize: Font size for tick labels
    """
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([])  # Clear default labels

    # Get axis transform for positioning
    trans = ax.get_xaxis_transform()

    # Add colored labels below axis
    for x, ghg, yll in zip(x_ticks, ghg_values, yll_values):
        # Format values nicely
        ghg_str = f"{int(ghg)}" if ghg < 1000 else f"{int(ghg/1000)}k"
        yll_str = f"{int(yll)}" if yll < 1000 else f"{int(yll/1000)}k"

        # GHG label (top, dark green)
        ax.text(
            x,
            -0.02,
            ghg_str,
            transform=trans,
            ha="center",
            va="top",
            fontsize=fontsize,
            color=ghg_color,
            fontweight="bold",
        )
        # YLL label (bottom, dark blue)
        ax.text(
            x,
            -0.08,
            yll_str,
            transform=trans,
            ha="center",
            va="top",
            fontsize=fontsize,
            color=yll_color,
            fontweight="bold",
        )


def set_dual_xlabel(
    ax: plt.Axes,
    ghg_color: str = "darkgreen",
    yll_color: str = "darkblue",
    fontsize: int = 8,
):
    """Set dual-colored x-axis label for combined GHG/YLL sensitivity.

    Args:
        ax: Matplotlib axes
        ghg_color: Color for GHG part
        yll_color: Color for YLL part
        fontsize: Font size
    """
    # Use a two-line xlabel with colored text
    ax.set_xlabel("")  # Clear default

    ax.text(
        0.5,
        -0.18,
        "GHG price [USD/tCO2eq]",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=fontsize,
        color=ghg_color,
    )
    ax.text(
        0.5,
        -0.26,
        "Health value [USD/YLL]",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=fontsize,
        color=yll_color,
    )


def plot_stacked_sensitivity(
    df: pd.DataFrame,
    colors: dict,
    ax: plt.Axes,
    xlabel: str,
    ylabel: str,
    panel_label: str,
    x_ticks: list[float],
    x_ticklabels: list[str],
    label_x_positions: dict | None = None,
    label_skip: set | None = None,
    min_height_for_label: float = 30,
    y_max: float | None = None,
    pretty_names: dict | None = None,
):
    """Create a stacked area plot with logarithmic x-axis.

    Args:
        df: DataFrame with parameter values as index and groups as columns
        colors: Dict mapping group names to colors
        ax: Matplotlib axes to plot on
        xlabel: X-axis label
        ylabel: Y-axis label
        panel_label: Panel label (e.g., 'a', 'b', 'c')
        x_ticks: X-axis tick positions (in original scale, 0 maps to 1)
        x_ticklabels: X-axis tick labels
        label_x_positions: Manual x-positions for labels (optional)
        label_skip: Set of group names to skip labeling (optional)
        min_height_for_label: Minimum height to show a label
        y_max: Maximum y-axis value (optional)
        pretty_names: Custom pretty names dict (falls back to PRETTY_NAMES)
    """
    if label_x_positions is None:
        label_x_positions = {}
    if label_skip is None:
        label_skip = set()
    if pretty_names is None:
        pretty_names = PRETTY_NAMES

    x_values = df.index.values
    groups = df.columns.tolist()

    # Handle x=0 for log scale
    x_plot = np.where(x_values == 0, 1, x_values)

    x_min, x_max = 1, x_plot.max() * 1.1
    x_smooth = np.logspace(np.log10(x_min), np.log10(x_max), 200)

    y_smooth = {}
    for group in groups:
        y_smooth[group] = np.interp(
            np.log10(x_smooth), np.log10(x_plot), df[group].values
        )

    y_stacks = [np.zeros(len(x_smooth))]
    for group in groups:
        y_stacks.append(y_stacks[-1] + y_smooth[group])

    for i, group in enumerate(groups):
        y_bottom = y_stacks[i]
        y_top = y_stacks[i + 1]
        ax.fill_between(
            x_smooth,
            y_bottom,
            y_top,
            label=group,
            color=colors[group],
            alpha=0.8,
            edgecolor="none",
            linewidth=0,
        )

    # Add labels
    label_fontsize = 5
    bbox_style = {
        "boxstyle": "round,pad=0.15",
        "facecolor": "white",
        "alpha": 0.7,
        "edgecolor": "none",
    }

    log_x_min, log_x_max = np.log10(x_min), np.log10(x_max)
    margin_frac = 0.15
    log_margin = (log_x_max - log_x_min) * margin_frac

    for i, group in enumerate(groups):
        if group in label_skip:
            continue

        heights = y_smooth[group]
        max_height = heights.max()

        if max_height < min_height_for_label:
            continue

        max_idx = np.argmax(heights)
        x_pos = x_smooth[max_idx]

        if group in label_x_positions:
            x_pos = label_x_positions[group]

        log_x_pos = np.log10(x_pos)
        log_x_pos = np.clip(log_x_pos, log_x_min + log_margin, log_x_max - log_margin)
        x_pos = 10**log_x_pos

        idx = np.argmin(np.abs(x_smooth - x_pos))
        y_bottom = y_stacks[i]
        y_top = y_stacks[i + 1]
        y_pos = (y_bottom[idx] + y_top[idx]) / 2

        label_text = pretty_names.get(group, PRETTY_NAMES.get(group, group))
        ax.text(
            x_pos,
            y_pos,
            label_text,
            ha="center",
            va="center",
            fontsize=label_fontsize,
            fontweight="bold",
            color="black",
            bbox=bbox_style,
        )

    ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)

    ax.text(
        -0.10,
        1.05,
        panel_label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
    )

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels)
    ax.tick_params(axis="both", labelsize=7)
    ax.set_xlim(x_min, x_max)
    if y_max is not None:
        ax.set_ylim(0, y_max)
    else:
        ax.set_ylim(0, None)

    ax.grid(True, alpha=0.3, which="both")
    ax.set_axisbelow(True)


def plot_objective_sensitivity(
    df: pd.DataFrame,
    ax: plt.Axes,
    xlabel: str,
    panel_label: str,
    x_ticks: list[float],
    x_ticklabels: list[str],
    health_value: float | None = None,
    ghg_price: float | None = None,
    label_x_positions: dict | None = None,
    highlight_cat: str | None = None,
):
    """Create stacked area plot for objective breakdown with positive/negative categories.

    Args:
        df: DataFrame with parameter values as index and cost categories as columns
        ax: Matplotlib axes to plot on
        xlabel: X-axis label
        panel_label: Panel label (e.g., 'c')
        x_ticks: X-axis tick positions
        x_ticklabels: X-axis tick labels
        health_value: Health value to display in note box
        ghg_price: GHG price to display in note box
        label_x_positions: Manual x-positions for labels (optional)
        highlight_cat: Category to highlight with hatching (None for no highlighting)
    """
    if label_x_positions is None:
        label_x_positions = {}

    x_values = df.index.values
    categories = df.columns.tolist()

    x_plot = np.where(x_values == 0, 1, x_values)
    x_min, x_max = 1, x_plot.max() * 1.1
    x_smooth = np.logspace(np.log10(x_min), np.log10(x_max), 200)

    y_smooth = {}
    for cat in categories:
        y_smooth[cat] = np.interp(np.log10(x_smooth), np.log10(x_plot), df[cat].values)

    min_magnitude = 1.0

    crossing_cats = []
    purely_pos_cats = []
    purely_neg_cats = []

    for cat in categories:
        y = y_smooth[cat]
        max_abs = np.max(np.abs(y))
        if max_abs < min_magnitude:
            continue

        has_pos = np.any(y > 1e-6)
        has_neg = np.any(y < -1e-6)
        if has_pos and has_neg:
            crossing_cats.append(cat)
        elif has_pos:
            purely_pos_cats.append(cat)
        elif has_neg:
            purely_neg_cats.append(cat)

    y_smooth_split = {}
    cmap_obj = plt.colormaps["tab20c"]

    for cat in purely_pos_cats:
        y_smooth_split[cat] = y_smooth[cat]
    for cat in purely_neg_cats:
        y_smooth_split[cat] = y_smooth[cat]

    for cat in crossing_cats:
        y = y_smooth[cat]
        y_pos = np.maximum(y, 0)
        y_neg = np.minimum(y, 0)

        if np.max(y_pos) > min_magnitude:
            pos_name = f"{cat} (positive)"
            y_smooth_split[pos_name] = y_pos
            purely_pos_cats.append(pos_name)

        if np.min(y_neg) < -min_magnitude:
            neg_name = f"{cat} (negative)"
            y_smooth_split[neg_name] = y_neg
            purely_neg_cats.append(neg_name)

    if highlight_cat is not None and highlight_cat in purely_pos_cats:
        purely_pos_cats.remove(highlight_cat)
        purely_pos_cats.append(highlight_cat)

    split_colors = {}
    for i, cat in enumerate(purely_pos_cats):
        if cat == highlight_cat:
            split_colors[cat] = "grey"
        else:
            split_colors[cat] = cmap_obj(4 + (i % 4))
    for i, cat in enumerate(purely_neg_cats):
        split_colors[cat] = cmap_obj(i % 4)

    y_pos_stacks = [np.zeros(len(x_smooth))]
    for cat in purely_pos_cats:
        y_pos_stacks.append(y_pos_stacks[-1] + y_smooth_split[cat])

    y_neg_stacks = [np.zeros(len(x_smooth))]
    for cat in purely_neg_cats:
        y_neg_stacks.append(y_neg_stacks[-1] + y_smooth_split[cat])

    for i, cat in enumerate(purely_pos_cats):
        y_bottom = y_pos_stacks[i]
        y_top = y_pos_stacks[i + 1]
        if cat == highlight_cat:
            ax.fill_between(
                x_smooth,
                y_bottom,
                y_top,
                label=cat,
                color=split_colors[cat],
                alpha=0.4,
                edgecolor="none",
                linewidth=0,
                hatch="///",
            )
        else:
            ax.fill_between(
                x_smooth,
                y_bottom,
                y_top,
                label=cat,
                color=split_colors[cat],
                alpha=0.8,
                edgecolor="none",
                linewidth=0,
            )

    for i, cat in enumerate(purely_neg_cats):
        y_top = y_neg_stacks[i]
        y_bottom = y_neg_stacks[i + 1]
        ax.fill_between(
            x_smooth,
            y_bottom,
            y_top,
            label=cat,
            color=split_colors[cat],
            alpha=0.8,
            edgecolor="none",
            linewidth=0,
        )

    label_fontsize = 5
    bbox_style = {
        "boxstyle": "round,pad=0.15",
        "facecolor": "white",
        "alpha": 0.7,
        "edgecolor": "none",
    }

    log_x_min, log_x_max = np.log10(x_min), np.log10(x_max)
    margin_frac = 0.15
    log_margin = (log_x_max - log_x_min) * margin_frac

    for i, cat in enumerate(purely_pos_cats):
        heights = y_smooth_split[cat]
        max_height = heights.max()

        if max_height < 15:
            continue

        max_idx = np.argmax(heights)
        x_pos = x_smooth[max_idx]

        if cat in label_x_positions:
            x_pos = label_x_positions[cat]

        log_x_pos = np.log10(x_pos)
        log_x_pos = np.clip(log_x_pos, log_x_min + log_margin, log_x_max - log_margin)
        x_pos = 10**log_x_pos

        idx = np.argmin(np.abs(x_smooth - x_pos))
        y_bottom = y_pos_stacks[i]
        y_top = y_pos_stacks[i + 1]
        y_pos = (y_bottom[idx] + y_top[idx]) / 2

        pretty_name = PRETTY_NAMES_OBJ.get(cat, cat)
        ax.text(
            x_pos,
            y_pos,
            pretty_name,
            ha="center",
            va="center",
            fontsize=label_fontsize,
            fontweight="bold",
            color="black",
            bbox=bbox_style,
        )

    for i, cat in enumerate(purely_neg_cats):
        heights = np.abs(y_smooth_split[cat])
        max_height = heights.max()

        if max_height < 15:
            continue

        max_idx = np.argmax(heights)
        x_pos = x_smooth[max_idx]

        if cat in label_x_positions:
            x_pos = label_x_positions[cat]

        log_x_pos = np.log10(x_pos)
        log_x_pos = np.clip(log_x_pos, log_x_min + log_margin, log_x_max - log_margin)
        x_pos = 10**log_x_pos

        idx = np.argmin(np.abs(x_smooth - x_pos))
        y_top = y_neg_stacks[i]
        y_bottom = y_neg_stacks[i + 1]
        y_pos = (y_bottom[idx] + y_top[idx]) / 2

        pretty_name = PRETTY_NAMES_OBJ.get(cat, cat)
        ax.text(
            x_pos,
            y_pos,
            pretty_name,
            ha="center",
            va="center",
            fontsize=label_fontsize,
            fontweight="bold",
            color="black",
            bbox=bbox_style,
        )

    if health_value is not None or ghg_price is not None:
        note_lines = ["Fixed in this plot:"]
        if health_value is not None:
            note_lines.append(f"  Health: ${health_value:,.0f}/YLL")
        if ghg_price is not None:
            note_lines.append(f"  GHG: ${ghg_price:,.0f}/tCO2eq")
        note_text = "\n".join(note_lines)
        note_bbox = {
            "boxstyle": "round,pad=0.3",
            "facecolor": "lightyellow",
            "edgecolor": "none",
            "alpha": 0.9,
        }
        ax.text(
            0.98,
            0.97,
            note_text,
            transform=ax.transAxes,
            fontsize=5,
            va="top",
            ha="right",
            bbox=note_bbox,
        )

    ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("Cost [billion USD]", fontsize=8)

    ax.text(
        -0.10,
        1.05,
        panel_label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
    )

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels)
    ax.tick_params(axis="both", labelsize=7)
    ax.set_xlim(x_min, x_max)

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_axisbelow(True)
