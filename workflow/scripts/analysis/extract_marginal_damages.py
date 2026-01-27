# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Extract marginal GHG and health damages by food group and country.

This script computes consumption-attributed marginal damages:
- GHG: MtCO2e per Mt of food consumed, traced through trade/processing to production
- Health: million YLL per Mt of food consumed, based on dose-response curve slopes
"""

from collections import defaultdict
from dataclasses import dataclass
import logging
from math import exp
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

from workflow.scripts.constants import DAYS_PER_YEAR, GRAMS_PER_MEGATONNE, PER_100K

logger = logging.getLogger(__name__)


@dataclass
class HealthData:
    """Container for health-related input data."""

    risk_breakpoints: pd.DataFrame
    cluster_cause: pd.DataFrame
    cause_log_breakpoints: pd.DataFrame
    country_clusters: pd.DataFrame
    population: pd.DataFrame


def load_health_data(inputs: dict) -> HealthData:
    """Load health data files from snakemake inputs."""
    return HealthData(
        risk_breakpoints=pd.read_csv(inputs["risk_breakpoints"]),
        cluster_cause=pd.read_csv(inputs["health_cluster_cause"]),
        cause_log_breakpoints=pd.read_csv(inputs["health_cause_log"]),
        country_clusters=pd.read_csv(inputs["health_clusters"]),
        population=pd.read_csv(inputs["population"]),
    )


def get_cluster_population(
    country_clusters: pd.DataFrame,
    population: pd.DataFrame,
) -> dict[int, float]:
    """Compute total population per health cluster."""
    clusters = country_clusters.assign(
        country_iso3=lambda df: df["country_iso3"].str.upper()
    )
    cluster_lookup = (
        clusters.set_index("country_iso3")["health_cluster"].astype(int).to_dict()
    )

    pop = population.assign(iso3=lambda df: df["iso3"].str.upper())
    pop_map = pop.set_index("iso3")["population"].astype(float).to_dict()

    result: dict[int, float] = defaultdict(float)
    for iso3, cluster in cluster_lookup.items():
        result[int(cluster)] += pop_map.get(iso3, 0.0)

    return dict(result)


def get_country_cluster_lookup(country_clusters: pd.DataFrame) -> dict[str, int]:
    """Map country ISO3 codes to health cluster IDs."""
    clusters = country_clusters.assign(
        country_iso3=lambda df: df["country_iso3"].str.upper()
    )
    return clusters.set_index("country_iso3")["health_cluster"].astype(int).to_dict()


# -----------------------------------------------------------------------------
# GHG Marginal Computation via Sparse Matrix Flow Attribution
# -----------------------------------------------------------------------------


def compute_ghg_marginals(
    n: pypsa.Network,
    food_groups: pd.DataFrame,
    ch4_gwp: float,
    n2o_gwp: float,
) -> pd.DataFrame:
    """Compute GHG emissions per Mt consumed, by food and country.

    Uses flow-based attribution via sparse matrix solve. Builds the network
    flow graph and solves (I - M) * rho = e where:
    - rho[b] = emission intensity at bus b (MtCO2e per Mt flow)
    - M = weighted adjacency matrix (flow fractions)
    - e = direct emission intensities

    Returns DataFrame with columns: country, food, food_group, consumption_mt,
    ghg_mtco2e_per_mt
    """

    # Build food -> food_group mapping
    food_to_group = food_groups.set_index("food")["group"].to_dict()

    # Get snapshot and flows
    snapshot = n.snapshots[-1]
    p0 = (
        n.links.dynamic.p0.loc[snapshot]
        if snapshot in n.links.dynamic.p0.index
        else pd.Series(dtype=float)
    )

    # Build links DataFrame with flows and emissions
    links_df = build_ghg_links_dataframe(n, p0, ch4_gwp, n2o_gwp)

    if links_df.empty:
        logger.warning("No links with positive flow found")
        return pd.DataFrame(
            columns=[
                "country",
                "food",
                "food_group",
                "consumption_mt",
                "ghg_mtco2e_per_mt",
            ]
        )

    # Compute emission intensities at each bus via sparse matrix solve
    bus_intensities = solve_emission_intensities(links_df)

    # Extract results for consumption links using carrier-based filtering
    links_static = n.links.static
    consume_mask = links_static["carrier"] == "food_consumption"
    consume_links = links_static.loc[consume_mask].copy()

    # Get flows for consume links
    consume_df = links_df.loc[links_df["link_name"].isin(consume_links.index)].copy()

    # Map bus intensities to consumption links
    consume_df["intensity"] = consume_df["bus0"].map(bus_intensities).fillna(0.0)

    # Use domain columns instead of name parsing
    records = []
    for _, row in consume_df.iterrows():
        link_name = row["link_name"]
        food = links_static.at[link_name, "food"]
        country = links_static.at[link_name, "country"]

        food_group = food_to_group.get(food)
        if not food_group:
            continue

        records.append(
            {
                "country": country,
                "food": food,
                "food_group": food_group,
                "consumption_mt": row["flow"],
                "ghg_mtco2e_per_mt": row["intensity"],
            }
        )

    logger.info(
        "Computed GHG intensities for %d buses, %d consumption links",
        len(bus_intensities),
        len(records),
    )

    return pd.DataFrame(records)


def build_ghg_links_dataframe(
    n: pypsa.Network,
    p0: pd.Series,
    ch4_gwp: float,
    n2o_gwp: float,
) -> pd.DataFrame:
    """Build DataFrame of links with flows and GHG emissions.

    Returns DataFrame with columns:
    - link_name, bus0, bus1, flow, efficiency, emissions_co2e
    """
    # GWP factors (CH4/N2O are in tonnes, convert to MtCO2e)
    gwp = {"co2": 1.0, "ch4": ch4_gwp * 1e-6, "n2o": n2o_gwp * 1e-6}

    links = n.links.static.copy()
    links["link_name"] = links.index
    links["flow"] = p0.reindex(links.index).fillna(0.0)

    # Filter to positive flows only
    links = links[links["flow"] > 1e-12].copy()

    if links.empty:
        return pd.DataFrame()

    # Ensure efficiency is filled
    links["efficiency"] = links["efficiency"].fillna(1.0)

    # Compute emissions per unit of input flow (summing bus2/3/4 contributions)
    links["emissions_co2e"] = 0.0

    for bus_col, eff_col in [
        ("bus2", "efficiency2"),
        ("bus3", "efficiency3"),
        ("bus4", "efficiency4"),
    ]:
        if bus_col not in links.columns:
            continue

        # Get the emission bus and efficiency for each link
        emission_bus = links[bus_col].fillna("")
        eff = links[eff_col].fillna(0.0) if eff_col in links.columns else 0.0

        # Only positive efficiencies are emissions (negative = inputs)
        # Map to GWP factor based on bus name
        for gas, gwp_factor in gwp.items():
            mask = (emission_bus == gas) & (eff > 0)
            links.loc[mask, "emissions_co2e"] += eff[mask] * gwp_factor

    return links[["link_name", "bus0", "bus1", "flow", "efficiency", "emissions_co2e"]]


def solve_emission_intensities(links_df: pd.DataFrame) -> dict[str, float]:
    """Solve for emission intensity at each bus using sparse matrix.

    The intensity rho[b] at bus b satisfies:
        rho[b] = e[b] + sum over incoming links l: w[l] * rho[bus0[l]]

    Where:
        w[l] = flow[l] / total_outflow[bus1[l]]  (weight of link)
        e[b] = sum(flow[l] * emissions[l]) / total_outflow[b]  (direct emissions)

    This gives: (I - M) * rho = e, solved via sparse linear algebra.
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    # Get unique buses and create integer indices
    all_buses = pd.concat([links_df["bus0"], links_df["bus1"]]).unique()
    bus_to_idx = {bus: i for i, bus in enumerate(all_buses)}
    n_buses = len(all_buses)

    # Map bus names to indices (vectorized)
    links_df = links_df.copy()
    links_df["idx0"] = links_df["bus0"].map(bus_to_idx)
    links_df["idx1"] = links_df["bus1"].map(bus_to_idx)

    # Compute total outflow at each destination bus: sum(flow * efficiency)
    links_df["outflow"] = links_df["flow"] * links_df["efficiency"]
    total_outflow = links_df.groupby("idx1")["outflow"].transform("sum")

    # Compute weights: flow / total_outflow[bus1]
    links_df["weight"] = links_df["flow"] / total_outflow

    # Compute direct emission contribution: flow * emissions / total_outflow
    links_df["emission_contrib"] = (
        links_df["flow"] * links_df["emissions_co2e"] / total_outflow
    )

    # Build sparse matrix M where M[i, j] = sum of weights for links from j to i
    # Using COO format for efficient construction
    row = links_df["idx1"].values
    col = links_df["idx0"].values
    data = links_df["weight"].values

    adj_matrix = sparse.coo_matrix((data, (row, col)), shape=(n_buses, n_buses))
    adj_matrix = adj_matrix.tocsr()  # Convert to CSR for efficient arithmetic

    # Build emission vector e[i] = sum of emission contributions to bus i
    e = np.zeros(n_buses)
    np.add.at(e, links_df["idx1"].values, links_df["emission_contrib"].values)

    # Solve (I - M) * rho = e
    identity = sparse.eye(n_buses, format="csr")
    system_matrix = identity - adj_matrix

    try:
        rho = spsolve(system_matrix, e)
    except Exception as ex:
        logger.warning("Sparse solve failed: %s, falling back to iterative", ex)
        rho = solve_iterative(adj_matrix, e, max_iter=50, tol=1e-9)

    # Map back to bus names
    idx_to_bus = {i: bus for bus, i in bus_to_idx.items()}
    return {idx_to_bus[i]: float(rho[i]) for i in range(n_buses)}


def solve_iterative(
    adj_matrix: "sparse.csr_matrix",
    e: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-9,
) -> np.ndarray:
    """Iterative solver for (I - M) * rho = e, i.e., rho = e + M @ rho."""
    rho = e.copy()
    for i in range(max_iter):
        rho_new = e + adj_matrix @ rho
        diff = np.abs(rho_new - rho).max()
        rho = rho_new
        if diff < tol:
            logger.info("Iterative solver converged in %d iterations", i + 1)
            break
    else:
        logger.warning(
            "Iterative solver did not converge after %d iterations (diff=%.2e)",
            max_iter,
            diff,
        )
    return rho


# -----------------------------------------------------------------------------
# Health Marginal Computation
# -----------------------------------------------------------------------------


def compute_health_marginals(
    n: pypsa.Network,
    food_groups: pd.DataFrame,
    health_data: HealthData,
    risk_factors: list[str],
) -> pd.DataFrame:
    """Compute marginal YLL per Mt consumed, by food group and country.

    Computes the derivative of the piecewise linear dose-response curve
    at the current intake level, then converts to per-Mt units.

    Returns DataFrame with columns: country, food_group, yll_myll_per_mt
    """
    # Build lookups
    cluster_lookup = get_country_cluster_lookup(health_data.country_clusters)
    cluster_population = get_cluster_population(
        health_data.country_clusters, health_data.population
    )

    # Food to risk factor mapping (only configured risk factors)
    food_risk_map = food_groups[food_groups["group"].isin(risk_factors)].copy()
    food_to_risk = food_risk_map.set_index("food")["group"].to_dict()

    # Build risk tables: risk_factor -> DataFrame(intake, cause -> log_rr)
    risk_tables = {}
    for risk, group in health_data.risk_breakpoints.groupby("risk_factor"):
        pivot = (
            group.sort_values(["intake_g_per_day", "cause"])
            .pivot_table(
                index="intake_g_per_day",
                columns="cause",
                values="log_rr",
                aggfunc="first",
            )
            .sort_index()
        )
        risk_tables[str(risk)] = pivot

    # Build cause log breakpoint tables
    cause_tables = {
        str(cause): df.sort_values("log_rr_total")
        for cause, df in health_data.cause_log_breakpoints.groupby("cause")
    }

    # Cluster-cause baseline data
    cluster_cause = health_data.cluster_cause.assign(
        health_cluster=lambda df: df["health_cluster"].astype(int)
    ).set_index(["health_cluster", "cause"])

    # Compute current intake per (cluster, risk_factor)
    intake_totals = compute_intake_by_cluster_risk(
        n, food_to_risk, cluster_lookup, cluster_population
    )

    # Compute marginal YLL per g/day for each (cluster, risk_factor)
    # Iterate over all clusters and risks, not just those with non-zero intake
    marginal_yll_per_g: dict[tuple[int, str], float] = {}

    all_clusters = set(cluster_population.keys())
    all_risks = set(risk_tables.keys())

    for cluster in all_clusters:
        cluster_pop = cluster_population[cluster]
        if cluster_pop <= 0:
            continue

        for risk in all_risks:
            intake_g = intake_totals.get((cluster, risk), 0.0)
            risk_table = risk_tables[risk]

            # Sum marginal YLL across all causes
            total_marginal = 0.0

            for cause in risk_table.columns:
                if (cluster, cause) not in cluster_cause.index:
                    continue

                row = cluster_cause.loc[(cluster, cause)]

                # Reconstruct absolute YLL from rate using planning-year population
                yll_attrib_rate = float(row["yll_attrib_rate_per_100k"])
                yll_base = (yll_attrib_rate / PER_100K) * cluster_pop

                log_rr_ref = float(row["log_rr_total_ref"])
                rr_ref = exp(log_rr_ref)

                if yll_base <= 0 or rr_ref <= 0:
                    continue

                # Get breakpoints for this (risk, cause)
                xs = risk_table.index.to_numpy(dtype=float)
                ys = risk_table[cause].to_numpy(dtype=float)

                if len(xs) < 2:
                    continue

                # Find current segment and compute slope
                d_log_rr = compute_piecewise_slope(xs, ys, intake_g)

                # Compute total log(RR) to get current RR
                log_rr = float(np.interp(intake_g, xs, ys))

                # Get cause breakpoints for RR interpolation
                cause_bp = cause_tables.get(cause)
                if cause_bp is None or cause_bp.empty:
                    # Approximate: use exp(log_rr) directly
                    rr = exp(log_rr)
                else:
                    log_points = cause_bp["log_rr_total"].to_numpy(dtype=float)
                    rr_points = cause_bp["rr_total"].to_numpy(dtype=float)
                    rr = float(np.interp(log_rr, log_points, rr_points))

                # Chain rule: d(YLL)/d(intake) = d(YLL)/d(RR) * d(RR)/d(log_RR) * d(log_RR)/d(intake)
                # d(YLL)/d(RR) = yll_base / rr_ref
                # d(RR)/d(log_RR) = RR (derivative of exp)
                marginal_yll = (yll_base / rr_ref) * rr * d_log_rr

                total_marginal += marginal_yll

            marginal_yll_per_g[(cluster, risk)] = total_marginal

    # Convert to per-country, per-food_group output
    # For each country, aggregate food group consumption and compute marginal
    records = []

    for country, cluster in cluster_lookup.items():
        cluster_pop = cluster_population[cluster]
        if cluster_pop <= 0:
            continue

        for risk in risk_factors:
            marginal_g = marginal_yll_per_g.get((cluster, risk), 0.0)

            # Convert from YLL per (g/capita/day) to million YLL per Mt
            # 1 Mt = 1e12 g, 1 year = 365 days
            # marginal_g is YLL per (g/capita/day) for the whole cluster
            # We want million YLL per Mt consumed (affecting cluster health)
            # Consumption of 1 Mt affects cluster intake by 1e12 / (365 * cluster_pop)
            # So marginal per Mt = marginal_g * 1e12 / (365 * cluster_pop) / 1e6
            intake_per_mt = GRAMS_PER_MEGATONNE / (DAYS_PER_YEAR * cluster_pop)
            marginal_myll_per_mt = marginal_g * intake_per_mt / 1e6

            records.append(
                {
                    "country": country,
                    "food_group": risk,
                    "yll_myll_per_mt": marginal_myll_per_mt,
                }
            )

    return pd.DataFrame(records)


def compute_intake_by_cluster_risk(
    n: pypsa.Network,
    food_to_risk: dict[str, str],
    cluster_lookup: dict[str, int],
    cluster_population: dict[int, float],
) -> dict[tuple[int, str], float]:
    """Compute current intake in g/capita/day by (cluster, risk_factor)."""
    intake_totals: dict[tuple[int, str], float] = defaultdict(float)

    snapshot = n.snapshots[-1]
    p0 = (
        n.links.dynamic.p0.loc[snapshot]
        if snapshot in n.links.dynamic.p0.index
        else pd.Series()
    )

    links_static = n.links.static

    # Filter to consume links using carrier-based filtering
    consume_mask = links_static["carrier"] == "food_consumption"
    consume_links = links_static.loc[consume_mask]

    for link_name in consume_links.index:
        # Use domain columns instead of name parsing
        food = links_static.at[link_name, "food"]
        country = str(links_static.at[link_name, "country"]).upper()

        risk = food_to_risk.get(food)
        if not risk:
            continue

        cluster = cluster_lookup.get(country)
        if cluster is None:
            continue

        population = cluster_population.get(int(cluster), 0.0)
        if population <= 0:
            continue

        flow_mt = float(p0.get(link_name, 0.0))
        if flow_mt <= 0:
            continue

        # Convert Mt/year to g/capita/day
        scale = GRAMS_PER_MEGATONNE / (DAYS_PER_YEAR * population)
        intake_totals[(int(cluster), risk)] += flow_mt * scale

    return dict(intake_totals)


def compute_piecewise_slope(
    x_breakpoints: np.ndarray,
    y_breakpoints: np.ndarray,
    x_current: float,
) -> float:
    """Compute the slope of a piecewise linear function at a given point.

    Returns the derivative (slope) of the line segment containing x_current.
    """
    if len(x_breakpoints) < 2:
        return 0.0

    # Find the segment containing x_current
    # np.searchsorted returns the index where x_current would be inserted
    idx = np.searchsorted(x_breakpoints, x_current)

    # Clamp to valid segment range
    if idx == 0:
        idx = 1
    elif idx >= len(x_breakpoints):
        idx = len(x_breakpoints) - 1

    # Segment is [idx-1, idx]
    x0, x1 = x_breakpoints[idx - 1], x_breakpoints[idx]
    y0, y1 = y_breakpoints[idx - 1], y_breakpoints[idx]

    dx = x1 - x0
    if abs(dx) < 1e-12:
        return 0.0

    return (y1 - y0) / dx


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def aggregate_by_food_group(
    ghg_df: pd.DataFrame,
    health_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate results by country and food group.

    For GHG: consumption-weighted average of per-Mt emissions
    For health: already at food_group level
    """
    # Aggregate GHG by food group (consumption-weighted average)
    if not ghg_df.empty:
        ghg_agg = (
            ghg_df.groupby(["country", "food_group"])
            .apply(
                lambda g: pd.Series(
                    {
                        "consumption_mt": g["consumption_mt"].sum(),
                        "ghg_mtco2e_per_mt": (
                            (g["ghg_mtco2e_per_mt"] * g["consumption_mt"]).sum()
                            / g["consumption_mt"].sum()
                            if g["consumption_mt"].sum() > 0
                            else 0.0
                        ),
                    }
                ),
                include_groups=False,
            )
            .reset_index()
        )
    else:
        ghg_agg = pd.DataFrame(
            columns=["country", "food_group", "consumption_mt", "ghg_mtco2e_per_mt"]
        )

    # Merge with health data
    if not health_df.empty:
        result = ghg_agg.merge(
            health_df[["country", "food_group", "yll_myll_per_mt"]],
            on=["country", "food_group"],
            how="outer",
        )
    else:
        result = ghg_agg.copy()
        result["yll_myll_per_mt"] = 0.0

    # Fill NaN values
    result = result.fillna(
        {"consumption_mt": 0.0, "ghg_mtco2e_per_mt": 0.0, "yll_myll_per_mt": 0.0}
    )

    return result


def add_monetary_values(
    df: pd.DataFrame,
    ghg_price: float,
    value_per_yll: float,
) -> pd.DataFrame:
    """Add USD per tonne columns for GHG and health damages.

    Parameters
    ----------
    df : DataFrame with ghg_mtco2e_per_mt and yll_myll_per_mt columns
    ghg_price : USD per tonne CO2e
    value_per_yll : USD per YLL

    Returns DataFrame with additional ghg_usd_per_t and health_usd_per_t columns
    """
    df = df.copy()

    # GHG: MtCO2e/Mt → tCO2e/t (same ratio), then multiply by price
    df["ghg_usd_per_t"] = df["ghg_mtco2e_per_mt"] * ghg_price

    # Health: mYLL/Mt → YLL/t (multiply by 1e6/1e6 = 1), then multiply by value
    df["health_usd_per_t"] = df["yll_myll_per_mt"] * value_per_yll

    return df


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load network
    n = pypsa.Network(snakemake.input.network)
    logger.info("Loaded network with %d links", len(n.links))

    # Load food groups
    food_groups = pd.read_csv(snakemake.input.food_groups)

    # Load health data
    health_data = load_health_data(
        {
            "risk_breakpoints": snakemake.input.risk_breakpoints,
            "health_cluster_cause": snakemake.input.health_cluster_cause,
            "health_cause_log": snakemake.input.health_cause_log,
            "health_clusters": snakemake.input.health_clusters,
            "population": snakemake.input.population,
        }
    )

    # Get params
    ghg_price = float(snakemake.params.ghg_price)
    value_per_yll = float(snakemake.params.value_per_yll)
    ch4_gwp = float(snakemake.params.ch4_gwp)
    n2o_gwp = float(snakemake.params.n2o_gwp)
    risk_factors = list(snakemake.params.health_risk_factors)

    logger.info("Computing GHG marginals...")
    ghg_df = compute_ghg_marginals(n, food_groups, ch4_gwp, n2o_gwp)
    logger.info("Computed GHG for %d food-country pairs", len(ghg_df))

    logger.info("Computing health marginals...")
    health_df = compute_health_marginals(n, food_groups, health_data, risk_factors)
    logger.info("Computed health for %d food_group-country pairs", len(health_df))

    logger.info("Aggregating by food group...")
    result = aggregate_by_food_group(ghg_df, health_df)

    logger.info("Adding monetary values...")
    result = add_monetary_values(result, ghg_price, value_per_yll)

    # Sort for consistent output
    result = result.sort_values(["country", "food_group"]).reset_index(drop=True)

    # Write output
    output_path = Path(snakemake.output.csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    logger.info("Wrote marginal damages to %s (%d rows)", output_path, len(result))


if __name__ == "__main__":
    main()
