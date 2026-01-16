# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Health objective constraints for the food systems optimization model.

This module implements health cost constraints as described in docs/health.rst.
The health objective quantifies the cost of dietary choices in terms of years
of life lost (YLL), using epidemiological dose-response relationships from
the Global Burden of Disease (GBD) Study.

Mathematical Formulation
------------------------

The health cost for cluster c and disease d is (see docs/health.rst):

    Cost_{c,d}(x) = V * (YLL_{c,d} / RR_d(x^base)) * (RR_d(x) - RR_d^ref)

where:
    - V = value per year of life lost (USD/YLL)
    - YLL_{c,d} = baseline years of life lost
    - RR_d(x) = relative risk at intake x (product over risk factors r)
    - RR_d^ref = RR at TMREL (theoretical minimum risk exposure level)
    - x^base = baseline intake

The combined relative risk is multiplicative across risk factors:

    RR_d(x) = ∏_r RR_{r,d}(x_r)

Implementation Strategy
-----------------------

To handle the nonlinear multiplicative combination, we use a two-stage
piecewise-linear approximation:

    Stage 1: Intake x_r → log(RR_{r,d}) for each (cluster, risk) pair
    Stage 2: Σ_r log(RR_{r,d}) → exp(·) → RR_d → YLL store level

Both stages use SOS2 (Special Ordered Sets Type 2) constraints for
piecewise-linear interpolation.
"""

from collections import defaultdict
import itertools
import logging
import math

import linopy
import numpy as np
import pandas as pd
import pypsa
import xarray as xr

from .. import constants
from ..population import get_health_cluster_population

logger = logging.getLogger(__name__)


# =============================================================================
# Module State for Auxiliary Variable Tracking
# =============================================================================

# Auxiliary variables (SOS2 segment binaries) must be removed before
# PyPSA solution assignment to avoid polluting the solved network.
HEALTH_AUX_MAP: dict[int, set[str]] = {}

# Counters for unique variable naming
_SOS2_COUNTER = [0]
_LAMBDA_GROUP_COUNTER = itertools.count()
_TOTAL_GROUP_COUNTER = itertools.count()


def _register_auxiliary_variable(m: linopy.Model, name: str) -> None:
    """Track an auxiliary variable for post-solve cleanup."""
    aux = HEALTH_AUX_MAP.setdefault(id(m), set())
    aux.add(name)


# =============================================================================
# SOS2 Constraint Helper
# =============================================================================


def _add_sos2_with_fallback(
    m: linopy.Model, variable: xr.DataArray, sos_dim: str, solver_name: str
) -> list[str]:
    """Add SOS2 constraint or binary fallback depending on solver support.

    SOS2 constraints ensure at most two adjacent λ_k variables are nonzero,
    enabling piecewise-linear interpolation:

        x = Σ_k λ_k x_k,  f(x) ≈ Σ_k λ_k f(x_k)

    Parameters
    ----------
    m
        The linopy model.
    variable
        Lambda variables with dimension ``sos_dim`` for interpolation points.
    sos_dim
        The dimension along which SOS2 adjacency is enforced.
    solver_name
        Solver name; HiGHS requires binary fallback since it lacks native SOS2.

    Returns
    -------
    list[str]
        Names of any auxiliary binary variables created (for cleanup tracking).
    """
    if solver_name.lower() != "highs":
        m.add_sos_constraints(variable, sos_type=2, sos_dim=sos_dim)
        return []

    # HiGHS fallback: implement SOS2 via binary variables
    coords = variable.coords[sos_dim]
    n_points = len(coords)
    if n_points <= 1:
        return []

    other_dims = [dim for dim in variable.dims if dim != sos_dim]

    # Create unique interval dimension name
    interval_dim = f"{sos_dim}_interval"
    suffix = 1
    while interval_dim in variable.dims:
        interval_dim = f"{sos_dim}_interval{suffix}"
        suffix += 1

    interval_index = pd.Index(range(n_points - 1), name=interval_dim)
    binary_coords = [variable.coords[d] for d in other_dims] + [interval_index]

    # Create binary segment variables
    _SOS2_COUNTER[0] += 1
    base_name = f"{variable.name}_segment" if variable.name else "health_segment"
    binary_name = f"{base_name}_{_SOS2_COUNTER[0]}"

    binaries = m.add_variables(coords=binary_coords, binary=True, name=binary_name)

    # Exactly one segment active
    m.add_constraints(binaries.sum(interval_dim) == 1)

    # Adjacency: λ_j ≤ binary_{j-1} + binary_j
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


# =============================================================================
# Data Loading
# =============================================================================


def _load_health_data(
    n: pypsa.Network,
    risk_breakpoints_path: str,
    cluster_cause_path: str,
    cause_log_path: str,
    cluster_summary_path: str,
    clusters_path: str,
) -> dict:
    """Load and preprocess all health-related input data.

    Returns a dictionary with all preprocessed data needed for constraint
    construction.
    """
    risk_breakpoints = pd.read_csv(risk_breakpoints_path)
    cluster_cause = pd.read_csv(cluster_cause_path)
    cause_log_breakpoints = pd.read_csv(cause_log_path)
    cluster_summary = pd.read_csv(cluster_summary_path)
    cluster_summary["health_cluster"] = cluster_summary["health_cluster"].astype(int)
    cluster_map = pd.read_csv(clusters_path)

    # Cluster lookups
    cluster_lookup = cluster_map.set_index("country_iso3")["health_cluster"].to_dict()

    # Cluster-cause metadata (baseline YLL, RR values)
    cluster_cause_metadata = cluster_cause.set_index(["health_cluster", "cause"])

    # Get cluster population from network metadata (computed at build time)
    cluster_population = get_health_cluster_population(n)

    # Sort breakpoint tables
    risk_breakpoints = risk_breakpoints.sort_values(
        ["risk_factor", "intake_g_per_day", "cause"]
    )
    cause_log_breakpoints = cause_log_breakpoints.sort_values(["cause", "log_rr_total"])

    return {
        "risk_breakpoints": risk_breakpoints,
        "cluster_cause": cluster_cause,
        "cause_log_breakpoints": cause_log_breakpoints,
        "cluster_summary": cluster_summary,
        "cluster_cause_metadata": cluster_cause_metadata,
        "cluster_lookup": cluster_lookup,
        "cluster_population": cluster_population,
    }


# =============================================================================
# Stage 1: Intake → log(RR)
# =============================================================================


def _build_store_to_cluster_map(
    stores_df: pd.DataFrame,
    risk_factors: list[str],
    cluster_lookup: dict[str, int],
    cluster_population: dict[int, float],
) -> pd.DataFrame:
    """Map food group stores to health clusters with per-capita coefficients.

    For each food group store, computes the coefficient for converting store
    level (Mt/year) to per-capita intake (g/day):

        coeff = 10^12 / (365 * P_c)

    where P_c is the population of cluster c that country belongs to.

    Parameters
    ----------
    stores_df
        DataFrame of stores with 'food_group' and 'country' columns.
    risk_factors
        List of GBD risk factors (e.g., ['fruits', 'vegetables', ...]).
    cluster_lookup
        Mapping from country ISO3 to health cluster.
    cluster_population
        Population by health cluster.

    Returns
    -------
    pd.DataFrame
        Columns: store_name, risk_factor, country, cluster, coeff.
    """
    # Filter for food group stores matching risk factors
    fg_stores = stores_df[stores_df["food_group"].isin(risk_factors)].copy()

    if fg_stores.empty:
        return pd.DataFrame()

    # Build mapping DataFrame using food_group column directly
    df = pd.DataFrame(
        {
            "store_name": fg_stores.index,
            "risk_factor": fg_stores["food_group"].values,
            "country": fg_stores["country"].values,
        }
    )

    # Map to cluster - fail if any countries are unmapped
    df["cluster"] = df["country"].map(cluster_lookup)
    unmapped = df[df["cluster"].isna()]["country"].unique()
    if len(unmapped) > 0:
        raise ValueError(f"Countries not mapped to health clusters: {sorted(unmapped)}")
    df["cluster"] = df["cluster"].astype(int)

    # Get cluster population - fail if any clusters have zero/missing population
    df["population"] = df["cluster"].map(cluster_population)
    zero_pop_clusters = df[df["population"].isna() | (df["population"] <= 0)][
        "cluster"
    ].unique()
    if len(zero_pop_clusters) > 0:
        raise ValueError(
            f"Health clusters with zero or missing population: {sorted(zero_pop_clusters)}"
        )

    # Per-capita coefficient: grams/megatonne / (365 * cluster_population)
    df["coeff"] = constants.GRAMS_PER_MEGATONNE / (365.0 * df["population"])

    return df


def _build_intake_breakpoints(risk_breakpoints: pd.DataFrame) -> dict:
    """Build intake grids from RR breakpoint data.

    For each risk factor, creates:
        - intake_steps: Index of breakpoint positions
        - intake_values: xr.DataArray of intake values (g/day)
        - log_rr: DataFrame with log(RR) by (intake_step, cause)

    Parameters
    ----------
    risk_breakpoints
        DataFrame with columns: risk_factor, intake_g_per_day, cause, log_rr.

    Returns
    -------
    dict
        {risk_factor: {intake_steps, intake_values, log_rr}}
    """
    risk_data = {}
    for risk, grp in risk_breakpoints.groupby("risk_factor"):
        intakes = pd.Index(sorted(grp["intake_g_per_day"].unique()), name="intake")
        if intakes.empty:
            continue

        # Pivot to get log_rr by (intake, cause)
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

    return risk_data


def _group_cluster_risk_pairs(
    store_map: pd.DataFrame, intake_data: dict
) -> dict[tuple[float, ...], list[tuple[int, str]]]:
    """Group (cluster, risk) pairs by shared intake coordinate patterns.

    Pairs with identical breakpoint grids share a single SOS2 variable set,
    improving solver efficiency.
    """
    unique_pairs = store_map[["cluster", "risk_factor"]].drop_duplicates()

    intake_groups: dict[tuple[float, ...], list[tuple[int, str]]] = defaultdict(list)
    for _, row in unique_pairs.iterrows():
        cluster = int(row["cluster"])
        risk = row["risk_factor"]

        risk_table = intake_data.get(risk)
        if risk_table is None:
            continue

        coords_key = tuple(risk_table["intake_values"].values)
        intake_groups[coords_key].append((cluster, risk))

    return intake_groups


def _add_stage1_constraints(
    m: linopy.Model,
    store_map: pd.DataFrame,
    intake_groups: dict[tuple[float, ...], list[tuple[int, str]]],
    intake_data: dict,
    store_e: xr.DataArray,
    solver_name: str,
    value_per_yll: float,
) -> dict[tuple[int, str], linopy.LinearExpression]:
    """Add Stage 1 constraints: store level → log(RR_{r,d}).

    Stage 1 transforms food group store levels into log relative risk values
    using piecewise-linear interpolation (SOS2).

    For each (cluster, risk) pair:
        1. Create λ_k variables for intake breakpoints
        2. Add convexity: Σ_k λ_k = 1
        3. Add intake balance: I_{c,r} = Σ_k λ_k x_k
        4. Compute log(RR_{r,d}) = Σ_k λ_k log(RR_{r,d}(x_k))

    Parameters
    ----------
    m
        The linopy model.
    store_map
        Store mapping from _build_store_to_cluster_map.
    intake_groups
        (cluster, risk) pairs grouped by intake coordinates.
    intake_data
        Breakpoint data from _build_intake_breakpoints.
    store_e
        Store level variables (food group stores).
    solver_name
        Solver name for SOS2 implementation selection.
    value_per_yll
        Value per YLL; if zero, skip degeneracy perturbation.

    Returns
    -------
    dict
        {(cluster, cause): Σ_r log(RR_{r,d})} expressions for Stage 2.
    """
    log_rr_totals: dict[tuple[int, str], linopy.LinearExpression] = {}

    for coords_key, group_pairs in intake_groups.items():
        # Get risk data for this group (all pairs share same breakpoints)
        risk = group_pairs[0][1]
        risk_table = intake_data[risk]
        intake_steps = risk_table["intake_steps"]

        # Create cluster-risk labels for vectorized operations
        cluster_risk_labels = [f"c{cluster}_r{risk}" for cluster, risk in group_pairs]
        cluster_risk_index = pd.Index(cluster_risk_labels, name="cluster_risk")

        # Create λ variables (SOS2-bounded interpolation weights)
        risk_label = str(risk)
        lambda_var = m.add_variables(
            lower=0,
            upper=1,
            coords=[cluster_risk_index, intake_steps],
            name=f"health_lambda_group_{next(_LAMBDA_GROUP_COUNTER)}_{risk_label}",
        )

        _register_auxiliary_variable(m, lambda_var.name)

        # Add small perturbation to break degeneracy (if health has cost)
        if value_per_yll > 0:
            rng = np.random.default_rng(seed=42)
            n_steps = len(intake_steps)
            perturbation = xr.DataArray(
                1e-10 * rng.uniform(size=n_steps),
                coords={"intake_step": intake_steps},
                dims=["intake_step"],
            )
            m.objective += (perturbation * lambda_var).sum()

        # Add SOS2 constraints for adjacency
        aux_names = _add_sos2_with_fallback(
            m, lambda_var, sos_dim="intake_step", solver_name=solver_name
        )
        for aux_name in aux_names:
            _register_auxiliary_variable(m, aux_name)

        # Convexity: Σ_k λ_k = 1
        m.add_constraints(lambda_var.sum("intake_step") == 1)

        # --- Intake Balance ---
        # Build mapping from cluster_risk labels to store levels
        group_map_df = pd.DataFrame(group_pairs, columns=["cluster", "risk_factor"])
        group_map_df["cluster_risk"] = cluster_risk_labels

        merged_stores = store_map.merge(
            group_map_df, on=["cluster", "risk_factor"], how="inner"
        )

        if not merged_stores.empty:
            store_names = merged_stores["store_name"].values
            coeffs = xr.DataArray(
                merged_stores["coeff"].values,
                coords={"name": store_names},
                dims="name",
            )
            # Grouper name="cluster_risk" ensures the groupby result dimension
            # matches intake_expr's dimension for proper constraint alignment
            grouper = xr.DataArray(
                merged_stores["cluster_risk"].values,
                coords={"name": store_names},
                dims="name",
                name="cluster_risk",
            )

            # LHS: Aggregated store level expression by cluster_risk
            # Each store is one country's food group; sum stores within cluster
            store_expr = (store_e.sel(name=store_names) * coeffs).groupby(grouper).sum()

            # RHS: Intake interpolation
            coeff_intake = risk_table["intake_values"]
            intake_expr = (lambda_var * coeff_intake).sum("intake_step")

            # IMPORTANT: Align coordinates before creating constraint.
            # The groupby operation may produce different coordinate ordering than
            # the lambda variable index. We must explicitly reindex to ensure
            # store_expr[cluster_risk] matches intake_expr[cluster_risk].
            intake_expr = intake_expr.reindex(
                cluster_risk=store_expr.data.coords["cluster_risk"]
            )

            # I_{c,r} = Σ_k λ_k x_k
            m.add_constraints(
                store_expr == intake_expr,
                name=f"health_intake_balance_group_{hash(coords_key)}",
            )

        # --- log(RR) Calculation ---
        # Collect log_rr matrices for all (cluster, risk) pairs in this group
        log_rr_frames = [intake_data[risk]["log_rr"] for _cluster, risk in group_pairs]

        if not log_rr_frames:
            continue

        # Concat along cluster_risk dimension
        combined_log_rr = pd.concat(
            log_rr_frames,
            keys=cluster_risk_index,
            names=["cluster_risk", "intake_step"],
        )

        # Check for missing log_rr values - these would silently zero out health effects
        if combined_log_rr.isna().any().any():
            missing = combined_log_rr[combined_log_rr.isna().any(axis=1)]
            raise ValueError(
                f"Missing log_rr values in risk breakpoints for {len(missing)} "
                f"(cluster_risk, intake_step) combinations; first few: "
                f"{list(missing.index[:5])}"
            )

        # Convert to DataArray: (cluster_risk, intake_step, cause)
        s_log = combined_log_rr.stack()
        s_log.index.names = ["cluster_risk", "intake_step", "cause"]
        da_log = xr.DataArray.from_series(s_log)

        # log(RR_{c,r,d}) = Σ_k λ_k log(RR_{r,d}(x_k))
        contrib = (lambda_var * da_log).sum("intake_step")

        # Accumulate by cluster (sum over risk factors for each cause)
        c_map = group_map_df.set_index("cluster_risk")["cluster"]
        present_cr = contrib.coords["cluster_risk"].values
        cluster_grouper = xr.DataArray(
            c_map.loc[present_cr].values,
            coords={"cluster_risk": present_cr},
            dims="cluster_risk",
            name="cluster",
        )

        # Σ_r log(RR_{r,d}) for each (cluster, cause)
        group_total = contrib.groupby(cluster_grouper).sum()

        # Store expressions for Stage 2
        causes = group_total.coords["cause"].values
        clusters = group_total.coords["cluster"].values

        for c in clusters:
            for cause in causes:
                expr = group_total.sel(cluster=c, cause=cause)
                key = (c, cause)

                if key in log_rr_totals:
                    log_rr_totals[key] = log_rr_totals[key] + expr
                else:
                    log_rr_totals[key] = expr

    return log_rr_totals


# =============================================================================
# Stage 2: log(RR) → YLL Store Level
# =============================================================================


def _build_cause_breakpoints(cause_log_breakpoints: pd.DataFrame) -> dict:
    """Build log-RR breakpoint grids by cause.

    Returns
    -------
    dict
        {cause: DataFrame with columns log_rr_total, rr_total}
    """
    return {
        cause: df.sort_values("log_rr_total")
        for cause, df in cause_log_breakpoints.groupby("cause")
    }


def _group_cluster_cause_pairs(
    cluster_cause_metadata: pd.DataFrame,
    cause_breakpoints: dict,
    cluster_population: dict[int, float],
) -> tuple[dict, dict]:
    """Group (cluster, cause) pairs by shared log-RR coordinate patterns.

    Computes absolute YLL from stored rates using planning-year population.

    Returns
    -------
    tuple
        (log_total_groups, cluster_cause_data) where:
        - log_total_groups: {coords_key: [(cluster, cause), ...]}
        - cluster_cause_data: {(cluster, cause): {yll_total, rr_ref, rr_baseline, cause_bp}}
    """
    log_total_groups: dict[tuple[float, ...], list[tuple[int, str]]] = defaultdict(list)
    cluster_cause_data: dict[tuple[int, str], dict] = {}

    for (cluster, cause), row in cluster_cause_metadata.iterrows():
        cluster = int(cluster)
        cause = str(cause)

        # Reconstruct absolute YLL from rate using planning-year population
        yll_rate_per_100k = float(row["yll_rate_per_100k"])
        pop = cluster_population[cluster]
        yll_total = (yll_rate_per_100k / constants.PER_100K) * pop

        cause_bp = cause_breakpoints.get(cause)
        if cause_bp is None:
            continue

        coords_key = tuple(cause_bp["log_rr_total"].values)
        if len(coords_key) == 1:
            raise ValueError(
                "Need at least two breakpoints for piecewise linear approximation"
            )

        log_total_groups[coords_key].append((cluster, cause))

        # Store metadata for constraint construction
        log_rr_total_ref = float(row["log_rr_total_ref"])
        log_rr_total_baseline = float(row["log_rr_total_baseline"])
        cluster_cause_data[(cluster, cause)] = {
            "yll_total": yll_total,
            "log_rr_total_ref": log_rr_total_ref,
            "rr_ref": math.exp(log_rr_total_ref),
            "rr_baseline": math.exp(log_rr_total_baseline),
            "cause_bp": cause_bp,
        }

    return log_total_groups, cluster_cause_data


def _add_stage2_constraints(
    m: linopy.Model,
    log_rr_totals: dict[tuple[int, str], linopy.LinearExpression],
    log_total_groups: dict[tuple[float, ...], list[tuple[int, str]]],
    cluster_cause_data: dict[tuple[int, str], dict],
    health_stores: pd.DataFrame,
    store_e: xr.DataArray,
    solver_name: str,
    value_per_yll: float,
) -> int:
    """Add Stage 2 constraints: Σ_r log(RR_{r,d}) → YLL store level.

    Stage 2 transforms the summed log-RR values into YLL store levels using
    a second piecewise-linear interpolation to compute exp(·).

    The store level represents the health cost normalized by V (value per YLL):

        e_{c,d} = (RR_d - RR_d^ref) * (YLL_{c,d} / RR_d^base) * 10^{-6}

    Parameters
    ----------
    m
        The linopy model.
    log_rr_totals
        {(cluster, cause): Σ_r log(RR_{r,d})} from Stage 1.
    log_total_groups
        (cluster, cause) pairs grouped by log-RR coordinates.
    cluster_cause_data
        Metadata for each (cluster, cause) pair.
    health_stores
        DataFrame of YLL stores indexed by (health_cluster, cause).
    store_e
        Store energy level variables.
    solver_name
        Solver name for SOS2 implementation selection.
    value_per_yll
        Value per YLL; determines constraint type (equality vs inequality).

    Returns
    -------
    int
        Number of store level constraints added.
    """
    constraints_added = 0

    for coords_key, cluster_cause_pairs in log_total_groups.items():
        log_total_vals = np.asarray(coords_key, dtype=float)
        log_total_steps = pd.Index(range(len(log_total_vals)), name="log_total_step")

        # Create cluster-cause labels for vectorized operations
        cluster_cause_labels = [
            f"c{cluster}_cause{cause}" for cluster, cause in cluster_cause_pairs
        ]
        cluster_cause_index = pd.Index(cluster_cause_labels, name="cluster_cause")

        # Create λ variables for log-RR interpolation
        cause_label = str(cluster_cause_pairs[0][1])
        lambda_var = m.add_variables(
            lower=0,
            upper=1,
            coords=[cluster_cause_index, log_total_steps],
            name=f"health_lambda_total_group_{next(_TOTAL_GROUP_COUNTER)}_{cause_label}",
        )

        _register_auxiliary_variable(m, lambda_var.name)

        # Add small perturbation to break degeneracy (if health has cost)
        if value_per_yll > 0:
            rng_total = np.random.default_rng(seed=43)
            n_total_steps = len(log_total_steps)
            perturbation_total = xr.DataArray(
                1e-10 * rng_total.uniform(size=n_total_steps),
                coords={"log_total_step": log_total_steps},
                dims=["log_total_step"],
            )
            m.objective += (perturbation_total * lambda_var).sum()

        # Add SOS2 constraints
        aux_names = _add_sos2_with_fallback(
            m, lambda_var, sos_dim="log_total_step", solver_name=solver_name
        )
        for aux_name in aux_names:
            _register_auxiliary_variable(m, aux_name)

        # Convexity: Σ_k λ_k = 1
        m.add_constraints(lambda_var.sum("log_total_step") == 1)

        # Coefficient for log-RR interpolation
        coeff_log_total = xr.DataArray(
            log_total_vals,
            coords={"log_total_step": log_total_steps},
            dims=["log_total_step"],
        )

        # Process each (cluster, cause) pair
        for (cluster, cause), label in zip(cluster_cause_pairs, cluster_cause_labels):
            data = cluster_cause_data[(cluster, cause)]
            lambda_total = lambda_var.sel(cluster_cause=label)

            # Verify Stage 1 produced log_rr totals for this (cluster, cause) pair
            if (cluster, cause) not in log_rr_totals:
                raise ValueError(
                    f"No log_rr total from Stage 1 for cluster {cluster}, cause {cause}. "
                    f"This indicates no risk factors were processed for this cluster. "
                    f"Check that food group stores exist and map to health clusters."
                )
            total_expr = log_rr_totals[(cluster, cause)]
            cause_bp = data["cause_bp"]

            # log(RR_d) interpolation: Σ_k λ_k z_k = Σ_r log(RR_{r,d})
            log_interp = m.linexpr((coeff_log_total, lambda_total)).sum(
                "log_total_step"
            )
            m.add_constraints(
                log_interp == total_expr,
                name=f"health_total_balance_c{cluster}_cause{cause}",
            )

            # RR_d interpolation: Σ_k λ_k exp(z_k)
            coeff_rr = xr.DataArray(
                cause_bp["rr_total"].values,
                coords={"log_total_step": log_total_steps},
                dims=["log_total_step"],
            )
            rr_interp = m.linexpr((coeff_rr, lambda_total)).sum("log_total_step")

            # Get store name
            if (cluster, cause) not in health_stores.index:
                raise ValueError(
                    f"No YLL store found for cluster {cluster}, cause {cause}. "
                    f"Check that health stores were created during model build."
                )
            store_name = health_stores.loc[(cluster, cause), "name"]

            if data["yll_total"] <= 0:
                logger.warning(
                    "Health store has non-positive yll_total (cluster=%d, cause=%s); "
                    "constraint will be non-binding",
                    cluster,
                    cause,
                )

            # Store level = (RR - RR^ref) * (YLL / RR^base) * 10^{-6}
            #
            # Health cost is zero at TMREL (where RR = RR^ref) and increases
            # with deviation from optimal intake. Since TMREL minimizes RR,
            # we have RR ≥ RR^ref always, so store levels are non-negative.
            #
            # Normalizing by RR^base ensures populations with identical
            # underlying susceptibility incur the same health cost for the
            # same diet, regardless of baseline consumption patterns.
            yll_expr_myll = (
                (rr_interp - data["rr_ref"])
                * (data["yll_total"] / data["rr_baseline"])
                * constants.YLL_TO_MILLION_YLL
            )

            # Constraint type depends on value_per_yll:
            # - If zero: use equality (store level tracks YLL exactly)
            # - If positive: use inequality (objective minimizes store level)
            if value_per_yll == 0:
                m.add_constraints(
                    store_e.sel(name=store_name) == yll_expr_myll,
                    name=f"health_store_level_c{cluster}_cause{cause}",
                )
            elif value_per_yll > 0:
                m.add_constraints(
                    store_e.sel(name=store_name) >= yll_expr_myll,
                    name=f"health_store_level_c{cluster}_cause{cause}",
                )
            constraints_added += 1

    return constraints_added


# =============================================================================
# Main Entry Point
# =============================================================================


def add_health_objective(
    n: pypsa.Network,
    risk_breakpoints_path: str,
    cluster_cause_path: str,
    cause_log_path: str,
    cluster_summary_path: str,
    clusters_path: str,
    risk_factors: list[str],
    risk_cause_map: dict[str, list[str]],
    solver_name: str,
    value_per_yll: float,
) -> None:
    """Add health cost constraints to the optimization model.

    This implements the health cost formulation from docs/health.rst:

        Cost_{c,d}(x) = V * (YLL_{c,d} / RR_d(x^base)) * (RR_d(x) - RR_d^ref)

    where:
        - V = value_per_yll (USD per year of life lost)
        - YLL_{c,d} = baseline years of life lost for cluster c, disease d
        - RR_d(x) = relative risk at intake x (product over risk factors)
        - RR_d^ref = RR at TMREL (theoretical minimum risk exposure level)
        - x^base = baseline intake

    The implementation uses two-stage SOS2 interpolation to handle the
    nonlinear multiplicative combination of relative risks:

        Stage 1: Intake x_r → log(RR_{r,d})
        Stage 2: Σ_r log(RR_{r,d}) → RR_d → YLL store level

    Parameters
    ----------
    n
        The PyPSA network with health stores already added. Population data
        for health clusters is read from the network metadata.
    risk_breakpoints_path
        Path to CSV with (risk_factor, intake_g_per_day, cause, log_rr).
    cluster_cause_path
        Path to CSV with (health_cluster, cause, yll_total, log_rr_total_ref,
        log_rr_total_baseline).
    cause_log_path
        Path to CSV with (cause, log_rr_total, rr_total) breakpoints.
    cluster_summary_path
        Path to CSV with cluster metadata.
    clusters_path
        Path to CSV mapping countries to health clusters.
    risk_factors
        List of risk factors to include (e.g., ['fruits', 'vegetables', ...]).
    risk_cause_map
        Mapping from risk factor to list of affected causes.
    solver_name
        Solver name ('gurobi', 'highs', etc.).
    value_per_yll
        Monetary value per year of life lost (USD).
    """
    m = n.model

    # --- Load Data ---
    data = _load_health_data(
        n,
        risk_breakpoints_path,
        cluster_cause_path,
        cause_log_path,
        cluster_summary_path,
        clusters_path,
    )

    risk_breakpoints = data["risk_breakpoints"]
    cause_log_breakpoints = data["cause_log_breakpoints"]
    cluster_cause_metadata = data["cluster_cause_metadata"]
    cluster_lookup = data["cluster_lookup"]
    cluster_population = data["cluster_population"]

    logger.info(
        "Health data: %d risk breakpoints across %d risks / %d causes; %d cause breakpoints",
        len(risk_breakpoints),
        risk_breakpoints["risk_factor"].nunique(),
        risk_breakpoints["cause"].nunique(),
        len(cause_log_breakpoints),
    )

    # --- Validate Risk-Cause Pairs ---
    available_risks = set(risk_breakpoints["risk_factor"].unique())
    risk_cause_map = {
        r: causes for r, causes in risk_cause_map.items() if r in available_risks
    }

    allowed_pairs = {(r, c) for r, causes in risk_cause_map.items() for c in causes}
    rb_pairs = set(zip(risk_breakpoints["risk_factor"], risk_breakpoints["cause"]))
    missing_pairs = sorted(allowed_pairs - rb_pairs)
    if missing_pairs:
        text = ", ".join([f"{r}:{c}" for r, c in missing_pairs])
        raise ValueError(f"Risk breakpoints missing required pairs: {text}")

    # --- Build Store Map ---
    # Map food group stores to health clusters with per-capita coefficients.
    # Using store levels instead of consumption link flows ensures the health
    # constraints operate on the same variables as food_group_equal_ constraints.
    store_e = m.variables["Store-e"].sel(snapshot="now")

    store_map = _build_store_to_cluster_map(
        n.stores.static,
        risk_factors,
        cluster_lookup,
        cluster_population,
    )

    if store_map.empty:
        logger.info("No food group stores map to health risk factors; skipping")
        return

    logger.info(
        "Health intake mapping: %d stores -> %d cluster-risk pairs across %d clusters",
        len(store_map),
        len(store_map[["cluster", "risk_factor"]].drop_duplicates()),
        store_map["cluster"].nunique(),
    )

    # --- Stage 1: Store Level → log(RR) ---
    intake_data = _build_intake_breakpoints(risk_breakpoints)
    intake_groups = _group_cluster_risk_pairs(store_map, intake_data)

    log_rr_totals = _add_stage1_constraints(
        m, store_map, intake_groups, intake_data, store_e, solver_name, value_per_yll
    )

    # --- Stage 2: log(RR) → YLL Store ---
    cause_breakpoints = _build_cause_breakpoints(cause_log_breakpoints)
    log_total_groups, cluster_cause_data = _group_cluster_cause_pairs(
        cluster_cause_metadata, cause_breakpoints, cluster_population
    )

    logger.info(
        "Health risk aggregation: %d (cluster, cause) pairs grouped into %d log-RR grids",
        len(cluster_cause_data),
        len(log_total_groups),
    )

    # Get health store mapping (store_e already loaded above for Stage 1)
    health_stores = (
        n.stores.static[
            n.stores.static["carrier"].notna()
            & n.stores.static["carrier"].str.startswith("yll_")
        ]
        .reset_index()
        .set_index(["health_cluster", "cause"])
    )

    constraints_added = _add_stage2_constraints(
        m,
        log_rr_totals,
        log_total_groups,
        cluster_cause_data,
        health_stores,
        store_e,
        solver_name,
        value_per_yll,
    )

    logger.info("Added %d health store level constraints", constraints_added)
