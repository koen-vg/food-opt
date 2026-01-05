# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Solve model with fixed consumption constraints for tax extraction.

This script:
1. Loads the optimal consumption from Stage 1 (optimize scenario)
2. Adds equality constraints on food group stores
3. Solves with production costs only (no health/GHG objectives)
4. Exports the solved network with dual variables for tax extraction

The dual variables of the consumption constraints represent the optimal
Pigouvian taxes/subsidies needed to incentivize the health/GHG-optimal diet.
"""

from logging_config import setup_script_logging
import pandas as pd
import pypsa
import xarray as xr

# Enable new PyPSA components API
pypsa.options.api.new_components_api = True

GRAMS_PER_MEGATONNE = 1e12
DAYS_PER_YEAR = 365


def _per_capita_mass_to_mt_per_year(g_per_day: float, population: float) -> float:
    """Convert per-capita g/day to annual Mt."""
    return g_per_day * population * DAYS_PER_YEAR / GRAMS_PER_MEGATONNE


def add_fixed_consumption_constraints(
    n: pypsa.Network,
    consumption_df: pd.DataFrame,
    population_map: dict[str, float],
) -> None:
    """Add equality constraints fixing food group consumption levels.

    Parameters
    ----------
    n : pypsa.Network
        Network with linopy model attached.
    consumption_df : pd.DataFrame
        Columns: group, country, consumption_g_per_day
    population_map : dict[str, float]
        Population by country ISO3 code.
    """
    m = n.model
    store_e = m.variables["Store-e"].sel(snapshot="now")
    stores_df = n.stores.static

    # Group by food group
    for group, group_df in consumption_df.groupby("group"):
        group_stores = stores_df[stores_df["carrier"] == f"group_{group}"]
        if group_stores.empty:
            logger.warning("No stores found for group %s", group)
            continue

        countries = group_stores["country"].astype(str).str.upper()
        lhs = store_e.sel(name=group_stores.index)

        # Build RHS from per-country consumption values
        rhs_vals = []
        for country in countries:
            pop = population_map.get(country, 0.0)
            row = group_df[group_df["country"] == country]
            if row.empty:
                g_per_day = 0.0
            else:
                g_per_day = float(row["consumption_g_per_day"].iloc[0])
            rhs_vals.append(_per_capita_mass_to_mt_per_year(g_per_day, pop))

        rhs = xr.DataArray(rhs_vals, coords={"name": group_stores.index}, dims="name")

        constr_name = f"food_group_equal_{group}"
        m.add_constraints(lhs == rhs, name=f"GlobalConstraint-{constr_name}")
        n.global_constraints.add(
            f"{constr_name}_" + group_stores.index,
            sense="==",
            constant=rhs.values,
            type="nutrition",
        )

    logger.info(
        "Added fixed consumption constraints for %d food groups",
        consumption_df["group"].nunique(),
    )


def _apply_solver_threads_option(
    solver_options: dict, solver_name: str, threads: int
) -> dict:
    """Add thread limit to solver options."""
    if threads <= 0:
        return solver_options

    solver_options = dict(solver_options)
    name_lower = solver_name.lower()

    if name_lower == "gurobi":
        solver_options.setdefault("Threads", threads)
    elif name_lower == "highs" or name_lower == "cplex":
        solver_options.setdefault("threads", threads)

    return solver_options


if __name__ == "__main__":
    logger = setup_script_logging(
        log_file=snakemake.log[0] if snakemake.log else None  # type: ignore[name-defined]
    )

    # Load network
    logger.info("Loading network from %s", snakemake.input.network)
    n = pypsa.Network(snakemake.input.network)

    # Create the linopy model
    logger.info("Creating linopy model...")
    n.optimize.create_model()
    logger.info("Linopy model created.")

    # Load optimal consumption
    consumption_df = pd.read_csv(snakemake.input.optimal_consumption)
    consumption_df["country"] = consumption_df["country"].astype(str).str.upper()

    # Load population
    population_df = pd.read_csv(snakemake.input.population)
    population_df["iso3"] = population_df["iso3"].astype(str).str.upper()
    population_map = (
        population_df.set_index("iso3")["population"].astype(float).to_dict()
    )

    # Add fixed consumption constraints
    add_fixed_consumption_constraints(n, consumption_df, population_map)

    # Solver setup
    solver_name = snakemake.params.solver
    solver_threads = snakemake.params.solver_threads
    solver_options = _apply_solver_threads_option(
        dict(snakemake.params.solver_options or {}),
        solver_name,
        solver_threads,
    )
    io_api = snakemake.params.io_api
    netcdf_compression = snakemake.params.netcdf_compression

    # Configure Gurobi to write detailed logs
    if solver_name.lower() == "gurobi" and snakemake.log:
        if "LogFile" not in solver_options:
            solver_options["LogFile"] = snakemake.log[0]
        if "LogToConsole" not in solver_options:
            solver_options["LogToConsole"] = 1

    # Solve
    status, condition = n.model.solve(
        solver_name=solver_name,
        io_api=io_api,
        calculate_fixed_duals=True,  # Required for dual extraction
        **solver_options,
    )

    if status == "ok":
        n.optimize.assign_solution()
        n.optimize.assign_duals(False)
        n.optimize.post_processing()
        n.export_to_netcdf(
            snakemake.output.network,
            compression=netcdf_compression,
        )
        logger.info("Solved successfully, saved to %s", snakemake.output.network)
    else:
        logger.error("Optimization failed: status=%s, condition=%s", status, condition)
        raise RuntimeError(f"Optimization failed: {status}, {condition}")
