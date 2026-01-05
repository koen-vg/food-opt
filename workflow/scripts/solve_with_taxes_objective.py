# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Solve model with optimal taxes/subsidies applied in the objective.

This script:
1. Loads the production-cost-only network (Stage 2 build)
2. Applies optimal taxes/subsidies as marginal costs on food group stores
3. Solves without fixed consumption constraints

The resulting consumption should match the health/GHG-optimal diet if the
taxes/subsidies were computed consistently.
"""

from logging_config import setup_script_logging
import pandas as pd
import pypsa

# Enable new PyPSA components API
pypsa.options.api.new_components_api = True


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


def add_taxes_to_objective(n: pypsa.Network, taxes_path: str) -> None:
    """Add taxes/subsidies to the objective via store marginal costs."""
    taxes_df = pd.read_csv(taxes_path)
    taxes_df["country"] = taxes_df["country"].astype(str).str.upper()
    taxes_df["store_name"] = "store_" + taxes_df["group"] + "_" + taxes_df["country"]

    store_index = n.stores.static.index
    if "marginal_cost_storage" in n.stores.static.columns:
        base_costs = n.stores.static["marginal_cost_storage"].copy()
        base_costs = base_costs.fillna(0.0)
    else:
        base_costs = pd.Series(0.0, index=store_index)
        n.stores.static["marginal_cost_storage"] = 0.0

    applied = 0
    for _, row in taxes_df.iterrows():
        store_name = row["store_name"]
        if store_name not in store_index:
            logger.warning("No store found for tax entry %s", store_name)
            continue
        base_costs.loc[store_name] = base_costs.loc[store_name] + float(
            row["tax_bnusd_per_mt"]
        )
        applied += 1

    n.stores.static["marginal_cost_storage"] = base_costs
    logger.info("Applied taxes to %d food group stores", applied)


if __name__ == "__main__":
    logger = setup_script_logging(
        log_file=snakemake.log[0] if snakemake.log else None  # type: ignore[name-defined]
    )

    logger.info("Loading network from %s", snakemake.input.network)
    n = pypsa.Network(snakemake.input.network)

    add_taxes_to_objective(n, snakemake.input.taxes)

    logger.info("Creating linopy model...")
    n.optimize.create_model()
    logger.info("Linopy model created.")

    solver_name = snakemake.params.solver
    solver_threads = snakemake.params.solver_threads
    solver_options = _apply_solver_threads_option(
        dict(snakemake.params.solver_options or {}),
        solver_name,
        solver_threads,
    )
    io_api = snakemake.params.io_api
    netcdf_compression = snakemake.params.netcdf_compression

    if solver_name.lower() == "gurobi" and snakemake.log:
        if "LogFile" not in solver_options:
            solver_options["LogFile"] = snakemake.log[0]
        if "LogToConsole" not in solver_options:
            solver_options["LogToConsole"] = 1

    status, condition = n.model.solve(
        solver_name=solver_name,
        io_api=io_api,
        **solver_options,
    )

    if status == "ok":
        n.optimize.assign_solution()
        n.optimize.post_processing()
        n.export_to_netcdf(
            snakemake.output.network,
            compression=netcdf_compression,
        )
        logger.info("Solved successfully, saved to %s", snakemake.output.network)
    else:
        logger.error("Optimization failed: status=%s, condition=%s", status, condition)
        raise RuntimeError(f"Optimization failed: {status}, {condition}")
