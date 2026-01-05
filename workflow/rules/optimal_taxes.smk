# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Rules for optimal taxes/subsidies workflow.

This workflow:
1. Runs optimization with health/GHG objectives to derive optimal consumption
2. Extracts consumption patterns from the optimized model
3. Fixes consumption to optimal values and solves with production costs only
4. Extracts dual variables as optimal Pigouvian taxes/subsidies
"""

plotting_cfg = config.get("plotting", {})


rule extract_optimal_consumption:
    """Extract consumption by food group from optimized model (Stage 1 output).

    Consumption is extracted from food group stores in the solved network,
    converted to per-capita g/day format for use as constraints in Stage 2.
    """
    input:
        network="results/{name}/solved/model_scen-optimize.nc",
        population="processing/{name}/population.csv",
        food_groups="data/food_groups.csv",
    output:
        consumption="results/{name}/optimal_taxes/optimal_consumption.csv",
    log:
        "logs/{name}/extract_optimal_consumption.log",
    script:
        "../scripts/extract_optimal_consumption.py"


rule solve_with_fixed_consumption:
    """Solve model with fixed consumption to extract tax dual variables (Stage 2).

    This rule:
    1. Loads optimal consumption from Stage 1
    2. Adds equality constraints on food group stores
    3. Solves with production costs only (no health/GHG objectives)
    4. The dual variables of consumption constraints represent optimal taxes
    """
    input:
        network="results/{name}/build/model_scen-extract_taxes.nc",
        optimal_consumption="results/{name}/optimal_taxes/optimal_consumption.csv",
        population="processing/{name}/population.csv",
    params:
        solver=config["solving"]["solver"],
        solver_threads=config["solving"]["threads"],
        solver_options=solver_options_with_threads(config),
        io_api=config["solving"]["io_api"],
        netcdf_compression=config["solving"].get("netcdf_compression"),
    threads: config["solving"]["threads"]
    output:
        network="results/{name}/optimal_taxes/solved_fixed_consumption.nc",
    log:
        "logs/{name}/solve_with_fixed_consumption.log",
    script:
        "../scripts/solve_with_fixed_consumption.py"


rule extract_optimal_taxes:
    """Extract optimal taxes/subsidies from consumption constraint duals.

    Taxes are the dual variables (shadow prices) of the food group
    equality constraints from Stage 2, representing the Pigouvian
    tax/subsidy needed to incentivize optimal consumption.
    """
    input:
        network="results/{name}/optimal_taxes/solved_fixed_consumption.nc",
    output:
        taxes="results/{name}/optimal_taxes/taxes.csv",
    log:
        "logs/{name}/extract_optimal_taxes.log",
    script:
        "../scripts/extract_optimal_taxes.py"


rule solve_with_taxes_objective:
    """Solve with taxes/subsidies applied to the objective (no fixed diet)."""
    input:
        network="results/{name}/build/model_scen-extract_taxes.nc",
        taxes="results/{name}/optimal_taxes/taxes.csv",
    params:
        solver=config["solving"]["solver"],
        solver_threads=config["solving"]["threads"],
        solver_options=solver_options_with_threads(config),
        io_api=config["solving"]["io_api"],
        netcdf_compression=config["solving"].get("netcdf_compression"),
    threads: config["solving"]["threads"]
    output:
        network="results/{name}/optimal_taxes/solved_with_taxes.nc",
    log:
        "logs/{name}/solve_with_taxes_objective.log",
    script:
        "../scripts/solve_with_taxes_objective.py"


rule plot_optimal_taxes:
    """Visualize optimal taxes/subsidies by food group."""
    input:
        taxes="results/{name}/optimal_taxes/taxes.csv",
    output:
        taxes_pdf="results/{name}/plots/optimal_taxes/taxes_by_food_group.pdf",
        taxes_csv="results/{name}/plots/optimal_taxes/taxes_by_food_group.csv",
    params:
        group_colors=plotting_cfg.get("colors", {}).get("food_groups", {}),
    log:
        "logs/{name}/plot_optimal_taxes.log",
    script:
        "../scripts/plotting/plot_optimal_taxes.py"


rule plot_optimal_taxes_diet_comparison:
    """Compare global average diet between the two optimal taxes optimizations."""
    input:
        networks=[
            "results/{name}/solved/model_scen-optimize.nc",
            "results/{name}/optimal_taxes/solved_fixed_consumption.nc",
            "results/{name}/optimal_taxes/solved_with_taxes.nc",
        ],
        population="processing/{name}/population.csv",
        food_groups="data/food_groups.csv",
    output:
        pdf="results/{name}/plots/optimal_taxes/diet_comparison.pdf",
        csv="results/{name}/plots/optimal_taxes/diet_comparison.csv",
    params:
        wildcards=[
            "Health/GHG optimized",
            "Fixed consumption (costs only)",
            "Taxes in objective",
        ],
        group_colors=plotting_cfg.get("colors", {}).get("food_groups", {}),
    log:
        "logs/{name}/plot_optimal_taxes_diet_comparison.log",
    script:
        "../scripts/plotting/plot_food_consumption_comparison.py"
