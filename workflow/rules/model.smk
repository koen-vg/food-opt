# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Model building and solving rules.

Includes the main optimization model construction and solution rules,
along with helper functions for gathering input files.
"""

import itertools


def yield_inputs(wildcards):
    """Get all crop yield files for model building."""
    irr_cfg = config["irrigation"]["irrigated_crops"]
    if irr_cfg == "all":
        irrigated_crops = config["crops"]
    else:
        irrigated_crops = list(irr_cfg)

    return {
        f"{crop}_yield_{water_supply}": f"processing/{{name}}/crop_yields/{crop}_{water_supply}.csv"
        for crop, water_supply in (
            list(zip(config["crops"], itertools.repeat("r")))  # Rainfed
            + list(zip(irrigated_crops, itertools.repeat("i")))
        )
    }


def harvested_area_model_inputs(_wildcards):
    """Return harvested area files when actual production mode is enabled."""

    if not config["validation"]["use_actual_production"]:
        return {}

    irr_cfg = config["irrigation"]["irrigated_crops"]
    if irr_cfg == "all":
        irrigated_crops = config["crops"]
    else:
        irrigated_crops = list(irr_cfg)

    dataset = config["validation"]["harvest_area_source"]
    inputs = {
        f"{crop}_harvested_r": f"processing/{{name}}/harvested_area/{dataset}/{crop}_r.csv"
        for crop in config["crops"]
    }
    for crop in irrigated_crops:
        inputs[f"{crop}_harvested_i"] = (
            f"processing/{{name}}/harvested_area/{dataset}/{crop}_i.csv"
        )
    return inputs


rule build_model:
    input:
        unpack(yield_inputs),
        unpack(residue_yield_inputs),
        unpack(harvested_area_model_inputs),
        fertilizer_n_rates="processing/{name}/global_fertilizer_n_rates.csv",
        foods="data/foods.csv",
        moisture_content="data/crop_moisture_content.csv",
        ruminant_feed_categories="processing/{name}/ruminant_feed_categories.csv",
        ruminant_feed_mapping="processing/{name}/ruminant_feed_mapping.csv",
        monogastric_feed_categories="processing/{name}/monogastric_feed_categories.csv",
        monogastric_feed_mapping="processing/{name}/monogastric_feed_mapping.csv",
        feed_to_products="processing/{name}/feed_to_animal_products.csv",
        manure_ch4_emissions="processing/{name}/manure_ch4_emission_factors.csv",
        food_groups="data/food_groups.csv",
        nutrition="data/nutrition.csv",
        regions="processing/{name}/regions.geojson",
        land_area_by_class="processing/{name}/land_area_by_class.csv",
        cropland_baseline="processing/{name}/cropland_baseline_by_class.csv",
        multi_cropping_area="processing/{name}/multi_cropping/eligible_area.csv",
        multi_cropping_yields="processing/{name}/multi_cropping/cycle_yields.csv",
        edible_portion="processing/{name}/fao_edible_portion.csv",
        population="processing/{name}/population.csv",
        baseline_diet="processing/{name}/dietary_intake.csv",
        food_loss_waste="processing/{name}/food_loss_waste.csv",
        costs="processing/{name}/crop_costs.csv",
        animal_costs="processing/{name}/animal_costs.csv",
        grassland_yields="processing/{name}/grassland_yields.csv",
        monthly_region_water="processing/{name}/water/monthly_region_water.csv",
        growing_season_water="processing/{name}/water/region_growing_season_water.csv",
        blue_water_availability="processing/{name}/water/blue_water_availability.csv",
        luc_carbon_coefficients="processing/{name}/luc/luc_carbon_coefficients.csv",
        current_grassland_area="processing/{name}/luc/current_grassland_area_by_class.csv",
        grazing_only_land="processing/{name}/land_grazing_only_by_class.csv",
        health_cluster_summary="processing/{name}/health/cluster_summary.csv",
        health_cluster_cause="processing/{name}/health/cluster_cause_baseline.csv",
        build_scripts=expand(
            "workflow/scripts/build_model/{script}",
            script=[
                "animals.py",
                "biomass.py",
                "health.py",
                "constants.py",
                "crops.py",
                "food.py",
                "grassland.py",
                "infrastructure.py",
                "land.py",
                "nutrition.py",
                "primary_resources.py",
                "trade.py",
                "utils.py",
            ],
        ),
    params:
        crops=config["crops"],
        multiple_cropping=config["multiple_cropping"],
        countries=config["countries"],
        land=config["land"],
        fertilizer=config["fertilizer"],
        residues=config["residues"],
        biomass=config["biomass"],
        emissions=config["emissions"],
        food_groups=config["food_groups"]["included"],
        food_group_constraints=config["food_groups"].get("constraints", {}),
        macronutrients=config["macronutrients"],
        diet=config["diet"],
        byproducts=config["byproducts"],
        animal_products=config["animal_products"],
        trade=config["trade"],
        grazing=grazing_cfg,
        health_reference_year=config["health"]["reference_year"],
    output:
        network="results/{name}/build/model_scen-{scenario}.nc",
    log:
        "logs/{name}/build_model_scen-{scenario}.log",
    script:
        "../scripts/build_model.py"


def solve_model_inputs(w):
    """Get input files for solve_model rule.

    Includes validation-specific inputs (e.g., FAO animal production data)
    only when validation mode is enabled.
    """
    inputs = {
        "network": f"results/{w.name}/build/model_scen-{w.scenario}.nc",
        "health_risk_breakpoints": f"processing/{w.name}/health/risk_breakpoints.csv",
        "health_cluster_cause": f"processing/{w.name}/health/cluster_cause_baseline.csv",
        "health_cause_log": f"processing/{w.name}/health/cause_log_breakpoints.csv",
        "health_cluster_summary": f"processing/{w.name}/health/cluster_summary.csv",
        "health_clusters": f"processing/{w.name}/health/country_clusters.csv",
        "population": f"processing/{w.name}/population.csv",
        "food_groups": "data/food_groups.csv",
        "baseline_diet": f"processing/{w.name}/dietary_intake.csv",
    }

    # Add validation-specific inputs
    if config.get("validation", {}).get("use_actual_production", False):
        inputs["animal_production"] = (
            f"processing/{w.name}/faostat_animal_production.csv"
        )
        inputs["food_loss_waste"] = f"processing/{w.name}/food_loss_waste.csv"

    # Add production stability inputs
    stability_cfg = config["validation"]["production_stability"]
    if stability_cfg["enabled"]:
        if stability_cfg["crops"]["enabled"]:
            inputs["crop_production_baseline"] = (
                f"processing/{w.name}/faostat_crop_production.csv"
            )
        if stability_cfg["animals"]["enabled"]:
            inputs["animal_production_baseline"] = (
                f"processing/{w.name}/faostat_animal_production.csv"
            )
            inputs["food_loss_waste"] = f"processing/{w.name}/food_loss_waste.csv"

    return inputs


def get_solver_threads(cfg: dict) -> int:
    """Return configured solver threads as an int."""

    return int(cfg["solving"]["threads"])


def solver_options_with_threads(cfg: dict) -> dict:
    """Return solver options with a threads override applied when configured."""

    solver_name = cfg["solving"]["solver"]
    options = cfg["solving"].get(f"options_{solver_name}", {}) or {}
    threads = get_solver_threads(cfg)

    options = dict(options)
    solver_key = solver_name.lower()
    if solver_key == "gurobi":
        options["Threads"] = threads
    elif solver_key == "highs":
        options["threads"] = threads

    return options


rule solve_model:
    input:
        unpack(solve_model_inputs),
    threads: get_solver_threads(config)
    params:
        health_risk_factors=config["health"]["risk_factors"],
        health_risk_cause_map=config["health"]["risk_cause_map"],
        ghg_price=config["emissions"]["ghg_price"],
        solver=config["solving"]["solver"],
        solver_threads=get_solver_threads(config),
        solver_options=solver_options_with_threads(config),
        io_api=config["solving"]["io_api"],
        netcdf_compression=config["solving"].get("netcdf_compression"),
        macronutrients=config["macronutrients"],
        food_group_constraints=config["food_groups"].get("constraints", {}),
        diet=config["diet"],
        enforce_baseline=config["validation"]["enforce_gdd_baseline"],
        production_stability=config["validation"]["production_stability"],
    output:
        network="results/{name}/solved/model_scen-{scenario}.nc",
    log:
        "logs/{name}/solve_model_scen-{scenario}.log",
    script:
        "../scripts/solve_model.py"
