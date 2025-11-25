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
        build_scripts=expand(
            "workflow/scripts/build_model/{script}",
            script=[
                "animals.py",
                "biomass.py",
                "constants.py",
                "crops.py",
                "food.py",
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
        network="results/{name}/build/model.nc",
    log:
        "logs/{name}/build_model.log",
    script:
        "../scripts/build_model.py"


def solve_model_inputs(w):
    """Get input files for solve_model rule.

    Includes validation-specific inputs (e.g., FAO animal production data)
    only when validation mode is enabled.
    """
    inputs = {
        "network": f"results/{w.name}/build/model.nc",
        "health_risk_breakpoints": f"processing/{w.name}/health/risk_breakpoints.csv",
        "health_cluster_cause": f"processing/{w.name}/health/cluster_cause_baseline.csv",
        "health_cause_log": f"processing/{w.name}/health/cause_log_breakpoints.csv",
        "health_cluster_summary": f"processing/{w.name}/health/cluster_summary.csv",
        "health_clusters": f"processing/{w.name}/health/country_clusters.csv",
        "population": f"processing/{w.name}/population.csv",
        "food_groups": "data/food_groups.csv",
    }

    # Add validation-specific inputs
    if config.get("validation", {}).get("use_actual_production", False):
        inputs["animal_production"] = (
            f"processing/{w.name}/faostat_animal_production.csv"
        )

    return inputs


rule solve_model:
    input:
        unpack(solve_model_inputs),
    params:
        health_risk_factors=config["health"]["risk_factors"],
        health_value_per_yll=config["health"]["value_per_yll"],
        ghg_price=config["emissions"]["ghg_price"],
        solver=config["solving"]["solver"],
        solver_options=config["solving"].get(
            f"options_{config['solving']['solver']}", {}
        ),
        io_api=config["solving"]["io_api"],
        netcdf_compression=config["solving"].get("netcdf_compression"),
    output:
        network="results/{name}/solved/model_obj-{objective}.nc",
    log:
        "logs/{name}/solve_model_obj-{objective}.log",
    script:
        "../scripts/solve_model.py"
