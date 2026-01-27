# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later


rule extract_ghg_intensity:
    """Extract GHG intensity by food and country."""
    input:
        network="results/{name}/solved/model_scen-{scenario}.nc",
        food_groups="data/food_groups.csv",
        food_consumption="results/{name}/analysis/scen-{scenario}/food_consumption.csv",
    params:
        ghg_price=lambda w: get_effective_config(w.scenario)["emissions"]["ghg_price"],
        ch4_gwp=config["emissions"]["ch4_to_co2_factor"],
        n2o_gwp=config["emissions"]["n2o_to_co2_factor"],
    output:
        csv="results/{name}/analysis/scen-{scenario}/ghg_intensity.csv",
    log:
        "logs/{name}/extract_ghg_intensity_scen-{scenario}.log",
    script:
        "../scripts/analysis/extract_ghg_intensity.py"


rule extract_health_impacts:
    """Extract health impacts (marginal YLL) by food group and country."""
    input:
        food_group_consumption="results/{name}/analysis/scen-{scenario}/food_group_consumption.csv",
        risk_breakpoints="processing/{name}/health/scen-{scenario}/risk_breakpoints.csv",
        health_cluster_cause="processing/{name}/health/scen-{scenario}/cluster_cause_baseline.csv",
        health_cause_log="processing/{name}/health/scen-{scenario}/cause_log_breakpoints.csv",
        health_clusters="processing/{name}/health/scen-{scenario}/country_clusters.csv",
        population="processing/{name}/population.csv",
    params:
        value_per_yll=lambda w: get_effective_config(w.scenario)["health"][
            "value_per_yll"
        ],
        health_risk_factors=config["health"]["risk_factors"],
    output:
        csv="results/{name}/analysis/scen-{scenario}/health_impacts.csv",
    log:
        "logs/{name}/extract_health_impacts_scen-{scenario}.log",
    script:
        "../scripts/analysis/extract_health_impacts.py"


rule extract_statistics:
    """Extract production and consumption statistics."""
    input:
        network="results/{name}/solved/model_scen-{scenario}.nc",
    output:
        crop_production="results/{name}/analysis/scen-{scenario}/crop_production.csv",
        land_use="results/{name}/analysis/scen-{scenario}/land_use.csv",
        animal_production="results/{name}/analysis/scen-{scenario}/animal_production.csv",
        food_consumption="results/{name}/analysis/scen-{scenario}/food_consumption.csv",
        food_group_consumption="results/{name}/analysis/scen-{scenario}/food_group_consumption.csv",
    log:
        "logs/{name}/extract_statistics_scen-{scenario}.log",
    script:
        "../scripts/analysis/extract_statistics.py"
