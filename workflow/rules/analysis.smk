# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later


rule extract_marginal_damages:
    """Extract marginal GHG and health damages by food group and country."""
    input:
        network="results/{name}/solved/model_scen-{scenario}.nc",
        food_groups="data/food_groups.csv",
        risk_breakpoints="processing/{name}/health/scen-{scenario}/risk_breakpoints.csv",
        health_cluster_cause="processing/{name}/health/scen-{scenario}/cluster_cause_baseline.csv",
        health_cause_log="processing/{name}/health/scen-{scenario}/cause_log_breakpoints.csv",
        health_clusters="processing/{name}/health/scen-{scenario}/country_clusters.csv",
        population="processing/{name}/population.csv",
    params:
        ghg_price=lambda w: get_effective_config(w.scenario)["emissions"]["ghg_price"],
        value_per_yll=lambda w: get_effective_config(w.scenario)["health"][
            "value_per_yll"
        ],
        ch4_gwp=config["emissions"]["ch4_to_co2_factor"],
        n2o_gwp=config["emissions"]["n2o_to_co2_factor"],
        health_risk_factors=config["health"]["risk_factors"],
    output:
        csv="results/{name}/analysis/scen-{scenario}/marginal_damages.csv",
    log:
        "logs/{name}/extract_marginal_damages_scen-{scenario}.log",
    script:
        "../scripts/analysis/extract_marginal_damages.py"


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
