# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Health-related data preparation rules.

Includes dietary intake, mortality rates, relative risks, life tables,
and health cost calculations.
"""


rule prepare_gbd_mortality:
    input:
        gbd_mortality="data/manually_downloaded/IHME-GBD_2023-dealth-rates.csv",
    params:
        countries=config["countries"],
        causes=config["health"]["causes"],
        reference_year=config["health"]["reference_year"],
    output:
        mortality="processing/{name}/health/gbd_mortality_rates.csv",
    log:
        "logs/{name}/prepare_gbd_mortality.log",
    script:
        "../scripts/prepare_gbd_mortality.py"


rule prepare_gdd_dietary_intake:
    input:
        gdd_dir="data/manually_downloaded/GDD-dietary-intake",
    params:
        countries=config["countries"],
        food_groups=config["food_groups"]["included"],
        reference_year=config["health"]["reference_year"],
        ssb_sugar_g_per_100g=config["health"]["ssb_sugar_g_per_100g"],
    output:
        diet="processing/{name}/gdd_dietary_intake.csv",
    log:
        "logs/{name}/prepare_gdd_dietary_intake.log",
    script:
        "../scripts/prepare_gdd_dietary_intake.py"


rule retrieve_faostat_food_supply:
    params:
        countries=config["countries"],
        reference_year=config["health"]["reference_year"],
    output:
        supply="processing/{name}/faostat_food_supply.csv",
    log:
        "logs/{name}/retrieve_faostat_food_supply.log",
    script:
        "../scripts/retrieve_faostat_food_supply.py"


rule merge_dietary_sources:
    input:
        gdd="processing/{name}/gdd_dietary_intake.csv",
        faostat="processing/{name}/faostat_food_supply.csv",
        food_loss_waste="processing/{name}/food_loss_waste.csv",
    output:
        diet="processing/{name}/dietary_intake.csv",
    log:
        "logs/{name}/merge_dietary_sources.log",
    script:
        "../scripts/merge_dietary_sources.py"


rule prepare_food_loss_waste:
    input:
        m49="data/M49-codes.csv",
        animal_production="processing/{name}/faostat_animal_production.csv",
        faostat_food_supply="processing/{name}/faostat_food_supply.csv",
        population="processing/{name}/population.csv",
    params:
        countries=config["countries"],
        food_groups=config["food_groups"]["included"],
        health_reference_year=config["health"]["reference_year"],
    output:
        food_loss_waste="processing/{name}/food_loss_waste.csv",
    log:
        "logs/{name}/prepare_food_loss_waste.log",
    script:
        "../scripts/prepare_food_loss_waste.py"


rule prepare_relative_risks:
    input:
        gbd_rr="data/manually_downloaded/IHME_GBD_2019_RELATIVE_RISKS_Y2020M10D15.XLSX",
    params:
        risk_factors=config["health"]["risk_factors"],
        causes=config["health"]["causes"],
        omega3_per_100g=config["health"]["omega3_per_100g_fish"],
        ssb_sugar_g_per_100g=config["health"]["ssb_sugar_g_per_100g"],
    output:
        relative_risks="processing/{name}/health/relative_risks.csv",
    log:
        "logs/{name}/prepare_relative_risks.log",
    script:
        "../scripts/prepare_relative_risks.py"


rule prepare_life_table:
    input:
        wpp_life_table="data/downloads/WPP_life_table.csv.gz",
    params:
        reference_year=config["health"]["reference_year"],
    output:
        life_table="processing/{name}/health/life_table.csv",
    log:
        "logs/{name}/prepare_life_table.log",
    script:
        "../scripts/prepare_life_table.py"


rule prepare_health_costs:
    input:
        regions="processing/{name}/regions.geojson",
        diet="processing/{name}/dietary_intake.csv",
        relative_risks="processing/{name}/health/relative_risks.csv",
        dr="processing/{name}/health/gbd_mortality_rates.csv",
        population="processing/{name}/population_age.csv",
        life_table="processing/{name}/health/life_table.csv",
    params:
        countries=config["countries"],
        health=config["health"],
    output:
        risk_breakpoints="processing/{name}/health/risk_breakpoints.csv",
        cluster_cause="processing/{name}/health/cluster_cause_baseline.csv",
        cause_log="processing/{name}/health/cause_log_breakpoints.csv",
        cluster_summary="processing/{name}/health/cluster_summary.csv",
        clusters="processing/{name}/health/country_clusters.csv",
        cluster_risk_baseline="processing/{name}/health/cluster_risk_baseline.csv",
    log:
        "logs/{name}/prepare_health_costs.log",
    script:
        "../scripts/prepare_health_costs.py"
