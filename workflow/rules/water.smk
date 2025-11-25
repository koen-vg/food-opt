# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Water and fertilizer-related data preparation rules.

Includes fertilizer application rates, blue water availability,
and regional water resource calculations.
"""


rule prepare_fertilizer_application_rates:
    input:
        fubc_data="data/downloads/ifa_fubc_1_to_9_data.csv",
        mapping="data/ifa_fubc_crop_mapping.csv",
    output:
        "processing/{name}/fertilizer_application_rates.csv",
    log:
        "logs/{name}/prepare_fertilizer_application_rates.log",
    script:
        "../scripts/prepare_fertilizer_application_rates.py"


rule derive_global_fertilizer_rates:
    input:
        fertilizer_rates="processing/{name}/fertilizer_application_rates.csv",
    params:
        n_percentile=config["fertilizer"]["n_percentile"],
        crops=config["crops"],
    output:
        "processing/{name}/global_fertilizer_n_rates.csv",
    log:
        "logs/{name}/derive_global_fertilizer_rates.log",
    script:
        "../scripts/derive_global_fertilizer_rates.py"


rule extract_waterfootprint_appendix:
    input:
        zip_path="data/downloads/Report53_Appendix.zip",
    output:
        shapefile="data/downloads/Report53_Appendix/Report53-BlueWaterScarcity-ArcGIS-ShapeFile/Monthly_WS_GRDC_405_basins.shp",
        excel="data/downloads/Report53_Appendix/Report53-Appendices-VI-IX.xls",
    log:
        "logs/shared/extract_waterfootprint_appendix.log",
    shell:
        r"""
        unzip -o {input.zip_path} -d data/downloads > {log} 2>&1
        """


rule process_blue_water_availability:
    input:
        shapefile=rules.extract_waterfootprint_appendix.output.shapefile,
        excel=rules.extract_waterfootprint_appendix.output.excel,
    output:
        "processing/{name}/water/blue_water_availability.csv",
    log:
        "logs/{name}/process_blue_water_availability.log",
    script:
        "../scripts/process_blue_water_availability.py"


def crop_yield_file_list(w):
    return list(yield_inputs(w).values())


rule build_region_water_availability:
    input:
        shapefile=rules.extract_waterfootprint_appendix.output.shapefile,
        regions="processing/{name}/regions.geojson",
        monthly="processing/{name}/water/blue_water_availability.csv",
        crop_yields=crop_yield_file_list,
    output:
        monthly_region="processing/{name}/water/monthly_region_water.csv",
        region_growing="processing/{name}/water/region_growing_season_water.csv",
    log:
        "logs/{name}/build_region_water_availability.log",
    script:
        "../scripts/build_region_water_availability.py"
