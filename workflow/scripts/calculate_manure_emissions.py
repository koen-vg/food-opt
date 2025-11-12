# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Calculate methane emissions from livestock manure management.

Computes CH4 emissions per kg dry matter feed intake for different animal
products and feed categories, accounting for:
- Volatile solids (VS) excretion based on digestibility and ash content
- Manure management system distributions from GLEAM
- IPCC methane conversion factors and producing capacity

NOTE: Currently averages MCF values across climate zones. This will be refined
later when climate zone data is added to modeling regions, allowing for
country-specific and region-specific emission factors.
"""

import logging

from logging_config import setup_script_logging
import pandas as pd

# Logger will be configured in __main__ block
logger = logging.getLogger(__name__)

# Urinary energy excretion fractions (as fraction of gross energy intake)
URINARY_FRACTIONS = {
    "ruminant": 0.04,
    "pig": 0.02,
    "chicken": 0.0,
}

# Conversion factor from m³ CH₄ to kg CH₄
M3_CH4_TO_KG = 0.67

# Map animal products to urinary fraction categories
PRODUCT_TO_URINARY_CATEGORY = {
    "meat-cattle": "ruminant",
    "dairy": "ruminant",
    "meat-pig": "pig",
    "meat-chicken": "chicken",
    "eggs": "chicken",
}

# Map GLEAM animal names to our animal products
GLEAM_TO_PRODUCT = {
    "Cattle": ["meat-cattle", "dairy"],
    "Buffalo": ["meat-cattle", "dairy"],  # Treat buffalo like cattle
    "Pigs": ["meat-pig"],
    "Chickens": ["meat-chicken", "eggs"],
}


def calculate_volatile_solids(
    feed_categories: pd.DataFrame,
    product: str,
) -> pd.DataFrame:
    """Calculate VS excretion per kg DM feed intake for each feed category.

    Uses adapted IPCC equation 10.24:
    VS = (1 - digestibility + urinary_fraction) * (1 - ash_content)

    Parameters
    ----------
    feed_categories : pd.DataFrame
        Feed category properties including digestibility and ash_content_pct_dm
    product : str
        Animal product name (e.g., "meat-cattle", "meat-pig")

    Returns
    -------
    pd.DataFrame
        Feed categories with added VS_kg_per_kg_DMI column
    """
    # Get urinary fraction for this product
    urinary_category = PRODUCT_TO_URINARY_CATEGORY[product]
    urinary_fraction = URINARY_FRACTIONS[urinary_category]

    # Calculate VS (kg VS per kg DM intake)
    # ash_content_pct_dm is in percent, convert to fraction
    df = feed_categories.copy()
    df["VS_kg_per_kg_DMI"] = (1 - df["digestibility"] + urinary_fraction) * (
        1 - df["ash_content_pct_dm"] / 100
    )

    return df


def average_mcf_over_climate_zones(
    mcf_data: pd.DataFrame,
) -> pd.DataFrame:
    """Average MCF values across climate zones for each manure management system.

    NOTE: This is a temporary simplification. Will be refined when climate zone
    data is added to modeling regions.

    Parameters
    ----------
    mcf_data : pd.DataFrame
        MCF data with columns: manure management system, climate zone, methane conversion factor

    Returns
    -------
    pd.DataFrame
        Averaged MCF by manure management system
    """
    # Average over climate zones
    mcf_avg = (
        mcf_data.groupby("manure management system")["methane conversion factor"]
        .mean()
        .reset_index()
    )

    logger.info("Averaged MCF over climate zones (temporary approach)")
    logger.info("  Will be refined with regional climate data in future")

    return mcf_avg


def get_mms_fractions_for_product(
    mms_fractions: pd.DataFrame,
    product: str,
) -> pd.DataFrame:
    """Get manure management system fractions for a given animal product.

    Averages over all livestock production systems (LPS) types for the given
    GLEAM animal category that maps to this product.

    Parameters
    ----------
    mms_fractions : pd.DataFrame
        GLEAM manure management system fractions
    product : str
        Animal product name

    Returns
    -------
    pd.DataFrame
        MMS fractions (long format) for this product
    """
    # Find which GLEAM animal(s) correspond to this product
    gleam_animals = [
        gleam_animal
        for gleam_animal, products in GLEAM_TO_PRODUCT.items()
        if product in products
    ]

    if not gleam_animals:
        raise ValueError(f"No GLEAM animal mapping found for product: {product}")

    # Filter to relevant animals and average over LPS types
    df = mms_fractions[mms_fractions["animal"].isin(gleam_animals)]

    # Get all MMS columns (everything except area, animal, lps)
    mms_columns = [col for col in df.columns if col not in ["area", "animal", "lps"]]

    # Average over LPS types
    avg_fractions = df[mms_columns].mean()

    # Convert to long format
    result = pd.DataFrame(
        {
            "manure management system": avg_fractions.index,
            "fraction": avg_fractions.values / 100,  # Convert from percent to fraction
        }
    )

    return result


def calculate_manure_ch4_for_product(
    product: str,
    feed_categories: pd.DataFrame,
    b0_data: pd.DataFrame,
    mcf_avg: pd.DataFrame,
    mms_fractions: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate manure CH4 emissions for one animal product across feed categories.

    Parameters
    ----------
    product : str
        Animal product name
    feed_categories : pd.DataFrame
        Feed category properties with VS calculated
    b0_data : pd.DataFrame
        B0 values (maximum CH4 producing capacity) by animal product
    mcf_avg : pd.DataFrame
        Climate-averaged MCF by manure management system
    mms_fractions : pd.DataFrame
        MMS fractions for this product

    Returns
    -------
    pd.DataFrame
        CH4 emissions (kg CH4 per kg DM intake) by feed category
    """
    # Get B0 for this product
    b0 = b0_data.loc[b0_data["animal product"] == product, "B0"].values[0]

    # Calculate VS for each feed category
    vs_data = calculate_volatile_solids(feed_categories, product)

    # Merge MMS fractions with MCF values
    mms_with_mcf = mms_fractions.merge(
        mcf_avg,
        on="manure management system",
        how="left",
    )

    # Calculate weighted average MCF across all manure management systems
    weighted_mcf = (
        mms_with_mcf["fraction"] * mms_with_mcf["methane conversion factor"]
    ).sum()

    # Calculate CH4 emissions: VS * B0 * MCF * conversion_factor
    # Result in kg CH4 per kg DM intake
    vs_data["manure_ch4_kg_per_kg_DMI"] = (
        vs_data["VS_kg_per_kg_DMI"] * b0 * weighted_mcf * M3_CH4_TO_KG
    )

    # Select relevant columns
    result = vs_data[["category", "manure_ch4_kg_per_kg_DMI"]].copy()
    result["product"] = product

    logger.info(
        "  %s: weighted MCF=%.4f, avg CH4=%.4f kg/kg DMI",
        product,
        weighted_mcf,
        result["manure_ch4_kg_per_kg_DMI"].mean(),
    )

    return result


if __name__ == "__main__":
    # Configure logging
    logger = setup_script_logging(log_file=snakemake.log[0] if snakemake.log else None)

    # Read inputs
    ruminant_categories = pd.read_csv(snakemake.input.ruminant_feed_categories)
    monogastric_categories = pd.read_csv(snakemake.input.monogastric_feed_categories)
    b0_data = pd.read_csv(snakemake.input.b0_data, comment="#")
    mcf_data = pd.read_csv(snakemake.input.mcf_data, comment="#")
    mms_fractions = pd.read_csv(snakemake.input.mms_fractions, comment="#")

    # Get countries from config
    countries = snakemake.config["countries"]

    # Average MCF over climate zones
    mcf_avg = average_mcf_over_climate_zones(mcf_data)

    logger.info("Calculating manure CH4 emissions by product and feed category:")

    # Process each animal product
    results = []

    for product in b0_data["animal product"].unique():
        # Determine if ruminant or monogastric
        if product in ["meat-cattle", "dairy"]:
            feed_categories = ruminant_categories
            category_prefix = "ruminant_"
        else:
            feed_categories = monogastric_categories
            category_prefix = "monogastric_"

        # Get MMS fractions for this product
        mms_frac = get_mms_fractions_for_product(mms_fractions, product)

        # Calculate emissions
        product_result = calculate_manure_ch4_for_product(
            product,
            feed_categories,
            b0_data,
            mcf_avg,
            mms_frac,
        )

        # Add category prefix to match feed_to_animal_products.csv format
        product_result["feed_category"] = category_prefix + product_result["category"]
        product_result = product_result.drop(columns=["category"])

        results.append(product_result)

    # Combine all products
    emissions = pd.concat(results, ignore_index=True)

    # Expand to all countries (same values for now, will be refined later)
    # NOTE: Emission factors are currently identical across countries
    # Will be differentiated by climate zone and region in future
    country_emissions = []
    for country in countries:
        df = emissions.copy()
        df["country"] = country
        country_emissions.append(df)

    final = pd.concat(country_emissions, ignore_index=True)

    # Reorder columns: country, product, feed_category, manure_ch4_kg_per_kg_DMI
    final = final[["country", "product", "feed_category", "manure_ch4_kg_per_kg_DMI"]]

    # Write output
    final.to_csv(snakemake.output[0], index=False)

    logger.info("Manure CH4 emission factors calculated")
    logger.info("  Products: %d", len(emissions["product"].unique()))
    logger.info("  Feed categories: %d", len(emissions["feed_category"].unique()))
    logger.info("  Countries: %d", len(countries))
    logger.info("  Total rows: %d", len(final))
