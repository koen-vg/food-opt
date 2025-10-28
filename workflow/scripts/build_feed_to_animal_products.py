# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Generate feed-to-animal-product conversion efficiencies from Wirsenius (2000) data.

Uses regional feed energy requirements combined with feed category energy values
to calculate feed conversion efficiencies (tonnes product per tonne feed DM).

The approach:
1. For each animal product * region * feed category combination:
2. Get energy requirement from Wirsenius (MJ per kg product)
3. Get energy content of feed category from GLEAM (MJ per kg DM)
4. Calculate: efficiency = feed_energy / product_energy_requirement
   (in tonnes product per tonne feed DM)

For ruminants:
- Wirsenius provides NE_m (maintenance) and NE_g (growth) requirements
- We use NE = ME * efficiency factors from NRC (2000):
  - For maintenance: k_m ~ 0.60 (typical for mixed diets)
  - For growth: k_g ~ 0.40 (typical for mixed diets)
- Therefore: ME_required = NE_m/k_m + NE_g/k_g

For dairy:
- Wirsenius provides NE_l (lactation), NE_m, and NE_g
- We use: ME_required = NE_l/k_l + NE_m/k_m + NE_g/k_g
- Where k_l ~ 0.60 (typical)

For monogastrics:
- Wirsenius directly provides ME requirements
- No conversion needed

References:
- Wirsenius (2000): Feed energy requirements by region
- NRC (2000): Nutrient Requirements of Beef Cattle (k factors)
- GLEAM (2022): Feed energy content values
"""

import logging

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def calculate_ruminant_me_requirements(
    wirsenius_data: pd.DataFrame, k_m: float, k_g: float, k_l: float
) -> pd.DataFrame:
    """
    Convert Wirsenius NE requirements to ME requirements for ruminants.

    For beef cattle: ME = NE_m / k_m + NE_g / k_g
    For dairy cattle: ME = NE_l / k_l + NE_m / k_m + NE_g / k_g

    Parameters
    ----------
    wirsenius_data : pd.DataFrame
        Wirsenius energy requirements
    k_m : float
        Maintenance efficiency factor
    k_g : float
        Growth efficiency factor
    k_l : float
        Lactation efficiency factor

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: animal_product, region, ME_MJ_per_kg
    """
    results = []

    for product in wirsenius_data["animal_product"].unique():
        product_data = wirsenius_data[wirsenius_data["animal_product"] == product]

        for region in product_data["region"].unique():
            region_data = product_data[product_data["region"] == region]

            # Extract energy requirements
            ne_values = {}
            for _, row in region_data.iterrows():
                ne_values[row["unit"]] = row["value"]

            # Calculate ME requirement based on product type
            if product == "dairy":
                # Dairy: NE_l + NE_m + NE_g
                me_req = (
                    ne_values.get("NE_l", 0) / k_l
                    + ne_values.get("NE_m", 0) / k_m
                    + ne_values.get("NE_g", 0) / k_g
                )
            elif product == "cattle meat":
                # Beef: NE_m + NE_g
                me_req = ne_values.get("NE_m", 0) / k_m + ne_values.get("NE_g", 0) / k_g
            else:
                # Skip non-ruminants
                continue

            results.append(
                {
                    "animal_product": product,
                    "region": region,
                    "ME_MJ_per_kg": me_req,
                }
            )

    df = pd.DataFrame(results)
    logger.info(
        "Calculated ME requirements for %d ruminant product-region combinations",
        len(df),
    )

    return df


def get_monogastric_me_requirements(wirsenius_data: pd.DataFrame) -> pd.DataFrame:
    """
    Extract ME requirements for monogastrics (already in ME units).

    Returns DataFrame with columns: animal_product, region, ME_MJ_per_kg
    """
    # Filter to monogastric products
    monogastric_products = ["pig meat", "chicken meat", "eggs"]
    df = wirsenius_data[wirsenius_data["animal_product"].isin(monogastric_products)]

    # Should only have ME unit
    df = df[df["unit"] == "ME"].copy()
    df = df.rename(columns={"value": "ME_MJ_per_kg"})
    df = df[["animal_product", "region", "ME_MJ_per_kg"]]

    logger.info(
        "Extracted ME requirements for %d monogastric product-region combinations",
        len(df),
    )

    return df


def calculate_feed_efficiencies(
    me_requirements: pd.DataFrame,
    feed_categories: pd.DataFrame,
    animal_type: str,
) -> pd.DataFrame:
    """
    Calculate feed conversion efficiencies from ME requirements and feed energy values.

    efficiency = ME_content_feed / ME_requirement_product
    (in tonnes product per tonne feed DM)

    Parameters
    ----------
    me_requirements : pd.DataFrame
        Product ME requirements with columns: animal_product, region, ME_MJ_per_kg
    feed_categories : pd.DataFrame
        Feed category properties with columns: category, ME_MJ_per_kg_DM
    animal_type : str
        Either "ruminant" or "monogastric"

    Returns
    -------
    pd.DataFrame
        With columns: product, feed_category, region, efficiency
    """
    results = []

    for _, product_row in me_requirements.iterrows():
        product = product_row["animal_product"]
        region = product_row["region"]
        me_req = product_row["ME_MJ_per_kg"]  # MJ per kg product

        for _, feed_row in feed_categories.iterrows():
            category = feed_row["category"]
            me_feed = feed_row["ME_MJ_per_kg_DM"]  # MJ per kg feed DM

            # efficiency = kg_product / kg_feed_DM
            # = (MJ/kg_feed) / (MJ/kg_product)
            efficiency = me_feed / me_req

            # Convert to tonnes product per tonne feed DM
            efficiency_t = efficiency

            feed_category = f"{animal_type}_{category}"

            results.append(
                {
                    "product": product,
                    "feed_category": feed_category,
                    "region": region,
                    "efficiency": efficiency_t,
                }
            )

    df = pd.DataFrame(results)
    logger.info(
        "Calculated %d feed conversion efficiencies for %s",
        len(df),
        animal_type,
    )

    return df


def add_notes(efficiencies: pd.DataFrame) -> pd.DataFrame:
    """Add descriptive notes column."""
    efficiencies = efficiencies.copy()

    def create_note(row):
        feed_cat = row["feed_category"]
        region = row["region"]
        product = row["product"]
        eff = row["efficiency"]

        # Calculate feed requirement (inverse of efficiency)
        feed_req = 1.0 / eff if eff > 0 else float("inf")

        return (
            f"Wirsenius (2000) {region} {product}: "
            f"{feed_req:.1f} t DM {feed_cat} per tonne product"
        )

    efficiencies["notes"] = efficiencies.apply(create_note, axis=1)

    return efficiencies


def build_feed_to_animal_products(
    wirsenius_file: str,
    ruminant_categories_file: str,
    monogastric_categories_file: str,
    output_file: str,
    regions_to_average: list[str] | None = None,
    k_m: float = 0.60,
    k_g: float = 0.40,
    k_l: float = 0.60,
) -> None:
    """
    Generate feed-to-animal-product conversion table from Wirsenius data.

    Parameters
    ----------
    wirsenius_file : str
        Path to Wirsenius feed energy requirements CSV
    ruminant_categories_file : str
        Path to ruminant feed categories CSV
    monogastric_categories_file : str
        Path to monogastric feed categories CSV
    output_file : str
        Path to output feed_to_animal_products.csv
    regions_to_average : list[str] | None
        List of Wirsenius regions to average. If None or empty, use all regions.
    k_m : float
        NRC maintenance efficiency factor (default 0.60)
    k_g : float
        NRC growth efficiency factor (default 0.40)
    k_l : float
        NRC lactation efficiency factor (default 0.60)
    """
    # Load data
    wirsenius = pd.read_csv(wirsenius_file, comment="#")
    ruminant_cats = pd.read_csv(ruminant_categories_file)
    monogastric_cats = pd.read_csv(monogastric_categories_file)

    logger.info("Loaded Wirsenius data: %d rows", len(wirsenius))
    logger.info("Loaded ruminant categories: %d", len(ruminant_cats))
    logger.info("Loaded monogastric categories: %d", len(monogastric_cats))
    logger.info("NRC efficiency factors: k_m=%.2f, k_g=%.2f, k_l=%.2f", k_m, k_g, k_l)

    # Calculate ME requirements
    ruminant_me = calculate_ruminant_me_requirements(wirsenius, k_m, k_g, k_l)
    monogastric_me = get_monogastric_me_requirements(wirsenius)

    # Calculate feed conversion efficiencies
    ruminant_eff = calculate_feed_efficiencies(ruminant_me, ruminant_cats, "ruminant")
    monogastric_eff = calculate_feed_efficiencies(
        monogastric_me, monogastric_cats, "monogastric"
    )

    # Combine
    all_eff = pd.concat([ruminant_eff, monogastric_eff], ignore_index=True)

    # Filter to specified regions if provided
    if regions_to_average:
        logger.info("Filtering to regions: %s", ", ".join(regions_to_average))
        all_eff = all_eff[all_eff["region"].isin(regions_to_average)].copy()
        if all_eff.empty:
            raise ValueError(
                f"No data found for specified regions: {regions_to_average}"
            )
        region_label = " & ".join(regions_to_average)
    else:
        logger.info("Using all regions")
        region_label = "Global average"

    # Average across the selected regions
    logger.info("Averaging efficiencies across selected regions")
    all_eff = all_eff.groupby(["product", "feed_category"], as_index=False).agg(
        {"efficiency": "mean"}
    )
    all_eff["region"] = region_label

    # Add notes
    all_eff = add_notes(all_eff)

    # Sort
    all_eff = all_eff.sort_values(["product", "feed_category"])

    # Write output with header
    header = """# SPDX-FileCopyrightText: 2000 Stefan Wirsenius
# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: CC-BY-4.0
#
# Feed-to-animal-product conversion efficiencies
#
# Generated from Wirsenius (2000) regional feed energy requirements combined with
# GLEAM 3.0 feed category energy values.
#
# Source: Wirsenius, S. (2000). Human Use of Land and Organic Materials:
# Modeling the Turnover of Biomass in the Global Food System.
# Chalmers University of Technology and Göteborg University, Sweden.
# Table 3.9, ISBN 91-7197-886-0.
# https://publications.lib.chalmers.se/records/fulltext/827.pdf
#
# Methodology:
# 1. Wirsenius provides regional energy requirements (NE or ME) per kg product
# 2. For ruminants: Convert NE to ME using NRC (2000) efficiency factors
#    - k_m = 0.60 (maintenance), k_g = 0.40 (growth), k_l = 0.60 (lactation)
# 3. Feed categories provide ME content (MJ per kg DM) from GLEAM
# 4. Efficiency = ME_feed / ME_requirement (tonnes product per tonne feed DM)
#
# Columns:
#   product: Animal product name
#   feed_category: Feed quality category (ruminant_* or monogastric_*)
#   region: Geographic region from Wirsenius
#   efficiency: Feed conversion efficiency (t product / t feed DM)
#   notes: Descriptive text with feed requirement in inverse form
#
"""

    with open(output_file, "w") as f:
        f.write(header)
        all_eff.to_csv(f, index=False)

    logger.info("Wrote %d feed conversion entries to %s", len(all_eff), output_file)

    # Log summary statistics
    logger.info("\nSummary by product:")
    for product in all_eff["product"].unique():
        product_data = all_eff[all_eff["product"] == product]
        logger.info(
            "  %s: %d entries (%.3f-%.3f t/t)",
            product,
            len(product_data),
            product_data["efficiency"].min(),
            product_data["efficiency"].max(),
        )


if __name__ == "__main__":
    # Get regions to average from config
    regions = snakemake.params.wirsenius_regions

    # Get net-to-ME conversion efficiency factors from config
    conversion_factors = snakemake.params.net_to_me_conversion
    k_m = conversion_factors["k_m"]
    k_g = conversion_factors["k_g"]
    k_l = conversion_factors["k_l"]

    build_feed_to_animal_products(
        wirsenius_file=snakemake.input.wirsenius,
        ruminant_categories_file=snakemake.input.ruminant_categories,
        monogastric_categories_file=snakemake.input.monogastric_categories,
        output_file=snakemake.output[0],
        regions_to_average=regions,
        k_m=k_m,
        k_g=k_g,
        k_l=k_l,
    )
