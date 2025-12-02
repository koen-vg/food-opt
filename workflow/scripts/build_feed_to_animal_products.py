# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Generate feed-to-animal-product conversion efficiencies from Wirsenius (2000) data.

Uses regional feed energy requirements combined with feed category energy values
to calculate feed conversion efficiencies (tonnes product per tonne feed DM).

UNIT CONVERSIONS:
1. Feed inputs: DRY MATTER (tonnes DM)
2. Animal product outputs: FRESH WEIGHT, RETAIL MEAT (tonnes fresh weight)
3. Wirsenius (2000) provides requirements per kg CARCASS WEIGHT (dressed, bone-in)
4. We apply carcass-to-retail conversion to get retail meat weight

The approach:
1. For each animal product * region * feed category combination:
2. Get energy requirement from Wirsenius (MJ per kg CARCASS)
3. Convert to retail meat using carcass_to_retail factors (MJ per kg RETAIL MEAT)
4. Get energy content of feed category from GLEAM (MJ per kg DM)
5. Calculate: efficiency = feed_energy / product_energy_requirement
   (in tonnes RETAIL MEAT per tonne FEED DM)

For ruminants:
- Wirsenius provides NE_m (maintenance) and NE_g (growth) requirements per kg carcass
- Convert NE to ME using NRC (2000) efficiency factors:
  - For maintenance: k_m ~ 0.60 (typical for mixed diets)
  - For growth: k_g ~ 0.40 (typical for mixed diets)
- ME_required_carcass = NE_m/k_m + NE_g/k_g
- ME_required_retail = ME_required_carcass / carcass_to_retail_factor

For dairy:
- Wirsenius provides NE_l (lactation), NE_m, and NE_g per kg whole milk
- ME_required = NE_l/k_l + NE_m/k_m + NE_g/k_g
- No carcass conversion (milk is already retail product)

For monogastrics:
- Wirsenius directly provides ME requirements per kg carcass
- Apply carcass-to-retail conversion to get ME per kg retail meat

References:
- Wirsenius (2000): Feed energy requirements by region (Table 3.9)
- NRC (2000): Nutrient Requirements of Beef Cattle (NE to ME conversion factors)
- GLEAM (2022): Feed energy content values
- USDA/FAO: Carcass-to-retail conversion factors
"""

import logging

from logging_config import setup_script_logging
import pandas as pd

# Logger will be configured in __main__ block
logger = logging.getLogger(__name__)


def calculate_ruminant_me_requirements(
    wirsenius_data: pd.DataFrame,
    k_m: float,
    k_g: float,
    k_l: float,
    carcass_to_retail: dict[str, float],
    feed_proxy_map: dict[str, str],
) -> pd.DataFrame:
    """
    Convert Wirsenius NE requirements to ME requirements for ruminants.

    For beef cattle: ME_carcass = NE_m / k_m + NE_g / k_g
                     ME_retail = ME_carcass / carcass_to_retail_factor
    For dairy cattle: ME = NE_l / k_l + NE_m / k_m + NE_g / k_g
                      (no carcass conversion, milk is already retail product)

    Supports proxy products that use feed requirements from other products,
    configured via feed_proxy_map (e.g., dairy-buffalo -> dairy).

    Parameters
    ----------
    wirsenius_data : pd.DataFrame
        Wirsenius energy requirements (per kg CARCASS for meat, per kg PRODUCT for milk/eggs)
    k_m : float
        Maintenance efficiency factor (NE to ME conversion)
    k_g : float
        Growth efficiency factor (NE to ME conversion)
    k_l : float
        Lactation efficiency factor (NE to ME conversion)
    carcass_to_retail : dict[str, float]
        Carcass-to-retail conversion factors by product
    feed_proxy_map : dict[str, str]
        Proxy products mapped to their source products

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: animal_product, region, ME_MJ_per_kg_RETAIL
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
                # Dairy: NE_l + NE_m + NE_g (per kg whole milk, already retail product)
                me_req_carcass = (
                    ne_values.get("NE_l", 0) / k_l
                    + ne_values.get("NE_m", 0) / k_m
                    + ne_values.get("NE_g", 0) / k_g
                )
                # No carcass conversion for dairy
                me_req_retail = me_req_carcass
            elif product == "meat-cattle":
                # Beef: NE_m + NE_g (per kg CARCASS)
                me_req_carcass = (
                    ne_values.get("NE_m", 0) / k_m + ne_values.get("NE_g", 0) / k_g
                )
                # Convert from per kg carcass to per kg retail meat
                conversion_factor = carcass_to_retail[product]
                me_req_retail = me_req_carcass / conversion_factor
            else:
                # Skip non-ruminants
                continue

            results.append(
                {
                    "animal_product": product,
                    "region": region,
                    "ME_MJ_per_kg": me_req_retail,
                }
            )

    df = pd.DataFrame(results)

    # ADD PROXY PRODUCTS using their source's ME requirements
    proxy_results = []
    for proxy_product, source_product in feed_proxy_map.items():
        if source_product in df["animal_product"].values:
            # Copy source product's requirements to proxy product
            source_data = df[df["animal_product"] == source_product].copy()
            source_data["animal_product"] = proxy_product

            # Adjust for carcass-to-retail conversion if different
            if (
                proxy_product in carcass_to_retail
                and source_product in carcass_to_retail
            ):
                source_factor = carcass_to_retail[source_product]
                proxy_factor = carcass_to_retail[proxy_product]
                if source_factor > 0 and proxy_factor > 0:
                    # Adjust ME requirements per kg retail meat
                    adjustment = source_factor / proxy_factor
                    source_data["ME_MJ_per_kg"] = (
                        source_data["ME_MJ_per_kg"] * adjustment
                    )

            proxy_results.append(source_data)

    if proxy_results:
        df = pd.concat([df, *proxy_results], ignore_index=True)
        logger.info(
            "Added proxy products: %s",
            ", ".join(feed_proxy_map.keys()),
        )

    logger.info(
        "Calculated ME requirements for %d ruminant product-region combinations",
        len(df),
    )

    return df


def get_monogastric_me_requirements(
    wirsenius_data: pd.DataFrame, carcass_to_retail: dict[str, float]
) -> pd.DataFrame:
    """
    Extract ME requirements for monogastrics (already in ME units).

    Wirsenius provides ME per kg CARCASS for meat products.
    We convert to ME per kg RETAIL MEAT using carcass_to_retail factors.
    Eggs are already retail product (no conversion).

    Parameters
    ----------
    wirsenius_data : pd.DataFrame
        Wirsenius energy requirements
    carcass_to_retail : dict[str, float]
        Carcass-to-retail conversion factors by product

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: animal_product, region, ME_MJ_per_kg_RETAIL
    """
    # Filter to monogastric products
    monogastric_products = ["meat-pig", "meat-chicken", "eggs"]
    df = wirsenius_data[wirsenius_data["animal_product"].isin(monogastric_products)]

    # Should only have ME unit
    df = df[df["unit"] == "ME"].copy()
    df = df.rename(columns={"value": "ME_MJ_per_kg_carcass"})

    # Apply carcass-to-retail conversion
    df["conversion_factor"] = df["animal_product"].map(carcass_to_retail)
    df["ME_MJ_per_kg"] = df["ME_MJ_per_kg_carcass"] / df["conversion_factor"]

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
    regions_to_average: list[str],
    k_m: float,
    k_g: float,
    k_l: float,
    carcass_to_retail: dict[str, float],
    feed_proxy_map: dict[str, str],
) -> None:
    """
    Generate feed-to-animal-product conversion table from Wirsenius data.

    Converts Wirsenius carcass-weight-based feed requirements to retail-meat-based
    feed conversion efficiencies.

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
        List of Wirsenius regions to average.
    k_m : float
        NRC maintenance efficiency factor
    k_g : float
        NRC growth efficiency factor
    k_l : float
        NRC lactation efficiency factor
    carcass_to_retail : dict[str, float]
        Carcass-to-retail conversion factors by product
    feed_proxy_map : dict[str, str]
        Mapping of proxy products to source products for feed requirements
    """
    # Load data
    wirsenius = pd.read_csv(wirsenius_file, comment="#")
    ruminant_cats = pd.read_csv(ruminant_categories_file)
    monogastric_cats = pd.read_csv(monogastric_categories_file)

    logger.info("Loaded Wirsenius data: %d rows", len(wirsenius))
    logger.info("Loaded ruminant categories: %d", len(ruminant_cats))
    logger.info("Loaded monogastric categories: %d", len(monogastric_cats))
    logger.info("NRC efficiency factors: k_m=%.2f, k_g=%.2f, k_l=%.2f", k_m, k_g, k_l)
    logger.info("Carcass-to-retail conversion factors:")
    for product, factor in carcass_to_retail.items():
        logger.info("  %s: %.2f", product, factor)

    # Calculate ME requirements (converted to per kg RETAIL product)
    ruminant_me = calculate_ruminant_me_requirements(
        wirsenius,
        k_m,
        k_g,
        k_l,
        carcass_to_retail,
        feed_proxy_map,
    )
    monogastric_me = get_monogastric_me_requirements(wirsenius, carcass_to_retail)

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

    # Write output
    all_eff.to_csv(output_file, index=False)

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
    # Configure logging
    logger = setup_script_logging(log_file=snakemake.log[0] if snakemake.log else None)

    # Get regions to average from config
    regions = snakemake.params.wirsenius_regions

    # Get net-to-ME conversion efficiency factors from config
    conversion_factors = snakemake.params.net_to_me_conversion
    k_m = conversion_factors["k_m"]
    k_g = conversion_factors["k_g"]
    k_l = conversion_factors["k_l"]

    # Get carcass-to-retail conversion factors from config
    carcass_to_retail = snakemake.params.carcass_to_retail

    build_feed_to_animal_products(
        wirsenius_file=snakemake.input.wirsenius,
        ruminant_categories_file=snakemake.input.ruminant_categories,
        monogastric_categories_file=snakemake.input.monogastric_categories,
        output_file=snakemake.output[0],
        regions_to_average=regions,
        k_m=k_m,
        k_g=k_g,
        k_l=k_l,
        carcass_to_retail=carcass_to_retail,
        feed_proxy_map=snakemake.params.feed_proxy_map,
    )
