# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Plot emissions breakdown by source for CO2, CH4, and N2O."""

from collections import defaultdict
import logging
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
import pandas as pd
import pypsa

logger = logging.getLogger(__name__)


def categorize_emission_carrier(carrier: str, bus_carrier: str) -> str:
    """Categorize an emission source by its carrier and gas type.

    Parameters
    ----------
    carrier : str
        Link carrier name
    bus_carrier : str
        The emission bus being fed ("co2", "ch4", "n2o")

    Returns
    -------
    str
        Category name for plotting
    """
    # Map specific carriers to categories based on documentation
    # Note: use link carriers, not bus/store carriers
    carrier_map = {
        "residue_incorporation": "Crop residue incorporation",
        "spare_land": "Carbon sequestration",  # Link carrier (not "spared_land" which is bus/store)
        "fertilizer_distribution": "Synthetic fertilizer application",
        "land_conversion": "Land Use Change",  # Link carrier for land expansion
    }

    if carrier in carrier_map:
        return carrier_map[carrier]

    # Pattern-based categorization
    if carrier == "crop_production":
        if bus_carrier == "ch4":
            return "Rice cultivation"
        if bus_carrier == "co2":
            return "Land Use Change"
        return "Crop production"
    elif carrier == "crop_production_multi":
        if bus_carrier == "co2":
            return "Land Use Change"
        return "Multi-cropping"
    elif carrier == "animal_production":
        # Animal production carrier
        if bus_carrier == "n2o":
            return "Manure management & application"
        elif bus_carrier == "ch4":
            # Combined enteric + manure CH4
            return "Enteric fermentation & Manure management"
        return "Livestock production"
    elif carrier == "grassland_production":
        if bus_carrier == "co2":
            return "Land Use Change"
        return "Grassland"
    elif carrier == "food_processing":
        return "Food processing"
    elif carrier.startswith("trade_"):
        return "Trade"
    else:
        # Return carrier name for unknown types
        return f"Other ({carrier})"


def extract_emissions_by_source(
    n: pypsa.Network,
    ch4_gwp: float,
    n2o_gwp: float,
) -> dict[str, dict[str, float]]:
    """Extract emissions by gas type and source category in CO2eq units.

    Uses n.statistics.energy_balance() to efficiently extract emission flows.
    Excludes conversion links (co2, ch4, n2o) that move emissions to the GHG bus.

    Parameters
    ----------
    n : pypsa.Network
        Solved network
    ch4_gwp : float
        Global warming potential for CH4 (kg CO2eq / kg CH4)
    n2o_gwp : float
        Global warming potential for N2O (kg CO2eq / kg N2O)

    Returns
    -------
    dict[str, dict[str, float]]
        Nested dict: {gas: {source: amount}}
        All values in MtCO2eq
    """
    # Initialize nested dict for emissions by gas and source
    emissions: dict[str, dict[str, float]] = {
        "CO2": defaultdict(float),
        "CH4": defaultdict(float),
        "N2O": defaultdict(float),
    }

    # GWP factors for each gas
    gwp_factors = {
        "co2": ("CO2", 1.0),
        "ch4": ("CH4", ch4_gwp),
        "n2o": ("N2O", n2o_gwp),
    }

    # Carriers representing conversion links to be excluded (sinks)
    # - co2, ch4, n2o: links that feed into individual gas buses
    # - emission_aggregation: links that move emissions from gas buses to GHG bus
    conversion_carriers = {"co2", "ch4", "n2o", "emission_aggregation"}

    # Get energy balance with grouping by bus_carrier and carrier
    # This gives us flows into each bus, grouped by component carrier
    try:
        balance = n.statistics.energy_balance(groupby=["bus_carrier", "carrier"])
    except Exception as e:
        logger.error("Failed to compute energy balance: %s", e)
        return emissions

    # The balance is a multi-indexed Series with (component, bus_carrier, carrier)
    # We want to extract flows into co2, ch4, and n2o buses
    for (_component, bus_carrier, carrier), value in balance.items():
        # Skip if not an emission bus
        if bus_carrier not in gwp_factors:
            continue

        # Skip conversion links (which appear as negative flows/sinks)
        if carrier in conversion_carriers:
            continue

        # Skip zero or negligible flows
        if abs(value) < 1e-9:
            continue

        gas_name, gwp_factor = gwp_factors[bus_carrier]

        # Convert to CO2eq; CH4 and N2O flows are in tonnes
        value_mt = value * 1e-6 if gas_name in ["CH4", "N2O"] else value

        emission_co2eq = value_mt * gwp_factor

        # Categorize by carrier, passing the bus_carrier (gas type) context
        category = categorize_emission_carrier(carrier, bus_carrier)

        # Add to the appropriate category
        emissions[gas_name][category] += emission_co2eq
        logger.debug(
            "Added %.3f MtCO2eq of %s from %s (carrier: %s)",
            emission_co2eq,
            gas_name,
            category,
            carrier,
        )

    # --- Split manure N2O into pasture vs managed using link-level shares ------
    # Each animal production link has a pasture_n2o_share attribute that gives
    # the fraction of its N2O coming from pasture deposition vs managed systems.
    # This is based on MMS (Manure Management System) distributions from GLEAM.
    if "Manure management & application" in emissions.get("N2O", {}):
        links_df = n.links.static
        produce_mask = links_df.carrier == "animal_production"
        pasture_share = (
            links_df.loc[produce_mask, "pasture_n2o_share"].fillna(0.0).astype(float)
        )

        p4 = n.links.dynamic["p4"].loc[:, produce_mask]
        weights = n.snapshot_weightings["objective"]
        pasture_t_n2o = -(
            p4.multiply(pasture_share, axis=1).multiply(weights, axis=0).sum().sum()
        )
        pasture_mtco2eq = pasture_t_n2o * n2o_gwp * 1e-6

        total_mtco2eq = emissions["N2O"].get("Manure management & application", 0.0)
        managed_mtco2eq = max(total_mtco2eq - pasture_mtco2eq, 0.0)

        emissions["N2O"].pop("Manure management & application", None)
        emissions["N2O"]["Manure: pasture deposition"] = pasture_mtco2eq
        emissions["N2O"]["Manure: managed systems"] = managed_mtco2eq

    # --- Split CH4 into enteric vs manure using link-level shares -------------
    if "Enteric fermentation & Manure management" in emissions.get("CH4", {}):
        links_df = n.links.static
        produce_mask = links_df.carrier == "animal_production"
        manure_share = (
            links_df.loc[produce_mask, "manure_ch4_share"].fillna(0.0).astype(float)
        )

        p2 = n.links.dynamic["p2"].loc[:, produce_mask]
        weights = n.snapshot_weightings["objective"]
        manure_t_ch4 = -(
            p2.multiply(manure_share, axis=1).multiply(weights, axis=0).sum().sum()
        )
        manure_mtco2eq = manure_t_ch4 * ch4_gwp * 1e-6

        total_mtco2eq = emissions["CH4"].get(
            "Enteric fermentation & Manure management", 0.0
        )
        enteric_mtco2eq = max(total_mtco2eq - manure_mtco2eq, 0.0)

        emissions["CH4"].pop("Enteric fermentation & Manure management", None)
        emissions["CH4"]["Enteric fermentation"] = enteric_mtco2eq
        emissions["CH4"]["Manure: managed systems"] = manure_mtco2eq

    return emissions


def process_faostat_emissions(
    faostat_df: pd.DataFrame,
    ch4_gwp: float,
    n2o_gwp: float,
) -> dict[str, dict[str, float]]:
    """Process raw FAOSTAT emissions data into a categorized dict in MtCO2eq.

    Parameters
    ----------
    faostat_df : pd.DataFrame
        Raw FAOSTAT GT emissions data.
    ch4_gwp : float
        Global warming potential for CH4 (kg CO2eq / kg CH4)
    n2o_gwp : float
        Global warming potential for N2O (kg CO2eq / kg N2O)

    Returns
    -------
    dict[str, dict[str, float]]
        Nested dict: {gas: {source: amount}}
        All values in MtCO2eq
    """
    faostat_emissions: dict[str, dict[str, float]] = {
        "CO2": defaultdict(float),
        "CH4": defaultdict(float),
        "N2O": defaultdict(float),
    }

    # Mapping of FAOSTAT items to our categories
    item_to_category = {
        "Crop Residues": "Crop residue incorporation",
        "Rice Cultivation": "Rice cultivation",
        "Burning - Crop residues": "Crop residue burning",
        "Synthetic Fertilizers": "Synthetic fertilizer application",
        "Drained organic soils": "Drained organic soils",
        "Drained organic soils (CO2)": "Drained organic soils",  # Handle variants
        "Drained organic soils (N2O)": "Drained organic soils",
        "Enteric Fermentation": "Enteric fermentation",
        "Manure Management": "Manure: managed systems",
        "Manure applied to Soils": "Manure: managed systems",
        "Manure left on Pasture": "Manure: pasture deposition",
        "Net Forest conversion": "Land Use Change",  # Positive emission
        # "Forestland": "Carbon sequestration",  # Excluded: represents standing forest sink
        # "Food Processing": "Food processing", # Excluded per user request
        # "Food Transport": "Trade", # Excluded per user request
        # "On-farm energy use": "Other (On-farm energy use)", # Excluded per user request
    }

    # Mapping of FAOSTAT elements to gas types
    element_to_gas = {
        "Emissions (CH4)": ("CH4", ch4_gwp),
        "Emissions (N2O)": ("N2O", n2o_gwp),
        "Emissions (CO2)": ("CO2", 1.0),
    }

    for _, row in faostat_df.iterrows():
        item = row["item"]
        element = row["element"]
        value_kt = row["value_kt"]

        # Handle "Drained organic soils" which might appear with suffix in some datasets or processing
        category = item_to_category.get(item)
        if category is None:
            # Try matching prefix for drained organic soils
            if item.startswith("Drained organic soils"):
                category = "Drained organic soils"
            else:
                logger.debug("Skipping unknown FAOSTAT item: %s", item)
                continue

        gas_info = element_to_gas.get(element)
        if gas_info is None:
            logger.debug("Skipping unknown FAOSTAT element: %s", element)
            continue

        gas_name, gwp_factor = gas_info

        # Convert kilotonnes to Mt, then to MtCO2eq
        value_mtco2eq = value_kt * 1e-3 * gwp_factor

        faostat_emissions[gas_name][category] += value_mtco2eq

    return faostat_emissions


def load_emissions_csv(path: Path) -> dict[str, dict[str, float]]:
    """Load emissions CSV (gas, source, emissions_mtco2eq) into nested dict."""

    df = pd.read_csv(path, comment="#")
    result: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for _, row in df.iterrows():
        result[row["gas"]][row["source"]] += float(row["emissions_mtco2eq"])
    return {gas: dict(srcs) for gas, srcs in result.items()}


def plot_emissions_breakdown(
    emissions: dict[str, dict[str, float]],
    faostat_emissions: dict[str, dict[str, float]],
    gleam_emissions: dict[str, dict[str, float]] | None,
    output_path: Path,
) -> None:
    """Create side-by-side stacked bar plots for each gas in CO2eq units, comparing modeled, FAOSTAT, and GLEAM.

    Parameters
    ----------
    emissions : dict[str, dict[str, float]]
        Modeled emissions data by gas and source (all in MtCO2eq)
    faostat_emissions : dict[str, dict[str, float]]
        FAOSTAT actual emissions data by gas and source (all in MtCO2eq)
    output_path : Path
        Path to save the PDF plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 7), sharey=True)

    fig.suptitle(
        "Global Emissions Breakdown by Source: Modeled vs. FAOSTAT",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    # Define gas-specific base colormaps
    gas_cmaps = {
        "CO2": "Greys",
        "CH4": "Greens",
        "N2O": "Oranges",
    }

    # Iterate through gases
    for idx, gas in enumerate(["CO2", "CH4", "N2O"]):
        ax = axes[idx]

        modeled_data = emissions.get(gas, {})
        actual_data = faostat_emissions.get(gas, {})
        gleam_data = gleam_emissions.get(gas, {}) if gleam_emissions else {}

        all_sources = sorted(
            set(modeled_data.keys()) | set(actual_data.keys()) | set(gleam_data.keys())
        )
        n_cats = len(all_sources)

        if not modeled_data and not actual_data:
            ax.text(0.5, 0.5, f"No {gas} emissions", ha="center", va="center")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            continue

        # Get colormap for this gas
        cmap_name = gas_cmaps.get(gas, "Blues")
        cmap = matplotlib.colormaps[cmap_name]

        # Generate a range of colors for categories within this gas
        if n_cats <= 1:
            colors_for_categories = [cmap(0.6)]
        else:
            # Use a range from 0.3 to 0.9 for shades, so largest gets darkest (0.9)
            colors_for_categories = [
                cmap(0.3 + 0.6 * i / (n_cats - 1)) for i in range(n_cats)
            ]
        colors_for_categories.reverse()  # Largest value gets darker shade
        category_colors = dict(zip(all_sources, colors_for_categories))

        bar_width = 0.5
        x_modeled = 0.0
        x_actual = 1.0
        x_gleam = 2.0

        def stacked_bar(
            x_pos: float,
            data: dict[str, float],
            sources: list[str],
            colors: dict[str, str],
            axis: plt.Axes,
            width: float,
        ) -> None:
            bottom_pos = 0.0
            bottom_neg = 0.0
            for source in sources:
                value = data.get(source, 0.0)
                color = colors.get(source, "#d9d9d9")
                if value > 0:
                    axis.bar(
                        x_pos,
                        value,
                        bottom=bottom_pos,
                        width=width,
                        color=color,
                        edgecolor="black",
                        linewidth=0.5,
                        label=source
                        if source not in axis.get_legend_handles_labels()[1]
                        else "",
                    )
                    bottom_pos += value
                elif value < 0:
                    axis.bar(
                        x_pos,
                        value,
                        bottom=bottom_neg,
                        width=width,
                        color=color,
                        edgecolor="black",
                        linewidth=0.5,
                        label=source
                        if source not in axis.get_legend_handles_labels()[1]
                        else "",
                    )
                    bottom_neg += value

        stacked_bar(
            x_modeled, modeled_data, all_sources, category_colors, ax, bar_width
        )
        stacked_bar(x_actual, actual_data, all_sources, category_colors, ax, bar_width)
        if gleam_emissions is not None:
            stacked_bar(
                x_gleam, gleam_data, all_sources, category_colors, ax, bar_width
            )

        # Set title and labels
        ax.set_title(gas, fontsize=14, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("Emissions (MtCO₂eq)", fontsize=12)
        xticks = [x_modeled, x_actual]
        xticklabels = ["Modeled", "FAOSTAT"]
        if gleam_emissions is not None:
            xticks.append(x_gleam)
            xticklabels.append("GLEAM\n(livestock)")
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=10)
        # Add some horizontal padding so bars sit centered under their labels
        xmin = x_modeled - bar_width * 0.75
        xmax = (
            x_gleam + bar_width * 0.75
            if gleam_emissions is not None
            else x_actual + bar_width * 0.75
        )
        ax.set_xlim(xmin, xmax)

        # Add gridlines
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.axhline(y=0, color="black", linewidth=0.8)

        # Add individual legend for this gas, sorted in reverse
        # Filter out empty labels from the bars to avoid duplicate entries
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = []
        unique_handles = []
        for handle, label in zip(handles, labels):
            if label and label not in unique_labels:
                unique_labels.append(label)
                unique_handles.append(handle)

        # Sort unique_labels based on the order of all_sources
        sorted_unique_labels = [
            label for label in all_sources if label in unique_labels
        ]
        sorted_unique_handles = [
            unique_handles[unique_labels.index(label)] for label in sorted_unique_labels
        ]

        ax.legend(
            reversed(sorted_unique_handles),
            reversed(sorted_unique_labels),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=1,
            frameon=False,
            fontsize=8,
            title="Sources",
            title_fontsize=9,
        )

    # Adjust layout to prevent legends from overlapping
    plt.tight_layout()
    plt.subplots_adjust(
        bottom=0.35, wspace=0.3
    )  # Increase bottom margin significantly for legends

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info("Wrote emissions breakdown plot to %s", output_path)


def save_emissions_table(
    emissions: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    """Save emissions data as a CSV table.

    Parameters
    ----------
    emissions : dict[str, dict[str, float]]
        Emissions data by gas and source (all in MtCO2eq)
    output_path : Path
        Path to save the CSV file
    """
    # Convert nested dict to DataFrame
    rows = []
    for gas, sources in emissions.items():
        for source, amount in sources.items():
            rows.append(
                {
                    "gas": gas,
                    "source": source,
                    "emissions_mtco2eq": amount,
                }
            )

    df = pd.DataFrame(rows)

    if df.empty:
        df = pd.DataFrame(columns=["gas", "source", "emissions_mtco2eq"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Wrote emissions breakdown table to %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    network = pypsa.Network(snakemake.input.network)
    ch4_gwp = float(snakemake.params.ch4_gwp)
    n2o_gwp = float(snakemake.params.n2o_gwp)

    logger.info("Extracting emissions from network using energy balance statistics")
    emissions = extract_emissions_by_source(network, ch4_gwp, n2o_gwp)

    logger.info("Loading and processing FAOSTAT emissions data")
    faostat_emissions_df = pd.read_csv(snakemake.input.faostat_emissions)
    faostat_emissions_processed = process_faostat_emissions(
        faostat_emissions_df, ch4_gwp, n2o_gwp
    )

    gleam_emissions = load_emissions_csv(Path(snakemake.input.gleam_emissions))

    # Log summary
    for gas, sources in emissions.items():
        total = sum(sources.values())
        logger.info("%s total: %.2f MtCO2eq", gas, total)
        for source, amount in sorted(
            sources.items(), key=lambda x: abs(x[1]), reverse=True
        ):
            logger.info("  %s: %.2f MtCO2eq", source, amount)

    pdf_path = Path(snakemake.output.pdf)
    csv_path = Path(snakemake.output.csv)

    save_emissions_table(emissions, csv_path)
    plot_emissions_breakdown(
        emissions,
        faostat_emissions_processed,
        gleam_emissions,
        pdf_path,
    )
