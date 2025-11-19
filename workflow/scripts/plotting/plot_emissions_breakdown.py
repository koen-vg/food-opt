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

# Enable new PyPSA components API
pypsa.options.api.new_components_api = True


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
    carrier_map = {
        "residue_incorporation": "Crop residue incorporation",
        "spared_land": "Carbon sequestration",
        "fertilizer": "Synthetic fertilizer application",
    }

    if carrier in carrier_map:
        return carrier_map[carrier]

    # Pattern-based categorization
    if carrier.startswith("crop_"):
        if bus_carrier == "ch4":
            return "Rice cultivation"
        return "Crop production"
    elif carrier.startswith("multi_crop_"):
        return "Multi-cropping"
    elif carrier.startswith("produce_"):
        # Animal production carriers
        if bus_carrier == "n2o":
            return "Manure management & application"
        elif bus_carrier == "ch4":
            # Combined enteric + manure CH4
            return "Enteric fermentation & Manure management"
        return "Livestock production"
    elif carrier.startswith("feed_"):
        return "Grassland"
    elif carrier.startswith("food_"):
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
        "CO₂": defaultdict(float),
        "CH₄": defaultdict(float),
        "N₂O": defaultdict(float),
    }

    # GWP factors for each gas
    gwp_factors = {
        "co2": ("CO₂", 1.0),
        "ch4": ("CH₄", ch4_gwp),
        "n2o": ("N₂O", n2o_gwp),
    }

    # Carriers representing conversion links to be excluded (sinks)
    conversion_carriers = {"co2", "ch4", "n2o"}

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
        value_mt = value * 1e-6 if gas_name in ["CH₄", "N₂O"] else value

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

    return emissions


def plot_emissions_breakdown(
    emissions: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    """Create side-by-side stacked bar plots for each gas in CO2eq units.

    Parameters
    ----------
    emissions : dict[str, dict[str, float]]
        Emissions data by gas and source (all in MtCO2eq)
    output_path : Path
        Path to save the PDF plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)

    # Define consistent colors for sources across all plots
    source_colors = {
        "Crop production": "#8dd3c7",
        "Multi-cropping": "#80b1d3",
        "Livestock production": "#fb8072",
        "Enteric fermentation & Manure management": "#fb8072",  # Same color as livestock
        "Manure management & application": "#d95f02",  # Darker orange/brown for manure N2O
        "Grassland": "#bebada",
        "Crop residue incorporation": "#fdb462",
        "Synthetic fertilizer application": "#b3de69",
        "Rice cultivation": "#a6cee3",  # Light blue for flooded rice methane
        "Carbon sequestration": "#fccde5",
        "Food processing": "#ffffb3",
        "Trade": "#bc80bd",
    }

    fig.suptitle(
        "Global Emissions Breakdown by Source", fontsize=16, fontweight="bold", y=1.02
    )

    # Plot each gas
    for idx, (gas, gas_data) in enumerate(emissions.items()):
        ax = axes[idx]

        if not gas_data:
            ax.text(0.5, 0.5, f"No {gas} emissions", ha="center", va="center")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            continue

        # Sort sources by total emission (descending)
        sorted_sources = sorted(
            gas_data.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        # Separate positive (emissions) and negative (sequestration) values
        positive_sources = [(s, v) for s, v in sorted_sources if v > 0]
        negative_sources = [(s, v) for s, v in sorted_sources if v < 0]

        # Stack positive emissions
        bottom = 0.0
        bars = []
        for source, value in positive_sources:
            color = source_colors.get(source, "#d9d9d9")
            bar = ax.bar(
                0,
                value,
                bottom=bottom,
                color=color,
                edgecolor="black",
                linewidth=0.5,
                label=source,
            )
            bars.append(bar)
            bottom += value

        # Stack negative emissions (sequestration) below zero
        bottom = 0.0
        for source, value in negative_sources:
            color = source_colors.get(source, "#d9d9d9")
            bar = ax.bar(
                0,
                value,
                bottom=bottom,
                color=color,
                edgecolor="black",
                linewidth=0.5,
                label=source,
            )
            bars.append(bar)
            bottom += value

        # Set title and labels
        ax.set_title(gas, fontsize=14, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("Emissions (MtCO₂eq)", fontsize=12)
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])

        # Add gridlines
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.axhline(y=0, color="black", linewidth=0.8)

    # Create a single legend for all subplots
    # Collect all unique sources that appear in any plot
    all_sources = set()
    for gas_data in emissions.values():
        all_sources.update(gas_data.keys())

    # Sort sources for consistent legend order
    sorted_all_sources = sorted(all_sources)

    # Create legend handles
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=source_colors.get(source, "#d9d9d9"))
        for source in sorted_all_sources
    ]

    fig.legend(
        legend_handles,
        sorted_all_sources,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        frameon=False,
        fontsize=10,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

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
    plot_emissions_breakdown(emissions, pdf_path)
