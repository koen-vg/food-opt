# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Plot food consumption by health region, showing breakdowns by food and by food group."""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa

from workflow.scripts.constants import DAYS_PER_YEAR, GRAMS_PER_MEGATONNE
from workflow.scripts.plotting.color_utils import categorical_colors

logger = logging.getLogger(__name__)
EPSILON = 1e-6  # Threshold for considering values as zero
BAR_WIDTH = 0.35  # Width of each bar (we'll have 2 per cluster)
BAR_SPACING = 0.05  # Space between the two bars
CLUSTER_WIDTH_FACTOR = 1.2
MIN_FIGURE_WIDTH = 8
FIGURE_WIDTH_PADDING = 2
FIGURE_HEIGHT = 8


def _select_snapshot(network: pypsa.Network) -> str:
    if "now" in network.snapshots:
        return "now"
    if len(network.snapshots) == 1:
        return network.snapshots[0]
    raise ValueError("Expected snapshot 'now' or single snapshot")


def _get_cluster_map(clusters_path: str) -> dict[str, int]:
    df = pd.read_csv(clusters_path)
    if "country_iso3" not in df.columns or "health_cluster" not in df.columns:
        raise ValueError("Invalid clusters file format")
    return (
        df.assign(country_iso3=lambda x: x.country_iso3.str.upper())
        .set_index("country_iso3")["health_cluster"]
        .astype(int)
        .to_dict()
    )


def _get_cluster_population(
    population_path: str, cluster_map: dict[str, int]
) -> dict[int, float]:
    df = pd.read_csv(population_path)
    df["iso3"] = df["iso3"].str.upper()
    df["cluster"] = df["iso3"].map(cluster_map)
    return df.groupby("cluster")["population"].sum().to_dict()


def _aggregate_by_cluster(
    data: pd.DataFrame, clusters: list[int], value_column: str = "value_g_person_day"
) -> list[float]:
    """Aggregate values by cluster, maintaining cluster order."""
    if data.empty:
        return [0.0] * len(clusters)
    cluster_sums = data.groupby("cluster")[value_column].sum()
    return [cluster_sums.get(cluster, 0.0) for cluster in clusters]


def main() -> None:
    try:
        snakemake
    except NameError as exc:
        raise RuntimeError("Must be run via Snakemake") from exc

    network_path = snakemake.input.network
    population_path = snakemake.input.population
    clusters_path = snakemake.input.clusters
    food_groups_path = snakemake.input.food_groups
    output_pdf = Path(snakemake.output.pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    group_colors = snakemake.params.get("group_colors", {})

    # Load Data
    n = pypsa.Network(network_path)
    snapshot = _select_snapshot(n)
    cluster_map = _get_cluster_map(clusters_path)
    cluster_pop = _get_cluster_population(population_path, cluster_map)

    food_groups_df = pd.read_csv(food_groups_path)
    food_to_group = food_groups_df.set_index("food")["group"].to_dict()

    # Collect consumption data from consume links
    consumption_data = []  # (cluster, group, food, value_mt)

    # Process Consumption Links
    consume_links = n.links[n.links.index.str.startswith("consume_")]
    p0 = (
        n.links_t.p0.loc[snapshot]
        if "p0" in n.links_t
        else pd.Series(0, index=n.links.index)
    )

    for link in consume_links.index:
        val = p0.get(link, 0.0)
        if val <= EPSILON:
            continue

        # Extract food and country from link attributes
        food = n.links.at[link, "food"]
        country = n.links.at[link, "country"]

        cluster = cluster_map.get(country)
        if cluster is None:
            continue

        group = food_to_group.get(food, "Unknown")

        consumption_data.append(
            {
                "cluster": cluster,
                "group": group,
                "food": food,
                "value": val,
            }
        )

    # Convert to DataFrame
    df = pd.DataFrame(consumption_data)

    if df.empty:
        logger.warning("No consumption data to plot.")
        fig, ax = plt.subplots(figsize=(MIN_FIGURE_WIDTH, FIGURE_HEIGHT))
        ax.text(0.5, 0.5, "No consumption data", ha="center", va="center")
        ax.axis("off")
        fig.savefig(output_pdf, bbox_inches="tight")
        plt.close(fig)
        return

    # Get unique clusters and groups
    clusters = sorted(df["cluster"].unique())
    all_groups = sorted(df["group"].unique())

    # Validate population data
    missing_clusters = [c for c in clusters if c not in cluster_pop]
    if missing_clusters:
        raise ValueError(
            f"Population data missing for clusters: {missing_clusters}. "
            "Cannot compute per-capita values."
        )

    # Convert to g/person/day
    df["population"] = df["cluster"].map(cluster_pop)
    df["value_g_person_day"] = (
        df["value"] * GRAMS_PER_MEGATONNE / (df["population"] * DAYS_PER_YEAR)
    )

    # Create figure
    fig_width = max(
        MIN_FIGURE_WIDTH, len(clusters) * CLUSTER_WIDTH_FACTOR + FIGURE_WIDTH_PADDING
    )
    fig, ax = plt.subplots(figsize=(fig_width, FIGURE_HEIGHT))

    # Assign colors
    colors = categorical_colors(all_groups, group_colors)

    # X-axis positions for clusters
    x = np.arange(len(clusters))

    # --- BAR 1: Consumption by Food Group ---
    # For each cluster, stack by food group
    bottom_groups = np.zeros(len(clusters))
    for group in all_groups:
        group_data = df[df["group"] == group]
        group_vals = _aggregate_by_cluster(group_data, clusters)

        if np.sum(group_vals) > 0:
            ax.bar(
                x - BAR_WIDTH / 2 - BAR_SPACING / 2,
                group_vals,
                bottom=bottom_groups,
                width=BAR_WIDTH,
                color=colors.get(group, "#333333"),
                edgecolor="black",
                linewidth=0.5,
            )
            bottom_groups += np.array(group_vals)

    # --- BAR 2: Consumption by Specific Food ---
    # For each cluster, stack by specific food (colored by group)
    bottom_foods = np.zeros(len(clusters))

    # Hatches cycle for differentiating foods within groups
    hatches_cycle = [None, "//", "..", "xx", "\\\\", "||", "--", "++", "**", "OO"]
    food_legend_entries = []  # (food_name, patch)

    hatch_idx = 0
    for group in all_groups:
        # Get foods in this group
        group_foods = sorted(df[df["group"] == group]["food"].unique())

        for food in group_foods:
            food_data = df[df["food"] == food]
            food_vals = np.array(_aggregate_by_cluster(food_data, clusters))

            if np.sum(food_vals) == 0:
                continue

            # Use hatch to differentiate foods
            hatch = hatches_cycle[hatch_idx % len(hatches_cycle)]
            hatch_idx += 1

            ax.bar(
                x + BAR_WIDTH / 2 + BAR_SPACING / 2,
                food_vals,
                bottom=bottom_foods,
                width=BAR_WIDTH,
                color=colors.get(group, "#333333"),
                edgecolor="white",
                linewidth=0.5,
                hatch=hatch,
                alpha=0.9,
            )

            # Create legend entry
            proxy = Patch(
                facecolor=colors.get(group, "#333333"),
                edgecolor="black",
                alpha=0.9,
                hatch=hatch,
                linewidth=0.5,
            )
            food_legend_entries.append((food, proxy))

            bottom_foods += food_vals

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels([f"Cluster {c}" for c in clusters], rotation=0, ha="center")
    ax.set_ylabel("g / person / day")
    ax.set_title(
        "Food Consumption by Health Cluster\n(Left: by Food Group, Right: by Specific Food)"
    )
    ax.grid(axis="y", alpha=0.3)

    # --- Legends ---
    # 1. Food Groups (Colors)
    group_handles = []
    group_labels = []
    for group in all_groups:
        patch = Patch(
            facecolor=colors.get(group, "#333333"),
            edgecolor="black",
            linewidth=0.5,
        )
        group_handles.append(patch)
        group_labels.append(group)

    legend1 = ax.legend(
        group_handles,
        group_labels,
        title="Food Groups",
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        fontsize="small",
    )
    ax.add_artist(legend1)

    # 2. Specific Foods (Styles)
    food_handles = [e[1] for e in food_legend_entries]
    food_labels = [e[0] for e in food_legend_entries]

    ax.legend(
        food_handles,
        food_labels,
        title="Specific Foods",
        loc="upper left",
        bbox_to_anchor=(1.0, 0.6),
        fontsize="small",
        ncol=1 if len(food_labels) < 15 else 2,
    )

    fig.tight_layout(rect=[0, 0, 0.85, 1])
    fig.savefig(output_pdf, bbox_inches="tight")
    logger.info("Saved consumption balance plot to %s", output_pdf)


if __name__ == "__main__":
    main()
