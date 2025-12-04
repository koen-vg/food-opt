# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Plot dry-matter feed use by animal and feed category."""

import logging
from pathlib import Path

from color_utils import categorical_colors
import matplotlib
import pandas as pd
import pypsa

matplotlib.use("pdf")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


PRODUCT_TO_ANIMAL = {
    "meat-cattle": "Cattle",
    "dairy": "Cattle",
    "meat-pig": "Pigs",
    "meat-chicken": "Chicken",
    "eggs": "Chicken",
    "meat-sheep": "Sheep",
    "meat-goat": "Goats",
    "meat-buffalo": "Buffalo",
    "milk-sheep": "Sheep",
    "milk-goat": "Goats",
    "milk-buffalo": "Buffalo",
}

FEED_CATEGORY_LABELS = {
    "ruminant_grassland": "Grass & leaves",
    "ruminant_forage": "Fodder crops",
    "ruminant_roughage": "Crop residues",
    "ruminant_grain": "Grains",
    "ruminant_protein": "Oilseed cakes",
    "monogastric_grain": "Grains",
    "monogastric_energy": "Fodder crops",
    "monogastric_low_quality": "By-products",
    "monogastric_protein": "Oilseed cakes",
}

# Order used for both legend and bar stacking
FEED_ORDER = [
    "Grass & leaves",
    "Crop residues",
    "Fodder crops",
    "Oilseed cakes",
    "By-products",
    "Grains",
]

FEED_COLOR_OVERRIDES = {
    "Grass & leaves": "#4f9d69",
    "Crop residues": "#8c6b4f",
    "Fodder crops": "#a6d96a",
    "Oilseed cakes": "#b8de6f",
    "By-products": "#7b6ba8",
    "Grains": "#d95f02",
}


def _map_animal(product: str) -> str:
    """Convert model product names to display animal categories."""

    if not product:
        return "Unknown"

    if product in PRODUCT_TO_ANIMAL:
        return PRODUCT_TO_ANIMAL[product]

    if product.startswith("meat-"):
        return product.split("-", 1)[1].replace("_", " ").title()

    return str(product).replace("_", " ").title()


def _map_feed_category(bus0: str) -> str:
    """Convert feed bus names to human-friendly categories."""

    raw = str(bus0)
    if raw.startswith("feed_"):
        raw = raw[5:]

    # Try exact match first
    if raw in FEED_CATEGORY_LABELS:
        return FEED_CATEGORY_LABELS[raw]

    # Try stripping the country suffix (last underscore component)
    if "_" in raw:
        prefix = raw.rsplit("_", 1)[0]
        if prefix in FEED_CATEGORY_LABELS:
            return FEED_CATEGORY_LABELS[prefix]

        # If not in labels, return the prefix formatted
        return prefix.replace("_", " ").title()

    return FEED_CATEGORY_LABELS.get(raw, raw.replace("_", " ").title())


def _extract_feed_use(n: pypsa.Network) -> pd.DataFrame:
    """Return long-form feed use (Mt DM) by animal and feed category."""

    if n.links.empty:
        return pd.DataFrame(columns=["animal", "feed_category", "feed_mt"])

    links_df = n.links
    feed_links = links_df[
        links_df["product"].notna() & links_df["bus0"].str.startswith("feed_")
    ]

    if feed_links.empty:
        logger.info("No feed→animal links found in network")
        return pd.DataFrame(columns=["animal", "feed_category", "feed_mt"])

    # Sum over snapshots with objective weightings (keeps units in Mt/year)
    if "p0" not in n.links_t:
        logger.warning("Network is missing link flow data (p0); returning empty frame")
        return pd.DataFrame(columns=["animal", "feed_category", "feed_mt"])

    weights = n.snapshot_weightings["objective"]
    p0 = n.links_t.p0.loc[:, feed_links.index]
    weighted = p0.multiply(weights, axis=0)
    totals = weighted.sum()

    records = []
    for link_name, value in totals.items():
        flow_mt = abs(float(value))
        if flow_mt <= 0.0:
            continue

        product = str(feed_links.at[link_name, "product"])
        animal = _map_animal(product)

        # Skip links without a valid animal product (e.g., trade links)
        if animal == "Unknown":
            continue

        feed_category = _map_feed_category(feed_links.at[link_name, "bus0"])

        records.append(
            {
                "animal": animal,
                "feed_category": feed_category,
                "feed_mt": flow_mt,
            }
        )

    return pd.DataFrame(records)


def _pivot_for_plot(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-form feed use into animal x feed_category wide format."""

    if df.empty:
        return pd.DataFrame()

    wide = df.pivot_table(
        index="animal",
        columns="feed_category",
        values="feed_mt",
        aggfunc="sum",
        fill_value=0.0,
    )

    totals = wide.sum(axis=1).sort_values(ascending=False)
    wide = wide.loc[totals.index]

    ordered_cols = [cat for cat in FEED_ORDER if cat in wide.columns]
    unordered_cols = [c for c in wide.columns if c not in ordered_cols]
    return wide[ordered_cols + unordered_cols]


def _plot_feed_breakdown(wide: pd.DataFrame, output_pdf: Path) -> None:
    """Render stacked horizontal bars of feed use."""

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    if wide.empty:
        ax.text(0.5, 0.5, "No feed flows in network", ha="center", va="center")
        ax.axis("off")
    else:
        categories = list(wide.columns)
        colors = categorical_colors(categories, overrides=FEED_COLOR_OVERRIDES)
        left = pd.Series(0.0, index=wide.index)

        for cat in categories:
            values = wide[cat]
            ax.barh(
                wide.index,
                values,
                left=left,
                color=colors[cat],
                edgecolor="black",
                linewidth=0.4,
                label=cat,
            )
            left = left + values

        ax.set_xlabel("Mt of DM")
        ax.set_ylabel("Animal")
        ax.invert_yaxis()  # Largest animals at top
        ax.grid(axis="x", alpha=0.3)
        ax.legend(
            title="Feed",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0,
        )
        plt.tight_layout()

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_pdf, bbox_inches="tight", dpi=300)
    plt.close()
    logger.info("Wrote feed breakdown plot to %s", output_pdf)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    network = pypsa.Network(snakemake.input.network)

    feed_long = _extract_feed_use(network)

    wide = _pivot_for_plot(feed_long)

    csv_path = Path(snakemake.output.csv)
    pdf_path = Path(snakemake.output.pdf)

    feed_long.to_csv(csv_path, index=False)
    logger.info("Wrote feed breakdown table to %s", csv_path)

    _plot_feed_breakdown(wide, pdf_path)
