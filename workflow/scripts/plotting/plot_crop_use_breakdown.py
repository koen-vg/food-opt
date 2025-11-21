# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Create a stacked bar chart of global crop uses (human vs. animal feed)."""

from collections import defaultdict
import logging
from pathlib import Path

import matplotlib
import pandas as pd
import pypsa

matplotlib.use("pdf")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def _extract_crop_production(
    n: pypsa.Network,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Aggregate crop production (megatonnes) over all regions/resource classes.

    Returns total, irrigated, and rainfed production separately.
    """

    if "now" not in n.snapshots:
        raise ValueError("Expected snapshot 'now' in solved network")

    production = {}
    irrigated = {}
    rainfed = {}
    flows = n.links_t.p1.loc["now"] if "p1" in n.links_t else pd.Series(dtype=float)
    links_df = n.links

    for link in n.links.index:
        if not link.startswith("produce_"):
            continue

        bus1 = str(links_df.at[link, "bus1"])
        if bus1.startswith(("co2", "ch4")):
            continue
        if not bus1.startswith("crop_"):
            continue

        # Extract crop from carrier (e.g., "crop_maize" -> "maize")
        carrier = str(links_df.at[link, "carrier"])
        if not carrier.startswith("crop_"):
            continue
        crop_token = carrier[5:]  # Remove "crop_" prefix

        # Get water supply from link attribute
        water_supply = str(links_df.at[link, "water_supply"])

        value = abs(float(flows.get(link, 0.0)))
        if value <= 0.0:
            continue

        production[crop_token] = production.get(crop_token, 0.0) + value

        if water_supply == "irrigated":
            irrigated[crop_token] = irrigated.get(crop_token, 0.0) + value
        elif water_supply == "rainfed":
            rainfed[crop_token] = rainfed.get(crop_token, 0.0) + value

    pasture_total = 0.0
    for link in n.links.index:
        if not (link.startswith(("grassland_region", "grassland_marginal"))):
            continue

        value = abs(float(flows.get(link, 0.0)))
        if value <= 0.0:
            continue

        pasture_total += value

    if pasture_total > 0.0:
        production["grassland"] = pasture_total
        rainfed["grassland"] = pasture_total

    return (
        pd.Series(production, dtype=float),
        pd.Series(irrigated, dtype=float),
        pd.Series(rainfed, dtype=float),
    )


def _extract_crop_use(n: pypsa.Network) -> tuple[pd.Series, pd.Series]:
    """Return crop use split into human consumption vs. animal feed (megatonnes)."""

    if "now" not in n.snapshots:
        raise ValueError("Expected snapshot 'now' in solved network")

    human_use: defaultdict[str, float] = defaultdict(float)
    feed_use: defaultdict[str, float] = defaultdict(float)

    flows_p0 = n.links_t.p0.loc["now"] if "p0" in n.links_t else pd.Series(dtype=float)
    flows_p1 = n.links_t.p1.loc["now"] if "p1" in n.links_t else pd.Series(dtype=float)
    links_df = n.links

    for link in n.links.index:
        flow_in = abs(float(flows_p0.get(link, 0.0)))
        if flow_in <= 0.0:
            continue

        bus0 = str(links_df.at[link, "bus0"])
        bus1 = str(links_df.at[link, "bus1"])
        carrier = str(links_df.at[link, "carrier"])

        if link.startswith("consume_"):
            # Use food attribute from consumption link
            crop = str(links_df.at[link, "food"])
            human_use[crop] += flow_in
        elif link.startswith("convert_") and (
            bus1.startswith("feed_") or carrier.startswith(("convert_to_feed", "feed_"))
        ):
            # Extract crop from bus0 (e.g., "crop_maize_USA" -> "maize")
            if bus0.startswith("crop_"):
                crop_with_country = bus0[5:]  # Remove "crop_" prefix
                crop = crop_with_country.rsplit("_", 1)[0]  # Remove country suffix
                feed_use[crop] += flow_in
        else:
            continue

    pasture_total = 0.0
    for link in n.links.index:
        if not (link.startswith(("grassland_region", "grassland_marginal"))):
            continue

        value = abs(float(flows_p1.get(link, 0.0)))
        if value <= 0.0:
            continue

        pasture_total += value

    if pasture_total > 0.0:
        feed_use["grassland"] = feed_use.get("grassland", 0.0) + pasture_total

    return pd.Series(human_use, dtype=float), pd.Series(feed_use, dtype=float)


def _build_dataframe(
    production: pd.Series,
    irrigated: pd.Series,
    rainfed: pd.Series,
    human_use: pd.Series,
    feed_use: pd.Series,
) -> pd.DataFrame:
    """Combine the individual series into a single dataframe for plotting/export."""

    df = pd.DataFrame(
        {
            "production_mt": production,
            "irrigated_mt": irrigated,
            "rainfed_mt": rainfed,
            "human_consumption_mt": human_use,
            "animal_feed_mt": feed_use,
        }
    ).fillna(0.0)

    if df.empty:
        return df

    df = df[(df > 0).any(axis=1)]

    df["irrigated_fraction"] = df["irrigated_mt"] / df["production_mt"].replace(
        0, float("nan")
    )

    df["residual_mt"] = df["production_mt"] - df[
        ["human_consumption_mt", "animal_feed_mt"]
    ].sum(axis=1)
    tolerance = 1e-6
    residual_mask = df["residual_mt"].abs() > tolerance
    if residual_mask.any():
        logger.warning(
            "Crop use does not sum to production for: %s",
            ", ".join(df.index[residual_mask]),
        )

    df.sort_values("production_mt", ascending=False, inplace=True)
    return df


def _plot(df: pd.DataFrame, output_pdf: Path) -> None:
    """Render stacked bar chart for crop uses with irrigation info."""

    plt.figure(figsize=(12, 7))

    if df.empty:
        plt.text(0.5, 0.5, "No crop production data", ha="center", va="center")
        plt.axis("off")
    else:
        x = range(len(df))
        human = df["human_consumption_mt"].to_numpy()
        feed = df["animal_feed_mt"].to_numpy()
        irrigated_frac = df["irrigated_fraction"].fillna(0).to_numpy()

        human_rainfed = human * (1 - irrigated_frac)
        human_irrigated = human * irrigated_frac
        feed_rainfed = feed * (1 - irrigated_frac)
        feed_irrigated = feed * irrigated_frac

        plt.bar(
            x,
            human_rainfed,
            label="Human consumption (rainfed)",
            color="#1f77b4",
            edgecolor="black",
            linewidth=0.5,
        )
        plt.bar(
            x,
            human_irrigated,
            bottom=human_rainfed,
            label="Human consumption (irrigated)",
            color="#1f77b4",
            hatch="///",
            edgecolor="black",
            linewidth=0.5,
        )
        plt.bar(
            x,
            feed_rainfed,
            bottom=human,
            label="Animal feed (rainfed)",
            color="#ff7f0e",
            edgecolor="black",
            linewidth=0.5,
        )
        plt.bar(
            x,
            feed_irrigated,
            bottom=human + feed_rainfed,
            label="Animal feed (irrigated)",
            color="#ff7f0e",
            hatch="///",
            edgecolor="black",
            linewidth=0.5,
        )

        plt.ylabel("Megatonnes (Mt)")
        plt.xticks(x, df.index, rotation=45, ha="right")
        plt.grid(axis="y", alpha=0.3)
        plt.legend()
        plt.tight_layout()

    plt.title("Global crop use breakdown (hatched = irrigated)")
    plt.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_pdf, bbox_inches="tight", dpi=300)
    plt.close()
    logger.info("Wrote stacked crop use plot to %s", output_pdf)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    network = pypsa.Network(snakemake.input.network)
    production, irrigated, rainfed = _extract_crop_production(network)
    human_use, feed_use = _extract_crop_use(network)

    df = _build_dataframe(production, irrigated, rainfed, human_use, feed_use)

    csv_path = Path(snakemake.output.csv)
    pdf_path = Path(snakemake.output.pdf)

    df.to_csv(csv_path)
    logger.info("Wrote crop use data to %s", csv_path)

    _plot(df, pdf_path)
