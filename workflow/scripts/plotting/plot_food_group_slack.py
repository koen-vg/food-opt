# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Plot positive/negative food group slack aggregated globally (Mt)."""

import logging
from pathlib import Path

import matplotlib
import pandas as pd
import pypsa

matplotlib.use("pdf")
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

try:  # Prefer package import when available (e.g., during documentation builds)
    from workflow.scripts.color_utils import categorical_colors
except ImportError:  # Fallback to Snakemake's script-directory loader
    from color_utils import categorical_colors  # type: ignore


logger = logging.getLogger(__name__)

POSITIVE_PREFIX = "slack_positive_group_"
NEGATIVE_PREFIX = "slack_negative_group_"


def _snapshot_weights(network: pypsa.Network) -> pd.Series:
    """Return per-snapshot weights; defaults to ones if missing."""

    weights = network.snapshot_weightings.get("objective")
    if weights is None:
        return pd.Series(1.0, index=network.snapshots)
    return weights


def _aggregate_positive_slack(network: pypsa.Network) -> pd.Series:
    """Aggregate positive (shortage) slack by food group in Mt."""

    generators = network.generators
    if generators.empty or "carrier" not in generators:
        return pd.Series(dtype=float)

    mask = generators["carrier"].astype(str).str.startswith(POSITIVE_PREFIX)
    if not mask.any():
        return pd.Series(dtype=float)

    dispatch = network.generators_t.p.loc[:, mask]
    weights = _snapshot_weights(network)
    weighted = dispatch.multiply(weights, axis=0)

    totals = weighted.clip(lower=0.0).sum(axis=0)
    carriers = generators.loc[mask, "carrier"]
    by_group = totals.groupby(carriers).sum()

    return by_group.rename(lambda c: c.replace(POSITIVE_PREFIX, "")).sort_index()


def _aggregate_negative_slack(network: pypsa.Network) -> pd.Series:
    """Aggregate negative (excess) slack by food group in Mt."""

    generators = network.generators
    if generators.empty or "carrier" not in generators:
        return pd.Series(dtype=float)

    mask = generators["carrier"].astype(str).str.startswith(NEGATIVE_PREFIX)
    if not mask.any():
        return pd.Series(dtype=float)

    dispatch = network.generators_t.p.loc[:, mask]
    weights = _snapshot_weights(network)
    weighted = dispatch.multiply(weights, axis=0)

    # Negative p values (consumption) correspond to absorbing surplus food
    absorption = -weighted.clip(upper=0.0).sum(axis=0)
    carriers = generators.loc[mask, "carrier"]
    by_group = absorption.groupby(carriers).sum()

    return by_group.rename(lambda c: c.replace(NEGATIVE_PREFIX, "")).sort_index()


def _plot_slack(positive: pd.Series, negative: pd.Series, output_pdf: Path) -> None:
    """Render bar plot with positive slack above and negative slack below zero."""

    df = pd.DataFrame({"positive_mt": positive, "negative_mt": negative}).fillna(0.0)
    df = df[(df["positive_mt"] > 0) | (df["negative_mt"] > 0)]

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    if df.empty:
        ax.text(0.5, 0.5, "No food group slack recorded", ha="center", va="center")
        ax.axis("off")
        plt.savefig(output_pdf, bbox_inches="tight", dpi=300)
        plt.close()
        logger.info("No slack to plot; wrote placeholder to %s", output_pdf)
        return

    df["total_mt"] = df["positive_mt"] + df["negative_mt"]
    df = df.sort_values("total_mt", ascending=False)

    groups = df.index.tolist()
    group_colors = getattr(snakemake.params, "group_colors", {}) or {}
    colors = categorical_colors(groups, overrides=group_colors)

    positions = range(len(groups))
    ax.bar(
        positions,
        df["positive_mt"],
        color=[colors[g] for g in groups],
        edgecolor="black",
        linewidth=0.4,
        label="Positive slack (shortage)",
    )
    ax.bar(
        positions,
        -df["negative_mt"],
        color=[colors[g] for g in groups],
        edgecolor="black",
        linewidth=0.4,
        alpha=0.45,
        label="Negative slack (excess)",
    )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(groups, rotation=35, ha="right")
    ax.set_ylabel("Mt")
    ax.set_title("Food group slack (global)")
    ax.grid(axis="y", alpha=0.3)

    legend_handles = [
        Patch(
            facecolor="#666666", edgecolor="black", label="Positive slack (shortage)"
        ),
        Patch(
            facecolor="#666666",
            edgecolor="black",
            alpha=0.45,
            label="Negative slack (excess)",
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper right")

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_pdf, bbox_inches="tight", dpi=300)
    plt.close()
    logger.info("Wrote food group slack plot to %s", output_pdf)


def _write_csv(positive: pd.Series, negative: pd.Series, output_csv: Path) -> None:
    df = pd.DataFrame(
        {
            "positive_mt": positive,
            "negative_mt": negative,
        }
    ).fillna(0.0)
    df["net_mt"] = df["positive_mt"] - df["negative_mt"]
    df = df.sort_index()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, float_format="%.6g")
    logger.info("Wrote food group slack totals to %s", output_csv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    logger.info("Loading solved network from %s", snakemake.input.network)
    network = pypsa.Network(snakemake.input.network)

    positive = _aggregate_positive_slack(network)
    negative = _aggregate_negative_slack(network)

    _plot_slack(positive, negative, Path(snakemake.output.pdf))
    _write_csv(positive, negative, Path(snakemake.output.csv))
