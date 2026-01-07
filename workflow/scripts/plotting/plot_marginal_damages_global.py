# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Plot consumption-weighted global average GHG and YLL impacts by food group."""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
import pandas as pd

from workflow.scripts.plotting.color_utils import categorical_colors

logger = logging.getLogger(__name__)


def compute_global_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Compute consumption-weighted global averages by food group.

    Input DataFrame has columns: country, food_group, consumption_mt,
    ghg_mtco2e_per_mt, yll_myll_per_mt

    Returns DataFrame with columns: food_group, consumption_mt,
    ghg_kgco2e_per_kg, yll_per_kg
    """
    # Filter to rows with positive consumption
    df = df[df["consumption_mt"] > 0].copy()

    if df.empty:
        return pd.DataFrame(
            columns=["food_group", "consumption_mt", "ghg_kgco2e_per_kg", "yll_per_kg"]
        )

    # Consumption-weighted average by food group
    def weighted_avg(group: pd.DataFrame) -> pd.Series:
        total_consumption = group["consumption_mt"].sum()
        ghg_weighted = (
            group["ghg_mtco2e_per_mt"] * group["consumption_mt"]
        ).sum() / total_consumption
        yll_weighted = (
            group["yll_myll_per_mt"] * group["consumption_mt"]
        ).sum() / total_consumption
        return pd.Series(
            {
                "consumption_mt": total_consumption,
                # MtCO2e/Mt = kgCO2e/kg (same ratio)
                "ghg_kgco2e_per_kg": ghg_weighted,
                # mYLL/Mt to YLL/kg: 1e6 YLL / 1e9 kg = 1e-3 YLL/kg
                "yll_per_kg": yll_weighted * 1e-3,
            }
        )

    result = df.groupby("food_group").apply(weighted_avg, include_groups=False)
    return result.reset_index()


def plot_ghg_bar(
    df: pd.DataFrame,
    output_path: Path,
    group_colors: dict[str, str] | None = None,
) -> None:
    """Create horizontal bar chart of GHG intensity by food group."""
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.axis("off")
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        return

    # Sort by GHG intensity descending
    df = df.sort_values("ghg_kgco2e_per_kg", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    groups = df["food_group"].tolist()
    colors = categorical_colors(groups, overrides=group_colors)

    ax.barh(
        range(len(groups)),
        df["ghg_kgco2e_per_kg"],
        color=[colors[g] for g in groups],
        edgecolor="black",
        linewidth=0.4,
    )

    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups)
    ax.set_xlabel("GHG intensity (kg CO$_2$eq / kg food)")
    ax.set_title("Global average GHG intensity by food group\n(consumption-weighted)")
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.axvline(0, color="black", linewidth=0.8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info("Wrote GHG bar chart to %s", output_path)


def plot_yll_bar(
    df: pd.DataFrame,
    output_path: Path,
    group_colors: dict[str, str] | None = None,
) -> None:
    """Create horizontal bar chart of YLL impact by food group."""
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.axis("off")
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        return

    # Sort by absolute YLL impact (largest absolute at top)
    df = df.copy()
    df["abs_yll"] = df["yll_per_kg"].abs()
    df = df.sort_values("abs_yll", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    groups = df["food_group"].tolist()
    colors = categorical_colors(groups, overrides=group_colors)

    bar_colors = [colors[g] for g in groups]

    ax.barh(
        range(len(groups)),
        df["yll_per_kg"],
        color=bar_colors,
        edgecolor="black",
        linewidth=0.4,
    )

    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups)
    ax.set_xlabel("Health impact (YLL / kg food)")
    ax.set_title(
        "Global average health impact by food group\n"
        "(consumption-weighted; negative = protective)"
    )
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.axvline(0, color="black", linewidth=0.8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info("Wrote YLL bar chart to %s", output_path)


def save_csv(df: pd.DataFrame, output_path: Path) -> None:
    """Save global averages to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, float_format="%.6g")
    logger.info("Wrote global averages to %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Load marginal damages
    marginal_df = pd.read_csv(snakemake.input.marginal_damages)
    logger.info("Loaded %d rows from marginal damages", len(marginal_df))

    # Compute global averages
    global_avg = compute_global_averages(marginal_df)
    logger.info("Computed averages for %d food groups", len(global_avg))

    # Get color overrides
    group_colors = getattr(snakemake.params, "group_colors", {}) or {}

    # Create plots
    plot_ghg_bar(global_avg, Path(snakemake.output.ghg_pdf), group_colors)
    plot_yll_bar(global_avg, Path(snakemake.output.yll_pdf), group_colors)
    save_csv(global_avg, Path(snakemake.output.csv))
