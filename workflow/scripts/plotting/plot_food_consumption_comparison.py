# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Compare food consumption across solved objectives."""

import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pypsa

matplotlib.use("pdf")
import matplotlib.pyplot as plt

from workflow.scripts.plotting.plot_food_consumption import (
    DAYS_PER_YEAR,
    GRAMS_PER_MEGATONNE,
    KCAL_PER_PJ,
    _aggregate_group_calories,
    _aggregate_group_mass,
    _assign_colors,
    _select_snapshot,
)

logger = logging.getLogger(__name__)


def _load_population(population_path: Path) -> float:
    population_df = pd.read_csv(population_path)
    if "population" not in population_df.columns:
        raise ValueError("Population file must contain a 'population' column")
    total_population = float(population_df["population"].sum())
    if total_population <= 0.0:
        raise ValueError("Total population must be positive for per-capita conversion")
    return total_population


def _per_capita_consumption(
    network_path: Path, food_to_group: dict[str, str], population_total: float
) -> tuple[pd.Series, pd.Series]:
    network = pypsa.Network(network_path)
    snapshot = _select_snapshot(network)

    mass_mt = _aggregate_group_mass(network, snapshot, food_to_group)
    calories_pj = _aggregate_group_calories(network, snapshot, food_to_group)

    mass_per_capita = mass_mt * GRAMS_PER_MEGATONNE / (population_total * DAYS_PER_YEAR)
    calories_per_capita = calories_pj * KCAL_PER_PJ / (population_total * DAYS_PER_YEAR)

    return mass_per_capita, calories_per_capita


def _ordered_groups(mass_df: pd.DataFrame, cal_df: pd.DataFrame) -> list[str]:
    order: list[str] = []
    order.extend(mass_df.sum(axis=0).sort_values(ascending=False).index.tolist())
    for group in cal_df.sum(axis=0).sort_values(ascending=False).index:
        if group not in order:
            order.append(group)
    return order


def _plot_comparison(
    mass_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    labels: list[str],
    colors: dict[str, str],
    output_pdf: Path,
) -> None:
    if mass_df.empty and cal_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No food group consumption data", ha="center", va="center")
        ax.axis("off")
        fig.savefig(output_pdf, bbox_inches="tight", dpi=300)
        plt.close(fig)
        return

    ordered_groups = _ordered_groups(mass_df, cal_df)
    bar_height = 0.7
    fig_height = max(4.0, 1.0 + bar_height * len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(14, fig_height))

    for ax, data, xlabel, title in [
        (axes[0], mass_df, "g/person/day", "Food Consumption (Mass)"),
        (axes[1], cal_df, "kcal/person/day", "Food Consumption (Calories)"),
    ]:
        _ = np.arange(len(labels))
        for idx, label in enumerate(labels):
            series = data.loc[label] if label in data.index else pd.Series(dtype=float)
            left = 0.0
            for group in ordered_groups:
                value = float(series.get(group, 0.0))
                if value <= 0.0:
                    continue
                ax.barh(idx, value, left=left, height=bar_height, color=colors[group])
                left += value
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_ylim(-0.5, len(labels) - 0.5)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)

    axes[1].set_yticklabels([])
    axes[1].tick_params(axis="y", length=0)

    legend_handles = [
        matplotlib.patches.Patch(color=colors[group], label=group)
        for group in ordered_groups
    ]
    if legend_handles:
        fig.legend(
            legend_handles[::-1],
            [patch.get_label() for patch in legend_handles][::-1],
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
        )
        fig.tight_layout(rect=(0, 0, 0.90, 1))
    else:
        fig.tight_layout()

    fig.savefig(output_pdf, bbox_inches="tight", dpi=300)
    plt.close(fig)


def _write_csv(
    mass_df: pd.DataFrame, cal_df: pd.DataFrame, labels: list[str], output_csv: Path
) -> None:
    groups: list[str] = []
    groups.extend(mass_df.columns.tolist())
    for group in cal_df.columns:
        if group not in groups:
            groups.append(group)

    records: list[dict[str, float | str]] = []
    for label in labels:
        mass_row = (
            mass_df.loc[label] if label in mass_df.index else pd.Series(dtype=float)
        )
        cal_row = cal_df.loc[label] if label in cal_df.index else pd.Series(dtype=float)
        for group in groups:
            records.append(
                {
                    "scenario": label,
                    "group": group,
                    "mass_g_person_day": float(mass_row.get(group, 0.0)),
                    "calories_kcal_person_day": float(cal_row.get(group, 0.0)),
                }
            )

    pd.DataFrame.from_records(records).to_csv(output_csv, index=False)


def main() -> None:
    try:
        snakemake  # type: ignore[name-defined]
    except NameError as exc:  # pragma: no cover - Snakemake injects the variable
        raise RuntimeError("This script must be run from Snakemake") from exc

    network_paths = [Path(p) for p in snakemake.input.networks]  # type: ignore[attr-defined]
    comparison_labels = list(snakemake.params.wildcards)  # type: ignore[attr-defined]
    population_path = Path(snakemake.input.population)  # type: ignore[attr-defined]
    food_groups_path = Path(snakemake.input.food_groups)  # type: ignore[attr-defined]
    output_pdf = Path(snakemake.output.pdf)  # type: ignore[attr-defined]
    output_csv = Path(snakemake.output.csv)  # type: ignore[attr-defined]

    if len(network_paths) != len(comparison_labels):
        raise ValueError(
            "Number of network inputs must match number of comparison labels"
        )

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading food group mapping from %s", food_groups_path)
    food_groups_df = pd.read_csv(food_groups_path)
    food_to_group = food_groups_df.set_index("food")["group"].to_dict()

    population_total = _load_population(population_path)
    logger.info("Total population for per-capita conversion: %.3e", population_total)

    mass_data: dict[str, pd.Series] = {}
    cal_data: dict[str, pd.Series] = {}

    for label, network_path in zip(comparison_labels, network_paths):
        logger.info("Aggregating consumption for %s", network_path)
        mass_pc, cal_pc = _per_capita_consumption(
            network_path, food_to_group, population_total
        )
        mass_data[label] = mass_pc
        cal_data[label] = cal_pc

    mass_df = (
        pd.DataFrame(mass_data).T.fillna(0.0).sort_index(axis=1)
        if mass_data
        else pd.DataFrame()
    )
    cal_df = (
        pd.DataFrame(cal_data).T.fillna(0.0).sort_index(axis=1)
        if cal_data
        else pd.DataFrame()
    )

    group_colors = _assign_colors(
        _ordered_groups(mass_df, cal_df),
        overrides=getattr(snakemake.params, "group_colors", {}) or None,  # type: ignore[attr-defined]
    )

    _plot_comparison(mass_df, cal_df, comparison_labels, group_colors, output_pdf)
    _write_csv(mass_df, cal_df, comparison_labels, output_csv)

    logger.info(
        "Food consumption comparison saved to %s and %s", output_pdf, output_csv
    )


if __name__ == "__main__":
    main()
