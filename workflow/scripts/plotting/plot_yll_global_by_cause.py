#! /usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Plot global diet-attributable YLL by cause (million YLL)."""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

YLL_TO_MILLION = 1e-6


def _prepare_summary(cluster_cause: pd.DataFrame) -> pd.DataFrame:
    required = {"cause", "yll_base"}
    missing = required - set(cluster_cause.columns)
    if missing:
        raise ValueError(f"Missing required columns in cluster_cause: {missing}")

    summary = (
        cluster_cause.groupby("cause", as_index=False)["yll_base"]
        .sum()
        .sort_values("yll_base", ascending=False)
    )
    summary["yll_million"] = summary["yll_base"] * YLL_TO_MILLION
    return summary


def _plot(summary: pd.DataFrame, output_pdf: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    bars = ax.bar(summary["cause"], summary["yll_million"], color="#4e79a7")

    ax.set_ylabel("YLL lost (million)")
    ax.set_title("Global diet-attributable YLL by cause")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, value in zip(bars, summary["yll_million"]):
        ax.annotate(
            f"{value:,.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    fig.savefig(output_pdf, bbox_inches="tight", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    cluster_cause_path = Path(snakemake.input.cluster_cause)  # type: ignore[name-defined]
    output_pdf = Path(snakemake.output.pdf)  # type: ignore[name-defined]
    output_csv = Path(snakemake.output.csv)  # type: ignore[name-defined]

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    cluster_cause = pd.read_csv(cluster_cause_path)
    summary = _prepare_summary(cluster_cause)

    summary.to_csv(output_csv, index=False)
    _plot(summary, output_pdf)

    logger.info("Saved global YLL bar chart to %s", output_pdf)
