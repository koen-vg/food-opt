# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Plot regional water use with pies for natural vs slack supply (Mm³)."""

import logging
from pathlib import Path

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa

matplotlib.use("pdf")

logger = logging.getLogger(__name__)


def _setup_regions(regions_path: str) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    gdf = gpd.read_file(regions_path)
    if gdf.crs is None:
        logger.warning("Regions GeoDataFrame missing CRS; assuming EPSG:4326")
        gdf = gdf.set_crs(4326, allow_override=True)
    else:
        gdf = gdf.to_crs(4326)

    if "region" not in gdf.columns:
        raise ValueError("Regions GeoDataFrame must contain a 'region' column")

    gdf = gdf.set_index("region", drop=False)
    gdf_eq = gdf.to_crs("+proj=eqearth")
    gdf_eq = gdf_eq.set_index("region", drop=False)
    return gdf, gdf_eq


def _aggregate_water_use(n: pypsa.Network, snapshot: str) -> pd.DataFrame:
    natural: dict[str, float] = {}
    slack: dict[str, float] = {}

    # Natural supply: discharge from water stores
    if not n.stores.empty and "p" in n.stores_t and not n.stores_t.p.empty:
        store_mask = n.stores["bus"].astype(str).str.startswith("water_")
        p_store = n.stores_t.p.loc[snapshot, store_mask]
        for name, val in p_store.items():
            if val <= 0:
                continue
            bus = str(n.stores.at[name, "bus"])
            region = bus.replace("water_", "", 1)
            natural[region] = natural.get(region, 0.0) + float(val)

    # Slack supply: dispatch of water_slack_* generators
    if not n.generators.empty and "p" in n.generators_t:
        mask = n.generators.index.to_series().str.startswith("water_slack_")
        if mask.any():
            dispatch = n.generators_t.p.loc[snapshot, mask]
            for name, val in dispatch.items():
                if val <= 0:
                    continue
                region = str(name).replace("water_slack_", "", 1)
                slack[region] = slack.get(region, 0.0) + float(val)

    regions = sorted(set(natural) | set(slack))
    df = pd.DataFrame(index=pd.Index(regions, name="region"))
    df["natural_mm3"] = pd.Series(natural)
    df["slack_mm3"] = pd.Series(slack)
    df = df.fillna(0.0).infer_objects(copy=False)
    df["total_mm3"] = df["natural_mm3"] + df["slack_mm3"]
    return df


def _draw_pie(
    ax: plt.Axes, x: float, y: float, sizes, colors, radius: float, transform
) -> None:
    total = float(sum(sizes))
    if total <= 0 or radius <= 0:
        return
    angles = np.cumsum([0.0] + [s / total * 360.0 for s in sizes])
    for i, size in enumerate(sizes):
        if size <= 0:
            continue
        wedge = mpatches.Wedge(
            center=(x, y),
            r=radius,
            theta1=angles[i],
            theta2=angles[i + 1],
            facecolor=colors[i],
            edgecolor="white",
            linewidth=0.4,
            alpha=0.85,
            transform=transform,
            zorder=10,
        )
        ax.add_patch(wedge)
    circ = mpatches.Circle(
        (x, y),
        radius=radius,
        facecolor="none",
        edgecolor="#444444",
        linewidth=0.3,
        alpha=0.7,
        transform=transform,
        zorder=11,
    )
    ax.add_patch(circ)


def _plot_map(by_region: pd.DataFrame, gdf: gpd.GeoDataFrame, output_pdf: Path) -> None:
    out = Path(output_pdf)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(
        figsize=(13, 6.5), dpi=150, subplot_kw={"projection": ccrs.EqualEarth()}
    )
    plate = ccrs.PlateCarree()
    ax.set_facecolor("#f7f9fb")
    ax.set_global()

    ax.add_geometries(
        gdf.geometry,
        crs=plate,
        facecolor="#e6eef2",
        edgecolor="#666666",
        linewidth=0.3,
        zorder=1,
    )

    if not by_region.empty:
        max_total = by_region["total_mm3"].max()
        # Use projected coordinates to keep pies circular on the map
        extent_proj = ax.get_extent(crs=ax.projection)
        width = extent_proj[1] - extent_proj[0]
        height = extent_proj[3] - extent_proj[2]
        max_radius = 0.03 * min(width, height)

        for region, row in by_region.iterrows():
            total = row["total_mm3"]
            if total <= 0 or region not in gdf.index:
                continue
            radius = max_radius * np.sqrt(total / max_total)
            geom = gdf.loc[region].geometry
            if geom.is_empty:
                continue
            centroid = geom.representative_point()
            x_proj, y_proj = ax.projection.transform_point(
                centroid.x, centroid.y, plate
            )
            _draw_pie(
                ax,
                x_proj,
                y_proj,
                [row["natural_mm3"], row["slack_mm3"]],
                ["#66c2a5", "#8da0cb"],
                radius,
                ax.transData,
            )

        # Size legend in axes coordinates (keeps circles circular and visible)
        legend_vals = [max_total, max_total / 4]
        legend_labels = [f"{v:,.0f} Mm³" for v in legend_vals]
        inset = ax.inset_axes([0.68, 0.04, 0.26, 0.22])
        inset.set_aspect("equal")
        inset.axis("off")
        x_center = 0.25
        max_r = 0.08
        y_positions = [0.65, 0.32]
        for y, val, label in zip(y_positions, legend_vals, legend_labels, strict=False):
            r = max_r * np.sqrt(val / max_total)
            circ = mpatches.Circle(
                (x_center, y),
                radius=r,
                facecolor="#ffffff",
                edgecolor="#333333",
                linewidth=0.6,
            )
            inset.add_patch(circ)
            inset.text(
                x_center + r + 0.06,
                y,
                label,
                ha="left",
                va="center",
                fontsize=8,
            )
        inset.set_xlim(0, 1)
        inset.set_ylim(0, 1)

    ax.set_title("Regional water use (natural vs slack)")
    # Legend for slice colors
    handles = [
        mpatches.Patch(facecolor="#66c2a5", edgecolor="black", label="Natural"),
        mpatches.Patch(facecolor="#8da0cb", edgecolor="black", label="Slack"),
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.02),
        frameon=True,
        framealpha=0.9,
        labelspacing=0.6,
        borderpad=0.5,
        handlelength=1.4,
        handletextpad=0.4,
        fontsize=9,
    )

    # Gridlines
    gl = ax.gridlines(draw_labels=False, linewidth=0.4, color="#aaaaaa", alpha=0.6)
    gl.xlocator = plt.MultipleLocator(60)
    gl.ylocator = plt.MultipleLocator(30)

    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info("Saved water use map to %s", out)


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    network = pypsa.Network(snakemake.input.network)  # type: ignore[name-defined]
    regions_path: str = snakemake.input.regions  # type: ignore[name-defined]
    output_pdf = Path(snakemake.output.pdf)  # type: ignore[name-defined]
    output_csv = Path(snakemake.output.csv)  # type: ignore[name-defined]

    snapshot = "now" if "now" in network.snapshots else network.snapshots[0]

    gdf, gdf_eq = _setup_regions(regions_path)
    water_use = _aggregate_water_use(network, snapshot)

    # Align to model regions
    water_use = water_use.reindex(gdf.index, fill_value=0.0)

    _plot_map(water_use, gdf, output_pdf)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    water_use.to_csv(output_csv, float_format="%.6g")
    logger.info("Saved regional water use table to %s", output_csv)


if __name__ == "__main__":
    main()
