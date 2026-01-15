#! /usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Plot cropland fraction maps with resource class detail at pixel resolution.

Inputs (via snakemake):
- input.network: solved PyPSA network (NetCDF)
- input.regions: regions GeoJSON with a 'region' column
- input.land_area_by_class: CSV with columns [region, water_supply, resource_class, area_ha]
  (NOTE: no longer used for total area calculation)
- input.resource_classes: NetCDF with variables 'resource_class' and 'region_id'

Outputs:
- PDF map under results/{name}/plots/
- CSV summary alongside the PDF (same stem, `_by_region_class.csv`)

Optional parameters (snakemake.params):
- water_supply: "i"/"irrigated", "r"/"rainfed", or omitted for all.
- title: figure title; defaults derived from `water_supply`.
- colorbar_label: colorbar caption; defaults derived from `water_supply`.
- cmap: Matplotlib colormap name (default: "YlOrRd").
- csv_prefix: prefix for CSV column names (default determined from `water_supply`).

Notes:
- Cropland use is computed from link flows for crop production links (carrier
  starting with 'crop_') and grazing links (carrier='feed_ruminant_grassland').
  Region, resource class, and water supply are read from domain columns on links.
  Water supply filtering (irrigated vs rainfed) happens via the optional
  parameter above. This reflects actual cropland allocated by the solver.
- Total land area per (region, resource class) pair is computed directly from
  the resource_classes grid by summing cell areas, NOT from land_area_by_class.csv
  (which contains suitability-weighted areas). This shows actual cropland as a
  fraction of total land area.
- Each pixel inherits the cropland fraction of its (region, resource class)
  combination, so within-region spatial patterns remain visible.
"""

import logging
from pathlib import Path

import cartopy.crs as ccrs
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
import geopandas as gpd
import matplotlib

matplotlib.use("pdf")
from affine import Affine
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from pyproj import Geod
import pypsa
from rasterio.transform import array_bounds
import xarray as xr

logger = logging.getLogger(__name__)

# Enable new PyPSA components API
pypsa.options.api.new_components_api = True


def _compute_total_land_area_by_region_class(
    classes_ds: xr.Dataset,
    regions_gdf: gpd.GeoDataFrame,
    region_name_to_id: dict[str, int],
) -> pd.Series:
    """Compute total land area per (region, resource_class) from grid.

    Args:
        classes_ds: xarray Dataset with resource_class and region_id variables
        regions_gdf: GeoDataFrame with regions (already indexed by region name)
        region_name_to_id: mapping from region name to integer ID

    Returns:
        Series indexed by (region, resource_class) with total area in hectares
    """
    class_grid = classes_ds["resource_class"].values.astype(np.int16)
    region_grid = classes_ds["region_id"].values.astype(np.int32)

    # Get transform from dataset attributes
    transform_gdal = classes_ds.attrs.get("transform")
    if transform_gdal is None:
        raise ValueError("Dataset missing 'transform' attribute")

    # Extract grid parameters
    height, width = class_grid.shape
    pixel_width_deg = abs(transform_gdal[1])
    pixel_height_deg = abs(transform_gdal[5])
    left = transform_gdal[0]
    top = transform_gdal[3]
    bottom = top + height * transform_gdal[5]  # transform[5] is negative

    # Calculate cell areas per row using Geod (same approach as raster_utils)
    lats = np.linspace(
        top - pixel_height_deg / 2, bottom + pixel_height_deg / 2, height
    )
    geod = Geod(ellps="WGS84")
    cell_areas_ha_1d = np.zeros(height, dtype=np.float32)

    for i, lat in enumerate(lats):
        lat_top = lat + pixel_height_deg / 2
        lat_bottom = lat - pixel_height_deg / 2
        lon_left = left
        lon_right = left + pixel_width_deg
        lons = [lon_left, lon_right, lon_right, lon_left, lon_left]
        lats_poly = [lat_bottom, lat_bottom, lat_top, lat_top, lat_bottom]
        area_m2, _ = geod.polygon_area_perimeter(lons, lats_poly)
        cell_areas_ha_1d[i] = abs(area_m2) / 10000.0

    # Broadcast to 2D for easier indexing
    cell_areas_ha = np.repeat(cell_areas_ha_1d[:, np.newaxis], width, axis=1)

    # Aggregate by (region, resource_class)
    rows: list[tuple[str, int, float]] = []
    valid_mask = (region_grid >= 0) & (class_grid >= 0)

    if not np.any(valid_mask):
        return pd.Series(dtype=float, name="total_ha")

    # Iterate over each region and resource class
    for region_name, region_id in region_name_to_id.items():
        region_mask = region_grid == region_id
        if not np.any(region_mask):
            continue

        # Get unique resource classes in this region
        classes_in_region = np.unique(class_grid[region_mask & valid_mask])

        for resource_class in classes_in_region:
            if resource_class < 0:
                continue

            # Mask for this (region, class) combination
            mask = region_mask & (class_grid == resource_class)

            # Sum cell areas
            total_area_ha = float(np.sum(cell_areas_ha[mask]))

            if total_area_ha > 0:
                rows.append((region_name, int(resource_class), total_area_ha))

    if not rows:
        return pd.Series(dtype=float, name="total_ha")

    df = pd.DataFrame(rows, columns=["region", "resource_class", "total_ha"])
    total_area = (
        df.groupby(["region", "resource_class"], sort=False)["total_ha"]
        .sum()
        .astype(float)
    )
    total_area.index = total_area.index.set_levels(
        total_area.index.levels[1].astype(int), level="resource_class"
    )
    return total_area


def _used_cropland_area_by_region_class(
    n: pypsa.Network, water_supply: str | None
) -> pd.Series:
    """Return used cropland and grassland area by region and resource class.

    Extracts land consumption from crop production and grazing links.
    Uses actual flow (p0) not capacity (p_nom_opt).

    Includes:
    - Crop production links (carrier starts with 'crop_')
    - Grazing links (carrier = 'feed_ruminant_grassland')

    Excludes spared land links (carrier='spared_land').

    Returns area in hectares.
    """

    links_static = n.links.static
    if links_static.empty:
        return pd.Series(dtype=float)

    # Find all links that consume from land buses, based on carrier
    # Include crop production and grazing, exclude spared land
    land_links = links_static[
        (
            links_static["carrier"].str.startswith("crop_", na=False)
            | (links_static["carrier"] == "feed_ruminant_grassland")
        )
        & (links_static["carrier"] != "spared_land")
    ]

    if land_links.empty:
        return pd.Series(dtype=float)

    # Filter by water supply if specified, using the domain column
    if water_supply is not None:
        land_links = land_links[land_links["water_supply"] == water_supply]
        if land_links.empty:
            return pd.Series(dtype=float)

    # Get snapshot - use "now" if available, otherwise first snapshot
    snapshot = "now" if "now" in n.snapshots else n.snapshots[0]

    # Filter to links that have valid resource_class (exclude links without it)
    land_links = land_links[land_links["resource_class"].notna()]
    if land_links.empty:
        return pd.Series(dtype=float)

    # Get actual flow on bus0 (land consumption) for all land links at this snapshot
    # p0 is in Mha, convert to ha
    p0_flows = n.links.dynamic.p0.loc[snapshot, land_links.index] * 1e6

    # Build DataFrame using domain columns directly
    df = pd.DataFrame(
        {
            "region": land_links["region"],
            "resource_class": land_links["resource_class"].astype(int),
            "used_ha": p0_flows.clip(lower=0.0).values,
        }
    )

    if df.empty:
        return pd.Series(dtype=float)

    used = (
        df.groupby(["region", "resource_class"], sort=False)["used_ha"]
        .sum()
        .astype(float)
    )
    used.index = pd.MultiIndex.from_tuples(
        [(region, int(resource_class)) for region, resource_class in used.index],
        names=["region", "resource_class"],
    )
    return used


def _normalize_water_supply(value: object) -> str | None:
    if value is None:
        return None

    text = str(value).strip().lower()
    if not text or text in {"all", "total", "both", "any"}:
        return None

    mapping = {
        "i": "i",
        "irr": "i",
        "irrigated": "i",
        "r": "r",
        "rainfed": "r",
    }

    try:
        return mapping[text]
    except KeyError as exc:
        raise ValueError(
            "Unsupported water_supply parameter (expected irrigated/rainfed)"
        ) from exc


def _level_values(index: pd.Index, name: str, fallback_level: int) -> pd.Index:
    """Return index level by name, with fallbacks for unnamed/empty indexes."""
    if not isinstance(index, pd.MultiIndex):
        return pd.Index([], dtype=int)
    try:
        return index.get_level_values(name)
    except KeyError:
        try:
            return index.get_level_values(fallback_level)
        except (KeyError, IndexError):
            return pd.Index([], dtype=int)


def main() -> None:
    n = pypsa.Network(snakemake.input.network)  # type: ignore[name-defined]
    regions_path: str = snakemake.input.regions  # type: ignore[name-defined]
    classes_path: str = snakemake.input.resource_classes  # type: ignore[name-defined]
    out_pdf = Path(snakemake.output.pdf)  # type: ignore[name-defined]

    params = getattr(snakemake, "params", None)  # type: ignore[name-defined]
    water_param = getattr(params, "water_supply", None) if params is not None else None
    water_supply = _normalize_water_supply(water_param)

    base_prefix = {
        None: "cropland",
        "i": "irrigated_cropland",
        "r": "rainfed_cropland",
    }[water_supply]

    csv_prefix = getattr(params, "csv_prefix", None) if params is not None else None
    if not csv_prefix:
        csv_prefix = base_prefix

    default_titles = {
        None: "Cropland Fraction by Region and Resource Class",
        "i": "Irrigated Cropland Fraction by Region and Resource Class",
        "r": "Rainfed Cropland Fraction by Region and Resource Class",
    }
    default_colorbars = {
        None: "Cropland / total land area",
        "i": "Irrigated cropland / total land area",
        "r": "Rainfed cropland / total land area",
    }

    title = (
        getattr(params, "title", None) if params is not None else None
    ) or default_titles[water_supply]
    colorbar_label = (
        getattr(params, "colorbar_label", None) if params is not None else None
    ) or default_colorbars[water_supply]
    cmap_name = (
        getattr(params, "cmap", None) if params is not None else None
    ) or "YlOrRd"

    gdf = gpd.read_file(regions_path)
    if "region" not in gdf.columns:
        raise ValueError("regions input must contain a 'region' column")
    gdf = gdf.reset_index(drop=True).to_crs(4326)
    region_name_to_id = {region: idx for idx, region in enumerate(gdf["region"])}
    gdf = gdf.set_index("region", drop=False)

    # Load resource class grid first (needed for total area computation)
    classes_ds = xr.open_dataset(classes_path, mode="r")
    if "resource_class" not in classes_ds or "region_id" not in classes_ds:
        raise ValueError("resource_classes input must contain required variables")

    used_ha = _used_cropland_area_by_region_class(n, water_supply)

    # Compute total land area (not suitability-weighted) per region/resource_class
    total_ha = _compute_total_land_area_by_region_class(
        classes_ds, gdf, region_name_to_id
    )

    classes = sorted(
        set(_level_values(total_ha.index, "resource_class", 1))
        | set(_level_values(used_ha.index, "resource_class", 1))
    )
    if not classes:
        raise ValueError("No resource classes found")

    full_index = pd.MultiIndex.from_product(
        [gdf.index, classes], names=["region", "resource_class"]
    )
    used_ha = used_ha.reindex(full_index).fillna(0.0)
    total_ha = total_ha.reindex(full_index).fillna(0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        fractions = (
            (used_ha / total_ha).replace([np.inf, -np.inf], np.nan).clip(0.0, 1.0)
        )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    class_grid = classes_ds["resource_class"].values.astype(np.int16)
    region_grid = classes_ds["region_id"].values.astype(np.int32)
    transform = Affine.from_gdal(*classes_ds.attrs["transform"])
    height, width = class_grid.shape
    lon_min, lat_min, lon_max, lat_max = array_bounds(height, width, transform)
    extent = (lon_min, lon_max, lat_min, lat_max)  # Fixed orientation!
    classes_ds.close()

    fraction_grid = np.full(class_grid.shape, np.nan, dtype=float)
    for (region, cls), frac in fractions.items():
        ridx = region_name_to_id.get(region)
        if ridx is not None:
            mask = (region_grid == ridx) & (class_grid == cls)
            fraction_grid[mask] = frac

    vmax = (
        max(0.5, min(1.0, np.nanmax(fraction_grid)))
        if not np.all(np.isnan(fraction_grid))
        else 0.5
    )

    cmap = plt.colormaps[cmap_name]
    fig, ax = plt.subplots(
        figsize=(13, 6.5), dpi=150, subplot_kw={"projection": ccrs.EqualEarth()}
    )
    ax.set_facecolor("#f7f9fb")
    ax.set_global()

    plate = ccrs.PlateCarree()
    im = ax.imshow(
        fraction_grid,
        origin="upper",
        extent=extent,
        transform=plate,
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        alpha=0.8,
        zorder=1,
    )

    ax.add_geometries(
        gdf.geometry,
        crs=plate,
        facecolor="none",
        edgecolor="#666666",
        linewidth=0.3,
        zorder=2,
    )

    gl = ax.gridlines(
        draw_labels=True,
        crs=plate,
        linewidth=0.35,
        color="#888888",
        alpha=0.45,
        linestyle="--",
    )
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 30))
    gl.ylocator = mticker.FixedLocator(np.arange(-60, 61, 15))
    gl.xformatter = LongitudeFormatter(number_format=".0f")
    gl.yformatter = LatitudeFormatter(number_format=".0f")
    gl.xlabel_style = {"size": 8, "color": "#555555"}
    gl.ylabel_style = {"size": 8, "color": "#555555"}
    gl.top_labels = gl.right_labels = False

    cbar = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.02)
    cbar.set_label(colorbar_label)

    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    plt.close(fig)

    data = pd.DataFrame(
        {
            f"{csv_prefix}_used_ha": used_ha,
            "total_land_area_ha": total_ha,
            f"{csv_prefix}_fraction": fractions,
        }
    )
    csv_out = out_pdf.with_suffix("").parent / f"{out_pdf.stem}_by_region_class.csv"
    data.reset_index().to_csv(csv_out, index=False)
    logger.info("Saved cropland fraction map to %s and CSV to %s", out_pdf, csv_out)


if __name__ == "__main__":
    main()
