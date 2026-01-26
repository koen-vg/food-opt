#! /usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

import logging
from pathlib import Path

from affine import Affine
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
import geopandas as gpd
import matplotlib
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

matplotlib.use("pdf")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from pyproj import Geod
import pypsa
from rasterio.transform import array_bounds
import xarray as xr

logger = logging.getLogger(__name__)

# Crop to group mapping for simplified visualization
CROP_TO_GROUP = {
    # Cereals (grains)
    "wheat": "Cereals",
    "dryland-rice": "Cereals",
    "wetland-rice": "Cereals",
    "maize": "Cereals",
    "barley": "Cereals",
    "oat": "Cereals",
    "rye": "Cereals",
    "sorghum": "Cereals",
    "buckwheat": "Cereals",
    "foxtail-millet": "Cereals",
    "pearl-millet": "Cereals",
    # Legumes
    "soybean": "Legumes",
    "dry-pea": "Legumes",
    "chickpea": "Legumes",
    "cowpea": "Legumes",
    "gram": "Legumes",
    "phaseolus-bean": "Legumes",
    "pigeonpea": "Legumes",
    # Roots & tubers
    "white-potato": "Roots & tubers",
    "sweet-potato": "Roots & tubers",
    "cassava": "Roots & tubers",
    "yam": "Roots & tubers",
    # Vegetables
    "tomato": "Vegetables",
    "carrot": "Vegetables",
    "onion": "Vegetables",
    "cabbage": "Vegetables",
    # Fruits
    "banana": "Fruits",
    "citrus": "Fruits",
    "coconut": "Fruits",
    # Oilseeds
    "sunflower": "Oilseeds",
    "rapeseed": "Oilseeds",
    "groundnut": "Oilseeds",
    "sesame": "Oilseeds",
    "oil-palm": "Oilseeds",
    "olive": "Oilseeds",
    # Sugar crops
    "sugarcane": "Sugar crops",
    "sugarbeet": "Sugar crops",
    # Feed/forage (non-food crops) and grassland
    "alfalfa": "Feed crops",
    "silage-maize": "Feed crops",
    "biomass-sorghum": "Feed crops",
    "grassland": "Feed crops",
}

# Colors for crop groups from Dark2 palette
_DARK2 = plt.get_cmap("Dark2").colors
CROP_GROUP_COLORS = {
    "Cereals": _DARK2[5],  # gold/amber - wheat color
    "Legumes": _DARK2[7],  # gray - peas, beans
    "Roots & tubers": _DARK2[6],  # brown - earth/soil
    "Vegetables": _DARK2[0],  # teal - fresh produce
    "Fruits": _DARK2[1],  # orange - citrus
    "Oilseeds": _DARK2[2],  # purple - distinct
    "Sugar crops": _DARK2[3],  # pink - sweet
    "Feed crops": _DARK2[4],  # green - grassland/animal feed
}


def _extract_land_use_by_region_class_crop(
    n: pypsa.Network,
    snapshot: str,
) -> pd.DataFrame:
    """Extract used land area by region, resource class, and crop.

    Returns DataFrame with columns: region, resource_class, crop, used_ha
    """
    links_static = n.links.static
    if links_static.empty:
        return pd.DataFrame(columns=["region", "resource_class", "crop", "used_ha"])

    # Find crop production links (produce_*) excluding grassland
    crop_links = links_static[
        links_static["carrier"].str.startswith("produce_", na=False)
        & (links_static["carrier"] != "produce_grassland")
    ]

    # Filter to links with valid resource_class and crop
    crop_links = crop_links[
        crop_links["resource_class"].notna() & crop_links["crop"].notna()
    ]

    rows = []

    if not crop_links.empty:
        p0_flows = n.links.dynamic.p0.loc[snapshot, crop_links.index] * 1e6
        for idx in crop_links.index:
            used_ha = max(float(p0_flows[idx]), 0.0)
            if used_ha > 0:
                rows.append(
                    {
                        "region": crop_links.at[idx, "region"],
                        "resource_class": int(crop_links.at[idx, "resource_class"]),
                        "crop": str(crop_links.at[idx, "crop"]),
                        "used_ha": used_ha,
                    }
                )

    # Handle grassland separately
    grass_links = links_static[links_static["carrier"] == "produce_grassland"]
    grass_links = grass_links[grass_links["resource_class"].notna()]

    if not grass_links.empty:
        p0_flows_grass = n.links.dynamic.p0.loc[snapshot, grass_links.index] * 1e6
        for idx in grass_links.index:
            used_ha = max(float(p0_flows_grass[idx]), 0.0)
            if used_ha > 0:
                rows.append(
                    {
                        "region": grass_links.at[idx, "region"],
                        "resource_class": int(grass_links.at[idx, "resource_class"]),
                        "crop": "grassland",
                        "used_ha": used_ha,
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["region", "resource_class", "crop", "used_ha"])

    return pd.DataFrame(rows)


def _load_resource_classes(path: str) -> dict:
    """Load resource class grid, region grid, extent, and cell areas.

    Returns dict with keys: class_grid, region_grid, extent, cell_areas_ha, shape
    """
    ds = xr.open_dataset(path)
    class_grid = ds["resource_class"].values.astype(np.int16)
    region_grid = ds["region_id"].values.astype(np.int32)
    transform_gdal = ds.attrs["transform"]
    transform = Affine.from_gdal(*transform_gdal)
    height, width = class_grid.shape
    lon_min, lat_min, lon_max, lat_max = array_bounds(height, width, transform)
    extent = (lon_min, lon_max, lat_min, lat_max)

    # Compute cell areas per row (varies by latitude)
    pixel_width_deg = abs(transform_gdal[1])
    pixel_height_deg = abs(transform_gdal[5])
    top = transform_gdal[3]
    bottom = top + height * transform_gdal[5]

    lats = np.linspace(
        top - pixel_height_deg / 2, bottom + pixel_height_deg / 2, height
    )
    geod = Geod(ellps="WGS84")
    cell_areas_ha_1d = np.zeros(height, dtype=np.float32)

    for i, lat in enumerate(lats):
        lat_top = lat + pixel_height_deg / 2
        lat_bottom = lat - pixel_height_deg / 2
        lon_left = lon_min
        lon_right = lon_min + pixel_width_deg
        lons = [lon_left, lon_right, lon_right, lon_left, lon_left]
        lats_poly = [lat_bottom, lat_bottom, lat_top, lat_top, lat_bottom]
        area_m2, _ = geod.polygon_area_perimeter(lons, lats_poly)
        cell_areas_ha_1d[i] = abs(area_m2) / 10000.0

    # Broadcast to 2D
    cell_areas_ha = np.repeat(cell_areas_ha_1d[:, np.newaxis], width, axis=1)

    ds.close()
    return {
        "class_grid": class_grid,
        "region_grid": region_grid,
        "extent": extent,
        "cell_areas_ha": cell_areas_ha,
        "shape": (height, width),
    }


def _load_potential_area(
    land_area_by_class_path: str,
    land_grazing_only_path: str,
) -> pd.Series:
    """Load potential cropland + grassland area by (region, resource_class).

    Combines:
    - Potential cropland: sum of rainfed + irrigated from land_area_by_class.csv
    - Potential grassland (marginal pasture): from land_grazing_only_by_class.csv

    Returns:
        Series indexed by (region, resource_class) with area in hectares.
    """
    # Load potential cropland (rainfed + irrigated)
    cropland_df = pd.read_csv(land_area_by_class_path)
    cropland_by_rc = cropland_df.groupby(["region", "resource_class"])["area_ha"].sum()

    # Load potential grassland (marginal pasture)
    grassland_df = pd.read_csv(land_grazing_only_path)
    grassland_by_rc = grassland_df.set_index(["region", "resource_class"])["area_ha"]

    # Combine: align indices and sum
    all_indices = cropland_by_rc.index.union(grassland_by_rc.index)
    potential_area = cropland_by_rc.reindex(
        all_indices, fill_value=0.0
    ) + grassland_by_rc.reindex(all_indices, fill_value=0.0)
    return potential_area


def _build_dominant_group_and_intensity_grids(
    land_use_df: pd.DataFrame,
    class_grid: np.ndarray,
    region_grid: np.ndarray,
    potential_area: pd.Series,
    region_name_to_id: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, dict[str, set[str]], pd.Series]:
    """Build pixel-level dominant crop group and intensity grids.

    Args:
        land_use_df: DataFrame with columns [region, resource_class, crop, used_ha]
        class_grid: 2D array of resource class IDs
        region_grid: 2D array of region IDs
        potential_area: Series indexed by (region, resource_class) with potential
            cropland + grassland area in hectares
        region_name_to_id: Mapping from region names to integer IDs

    Returns:
        dominant_group_grid: 2D array of group indices (-1 for no data)
        intensity_grid: 2D array of utilization fractions (0-1)
        crops_by_group: dict mapping group names to sets of crops present
        area_by_crop: Series with total area (ha) per crop
    """
    # Initialize output grids
    intensity_grid = np.full(class_grid.shape, np.nan, dtype=np.float32)
    dominant_group_grid = np.full(class_grid.shape, -1, dtype=np.int8)

    # Build group name to index mapping
    group_names = list(CROP_GROUP_COLORS.keys())
    group_to_idx = {name: idx for idx, name in enumerate(group_names)}

    # Track which crops appear in each group
    crops_by_group: dict[str, set[str]] = {g: set() for g in group_names}

    # Aggregate land use by (region, resource_class) and find dominant crop
    grouped = land_use_df.groupby(["region", "resource_class"])

    for (region, rc), group_df in grouped:
        total_used_ha = group_df["used_ha"].sum()
        if total_used_ha <= 0:
            continue

        # Find dominant crop by area
        crop_areas = group_df.groupby("crop")["used_ha"].sum()
        dominant_crop = crop_areas.idxmax()
        dominant_group = CROP_TO_GROUP.get(dominant_crop, "Other")

        # Track crops present
        for crop in crop_areas.index:
            crop_group = CROP_TO_GROUP.get(crop, "Other")
            if crop_group in crops_by_group:
                crops_by_group[crop_group].add(crop)

        # Compute intensity using potential area (cropland + grassland)
        potential_ha = potential_area.get((region, int(rc)), 0.0)
        intensity = min(total_used_ha / potential_ha, 1.0) if potential_ha > 0 else 0.0

        # Assign to pixels
        region_id = region_name_to_id.get(region)
        if region_id is not None and dominant_group in group_to_idx:
            mask = (region_grid == region_id) & (class_grid == int(rc))
            intensity_grid[mask] = intensity
            dominant_group_grid[mask] = group_to_idx[dominant_group]

    # Compute total area by crop across all regions/classes
    area_by_crop = land_use_df.groupby("crop")["used_ha"].sum()

    return dominant_group_grid, intensity_grid, crops_by_group, area_by_crop


def _setup_regions(regions_path: str) -> gpd.GeoDataFrame:
    """Load and prepare regions GeoDataFrame."""
    gdf = gpd.read_file(regions_path)
    if gdf.crs is None:
        logger.warning("Regions GeoDataFrame missing CRS; assuming EPSG:4326")
        gdf = gdf.set_crs(4326, allow_override=True)
    else:
        gdf = gdf.to_crs(4326)

    if "region" not in gdf.columns:
        raise ValueError("Regions GeoDataFrame must contain a 'region' column")

    gdf = gdf.set_index("region", drop=False)
    return gdf


def _plot_gridcell_intensity(
    dominant_group_grid: np.ndarray,
    intensity_grid: np.ndarray,
    extent: tuple,
    gdf: gpd.GeoDataFrame,
    crops_by_group: dict[str, set[str]],
    area_by_crop: pd.Series,
    output_path: str,
    title: str = "Dominant Crop and Land Use Intensity",
) -> None:
    """Plot dominant crop group with intensity at gridcell resolution.

    Args:
        dominant_group_grid: 2D array of group indices (-1 for no data)
        intensity_grid: 2D array of intensity values (0-1)
        extent: (lon_min, lon_max, lat_min, lat_max)
        gdf: GeoDataFrame with region boundaries
        crops_by_group: dict mapping group names to sets of crops present
        area_by_crop: Series with total area (ha) per crop
        output_path: Path to save PDF
        title: Figure title
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(
        figsize=(13, 6.5),
        dpi=150,
        subplot_kw={"projection": ccrs.EqualEarth()},
    )
    ax.set_facecolor("#ffffff")
    ax.set_global()
    plate = ccrs.PlateCarree()

    # Build RGBA image from dominant group and intensity
    group_names = list(CROP_GROUP_COLORS.keys())
    height, width = dominant_group_grid.shape
    rgba = np.ones((height, width, 4), dtype=np.float32)  # Start with white, alpha=1

    for idx, group_name in enumerate(group_names):
        color = CROP_GROUP_COLORS[group_name]
        # Convert to RGB if needed
        if isinstance(color, str):
            color = mcolors.to_rgb(color)
        mask = dominant_group_grid == idx
        if not np.any(mask):
            continue

        # Get intensity for these pixels
        intensities = intensity_grid[mask]

        # Set RGB from group color, alpha from intensity
        rgba[mask, 0] = color[0]
        rgba[mask, 1] = color[1]
        rgba[mask, 2] = color[2]
        rgba[mask, 3] = np.clip(intensities, 0.05, 1.0)  # Min alpha for visibility

    # Set pixels with no data to fully transparent
    no_data_mask = dominant_group_grid < 0
    rgba[no_data_mask, 3] = 0.0

    ax.imshow(
        rgba,
        origin="upper",
        extent=extent,
        transform=plate,
        interpolation="nearest",
        zorder=1,
    )

    # Add region boundaries in subtle grey
    ax.add_geometries(
        gdf.geometry,
        crs=plate,
        facecolor="none",
        edgecolor="#999999",
        linewidth=0.2,
        zorder=2,
    )

    # Style spines
    for name, spine in ax.spines.items():
        if name == "geo":
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_edgecolor("#cccccc")
        else:
            spine.set_visible(False)

    # Add gridlines
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
    gl.xlabel_style = {"size": 6, "color": "#555555"}
    gl.ylabel_style = {"size": 6, "color": "#555555"}
    gl.top_labels = False
    gl.right_labels = False

    # Build inset stacked bar chart showing land use breakdown by crop group
    # Compute area by group and crop
    group_data = []
    for group_name in group_names:
        crops_in_group = crops_by_group.get(group_name, set())
        if not crops_in_group:
            continue
        # Get areas for crops in this group, sorted by area (largest first)
        crop_areas = []
        for crop in crops_in_group:
            area_ha = area_by_crop.get(crop, 0.0)
            if area_ha > 0:
                crop_areas.append((crop, area_ha))
        if not crop_areas:
            continue
        crop_areas.sort(key=lambda x: -x[1])  # Sort by area descending
        total_area = sum(a for _, a in crop_areas)
        group_data.append((group_name, total_area, crop_areas))

    # Sort groups by total area (largest first = top row)
    group_data.sort(key=lambda x: -x[1])

    if group_data:
        # Create inset axes in bottom-left corner
        inset_width_frac = 0.29
        inset_ax = fig.add_axes([0.0, 0.0, inset_width_frac, 0.42])
        inset_ax.set_facecolor("#ffffff")
        inset_ax.patch.set_alpha(1.0)
        inset_ax.set_zorder(10)

        n_groups = len(group_data)
        bar_height = 0.5
        row_spacing = 1.0
        y_positions = np.arange(n_groups)[::-1] * row_spacing

        # Find max total area for x-axis scale
        max_area_mha = max(g[1] for g in group_data) / 1e6

        # Minimum segment width for labeling (as fraction of max area)
        min_label_frac = 0.025
        min_labels_per_group = 3

        # Setup for text width measurement
        fontsize = 5
        font_props = FontProperties(size=fontsize, family="monospace")
        x_margin_factor = 1.22
        x_range = max_area_mha * x_margin_factor

        # Calculate conversion factor from points to data coordinates
        fig_width_points = 13 * 72
        inset_width_points = fig_width_points * inset_width_frac
        points_to_data = x_range / inset_width_points

        def get_text_width_data(text: str) -> float:
            """Get text width in data coordinates using TextPath."""
            tp = TextPath((0, 0), text, prop=font_props)
            bbox = tp.get_extents()
            return bbox.width * points_to_data

        for i, (group_name, _total_area, crop_areas) in enumerate(group_data):
            y = y_positions[i]
            base_color = CROP_GROUP_COLORS[group_name]
            if isinstance(base_color, str):
                base_color = mcolors.to_rgb(base_color)

            left = 0.0
            for _crop, area_ha in crop_areas:
                area_mha = area_ha / 1e6

                inset_ax.barh(
                    y,
                    area_mha,
                    height=bar_height,
                    left=left,
                    color=base_color,
                    edgecolor="white",
                    linewidth=1.0,
                )

                left += area_mha

            # Add horizontal labels with smart overlap handling
            label_y = y + bar_height / 2 + 0.08
            label_x_nudge = points_to_data * 2.0
            last_label_right_x = 0.0

            n_crops_in_group = len(crop_areas)
            guaranteed_labels = min(n_crops_in_group, min_labels_per_group)

            left = 0.0
            for j, (crop, area_ha) in enumerate(crop_areas):
                area_mha = area_ha / 1e6
                seg_frac = area_mha / max_area_mha

                if seg_frac >= min_label_frac or j < guaranteed_labels:
                    desired_x = left + label_x_nudge

                    if desired_x < last_label_right_x:
                        label_x = last_label_right_x
                        label_text = ", " + crop
                    else:
                        label_x = desired_x
                        label_text = crop

                    inset_ax.text(
                        label_x,
                        label_y,
                        label_text,
                        ha="left",
                        va="bottom",
                        fontsize=fontsize,
                        fontfamily="monospace",
                        color="#333333",
                    )

                    label_width = get_text_width_data(label_text)
                    padding = points_to_data * 1.0
                    last_label_right_x = label_x + label_width + padding

                left += area_mha

        # Style the inset axes
        inset_ax.set_yticks(y_positions)
        inset_ax.set_yticklabels([g[0] for g in group_data], fontsize=6)
        inset_ax.set_xlabel("Land use (Mha)", fontsize=6)
        inset_ax.tick_params(axis="x", labelsize=5)
        inset_ax.tick_params(axis="y", length=0)

        inset_ax.set_xlim(0, x_range)
        y_max = y_positions[0] + bar_height / 2 + 0.9
        y_min = y_positions[-1] - bar_height / 2 - 0.3
        inset_ax.set_ylim(y_min, y_max)

        inset_ax.xaxis.grid(True, linestyle="-", alpha=0.3, linewidth=0.5)
        inset_ax.set_axisbelow(True)

        for spine in inset_ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_color("#cccccc")

    # Add intensity colorbar
    cmap_colors = np.zeros((256, 4))
    cmap_colors[:, 0] = 0.4  # R (gray)
    cmap_colors[:, 1] = 0.4  # G (gray)
    cmap_colors[:, 2] = 0.4  # B (gray)
    cmap_colors[:, 3] = np.linspace(0, 1, 256)  # Alpha gradient
    intensity_cmap = mcolors.ListedColormap(cmap_colors)

    sm = plt.cm.ScalarMappable(
        cmap=intensity_cmap, norm=mcolors.Normalize(vmin=0, vmax=100)
    )
    sm.set_array([])

    # Add colorbar with white background
    cbar_bg_ax = fig.add_axes([0.44, 0.07, 0.26, 0.08])
    cbar_bg_ax.set_facecolor("#ffffff")
    cbar_bg_ax.patch.set_alpha(1.0)
    cbar_bg_ax.set_zorder(9)
    cbar_bg_ax.set_xticks([])
    cbar_bg_ax.set_yticks([])
    for spine in cbar_bg_ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color("#cccccc")

    cbar_ax = fig.add_axes([0.48, 0.12, 0.18, 0.018])
    cbar_ax.set_zorder(10)
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks([0, 50, 100])
    cbar.set_ticklabels(["0%", "50%", "100%"])
    cbar.ax.tick_params(labelsize=6, length=2, color="#cccccc")
    cbar.set_label(
        "Utilization of potential cropland and grassland (excl. protected/unsuitable)",
        fontsize=6,
    )
    cbar.outline.set_linewidth(0.5)
    cbar.outline.set_edgecolor("#cccccc")

    ax.set_title(title, fontsize=8)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)

    logger.info("Saved gridcell intensity map to %s", out)


def main() -> None:
    n = pypsa.Network(snakemake.input.network)  # type: ignore[name-defined]
    regions_path: str = snakemake.input.regions  # type: ignore[name-defined]
    resource_classes_path: str = snakemake.input.resource_classes  # type: ignore[name-defined]
    land_area_by_class_path: str = snakemake.input.land_area_by_class  # type: ignore[name-defined]
    land_grazing_only_path: str = snakemake.input.land_grazing_only  # type: ignore[name-defined]
    output_pdf: str = snakemake.output.pdf  # type: ignore[name-defined]

    snapshot = "now" if "now" in n.snapshots else n.snapshots[0]

    gdf = _setup_regions(regions_path)
    region_name_to_id = {region: idx for idx, region in enumerate(gdf["region"])}

    rc_data = _load_resource_classes(resource_classes_path)
    potential_area = _load_potential_area(
        land_area_by_class_path, land_grazing_only_path
    )
    land_use_by_rc_crop = _extract_land_use_by_region_class_crop(n, snapshot)

    if not land_use_by_rc_crop.empty:
        dominant_group_grid, intensity_grid, crops_by_group, area_by_crop = (
            _build_dominant_group_and_intensity_grids(
                land_use_by_rc_crop,
                rc_data["class_grid"],
                rc_data["region_grid"],
                potential_area,
                region_name_to_id,
            )
        )
        _plot_gridcell_intensity(
            dominant_group_grid,
            intensity_grid,
            rc_data["extent"],
            gdf,
            crops_by_group,
            area_by_crop,
            output_pdf,
            title="Dominant Crop and Land Use Intensity",
        )
    else:
        logger.warning("No land use data by resource class; skipping gridcell plot")


if __name__ == "__main__":
    main()
