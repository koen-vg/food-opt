"""
SPDX-FileCopyrightText: 2025 Koen van Greevenbroek

SPDX-License-Identifier: GPL-3.0-or-later

Aggregate CROPGRIDS harvested area to regions/resource classes and split
between irrigated and rainfed using GAEZ irrigated shares.

Outputs match the format expected by `_load_crop_yield_table`, with
variables `harvested_area`, `crop_area`, and `cropping_intensity` per
region/resource_class/water_supply.
"""

import gc
from pathlib import Path
import sys

from exactextract import exact_extract
from exactextract.raster import NumPyRasterSource
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from raster_utils import raster_bounds, read_raster_float

RES06_HAR_SCALE_TO_HA = 1_000.0  # RES06-HAR stored in kha


def _load_cropgrids_arrays(
    nc_file: Path,
) -> tuple[np.ndarray, np.ndarray, rasterio.Affine]:
    """Return (harvested_area, crop_area, transform) arrays in ha."""

    if not nc_file.exists():
        raise FileNotFoundError(f"No CROPGRIDS NetCDF found at {nc_file}")

    with rasterio.open(f"NETCDF:{nc_file}:harvarea") as src_h:
        harv = src_h.read(1, masked=True).filled(0.0).astype(np.float32)
    with rasterio.open(f"NETCDF:{nc_file}:croparea") as src_c:
        crop = src_c.read(1, masked=True).filled(0.0).astype(np.float32)
    transform = from_origin(-180.0, 90.0, 0.05, 0.05)
    return harv, crop, transform


def _aggregate_by_region(
    value_arr: np.ndarray, regions_gdf: gpd.GeoDataFrame, transform
) -> pd.DataFrame:
    xmin, ymin, xmax, ymax = raster_bounds(
        transform, value_arr.shape[1], value_arr.shape[0]
    )
    src = NumPyRasterSource(
        value_arr,
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
        nodata=np.nan,
        srs_wkt=CRS.from_epsg(4326).to_wkt(),
    )
    stats = exact_extract(
        src,
        regions_gdf[["region", "geometry"]],
        ["sum"],
        include_cols=["region"],
        output="pandas",
    )
    if stats.empty:
        return pd.DataFrame(columns=["region", "value"])
    return stats.rename(columns={"sum": "value"})


if __name__ == "__main__":
    cropgrids_nc_path = Path(snakemake.input.cropgrids_nc)  # type: ignore[name-defined]
    gaez_r_path = Path(snakemake.input.gaez_harvest_r)  # type: ignore[name-defined]
    gaez_i_path = Path(snakemake.input.gaez_harvest_i)  # type: ignore[name-defined]
    class_frac_path = Path(snakemake.input.class_fractions)  # type: ignore[name-defined]
    regions_path = Path(snakemake.input.regions)  # type: ignore[name-defined]
    output_path = Path(snakemake.output[0])  # type: ignore[name-defined]
    ws = str(snakemake.wildcards.water_supply)  # type: ignore[name-defined]

    # If the cropgrids file is empty (crop not in CROPGRIDS), output empty CSV
    if cropgrids_nc_path.stat().st_size == 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            columns=["region", "resource_class", "variable", "unit", "value"]
        ).to_csv(output_path, index=False)
        raise SystemExit(0)

    # Load CROPGRIDS harvarea and croparea
    harv_cg, crop_cg, cg_transform = _load_cropgrids_arrays(cropgrids_nc_path)
    cg_shape = harv_cg.shape
    cg_crs = CRS.from_epsg(4326)

    # Load class fractions on the CROPGRIDS grid
    # Use open_dataset instead of load_dataset to avoid loading all data into memory
    class_frac_ds = xr.open_dataset(class_frac_path, chunks="auto")

    # Load GAEZ irrigated/rainfed harvested area and scale to ha, reproject to CROPGRIDS grid
    gaez_r, gaez_src_r = read_raster_float(gaez_r_path)
    gaez_i, gaez_src_i = read_raster_float(gaez_i_path)
    # Ensure float32
    gaez_r = (gaez_r * RES06_HAR_SCALE_TO_HA).astype(np.float32)
    gaez_i = (gaez_i * RES06_HAR_SCALE_TO_HA).astype(np.float32)
    gaez_transform = gaez_src_r.transform
    gaez_crs = gaez_src_r.crs
    gaez_src_r.close()
    gaez_src_i.close()

    gaez_r_cg = np.zeros(cg_shape, dtype=np.float32)
    gaez_i_cg = np.zeros(cg_shape, dtype=np.float32)
    reproject(
        gaez_r,
        gaez_r_cg,
        src_transform=gaez_transform,
        src_crs=gaez_crs,
        dst_transform=cg_transform,
        dst_crs=cg_crs,
        resampling=Resampling.average,
    )
    reproject(
        gaez_i,
        gaez_i_cg,
        src_transform=gaez_transform,
        src_crs=gaez_crs,
        dst_transform=cg_transform,
        dst_crs=cg_crs,
        resampling=Resampling.average,
    )
    # Free original GAEZ arrays
    del gaez_r, gaez_i
    gc.collect()

    gaez_sum = gaez_r_cg + gaez_i_cg
    irrig_share = np.divide(
        gaez_i_cg, gaez_sum, out=np.zeros_like(gaez_i_cg), where=gaez_sum > 0
    )
    irrig_share = np.clip(irrig_share, 0.0, 1.0)

    # Free reprojected GAEZ arrays
    del gaez_r_cg, gaez_i_cg, gaez_sum
    gc.collect()

    regions = gpd.read_file(regions_path)[["region", "geometry"]]

    harv_records: list[pd.DataFrame] = []
    crop_records: list[pd.DataFrame] = []

    # Iterate over variables directly instead of pre-loading
    for var_name in class_frac_ds.data_vars:
        cls = int(var_name.split("_")[-1])
        # Load only the current fraction array
        frac_arr = class_frac_ds[var_name].values.astype(np.float32)

        harv_cls = harv_cg * frac_arr
        crop_cls = crop_cg * frac_arr

        harv_i = harv_cls * irrig_share
        harv_r = harv_cls - harv_i
        crop_i = crop_cls * irrig_share
        crop_r = crop_cls - crop_i

        harv_arr = harv_i if ws == "i" else harv_r
        crop_arr = crop_i if ws == "i" else crop_r

        harv_df = _aggregate_by_region(harv_arr, regions, cg_transform)
        crop_df = _aggregate_by_region(crop_arr, regions, cg_transform)

        if not harv_df.empty:
            harv_df["resource_class"] = cls
            harv_records.append(harv_df)
        if not crop_df.empty:
            crop_df["resource_class"] = cls
            crop_records.append(crop_df)

        # Clean up intermediate arrays for this iteration
        del (
            frac_arr,
            harv_cls,
            crop_cls,
            harv_i,
            harv_r,
            crop_i,
            crop_r,
            harv_arr,
            crop_arr,
        )
        gc.collect()

    class_frac_ds.close()

    harvested_df = (
        pd.concat(harv_records, ignore_index=True)
        if harv_records
        else pd.DataFrame(columns=["region", "value", "resource_class"])
    )
    crop_df = (
        pd.concat(crop_records, ignore_index=True)
        if crop_records
        else pd.DataFrame(columns=["region", "value", "resource_class"])
    )

    combined = harvested_df.merge(
        crop_df.rename(columns={"value": "crop_area"}),
        on=["region", "resource_class"],
        how="outer",
    ).fillna(0.0)
    combined["cropping_intensity"] = np.where(
        combined["crop_area"] > 0, combined["value"] / combined["crop_area"], np.nan
    )

    records: list[dict[str, object]] = []
    for _, row in combined.iterrows():
        region = str(row["region"])
        cls = int(row["resource_class"])
        if row["value"] > 0:
            records.append(
                {
                    "region": region,
                    "resource_class": cls,
                    "variable": "harvested_area",
                    "unit": "ha",
                    "value": float(row["value"]),
                }
            )
        if row["crop_area"] > 0:
            records.append(
                {
                    "region": region,
                    "resource_class": cls,
                    "variable": "crop_area",
                    "unit": "ha",
                    "value": float(row["crop_area"]),
                }
            )
        if np.isfinite(row["cropping_intensity"]) and row["cropping_intensity"] > 0:
            records.append(
                {
                    "region": region,
                    "resource_class": cls,
                    "variable": "cropping_intensity",
                    "unit": "-",
                    "value": float(row["cropping_intensity"]),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_path, index=False)
