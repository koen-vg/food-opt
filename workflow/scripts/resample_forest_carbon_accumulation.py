"""
SPDX-FileCopyrightText: 2025 Koen van Greevenbroek

SPDX-License-Identifier: GPL-3.0-or-later
"""

import os
from pathlib import Path

from affine import Affine
import numpy as np
from osgeo import gdal, osr
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
import xarray as xr

NO_DATA = -9999.0

# Enable GDAL exceptions for better error messages
gdal.UseExceptions()
osr.UseExceptions()

# Enable GDAL multi-threading for better performance
os.environ["GDAL_NUM_THREADS"] = "ALL_CPUS"
os.environ["GDAL_CACHEMAX"] = "512"


def _load_target_grid(
    grid_path: str,
) -> tuple[Affine, CRS, tuple[int, int], dict[str, np.ndarray]]:
    ds = xr.load_dataset(grid_path)
    try:
        transform = Affine(*ds.attrs["transform"])
    except KeyError as exc:
        raise ValueError("grid definition missing affine transform metadata") from exc
    try:
        crs = CRS.from_wkt(ds.attrs["crs_wkt"])
    except KeyError as exc:
        raise ValueError("grid definition missing CRS metadata") from exc

    height = int(ds.attrs.get("height", ds.sizes["y"]))
    width = int(ds.attrs.get("width", ds.sizes["x"]))

    cols = np.arange(width, dtype=np.float64)
    rows = np.arange(height, dtype=np.float64)
    lon = transform.c + (cols + 0.5) * transform.a
    lat = transform.f + (rows + 0.5) * transform.e

    coords = {
        "y": lat.astype(np.float32),
        "x": lon.astype(np.float32),
    }

    return transform, crs, (height, width), coords


def main() -> None:
    grid_path: str = snakemake.input.grid  # type: ignore[name-defined]
    regrowth_path: str = snakemake.input.regrowth_raw  # type: ignore[name-defined]
    output_path: str = snakemake.output.regrowth  # type: ignore[name-defined]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    (
        target_transform,
        target_crs,
        target_shape,
        coords,
    ) = _load_target_grid(grid_path)

    # Use WarpedVRT for efficient resampling - much faster than reproject()
    # for large files as it uses GDAL's optimized warping internally
    with rasterio.open(regrowth_path) as src:
        if src.crs is None:
            raise ValueError("regrowth raster missing CRS information")

        # Create a virtual warped dataset with the target resolution
        # Using bilinear resampling for performance (much faster than average)
        with WarpedVRT(
            src,
            crs=target_crs,
            transform=target_transform,
            height=target_shape[0],
            width=target_shape[1],
            resampling=Resampling.bilinear,
            src_nodata=src.nodata,
            nodata=NO_DATA,
        ) as vrt:
            # Read the entire warped dataset at once
            # This is efficient because GDAL handles the resampling internally
            dst = vrt.read(1, out_dtype=np.float32)

    dst[dst == NO_DATA] = np.nan

    ds_out = xr.Dataset(
        {
            "regrowth_tc_per_ha_yr": (("y", "x"), dst.astype(np.float32)),
        },
        coords=coords,
        attrs={
            "transform": target_transform.to_gdal(),
            "crs_wkt": target_crs.to_wkt(),
        },
    )

    encoding = {
        "regrowth_tc_per_ha_yr": {"zlib": True, "complevel": 4, "dtype": "float32"}
    }
    ds_out.to_netcdf(output_path, encoding=encoding)


if __name__ == "__main__":
    main()
