"""
SPDX-FileCopyrightText: 2025 Koen van Greevenbroek

SPDX-License-Identifier: GPL-3.0-or-later

Compute fractional overlap of resource classes on the CROPGRIDS 0.05° grid.

Output: NetCDF with one band per resource class (class_fraction_{k}), storing
the fraction of each 0.05° cell covered by that class. This conserves total
area and avoids nearest-neighbour artefacts when aggregating CROPGRIDS data.
"""

from pathlib import Path

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import Affine, from_origin
from rasterio.warp import Resampling, reproject
import xarray as xr


def _target_grid():
    # 0.05° global grid matching CROPGRIDS
    transform = from_origin(-180.0, 90.0, 0.05, 0.05)
    width = 7200
    height = 3600
    return transform, width, height


if __name__ == "__main__":
    classes_path = Path(snakemake.input.classes)  # type: ignore[name-defined]
    output_path = Path(snakemake.output[0])  # type: ignore[name-defined]

    # Open resource_classes.nc as an xarray dataset to properly read global attributes
    ds_src = xr.open_dataset(classes_path)
    class_arr = ds_src["resource_class"].values.astype(np.int16)

    # Extract georeferencing from global attributes, parsing the transform string
    src_transform = Affine.from_gdal(*ds_src.attrs["transform"])
    src_crs = CRS.from_wkt(ds_src.attrs["crs_wkt"])
    ds_src.close()  # Close dataset after reading necessary data

    transform_tgt, width_tgt, height_tgt = _target_grid()
    bands: dict[str, np.ndarray] = {}

    classes = np.unique(class_arr[np.isfinite(class_arr)])
    classes = classes[classes >= 0]

    for cls in classes:
        mask = (class_arr == cls).astype(np.float32)
        dst = np.zeros((height_tgt, width_tgt), dtype=np.float32)
        reproject(
            mask,
            dst,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=transform_tgt,
            dst_crs=CRS.from_epsg(4326),
            resampling=Resampling.average,
        )
        bands[f"class_fraction_{int(cls)}"] = dst

    # Write NetCDF
    coords = {
        "y": np.arange(height_tgt),
        "x": np.arange(width_tgt),
    }
    data_vars = {name: (("y", "x"), arr) for name, arr in bands.items()}
    ds = xr.Dataset(data_vars, coords=coords)
    ds.attrs["crs"] = src_crs.to_wkt() if src_crs else ""
    ds.attrs["transform"] = list(transform_tgt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_path)
