"""
SPDX-FileCopyrightText: 2025 Koen van Greevenbroek

SPDX-License-Identifier: GPL-3.0-or-later
"""

from pathlib import Path

from affine import Affine
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

try:  # Prefer package import when executed via Snakemake
    from workflow.scripts.raster_utils import calculate_all_cell_areas, raster_bounds
except ImportError:  # pragma: no cover - fallback when run via `snakemake --script`
    from raster_utils import calculate_all_cell_areas, raster_bounds  # type: ignore


def _build_dummy_raster(transform: Affine, width: int, height: int):
    class _DummyRaster:
        def __init__(self, transform: Affine, width: int, height: int) -> None:
            self.transform = transform
            self.shape = (height, width)
            xmin, ymin, xmax, ymax = raster_bounds(transform, width, height)
            self.bounds = (xmin, ymin, xmax, ymax)

    return _DummyRaster(transform, width, height)


def _transform_from_attrs(ds: xr.Dataset) -> Affine:
    try:
        return Affine.from_gdal(*ds.attrs["transform"])
    except KeyError as exc:  # pragma: no cover - sanity guard
        raise ValueError(
            "resource_classes.nc missing affine transform metadata"
        ) from exc


if __name__ == "__main__":
    classes_path: str = snakemake.input.classes  # type: ignore[name-defined]
    lc_masks_path: str = snakemake.input.lc_masks  # type: ignore[name-defined]
    regions_path: str = snakemake.input.regions  # type: ignore[name-defined]
    output_path = Path(snakemake.output[0])  # type: ignore[name-defined]

    classes_ds = xr.load_dataset(classes_path)
    region_id = classes_ds["region_id"].astype(np.int32).values
    resource_class = classes_ds["resource_class"].astype(np.int16).values
    transform = _transform_from_attrs(classes_ds)
    height, width = region_id.shape

    lc_ds = xr.load_dataset(lc_masks_path)
    grass_frac = lc_ds["grassland_fraction"].astype(np.float32).values
    if grass_frac.shape != region_id.shape:
        raise ValueError(
            "Grassland fraction grid does not match the resource_classes grid"
        )

    np.copyto(grass_frac, 0.0, where=~np.isfinite(grass_frac))
    np.clip(grass_frac, 0.0, 1.0, out=grass_frac)

    dummy_raster = _build_dummy_raster(transform, width, height)
    cell_area = calculate_all_cell_areas(dummy_raster)
    grass_area = grass_frac * cell_area

    valid = (
        np.isfinite(grass_area)
        & (grass_area > 0.0)
        & np.isfinite(region_id)
        & np.isfinite(resource_class)
        & (region_id >= 0)
        & (resource_class >= 0)
    )
    if not np.any(valid):
        df = pd.DataFrame(columns=["region", "resource_class", "area_ha"])
    else:
        region_vals = region_id[valid].astype(np.int32, copy=False)
        class_vals = resource_class[valid].astype(np.int32, copy=False)
        area_vals = grass_area[valid].astype(np.float64, copy=False)

        regions_gdf = gpd.read_file(regions_path)
        if "region" not in regions_gdf.columns:
            raise ValueError("regions.geojson must contain a 'region' column")
        region_lookup = (
            regions_gdf.reset_index().set_index("index")["region"].astype(str).to_dict()
        )

        df = (
            pd.DataFrame(
                {
                    "region_id": region_vals,
                    "resource_class": class_vals,
                    "area_ha": area_vals,
                }
            )
            .groupby(["region_id", "resource_class"], as_index=False)["area_ha"]
            .sum()
        )
        df["region"] = df["region_id"].map(region_lookup)
        missing = df["region"].isna()
        if missing.any():
            missing_ids = sorted(df.loc[missing, "region_id"].unique().tolist())
            raise ValueError(
                "Region IDs in resource_classes.nc missing from regions.geojson: "
                + ", ".join(str(mid) for mid in missing_ids)
            )
        df = df[["region", "resource_class", "area_ha"]]
        df = df.sort_values(["region", "resource_class"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
