"""
SPDX-FileCopyrightText: 2025 Koen van Greevenbroek

SPDX-License-Identifier: GPL-3.0-or-later
"""

from pathlib import Path
import sys

from osgeo import gdal, osr

gdal.UseExceptions()
osr.UseExceptions()

from exactextract import exact_extract  # noqa: E402
from exactextract.raster import NumPyRasterSource  # noqa: E402
import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

# Ensure workflow/scripts is on path for raster_utils
sys.path.insert(0, str(Path(__file__).parent))

from raster_utils import (  # noqa: E402
    calculate_all_cell_areas,
    raster_bounds,
    read_raster_float,
)


def _load_mapping(mapping_path: Path) -> pd.DataFrame:
    df = pd.read_csv(mapping_path)
    df["crop_name"] = df["crop_name"].astype(str).str.strip()
    df["res06_code"] = df["res06_code"].astype(str).str.strip().str.upper()
    return df


def _shares_for_crop(
    crop: str,
    mapping_df: pd.DataFrame,
    production_df: pd.DataFrame,
) -> tuple[dict[str, float], float]:
    """Return country-specific shares and fallback share for the given crop."""

    row = mapping_df[mapping_df["crop_name"] == crop]
    if row.empty:
        raise ValueError(f"Crop '{crop}' missing from RES06 mapping table")
    module_code = str(row.iloc[0]["res06_code"]).upper()

    crops_in_module: list[str] = (
        mapping_df[mapping_df["res06_code"].str.upper() == module_code]["crop_name"]
        .astype(str)
        .str.strip()
        .tolist()
    )
    crops_in_module = sorted(set(crops_in_module))
    if crop not in crops_in_module:
        crops_in_module.append(crop)

    if len(crops_in_module) == 1:
        return {}, 1.0

    production_df = production_df.copy()
    production_df["crop"] = production_df["crop"].astype(str).str.strip()
    production_df["country"] = production_df["country"].astype(str).str.upper()
    production_df = production_df[production_df["crop"].isin(crops_in_module)]

    if production_df.empty:
        uniform_share = 1.0 / len(crops_in_module)
        return {}, uniform_share

    production_df["production_tonnes"] = pd.to_numeric(
        production_df["production_tonnes"], errors="coerce"
    ).fillna(0.0)

    by_country = (
        production_df.groupby(["country", "crop"])["production_tonnes"]
        .sum()
        .rename("crop_total")
        .reset_index()
    )
    country_totals = (
        by_country.groupby("country")["crop_total"].sum().rename("country_total")
    )
    by_country = by_country.merge(country_totals, on="country", how="left")
    by_country["share"] = np.where(
        by_country["country_total"] > 0,
        by_country["crop_total"] / by_country["country_total"],
        np.nan,
    )

    shares_lookup: dict[str, float] = {
        country: float(share)
        for country, share in zip(
            by_country[by_country["crop"] == crop]["country"],
            by_country[by_country["crop"] == crop]["share"],
        )
        if np.isfinite(share)
    }

    global_totals = (
        production_df.groupby("crop")["production_tonnes"].sum().rename("global_total")
    )
    global_denominator = float(global_totals.sum())
    if global_denominator > 0:
        fallback_share = float(global_totals.get(crop, 0.0) / global_denominator)
    else:
        fallback_share = 1.0 / len(crops_in_module)

    if fallback_share == 0.0:
        fallback_share = 1.0 / len(crops_in_module)

    return shares_lookup, fallback_share


def _apply_shares(
    df: pd.DataFrame,
    crop: str,
    shares_lookup: dict[str, float],
    fallback_share: float,
) -> pd.DataFrame:
    def _share(country: str) -> float:
        return shares_lookup.get(country, fallback_share)

    df = df.copy()
    df["share"] = df["country"].map(_share).fillna(fallback_share)
    df["value"] = df["value"] * df["share"]
    return df.drop(columns=["share"])


def _extract_harvested_area(
    raster: np.ndarray,
    transform,
    crs_wkt: str | None,
    class_labels: np.ndarray,
    regions: gpd.GeoDataFrame,
) -> pd.DataFrame:
    xmin, ymin, xmax, ymax = raster_bounds(transform, raster.shape[1], raster.shape[0])

    regions_for_extract = regions.reset_index(drop=True)

    records: list[pd.DataFrame] = []
    n_classes = (
        int(np.nanmax(class_labels)) + 1 if np.isfinite(class_labels).any() else 0
    )
    for cls in range(n_classes):
        mask = class_labels == cls
        if not np.any(mask):
            continue
        masked = np.where(mask, raster, np.nan)
        raster_src = NumPyRasterSource(
            masked,
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            nodata=np.nan,
            srs_wkt=crs_wkt,
        )
        stats = exact_extract(
            raster_src,
            regions_for_extract,
            ["sum"],
            include_cols=["region"],
            output="pandas",
        )
        if stats.empty:
            continue
        stats = stats.rename(columns={"sum": "value"})
        stats["resource_class"] = cls
        records.append(stats)

    if not records:
        return pd.DataFrame(columns=["region", "resource_class", "value"])

    combined = pd.concat(records, ignore_index=True)
    combined["resource_class"] = combined["resource_class"].astype(int)
    return combined


def main() -> None:
    classes_nc = Path(snakemake.input.classes)  # type: ignore[name-defined]
    raster_path = Path(snakemake.input.harvested_area_raster)  # type: ignore[name-defined]
    regions_path = Path(snakemake.input.regions)  # type: ignore[name-defined]
    mapping_path = Path(snakemake.input.crop_mapping)  # type: ignore[name-defined]
    production_path = Path(snakemake.input.faostat_production)  # type: ignore[name-defined]
    output_path = Path(snakemake.output[0])  # type: ignore[name-defined]
    crop = str(snakemake.wildcards.crop)  # type: ignore[name-defined]

    ds = xr.load_dataset(classes_nc)
    class_labels = ds["resource_class"].values.astype(np.int16)

    harvested_raw, src = read_raster_float(raster_path)
    try:
        cell_area_ha = calculate_all_cell_areas(src)
        harvested_raw = harvested_raw * cell_area_ha
        transform = src.transform
        crs_wkt = src.crs.to_wkt() if src.crs else None
    finally:
        src.close()

    regions = gpd.read_file(regions_path)[["region", "country", "geometry"]]
    regions["country"] = regions["country"].astype(str).str.upper()

    mapping_df = _load_mapping(mapping_path)
    production_df = pd.read_csv(production_path)

    shares_lookup, fallback_share = _shares_for_crop(crop, mapping_df, production_df)

    extracted = _extract_harvested_area(
        harvested_raw,
        transform,
        crs_wkt,
        class_labels,
        regions,
    )
    if extracted.empty:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            columns=["region", "resource_class", "variable", "unit", "value"]
        ).to_csv(output_path, index=False)
        return

    extracted = extracted.merge(regions[["region", "country"]], on="region", how="left")
    extracted = _apply_shares(extracted, crop, shares_lookup, fallback_share)

    extracted["variable"] = "harvested_area"
    extracted["unit"] = "ha"

    extracted = extracted.loc[
        :, ["region", "resource_class", "variable", "unit", "value"]
    ]
    extracted = extracted[extracted["value"] > 0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    extracted.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
