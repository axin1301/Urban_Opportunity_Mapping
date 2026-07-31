import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
import tqdm


city_list_candidates = [
    Path("selected_50_fua_cities_nonempty_labels.csv"),
    Path("../selected_50_fua_cities_nonempty_labels.csv"),
    Path("selected_50_fua_cities.csv"),
    Path("../selected_50_fua_cities.csv"),
]

output_root = Path("../output_files")
tile_dir_name = "arcgis-imagery-FUA"
default_patch_filename = "{city_name}_arcgis_tiles_z14.gpkg"

# ESA WorldCover class code -> column suffix
esa_landuse_map = {
    10: "tree_cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built_up",
    60: "bare_sparse",
    70: "snow_ice",
    80: "permanent_water",
    90: "herbaceous_wetland",
    95: "mangroves",
    100: "moss_lichen",
}


def resolve_city_list_csv():
    for path in city_list_candidates:
        if path.exists():
            return path
    raise FileNotFoundError("No city list CSV found.")


def ensure_raster_path(path_str, name):
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"{name} raster not found: {path}")
    if path.suffix.lower() == ".gz":
        raise ValueError(
            f"{name} raster is gzipped: {path}. Please decompress it to .tif before running."
        )
    return path


def compute_continuous_stats(dataset, geometry):
    try:
        data, _ = mask(dataset, [geometry], crop=True, filled=False)
    except ValueError:
        return {
            "mean": np.nan,
            "sum": np.nan,
            "count": 0,
        }

    band = np.ma.array(data[0])
    valid = band.compressed()
    if valid.size == 0:
        return {
            "mean": np.nan,
            "sum": np.nan,
            "count": 0,
        }

    return {
        "mean": float(valid.mean()),
        "sum": float(valid.sum()),
        "count": int(valid.size),
    }


def compute_categorical_fractions(dataset, geometry, class_map):
    try:
        data, _ = mask(dataset, [geometry], crop=True, filled=False)
    except ValueError:
        return {f"landuse_{suffix}_frac": np.nan for suffix in class_map.values()}

    band = np.ma.array(data[0])
    valid = band.compressed()
    if valid.size == 0:
        return {f"landuse_{suffix}_frac": np.nan for suffix in class_map.values()}

    total = float(valid.size)
    fractions = {}
    for class_code, suffix in class_map.items():
        fractions[f"landuse_{suffix}_frac"] = float((valid == class_code).sum() / total)
    return fractions


def empty_landuse_fraction_row():
    return {f"landuse_{suffix}_frac": np.nan for suffix in esa_landuse_map.values()}


def load_patch_gdf(city_name):
    patch_path = output_root / city_name / tile_dir_name / default_patch_filename.format(city_name=city_name)
    if not patch_path.exists():
        raise FileNotFoundError(f"Patch gpkg not found: {patch_path}")
    patch_gdf = gpd.read_file(patch_path)
    if "image_name" not in patch_gdf.columns:
        raise ValueError(f"{patch_path} missing image_name field")
    patch_gdf["city_query"] = city_name
    return patch_gdf


def compute_patch_covariates(city_name, patch_gdf, ntl_ds, pop_ds, landuse_ds=None):
    ntl_gdf = patch_gdf.to_crs(ntl_ds.crs)
    pop_gdf = patch_gdf.to_crs(pop_ds.crs)
    landuse_gdf = patch_gdf.to_crs(landuse_ds.crs) if landuse_ds is not None else None

    rows = []
    patch_iterator = patch_gdf.itertuples(index=False)
    ntl_geoms = ntl_gdf.geometry.tolist()
    pop_geoms = pop_gdf.geometry.tolist()
    landuse_geoms = landuse_gdf.geometry.tolist() if landuse_gdf is not None else [None] * len(patch_gdf)

    iterator = zip(patch_iterator, ntl_geoms, pop_geoms, landuse_geoms)

    for patch_row, ntl_geom, pop_geom, landuse_geom in tqdm.tqdm(
        iterator,
        total=len(patch_gdf),
        desc=f"{city_name} raster covariates",
        leave=True,
    ):
        ntl_stats = compute_continuous_stats(ntl_ds, ntl_geom)
        pop_stats = compute_continuous_stats(pop_ds, pop_geom)
        if landuse_ds is not None and landuse_geom is not None:
            landuse_fracs = compute_categorical_fractions(landuse_ds, landuse_geom, esa_landuse_map)
        else:
            landuse_fracs = empty_landuse_fraction_row()

        row = {
            "city_query": city_name,
            "image_name": patch_row.image_name,
            "tile_x": getattr(patch_row, "tile_x", np.nan),
            "tile_y": getattr(patch_row, "tile_y", np.nan),
            "min_lon": patch_row.min_lon,
            "min_lat": patch_row.min_lat,
            "max_lon": patch_row.max_lon,
            "max_lat": patch_row.max_lat,
            "nightlight_mean": ntl_stats["mean"],
            "nightlight_sum": ntl_stats["sum"],
            "nightlight_valid_pixels": ntl_stats["count"],
            "population_mean": pop_stats["mean"],
            "population_sum": pop_stats["sum"],
            "population_valid_pixels": pop_stats["count"],
        }
        row.update(landuse_fracs)
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Build patch-level raster covariates for ArcGIS FUA tiles"
    )
    parser.add_argument("--ntl-raster", type=str, required=True, help="Path to decompressed VIIRS nightlight .tif")
    parser.add_argument("--population-raster", type=str, required=True, help="Path to GHSL population raster .tif")
    parser.add_argument("--landuse-raster", type=str, default=None, help="Optional path to ESA WorldCover raster .tif")
    parser.add_argument("--city-list-csv", type=str, default=None)
    parser.add_argument("--city", type=str, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    ntl_raster = ensure_raster_path(args.ntl_raster, "Nightlight")
    population_raster = ensure_raster_path(args.population_raster, "Population")
    landuse_raster = ensure_raster_path(args.landuse_raster, "Land-use") if args.landuse_raster else None

    city_list_csv = Path(args.city_list_csv) if args.city_list_csv else resolve_city_list_csv()
    if not city_list_csv.exists():
        raise FileNotFoundError(f"City list CSV not found: {city_list_csv}")

    city_df = pd.read_csv(city_list_csv)
    city_df["city"] = city_df["city"].astype(str)

    if args.city:
        city_df = city_df[city_df["city"].str.lower() == args.city.lower()].copy()
        if city_df.empty:
            raise ValueError(f"City not found in {city_list_csv}: {args.city}")
    else:
        start = args.start or 0
        end = args.end if args.end is not None else len(city_df)
        city_df = city_df.iloc[start:end].copy()

    print(f"Using city list: {city_list_csv}")
    print(f"Loaded {len(city_df)} cities")

    with rasterio.open(ntl_raster) as ntl_ds, rasterio.open(population_raster) as pop_ds:
        if landuse_raster is not None:
            with rasterio.open(landuse_raster) as landuse_ds:
                for _, city_row in tqdm.tqdm(city_df.iterrows(), total=len(city_df), desc="cities", leave=True):
                    city_name = city_row["city"]
                    city_dir = output_root / city_name / tile_dir_name
                    output_csv = city_dir / f"{city_name}_patch_raster_covariates.csv"
                    output_parquet = city_dir / f"{city_name}_patch_raster_covariates.parquet"

                    if output_parquet.exists() and not args.overwrite:
                        print(f"{city_name}: existing raster covariate parquet found, skip -> {output_parquet}")
                        continue

                    try:
                        patch_gdf = load_patch_gdf(city_name)
                        covariate_df = compute_patch_covariates(
                            city_name=city_name,
                            patch_gdf=patch_gdf,
                            ntl_ds=ntl_ds,
                            pop_ds=pop_ds,
                            landuse_ds=landuse_ds,
                        )
                        covariate_df.to_csv(output_csv, index=False)
                        covariate_df.to_parquet(output_parquet, index=False)
                        print(f"{city_name}: saved raster covariates -> {output_parquet}")
                    except Exception as exc:
                        print(f"{city_name}: failed to compute raster covariates: {exc}")
        else:
            for _, city_row in tqdm.tqdm(city_df.iterrows(), total=len(city_df), desc="cities", leave=True):
                city_name = city_row["city"]
                city_dir = output_root / city_name / tile_dir_name
                output_csv = city_dir / f"{city_name}_patch_raster_covariates.csv"
                output_parquet = city_dir / f"{city_name}_patch_raster_covariates.parquet"

                if output_parquet.exists() and not args.overwrite:
                    print(f"{city_name}: existing raster covariate parquet found, skip -> {output_parquet}")
                    continue

                try:
                    patch_gdf = load_patch_gdf(city_name)
                    covariate_df = compute_patch_covariates(
                        city_name=city_name,
                        patch_gdf=patch_gdf,
                        ntl_ds=ntl_ds,
                        pop_ds=pop_ds,
                        landuse_ds=None,
                    )
                    covariate_df.to_csv(output_csv, index=False)
                    covariate_df.to_parquet(output_parquet, index=False)
                    print(f"{city_name}: saved raster covariates -> {output_parquet}")
                except Exception as exc:
                    print(f"{city_name}: failed to compute raster covariates: {exc}")


if __name__ == "__main__":
    main()
