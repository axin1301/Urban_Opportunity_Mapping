import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
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


def resolve_city_list_csv():
    for path in city_list_candidates:
        if path.exists():
            return path
    raise FileNotFoundError("No city list CSV found.")


def estimate_utm_epsg(lon, lat):
    zone = int((lon + 180) / 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone


def load_patch_gdf(city_name):
    patch_path = output_root / city_name / tile_dir_name / default_patch_filename.format(city_name=city_name)
    if not patch_path.exists():
        raise FileNotFoundError(f"Patch gpkg not found: {patch_path}")

    patch_gdf = gpd.read_file(patch_path).to_crs("EPSG:4326")
    if "image_name" not in patch_gdf.columns:
        raise ValueError(f"{patch_path} missing image_name field")

    patch_gdf["city_query"] = city_name
    patch_gdf["patch_id"] = patch_gdf["image_name"].astype(str).str.replace(".png", "", regex=False)
    return patch_gdf


def load_patch_pois(city_name):
    city_dir = output_root / city_name / tile_dir_name
    parquet_path = city_dir / f"{city_name}_patch_pois.parquet"
    csv_path = city_dir / f"{city_name}_patch_pois.csv"

    if parquet_path.exists():
        poi_df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        poi_df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"No patch POI file found for {city_name}")

    if "image_name" not in poi_df.columns:
        raise ValueError(f"Patch POI file for {city_name} missing image_name column")

    poi_df["city_query"] = city_name
    poi_df["image_name"] = poi_df["image_name"].astype(str)
    poi_df["category"] = poi_df["category"].fillna("unknown").astype(str)
    return poi_df


def collect_global_top_categories(city_names, top_k):
    category_counts = {}
    for city_name in tqdm.tqdm(city_names, desc="scan poi categories", leave=True):
        try:
            poi_df = load_patch_pois(city_name)
        except Exception as exc:
            print(f"{city_name}: skip category scan -> {exc}")
            continue

        counts = poi_df["category"].value_counts()
        for category, count in counts.items():
            category_counts[category] = category_counts.get(category, 0) + int(count)

    sorted_categories = sorted(category_counts.items(), key=lambda item: item[1], reverse=True)
    return [category for category, _ in sorted_categories[:top_k]]


def build_patch_poi_covariates(city_name, patch_gdf, poi_df, top_categories):
    city_union = patch_gdf.geometry.union_all().centroid
    utm_epsg = estimate_utm_epsg(city_union.x, city_union.y)

    patch_metric = patch_gdf.to_crs(epsg=utm_epsg)
    patch_area_km2 = patch_metric.geometry.area / 1_000_000.0

    patch_base = patch_gdf.drop(columns="geometry").copy()
    patch_base["patch_area_km2"] = patch_area_km2.to_numpy()
    patch_base["patch_lon"] = patch_metric.geometry.centroid.to_crs("EPSG:4326").x
    patch_base["patch_lat"] = patch_metric.geometry.centroid.to_crs("EPSG:4326").y

    if poi_df.empty:
        patch_base["poi_count"] = 0
        patch_base["log1p_poi_count"] = 0.0
        patch_base["poi_density"] = 0.0
        patch_base["unique_poi_category_count"] = 0
        for category in top_categories:
            safe = category.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
            patch_base[f"poi_count_{safe}"] = 0
            patch_base[f"poi_density_{safe}"] = 0.0
        return patch_base

    patch_counts = poi_df.groupby("image_name").size().rename("poi_count").reset_index()
    unique_category_counts = (
        poi_df.groupby("image_name")["category"]
        .nunique()
        .rename("unique_poi_category_count")
        .reset_index()
    )

    patch_cov = patch_base.merge(patch_counts, on="image_name", how="left")
    patch_cov = patch_cov.merge(unique_category_counts, on="image_name", how="left")
    patch_cov["poi_count"] = patch_cov["poi_count"].fillna(0).astype(int)
    patch_cov["unique_poi_category_count"] = patch_cov["unique_poi_category_count"].fillna(0).astype(int)
    patch_cov["log1p_poi_count"] = np.log1p(patch_cov["poi_count"])
    patch_cov["poi_density"] = np.where(
        patch_cov["patch_area_km2"] > 0,
        patch_cov["poi_count"] / patch_cov["patch_area_km2"],
        np.nan,
    )

    category_df = poi_df[poi_df["category"].isin(top_categories)].copy()
    if not category_df.empty:
        category_counts = (
            category_df.groupby(["image_name", "category"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        patch_cov = patch_cov.merge(category_counts, on="image_name", how="left")
    else:
        category_counts = None

    for category in top_categories:
        safe = category.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
        if category not in patch_cov.columns:
            patch_cov[category] = 0
        patch_cov[category] = patch_cov[category].fillna(0).astype(int)
        patch_cov[f"poi_count_{safe}"] = patch_cov[category]
        patch_cov[f"poi_density_{safe}"] = np.where(
            patch_cov["patch_area_km2"] > 0,
            patch_cov[category] / patch_cov["patch_area_km2"],
            np.nan,
        )

    if category_counts is not None:
        patch_cov = patch_cov.drop(columns=[category for category in top_categories if category in patch_cov.columns])

    return patch_cov


def main():
    parser = argparse.ArgumentParser(description="Aggregate patch-level POI covariates from patch POI tables")
    parser.add_argument("--city-list-csv", type=str, default=None)
    parser.add_argument("--city", type=str, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--top-k-categories", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

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

    city_names = list(city_df["city"])
    print(f"Using city list: {city_list_csv}")
    print(f"Loaded {len(city_names)} cities")

    top_categories = collect_global_top_categories(city_names, args.top_k_categories)
    category_vocab_path = output_root / "patch_poi_top_categories.json"
    category_vocab_path.write_text(json.dumps(top_categories, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved top POI categories -> {category_vocab_path}")

    for city_name in tqdm.tqdm(city_names, desc="cities", leave=True):
        city_dir = output_root / city_name / tile_dir_name
        output_csv = city_dir / f"{city_name}_patch_poi_covariates.csv"
        output_parquet = city_dir / f"{city_name}_patch_poi_covariates.parquet"

        if output_parquet.exists() and not args.overwrite:
            print(f"{city_name}: existing patch POI covariates found, skip -> {output_parquet}")
            continue

        try:
            patch_gdf = load_patch_gdf(city_name)
            poi_df = load_patch_pois(city_name)
            covariates_df = build_patch_poi_covariates(city_name, patch_gdf, poi_df, top_categories)
            covariates_df.to_csv(output_csv, index=False)
            covariates_df.to_parquet(output_parquet, index=False)
            print(f"{city_name}: saved patch POI covariates -> {output_parquet}")
        except Exception as exc:
            print(f"{city_name}: failed to build patch POI covariates: {exc}")


if __name__ == "__main__":
    main()
