import argparse
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import tqdm


city_list_candidates = [
    Path("selected_50_fua_cities.csv"),
    Path("../selected_50_fua_cities.csv"),
    Path("selected_10_fua_cities.csv"),
    Path("../selected_10_fua_cities.csv"),
    Path("cities_with_FUA_ge_1M.csv"),
    Path("../cities_with_FUA_ge_1M.csv"),
]

output_root = Path("../output_files")
tile_dir_name = "arcgis-imagery-FUA"
default_beta_meters = 1000
default_patch_filename = "{city_name}_arcgis_tiles_z14.gpkg"

required_cell_columns = [
    "LON",
    "LAT",
]

def resolve_city_list_csv():
    for path in city_list_candidates:
        if path.exists():
            return path
    raise FileNotFoundError("No city list CSV found.")


def normalize_radio(value):
    if pd.isna(value):
        return "OTHER"

    radio = str(value).strip().upper()
    if "NR" == radio or radio.endswith("NR") or "5G" in radio:
        return "NR"
    if "LTE" in radio or "4G" in radio:
        return "LTE"
    if "UMTS" in radio or "WCDMA" in radio or "3G" in radio:
        return "UMTS"
    if "GSM" in radio or "2G" in radio:
        return "GSM"
    return "OTHER"


def estimate_utm_epsg(lon, lat):
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        return 32600 + zone
    return 32700 + zone


def compute_city_percentile(values):
    values = pd.to_numeric(values, errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=values.index)
    return values.rank(method="average", pct=True)


def read_all_cell_csvs(cell_dir):
    csv_paths = sorted(Path(cell_dir).rglob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {cell_dir}")

    frames = []
    for csv_path in tqdm.tqdm(csv_paths, desc="load cell csv", leave=True):
        df = pd.read_csv(csv_path)
        missing = [column for column in required_cell_columns if column not in df.columns]
        if missing:
            raise ValueError(f"{csv_path} missing required columns: {missing}")

        df = df.copy()
        df["source_csv"] = csv_path.name
        frames.append(df)

    cells = pd.concat(frames, ignore_index=True)
    cells = cells.dropna(subset=["LON", "LAT"]).copy()
    cells["LON"] = pd.to_numeric(cells["LON"], errors="coerce")
    cells["LAT"] = pd.to_numeric(cells["LAT"], errors="coerce")
    cells = cells.dropna(subset=["LON", "LAT"]).copy()
    cells = cells[(cells["LON"].between(-180, 180)) & (cells["LAT"].between(-90, 90))].copy()
    if "radio" in cells.columns:
        cells["radio_norm"] = cells["radio"].map(normalize_radio)
    else:
        cells["radio_norm"] = "UNKNOWN"
    return cells


def cells_to_gdf(cells_df):
    return gpd.GeoDataFrame(
        cells_df.copy(),
        geometry=gpd.points_from_xy(cells_df["LON"], cells_df["LAT"]),
        crs="EPSG:4326",
    )


def compute_digital_infrastructure_intensity(patch_centroids_metric, cell_points_metric, beta_meters):
    if len(cell_points_metric) == 0:
        return np.zeros(len(patch_centroids_metric), dtype=float)

    patch_xy = np.column_stack([patch_centroids_metric.x.to_numpy(), patch_centroids_metric.y.to_numpy()])
    cell_xy = np.column_stack([cell_points_metric.x.to_numpy(), cell_points_metric.y.to_numpy()])

    values = np.zeros(len(patch_xy), dtype=float)
    for index, (patch_x, patch_y) in enumerate(patch_xy):
        distances = np.sqrt((cell_xy[:, 0] - patch_x) ** 2 + (cell_xy[:, 1] - patch_y) ** 2)
        values[index] = np.sum(np.exp(-distances / beta_meters))
    return values


def build_patch_labels_for_city(city_name, patch_gdf, cells_gdf, beta_meters):
    patch_gdf = patch_gdf.copy()
    if "image_name" not in patch_gdf.columns:
        raise ValueError(f"{city_name} patch GPKG missing image_name field.")

    patch_gdf["patch_id"] = patch_gdf["image_name"].astype(str).str.replace(".png", "", regex=False)
    patch_gdf["city_query"] = city_name

    bounds = patch_gdf.total_bounds
    approx_degree_buffer = beta_meters / 111320.0
    cells_candidate = cells_gdf.cx[
        bounds[0] - approx_degree_buffer : bounds[2] + approx_degree_buffer,
        bounds[1] - approx_degree_buffer : bounds[3] + approx_degree_buffer,
    ].copy()

    cells_in_patch = gpd.sjoin(
        cells_candidate,
        patch_gdf[["patch_id", "image_name", "geometry"]],
        predicate="within",
        how="inner",
    ).drop(columns=["index_right"])

    patch_labels = patch_gdf.copy()

    city_centroid = patch_gdf.to_crs("EPSG:4326").geometry.union_all().centroid
    utm_epsg = estimate_utm_epsg(city_centroid.x, city_centroid.y)

    patch_metric = patch_gdf.to_crs(epsg=utm_epsg)
    cells_metric = cells_candidate.to_crs(epsg=utm_epsg)
    patch_centroids_metric = patch_metric.geometry.centroid

    dii_raw = compute_digital_infrastructure_intensity(
        patch_centroids_metric,
        cells_metric.geometry,
        beta_meters,
    )
    dii_log1p = np.log1p(dii_raw)
    patch_labels["digital_total_cell_intensity_raw"] = dii_raw
    patch_labels["digital_total_cell_intensity_log1p"] = dii_log1p
    patch_labels["digital_total_cell_intensity_city_pct"] = compute_city_percentile(
        pd.Series(dii_log1p, index=patch_labels.index)
    )
    patch_labels["digital_total_cell_intensity"] = patch_labels["digital_total_cell_intensity_city_pct"]
    patch_labels["dii_beta_m"] = beta_meters
    patch_labels["metric_crs_epsg"] = utm_epsg

    patch_centroids_wgs84 = patch_metric.geometry.centroid.to_crs("EPSG:4326")
    patch_labels["patch_lon"] = patch_centroids_wgs84.x
    patch_labels["patch_lat"] = patch_centroids_wgs84.y
    patch_labels["min_lon"] = patch_labels.geometry.bounds["minx"]
    patch_labels["min_lat"] = patch_labels.geometry.bounds["miny"]
    patch_labels["max_lon"] = patch_labels.geometry.bounds["maxx"]
    patch_labels["max_lat"] = patch_labels.geometry.bounds["maxy"]

    return patch_labels, cells_in_patch


def main():
    parser = argparse.ArgumentParser(
        description="Build patch-level Digital Infrastructure Intensity (DII) for ArcGIS FUA tiles"
    )
    parser.add_argument("--cell-dir", type=str, required=True, help="Directory containing base station CSV files")
    parser.add_argument("--city", type=str, default=None, help="Optional single city name")
    parser.add_argument("--city-list-csv", type=str, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument(
        "--beta",
        type=int,
        default=default_beta_meters,
        help="Distance-decay bandwidth in meters for DII",
    )
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

    print(f"Using city list: {city_list_csv}")
    print(f"Loaded {len(city_df)} cities for label building")

    cells_df = read_all_cell_csvs(args.cell_dir)
    cells_gdf = cells_to_gdf(cells_df)
    print(f"Loaded {len(cells_gdf)} cell records from {args.cell_dir}")

    combined_city_rows = []

    for _, city_row in tqdm.tqdm(city_df.iterrows(), total=len(city_df), desc="cities", leave=True):
        city_name = city_row["city"]
        city_dir = output_root / city_name / tile_dir_name
        patch_path = city_dir / default_patch_filename.format(city_name=city_name)

        if not patch_path.exists():
            print(f"{city_name}: missing patch file, skip -> {patch_path}")
            continue

        try:
            patch_gdf = gpd.read_file(patch_path)
            patch_labels, matched_cells = build_patch_labels_for_city(
                city_name=city_name,
                patch_gdf=patch_gdf,
                cells_gdf=cells_gdf,
                beta_meters=args.beta,
            )

            label_csv = city_dir / f"{city_name}_base_station_labels.csv"
            label_parquet = city_dir / f"{city_name}_base_station_labels.parquet"
            label_gpkg = city_dir / f"{city_name}_base_station_labels.gpkg"
            cell_match_csv = city_dir / f"{city_name}_matched_cells.csv"

            patch_labels.to_csv(label_csv, index=False)
            patch_labels.to_parquet(label_parquet, index=False)
            if label_gpkg.exists():
                label_gpkg.unlink()
            patch_labels.to_file(label_gpkg, driver="GPKG")

            matched_cells.drop(columns="geometry").to_csv(cell_match_csv, index=False)

            city_summary = pd.DataFrame(
                {
                    "city": [city_name],
                    "patch_count": [len(patch_labels)],
                    "matched_cell_count": [len(matched_cells)],
                    "mean_dii_raw": [patch_labels["digital_total_cell_intensity_raw"].mean()],
                    "mean_dii_log1p": [patch_labels["digital_total_cell_intensity_log1p"].mean()],
                    "mean_dii_city_pct": [patch_labels["digital_total_cell_intensity"].mean()],
                }
            )
            combined_city_rows.append(city_summary)

            print(
                f"{city_name}: patches={len(patch_labels)}, matched_cells={len(matched_cells)}, "
                f"mean_dii={patch_labels['digital_total_cell_intensity'].mean():.3f}"
            )
        except Exception as exc:
            print(f"{city_name}: failed to build labels: {exc}")

    if combined_city_rows:
        combined_summary = pd.concat(combined_city_rows, ignore_index=True)
        combined_summary.to_csv(output_root / "arcgis_fua_city_label_summary.csv", index=False)


if __name__ == "__main__":
    main()
