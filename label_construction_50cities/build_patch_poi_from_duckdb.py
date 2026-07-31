from pathlib import Path
import os

import duckdb
import geopandas as gpd
import pandas as pd
import tqdm


city_list_candidates = [
    Path("selected_50_fua_cities_nonempty_labels.csv"),
    Path("../selected_50_fua_cities_nonempty_labels.csv"),
    Path("selected_50_fua_cities.csv"),
    Path("../selected_50_fua_cities.csv"),
]

output_root = Path("../output_files")
overture_release = "2026-05-20.0"
tile_dir_name = "arcgis-imagery-FUA"
default_patch_filename = "{city_name}_arcgis_tiles_z14.gpkg"

for k in [
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_PROFILE",
    "AWS_DEFAULT_PROFILE",
]:
    os.environ.pop(k, None)


def resolve_city_list_csv():
    for path in city_list_candidates:
        if path.exists():
            return path
    raise FileNotFoundError("No city list CSV found.")


def load_patch_gdf(city_name):
    patch_path = output_root / city_name / tile_dir_name / default_patch_filename.format(city_name=city_name)
    if not patch_path.exists():
        raise FileNotFoundError(f"Patch gpkg not found: {patch_path}")

    patch_gdf = gpd.read_file(patch_path)
    if "image_name" not in patch_gdf.columns:
        raise ValueError(f"{patch_path} missing image_name field")

    patch_gdf = patch_gdf.to_crs("EPSG:4326")
    patch_gdf["city_query"] = city_name
    patch_gdf["patch_id"] = patch_gdf["image_name"].astype(str).str.replace(".png", "", regex=False)
    return patch_gdf


def get_patch_bbox(patch_gdf):
    bounds = patch_gdf.total_bounds
    return {
        "west": float(bounds[0]),
        "south": float(bounds[1]),
        "east": float(bounds[2]),
        "north": float(bounds[3]),
    }


def get_overture_pois_for_bbox(city_name, west, south, east, north):
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET s3_region='us-west-2';")
    con.execute("SET s3_access_key_id='';")
    con.execute("SET s3_secret_access_key='';")
    con.execute("SET s3_session_token='';")
    con.execute("SET s3_endpoint='s3.us-west-2.amazonaws.com';")
    con.execute("SET s3_url_style='path';")

    parquet_path = (
        f"s3://overturemaps-us-west-2/release/{overture_release}/"
        f"theme=places/type=place/*"
    )
    query = f"""
    SELECT
        id,
        names.primary AS name,
        categories.primary AS category,
        ST_X(geometry) AS lon,
        ST_Y(geometry) AS lat
    FROM read_parquet(
        '{parquet_path}',
        filename=true,
        hive_partitioning=1
    )
    WHERE
        bbox.xmin <= {east}
        AND bbox.xmax >= {west}
        AND bbox.ymin <= {north}
        AND bbox.ymax >= {south}
    """

    try:
        df = con.execute(query).df()
        print(f"{city_name}: Overture path worked -> {parquet_path}")
    except Exception as exc:
        con.close()
        raise RuntimeError(
            f"Failed to query Overture POIs for {city_name} using path {parquet_path}. "
            f"Original error: {exc}"
        ) from exc
    con.close()
    df["city_query"] = city_name
    return df


def assign_pois_to_patches(city_name, patch_gdf, poi_df):
    if poi_df.empty:
        empty_cols = [
            "city_query",
            "patch_id",
            "image_name",
            "poi_id",
            "name",
            "category",
            "lon",
            "lat",
        ]
        return pd.DataFrame(columns=empty_cols)

    poi_gdf = gpd.GeoDataFrame(
        poi_df.copy(),
        geometry=gpd.points_from_xy(poi_df["lon"], poi_df["lat"]),
        crs="EPSG:4326",
    )

    joined = gpd.sjoin(
        poi_gdf,
        patch_gdf[["patch_id", "image_name", "geometry"]],
        predicate="within",
        how="inner",
    ).drop(columns=["index_right", "geometry"])

    joined = joined.rename(columns={"id": "poi_id"})
    ordered_cols = [
        "city_query",
        "patch_id",
        "image_name",
        "poi_id",
        "name",
        "category",
        "lon",
        "lat",
    ]
    return joined[ordered_cols].copy()


def main():
    city_list_csv = resolve_city_list_csv()
    city_df = pd.read_csv(city_list_csv)
    city_df["city"] = city_df["city"].astype(str)

    print(f"Using city list: {city_list_csv}")
    print(f"Loaded {len(city_df)} cities")

    for _, city_row in tqdm.tqdm(city_df.iterrows(), total=len(city_df), desc="cities", leave=True):
        city_name = city_row["city"]
        city_dir = output_root / city_name / tile_dir_name
        output_csv = city_dir / f"{city_name}_patch_pois.csv"
        output_parquet = city_dir / f"{city_name}_patch_pois.parquet"

        if output_parquet.exists():
            print(f"{city_name}: existing patch POI parquet found, skip -> {output_parquet}")
            continue

        try:
            patch_gdf = load_patch_gdf(city_name)
            bbox = get_patch_bbox(patch_gdf)
            pois = get_overture_pois_for_bbox(
                city_name=city_name,
                west=bbox["west"],
                south=bbox["south"],
                east=bbox["east"],
                north=bbox["north"],
            )

            print(
                f"{city_name} patch bbox: west={bbox['west']}, south={bbox['south']}, "
                f"east={bbox['east']}, north={bbox['north']}"
            )
            print(f"{city_name} raw POI count in bbox: {len(pois)}")

            patch_pois = assign_pois_to_patches(city_name, patch_gdf, pois)
            print(f"{city_name} matched patch POI count: {len(patch_pois)}")

            patch_pois.to_csv(output_csv, index=False)
            patch_pois.to_parquet(output_parquet, index=False)
            print(f"Saved patch POIs to: {output_parquet}")
        except Exception as exc:
            print(f"{city_name}: failed to fetch patch POIs: {exc}")


if __name__ == "__main__":
    main()
