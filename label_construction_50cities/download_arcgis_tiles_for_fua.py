import argparse
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
import tqdm
from shapely.geometry import box


city_list_candidates = [
    Path("selected_50_fua_cities.csv"),
]


def resolve_city_list_csv():
    for path in city_list_candidates:
        if path.exists():
            return path
    raise FileNotFoundError("No city list CSV found.")


city_list_csv = resolve_city_list_csv()
csv_fua = pd.read_csv(city_list_csv)
city_names = list(csv_fua["city"])

output_root = Path("../output_files")
tile_output_dir_name = "arcgis-imagery-FUA"
boundary_dir_name = "boundary"
fua_path = "../GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0/GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.gpkg"
name_col = "eFUA_name"
country_col = "Cntry_name"
min_tile_overlap_ratio = 0.75

tile_url_template = (
    "https://services.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{zoom}/{y_tile}/{x_tile}"
)


def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int(
        (1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n
    )
    return ytile, xtile


def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg


def tile_bbox_from_xyz(x_tile, y_tile, zoom):
    north, west = num2deg(x_tile, y_tile, zoom)
    south, east = num2deg(x_tile + 1, y_tile + 1, zoom)
    return {
        "west": west,
        "south": south,
        "east": east,
        "north": north,
    }


def resolve_city_fua(city_name, country_name, fua):
    city_fua = fua[fua[name_col].astype(str).str.contains(city_name, case=False, na=False)].copy()
    if city_fua.empty:
        return None

    if country_name and country_col in city_fua.columns:
        country_matches = city_fua[
            city_fua[country_col].astype(str).str.lower() == str(country_name).lower()
        ].copy()
        if not country_matches.empty:
            city_fua = country_matches

    return city_fua.iloc[[0]].copy()


def build_expanded_city_geometry(city_name, city_fua, buffer_meters=1000):
    city_geom = city_fua.copy()
    if not city_geom.geometry.is_valid.all():
        try:
            city_geom["geometry"] = city_geom.geometry.make_valid()
        except Exception:
            pass

    if not city_geom.geometry.is_valid.all():
        city_geom["geometry"] = city_geom.buffer(0)

    if city_geom.geometry.is_empty.any() or not city_geom.geometry.is_valid.all():
        raise ValueError(
            f"Invalid FUA geometry after repair for {city_name}: "
            f"is_empty={city_geom.geometry.is_empty.tolist()}, "
            f"is_valid={city_geom.geometry.is_valid.tolist()}"
        )

    merged_geom = city_geom.geometry.union_all()
    merged_gdf = gpd.GeoDataFrame(
        {"city_query": [city_name]},
        geometry=[merged_geom],
        crs=city_geom.crs,
    )
    if not merged_gdf.geometry.is_valid.all():
        try:
            merged_gdf["geometry"] = merged_gdf.geometry.make_valid()
        except Exception:
            pass
    if not merged_gdf.geometry.is_valid.all():
        merged_gdf["geometry"] = merged_gdf.buffer(0)
    if merged_gdf.geometry.is_empty.any() or not merged_gdf.geometry.is_valid.all():
        raise ValueError(
            f"Invalid merged FUA geometry for {city_name}: "
            f"is_empty={merged_gdf.geometry.is_empty.tolist()}, "
            f"is_valid={merged_gdf.geometry.is_valid.tolist()}"
        )

    expanded_geom = merged_gdf.geometry.buffer(buffer_meters)

    expanded_geom_gdf = gpd.GeoDataFrame(
        {"city_query": [city_name]},
        geometry=expanded_geom,
        crs=merged_gdf.crs,
    )

    if not expanded_geom_gdf.geometry.is_valid.all():
        try:
            expanded_geom_gdf["geometry"] = expanded_geom_gdf.geometry.make_valid()
        except Exception:
            pass

    if not expanded_geom_gdf.geometry.is_valid.all():
        expanded_geom_gdf["geometry"] = expanded_geom_gdf.buffer(0)

    if not expanded_geom_gdf.geometry.is_valid.all():
        expanded_geom_gdf["geometry"] = expanded_geom_gdf.geometry.convex_hull

    if expanded_geom_gdf.geometry.is_empty.any() or not expanded_geom_gdf.geometry.is_valid.all():
        raise ValueError(
            f"Invalid expanded FUA geometry for {city_name}: "
            f"is_empty={expanded_geom_gdf.geometry.is_empty.tolist()}, "
            f"is_valid={expanded_geom_gdf.geometry.is_valid.tolist()}"
        )

    bbox_gdf = gpd.GeoDataFrame(
        {
            "city_query": [city_name, city_name],
            "bbox_type": ["original_fua_bounds", "expanded_1km_bounds"],
        },
        geometry=[city_geom.geometry.iloc[0].envelope, expanded_geom_gdf.geometry.iloc[0].envelope],
        crs=city_geom.crs,
    ).to_crs("EPSG:4326")
    bbox_gdf["min_lon"] = bbox_gdf.geometry.bounds["minx"]
    bbox_gdf["min_lat"] = bbox_gdf.geometry.bounds["miny"]
    bbox_gdf["max_lon"] = bbox_gdf.geometry.bounds["maxx"]
    bbox_gdf["max_lat"] = bbox_gdf.geometry.bounds["maxy"]

    expanded_geom_4326 = expanded_geom_gdf.to_crs("EPSG:4326")

    return expanded_geom_4326, bbox_gdf


def enumerate_candidate_tiles(expanded_geom_4326, zoom):
    bounds = expanded_geom_4326.total_bounds
    west, south, east, north = bounds[0], bounds[1], bounds[2], bounds[3]
    y_ul, x_ul = deg2num(north, west, zoom)
    y_lr, x_lr = deg2num(south, east, zoom)

    x_min, x_max = sorted([x_ul, x_lr])
    y_min, y_max = sorted([y_ul, y_lr])

    records = []
    expanded_polygon = expanded_geom_4326.geometry.iloc[0]
    for x_tile in range(x_min, x_max + 1):
        for y_tile in range(y_min, y_max + 1):
            tile_bbox = tile_bbox_from_xyz(x_tile, y_tile, zoom)
            tile_polygon = box(
                tile_bbox["west"],
                tile_bbox["south"],
                tile_bbox["east"],
                tile_bbox["north"],
            )
            if not tile_polygon.intersects(expanded_polygon):
                continue

            intersection_area = tile_polygon.intersection(expanded_polygon).area
            tile_area = tile_polygon.area
            overlap_ratio_tile = intersection_area / tile_area if tile_area > 0 else 0.0
            if overlap_ratio_tile < min_tile_overlap_ratio:
                continue

            if any(
                pd.isna(value)
                for value in [
                    x_tile,
                    y_tile,
                    tile_bbox["west"],
                    tile_bbox["south"],
                    tile_bbox["east"],
                    tile_bbox["north"],
                ]
            ):
                continue

            records.append(
                {
                    "tile_x": x_tile,
                    "tile_y": y_tile,
                    "tile_min_lon": tile_bbox["west"],
                    "tile_min_lat": tile_bbox["south"],
                    "tile_max_lon": tile_bbox["east"],
                    "tile_max_lat": tile_bbox["north"],
                    "overlap_ratio_tile": overlap_ratio_tile,
                    "tile_polygon": tile_polygon,
                }
            )

    return records


def download_tile(session, zoom, y_tile, x_tile, output_png):
    tile_url = tile_url_template.format(zoom=zoom, y_tile=y_tile, x_tile=x_tile)
    response = session.get(tile_url, timeout=60)
    response.raise_for_status()
    output_png.write_bytes(response.content)
    return tile_url


def download_tile_task(zoom, tile_record, city_output_dir):
    session = requests.Session()
    x_tile = int(tile_record["tile_x"])
    y_tile = int(tile_record["tile_y"])
    output_png = city_output_dir / f"{y_tile}_{x_tile}.png"

    if output_png.exists():
        tile_url = tile_url_template.format(zoom=zoom, y_tile=y_tile, x_tile=x_tile)
    else:
        tile_url = download_tile(session, zoom, y_tile, x_tile, output_png)

    return {
        "tile_x": x_tile,
        "tile_y": y_tile,
        "tile_url": tile_url,
        "png_path": str(output_png),
    }


def main():
    parser = argparse.ArgumentParser(description="Download ArcGIS imagery tiles covering expanded FUA polygons")
    parser.add_argument("--city", type=str, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--zoom", type=int, default=14)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    fua = gpd.read_file(fua_path)
    print(f"Using city list: {city_list_csv}")
    print(f"Loaded {len(city_names)} cities")

    selected_city_names = city_names
    if args.city:
        selected_city_names = [city_name for city_name in city_names if str(city_name).lower() == args.city.lower()]
        if not selected_city_names:
            print(f"City not found: {args.city}")
            return

    end_index = args.end if args.end is not None else len(selected_city_names)
    end_index = min(end_index, len(selected_city_names))
    run_city_names = selected_city_names[args.start:end_index]

    for city_name in tqdm.tqdm(run_city_names, desc="cities"):
        try:
            country_name = None
            country_matches = csv_fua[csv_fua["city"].astype(str).str.lower() == str(city_name).lower()]
            if not country_matches.empty and "country" in country_matches.columns:
                country_name = country_matches.iloc[0]["country"]

            city_fua = resolve_city_fua(city_name, country_name, fua)
            if city_fua is None:
                print(f"{city_name}: no FUA found, skip")
                continue
            print(
                f"{city_name}: matched FUA -> "
                f"{name_col}={city_fua.iloc[0][name_col]}, "
                f"{country_col}={city_fua.iloc[0][country_col]}, "
                f"eFUA_ID={city_fua.iloc[0]['eFUA_ID']}, "
                f"is_valid={city_fua.geometry.is_valid.tolist()}"
            )

            expanded_geom_4326, bbox_gdf = build_expanded_city_geometry(city_name, city_fua, buffer_meters=1000)

            city_output_dir = output_root / city_name / tile_output_dir_name
            city_output_dir.mkdir(parents=True, exist_ok=True)
            boundary_dir = output_root / city_name / boundary_dir_name
            boundary_dir.mkdir(parents=True, exist_ok=True)

            bbox_gdf[["city_query", "bbox_type", "min_lon", "min_lat", "max_lon", "max_lat"]].to_csv(
                boundary_dir / f"{city_name}_fua_bounds_expanded_1km.csv",
                index=False,
            )

            output_csv = city_output_dir / f"{city_name}_arcgis_tiles_z{args.zoom}.csv"
            output_parquet = city_output_dir / f"{city_name}_arcgis_tiles_z{args.zoom}.parquet"
            output_gpkg = city_output_dir / f"{city_name}_arcgis_tiles_z{args.zoom}.gpkg"

            if output_csv.exists() and output_parquet.exists() and output_gpkg.exists():
                continue

            tile_records = enumerate_candidate_tiles(expanded_geom_4326, args.zoom)
            if not tile_records:
                print(f"{city_name}: no tiles intersect expanded FUA")
                continue

            result_rows = []
            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
                future_to_tile = {
                    executor.submit(download_tile_task, args.zoom, tile_record, city_output_dir): tile_record
                    for tile_record in tile_records
                }
                for future in tqdm.tqdm(
                    as_completed(future_to_tile),
                    total=len(future_to_tile),
                    desc=f"{city_name} tiles",
                    leave=True,
                ):
                    tile_record = future_to_tile[future]
                    result = future.result()
                    if any(
                        pd.isna(value)
                        for value in [
                            tile_record["tile_x"],
                            tile_record["tile_y"],
                            tile_record["tile_min_lon"],
                            tile_record["tile_min_lat"],
                            tile_record["tile_max_lon"],
                            tile_record["tile_max_lat"],
                        ]
                    ):
                        continue
                    result_rows.append(
                        {
                            "city_query": city_name,
                            "zoom": args.zoom,
                            "image_name": Path(result["png_path"]).name,
                            "tile_x": result["tile_x"],
                            "tile_y": result["tile_y"],
                            "min_lon": tile_record["tile_min_lon"],
                            "min_lat": tile_record["tile_min_lat"],
                            "max_lon": tile_record["tile_max_lon"],
                            "max_lat": tile_record["tile_max_lat"],
                            "overlap_ratio_tile": tile_record["overlap_ratio_tile"],
                            "tile_url": result["tile_url"],
                            "png_path": result["png_path"],
                        }
                    )

            result_df = pd.DataFrame(result_rows).sort_values(by=["tile_y", "tile_x"]).reset_index(drop=True)
            result_df.to_csv(output_csv, index=False)
            result_df.to_parquet(output_parquet, index=False)
            result_gdf = gpd.GeoDataFrame(
                result_df.copy(),
                geometry=result_df.apply(
                    lambda row: box(row["min_lon"], row["min_lat"], row["max_lon"], row["max_lat"]),
                    axis=1,
                ),
                crs="EPSG:4326",
            )
            result_gdf.to_file(output_gpkg, driver="GPKG")
            print(f"{city_name}: saved {len(result_df)} tiles to {city_output_dir}")
        except Exception as exc:
            print(f"{city_name}: failed to download FUA tiles: {exc}")
            continue


if __name__ == "__main__":
    main()
