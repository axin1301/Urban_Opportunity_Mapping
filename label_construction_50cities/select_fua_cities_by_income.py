import argparse
import re
import unicodedata
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests


DEFAULT_INPUT_CANDIDATES = [
    Path("cities_with_FUA_ge_1M.csv"),
    Path("../cities_with_FUA_ge_1M.csv"),
]
FUA_PATH = Path("../GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0/GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.gpkg")
FUA_NAME_COL = "eFUA_name"
OUTPUT_ENRICHED = Path("cities_with_FUA_ge_1M_income_groups.csv")
OUTPUT_SELECTED = Path("selected_50_fua_cities.csv")


COUNTRY_ALIASES = {
    "united states": "united states",
    "usa": "united states",
    "uk": "united kingdom",
    "south korea": "korea, rep.",
    "egypt": "egypt, arab rep.",
    "turkey": "türkiye",
    "slovakia": "slovak republic",
    "czech republic": "czechia",
}


def normalize_text(value):
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = re.sub(r"[\s'`´\-–—\.,/\\\(\)\[\]{}]+", "", text)
    return text


def resolve_input_csv():
    for path in DEFAULT_INPUT_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find cities_with_FUA_ge_1M.csv in current or parent directory."
    )


def load_world_bank_income_table():
    url = "https://api.worldbank.org/v2/country?format=json&per_page=400"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    payload = response.json()
    records = payload[1]

    rows = []
    for item in records:
        rows.append(
            {
                "wb_country_name": item.get("name"),
                "wb_country_id": item.get("id"),
                "income_level": item.get("incomeLevel", {}).get("value"),
                "region": item.get("region", {}).get("value"),
                "lending_type": item.get("lendingType", {}).get("value"),
            }
        )

    wb_df = pd.DataFrame(rows)
    wb_df["country_norm"] = wb_df["wb_country_name"].apply(normalize_text)
    return wb_df


def attach_income_groups(city_df, wb_df):
    city_df = city_df.copy()
    city_df["country_for_match"] = city_df["country"].astype(str).str.strip()
    city_df["country_for_match"] = city_df["country_for_match"].str.lower().map(
        lambda x: COUNTRY_ALIASES.get(x, x)
    )
    city_df["country_norm"] = city_df["country_for_match"].apply(normalize_text)

    merged = city_df.merge(
        wb_df[["country_norm", "wb_country_name", "wb_country_id", "income_level", "region", "lending_type"]],
        on="country_norm",
        how="left",
    )
    return merged.drop(columns=["country_norm"])


def attach_hemisphere(city_df):
    city_df = city_df.copy()
    fua = gpd.read_file(FUA_PATH)
    fua = fua[[FUA_NAME_COL, "geometry"]].copy()
    fua["city_norm"] = fua[FUA_NAME_COL].apply(normalize_text)

    city_df["city_norm"] = city_df["city"].apply(normalize_text)
    merged = city_df.merge(fua[["city_norm", "geometry"]], on="city_norm", how="left")
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=fua.crs if not fua.empty else None)

    if merged.crs is not None:
        merged = merged.to_crs("EPSG:4326")
        merged["centroid_lat"] = merged.geometry.centroid.y
        merged["centroid_lon"] = merged.geometry.centroid.x
        merged["hemisphere"] = merged["centroid_lat"].apply(
            lambda lat: "north" if pd.notna(lat) and lat >= 0 else "south"
        )
    else:
        merged["centroid_lat"] = None
        merged["centroid_lon"] = None
        merged["hemisphere"] = None

    return pd.DataFrame(merged.drop(columns=["geometry", "city_norm"]))


def sample_cities(city_df, total_target=50, random_state=42):
    df = city_df.copy()
    df = df.dropna(subset=["income_level", "hemisphere"])

    desired_counts = {
        ("north", "High income"): 15,
        ("north", "Upper middle income"): 12,
        ("north", "Lower middle income"): 8,
        ("north", "Low income"): 3,
        ("south", "High income"): 4,
        ("south", "Upper middle income"): 4,
        ("south", "Lower middle income"): 3,
        ("south", "Low income"): 1,
    }

    selected_parts = []
    for (hemisphere, income_level), target_count in desired_counts.items():
        group = df[(df["hemisphere"] == hemisphere) & (df["income_level"] == income_level)].copy()
        if group.empty:
            continue
        sample_n = min(target_count, len(group))
        selected_parts.append(group.sample(n=sample_n, random_state=random_state))

    selected_df = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame(columns=df.columns)
    selected_df = selected_df.drop_duplicates(subset=["city"], keep="first")

    if len(selected_df) < total_target:
        remaining = df[~df["city"].isin(selected_df["city"])].copy()
        fill_n = min(total_target - len(selected_df), len(remaining))
        if fill_n > 0:
            selected_df = pd.concat(
                [selected_df, remaining.sample(n=fill_n, random_state=random_state)],
                ignore_index=True,
            )

    return selected_df.head(total_target).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Enrich FUA city list with World Bank income groups and select 50 cities")
    parser.add_argument("--target", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_csv = resolve_input_csv()
    city_df = pd.read_csv(input_csv)
    wb_df = load_world_bank_income_table()
    enriched_df = attach_income_groups(city_df, wb_df)
    enriched_df = attach_hemisphere(enriched_df)

    enriched_df.to_csv(OUTPUT_ENRICHED, index=False)
    selected_df = sample_cities(enriched_df, total_target=args.target, random_state=args.seed)
    selected_df.to_csv(OUTPUT_SELECTED, index=False)

    print(f"Input cities: {len(city_df)}")
    print(f"Income matched: {int(enriched_df['income_level'].notna().sum())}")
    print(f"Hemisphere matched: {int(enriched_df['hemisphere'].notna().sum())}")
    print(f"Saved enriched file: {OUTPUT_ENRICHED}")
    print(f"Saved selected sample: {OUTPUT_SELECTED}")
    print(selected_df.groupby(['hemisphere', 'income_level']).size())


if __name__ == "__main__":
    main()
