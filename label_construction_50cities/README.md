# UrbanOpp benchmark construction (50-City Labels)

This folder contains the scripts used to build satellite tile (patch) labels for the 50-city urban opportunity dataset (UrbanOpp).

## Scripts
- `select_fua_cities_by_income.py`: selects FUA cities and attaches World Bank income/region metadata.
- `download_arcgis_tiles_for_fua.py`: downloads ArcGIS satellite image tiles inside each FUA boundary and saves patch geometries.
- `build_base_station_labels_for_arcgis_tiles.py`: matches cellular base-station records to image patches and builds only Digital Infrastructure Intensity (DII).
- `build_patch_raster_covariates.py`: extracts population and nighttime-light raster statistics for each patch.
- `build_patch_poi_from_duckdb.py`: downloads Overture POIs for each city using DuckDB.
- `build_patch_poi_covariates.py`: computes patch-level POI summary features.


## Main inputs

- `selected_50_fua_cities_nonempty_labels.csv`: final selected city list.
- FUA boundary file, for example `../GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0/GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.gpkg`.
- Base-station CSV files with at least `LON` and `LAT` columns. `radio` is optional for the DII-only script.
- Population raster, such as GHSL or WorldPop.
- Nighttime-light raster, such as VIIRS.

## Typical workflow

```bash
# 1. Download satellite image patches
python download_arcgis_tiles_for_fua.py \
  --city-list selected_50_fua_cities_nonempty_labels.csv \
  --zoom 14 \
  --workers 8

# 2. Build cellular infrastructure labels
python build_base_station_labels_for_arcgis_tiles.py \
  --city-list-csv selected_50_fua_cities_nonempty_labels.csv \
  --cell-dir /path/to/base_station_csv_dir \
  --beta 1000

# 3. Extract population and nighttime-light covariates
python build_patch_raster_covariates.py \
  --city-list selected_50_fua_cities_nonempty_labels.csv \
  --population-raster /path/to/population.tif \
  --nightlight-raster /path/to/nightlight.tif

# 4. Download POIs
python build_patch_poi_from_duckdb.py \
  --city-list selected_50_fua_cities_nonempty_labels.csv
```

## Output structure

Most city-level outputs are written to:

```text
../output_files/{city}/
```

Important derived outputs include:

- `../output_files/{city}/arcgis-imagery-FUA/*.png`
- `../output_files/{city}/arcgis-imagery-FUA/{city}_arcgis_tiles_z14.gpkg`
- `../output_files/{city}/arcgis-imagery-FUA/{city}_base_station_labels.csv`
- `../output_files/{city}/{city}_patch_raster_covariates.csv`
- `../output_files/{city}/{city}_patch_pois.csv`


## Indicators

Main indicators include:

- `Population Intensity (POP)`
- `Nighttime Luminosity Intensity (NTL)`
- `Education opportunity (EDU)`
- `Healthcare opportunity (HEA)`
- `Food opportunity (FOOD)`
- `Social opportunity (SOC)`
- `Mobility opportunity (MOB)`
- `Digital Infrastructure Intensity (DII)`
