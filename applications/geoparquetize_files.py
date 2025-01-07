import sys
import os
import subprocess
import requests

import pandas as pd
import geopandas as gpd

from loguru import logger

# Configure Loguru to log messages to a file and the console
logger.add("script.log", rotation="1 MB")  # Log rotation after 1 MB
logger.add(
    sys.stderr, format="{time} {level} {message}", level="INFO"
)  # Console logging


# DVF --------------------------------------------------------------------------

# DOWNLOAD =====================================

url = "https://files.data.gouv.fr/geo-dvf/latest/csv/2022/full.csv.gz"
file_name = "dvf.csv.gz"

logger.info("Starting the download process for DVF data.")

# Check if the file already exists
if not os.path.exists(file_name):
    logger.info(f"File '{file_name}' does not exist. Initiating download from {url}.")
    try:
        response = requests.get(url, timeout=60)  # Added timeout for network issues
        response.raise_for_status()  # Raises HTTPError for bad responses
        with open(file_name, "wb") as f:
            f.write(response.content)
        logger.success("Download completed successfully.")
    except Exception as e:
        logger.error(f"Errow while downloading: {e}")
        raise  # Re-raise the exception after logging
else:
    logger.info(f"The file '{file_name}' already exists. Skipping download.")


# READ, CLEAN ===========================

logger.info("Reading the downloaded CSV file into a pandas DataFrame.")
try:
    dvf = pd.read_csv(
        "dvf.csv.gz", dtype={"code_commune": "str", "code_departement": "str"}
    )
    logger.success("CSV file read successfully.")
except Exception as e:
    logger.error(f"Error reading CSV file: {e}")
    raise  # Re-raise the exception after logging

logger.info("Converting DataFrame to GeoDataFrame.")
try:
    gdf = gpd.GeoDataFrame(
        dvf, geometry=gpd.points_from_xy(x=dvf.longitude, y=dvf.latitude)
    )
    gdf.set_crs(epsg=4326, inplace=True)
    logger.success("GeoDataFrame created successfully.")
except Exception as e:
    logger.error(f"Error converting to GeoDataFrame: {e}")
    raise

logger.info("Converting object columns to string type.")
try:
    object_cols = gdf.select_dtypes(["object"]).columns
    gdf[object_cols] = gdf[object_cols].astype("string")
    logger.success("Object columns converted to string type.")
except Exception as e:
    logger.error(f"Error converting object columns: {e}")
    raise

logger.info("Shrink fields to simplify re-using.")
try:
    gdf = gdf[gdf['type_local'].isin(["Appartement", "Maison"])]
    gdf = gdf[gdf['valeur_fonciere'].notna()]
    gdf = gdf[(gdf['valeur_fonciere'] > 10000) & (gdf['valeur_fonciere'] < 10000000)]
    gdf = gdf[gdf['nature_mutation'] == "Vente"]
    gdf = gdf[gdf['longitude'].notna() | gdf['latitude'].notna()]
    gdf = gdf[gdf['nombre_pieces_principales'] > 0]
    gdf = gdf.drop_duplicates(subset='id_mutation')
except Exception as e:
    logger.error(f"Error simplifying dvf data: {e}")
    raise

logger.info("Reprojecting GeoDataFrame to EPSG:2154.")
gdf = gdf.to_crs(2154)


# FILOSOFI -----------------------------------


logger.info("Copying 'carreaux_200m_met.gpkg' from S3 to local storage.")

try:
    subprocess.check_call(
        "mc cp s3/projet-formation/r-lissage-spatial/carreaux_200m_met.gpkg carreaux.gpkg",
        shell=True,
    )
    logger.success("File copied successfully.")
except subprocess.CalledProcessError as e:
    logger.error(f"Error during file copy: {e}")
    raise

logger.info("Reading 'carreaux.gpkg' into a GeoDataFrame.")
try:
    carreaux = gpd.read_file("carreaux.gpkg")
    object_cols = carreaux.select_dtypes(["object"]).columns
    carreaux[object_cols] = carreaux[object_cols].astype("string")
    carreaux = carreaux.to_crs(2154)
    logger.success("Carreaux GeoDataFrame processed successfully.")
except Exception as e:
    logger.error(f"Error processing 'carreaux.gpkg': {e}")
    raise


# WRITE ------------------------

logger.info("Writing GeoDataFrames to Parquet files.")
try:
    gdf.to_parquet("dvf_geoparquet.parquet", compression="gzip")
    carreaux.to_parquet("carreaux_geoparquet.parquet", compression="gzip")
    logger.success("GeoDataFrames written to Parquet files successfully.")
except Exception as e:
    logger.error(f"Error writing Parquet files: {e}")
    raise


# COPY TO S3 --------------------------------------

logger.info("Writing to S3")

subprocess.call(
    "mc cp carreaux_geoparquet.parquet s3/projet-formation/nouvelles-sources/data/geoparquet/carreaux.parquet",
    shell=True
)
subprocess.call(
    "mc cp dvf_geoparquet.parquet s3/projet-formation/nouvelles-sources/data/geoparquet/dvf.parquet",
    shell=True
)
