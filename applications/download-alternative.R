# Define base URL for downloads
base_url <- "https://minio.lab.sspcloud.fr/projet-formation/nouvelles-sources/data/"

# Download the files
download.file(url = paste0(base_url, "geoparquet/dvf.parquet"),
              destfile = "data/dvf.parquet")

download.file(url = paste0(base_url, "geoparquet/carreaux.parquet"),
              destfile = "data/carreaux.parquet")

download.file(url = paste0(base_url, "triangle.geojson"),
              destfile = "data/triangle.geojson")

download.file(url = paste0(base_url, "malakoff.geojson"),
              destfile = "data/malakoff.geojson")

download.file(url = paste0(base_url, "montrouge.geojson"),
              destfile = "data/montrouge.geojson")
