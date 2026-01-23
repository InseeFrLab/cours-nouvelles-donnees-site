#!/bin/bash

mkdir appli1
cd appli1

mkdir data
echo "data/" >> .gitignore

# Download data from S3
curl https://minio.lab.sspcloud.fr/projet-formation/nouvelles-sources/data/geoparquet/dvf.parquet --output data/dvf.parquet  --retry 5 --retry-all-errors --max-time 15
curl https://minio.lab.sspcloud.fr/projet-formation/nouvelles-sources/data/geoparquet/carreaux.parquet --output data/carreaux.parquet  --retry 5 --retry-all-errors --max-time 15
curl https://minio.lab.sspcloud.fr/projet-formation/nouvelles-sources/data/triangle.geojson --output data/triangle.geojson  --retry 5 --retry-all-errors --max-time 15
curl https://minio.lab.sspcloud.fr/projet-formation/nouvelles-sources/data/malakoff.geojson --output data/malakoff.geojson  --retry 5 --retry-all-errors --max-time 15
curl https://minio.lab.sspcloud.fr/projet-formation/nouvelles-sources/data/montrouge.geojson --output data/montrouge.geojson  --retry 5 --retry-all-errors --max-time 15
