#!/bin/bash

mkdir appli1
cd appli1

mkdir data

# Download data from S3
curl https://minio.lab.sspcloud.fr/projet-formation/nouvelles-sources/data/geoparquet/dvf.parquet --output data/dvf.parquet --retry 3 --retry-all-errors --max-time 5
curl https://minio.lab.sspcloud.fr/projet-formation/nouvelles-sources/data/geoparquet/carreaux.parquet --output data/carreaux.parquet --retry 3 --retry-all-errors --max-time 5
curl https://minio.lab.sspcloud.fr/projet-formation/nouvelles-sources/data/triangle.geojson --output data/triangle.geojson --retry 3 --retry-all-errors --max-time 5
curl https://minio.lab.sspcloud.fr/projet-formation/nouvelles-sources/data/malakoff.geojson --output data/malakoff.geojson --retry 3 --retry-all-errors --max-time 5
curl https://minio.lab.sspcloud.fr/projet-formation/nouvelles-sources/data/montrouge.geojson --output data/montrouge.geojson --retry 3 --retry-all-errors --max-time 5
