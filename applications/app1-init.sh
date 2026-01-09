#!/bin/bash

mkdir appli1
cd appli1

# Initializing git repo and renaming branch to main
git init
git branch -m main

mkdir data
echo "data/" >> .gitignore

# Download data from S3
curl https://minio.lab.sspcloud.fr/projet-formation/nouvelles-sources/data/geoparquet/dvf.parquet --output data/dvf.parquet
curl https://minio.lab.sspcloud.fr/projet-formation/nouvelles-sources/data/geoparquet/carreaux.parquet --output data/carreaux.parquet
curl https://minio.lab.sspcloud.fr/projet-formation/nouvelles-sources/data/triangle.geojson --output data/triangle.geojson
curl https://minio.lab.sspcloud.fr/projet-formation/nouvelles-sources/data/malakoff.geojson --output data/malakoff.geojson
curl https://minio.lab.sspcloud.fr/projet-formation/nouvelles-sources/data/montrouge.geojson --output data/montrouge.geojson
