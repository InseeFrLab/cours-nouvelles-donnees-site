#!/bin/bash

mkdir -p appli1
cd appli1
echo "data/" >> .gitignore
git init
git branch -m main

mc cp s3/projet-formation/nouvelles-sources/data/geoparquet/dvf.parquet data/dvf.parquet
mc cp s3/projet-formation/nouvelles-sources/data/geoparquet/carreaux.parquet data/carreaux.parquet
mc cp s3/projet-formation/nouvelles-sources/data/triangle.geojson data/triangle.geojson
mc cp s3/projet-formation/nouvelles-sources/data/malakoff.geojson data/malakoff.geojson
mc cp s3/projet-formation/nouvelles-sources/data/montrouge.geojson data/montrouge.geojson
