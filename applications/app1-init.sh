#!/bin/bash

mkdir -p appli1
cd appli1

echo "data/" >> .gitignore

curl https://minio.lab.sspcloud.fr/projet-formation/nouvelles-sources/data/geoparquet/dvf.parquet --output data/dvf.parquet
curl https://minio.lab.sspcloud.fr/projet-formation/nouvelles-sources/data/geoparquet/carreaux.parquet --output data/carreaux.parquet
curl https://minio.lab.sspcloud.fr/projet-formation/nouvelles-sources/data/triangle.geojson --output data/triangle.geojson
curl https://minio.lab.sspcloud.fr/projet-formation/nouvelles-sources/data/malakoff.geojson --output data/malakoff.geojson
curl https://minio.lab.sspcloud.fr/projet-formation/nouvelles-sources/data/montrouge.geojson --output data/montrouge.geojson
