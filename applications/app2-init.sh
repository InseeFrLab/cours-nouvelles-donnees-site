#!/bin/bash

mkdir ape
cd ape

mkdir data
echo "data/" >> .gitignore

curl ttps://minio.lab.sspcloud.fr/projet-formation/diffusion/mlops/data/firm_activity_data.parquet --output data/data.parquet
curl https://minio.lab.sspcloud.fr/projet-formation/nouvelles-sources/data/naf2008_liste_n5.xls --output data/naf.parquet

pip install wordcloud
pip install xlrd
pip install spacy
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pytorch_lightning
pip install torchTextClassifiers[huggingface]
