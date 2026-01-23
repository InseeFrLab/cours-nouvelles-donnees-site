#!/bin/bash

mkdir ape
cd ape

mkdir data
echo "data/" >> .gitignore

curl https://minio.lab.sspcloud.fr/projet-formation/diffusion/mlops/data/firm_activity_data.parquet --output data/data.parquet --retry 3 --retry-all-errors --max-time 5
curl https://minio.lab.sspcloud.fr/projet-formation/nouvelles-sources/data/naf2008_liste_n5.xls --output data/naf.xls --retry 3 --retry-all-errors --max-time 5
curl -O https://raw.githubusercontent.com/InseeFrLab/cours-nouvelles-donnees-site/main/applications/processor.py --retry 3 --retry-all-errors --max-time 5

pip install wordcloud
pip install xlrd
pip install spacy
pip install nltk
pip install unidecode
pip install pytorch_lightning
pip install torchTextClassifiers[huggingface]
pip install scikit-learn

echo -e "# Script du TP\nCi-dessous une balise pour voir les résultats en intéractif\n# %%" > script_tp.py
