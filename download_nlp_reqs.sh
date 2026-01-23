# !/bin/sh
python - <<'END_SCRIPT'
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
END_SCRIPT
python -m spacy download fr_core_news_sm

# App APE
mkdir applications/data

curl https://minio.lab.sspcloud.fr/projet-formation/diffusion/mlops/data/firm_activity_data.parquet --output applications/data/data.parquet  --retry 5 --retry-all-errors --max-time 15
curl https://minio.lab.sspcloud.fr/projet-formation/nouvelles-sources/data/naf2008_liste_n5.xls --output applications/data/naf.xls  --retry 5 --retry-all-errors --max-time 15

mkdir applications/model_ape

curl https://minio.lab.sspcloud.fr/projet-formation/nouvelles-sources/model_ape/metadata.pkl --output applications/model_ape/metadata.pkl  --retry 5 --retry-all-errors --max-time 15
curl https://minio.lab.sspcloud.fr/projet-formation/nouvelles-sources/model_ape/model_checkpoint.ckpt --output applications/model_ape/model_checkpoint.ckpt  --retry 5 --retry-all-errors --max-time 15
curl https://minio.lab.sspcloud.fr/projet-formation/nouvelles-sources/model_ape/tokenizer.pkl --output applications/model_ape/tokenizer.pkl --retry 5 --retry-all-errors --max-time 15
