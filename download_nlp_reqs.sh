# !/bin/sh
python - <<'END_SCRIPT'
import nltk
nltk.download('stopwords')
nltk.download('punkt')
END_SCRIPT
python -m spacy download fr_core_news_sm