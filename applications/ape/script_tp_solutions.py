# Script du TP
# Ci-dessous une balise pour voir les résultats en interactif
# %%
import os

import pandas as pd

# Import NAF classification
naf = pd.read_excel("data/naf.xls", skiprows=2)

# Import training data
train = pd.read_parquet("data/data.parquet")

# Merge classification info
naf["Code"] = naf["Code"].str.replace(".", "")
train = train.merge(naf, left_on="nace", right_on="Code")
train.head(5)

# %%
# ------------------------------------------
# Env --------------------------------------

# Charge la bibliothèque spaCy pour le traitement du langage naturel (NLP)
import spacy

# Télécharge le modèle français pour spaCy (nécessaire pour l'analyse morphosyntaxique en français)
os.system("python -m spacy download fr_core_news_sm")

# Charge la bibliothèque NLTK pour le traitement du texte
import nltk

# Télécharge le tokeniseur "punkt_tab" de NLTK (pour la segmentation en phrases/tokens)
nltk.download("punkt_tab")

# Télécharge la liste des stopwords français/anglais de NLTK (mots vides comme "le", "la", "the", etc.)
nltk.download("stopwords")

# %%
# -------------------------------------------
# Premiere partie : data cleaning -----------
# -------------------------------------------


def filter_train_data(train_data, sequence):
    sequence_capitalized = sequence.upper()
    mask = train_data["text"].str.contains(sequence_capitalized)
    nb_occurrence = mask.astype(int).sum()
    print(f"Nombre d'occurrences de la séquence '{sequence}': {nb_occurrence}")
    return train_data.loc[mask]


pd.set_option("display.max_colwidth", None)
filter_train_data(train, "data science").head(5)
filter_train_data(train, "boulanger").head(5)

# %%
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def graph_wordcloud(train_data, text_var="text", naf=None):
    if naf is not None:
        train_data = train_data.loc[train_data["nace"] == naf]

    txt = train_data[text_var]
    all_text = " ".join([text for text in txt])
    wordcloud = WordCloud(
        width=800,
        height=500,
        random_state=21,
        max_words=2000,
        background_color="white",
        colormap="Set2",
    ).generate(all_text)
    return wordcloud


wordcloud_corpus = graph_wordcloud(train.sample(10000))
plt.imshow(wordcloud_corpus, interpolation="bilinear")

wordcloud_corpus = graph_wordcloud(train, naf="1071C")
plt.imshow(wordcloud_corpus, interpolation="bilinear")

wordcloud_corpus = graph_wordcloud(train, naf="4942Z")
plt.imshow(wordcloud_corpus, interpolation="bilinear")

# %%
from nltk.tokenize import word_tokenize

nlp = spacy.load("fr_core_news_sm")
stop_words = nlp.Defaults.stop_words
stop_words = set(stop_words)


# Function to remove stopwords
def remove_stopwords(text):
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return " ".join(filtered_text)


def remove_single_letters(text):
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if len(word) > 1]
    return " ".join(filtered_text)


# Apply the function to the 'text' column
train["text_clean"] = train["text"].apply(remove_stopwords).apply(remove_single_letters)

wordcloud_corpus_cleaned = graph_wordcloud(train.sample(10000), "text_clean")
plt.imshow(wordcloud_corpus_cleaned, interpolation="bilinear")

# %%
# -------------------------------------------
# Partie TTC --------------------------------
# -------------------------------------------

from processor import Preprocessor

preprocessor = Preprocessor()

# Preprocess data before training and testing
TEXT_FEATURE = "text"
Y = "nace"

df = train.copy()

df = preprocessor.clean_text(df, TEXT_FEATURE).drop("text_clean", axis="columns")
df.head(2)

# %%

df = df.dropna(subset=[Y, TEXT_FEATURE])
X = df[TEXT_FEATURE].values
y = df[Y].values

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_encoded = le.fit_transform(y)  # Convertit ["cat", "dog"] → [0, 1]

# Première division : train (80 %) + test (20%)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X.to_numpy(),
    y_encoded,
    test_size=0.2,
    random_state=0,
    shuffle=True,
)
# Deuxième division pour aboutir à : train (60 % = 80% * 75%) + val =  (60 % = 80% * 25%) + test (20%)

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.25,
    random_state=0,
    shuffle=True,
)


from torchTextClassifiers.tokenizers.ngram import NGramTokenizer

tokenizer = NGramTokenizer(
    min_count=2,  # On considère un mot s'il est trouvé au moins 2 fois dans le corpus
    min_n=2,
    max_n=4,  # On fait des 2grams, 3grams et 4grams de caractères
    len_word_ngrams=2,  # On fait des 2grams de mots
    num_tokens=10000,  # Nombre max de tokens considérés dans le vocable
    training_text=X,  # Jeu d'entraînement du tokenizer
)

# %%
# Set model configs ---------------

import numpy as np
from torchTextClassifiers import ModelConfig

# Embedding dimension
embedding_dim = 64

# Count number of unique labels
unique_values, counts = np.unique(y, return_counts=True)
num_unique = len(unique_values)

model_config = ModelConfig(embedding_dim=embedding_dim, num_classes=num_unique)

# %%
# Instanciate a ttc model (nammed "classifier") ---------------

from torchTextClassifiers import torchTextClassifiers

classifier = torchTextClassifiers(tokenizer=tokenizer, model_config=model_config)

# %%
# Set the training configs ---------------

from torchTextClassifiers import TrainingConfig

# Training params (torch style)
training_config = TrainingConfig(
    num_epochs=30,
    batch_size=8,
    lr=1e-3,
    patience_early_stopping=7,
    num_workers=0,
    trainer_params={"deterministic": True},
)

# Training (too long) !

# classifier.train(
#     X_train,
#     y_train,
#     training_config,
#     X_val,
#     y_val,
#     verbose=True
# )

# %%
# Download a pre-trained instead to make it faster :

# Download the model
base_url = "https://minio.lab.sspcloud.fr/projet-formation/nouvelles-sources/model_ape"
files = ["metadata.pkl", "model_checkpoint.ckpt", "tokenizer.pkl"]

for file in files:
    os.system(f"curl {base_url}/{file} --output data/model_ape/{file} --create-dirs")

# %%
# Load it
classifier = torchTextClassifiers.load("model_ape")

# %%
# Sample testset

import numpy as np

n = X_test.shape[0]
sample_size = min(1000, n)

rng = np.random.default_rng(seed=42)
idx = rng.choice(n, size=sample_size, replace=False)

X_sample = X_test[idx]
y_sample = y_test[idx]

# %%
# Inference on sampled testset

result = classifier.predict(X_sample)
predictions = result["prediction"].squeeze().numpy()
accuracy = (predictions == y_sample).mean()
print(f"Test accuracy: {accuracy:.3f}")

# %%
# Try some predictions yourself
import numpy as np

searched_professions = np.array(
    [
        "Conseil datascience",
        "Concésion dans l'automobile",
        "Concession automobile",
        "peintre",
    ]
)
preds = classifier.predict(searched_professions)

for profession, prediction, confidence in zip(
    searched_professions, preds["prediction"], preds["confidence"]
):
    # Affichage de la profession recherchée
    print(f"\n🔍 Profession recherchée : {profession}")

    # Conversion sécurisée de la prédiction (compatible PyTorch/NumPy)
    pred_code = prediction.cpu().numpy()
    pred_labels = le.inverse_transform(pred_code)

    # Récupération des libellés NAF correspondants
    matching_labels = naf.loc[naf["Code"].isin(pred_labels), "Libellé"].tolist()

    # Affichage des résultats
    print(f"🏷️ Profession(s) trouvée(s) : {', '.join(matching_labels)}")
    print(
        f"📊 Confiance : {confidence.item() if hasattr(confidence, 'item') else confidence:.2f}"
    )


# %%
