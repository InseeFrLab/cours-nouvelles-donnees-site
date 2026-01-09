
# pip install wordcloud
# pip install xlrd
# pip install spacy
# python -m spacy download fr_core_news_sm
# pip install torch --index-url https://download.pytorch.org/whl/cpu
# pip install pytorch_lightning
# pip install torchTextClassifiers
# pip install torchTextClassifiers[huggingface]

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

DATA_PATH = "https://minio.lab.sspcloud.fr/projet-formation/diffusion/mlops/data/firm_activity_data.parquet"
NAF_PATH = "https://minio.lab.sspcloud.fr/projet-formation/nouvelles-sources/data/naf2008_liste_n5.xls"
naf = pd.read_excel(NAF_PATH, skiprows = 2)
naf['Code'] = naf['Code'].str.replace(".","")
train = pd.read_parquet(DATA_PATH)
train = train.merge(naf, left_on = "nace", right_on = "Code")
train.head(5)


# import spacy
# import nltk
# from nltk.tokenize import word_tokenize

# nltk.download('punkt_tab')
# nltk.download('stopwords')

# nlp = spacy.load("fr_core_news_sm")

# # Function to remove stopwords
# def remove_stopwords(text):
#     word_tokens = word_tokenize(text)
#     filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
#     return ' '.join(filtered_text)

# def remove_single_letters(text):
#     word_tokens = word_tokenize(text)
#     filtered_text = [word for word in word_tokens if len(word) > 1]
#     return ' '.join(filtered_text)

# # Apply the function to the 'text' column
# # train['text_clean'] = (train['text']
# #     .apply(remove_stopwords)
# #     .apply(remove_single_letters)
# # )

# train_data=train.copy()

# def filter_train_data(train_data, sequence):
#     sequence_capitalized = sequence.upper() 
#     mask = train_data['text'].str.contains(sequence_capitalized)
#     nb_occurrence = mask.astype(int).sum()
#     print(
#         f"Nombre d'occurrences de la séquence '{sequence}': {nb_occurrence}"
#     )
#     return train_data.loc[mask]

# filter_train_data(train, "data science").head(5)
# filter_train_data(train, "boulanger").head(5)

# def graph_wordcloud(train_data, text_var="text", naf=None):
#     if naf is not None:
#         train_data = train_data.loc[train_data['nace'] == naf]

#     txt = train_data[text_var]
#     all_text = ' '.join([text for text in txt])
#     wordcloud = WordCloud(
#         width=800,
#         height=500,
#         random_state=21,
#         max_words=2000,
#         background_color="white",
#         colormap='Set2'
#     ).generate(all_text)
#     return wordcloud


# wordcloud_corpus = graph_wordcloud(train)
# plt.figure()
# plt.imshow(wordcloud_corpus, interpolation="bilinear")
# plt.axis("off")
# plt.show()


# wordcloud_corpus = graph_wordcloud(train, naf = "1071C")
# plt.figure()
# plt.imshow(wordcloud_corpus, interpolation="bilinear")
# plt.axis("off")
# plt.show()

# wordcloud_corpus = graph_wordcloud(train, naf = "4942Z")
# plt.figure()
# plt.imshow(wordcloud_corpus, interpolation="bilinear")
# plt.axis("off")
# plt.show()

# from nltk.tokenize import word_tokenize
# import spacy

# nlp = spacy.load("fr_core_news_sm")
# stop_words = nlp.Defaults.stop_words

# stop_words = set(stop_words)

# train['text_clean'] = (train['text']
#     .apply(remove_stopwords)
#     .apply(remove_single_letters)
# )



# - [`constants.py`](https://github.com/InseeFrLab/cours-nouvelles-donnees-site/blob/main/applications/constants.py)
# - [`processor.py`](https://github.com/InseeFrLab/cours-nouvelles-donnees-site/blob/main/applications/processor.py)
# - [`utils.py`](https://github.com/InseeFrLab/cours-nouvelles-donnees-site/blob/main/applications/utils.py)

# pip install unidecode
from processor import Preprocessor
preprocessor = Preprocessor()

# Preprocess data before training and testing
TEXT_FEATURE = "text"
Y = "nace"

df = preprocessor.clean_text(train, TEXT_FEATURE).drop('text_clean', axis = "columns")
df.head(2)


import pathlib

params = {
    "dim": 25,
    "label_prefix": "__label__"
}



import warnings
from sklearn.model_selection import train_test_split

df = df.dropna(subset = [Y, TEXT_FEATURE])

df = df.sample(500)


X = df[TEXT_FEATURE].values
y = df[Y].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # Convertit ["cat", "dog"] → [0, 1]

# Première division : train (80%) + test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=0,
    shuffle=True,
)

# Deuxième division : train (75% des données initiales) et validation (5% des données initiales)
# Cela donne 60% train, 20% validation, 20% test si tu veux une répartition classique.
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.25,  # 25% de X_train = 20% des données initiales (car 0.25 * 0.8 = 0.2)
    random_state=0,
    shuffle=True,
)

type(X_train)


# Step 3: Create and Train Tokenizer
tokenizer = WordPieceTokenizer(vocab_size=5000, output_dim=128)
tokenizer.train(X_train.tolist())


# Step 4: Configure Model for 3 Classes
unique_values, counts = np.unique(y, return_counts=True)
num_unique = len(unique_values)

model_config = ModelConfig(
    embedding_dim=64,
    num_classes=num_unique
)

# Step 5: Create Classifier
classifier = torchTextClassifiers(
    tokenizer=tokenizer,
    model_config=model_config
)

# Step 6: Train Model
training_config = TrainingConfig(
    num_epochs=30,
    batch_size=8,
    lr=1e-3,
    patience_early_stopping=7,
    num_workers=0,
    trainer_params={'deterministic': True}
)

classifier.train(
    X_train, y_train,
    training_config,
    X_val, y_val,
    verbose=True
)

# Step 7: Make Predictions
result = classifier.predict(X_test)
predictions = result["prediction"].squeeze().numpy()

# Step 8: Evaluate
accuracy = (predictions == y_test).mean()
print(f"Test accuracy: {accuracy:.3f}")



# Tuto meilame 



import os
import numpy as np
import torch
from pytorch_lightning import seed_everything
from torchTextClassifiers import ModelConfig, TrainingConfig, torchTextClassifiers
from torchTextClassifiers.tokenizers import WordPieceTokenizer


SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
seed_everything(SEED, workers=True)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)

X_train = np.array([
    # Negative (class 0)
    "This product is terrible and I hate it completely.",
    "Worst purchase ever. Total waste of money.",
    "Absolutely awful quality. Very disappointed.",
    "Poor service and terrible product quality.",
    "I regret buying this. Complete failure.",

    # Neutral (class 1)
    "The product is okay, nothing special though.",
    "It works but could be better designed.",
    "Average quality for the price point.",
    "Not bad but not great either.",
    "It's fine, meets basic expectations.",

    # Positive (class 2)
    "Excellent product! Highly recommended!",
    "Amazing quality and great customer service.",
    "Perfect! Exactly what I was looking for.",
    "Outstanding value and excellent performance.",
    "Love it! Will definitely buy again."
])

y_train = np.array([0, 0, 0, 0, 0,  # negative
                    1, 1, 1, 1, 1,  # neutral
                    2, 2, 2, 2, 2]) # positive

# Validation data
X_val = np.array([
    "Bad quality, not recommended.",
    "It's okay, does the job.",
    "Great product, very satisfied!"
])
y_val = np.array([0, 1, 2])

# Test data
X_test = np.array([
    "This is absolutely horrible!",
    "It's an average product, nothing more.",
    "Fantastic! Love every aspect of it!",
    "Really poor design and quality.",
    "Works well, good value for money.",
    "Outstanding product with amazing features!"
])
y_test = np.array([0, 1, 2, 0, 1, 2])

