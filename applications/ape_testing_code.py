
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
import os
import nltk
import spacy
nltk.download('stopwords')
os.chdir("ape")
naf = pd.read_excel("data/naf.parquet", skiprows = 2)
naf['Code'] = naf['Code'].str.replace(".","")
train = pd.read_parquet("data/data.parquet")
train = train.merge(naf, left_on = "nace", right_on = "Code")
train.head(5)


# Part 1 ----------------------------------------------------------------

from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')
nltk.download('stopwords')

nlp = spacy.load("fr_core_news_sm")

# Function to remove stopwords
def remove_stopwords(text):
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_text)

def remove_single_letters(text):
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if len(word) > 1]
    return ' '.join(filtered_text)

# Apply the function to the 'text' column
# train['text_clean'] = (train['text']
#     .apply(remove_stopwords)
#     .apply(remove_single_letters)
# )

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

nlp = spacy.load("fr_core_news_sm")
stop_words = nlp.Defaults.stop_words

stop_words = set(stop_words)

train['text_clean'] = (train['text']
    .apply(remove_stopwords)
    .apply(remove_single_letters)
)




from processor import Preprocessor
preprocessor = Preprocessor()

# Preprocess data before training and testing
TEXT_FEATURE = "text"
Y = "nace"

df = preprocessor.clean_text(train, TEXT_FEATURE).drop('text_clean', axis = "columns")
df.head(2)


import pathlib
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
from torchTextClassifiers.tokenizers.ngram import NGramTokenizer
from torchTextClassifiers import ModelConfig, TrainingConfig, torchTextClassifiers
from torchTextClassifiers.tokenizers import WordPieceTokenizer


tokenizer = WordPieceTokenizer(vocab_size=5000, output_dim=128)
tokenizer.train(X_train.tolist())


# Step 4: Configure Model for 3 Classes
import numpy as np
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
