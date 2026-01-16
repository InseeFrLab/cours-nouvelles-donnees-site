import os
os.chdir("ape")
import pandas as pd

# Import NAF classification
naf = pd.read_excel("data/naf.parquet", skiprows = 2)
# Import training data
train = pd.read_parquet("data/data.parquet")
# train = train.sample(10000)

# Merge classification info
naf['Code'] = naf['Code'].str.replace(".","")
train = train.merge(naf, left_on = "nace", right_on = "Code")
train.head(5)



def filter_train_data(train_data, sequence):
   sequence_capitalized = sequence.upper() 
   mask = train_data['text'].str.contains(sequence_capitalized)
   nb_occurrence = mask.astype(int).sum()
   print(
       f"Nombre d'occurrences de la séquence '{sequence}': {nb_occurrence}"
   )
   return train_data.loc[mask]


from nltk.tokenize import word_tokenize
import spacy

os.system("python -m spacy download fr_core_news_sm")
nlp = spacy.load("fr_core_news_sm")
stop_words = nlp.Defaults.stop_words
stop_words = set(stop_words)

import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')

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
train['text_clean'] = (train['text']
    .apply(remove_stopwords)
    .apply(remove_single_letters)
)




from processor import Preprocessor
preprocessor = Preprocessor()


# Preprocess data before training and testing
TEXT_FEATURE = "text"
Y = "nace"

df = train.copy()
df = preprocessor.clean_text(df, TEXT_FEATURE).drop('text_clean', axis = "columns")
df.head(2)

df = df.dropna(subset = [Y, TEXT_FEATURE])
X = df[TEXT_FEATURE].values
y = df[Y].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # Convertit ["cat", "dog"] → [0, 1]

# Première division : train (80 %) + test (20%)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
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
    min_count=2, # On considère un mot s'il est trouvé au moins 2 fois dans le corpus
    min_n=2, 
    max_n=4, # On fait des 2grams, 3grams et 4grams de caractères
    len_word_ngrams=2, # On fait des 2grams de mots
    num_tokens=10000, # Nombre max de tokens considérés
    training_text=X,
)

from torchTextClassifiers import ModelConfig
import numpy as np

# Embedding dimension
embedding_dim = 64

# Count number of unique labels
unique_values, counts = np.unique(y, return_counts=True)
num_unique = len(unique_values)

model_config = ModelConfig(
    embedding_dim=embedding_dim,
    num_classes=num_unique
)

from torchTextClassifiers import torchTextClassifiers

classifier = torchTextClassifiers(
    tokenizer=tokenizer,
    model_config=model_config,
)


import torch
# s3_path = "s3://projet-formation/nouvelles-sources/model_ape.pth"

# state_dict = torch.load("model_ape.pth", map_location="cpu")



from torchTextClassifiers import TrainingConfig

# Training params (torch style)
training_config = TrainingConfig(
    num_epochs=30,
    batch_size=8,
    lr=1e-3,
    patience_early_stopping=7,
    num_workers=0,
    trainer_params={'deterministic': True},
    save_path="model_ape"
)

# Training !
classifier.train(
    X_train, 
    y_train,
    training_config,
    X_val, 
    y_val,
    verbose=True
)

# Inference on testset
result = classifier.predict(X_test)
predictions = result["prediction"].squeeze().numpy()

# Step 8: Evaluate
accuracy = (predictions == y_test).mean()
print(f"Test accuracy: {accuracy:.3f}")


# mc cp --recursive ape/model_ape s3/projet-formation/nouvelles-sources/
