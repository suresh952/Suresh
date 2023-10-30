# Suresh
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from scipy.special import softmax
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Preprocess the text data
vectorizer = CountVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(df['Text'])

# Apply LDA
lda = LatentDirichletAllocation(n_components=5, random_state=42)
topics = lda.fit_transform(X)

# Print topics and visualize
print("Top words for each topic:")
feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    top_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
    print(f"Topic #{topic_idx+1}: {' '.join(top_words)}")
    import spacy

# Load the English NER model from spaCy
nlp = spacy.load("en_core_web_sm")

# Apply NER to the text data
def extract_entities(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    return entities

# Apply NER to all tweets
df['Entities'] = df['Text'].apply(extract_entities)

# Print NER results for a sample tweet
sample_tweet = df.loc[0, 'Text']
sample_entities = df.loc[0, 'Entities']
print(f"Sample Tweet: {sample_tweet}")
print(f"Extracted Entities: {sample_entities}")
# Define a function to perform dependency parsing
def dependency_parsing(text):
    doc = nlp(text)
    dependencies = []

    for token in doc:
        dependency = {
            'Token': token.text,
            'Dep': token.dep_,
            'Head Token': token.head.text,
            'Head Dep': token.head.dep_
        }
        dependencies.append(dependency)

    return dependencies

# Apply dependency parsing to a sample tweet
sample_tweet = df.loc[0, 'Text']
sample_dependencies = dependency_parsing(sample_tweet)
print(f"Sample Dependency Parsing for '{sample_tweet}':")
for dependency in sample_dependencies:
    print(dependency)
