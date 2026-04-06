# embeddings.py

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

def fit_transform(texts):
    return vectorizer.fit_transform(texts)

def transform(texts):
    return vectorizer.transform(texts)

def get_vectorizer():
    return vectorizer