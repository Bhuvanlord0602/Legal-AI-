from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

def extract_features(texts):
    return vectorizer.fit_transform(texts)

def transform_features(texts):
    return vectorizer.transform(texts)