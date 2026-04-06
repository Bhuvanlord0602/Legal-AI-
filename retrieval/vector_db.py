# vector_db.py

import json
import pickle
from pathlib import Path

from scipy.sparse import load_npz, save_npz

class VectorDB:
    def __init__(self):
        self.texts = []
        self.vectors = None

    def add(self, vectors, texts):
        self.texts = texts
        self.vectors = vectors

    def save(self, index_dir, vectorizer):
        index_path = Path(index_dir)
        index_path.mkdir(parents=True, exist_ok=True)

        with (index_path / "texts.json").open("w", encoding="utf-8") as f:
            json.dump(self.texts, f, ensure_ascii=False)

        save_npz(index_path / "vectors.npz", self.vectors)

        with (index_path / "vectorizer.pkl").open("wb") as f:
            pickle.dump(vectorizer, f)

    @classmethod
    def load(cls, index_dir):
        index_path = Path(index_dir)
        db = cls()

        with (index_path / "texts.json").open("r", encoding="utf-8") as f:
            db.texts = json.load(f)

        db.vectors = load_npz(index_path / "vectors.npz")

        with (index_path / "vectorizer.pkl").open("rb") as f:
            loaded_vectorizer = pickle.load(f)

        return db, loaded_vectorizer

    def search(self, query_vector, top_k=3, min_similarity=0.05):
        if self.vectors is None or not self.texts:
            return []

        similarities = (self.vectors @ query_vector.T).toarray()
        scores = similarities.flatten()
        top_indices = scores.argsort()[-top_k:][::-1]

        filtered_indices = [i for i in top_indices if scores[i] >= min_similarity]
        return [self.texts[i] for i in filtered_indices]