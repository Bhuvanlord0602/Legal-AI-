"""Better offline evaluation using known glossary terms."""

import csv
from pathlib import Path

from retrieval.embeddings import fit_transform, get_vectorizer, transform
from nlp.preprocessing import preprocess_text
from retrieval.qa import answer_question
from translation.translator import translate
from nlp.utils import chunk_text
from retrieval.vector_db import VectorDB


# Load a few known terms from glossary to create test
def load_test_kannada_terms(n=10):
    glossary_path = Path("data/legal_glossary.csv")
    terms = []
    if glossary_path.exists():
        with glossary_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= n:
                    break
                kn = row.get("kannada", "").strip().split(",")[0]  # take first if multiple
                en = row.get("english", "").strip()
                if kn and en:
                    terms.append({"kannada": kn, "english": en})
    return terms


def test_translation_with_known_terms():
    known = load_test_kannada_terms(15)
    if not known:
        print("No glossary terms found.")
        return

    print("\n=== Translation Test with Known Glossary Terms ===")
    print(f"Testing {len(known)} known terms:\n")

    for item in known[:5]:
        kn = item["kannada"]
        expected_en = item["english"]
        translated = translate(kn)
        match = expected_en.lower() in translated.lower()
        status = "✓" if match else "✗"
        print(f"{status} {kn:30} -> {translated:40} (expected: {expected_en})")


def test_retrieval_with_english_queries():
    print("\n=== Retrieval Test with English Queries ===")

    # Use a known English term that will appear in translated text
    sample_kannada = "ಕಾನೂನು ಒಪ್ಪಂದ"
    clean = preprocess_text(sample_kannada)
    translated = translate(clean)

    print(f"Sample Kannada: {sample_kannada}")
    print(f"Translated: {translated}")

    chunks = chunk_text(translated, size=20)
    vectors = fit_transform(chunks)
    db = VectorDB()
    db.add(vectors, chunks)

    test_queries = ["law", "agreement"]
    for q in test_queries:
        query_vec = transform([q])
        results = db.search(query_vec, top_k=2, min_similarity=0.0)
        print(f"\nQuery '{q}':")
        print(f"  Results: {results}")
        print(f"  Match: {'Yes' if results else 'No'}")


if __name__ == "__main__":
    test_translation_with_known_terms()
    test_retrieval_with_english_queries()
