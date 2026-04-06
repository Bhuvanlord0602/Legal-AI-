"""Offline evaluation for translation coverage and retrieval quality.

Run:
python evaluate.py
"""

from retrieval.embeddings import fit_transform, get_vectorizer, transform
from nlp.preprocessing import preprocess_text
from retrieval.qa import answer_question
from translation.translator import translate
from nlp.utils import chunk_text
from retrieval.vector_db import VectorDB


SAMPLE_TEXT = """
ಬಂಧನ ಒಪ್ಪಂದ ಕಾನೂನು ಗಿ ಬದ್ಧ.
ಏಜೆನ್ಸಿ ದೊರೆಗಾಲು ಪ್ರಕರಣ ನಿಕಟಿ.
""".strip()

TEST_QUERIES = [
    {
        "query": "agreement",
        "expected_any": ["agreement"],
    },
    {
        "query": "law",
        "expected_any": ["law"],
    },
]


def evaluate_translation(clean_text):
    translated = translate(clean_text)
    clean_split = clean_text.split()
    trans_split = translated.split()
    total_tokens = len(clean_split) or 1
    changed_tokens = sum(1 for a, b in zip(clean_split, trans_split) if a != b)
    coverage = changed_tokens / total_tokens
    
    missing_terms = [c for c, t in zip(clean_split, trans_split) if c == t and '\u0C80' <= c[0] <= '\u0CFF']
    
    return translated, coverage, missing_terms


def evaluate_retrieval(translated_text):
    chunks = chunk_text(translated_text, size=40)
    vectors = fit_transform(chunks)

    db = VectorDB()
    db.add(vectors, chunks)

    hits = 0
    for item in TEST_QUERIES:
        query = item["query"]
        expected = item["expected_any"]

        query_vec = transform([query])
        contexts = db.search(query_vec, top_k=3, min_similarity=0.0)
        joined = " ".join(contexts).lower()
        if any(term.lower() in joined for term in expected):
            hits += 1

    return hits, len(TEST_QUERIES)


def main():
    clean_text = preprocess_text(SAMPLE_TEXT)
    translated, coverage, missing = evaluate_translation(clean_text)
    hits, total = evaluate_retrieval(translated)

    print("=== Kannada Legal AI Offline Evaluation ===")
    print(f"Input Kannada text: {SAMPLE_TEXT[:100]}...")
    print(f"Translated text: {translated}")
    print(f"Translation token-change coverage: {coverage:.2%}")
    if missing:
        print(f"⚠️  Untranslated Kannada terms ({len(missing)}): {', '.join(set(missing[:5]))}")
    else:
        print("✓ All Kannada terms translated")
    print(f"Retrieval pass rate: {hits}/{total} ({(hits / total):.2%})")
    if hits == total:
        print("✓ Retrieval working correctly")
    else:
        print("⚠️  Retrieval needs improvement")

    # This uses the QA fallback when LLM is unavailable.
    chunks = chunk_text(translated, size=40)
    db = VectorDB()
    db.add(fit_transform(chunks), chunks)
    answer = answer_question("What does the law require?", db, get_vectorizer())
    print(f"Sample QA response: {answer[:200]}")


if __name__ == "__main__":
    main()
