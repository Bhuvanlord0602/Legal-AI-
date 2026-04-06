# qa.py

from core.llm import ask_llm

def answer_question(query, db, vectorizer):
    query_vec = vectorizer.transform([query])
    results = db.search(query_vec)

    if not results:
        return "Not found in document"

    context = "\n".join(results)

    return ask_llm(context, query)