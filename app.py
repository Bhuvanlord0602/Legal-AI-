import hashlib
import os
from pathlib import Path

import streamlit as st

from nlp.utils import extract_text_from_pdf, extract_text_from_image, chunk_text
from nlp.preprocessing import preprocess_text
from translation.translator import translate as glossary_translate
from translation.neural_translator import HybridTranslator, load_glossary_from_dict, NeuralTranslator
from nlp.ner_legal import highlight_entities
from retrieval.embeddings import fit_transform, get_vectorizer
from retrieval.vector_db import VectorDB
from retrieval.qa import answer_question

st.title("⚖️ Kannada Legal AI (Hybrid NMT + RAG)")

has_groq = bool(os.getenv("GROQ_API_KEY"))
has_gemini = bool(os.getenv("GEMINI_API_KEY"))
# Initialize hybrid translator with fine-tuned model if available
fine_tuned_path = Path("models/legal_translation_model/final_model")
glossary = load_glossary_from_dict()
translator = HybridTranslator(
    glossary_dict=glossary,
    fine_tuned_model_path=str(fine_tuned_path) if fine_tuned_path.exists() else None,
    use_neural=True
)
translator_status = translator.get_status()

col1, col2 = st.columns(2)
with col1:
    if translator_status.get("neural_status", {}).get("model_available"):
        model_source = translator_status.get("neural_status", {}).get("model_source", "unknown")
        st.success(f"✓ Neural translation: Active ({model_source})")
    else:
        st.info(f"ℹ Translation: Glossary ({translator_status['glossary_size']} terms)")

with col2:
    if has_groq:
        st.success("✓ LLM: Groq (configured)")
    elif has_gemini:
        st.info("ℹ LLM: Gemini (configured)")
    else:
        st.warning("⚠ No LLM key found")

uploaded_file = st.file_uploader("Upload Document", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    document_id = hashlib.md5(file_bytes).hexdigest()
    index_dir = Path("data") / "indexes" / document_id

    if uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(uploaded_file)
    else:
        text = extract_text_from_image(uploaded_file)

    st.subheader("📄 Raw Text")
    st.write(text[:1000])

    # preprocess
    clean_text = preprocess_text(text)

    # translate (try NMT first, fallback to glossary)
    english_text, method = translator.translate(clean_text)
    
    # extract legal entities
    entities_info = highlight_entities(english_text)
    if entities_info["entity_count"] > 0:
        st.info(f"🔍 Found {entities_info['entity_count']} legal entities (acts, sections, courts, etc.)")

    st.subheader("🌍 Translated Text (English)")
    st.write(english_text[:1000])
    
    with st.expander("Show extracted legal entities"):
        st.json(entities_info["entities"])

    if (index_dir / "texts.json").exists() and (index_dir / "vectors.npz").exists() and (index_dir / "vectorizer.pkl").exists():
        db, active_vectorizer = VectorDB.load(index_dir)
        st.caption("Using cached document index")
    else:
        # chunk
        chunks = chunk_text(english_text)
        if not chunks:
            st.warning("No text chunks were generated from the document.")
            st.stop()

        # embeddings
        vectors = fit_transform(chunks)

        # store
        db = VectorDB()
        db.add(vectors, chunks)
        active_vectorizer = get_vectorizer()
        db.save(index_dir, active_vectorizer)
        st.caption("Created and saved document index")

    # ask question
    query = st.text_input("Ask a legal question")

    if query:
        answer = answer_question(query, db, active_vectorizer)

        st.subheader("📌 Answer")
        st.write(answer)
