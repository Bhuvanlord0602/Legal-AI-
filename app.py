import streamlit as st
from preprocessing import preprocess_text, tokenize_text, encode_text
from feature_extraction import extract_features
from utils import extract_text_from_pdf, extract_text_from_image, extract_entities, summarize_text

st.title("Kannada Legal NLP System")

uploaded_file = st.file_uploader("Upload Document", type=["pdf", "png", "jpg"])

if uploaded_file:
    # Extract text
    if uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(uploaded_file)
    else:
        text = extract_text_from_image(uploaded_file)

    st.subheader("📄 Raw Text")
    st.write(text)

    # Preprocessing
    clean_text = preprocess_text(text)
    st.subheader("🧹 Cleaned Text")
    st.write(clean_text)

    # Tokenization
    tokens = tokenize_text(clean_text)
    st.subheader("🔡 Tokens")
    st.write(tokens[:100])

    # Encoding
    encoded = encode_text(clean_text)
    st.subheader("🔢 Encoded")
    st.write(encoded[:100])

    # Features
    features = extract_features([clean_text])
    st.subheader("📊 TF-IDF Features Shape")
    st.write(features.shape)

    # Entities
    entities = extract_entities(clean_text)
    st.subheader("📌 Entities")
    st.write(entities)

    # Summary
    summary = summarize_text(clean_text)
    st.subheader("📝 Summary")
    st.write(summary)