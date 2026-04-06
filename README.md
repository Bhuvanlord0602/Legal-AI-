# Kannada Legal AI

English-to-Kannada legal document intelligence system with document extraction, hybrid translation, legal entity extraction, retrieval-based Q&A, and optional model fine-tuning.

## 1. Project Goal

The goal of this project is to make legal content easier to process and understand by combining:

- OCR and legal document text extraction
- Kannada text preprocessing and tokenization
- Hybrid translation (Neural + Glossary fallback)
- Legal-domain entity extraction
- Retrieval-Augmented Question Answering (RAG)

This allows users to upload legal files, convert text, identify legal entities, and ask questions grounded in document content.

## 2. End-to-End Pipeline

### Inference Pipeline

1. Document Upload
2. Text Extraction
3. Text Preprocessing
4. Hybrid Translation
5. Legal NER Extraction
6. Chunking + Vectorization
7. Index Creation or Cached Index Load
8. User Query -> Retrieval -> LLM Answer

### Pipeline Diagram

Upload File -> OCR/Text Extract -> Clean/Normalize -> Hybrid Translation -> Legal NER -> Chunking -> TF-IDF Embedding -> Vector Search -> Context Assembly -> LLM Response

## 3. Core Features

- Streamlit interface for legal document upload and Q&A
- OCR support for scanned legal pages
- Neural translation with glossary fallback behavior
- Legal NER for acts, sections, articles, courts, cases, years
- Retrieval QA over uploaded documents
- BLEU and ROUGE metrics for translation quality evaluation
- Fine-tuning pipeline for legal translation adaptation

## 4. Repository Layout

- [app.py](app.py): application entry point (UI + orchestration)
- [core](core): configuration and LLM provider integration
- [tokenization](tokenization): Kannada tokenizer and SentencePiece utilities
- [translation](translation): glossary dictionary, neural translator, hybrid translation logic
- [nlp](nlp): preprocessing, legal entity extraction, helper utilities
- [retrieval](retrieval): embeddings, vector DB, and QA logic
- [evaluation](evaluation): metrics and system validation scripts
- [training](training): corpus preparation and model fine-tuning scripts
- [scripts](scripts): glossary extraction and maintenance scripts
- [data](data): datasets and generated indexes
- [models](models): model checkpoints and final artifacts

Detailed module map: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

## 5. Technology Stack

- Python 3.13+
- Streamlit
- scikit-learn
- PyTorch
- transformers + datasets
- SentencePiece
- pdfplumber + pytesseract + Pillow
- Groq API and/or Gemini API

## 6. Installation and Setup

### Step 1: Environment

Create and activate a virtual environment.

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Configure LLM Keys

Set one of the following environment variables:

- GROQ_API_KEY (preferred)
- GEMINI_API_KEY

### Step 4: OCR Requirement

Install Tesseract on your system and ensure Kannada language data is available if processing Kannada OCR-heavy documents.

## 7. Running the Application

```bash
streamlit run app.py
```

After upload, the app will:

- extract text
- preprocess and translate
- extract legal entities
- build/load vector index
- answer legal questions from retrieved context

## 8. Training and Evaluation Workflow

### 8.1 Prepare Corpus Splits

```bash
python training/train_pipeline.py
```

Generates split metadata and train/validation/test artifacts.

### 8.2 Fine-Tune Translation Model

```bash
python training/finetune_model.py
```

Performs MarianMT-based fine-tuning on the legal bilingual corpus.

### 8.3 Evaluate System and Metrics

```bash
python evaluation/eval_hybrid_system.py
python evaluation/verify_implementation.py
```

## 9. Data and Model Artifacts

Important resources:

- [data/legal_corpus.csv](data/legal_corpus.csv): bilingual legal corpus
- [data/legal_glossary.csv](data/legal_glossary.csv): extracted glossary
- [data/Legal Kanna.pdf](data/Legal%20Kanna.pdf): source legal document

Generated artifacts:

- [data](data): index files, corpus info, split outputs
- [models](models): checkpoints and final fine-tuned model

## 10. Common Commands

```bash
streamlit run app.py
python training/train_pipeline.py
python training/finetune_model.py
python evaluation/eval_hybrid_system.py
python evaluation/verify_implementation.py
```

## 11. Known Behavior and Notes

- If neural translation is unavailable, glossary fallback is used automatically.
- Small bilingual corpora can train quickly but may produce unstable translation quality.
- Better legal translation quality requires larger, cleaner domain-aligned parallel data.
- Keep cache, checkpoint, and pycache artifacts out of commits where possible.

## 12. Roadmap

- Expand bilingual legal corpus significantly
- Improve legal term normalization and phrase-level translation
- Add stronger retrieval reranking
- Add automated regression checks for translation quality
- Add deployment configuration for hosted inference

## 13. Disclaimer

This project is for educational and research purposes. It is not legal advice.
