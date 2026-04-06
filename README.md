# Kannada Legal AI

English-to-Kannada legal document processing with hybrid translation, retrieval-based QA, glossary lookup, and optional neural translation fine-tuning.

## What It Does

- Extracts text from PDF and image legal documents
- Cleans and preprocesses legal text
- Translates English legal text into Kannada using glossary and neural fallback
- Detects legal entities such as acts, sections, courts, and cases
- Builds TF-IDF document indexes for question answering
- Supports fine-tuning a translation model on a bilingual legal corpus

## Project Layout

- `app.py` - Streamlit app entry point
- `core/` - configuration and LLM access
- `nlp/` - preprocessing, tokenization helpers, legal NER, text utilities
- `translation/` - glossary translation and neural translation modules
- `retrieval/` - embeddings, vector DB, QA, retrieval model helpers
- `evaluation/` - metrics, test runners, and verification scripts
- `training/` - corpus preparation and model fine-tuning
- `scripts/` - PDF glossary extraction and supporting scripts
- `data/` - corpus, glossary, and cached indexes
- `models/` - trained model outputs

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for a quick module map.

## Setup

1. Create and activate the virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

## Train the Translation Pipeline

Prepare splits and corpus metadata:

```bash
python training/train_pipeline.py
```

Fine-tune the model:

```bash
python training/finetune_model.py
```

## Key Files

- `data/legal_corpus.csv` - bilingual training data
- `data/legal_glossary.csv` - glossary terms extracted from the PDF
- `data/Legal Kanna.pdf` - source document used for glossary extraction

## Notes

- The neural translator falls back to glossary translation if a trained model is unavailable.
- The current corpus is small, so quality improves as you add more bilingual legal pairs.
- The repository structure is organized by concern, with compatibility kept through package imports.

## Requirements

- Python 3.13+
- Streamlit
- PyTorch
- HuggingFace Transformers
- SentencePiece
- scikit-learn
- Groq API key or Gemini API key for LLM-backed answering

## Common Commands

```bash
python training/train_pipeline.py
python training/finetune_model.py
python evaluation/verify_implementation.py
streamlit run app.py
```

## Status

This project is organized for a hybrid legal translation workflow:
- glossary-based translation
- neural translation fine-tuning
- legal NER
- retrieval-based QA
- Streamlit UI
