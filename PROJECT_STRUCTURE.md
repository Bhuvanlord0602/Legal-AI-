# Project Structure

## Entry Points
- `app.py` - Streamlit UI
- `TRAINING.md` - training and inference guide

## Core
- `core/config_manager.py` - model and training config
- `core/llm.py` - Groq/Gemini LLM access

## Tokenization
- `tokenization/tokenizer.py` - Kannada tokenizer helpers
- `tokenization/tokenizer_sp.py` - SentencePiece wrapper

## Translation
- `translation/translator.py` - glossary translation
- `translation/neural_translator.py` - neural + hybrid translation
- `translation/nmt_translator.py` - optional NMT wrapper
- `translation/dictionary.py` - glossary loading helpers

## NLP
- `nlp/preprocessing.py` - text cleanup
- `nlp/ner_legal.py` - legal entity extraction
- `nlp/feature_extraction.py` - text feature helpers
- `nlp/utils.py` - text and document utilities

## Retrieval
- `retrieval/embeddings.py` - TF-IDF vectorization
- `retrieval/vector_db.py` - vector store and persistence
- `retrieval/qa.py` - question answering
- `retrieval/model.py` - retrieval model helpers

## Evaluation
- `evaluation/metrics_eval.py` - BLEU and ROUGE
- `evaluation/evaluate.py` - evaluation runner
- `evaluation/eval_hybrid_system.py` - hybrid system checks
- `evaluation/test_system.py` - end-to-end test script
- `evaluation/verify_implementation.py` - structure and setup verifier

## Training
- `training/train_pipeline.py` - corpus loading and splitting
- `training/finetune_model.py` - fine-tuning script

## Scripts
- `scripts/build_glossary_from_pdf.py` - glossary extraction from PDF
- `scripts/check_terms.py` - glossary term checks

## Data and Outputs
- `data/` - corpora, glossary CSVs, cached indexes
- `models/` - trained model outputs
- `config/` - saved configs

## Root Compatibility Wrappers
The root-level `.py` files now forward to these organized folders so existing commands still work.
