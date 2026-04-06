"""Comprehensive evaluation script for legal translation system with BLEU/ROUGE metrics."""

from evaluation.metrics_eval import evaluate_translation_pair, batch_evaluate
from nlp.ner_legal import highlight_entities
from translation.nmt_translator import get_nmt_model
from translation.translator import translate as glossary_translate


# Sample legal test cases (reference translation pairs)
TEST_CASES = [
    {
        "english": "The agreement between the parties shall be binding.",
        "kannada_reference": "ಪಕ್ಷಗಳ ನಡುವಿನ ಒಪ್ಪಂದ ಬಂಧನಕಾರಕ ಆಗಿರುತ್ತದೆ.",
    },
    {
        "english": "Section 45 of the Indian Penal Code prescribes punishment.",
        "kannada_reference": "ಭಾರತೀಯ ದಂಡ ಸಂಹಿತೆಯ ವಿಭಾಗ 45 ಶಿಕ್ಷೆಯನ್ನು ನಿರ್ದಿಷ್ಟಪಡಿಸುತ್ತದೆ.",
    },
    {
        "english": "The High Court shall review the judgment.",
        "kannada_reference": "ಉಚ್ಚ ನ್ಯಾಯಾಲಯವು ತೀರ್ಪನ್ನು ಪರಿಶೀಲಿಸುವುದು.",
    },
    {
        "english": "Plaintiff versus Defendant case number 2024.",
        "kannada_reference": "ಮೂಲೆದಾರ ವಿರುದ್ಧ ಪ್ರತಿವಾದಿ ಪ್ರಕರಣ ಸಂಖ್ಯೆ 2024.",
    },
]


def evaluate_glossary_translation():
    """Evaluate glossary-based translation."""
    print("\n=== Glossary-Based Translation Evaluation ===")
    print("Note: Glossary performs word substitution only\n")

    for i, case in enumerate(TEST_CASES, 1):
        # For glossary, we use kannada as input
        translated = glossary_translate(case["kannada_reference"])
        print(f"Test case {i}:")
        print(f"  Reference Kannada: {case['kannada_reference']}")
        print(f"  Translated to English: {translated}")


def evaluate_nmt_translation():
    """Evaluate NMT translation (if available)."""
    print("\n=== Neural Machine Translation (IndicTrans) Evaluation ===")

    model = get_nmt_model()
    if not model.available:
        print("IndicTrans model not available in current environment.")
        print("To use NMT, install via: pip install IndicTransToolkit torch ")
        return

    print("IndicTrans model loaded. Evaluating English->Kannada translation:\n")

    references = []
    hypotheses = []

    for i, case in enumerate(TEST_CASES, 1):
        translated = model.translate_en_to_kn(case["english"])
        if translated:
            references.append(case["kannada_reference"])
            hypotheses.append(translated)

            print(f"Test case {i}:")
            print(f"  English: {case['english']}")
            print(f"  Expected Kannada: {case['kannada_reference']}")
            print(f"  Model Output: {translated}")

            metrics = evaluate_translation_pair(case["kannada_reference"], translated)
            print(f"  BLEU Score: {metrics['bleu']}")
            print(f"  ROUGE-F Score: {metrics['rouge_f_score']}\n")

    if hypotheses:
        print("\n=== Batch Statistics ===")
        batch_scores = batch_evaluate(references, hypotheses)
        print(f"Average BLEU: {batch_scores['average_bleu']}")
        print(f"Average ROUGE-F: {batch_scores['average_rouge_f']}")
        print(f"Samples evaluated: {batch_scores['num_samples']}")


def evaluate_ner():
    """Demonstrate Named Entity Recognition."""
    print("\n=== Legal Named Entity Recognition ===\n")

    sample_legal_text = """
    In the case of Plaintiff vs. Defendant (High Court, 2024),
    Section 45 of the Indian Penal Code 1860 was invoked.
    The Supreme Court under Article 32 ruled on jurisdiction matters.
    """

    entities = highlight_entities(sample_legal_text)

    print(f"Text: {entities['text'].strip()}\n")
    print(f"Total entities found: {entities['entity_count']}\n")

    for entity_type, entity_list in entities["entities"].items():
        if entity_list:
            print(f"{entity_type.upper()}: {entity_list}")


def main():
    print("=" * 70)
    print("Legal Translation System: Hybrid NMT + Glossary Evaluation")
    print("=" * 70)

    # Evaluate glossary
    evaluate_glossary_translation()

    # Evaluate NMT (if available)
    evaluate_nmt_translation()

    # Demonstrate NER
    evaluate_ner()

    print("\n" + "=" * 70)
    print("Evaluation complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
