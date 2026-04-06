"""Neural Machine Translation using fine-tuned or pre-trained models with glossary fallback."""

from pathlib import Path
from typing import Optional, Callable, Tuple
import json

# Optional imports
try:
    from transformers import MarianMTModel, MarianTokenizer, pipeline
    import torch

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class NeuralTranslator:
    """Neural translation with model management and fallback."""

    def __init__(self, fine_tuned_model_path: Optional[str] = None, use_gpu: bool = True):
        """Initialize neural translator.

        Args:
            fine_tuned_model_path: Path to fine-tuned model directory
            use_gpu: Whether to use GPU if available
        """
        self.fine_tuned_model_path = fine_tuned_model_path
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.translator_pipeline = None
        self.model_source = None
        
        if HF_AVAILABLE:
            self._load_model()

    def _load_model(self):
        """Load fine-tuned or pre-trained model."""
        if self.fine_tuned_model_path:
            fine_tuned_path = Path(self.fine_tuned_model_path)
            if fine_tuned_path.exists():
                try:
                    print(f"Loading fine-tuned model from {fine_tuned_path}")
                    self.model = MarianMTModel.from_pretrained(str(fine_tuned_path))
                    self.tokenizer = MarianTokenizer.from_pretrained(str(fine_tuned_path))
                    self.model.to(self.device)
                    self.model_source = "fine-tuned"
                    print(f"✓ Fine-tuned model loaded on device: {self.device}")
                    return
                except Exception as e:
                    print(f"Warning: Could not load fine-tuned model: {e}")

        # Fallback to pre-trained model
        try:
            print("Loading pre-trained MarianMT model (en-mul)...")
            self.model = MarianMTModel.from_pretrained("Helsinki-NLP/Opus-MT-en-mul")
            self.tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/Opus-MT-en-mul")
            self.model.to(self.device)
            self.model_source = "pre-trained"
            print(f"✓ Pre-trained model loaded on device: {self.device}")
        except Exception as e:
            print(f"✗ Failed to load any model: {e}")
            HF_AVAILABLE = False

    def translate(self, english_text: str, min_length: int = 10, max_length: int = 128) -> str:
        """Translate English text to Kannada.

        Args:
            english_text: English text to translate
            min_length: Minimum output length
            max_length: Maximum output length

        Returns:
            Translated Kannada text
        """
        if not HF_AVAILABLE or self.model is None:
            return english_text  # Return unchanged if model unavailable

        try:
            inputs = self.tokenizer(english_text, return_tensors="pt").to(self.device)
            translated_ids = self.model.generate(
                **inputs,
                min_length=min_length,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
            )
            translation = self.tokenizer.batch_decode(
                translated_ids, skip_special_tokens=True
            )[0]
            return translation
        except Exception as e:
            print(f"Translation error: {e}")
            return english_text

    def batch_translate(self, texts: list, batch_size: int = 32) -> list:
        """Translate multiple texts.

        Args:
            texts: List of English texts
            batch_size: Batch size for processing

        Returns:
            List of translated texts
        """
        translations = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_translations = [self.translate(text) for text in batch]
            translations.extend(batch_translations)
        return translations

    def get_status(self) -> dict:
        """Get model status information."""
        return {
            "model_available": HF_AVAILABLE and self.model is not None,
            "model_source": self.model_source,
            "device": self.device,
            "fine_tuned_path": self.fine_tuned_model_path,
        }


class HybridTranslator:
    """Hybrid translation combining neural model with glossary fallback."""

    def __init__(
        self,
        glossary_dict: Optional[dict] = None,
        fine_tuned_model_path: Optional[str] = None,
        use_neural: bool = True,
    ):
        """Initialize hybrid translator.

        Args:
            glossary_dict: Dictionary of term translations {english: kannada}
            fine_tuned_model_path: Path to fine-tuned model
            use_neural: Whether to use neural translation
        """
        self.glossary = glossary_dict or {}
        self.use_neural = use_neural
        self.neural_translator = None

        if use_neural and HF_AVAILABLE:
            self.neural_translator = NeuralTranslator(fine_tuned_model_path)

    def translate(self, english_text: str) -> Tuple[str, str]:
        """Translate with fallback chain.

        Args:
            english_text: English text to translate

        Returns:
            Tuple of (kannada_translation, method_used)
        """
        # Try neural translation first
        if self.neural_translator is not None:
            translation = self.neural_translator.translate(english_text)
            if translation != english_text:
                return translation, "neural"

        # Try glossary lookup
        if english_text.lower() in self.glossary:
            return self.glossary[english_text.lower()], "glossary"

        # Try partial glossary matching
        words = english_text.lower().split()
        translated_words = []
        for word in words:
            if word in self.glossary:
                translated_words.append(self.glossary[word])
            else:
                translated_words.append(word)

        if translated_words != words:
            return " ".join(translated_words), "glossary_partial"

        # Return original if no translation found
        return english_text, "none"

    def get_status(self) -> dict:
        """Get status of hybrid translator."""
        status = {
            "glossary_size": len(self.glossary),
            "neural_enabled": self.use_neural,
        }
        if self.neural_translator:
            status["neural_status"] = self.neural_translator.get_status()
        return status


def load_glossary_from_dict(
    glossary_path: str = "data/legal_glossary.csv",
) -> dict:
    """Load glossary from CSV file.

    Args:
        glossary_path: Path to glossary CSV (english,kannada format)

    Returns:
        Dictionary of translations
    """
    import csv

    glossary = {}
    glossary_file = Path(glossary_path)

    if glossary_file.exists():
        with open(glossary_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                en = row.get("english", "").strip().lower()
                kn = row.get("kannada", "").strip()
                if en and kn:
                    glossary[en] = kn

    return glossary


def main():
    """Example usage."""
    print("Neural Translation Module")
    print("=" * 70)

    # Initialize neural translator
    print("\n1. Neural Translator")
    print("-" * 70)
    neural = NeuralTranslator()
    print(f"Status: {neural.get_status()}")

    # Test translation
    test_text = "The contract is legally binding under Indian law."
    print(f"\nInput: {test_text}")
    translation = neural.translate(test_text)
    print(f"Output: {translation}")

    # Initialize hybrid translator
    print("\n2. Hybrid Translator")
    print("-" * 70)

    # Load glossary
    glossary = load_glossary_from_dict()
    print(f"Loaded {len(glossary)} glossary terms")

    # Create hybrid translator
    hybrid = HybridTranslator(
        glossary_dict=glossary, use_neural=True
    )

    # Test hybrid translation
    print(f"\nTest: {test_text}")
    translation, method = hybrid.translate(test_text)
    print(f"Translation: {translation}")
    print(f"Method: {method}")

    # Status
    print(f"\nHybrid Status: {hybrid.get_status()}")


if __name__ == "__main__":
    main()
