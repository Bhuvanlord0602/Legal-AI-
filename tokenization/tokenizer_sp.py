"""SentencePiece tokenizer wrapper for legal text."""

from pathlib import Path
from typing import List, Tuple

try:
    import sentencepiece as spm
except ImportError:
    spm = None


class LegalTokenizer:
    """Tokenizer for legal text using SentencePiece."""

    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.sp_models = {}  # Dict to store separate models for each language

    def train(
        self,
        texts: List[str],
        vocab_size: int = 32000,
        model_prefix: str = "legal_tokenizer",
        language: str = "en",
    ):
        """
        Train a SentencePiece model.

        Args:
            texts: list of texts to train on
            vocab_size: vocabulary size
            model_prefix: prefix for model files
            language: language code (en, kan, etc.)
        """
        if spm is None:
            raise ImportError("sentencepiece not installed. Install: pip install sentencepiece")

        # Save texts to temp file
        temp_file = f"/tmp/{language}_texts.txt"
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write("\n".join(texts))

        # Train model
        spm.SentencePieceTrainer.train(
            input=temp_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            normalization_rule_name="identity",  # Preserve Kannada script
            character_coverage=0.99,
            user_defined_symbols=["[ACT]", "[SECTION]", "[COURT]", "[CASE]"],
        )

        # Load model
        model_file = f"{model_prefix}.model"
        self.sp_models[language] = spm.SentencePieceProcessor()
        self.sp_models[language].Load(model_file)

        self.model_path = model_prefix
        return self.sp_models[language]

    def load(self, model_path: str, language: str = "en"):
        """Load a pre-trained SentencePiece model."""
        if spm is None:
            raise ImportError("sentencepiece not installed. Install: pip install sentencepiece")

        model_file = f"{model_path}.model"
        if not Path(model_file).exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        self.sp_models[language] = spm.SentencePieceProcessor()
        self.sp_models[language].Load(model_file)
        self.model_path = model_path

    def tokenize(self, text: str, language: str = "en") -> List[int]:
        """Tokenize text to IDs."""
        if language not in self.sp_models:
            raise ValueError(f"Model for {language} not loaded")

        return self.sp_models[language].EncodeAsIds(text)

    def tokenize_pieces(self, text: str, language: str = "en") -> List[str]:
        """Tokenize text to pieces."""
        if language not in self.sp_models:
            raise ValueError(f"Model for {language} not loaded")

        return self.sp_models[language].EncodeAsPieces(text)

    def decode(self, ids: List[int], language: str = "en") -> str:
        """Decode IDs back to text."""
        if language not in self.sp_models:
            raise ValueError(f"Model for {language} not loaded")

        return self.sp_models[language].DecodeIds(ids)

    def get_vocab_size(self, language: str = "en") -> int:
        """Get vocabulary size."""
        if language not in self.sp_models:
            return 0
        return self.sp_models[language].GetPieceSize()


def create_bilingual_tokenizer(
    en_texts: List[str],
    kn_texts: List[str],
    vocab_size: int = 32000,
    model_prefix: str = "legal_bilingual",
) -> Tuple[LegalTokenizer, LegalTokenizer]:
    """
    Create separate tokenizers for English and Kannada.

    Args:
        en_texts: English texts
        kn_texts: Kannada texts
        vocab_size: vocabulary size for each
        model_prefix: prefix for model files

    Returns:
        Tuple of (en_tokenizer, kn_tokenizer)
    """
    en_tokenizer = LegalTokenizer()
    kn_tokenizer = LegalTokenizer()

    en_tokenizer.train(en_texts, vocab_size, f"{model_prefix}_en", language="en")
    kn_tokenizer.train(kn_texts, vocab_size, f"{model_prefix}_kn", language="kan")

    return en_tokenizer, kn_tokenizer
