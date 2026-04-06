"""Neural Machine Translation using IndicTrans for English->Kannada legal text."""

import os

try:
    from IndicTransToolkit import IndicTransToolkit
except ImportError:
    IndicTransToolkit = None


class IndicTransNMT:
    """Wrapper for IndicTrans neural machine translation."""

    def __init__(self):
        self.model = None
        self.available = False
        self._initialize()

    def _initialize(self):
        if IndicTransToolkit is None:
            return

        try:
            self.model = IndicTransToolkit(
                gpu=True,
                model="en-indic",
                quantization=False,
            )
            self.available = True
        except Exception as e:
            print(f"IndicTrans initialization failed: {e}. Falling back to glossary mode.")
            self.available = False

    def translate_en_to_kn(self, english_text: str) -> str:
        """Translate English to Kannada using IndicTrans."""
        if not self.available or self.model is None:
            return None

        try:
            result = self.model.translate_paragraph(english_text, src_lang="eng_Latn", tgt_lang="kan_Knda")
            return result
        except Exception as e:
            print(f"IndicTrans translation failed: {e}")
            return None


# Global instance
_nmt_model = None


def get_nmt_model():
    global _nmt_model
    if _nmt_model is None:
        _nmt_model = IndicTransNMT()
    return _nmt_model


def translate_neural(english_text: str, fallback_fn=None) -> str:
    """
    Translate using IndicTrans NMT. If unavailable or fails, use fallback function.

    Args:
        english_text: text to translate
        fallback_fn: fallback translation function (e.g., glossary-based)

    Returns:
        Translated Kannada text
    """
    model = get_nmt_model()

    if model.available:
        result = model.translate_en_to_kn(english_text)
        if result:
            return result

    if fallback_fn:
        return fallback_fn(english_text)

    return english_text
