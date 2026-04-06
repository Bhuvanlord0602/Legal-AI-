import re

from translation.dictionary import legal_dict_kn_en


# Kannada Unicode block and Latin letters/numbers are preserved as terms.
TOKEN_PATTERN = re.compile(r"[\u0C80-\u0CFFA-Za-z0-9_]+|[^\s]")


def translate(text: str) -> str:
    tokens = TOKEN_PATTERN.findall(text)
    translated = [legal_dict_kn_en.get(token, token) for token in tokens]
    return " ".join(translated)
