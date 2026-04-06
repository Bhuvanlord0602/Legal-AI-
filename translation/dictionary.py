"""Legal glossary helpers.

If a glossary file is placed in data/, it is loaded automatically.
Supported files: legal_glossary.csv, legal_glossary.tsv, legal_glossary.json
Expected columns/keys: english, kannada
"""

import csv
import json
import re
from pathlib import Path


DEFAULT_LEGAL_DICT_EN_KN = {
    "abandon": "ತ್ಯಜಿಸು",
    "abandonment": "ತ್ಯಾಗ",
    "law": "ಕಾನೂನು",
    "court": "ನ್ಯಾಯಾಲಯ",
    "agreement": "ಒಪ್ಪಂದ",
}


def _normalize_term(term: str) -> str:
    return term.strip().lower()


def _normalize_kn_term(term: str) -> str:
    return re.sub(r"\s+", " ", term).strip()


def _load_from_csv(path: Path, delimiter: str = ",") -> dict:
    glossary = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if not reader.fieldnames:
            return glossary

        field_map = {name.lower().strip(): name for name in reader.fieldnames}
        en_key = field_map.get("english")
        kn_key = field_map.get("kannada")
        if not en_key or not kn_key:
            return glossary

        for row in reader:
            en = (row.get(en_key) or "").strip()
            kn = (row.get(kn_key) or "").strip()
            if en and kn:
                glossary[_normalize_term(en)] = kn
    return glossary


def _load_from_json(path: Path) -> dict:
    glossary = {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        for en, kn in data.items():
            en = str(en).strip()
            kn = str(kn).strip()
            if en and kn:
                glossary[_normalize_term(en)] = kn
        return glossary

    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            en = str(item.get("english", "")).strip()
            kn = str(item.get("kannada", "")).strip()
            if en and kn:
                glossary[_normalize_term(en)] = kn
    return glossary


def load_legal_dictionary() -> dict:
    data_dir = Path(__file__).resolve().parent / "data"
    candidates = [
        (data_dir / "legal_glossary.csv", ","),
        (data_dir / "legal_glossary.sample.csv", ","),
        (data_dir / "legal_glossary.tsv", "\t"),
        (data_dir / "legal_glossary.json", None),
    ]

    for path, delimiter in candidates:
        if not path.exists():
            continue
        loaded = _load_from_json(path) if path.suffix == ".json" else _load_from_csv(path, delimiter)
        if loaded:
            return loaded

    return {_normalize_term(k): v for k, v in DEFAULT_LEGAL_DICT_EN_KN.items()}


legal_dict_en_kn = load_legal_dictionary()

# reverse dictionary for Kannada -> English translation
legal_dict_kn_en = {}
for en, kn_value in legal_dict_en_kn.items():
    for piece in re.split(r"[,;|/]", kn_value):
        kn_piece = _normalize_kn_term(piece)
        if kn_piece and kn_piece not in legal_dict_kn_en:
            legal_dict_kn_en[kn_piece] = en