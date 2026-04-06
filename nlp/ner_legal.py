"""Simple Named Entity Recognition for legal documents."""

import re
from typing import List, Tuple


class LegalNER:
    """Extract legal entities from text."""

    # Common legal entity patterns
    ACT_PATTERN = re.compile(r"\b(\w+\s+Act(?:\s+of\s+\d+)?)\b", re.IGNORECASE)
    SECTION_PATTERN = re.compile(r"\bSection\s+(\d+[A-Z]?)", re.IGNORECASE)
    ARTICLE_PATTERN = re.compile(r"\bArticle\s+(\d+[A-Z]?)", re.IGNORECASE)
    COURT_PATTERN = re.compile(r"\b(High Court|Supreme Court|District Court|Magistrate Court)\b", re.IGNORECASE)
    CASE_PATTERN = re.compile(r"\b([A-Z][a-z\s.]+)\s+v[s.]?\s+([A-Z][a-z\s.]+)\b")
    YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")

    @classmethod
    def extract_acts(cls, text: str) -> List[str]:
        """Extract Act references."""
        return cls.ACT_PATTERN.findall(text)

    @classmethod
    def extract_sections(cls, text: str) -> List[str]:
        """Extract Section references."""
        return cls.SECTION_PATTERN.findall(text)

    @classmethod
    def extract_articles(cls, text: str) -> List[str]:
        """Extract Article references."""
        return cls.ARTICLE_PATTERN.findall(text)

    @classmethod
    def extract_courts(cls, text: str) -> List[str]:
        """Extract court references."""
        return cls.COURT_PATTERN.findall(text)

    @classmethod
    def extract_cases(cls, text: str) -> List[Tuple[str, str]]:
        """Extract case references (v/vs pattern)."""
        return cls.CASE_PATTERN.findall(text)

    @classmethod
    def extract_years(cls, text: str) -> List[str]:
        """Extract year references."""
        return cls.YEAR_PATTERN.findall(text)

    @classmethod
    def extract_all(cls, text: str) -> dict:
        """Extract all legal entities."""
        return {
            "acts": cls.extract_acts(text),
            "sections": cls.extract_sections(text),
            "articles": cls.extract_articles(text),
            "courts": cls.extract_courts(text),
            "cases": cls.extract_cases(text),
            "years": cls.extract_years(text),
        }


def highlight_entities(text: str) -> dict:
    """Extract and return entity information from legal text."""
    entities = LegalNER.extract_all(text)
    return {
        "text": text,
        "entities": entities,
        "entity_count": sum(len(v) if isinstance(v, list) else len(v) for v in entities.values()),
    }
