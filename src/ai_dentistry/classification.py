from __future__ import annotations

import re
from typing import Callable


def _build_pattern(terms: list[str]) -> re.Pattern[str]:
    escaped = [re.escape(term).replace(r"\ ", r"\s+") for term in terms]
    joined = "|".join(escaped)
    return re.compile(rf"\b({joined})\b", re.IGNORECASE)


def compile_category_patterns(classification_cfg: dict[str, list[str]]) -> dict[str, re.Pattern[str]]:
    return {
        "Dental": _build_pattern(classification_cfg.get("dental", [])),
        "Medical": _build_pattern(classification_cfg.get("medical", [])),
        "Technical": _build_pattern(classification_cfg.get("technical", [])),
    }


def classify_institution(name: str, patterns: dict[str, re.Pattern[str]]) -> str:
    for category in ("Dental", "Medical", "Technical"):
        if patterns[category].search(name):
            return category
    return "Other"


def make_classifier(patterns: dict[str, re.Pattern[str]]) -> Callable[[str], str]:
    def _classifier(name: str) -> str:
        return classify_institution(name, patterns)

    return _classifier
