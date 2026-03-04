from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


FUNDING_US_FEDERAL = "US_FEDERAL_FUNDED"
FUNDING_NON_US = "NON_US_OR_INTL_FUNDED"
FUNDING_NONE = "NO_GRANT_LISTED"

FUNDING_OUTPUT_ORDER = [
    FUNDING_US_FEDERAL,
    FUNDING_NON_US,
    FUNDING_NONE,
]

_DEFAULT_US_FEDERAL_KEYWORDS = [
    "nih",
    "national institutes of health",
    "hhs",
    "ninds",
    "niaid",
    "nci",
    "nibib",
    "nidcr",
    "nigms",
    "nhlbi",
    "nida",
    "nccih",
    "cdc",
    "centers for disease control",
    "nsf",
    "national science foundation",
    "ahrq",
    "agency for healthcare research and quality",
    "hrsa",
    "health resources and services administration",
    "dod",
    "department of defense",
    "darpa",
    "va",
    "veterans affairs",
    "usda",
    "department of agriculture",
    "department of energy",
    "doe",
    "nasa",
]

_US_TEXT_HINTS = [
    "united states",
    "u.s.",
    "usa",
    "federal",
]


@dataclass(frozen=True)
class FundingFlags:
    has_grant_support: bool
    has_us_federal_signal: bool
    has_non_us_signal: bool
    has_mixed_us_non_us: bool

    def as_dict(self) -> dict[str, bool]:
        return {
            "has_grant_support": self.has_grant_support,
            "has_us_federal_signal": self.has_us_federal_signal,
            "has_non_us_signal": self.has_non_us_signal,
            "has_mixed_us_non_us": self.has_mixed_us_non_us,
        }


def default_funding_config() -> dict[str, Any]:
    return {
        "enabled": True,
        "classification_policy": "hierarchical_3_level",
        "us_federal_keywords": list(_DEFAULT_US_FEDERAL_KEYWORDS),
        "output_categories": list(FUNDING_OUTPUT_ORDER),
    }


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _normalize_grant_entry(raw: str) -> str:
    return _normalize_spaces(str(raw).replace("\n", " "))


def extract_grant_entries(record: dict[str, Any]) -> list[dict[str, str]]:
    raw_value = record.get("GR")
    values: list[str] = []
    if isinstance(raw_value, list):
        values = [str(item) for item in raw_value if str(item).strip()]
    elif isinstance(raw_value, str) and raw_value.strip():
        values = [raw_value]

    entries: list[dict[str, str]] = []
    for raw in values:
        normalized = _normalize_grant_entry(raw)
        if not normalized:
            continue

        parts = [part.strip() for part in normalized.split("/") if part.strip()]
        grant_id = parts[0] if parts else ""
        agency = ""
        country = ""
        if len(parts) >= 4:
            agency = parts[-2]
            country = parts[-1]
        elif len(parts) == 3:
            agency = parts[-2]
            country = parts[-1]
        elif len(parts) == 2:
            agency = parts[-1]

        entries.append(
            {
                "raw": normalized,
                "grant_id": grant_id,
                "agency": agency,
                "country": country,
            }
        )

    return entries


def _is_us_federal(entry: dict[str, str], keywords: list[str]) -> bool:
    text = " ".join(
        [
            str(entry.get("raw", "")),
            str(entry.get("agency", "")),
            str(entry.get("country", "")),
        ]
    ).lower()
    if any(keyword.lower() in text for keyword in keywords):
        return True
    # Handle sparse records where "United States" appears but agency tokens are partial.
    if any(hint in text for hint in _US_TEXT_HINTS) and (
        "nih" in text or "hhs" in text or "cdc" in text or "nsf" in text
    ):
        return True
    return False


def classify_funding(
    grant_entries: list[dict[str, str]],
    policy_cfg: dict[str, Any] | None = None,
) -> tuple[str, FundingFlags]:
    cfg = {**default_funding_config(), **(policy_cfg or {})}
    policy = str(cfg.get("classification_policy", "hierarchical_3_level")).strip().lower()
    if policy not in {"hierarchical_3_level"}:
        raise ValueError(
            f"Unsupported funding classification_policy '{policy}'. "
            "Supported values: hierarchical_3_level."
        )

    keywords = [str(value).strip().lower() for value in cfg.get("us_federal_keywords", []) if str(value).strip()]
    if not keywords:
        keywords = list(_DEFAULT_US_FEDERAL_KEYWORDS)

    has_grant_support = len(grant_entries) > 0
    has_us = False
    has_non_us = False
    for entry in grant_entries:
        if _is_us_federal(entry, keywords):
            has_us = True
        else:
            has_non_us = True

    flags = FundingFlags(
        has_grant_support=has_grant_support,
        has_us_federal_signal=has_us,
        has_non_us_signal=has_non_us,
        has_mixed_us_non_us=has_us and has_non_us,
    )

    if not has_grant_support:
        return FUNDING_NONE, flags
    if has_us:
        return FUNDING_US_FEDERAL, flags
    return FUNDING_NON_US, flags
