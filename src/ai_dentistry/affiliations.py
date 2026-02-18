from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
from dateutil import parser

from ai_dentistry.geography import extract_country_from_affiliation

INSTITUTION_EXTRACTION_MODE_EXACT = "exact_affiliation"
INSTITUTION_EXTRACTION_MODE_KEYWORD = "keyword_hint_segment"

MEDLINE_RENAME_MAP = {
    "AB": "Abstract",
    "AD": "Affiliation",
    "AU": "Author",
    "FAU": "Full Author",
    "DP": "Date of Publication",
    "PMID": "PubMed Unique Identifier",
    "PL": "Place of Publication",
    "PT": "Publication Type",
    "TI": "Title",
    "JT": "Journal Title",
    "EDAT": "Entrez Date",
    "LR": "Date Last Revised",
}


def read_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    path_obj = Path(path)
    rows: list[dict[str, Any]] = []
    with path_obj.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def approximate_publication_date(value: Any) -> pd.Timestamp | pd.NaT:
    if value is None:
        return pd.NaT
    text = str(value).strip()
    if not text:
        return pd.NaT
    text = re.sub(r"([A-Za-z]{3,9})-[A-Za-z]{3,9}", r"\1", text)
    if re.fullmatch(r"\d{4}", text):
        text = f"{text} Jan"
    try:
        return pd.Timestamp(parser.parse(text, default=parser.parse("1900-01-01")))
    except Exception:
        return pd.NaT


def _ascii_fold(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def normalize_affiliation_text(text: str, remove_punctuation: bool = True) -> str:
    value = _ascii_fold(text).lower().strip()
    if remove_punctuation:
        value = re.sub(r"[^a-z0-9\\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def extract_institution_from_affiliation(
    affiliation: str,
    keyword_hints: list[str],
    remove_punctuation: bool = True,
    extraction_mode: str = INSTITUTION_EXTRACTION_MODE_EXACT,
) -> str:
    raw = str(affiliation).strip()
    if not raw:
        return ""

    if extraction_mode == INSTITUTION_EXTRACTION_MODE_EXACT:
        return raw
    if extraction_mode != INSTITUTION_EXTRACTION_MODE_KEYWORD:
        raise ValueError(
            f"Unsupported institution extraction mode '{extraction_mode}'. "
            f"Expected '{INSTITUTION_EXTRACTION_MODE_EXACT}' or '{INSTITUTION_EXTRACTION_MODE_KEYWORD}'."
        )

    pieces = [p.strip() for p in re.split(r"[;,]", raw) if p.strip()]
    if not pieces:
        return ""

    lowered_hints = tuple(h.lower() for h in keyword_hints)
    selected = ""
    for piece in pieces:
        candidate = piece.lower()
        if any(h in candidate for h in lowered_hints):
            selected = piece
            break
    if not selected:
        selected = pieces[0]

    return normalize_affiliation_text(selected, remove_punctuation=remove_punctuation)


def extract_affiliation_entries(
    affiliation_field: Any,
    keyword_hints: list[str],
    remove_punctuation: bool = True,
    extraction_mode: str = INSTITUTION_EXTRACTION_MODE_EXACT,
) -> list[dict[str, str]]:
    raw_affiliations: list[str] = []
    if isinstance(affiliation_field, list):
        raw_affiliations = [str(v) for v in affiliation_field if isinstance(v, str) and v.strip()]
    elif isinstance(affiliation_field, str) and affiliation_field.strip():
        raw_affiliations = [affiliation_field]

    entries: list[dict[str, str]] = []
    for raw in raw_affiliations:
        institution = extract_institution_from_affiliation(
            raw,
            keyword_hints=keyword_hints,
            remove_punctuation=remove_punctuation,
            extraction_mode=extraction_mode,
        )
        if not institution:
            continue
        country = extract_country_from_affiliation(raw)
        entries.append({"institution": institution, "country": country or ""})

    unique_pairs = dedupe_preserve_order(
        [f"{entry['institution']}|||{entry['country']}" for entry in entries]
    )
    return [
        {"institution": item.split("|||", 1)[0], "country": item.split("|||", 1)[1]}
        for item in unique_pairs
    ]


def transform_records_to_publications(
    records: list[dict[str, Any]],
    keyword_hints: list[str],
    remove_punctuation: bool = True,
    extraction_mode: str = INSTITUTION_EXTRACTION_MODE_EXACT,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in records:
        date_value = record.get("DP")
        pub_date = approximate_publication_date(date_value)
        pub_year = int(pub_date.year) if not pd.isna(pub_date) else None

        entries = extract_affiliation_entries(
            record.get("AD"),
            keyword_hints=keyword_hints,
            remove_punctuation=remove_punctuation,
            extraction_mode=extraction_mode,
        )
        institutions = dedupe_preserve_order([e["institution"] for e in entries if e["institution"]])
        countries = dedupe_preserve_order([e["country"] for e in entries if e["country"]])

        if not institutions:
            continue

        output.append(
            {
                "pmid": str(record.get("PMID", "")).strip(),
                "title": str(record.get("TI", "")).strip(),
                "publication_date_raw": str(date_value or ""),
                "publication_year": pub_year,
                "affiliation_raw": record.get("AD"),
                "institutions": institutions,
                "countries": countries,
                "institution_country_pairs": entries,
            }
        )
    return output


def deduplicate_publications(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen_pmids: set[str] = set()
    seen_title_year: set[tuple[str, int | None]] = set()

    for row in rows:
        pmid = row.get("pmid", "")
        title = row.get("title", "")
        year = row.get("publication_year")
        if pmid:
            if pmid in seen_pmids:
                continue
            seen_pmids.add(pmid)
        else:
            key = (title, year)
            if key in seen_title_year:
                continue
            seen_title_year.add(key)
        deduped.append(row)
    return deduped
