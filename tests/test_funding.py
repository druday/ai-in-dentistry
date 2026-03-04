from ai_dentistry.funding import (
    FUNDING_NONE,
    FUNDING_NON_US,
    FUNDING_US_FEDERAL,
    classify_funding,
    extract_grant_entries,
)


def test_extract_grant_entries_parses_medline_gr_field() -> None:
    record = {
        "GR": [
            "R01 DE000001/DE/NIDCR NIH HHS/United States",
            "81400550/National Natural Science Foundation of China/",
        ]
    }
    entries = extract_grant_entries(record)
    assert len(entries) == 2
    assert entries[0]["grant_id"] == "R01 DE000001"
    assert entries[0]["agency"] == "NIDCR NIH HHS"
    assert entries[0]["country"] == "United States"


def test_classify_funding_us_federal_only() -> None:
    grants = [
        {
            "raw": "R01 DE000001/DE/NIDCR NIH HHS/United States",
            "grant_id": "R01 DE000001",
            "agency": "NIDCR NIH HHS",
            "country": "United States",
        }
    ]
    category, flags = classify_funding(grants)
    assert category == FUNDING_US_FEDERAL
    assert flags.has_grant_support
    assert flags.has_us_federal_signal
    assert not flags.has_non_us_signal


def test_classify_funding_non_us_only() -> None:
    grants = [
        {
            "raw": "81400550/National Natural Science Foundation of China/",
            "grant_id": "81400550",
            "agency": "National Natural Science Foundation of China",
            "country": "",
        }
    ]
    category, flags = classify_funding(grants)
    assert category == FUNDING_NON_US
    assert flags.has_grant_support
    assert not flags.has_us_federal_signal
    assert flags.has_non_us_signal


def test_classify_funding_mixed_hierarchical_to_us() -> None:
    grants = [
        {
            "raw": "R01 DE000001/DE/NIDCR NIH HHS/United States",
            "grant_id": "R01 DE000001",
            "agency": "NIDCR NIH HHS",
            "country": "United States",
        },
        {
            "raw": "81400550/National Natural Science Foundation of China/",
            "grant_id": "81400550",
            "agency": "National Natural Science Foundation of China",
            "country": "",
        },
    ]
    category, flags = classify_funding(grants)
    assert category == FUNDING_US_FEDERAL
    assert flags.has_mixed_us_non_us


def test_classify_funding_no_grant() -> None:
    category, flags = classify_funding([])
    assert category == FUNDING_NONE
    assert not flags.has_grant_support

