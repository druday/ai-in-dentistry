from ai_dentistry.classification import classify_institution, compile_category_patterns


def test_classification_patterns_detect_expected_categories() -> None:
    cfg = {
        "dental": ["periodontology", "implantology"],
        "medical": ["life science", "pediatrics"],
        "technical": ["bioengineering", "computer science"],
    }
    patterns = compile_category_patterns(cfg)

    assert classify_institution("department of periodontology and implantology", patterns) == "Dental"
    assert classify_institution("center for life science pediatrics", patterns) == "Medical"
    assert classify_institution("school of bioengineering and computer science", patterns) == "Technical"
    assert classify_institution("school of humanities", patterns) == "Other"
