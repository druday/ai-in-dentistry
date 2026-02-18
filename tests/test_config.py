from ai_dentistry.config import Period, assign_period_label, resolve_periods


def test_assign_period_label() -> None:
    periods = [
        Period(label="1946-2003", start_year=1946, end_year=2003),
        Period(label="2004-2013", start_year=2004, end_year=2013),
        Period(label="2024-2025", start_year=2024, end_year=2025),
    ]
    assert assign_period_label(1950, periods) == "1946-2003"
    assert assign_period_label(2010, periods) == "2004-2013"
    assert assign_period_label(2025, periods) == "2024-2025"
    assert assign_period_label(2030, periods) is None


def test_resolve_periods_fixed_mode() -> None:
    protocol = {
        "temporal_segmentation": {
            "mode": "fixed",
            "fixed_periods": [
                {"label": "2000-2005", "start_year": 2000, "end_year": 2005},
                {"label": "2006-2010", "start_year": 2006, "end_year": 2010},
            ],
        }
    }
    periods = resolve_periods(protocol, publication_years=[2001, 2007])
    assert [p.label for p in periods] == ["2000-2005", "2006-2010"]


def test_resolve_periods_balanced_mode() -> None:
    years = [2000] * 4 + [2001] * 3 + [2002] * 2 + [2003] * 4 + [2004] * 3 + [2005] * 2
    protocol = {
        "temporal_segmentation": {
            "mode": "balanced",
            "dynamic": {
                "n_bins": 3,
                "min_year": 2000,
                "max_year": 2005,
                "min_years_per_bin": 1,
                "use_observed_year_range": True,
            },
        }
    }
    periods = resolve_periods(protocol, publication_years=years)
    assert len(periods) == 3
    assert periods[0].start_year == 2000
    assert periods[-1].end_year == 2005
