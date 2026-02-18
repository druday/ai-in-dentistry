from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


@dataclass(frozen=True)
class Period:
    label: str
    start_year: int
    end_year: int

    def contains(self, year: int | None) -> bool:
        if year is None:
            return False
        return self.start_year <= year <= self.end_year


def load_protocol(path: str | Path) -> dict[str, Any]:
    protocol_path = Path(path)
    if not protocol_path.exists():
        raise FileNotFoundError(f"Protocol file not found: {protocol_path}")

    with protocol_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError("Protocol file must parse to a mapping at the top level.")

    required_keys = {"queries", "classification", "preprocessing", "outputs"}
    missing = required_keys - set(data.keys())
    if missing:
        raise ValueError(f"Protocol missing required keys: {sorted(missing)}")

    if "periods" not in data and "temporal_segmentation" not in data:
        raise ValueError("Protocol must define either 'periods' or 'temporal_segmentation'.")

    return data


def _rows_to_periods(rows: Iterable[dict[str, Any]]) -> list[Period]:
    periods: list[Period] = []
    for row in rows:
        periods.append(
            Period(
                label=str(row["label"]),
                start_year=int(row["start_year"]),
                end_year=int(row["end_year"]),
            )
        )
    return periods


def load_periods(protocol: dict[str, Any]) -> list[Period]:
    return _rows_to_periods(protocol.get("periods", []))


def assign_period_label(year: int | None, periods: list[Period]) -> str | None:
    for period in periods:
        if period.contains(year):
            return period.label
    return None


def _clean_publication_years(values: Iterable[Any]) -> list[int]:
    clean: list[int] = []
    for value in values:
        if value is None:
            continue
        try:
            if value != value:  # NaN check without pandas dependency
                continue
        except Exception:
            pass
        try:
            clean.append(int(value))
        except (TypeError, ValueError):
            continue
    return clean


def _normalize_year_bounds(
    years: list[int],
    min_year: int | None,
    max_year: int | None,
    use_observed_year_range: bool,
) -> tuple[int, int]:
    observed_min = min(years)
    observed_max = max(years)

    if use_observed_year_range:
        start = observed_min if min_year is None else max(min_year, observed_min)
        end = observed_max if max_year is None else min(max_year, observed_max)
    else:
        start = observed_min if min_year is None else min_year
        end = observed_max if max_year is None else max_year

    if end < start:
        raise ValueError(
            f"Invalid year bounds after applying constraints: start={start}, end={end}."
        )
    return start, end


def _balanced_period_boundaries(
    counts: list[int],
    n_bins: int,
    min_years_per_bin: int,
) -> list[tuple[int, int]]:
    n_years = len(counts)
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1.")
    if min_years_per_bin < 1:
        raise ValueError("min_years_per_bin must be >= 1.")
    if n_bins * min_years_per_bin > n_years:
        raise ValueError(
            f"Cannot split {n_years} years into {n_bins} bins with "
            f"min_years_per_bin={min_years_per_bin}."
        )

    prefix = [0]
    for value in counts:
        prefix.append(prefix[-1] + value)

    total = prefix[-1]
    target = total / n_bins

    inf = float("inf")
    dp = [[inf] * (n_years + 1) for _ in range(n_bins + 1)]
    back: list[list[int | None]] = [[None] * (n_years + 1) for _ in range(n_bins + 1)]
    dp[0][0] = 0.0

    def segment_cost(start_idx: int, end_idx: int) -> float:
        # end_idx is exclusive.
        segment_total = prefix[end_idx] - prefix[start_idx]
        return float((segment_total - target) ** 2)

    for k in range(1, n_bins + 1):
        j_min = k * min_years_per_bin
        j_max = n_years - (n_bins - k) * min_years_per_bin
        for j in range(j_min, j_max + 1):
            i_min = (k - 1) * min_years_per_bin
            i_max = j - min_years_per_bin
            best_val = inf
            best_i: int | None = None
            for i in range(i_min, i_max + 1):
                if dp[k - 1][i] == inf:
                    continue
                candidate = dp[k - 1][i] + segment_cost(i, j)
                if candidate < best_val:
                    best_val = candidate
                    best_i = i
            dp[k][j] = best_val
            back[k][j] = best_i

    if dp[n_bins][n_years] == inf:
        raise ValueError("Unable to compute balanced period boundaries for provided settings.")

    segments: list[tuple[int, int]] = []
    k = n_bins
    j = n_years
    while k > 0:
        i = back[k][j]
        if i is None:
            raise ValueError("Backtracking failed while computing balanced periods.")
        segments.append((i, j))
        j = i
        k -= 1
    segments.reverse()
    return segments


def build_balanced_periods(
    publication_years: Iterable[Any],
    n_bins: int,
    min_year: int | None = None,
    max_year: int | None = None,
    min_years_per_bin: int = 1,
    use_observed_year_range: bool = True,
) -> list[Period]:
    clean_years = _clean_publication_years(publication_years)
    if not clean_years:
        raise ValueError("Cannot build balanced periods: no valid publication years available.")

    start_year, end_year = _normalize_year_bounds(
        years=clean_years,
        min_year=min_year,
        max_year=max_year,
        use_observed_year_range=use_observed_year_range,
    )

    bounded = [year for year in clean_years if start_year <= year <= end_year]
    if not bounded:
        raise ValueError(
            f"No publication years found inside configured dynamic bounds: {start_year}-{end_year}."
        )

    year_axis = list(range(start_year, end_year + 1))
    year_counts = Counter(bounded)
    counts = [int(year_counts.get(year, 0)) for year in year_axis]

    segments = _balanced_period_boundaries(
        counts=counts,
        n_bins=n_bins,
        min_years_per_bin=min_years_per_bin,
    )

    periods: list[Period] = []
    for start_idx, end_idx in segments:
        seg_start = year_axis[start_idx]
        seg_end = year_axis[end_idx - 1]
        periods.append(
            Period(
                label=f"{seg_start}-{seg_end}",
                start_year=seg_start,
                end_year=seg_end,
            )
        )
    return periods


def resolve_periods(
    protocol: dict[str, Any],
    publication_years: Iterable[Any],
    mode_override: str | None = None,
) -> list[Period]:
    segmentation = protocol.get("temporal_segmentation")
    if not segmentation:
        mode = (mode_override or "fixed").lower()
        if mode != "fixed":
            raise ValueError(
                "Dynamic period mode requested, but 'temporal_segmentation' is missing in protocol."
            )
        return load_periods(protocol)

    mode = (mode_override or segmentation.get("mode", "fixed")).lower()
    if mode == "fixed":
        fixed_rows = segmentation.get("fixed_periods") or protocol.get("periods", [])
        return _rows_to_periods(fixed_rows)

    if mode in {"balanced", "dynamic"}:
        dynamic_cfg = segmentation.get("dynamic", {})
        return build_balanced_periods(
            publication_years=publication_years,
            n_bins=int(dynamic_cfg.get("n_bins", 6)),
            min_year=dynamic_cfg.get("min_year"),
            max_year=dynamic_cfg.get("max_year"),
            min_years_per_bin=int(dynamic_cfg.get("min_years_per_bin", 1)),
            use_observed_year_range=bool(dynamic_cfg.get("use_observed_year_range", True)),
        )

    raise ValueError(f"Unsupported temporal segmentation mode: {mode}")


def periods_to_rows(periods: list[Period]) -> list[dict[str, Any]]:
    return [
        {
            "label": period.label,
            "start_year": period.start_year,
            "end_year": period.end_year,
        }
        for period in periods
    ]
