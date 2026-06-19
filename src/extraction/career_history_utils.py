from __future__ import annotations

from datetime import date, datetime
from typing import Any

_LOW_CONFIDENCE_THRESHOLD = 0.5


def latest_role(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not history:
        return None
    current = [r for r in history if r.get("is_current")]
    if current:
        return current[0]
    return max(
        history,
        key=lambda r: _parse_date_for_sort(r.get("start_date")) or date.min,
    )


def compute_years_from_dates(
    history: list[dict[str, Any]],
) -> tuple[float | None, int, str]:
    today = date.today()
    total_days = 0.0
    valid_entries = 0
    assumptions = 0

    intervals: list[tuple[date, date]] = []
    for entry in history:
        start = _parse_date(entry.get("start_date"))
        if start is None:
            continue
        end_raw = entry.get("end_date")
        is_current = entry.get("is_current", False)
        if end_raw:
            end = _parse_date(end_raw)
            if end is None:
                continue
            if end > today:
                end = today
                assumptions += 1
            if end <= start:
                continue
        elif is_current:
            end = today
        else:
            end = date(start.year + 1, start.month, start.day)
            assumptions += 1
        intervals.append((start, end))
        valid_entries += 1

    if not intervals:
        return None, 0, "low"

    merged = _merge_intervals(intervals)
    for s, e in merged:
        total_days += (e - s).days

    total_years = round(total_days / 365.25, 1)

    if valid_entries == 0:
        return None, 0, "low"
    ratio = assumptions / valid_entries
    if ratio > _LOW_CONFIDENCE_THRESHOLD:
        confidence = "low"
    elif ratio > 0:
        confidence = "medium"
    else:
        confidence = "high"

    return total_years, valid_entries, confidence


def _parse_date(raw: str | None) -> date | None:
    if not raw:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y/%m/%d", "%Y/%m", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(raw.strip(), fmt).date()
        except ValueError:
            continue
    try:
        return datetime.strptime(raw.strip()[:10], "%Y-%m-%d").date()
    except (ValueError, IndexError):
        pass
    try:
        year = int(raw.strip()[:4])
        return date(year, 1, 1)
    except (ValueError, IndexError):
        return None


def _parse_date_for_sort(raw: str | None) -> date | None:
    d = _parse_date(raw)
    if d is not None:
        return d
    if raw:
        try:
            year = int(raw.strip()[:4])
            return date(year, 1, 1)
        except (ValueError, IndexError):
            pass
    return None


def _merge_intervals(intervals: list[tuple[date, date]]) -> list[tuple[date, date]]:
    sorted_iv = sorted(intervals, key=lambda x: x[0])
    merged: list[tuple[date, date]] = []
    for start, end in sorted_iv:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged
