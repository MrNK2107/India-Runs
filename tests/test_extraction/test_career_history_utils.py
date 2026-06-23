from __future__ import annotations

from datetime import date

from src.extraction.career_history_utils import _parse_date, compute_years_from_dates, latest_role


class TestLatestRole:
    def test_empty_history(self):
        assert latest_role([]) is None

    def test_prefers_current(self):
        history = [
            {"title": "Backend Engineer", "company": "Mindtree", "is_current": True,
             "start_date": "2024-01"},
            {"title": "Junior Dev", "company": "OldCo", "is_current": False,
             "start_date": "2022-01", "end_date": "2023-12"},
        ]
        role = latest_role(history)
        assert role["company"] == "Mindtree"

    def test_most_recent_when_no_current(self):
        history = [
            {"title": "Junior Dev", "company": "OldCo", "is_current": False,
             "start_date": "2022-01", "end_date": "2023-12"},
            {"title": "Intern", "company": "FirstCo", "is_current": False,
             "start_date": "2021-01", "end_date": "2021-12"},
        ]
        role = latest_role(history)
        assert role["company"] == "OldCo"


class TestComputeYearsFromDates:
    def test_simple_non_overlapping(self):
        history = [
            {"start_date": "2020-01", "end_date": "2022-01", "is_current": False},
            {"start_date": "2022-02", "end_date": "2024-02", "is_current": False},
        ]
        years, count, conf = compute_years_from_dates(history)
        assert count == 2
        assert conf == "high"
        assert years is not None and 3.9 < years < 4.1

    def test_current_role_uses_today(self):
        history = [
            {"start_date": "2022-01", "is_current": True},
        ]
        years, count, conf = compute_years_from_dates(history)
        assert count == 1
        assert conf == "high"
        assert years is not None and years > 0

    def test_missing_end_date_assumes_one_year(self):
        history = [
            {"start_date": "2023-01", "is_current": False},
        ]
        years, count, conf = compute_years_from_dates(history)
        assert count == 1
        assert conf == "low"
        assert years is not None and 0.9 < years < 1.1

    def test_future_date_capped_to_today(self):
        history = [
            {"start_date": "2020-01", "end_date": "2030-01", "is_current": False},
        ]
        years, count, conf = compute_years_from_dates(history)
        assert count == 1
        assert conf == "low"

    def test_swapped_dates_skipped(self):
        history = [
            {"start_date": "2024-01", "end_date": "2022-01", "is_current": False},
        ]
        years, count, conf = compute_years_from_dates(history)
        assert years is None

    def test_missing_start_date_skipped(self):
        history = [
            {"end_date": "2024-01", "is_current": False},
        ]
        years, count, conf = compute_years_from_dates(history)
        assert years is None

    def test_overlapping_interval_merged(self):
        history = [
            {"start_date": "2020-01", "end_date": "2023-06", "is_current": False},
            {"start_date": "2023-01", "end_date": "2024-01", "is_current": False},
        ]
        years, count, conf = compute_years_from_dates(history)
        assert count == 2
        assert years is not None and 3.9 < years < 4.1

    def test_empty_history_returns_none(self):
        years, count, conf = compute_years_from_dates([])
        assert years is None

    def test_low_confidence_when_mostly_assumptions(self):
        history = [
            {"start_date": "2020-01", "is_current": False},
            {"start_date": "2021-01", "end_date": "2022-01", "is_current": False},
        ]
        years, count, conf = compute_years_from_dates(history)
        assert conf == "medium"


class TestParseDate:
    def test_iso_date(self):
        assert _parse_date("2024-01-15") == date(2024, 1, 15)

    def test_year_month(self):
        assert _parse_date("2024-01") == date(2024, 1, 1)

    def test_year_only(self):
        assert _parse_date("2024") == date(2024, 1, 1)

    def test_invalid_returns_none(self):
        assert _parse_date("not-a-date") is None

    def test_none_returns_none(self):
        assert _parse_date(None) is None
