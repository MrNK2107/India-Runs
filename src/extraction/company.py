from __future__ import annotations

from typing import Any

from src.extraction.career_history_utils import latest_role


def extract_company(
    prof: dict[str, Any],
    history: list[dict[str, Any]],
) -> tuple[str | None, str]:
    direct = prof.get("current_company")
    if direct and isinstance(direct, str) and direct.strip():
        return direct.strip(), "direct"

    latest = latest_role(history)
    if latest:
        company = latest.get("company")
        if company and isinstance(company, str) and company.strip():
            return company.strip(), "history"

    return None, "not_found"
