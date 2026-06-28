from __future__ import annotations

from src.core.models import Profile, SearchFilters


class SearchFilter:
    def __init__(self, filters: SearchFilters) -> None:
        self.filters = filters

    def passes(self, profile: Profile) -> bool:
        if not self._check_location(profile):
            return False
        if not self._check_experience(profile):
            return False
        if not self._check_companies(profile):
            return False
        return True

    def filter_profiles(self, profiles: list[Profile]) -> list[Profile]:
        return [p for p in profiles if self.passes(p)]

    def _check_location(self, profile: Profile) -> bool:
        if self.filters.remote_ok:
            is_candidate_remote = False
            if profile.personal and profile.personal.location:
                is_candidate_remote = (
                    bool(profile.personal.location.is_remote_ok) or
                    (profile.personal.location.city and profile.personal.location.city.lower().strip() == "remote")
                )
            if not is_candidate_remote:
                return False

        if not self.filters.location:
            return True

        location_filter = self.filters.location.lower()
        city = profile.personal.location.city if profile.personal and profile.personal.location else None
        country = profile.personal.location.country if profile.personal and profile.personal.location else None
        if location_filter in ("remote", "anywhere"):
            return True
        if city and city.lower() == location_filter:
            return True
        if country and country.lower() == location_filter:
            return True
        for exp in profile.experience:
            if exp.location and exp.location.lower().strip() == location_filter:
                return True
        if self.filters.remote_ok and profile.personal and profile.personal.location and profile.personal.location.is_remote_ok:
            return True
        return False

    def _check_experience(self, profile: Profile) -> bool:
        total = profile.professional.total_experience_years
        min_yrs = self.filters.min_experience_years
        max_yrs = self.filters.max_experience_years
        if total is None:
            if min_yrs is not None and min_yrs > 0:
                return False
            return True
        if min_yrs is not None and total < min_yrs:
            return False
        if max_yrs is not None and total > max_yrs:
            return False
        return True

    def _check_companies(self, profile: Profile) -> bool:
        if self.filters.include_companies:
            company_names = {exp.company.lower() for exp in profile.experience}
            if not any(c.lower() in company_names for c in self.filters.include_companies):
                return False
        if self.filters.exclude_companies:
            company_names = {exp.company.lower() for exp in profile.experience}
            if any(c.lower() in company_names for c in self.filters.exclude_companies):
                return False
        return True
