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
        if not self.filters.location:
            return True
        location_filter = self.filters.location.lower()
        city = profile.personal.location.city
        country = profile.personal.location.country
        if city and location_filter in city.lower():
            return True
        if country and location_filter in country.lower():
            return True
        for exp in profile.experience:
            if exp.location and location_filter in exp.location.lower():
                return True
        if self.filters.remote_ok and profile.personal.location.is_remote_ok:
            return True
        return False

    def _check_experience(self, profile: Profile) -> bool:
        total = profile.professional.total_experience_years
        if total is None:
            return True
        min_yrs = self.filters.min_experience_years
        max_yrs = self.filters.max_experience_years
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
