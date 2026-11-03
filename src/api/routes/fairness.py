"""Fairness check for a ranked list: do candidates from tier-1 cities, top institutes or
English-language profiles score systematically higher than the rest of the same list?

Uses the existing BiasDetector (mean-score gap > 0.1 between groups is flagged; an empty
group is "insufficient data", never bias). It reports; it does not re-rank.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.api.routes import profiles as profiles_route
from src.core.models import MatchResult, MatchScores, Profile
from src.fairness.bias_detector import BiasDetector

router = APIRouter()

_DIMENSIONS = (
    ("location", "City tier", "Tier-1 cities vs tier-2/3"),
    ("university", "Institute", "IIT/NIT/BITS vs other institutes"),
    ("language", "Profile language", "English vs other languages"),
)


class FairnessItem(BaseModel):
    profile_id: str = Field(min_length=1, max_length=200)
    score: float = Field(ge=0.0, le=1.0)


class FairnessRequest(BaseModel):
    items: list[FairnessItem] = Field(min_length=1, max_length=100)


class FairnessDimension(BaseModel):
    dimension: str
    label: str
    comparison: str
    status: str  # "ok" | "flag" | "insufficient"
    gap: float
    observations: list[str]


class FairnessReport(BaseModel):
    candidates: int
    flagged: int
    dimensions: list[FairnessDimension]


@router.post("/fairness", response_model=FairnessReport)
async def fairness_check(request: FairnessRequest) -> FairnessReport:
    store = profiles_route._profile_store
    if store is None:
        raise HTTPException(status_code=503, detail="Profile store not initialized")
    profiles: dict[str, Profile] = {}
    matches: list[MatchResult] = []
    for i, item in enumerate(request.items, start=1):
        profile = store.get(item.profile_id)
        if profile is None:
            continue
        profiles[item.profile_id] = profile
        matches.append(MatchResult(query_id="", profile_id=item.profile_id, rank=i,
                                   scores=MatchScores(overall=item.score)))
    detector = BiasDetector()
    dims: list[FairnessDimension] = []
    for key, label, comparison in _DIMENSIONS:
        result = detector.detect_bias(matches, profiles, key)
        observations = [str(o) for o in result.get("observations", [])]
        insufficient = any("insufficient data" in o for o in observations) or not matches
        gap = float(result.get("details", {}).get("diff", 0.0) or 0.0)
        status = "flag" if result.get("detected") else "insufficient" if insufficient else "ok"
        dims.append(FairnessDimension(dimension=key, label=label, comparison=comparison,
                                      status=status, gap=round(gap, 4),
                                      observations=observations))
    return FairnessReport(candidates=len(matches),
                          flagged=sum(d.status == "flag" for d in dims), dimensions=dims)
