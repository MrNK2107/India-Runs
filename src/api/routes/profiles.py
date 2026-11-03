from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

from src.core.models import Profile
from src.core.profile_store import ProfileStore

logger = logging.getLogger(__name__)

router = APIRouter()

_profile_store: ProfileStore | None = None


def init_profiles(profiles: ProfileStore) -> None:
    global _profile_store
    _profile_store = profiles


@router.get("/profiles/{profile_id}", response_model=Profile)
async def get_profile(profile_id: str) -> Profile:
    if _profile_store is None:
        raise HTTPException(status_code=503, detail="Profile store not initialized")
    profile = _profile_store.get(profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Profile not found: {profile_id}")
    return profile


@router.get("/profiles", response_model=list[Profile])
async def list_profiles(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
) -> list[Profile]:
    if _profile_store is None:
        raise HTTPException(status_code=503, detail="Profile store not initialized")
    # Sample profiles first, then profiles added via POST /ingest (an ingested
    # profile with a sample's id replaces it).
    merged = {**_profile_store.get_all_sample(), **_profile_store.get_ingested()}
    all_profiles = list(merged.values())
    return all_profiles[skip : skip + limit]
