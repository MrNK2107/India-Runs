from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

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
async def list_profiles(skip: int = 0, limit: int = 20) -> list[Profile]:
    if _profile_store is None:
        raise HTTPException(status_code=503, detail="Profile store not initialized")
    all_profiles = list(_profile_store.get_all_sample().values())
    return all_profiles[skip : skip + limit]
