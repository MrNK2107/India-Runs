from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from src.core.models import Profile

logger = logging.getLogger(__name__)

router = APIRouter()

_profiles_store: dict[str, Profile] = {}


def init_profiles(profiles: dict[str, Profile]) -> None:
    global _profiles_store
    _profiles_store = profiles


@router.get("/profiles/{profile_id}", response_model=Profile)
async def get_profile(profile_id: str) -> Profile:
    profile = _profiles_store.get(profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Profile not found: {profile_id}")
    return profile


@router.get("/profiles", response_model=list[Profile])
async def list_profiles(skip: int = 0, limit: int = 20) -> list[Profile]:
    all_profiles = list(_profiles_store.values())
    return all_profiles[skip : skip + limit]
