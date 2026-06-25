from __future__ import annotations

import numpy as np
import pytest

from src.core.models import (
    Education,
    Location,
    PersonalInfo,
    ProfessionalInfo,
    Profile,
    ProfileSource,
    Skill,
    SkillCategory,
    ProficiencyLevel,
    WorkExperience,
)


@pytest.fixture
def sample_profile() -> Profile:
    return Profile(
        profile_id="test-001",
        source=ProfileSource.MANUAL,
        raw_text="Senior Python developer with 5 years experience in Django and AWS",
        personal=PersonalInfo(
            name="Test User",
            location=Location(city="Bangalore", country="India"),
        ),
        professional=ProfessionalInfo(
            current_title="Senior Engineer",
            current_company="Acme Corp",
            total_experience_years=5.0,
        ),
        skills=[
            Skill(name="Python", category=SkillCategory.PROGRAMMING_LANGUAGE,
                  proficiency=ProficiencyLevel.ADVANCED),
            Skill(name="Django", category=SkillCategory.FRAMEWORK, proficiency=ProficiencyLevel.ADVANCED),
            Skill(name="AWS", category=SkillCategory.TOOL, proficiency=ProficiencyLevel.INTERMEDIATE),
        ],
        experience=[
            WorkExperience(title="Senior Engineer", company="Acme Corp",
                           start_date="2020-01", is_current=True),
        ],
        education=[
            Education(institution="IIT Bombay", degree="B.Tech", field="Computer Science"),
        ],
    )


@pytest.fixture
def sample_profiles() -> list[Profile]:
    return [
        Profile(
            profile_id=f"test-{i:03d}",
            source=ProfileSource.MANUAL,
            raw_text=f"Profile {i}",
            personal=PersonalInfo(name=f"User {i}"),
            professional=ProfessionalInfo(total_experience_years=float(i)),
            skills=[
                Skill(name="Python", category=SkillCategory.PROGRAMMING_LANGUAGE),
            ],
        )
        for i in range(1, 6)
    ]


@pytest.fixture
def sample_embeddings() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(size=(5, 384)).astype(np.float32)


@pytest.fixture
def sample_query_text() -> str:
    return "Find a senior Python developer with experience in Django and AWS"


@pytest.fixture
def vector_search_instance():
    from src.search.vector_search import VectorSearch
    return VectorSearch(dimension=384)


@pytest.fixture
def bm25_search_instance():
    from src.search.bm25_search import BM25Search
    return BM25Search()
