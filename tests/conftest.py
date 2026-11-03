from __future__ import annotations

import hashlib
import os

# Keep the suite offline: importing gradio (src/ui) otherwise calls api.gradio.app.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import numpy as np
import pytest

from src.core.models import (
    Education,
    Location,
    PersonalInfo,
    ProfessionalInfo,
    ProficiencyLevel,
    Profile,
    ProfileSource,
    Skill,
    SkillCategory,
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
            Skill(name="Django", category=SkillCategory.FRAMEWORK,
                  proficiency=ProficiencyLevel.ADVANCED),
            Skill(name="AWS", category=SkillCategory.TOOL,
                  proficiency=ProficiencyLevel.INTERMEDIATE),
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


# ---------------------------------------------------------------------------
# Offline test doubles
#
# Unit tests must not download HuggingFace models or call paid LLM APIs.
# These autouse fixtures swap the sentence-transformers models for small
# deterministic fakes and make every LLM client unavailable, so agents take
# their rule-based fallback paths. Set INDIA_RUNS_REAL_MODELS=1 to opt out.
# ---------------------------------------------------------------------------

_REAL_MODELS = os.environ.get("INDIA_RUNS_REAL_MODELS") == "1"


class FakeSentenceTransformer:
    """Deterministic stand-in for ``SentenceTransformer`` (384-dim)."""

    def __init__(self, *args, **kwargs) -> None:
        self.dimension = 384

    def _vec(self, text: str) -> np.ndarray:
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        vec = np.random.default_rng(seed).normal(size=self.dimension).astype(np.float32)
        return vec / np.linalg.norm(vec)

    def encode(self, sentences, normalize_embeddings: bool = True, **kwargs):
        if isinstance(sentences, str):
            return self._vec(sentences)
        return np.stack([self._vec(s) for s in sentences])

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension


class FakeCrossEncoder:
    """Stand-in for ``CrossEncoder``: logit = token overlap between query and doc."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def predict(self, pairs, **kwargs):
        scores = []
        for query, doc in pairs:
            q = set(str(query).lower().split())
            d = set(str(doc).lower().split())
            scores.append(float(len(q & d)) - 1.0)
        return np.asarray(scores, dtype=np.float32)


def _no_llm(*args, **kwargs):
    raise RuntimeError("LLM clients are disabled in unit tests")


@pytest.fixture(autouse=True)
def _offline_models(monkeypatch):
    if _REAL_MODELS:
        yield
        return
    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(sentence_transformers, "CrossEncoder", FakeCrossEncoder)
    monkeypatch.setattr(
        "src.language.multilingual.SentenceTransformer", FakeSentenceTransformer
    )
    for target in (
        "src.core.config.get_llm_client",
        "src.agents.planner.get_llm_client",
        "src.agents.reflector.get_llm_client",
        "src.ranking.listwise_ranker.get_llm_client",
        "src.rationale.generator.get_llm_client",
    ):
        monkeypatch.setattr(target, _no_llm)
    yield
