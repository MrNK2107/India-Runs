from __future__ import annotations

import numpy as np

from src.search.bm25_search import BM25Search
from src.search.filters import SearchFilter
from src.search.hybrid import HybridSearch
from src.search.vector_search import VectorSearch


def test_vector_build_and_search(vector_search_instance, sample_embeddings):
    vs = vector_search_instance
    profile_ids = ["a", "b", "c", "d", "e"]
    vs.build_index(sample_embeddings, profile_ids)
    assert vs.size == 5
    query_vec = sample_embeddings[0]
    results = vs.search(query_vec, top_k=3)
    assert len(results) == 3
    assert results[0][0] == "a"


def test_vector_save_load(tmp_path, vector_search_instance, sample_embeddings):
    vs = vector_search_instance
    vs.build_index(sample_embeddings, ["a", "b", "c", "d", "e"])
    index_path = tmp_path / "test_index.bin"
    id_map_path = tmp_path / "test_ids.json"
    vs.save(index_path, id_map_path)
    vs2 = VectorSearch()
    vs2.load(index_path, id_map_path)
    assert vs2.size == 5


def test_vector_empty_search(vector_search_instance):
    query_vec = np.zeros(384, dtype=np.float32)
    results = vector_search_instance.search(query_vec)
    assert results == []


def test_bm25_build_and_search(bm25_search_instance):
    bm = bm25_search_instance
    docs = ["python developer django", "java spring boot", "devops aws kubernetes"]
    ids = ["p1", "p2", "p3"]
    bm.build_index(docs, ids)
    assert bm.size == 3
    results = bm.search("python django", top_k=2)
    assert len(results) >= 1
    assert results[0][0] == "p1"


def test_bm25_save_load(tmp_path, bm25_search_instance):
    bm = bm25_search_instance
    bm.build_index(["python developer"], ["p1"])
    path = tmp_path / "bm25.pkl"
    bm.save(path)
    bm2 = BM25Search()
    bm2.load(path)
    assert bm2.size == 1


def test_bm25_tokenize(bm25_search_instance):
    tokens = bm25_search_instance._tokenize("Python Developer AWS")
    assert tokens == ["python", "developer", "aws"]


def test_hybrid_rrf():
    vs = VectorSearch()
    bm = BM25Search()
    from src.language.multilingual import MultilingualEmbedder
    embedder = MultilingualEmbedder()
    hybrid = HybridSearch(vs, bm, embedder)
    rankings = [
        [("a", 0.9), ("b", 0.8), ("c", 0.7)],
        [("b", 0.9), ("c", 0.8), ("a", 0.7)],
    ]
    fused = hybrid.reciprocal_rank_fusion(rankings, k=60)
    assert len(fused) == 3
    assert fused[0][0] == "b"


def test_filters_location(sample_profile):
    from src.core.models import SearchFilters
    sf = SearchFilter(SearchFilters(location="Bangalore"))
    assert sf.passes(sample_profile)
    sf2 = SearchFilter(SearchFilters(location="Mumbai"))
    assert not sf2.passes(sample_profile)


def test_filters_experience(sample_profile):
    from src.core.models import SearchFilters
    sf = SearchFilter(SearchFilters(min_experience_years=3, max_experience_years=10))
    assert sf.passes(sample_profile)
    sf2 = SearchFilter(SearchFilters(min_experience_years=10))
    assert not sf2.passes(sample_profile)
