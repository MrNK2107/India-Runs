from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import src.core.profile_store as ps_module
from src.core.profile_store import ProfileStore


def _raw(pid: str, name: str = "Ana Rao", key: str = "candidate_id") -> dict[str, Any]:
    return {
        key: pid,
        "profile": {"anonymized_name": name, "current_title": "Engineer", "location": "Pune"},
        "career_history": [],
    }


def _write_jsonl(path: Path, rows: list[Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write((row if isinstance(row, str) else json.dumps(row, ensure_ascii=False)) + "\n")


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate the store from the repo's real data/ directory."""
    monkeypatch.delenv("PROFILES_PATH", raising=False)
    monkeypatch.setattr(ps_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(ps_module, "SAMPLE_PATH", tmp_path / "samples" / "sample.json")
    (tmp_path / "samples").mkdir()
    (tmp_path / "indexes").mkdir()
    return tmp_path


def _write_sample(data_dir: Path, rows: list[dict[str, Any]]) -> None:
    (data_dir / "samples" / "sample.json").write_text(
        json.dumps(rows, ensure_ascii=False), encoding="utf-8"
    )


def test_empty_store(data_dir: Path) -> None:
    store = ProfileStore(jsonl_path=data_dir / "missing.jsonl")
    assert len(store) == 0
    assert store.get("x") is None
    assert "x" not in store
    assert store.get_all_pids() == []
    assert store.get_all_sample() == {}


def test_path_autodetect_prefers_candidates_then_sample_500(data_dir: Path) -> None:
    assert ProfileStore().path == data_dir / "samples" / "sample_candidates.json"
    (data_dir / "samples" / "sample_500.jsonl").write_text("", encoding="utf-8")
    assert ProfileStore().path == data_dir / "samples" / "sample_500.jsonl"
    (data_dir / "profiles").mkdir()
    (data_dir / "profiles" / "candidates.jsonl").write_text("", encoding="utf-8")
    assert ProfileStore().path == data_dir / "profiles" / "candidates.jsonl"


def test_path_env_override(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROFILES_PATH", str(data_dir / "custom.jsonl"))
    assert ProfileStore().path == data_dir / "custom.jsonl"
    # An explicit argument still wins over the env var.
    assert ProfileStore(jsonl_path=data_dir / "x.jsonl").path == data_dir / "x.jsonl"


def test_lazy_jsonl_lookup_by_offset(data_dir: Path) -> None:
    path = data_dir / "c.jsonl"
    _write_jsonl(path, [_raw("A1", "Asha"), "", _raw("B2", "Bilal", key="id"), _raw("C3")])
    store = ProfileStore(jsonl_path=path)
    b = store.get("B2")
    assert b is not None and b.personal.name == "Bilal"
    a = store.get("A1")
    assert a is not None and a.personal.name == "Asha"
    assert store.get("ZZ") is None
    assert "C3" in store
    assert len(store) == 3
    assert store.get_all_pids() == ["A1", "B2", "C3"]


def test_extract_id_variants() -> None:
    assert ProfileStore._extract_id({"profile_id": "p"}) == "p"
    assert ProfileStore._extract_id({"candidate_id": 12}) == "12"
    assert ProfileStore._extract_id({"profile": {"id": "nested"}}) == "nested"
    assert ProfileStore._extract_id({"profile": "not-a-dict"}) is None
    assert ProfileStore._extract_id({}) is None


def test_bad_lines_are_skipped(data_dir: Path) -> None:
    # Regression: a valid-JSON line that isn't an object (e.g. `[1, 2]`)
    # raised AttributeError in _extract_id and aborted the whole index build.
    path = data_dir / "c.jsonl"
    _write_jsonl(path, ["{broken", "[1, 2]", "42", _raw("OK1")])
    store = ProfileStore(jsonl_path=path)
    assert store.get_all_pids() == ["OK1"]
    assert store.get("OK1") is not None


def test_duplicate_ids_keep_first(data_dir: Path) -> None:
    path = data_dir / "c.jsonl"
    _write_jsonl(path, [_raw("D", "First"), _raw("D", "Second")])
    p = ProfileStore(jsonl_path=path).get("D")
    assert p is not None and p.personal.name == "First"


def test_lru_cache_evicts_oldest(data_dir: Path) -> None:
    path = data_dir / "c.jsonl"
    _write_jsonl(path, [_raw(f"P{i}") for i in range(3)])
    store = ProfileStore(jsonl_path=path, max_cache=2)
    store.get("P0")
    store.get("P1")
    store.get("P0")  # P0 becomes most recent
    store.get("P2")  # evicts P1
    assert list(store._cache) == ["P0", "P2"]
    # Cached object is returned as-is.
    assert store.get("P0") is store.get("P0")


def test_samples_loaded_and_served(data_dir: Path) -> None:
    _write_sample(data_dir, [_raw("S1", "Sita"), {"profile": "garbage"}])
    store = ProfileStore(jsonl_path=data_dir / "missing.jsonl")
    assert set(store.get_all_sample()) == {"S1"}
    s = store.get("S1")
    assert s is not None and s.personal.name == "Sita"
    assert "S1" in store


def test_sample_with_single_object(data_dir: Path) -> None:
    (data_dir / "samples" / "sample.json").write_text(json.dumps(_raw("ONE")), encoding="utf-8")
    assert set(ProfileStore(jsonl_path=data_dir / "x.jsonl").get_all_sample()) == {"ONE"}


def test_sample_utf8_names_decoded_correctly(data_dir: Path) -> None:
    # Regression: load_sample opened the file without encoding=, so on Windows
    # (cp1252 default) UTF-8 names became mojibake ("JosÃ©").
    _write_sample(data_dir, [_raw("U1", "José Ñúñez — ராம்")])
    p = ProfileStore(jsonl_path=data_dir / "x.jsonl").get("U1")
    assert p is not None and p.personal.name == "José Ñúñez — ராம்"


def test_len_does_not_double_count_samples(data_dir: Path) -> None:
    # Regression: samples are also registered in the offset index (offset -1),
    # and __len__ added both, so every sample was counted twice.
    _write_sample(data_dir, [_raw("S1"), _raw("S2")])
    path = data_dir / "c.jsonl"
    _write_jsonl(path, [_raw("S1"), _raw("J1")])
    store = ProfileStore(jsonl_path=path)
    assert len(store) == 3
    assert sorted(store.get_all_pids()) == ["J1", "S1", "S2"]


def test_offset_index_roundtrip_keeps_samples(data_dir: Path) -> None:
    path = data_dir / "c.jsonl"
    _write_jsonl(path, [_raw("J1", "Jaya"), _raw("J2", "Jai")])
    first = ProfileStore(jsonl_path=path)
    first.get_all_pids()
    first.save_offset_index(data_dir / "indexes" / "offset_index.json")

    # A new store auto-loads the saved index and never re-scans the file.
    _write_sample(data_dir, [_raw("S1")])
    second = ProfileStore(jsonl_path=path)
    assert second._index_built
    j2 = second.get("J2")
    assert j2 is not None and j2.personal.name == "Jai"
    # Regression: loading the offset index replaced the dict and dropped the
    # sample IDs registered just before, so they vanished from get_all_pids().
    assert sorted(second.get_all_pids()) == ["J1", "J2", "S1"]
    assert len(second) == 3


def test_stale_offset_returns_none(data_dir: Path) -> None:
    path = data_dir / "c.jsonl"
    _write_jsonl(path, [_raw("J1")])
    (data_dir / "indexes" / "offset_index.json").write_text(
        json.dumps({"J1": 0, "GONE": 10_000}), encoding="utf-8"
    )
    store = ProfileStore(jsonl_path=path)
    assert store.get("J1") is not None
    assert store.get("GONE") is None  # reads "" at EOF -> JSON error -> None
