"""Edge cases for POST /api/v1/ingest (real lifespan + fake models, see test_ingest.py)."""
from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient

import src.main as main_module
from tests.test_ingestion.test_service import raw_profile


def _post(c: TestClient, body: bytes, name: str = "p.json") -> Any:
    return c.post("/api/v1/ingest", files={"file": (name, body, "application/json")})


def test_nan_or_infinity_in_profile_is_400_and_not_persisted(app_env: dict[str, Any]) -> None:
    # Python's json accepts NaN/Infinity, which are not valid JSON; they were stored raw in
    # ingested.jsonl (unreadable by strict JSON tools) and fed into the content hash.
    p = raw_profile("NAN-1")
    body = json.dumps(p).replace('"years_of_experience": 5', '"years_of_experience": NaN')
    assert "NaN" in body
    with TestClient(main_module.app, raise_server_exceptions=False) as c:
        r = _post(c, body.encode())
        assert r.status_code == 400
        assert "NaN" in r.json()["detail"]
        assert c.get("/api/v1/profiles/NAN-1").status_code == 404
    assert not app_env["ingested"].exists()


def test_uppercase_json_extension_is_accepted(app_env: dict[str, Any]) -> None:
    with TestClient(main_module.app) as c:
        r = _post(c, json.dumps(raw_profile("UP-1")).encode(), name="EXPORT.JSON")
        assert r.status_code == 200, r.text
        assert r.json()["created"] == 1


def test_wrong_extension_message_names_the_file(app_env: dict[str, Any]) -> None:
    with TestClient(main_module.app) as c:
        r = _post(c, b"a,b\n1,2\n", name="people.csv")
        assert r.status_code == 400
        assert "people.csv" in r.json()["detail"] and ".json" in r.json()["detail"]


def test_utf8_bom_file_is_accepted(app_env: dict[str, Any]) -> None:
    # Windows Notepad / PowerShell `Out-File` write a BOM.
    body = "﻿".encode() + json.dumps(raw_profile("BOM-1")).encode()
    with TestClient(main_module.app) as c:
        r = _post(c, body)
        assert r.status_code == 200, r.text
        assert r.json()["profile_ids"] == ["BOM-1"]


def test_unicode_profile_roundtrips(app_env: dict[str, Any]) -> None:
    p = raw_profile("UNI-1", title="डेटा वैज्ञानिक")
    p["profile"]["anonymized_name"] = "Anaïs Ñúñez"
    with TestClient(main_module.app) as c:
        assert _post(c, json.dumps(p, ensure_ascii=False).encode()).status_code == 200
        got = c.get("/api/v1/profiles/UNI-1").json()
        assert got["personal"]["name"] == "Anaïs Ñúñez"


def test_empty_list_and_empty_file(app_env: dict[str, Any]) -> None:
    with TestClient(main_module.app) as c:
        r = _post(c, b"[]")
        assert r.status_code == 200
        assert r.json()["total_profiles"] == 0 and r.json()["successful"] == 0
        r = _post(c, b"")
        assert r.status_code == 400
        assert "Invalid JSON" in r.json()["detail"]


def test_same_id_twice_in_one_request_last_wins(app_env: dict[str, Any]) -> None:
    a = raw_profile("DUP-1", title="Java Developer")
    b = raw_profile("DUP-1", title="Go Developer")
    with TestClient(main_module.app) as c:
        r = _post(c, json.dumps([a, b]).encode())
        assert r.status_code == 200, r.text
        assert r.json()["failed"] == 0
        assert c.get("/api/v1/profiles/DUP-1").json()["professional"]["current_title"] == (
            "Go Developer"
        )
    lines = app_env["ingested"].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1  # one stored record per id


def test_oversized_upload_is_413_even_without_content_length(
    app_env: dict[str, Any], monkeypatch: Any,
) -> None:
    # The middleware only checks Content-Length; a chunked upload skipped it and the route read
    # the whole file into memory. The route now enforces its own byte limit.
    import src.api.routes.ingest as ingest_route

    monkeypatch.setattr(ingest_route, "MAX_UPLOAD_BYTES", 1000)
    big = json.dumps([raw_profile(f"BIG-{i}") for i in range(20)]).encode()
    assert len(big) > 1000
    boundary = "xBOUNDARYx"
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"big.json\"\r\n"
        f"Content-Type: application/json\r\n\r\n"
    ).encode() + big + f"\r\n--{boundary}--\r\n".encode()

    def chunks() -> Any:
        for i in range(0, len(body), 256):
            yield body[i : i + 256]

    with TestClient(main_module.app, raise_server_exceptions=False) as c:
        r = c.post(
            "/api/v1/ingest", content=chunks(),
            headers={"content-type": f"multipart/form-data; boundary={boundary}"},
        )
        assert r.status_code == 413, r.text
        assert "1000" in r.json()["detail"]
        # Small files still work under the same limit.
        assert _post(c, json.dumps(raw_profile("S-1")).encode()).status_code == 200
