from __future__ import annotations

import json

import pytest

from src.ingestion.normalizer import normalize_redrob
from src.ingestion.parser import ProfileParser
from src.ingestion.quality_scorer import compute_data_quality_score


@pytest.fixture
def parser():
    return ProfileParser()


def test_parse_jsonl_yields_generator(tmp_path, parser):
    lines = [
        json.dumps({
            "candidate_id": str(i),
            "profile": {"anonymized_name": f"User {i}"},
        })
        for i in range(1000)
    ]
    file_path = tmp_path / "test.jsonl"
    file_path.write_text("\n".join(lines))
    gen = parser.parse_jsonl_file(file_path)
    assert hasattr(gen, "__iter__")
    results = list(gen)
    assert len(results) == 1000


def test_parse_jsonl_streaming_not_memory_bound(tmp_path, parser):
    lines = [
        json.dumps({
            "candidate_id": str(i),
            "data": "x" * 1000,
        })
        for i in range(10000)
    ]
    file_path = tmp_path / "test.jsonl"
    file_path.write_text("\n".join(lines))
    count = 0
    for _ in parser.parse_jsonl_file(file_path):
        count += 1
    assert count == 10000


def test_parse_jsonl_gzip(tmp_path, parser):
    import gzip
    lines = [
        json.dumps({"candidate_id": str(i)})
        for i in range(5)
    ]
    file_path = tmp_path / "test.jsonl.gz"
    with gzip.open(file_path, "wt", encoding="utf-8") as f:
        f.write("\n".join(lines))
    results = list(parser.parse_jsonl_file(file_path))
    assert len(results) == 5


def test_parse_jsonl_invalid_json_skipped(tmp_path, parser):
    file_path = tmp_path / "test.jsonl"
    file_path.write_text('{"candidate_id": "1"}\nnot-json\n{"candidate_id": "2"}')
    results = list(parser.parse_jsonl_file(file_path))
    assert len(results) == 2
    assert parser.failed_profiles == 1


def test_parse_json(tmp_path, parser):
    data = [{"candidate_id": "1"}, {"candidate_id": "2"}]
    file_path = tmp_path / "test.json"
    file_path.write_text(json.dumps(data))
    results = parser.parse_json_file(file_path)
    assert len(results) == 2


def test_parse_json_single_object(tmp_path, parser):
    data = {"candidate_id": "1"}
    file_path = tmp_path / "test.json"
    file_path.write_text(json.dumps(data))
    results = parser.parse_json_file(file_path)
    assert len(results) == 1


def test_parse_json_empty_list(tmp_path, parser):
    file_path = tmp_path / "test.json"
    file_path.write_text("[]")
    results = parser.parse_json_file(file_path)
    assert len(results) == 0


def test_parse_empty_file(tmp_path, parser):
    file_path = tmp_path / "empty.jsonl"
    file_path.write_text("")
    results = list(parser.parse_jsonl_file(file_path))
    assert len(results) == 0


def test_parse_docx(tmp_path, parser):
    from xml.etree.ElementTree import Element, SubElement, tostring
    from zipfile import ZipFile

    docx_path = tmp_path / "test.docx"
    with ZipFile(docx_path, "w") as z:
        ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        body = Element(f"{{{ns}}}body")
        for text in ["Hello World", "Python Developer", "5 years experience"]:
            p = SubElement(body, f"{{{ns}}}p")
            r = SubElement(p, f"{{{ns}}}r")
            t = SubElement(r, f"{{{ns}}}t")
            t.text = text
        xml_str = b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' + tostring(body)
        z.writestr("word/document.xml", xml_str)

    result = parser.parse_docx(docx_path)
    assert result["candidate_id"] == "test"
    assert "Hello World" in result["raw_text"]
    assert parser.failed_profiles == 0


def test_parse_docx_unicode(tmp_path, parser):
    from xml.etree.ElementTree import Element, SubElement, tostring
    from zipfile import ZipFile

    docx_path = tmp_path / "résumé.docx"
    with ZipFile(docx_path, "w") as z:
        ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        body = Element(f"{{{ns}}}body")
        p = SubElement(body, f"{{{ns}}}p")
        r = SubElement(p, f"{{{ns}}}r")
        t = SubElement(r, f"{{{ns}}}t")
        t.text = "Software Engineer with 5+ years in Python and \u00e9l\u00e9vation"
        xml_str = b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' + tostring(body)
        z.writestr("word/document.xml", xml_str)

    result = parser.parse_docx(docx_path)
    assert "résumé" in result["candidate_id"]
    assert "élév" in result["raw_text"]


def test_normalizer_redrob():
    raw = {
        "candidate_id": "redrob-1",
        "profile": {
            "anonymized_name": "Priya Sharma",
            "current_title": "Software Engineer",
            "current_company": "Flipkart",
            "current_industry": "IT Services",
            "headline": "Backend Engineer | Python, SQL",
            "summary": "Experienced backend engineer",
            "location": "Bangalore",
            "country": "India",
            "years_of_experience": 5.0,
        },
        "career_history": [
            {
                "title": "Senior Engineer",
                "company": "Flipkart",
                "start_date": "2020-01",
                "end_date": "2023-06",
                "is_current": False,
                "description": "Built scalable microservices",
                "industry": "E-commerce",
            }
        ],
        "education": [
            {
                "institution": "IIT Bombay",
                "degree": "B.Tech",
                "field_of_study": "Computer Science",
                "start_year": 2015,
                "end_year": 2019,
                "grade": "8.5 CGPA",
            }
        ],
        "skills": [
            {"name": "Python", "proficiency": "advanced", "endorsements": 30},
            {"name": "SQL", "proficiency": "intermediate", "endorsements": 15},
            {"name": "AWS", "proficiency": "beginner", "endorsements": 5},
        ],
        "certifications": [{"name": "AWS Certified Solutions Architect"}],
        "languages": [  # noqa: E501
            {"language": "English", "proficiency": "native"},
            {"language": "Hindi", "proficiency": "fluent"},
        ],
    }
    profile = normalize_redrob(raw)
    assert profile.profile_id == "redrob-1"
    assert profile.personal.name == "Priya Sharma"
    assert profile.professional.current_title == "Software Engineer"
    assert profile.professional.current_company == "Flipkart"
    assert profile.personal.location.city == "Bangalore"
    assert profile.professional.industry == "IT Services"
    assert profile.professional.total_experience_years is not None
    assert profile.professional.total_experience_years != 5.0
    assert isinstance(profile.professional.seniority_level, int)
    assert len(profile.skills) == 3
    assert profile.skills[0].name == "Python"
    assert profile.skills[0].confidence > 0.5
    assert len(profile.experience) == 1
    assert profile.experience[0].title == "Senior Engineer"
    assert len(profile.education) == 1
    assert profile.education[0].institution == "IIT Bombay"
    assert len(profile.signals.certifications) == 1
    assert profile.signals.certifications[0] == "AWS Certified Solutions Architect"
    assert "English" in profile.personal.languages_spoken
    assert profile.personal.native_language == "English"
    assert profile.personal.location.is_remote_ok is False


def test_normalizer_redrob_remote():
    for mode, expected in [("remote", True), ("flexible", True), ("hybrid", True), ("onsite", False), (None, False)]:
        raw = {
            "candidate_id": "r-remote",
            "profile": {"anonymized_name": "Test User"},
            "redrob_signals": {"preferred_work_mode": mode} if mode else {}
        }
        profile = normalize_redrob(raw)
        assert profile.personal.location.is_remote_ok is expected


def test_normalizer_redrob_minimal():
    raw = {"candidate_id": "r2", "profile": {"anonymized_name": "Raj"}}
    profile = normalize_redrob(raw)
    assert profile.profile_id == "r2"
    assert profile.personal.name == "Raj"
    assert profile.professional.total_experience_years is None
    assert profile.personal.location.city is None
    assert profile.personal.location.is_remote_ok is False


def test_raw_text_matches_prd_spec():
    raw = {
        "candidate_id": "rt-1",
        "profile": {
            "anonymized_name": "Rahul Kumar",
            "current_title": "DevOps Engineer",
            "current_company": "Razorpay",
            "summary": "Expert in AWS and Kubernetes infrastructure",
            "location": "Bangalore",
        },
        "career_history": [
            {
                "title": "DevOps Engineer",
                "company": "Razorpay",
                "start_date": "2021-03",
                "end_date": None,
                "is_current": True,
                "description": "Managed Kubernetes clusters",
            }
        ],
        "education": [
            {
                "institution": "NIT Trichy",
                "degree": "B.Tech",
                "field_of_study": "Computer Science",
            }
        ],
        "skills": [
            {"name": "AWS", "proficiency": "expert"},
            {"name": "Kubernetes", "proficiency": "advanced"},
        ],
        "certifications": [{"name": "CKA"}],
        "languages": [{"language": "English"}],
    }
    profile = normalize_redrob(raw)
    rt = profile.raw_text
    assert "Name: Rahul Kumar." in rt
    assert "Title: DevOps Engineer." in rt
    assert "Company: Razorpay." in rt
    assert "Summary: Expert in AWS and Kubernetes infrastructure." in rt
    assert "Skills: AWS, Kubernetes." in rt
    assert "DevOps Engineer at Razorpay" in rt
    assert "B.Tech in Computer Science at NIT Trichy" in rt
    assert "Certifications: CKA." in rt
    assert "Languages: English." in rt
    assert rt.count(".") >= 9


def test_quality_scorer(sample_profile):
    score = compute_data_quality_score(sample_profile)
    assert 0.0 <= score <= 1.0
    assert score > 0.5


def test_quality_scorer_all_fields(sample_profile):
    score = compute_data_quality_score(sample_profile)
    assert 0.50 <= score <= 1.00


def test_quality_scorer_empty_profile():
    from src.core.models import PersonalInfo, Profile
    empty = Profile(profile_id="empty", personal=PersonalInfo(name=""))
    score = compute_data_quality_score(empty)
    assert score == 0.05  # No encoding artifacts bonus


def test_quality_scorer_encoding_artifacts():
    from src.core.models import PersonalInfo, Profile
    bad = Profile(
        profile_id="bad",
        personal=PersonalInfo(name="Test"),
        raw_text="Some text with Ã© weird chars",
    )
    score = compute_data_quality_score(bad)
    assert score < 0.5


def test_parse_jsonl_skip_noisy(tmp_path, parser):
    from src.core.models import PersonalInfo, Profile

    def fake_normalizer(d):
        return Profile(
            profile_id=d.get("candidate_id", ""),
            personal=PersonalInfo(name=d.get("candidate_id", "")),
            raw_text="",
        )

    def fake_scorer(p):
        return 0.1

    parser.normalizer = fake_normalizer
    lines = [json.dumps({"candidate_id": f"noisy_{i}"}) for i in range(5)]
    file_path = tmp_path / "noisy.jsonl"
    file_path.write_text("\n".join(lines))
    results = list(parser.parse_jsonl_file(file_path, skip_noisy=True, quality_scorer=fake_scorer))
    assert len(results) == 0
    assert parser.failed_profiles == 5


def test_parse_batch(parser):
    data = [
        {"candidate_id": "1", "profile": {"anonymized_name": "A"}},
        {"candidate_id": "2", "profile": {"anonymized_name": "B"}},
    ]
    successful, errors = parser.parse_batch(data)
    assert len(successful) == 2
    assert len(errors) == 0


def test_parse_batch_with_errors(parser):
    data = [
        {"candidate_id": "1"},
        None,
    ]
    successful, errors = parser.parse_batch(data)
    assert len(successful) == 1
    assert len(errors) == 1


def test_parse_csv_file(tmp_path, parser):
    csv_path = tmp_path / "submission.csv"
    csv_path.write_text(
        "candidate_id,rank,score,reasoning\n"
        "CAND_0000001,1,0.95,Good fit\n"
        "CAND_0000002,2,0.85,Moderate fit\n"
    )
    results = parser.parse_csv_file(csv_path)
    assert len(results) == 2
    assert results[0]["candidate_id"] == "CAND_0000001"
    assert results[0]["rank"] == 1
    assert results[0]["score"] == 0.95
    assert results[0]["reasoning"] == "Good fit"


def test_parse_csv_file_empty(tmp_path, parser):
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("candidate_id,rank,score,reasoning\n")
    results = parser.parse_csv_file(csv_path)
    assert len(results) == 0


def test_parse_csv_stream(tmp_path, parser):
    csv_path = tmp_path / "stream.csv"
    csv_path.write_text(
        "candidate_id,rank,score,reasoning\n"
        "CAND_0000001,1,0.99,Fast\n"
        "CAND_0000002,2,0.88,Fast2\n"
        "CAND_0000003,3,0.77,Fast3\n"
    )
    gen = parser.parse_csv_stream(csv_path)
    assert hasattr(gen, "__iter__")
    results = list(gen)
    assert len(results) == 3


def test_parse_csv_file_extra_columns(tmp_path, parser):
    csv_path = tmp_path / "extra.csv"
    csv_path.write_text(
        "candidate_id,rank,score,reasoning,extra\n"
        "CAND_0000001,1,0.95,Good fit,notes here\n"
    )
    results = parser.parse_csv_file(csv_path)
    assert len(results) == 1
    assert results[0]["extra"] == "notes here"
