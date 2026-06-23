from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

DEVANAGARI_RANGE = range(0x0900, 0x0980)
LATIN_RANGE = range(0x0041, 0x007B)
LATIN_ACCENTED_RANGE = range(0x00C0, 0x0250)

HINGLISH_KEYWORDS: set[str] = {
    "hai", "ho", "hain", "hoon", "hu", "ka", "ki", "ke", "ko", "se", "mein", "me",
    "nahi", "tha", "the", "thi", "thhe", "aur", "ya", "toh", "to",
    "bahut", "kuch", "kar", "karo", "karte", "karna", "baat", "liye",
    "wala", "wale", "wali", "saath", "yahan", "wahan", "abhi", "aaj",
    "kal", "raha", "rahi", "rahe", "chahiye", "sakta", "sakte", "sakti",
    "apna", "apne", "mera", "mere", "tera", "tere", "uska", "uske",
    "inn", "unn", "inhe", "unhe", "yeh", "woh", "jo", "so", "kya",
    "kyon", "kaise", "kab", "kahan", "kitna", "kitne", "kitni",
    "bada", "bade", "badi", "chota", "chote", "choti",
    "main", "mujhe", "tum", "tumhe", "aap", "hum",
    "ek", "do", "teen", "char", "kya", "kyu", "hona",
    "rakhta", "rakhti", "rakhte", "leta", "leti", "lete",
    "diya", "diye", "dijiye", "karo", "kijiye",
    "chalta", "chalti", "chalte", "chalega",
    "aata", "aati", "aate", "jaata", "jaati", "jaate",
    "dekh", "dekho", "dekhte", "dekhna",
    "bol", "bolo", "bolte", "bolna", "bolti",
    "samajh", "samajho", "samajhte",
    "aana", "jana", "khana", "pina", "sona",
    "ho sake", "sakta", "sakti", "sakte", "paana",
    "chahte", "chahti", "chahta", "chahiye",
    "thoda", "thodi", "thode", "zyada", "saara",
    "aadmi", "insaan", "log", "dost",
    "kaam", "ghar", "office", "school",
    "achha", "accha", "achhi", "acchi", "achhe", "acche",
    "bura", "buri", "bure", "sundar", "khoobsurat",
    "sahi", "galat", "pakka", "sach", "jhooth",
    "aise", "vaise", "jaise", "taise",
    "yaha", "waha", "idhar", "udhar", "kaha",
    "jab", "tab", "kabhi", "kabhi nahi", "hamesha",
    "shayad", "zaroor", "pakka", "bilkul",
    "matlab", "mtlb", "bas", "sirf", "kewal",
}

DEVANAGARI_PATTERN = re.compile(r"[\u0900-\u097F]")


@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0


class CodeMixedProcessor:
    def __init__(self, hinglish_model: str = "l3cube-pune/hing-bert") -> None:
        self.hinglish_model = hinglish_model
        self._ner_pipeline: Any | None = None

    def detect_code_mixed(self, text: str) -> bool:
        has_devanagari = bool(DEVANAGARI_PATTERN.search(text))
        latin_words = _count_latin_words(text)
        words = text.split()
        has_hinglish_keywords = any(w.lower() in HINGLISH_KEYWORDS for w in words)
        if has_devanagari and latin_words < 3:
            return True
        if has_devanagari and latin_words >= 3:
            return True
        if has_hinglish_keywords and latin_words >= 2:
            return True
        if has_hinglish_keywords and len(words) <= 8:
            return True
        return False

    def extract_entities(self, text: str) -> list[Entity]:
        return self._regex_ner_fallback(text)

    def _regex_ner_fallback(self, text: str) -> list[Entity]:
        entities: list[Entity] = []
        skill_pattern = re.compile(
            r"(Python|Java|JavaScript|SQL|AWS|Django|React|Angular|Node\.?js|"
            r"TypeScript|Go|Rust|Kubernetes|Docker|TensorFlow|PyTorch|"
            r"Machine\s*Learning|Deep\s*Learning|Data\s*Science|NLP|"
            r"Natural\s*Language\s*Processing|Computer\s*Vision|"
            r"DevOps|CI/CD|Cloud|Azure|GCP|API|REST|GraphQL)"
        )
        for match in skill_pattern.finditer(text):
            entities.append(Entity(
                text=match.group(), label="SKILL",
                start=match.start(), end=match.end(),
            ))

        org_pattern = re.compile(
            r"(Google|Microsoft|Amazon|Meta|Flipkart|Amazon|Paytm|"
            r"Razorpay|Swiggy|Zomato|Ola|Uber|TCS|Infosys|Wipro|"
            r"HCL|Tech\s*Mahindra|LTI|Cognizant|Accenture|Deloitte|PwC|"
            r"KPMG|EY|Goldman\s*Sachs|JPMorgan|Morgan\s*Stanley)"
        )
        for match in org_pattern.finditer(text):
            entities.append(Entity(
                text=match.group(), label="ORG",
                start=match.start(), end=match.end(),
            ))
        return entities

    def transliterate_hinglish(self, text: str) -> str:
        if not self.detect_code_mixed(text):
            return text
        translit_map: dict[str, str] = {
            "kaam": "work", "baat": "talk", "achha": "good", "bura": "bad",
            "chhota": "small", "bada": "big", "accha": "good", "thik": "fine",
            "sahi": "correct", "galat": "wrong", "zyada": "more", "kam": "less",
            "jaldi": "fast", "dheere": "slow", "andar": "inside", "bahar": "outside",
            "upar": "above", "neeche": "below", "aage": "ahead", "peeche": "behind",
            "paas": "near", "door": "far", "aaj": "today", "kal": "yesterday/tomorrow",
            "abhi": "now", "phir": "then/again", "hamesha": "always",
            "kabhi": "sometimes/never", "kuch": "some/any", "sab": "all",
            "bahut": "very/many", "thoda": "a little", "thode": "a few",
            "sakta": "can", "sakte": "can", "sakti": "can",
            "chahiye": "need/want", "karna": "to do", "karo": "do",
            "karte": "do/doing", "kar": "do", "ho": "are/be",
            "hai": "is", "hain": "are", "tha": "was", "the": "were",
            "thi": "was", "nahi": "no/not", "aur": "and",
            "ya": "or", "lekin": "but", "kyonki": "because",
        }
        words = text.split()
        result: list[str] = []
        for w in words:
            cleaned = re.sub(r"[^a-zA-Z\u0900-\u097F]", "", w).lower()
            if cleaned in translit_map:
                result.append(translit_map[cleaned])
            else:
                result.append(w)
        return " ".join(result)


def _count_latin_words(text: str) -> int:
    count = 0
    for word in text.split():
        cleaned = word.strip(".,!?;:\"'()[]{}")
        if cleaned and all(
            ord(c) in LATIN_RANGE or ord(c) in LATIN_ACCENTED_RANGE or c == "-"
            for c in cleaned
        ):
            count += 1
    return count
