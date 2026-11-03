"""Deterministic, model-free text embedder (the "lite" engine).

Feature hashing over word unigrams, word bigrams and character trigrams, signed
and L2-normalised into the same 384-dim space as the transformer embedder. It is
not a semantic model: it catches lexical and morphological overlap ("kubernetes"
~ "k8s" does not match; "postgres" ~ "postgresql" does via char trigrams). It
exists so the whole product (ingest, hybrid search, explanations, UI) runs
offline, on a free-tier container, in well under 500 MB, with reproducible
rankings - and so tests and demos never download a model.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#.\-]*")
# Light synonym folding so the most common Indian-market spellings meet.
_FOLD = {
    "bengaluru": "bangalore",
    "gurugram": "gurgaon",
    "bombay": "mumbai",
    "k8s": "kubernetes",
    "golang": "go",
    "js": "javascript",
    "ts": "typescript",
    "postgres": "postgresql",
    "ml": "machine-learning",
    "nodejs": "node.js",
    "reactjs": "react",
}


def _tokens(text: str) -> list[str]:
    toks = _TOKEN_RE.findall(text.lower())
    return [_FOLD.get(t.rstrip(".,-"), t.rstrip(".,-")) for t in toks if t.rstrip(".,-")]


def _bucket(feature: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    return value % dim, (1.0 if (value >> 63) & 1 else -1.0)


class HashingEmbedder:
    """Drop-in for MultilingualEmbedder (embed / embed_batch / embed_query)."""

    model_name = "hashing-v1"

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        self.device = "cpu"

    @property
    def model(self) -> HashingEmbedder:  # parity with MultilingualEmbedder warm-up
        return self

    def embed(self, text: str) -> np.ndarray[Any, Any]:
        vec = np.zeros(self.dimension, dtype=np.float32)
        toks = _tokens(text or "")
        feats: list[tuple[str, float]] = [(f"w:{t}", 1.0) for t in toks]
        feats += [(f"b:{a}_{b}", 0.7) for a, b in zip(toks, toks[1:])]
        for t in toks:
            padded = f"^{t}$"
            feats += [(f"c:{padded[i:i + 3]}", 0.25) for i in range(len(padded) - 2)]
        for feat, weight in feats:
            idx, sign = _bucket(feat, self.dimension)
            vec[idx] += sign * weight
        # Sub-linear term frequency keeps long profiles from dominating.
        vec = (np.sign(vec) * np.log1p(np.abs(vec))).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        out: np.ndarray[Any, Any] = vec
        return out

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> np.ndarray[Any, Any]:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.vstack([self.embed(t) for t in texts]).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray[Any, Any]:
        return self.embed(query)

    def cosine_similarity(self, vec_a: np.ndarray[Any, Any], vec_b: np.ndarray[Any, Any]) -> float:
        return float(np.dot(vec_a, vec_b))
