#!/usr/bin/env python3
"""Test if HF_HUB_OFFLINE is respected with uvicorn import chain."""
import os
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Simulate the server's import chain
from sentence_transformers import SentenceTransformer

t0 = time.time()
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
print(f"Model load: {time.time()-t0:.2f}s")

t1 = time.time()
vec = model.encode("test query", normalize_embeddings=True)
print(f"First encode: {time.time()-t1:.2f}s, shape: {vec.shape}")
print(f"Total: {time.time()-t0:.1f}s")
print("SUCCESS: Offline mode working")
