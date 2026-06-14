from __future__ import annotations

import numpy as np


def compute_score_variance(scores: list[float]) -> float:
    if len(scores) < 2:
        return 0.0
    return float(np.var(scores))


def compute_confidence(scores: dict[str, float | None]) -> float:
    values = [v for v in scores.values() if v is not None]
    if len(values) < 2:
        return 0.5
    std = float(np.std(values))
    return max(0.0, min(1.0, 1.0 - std))
