from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.core.models import MatchMetadata, MatchResult, MatchScores, SearchMethod
from src.ranking.listwise_ranker import PlackettLuceRanker


def _match(pid: str, score: float) -> MatchResult:
    return MatchResult(
        query_id="q", profile_id=pid, rank=1,
        scores=MatchScores(overall=score, semantic_similarity=0, keyword_match=0,
                           skill_match=0, experience_match=0, confidence=0.5),
        metadata=MatchMetadata(search_method=SearchMethod.HYBRID),
    )


class ScoreJudge:
    """Fake LLM: ranks each group by the overall_score printed in the prompt."""

    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, messages: list[Any]) -> Any:
        self.calls += 1
        text = messages[0].content
        scores = [float(s) for s in re.findall(r"overall_score=([0-9.]+)", text)]
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        return SimpleNamespace(content=",".join(map(str, order)))


def _ranker(client: Any, group_size: int = 4, rounds: int = 3) -> PlackettLuceRanker:
    r = PlackettLuceRanker()
    r.enabled = True
    r.group_size = group_size
    r.num_tournament_rounds = rounds
    r._client = client
    return r


async def test_merit_is_assigned_to_the_judged_candidates() -> None:
    # Regression: group-local indices (0..group_size-1) were used as global indices, so only
    # the first group_size candidates ever received merit; everyone else tied at ~0.
    import random

    random.seed(1)
    scores = [0.10, 0.95, 0.20, 0.30, 0.90, 0.40, 0.85, 0.50, 0.80, 0.60, 0.70, 0.15]
    cands = [_match(f"c{i}", s) for i, s in enumerate(scores)]
    judge = ScoreJudge()
    ranked = await _ranker(judge, group_size=4, rounds=4).arank(cands, query="python")
    assert judge.calls == 12  # 3 groups x 4 rounds
    order = [pid for pid, _ in ranked]
    by_score = [f"c{i}" for i in sorted(range(len(scores)), key=lambda i: -scores[i])]
    assert order[0] == by_score[0]
    # Top 3 of the tournament are among the top 5 by score; bottom 3 among the bottom 5.
    # (Candidates that always finish last get ~0 merit and tie, so exact tails can differ.)
    assert set(order[:3]) <= set(by_score[:5])
    assert set(order[-3:]) <= set(by_score[-5:])
    merits = dict(ranked)
    assert merits["c1"] > merits["c0"]
    assert merits["c4"] > merits["c2"]


async def test_llm_failure_keeps_input_order() -> None:
    class Down:
        async def ainvoke(self, messages: list[Any]) -> Any:
            raise TimeoutError("down")

    cands = [_match(f"c{i}", 0.5) for i in range(7)]
    ranked = await _ranker(Down()).arank(cands)
    assert [pid for pid, _ in ranked] == [f"c{i}" for i in range(7)]


async def test_disabled_or_no_client_uses_score_fallback() -> None:
    cands = [_match("a", 0.2), _match("b", 0.9), _match("c", 0.5)]
    r = _ranker(None)
    r._client = None
    r.client  # noqa: B018  (property must not raise)
    r.enabled = False
    assert [p for p, _ in await r.arank(cands)] == ["b", "c", "a"]


async def test_trivial_inputs() -> None:
    r = _ranker(ScoreJudge())
    assert await r.arank([]) == []
    assert await r.arank([_match("solo", 0.3)]) == [("solo", 1.0)]


@pytest.mark.parametrize("content,expected", [
    ("2,0,1,3", [2, 0, 1, 3]),
    (" [3, 2, 1, 0] ", [3, 2, 1, 0]),
    ("0,0,1,2", None),        # duplicate index: not a permutation
    ("0,1,2", None),          # too short
    ("0,1,2,4", None),        # out of range
    ("best is 2", None),
    ("", None),
])
def test_parse_ranking(content: str, expected: list[int] | None) -> None:
    assert _ranker(None)._parse_ranking(content, 4) == expected


def test_mm_algorithm_orders_consistent_winner_first() -> None:
    r = _ranker(None)
    gamma = r._mm_algorithm(np.ones(3), [[2, 0, 1], [2, 1, 0], [0, 1, 2]])
    assert gamma.argmax() == 2
    assert gamma.sum() == pytest.approx(3.0)


def test_create_groups_partitions_all_candidates() -> None:
    cands = [_match(f"c{i}", 0.5) for i in range(10)]
    groups = _ranker(None)._create_groups(cands, 4)
    assert [len(g) for g in groups] == [4, 4, 2]
    assert sorted(c.profile_id for g in groups for c in g) == sorted(c.profile_id for c in cands)
