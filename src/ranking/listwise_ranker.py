"""Plackett-Luce listwise tournament ranking.

Implementation of the Plackett-Luce model for ranking candidates through
tournament-style comparison. Candidates compete in small groups, judged by
an LLM, and aggregated via the MM (minorization-maximization) algorithm.

Reference: https://en.wikipedia.org/wiki/Plackett%E2%80%93Luce_model
"""

from __future__ import annotations

import json
import logging
import math
import random
from typing import Any

import numpy as np

from src.core.config import get_llm_client, get_scoring_config
from src.core.models import MatchResult

logger = logging.getLogger(__name__)


class PlackettLuceRanker:
    """Listwise tournament ranking using Plackett-Luce model.

    Candidates are divided into random groups, an LLM judges each group
    listwise (producing a ranking), and the MM algorithm aggregates partial
    rankings into a global merit score.
    """

    def __init__(self) -> None:
        config = get_scoring_config()
        lr_config = config.get("listwise_ranking", {})
        self.enabled = lr_config.get("enabled", True)
        self.group_size = lr_config.get("group_size", 5)
        self.max_em_iterations = lr_config.get("max_em_iterations", 20)
        self.num_tournament_rounds = lr_config.get("num_tournament_rounds", 3)
        self._client = None

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                self._client = get_llm_client()
            except Exception:
                logger.warning("LLM client unavailable for Plackett-Luce ranking")
                self._client = None
        return self._client

    def rank(
        self,
        candidates: list[MatchResult],
        anonymized_profiles: dict[str, dict] | None = None,
    ) -> list[tuple[str, float]]:
        """Rank candidates by Plackett-Luce merit score (sync, with fallback).

        Args:
            candidates: List of match results to rank
            anonymized_profiles: Optional dict of profile_id -> anonymized profile dicts

        Returns:
            List of (profile_id, merit_score) sorted descending by merit
        """
        if not candidates or len(candidates) < 2:
            return [(c.profile_id, 1.0) for c in candidates]

        if not self.enabled or self.client is None:
            logger.info("Plackett-Luce disabled or client unavailable, using score-based fallback")
            return self._pointwise_fallback(candidates)

        n_candidates = len(candidates)
        gamma = np.ones(n_candidates)

        for round_idx in range(self.num_tournament_rounds):
            groups = self._create_groups(candidates, self.group_size)
            all_rankings = []

            for group in groups:
                partial_ranking = self._judge_group(group, anonymized_profiles)
                if partial_ranking:
                    all_rankings.append(partial_ranking)

            if all_rankings:
                gamma = self._mm_algorithm(gamma, all_rankings)
                logger.info(
                    f"  Round {round_idx + 1}: gamma range "
                    f"[{gamma.min():.4f}, {gamma.max():.4f}]"
                )

        sorted_indices = np.argsort(-gamma)
        return [
            (candidates[i].profile_id, float(gamma[i]))
            for i in sorted_indices
        ]

    async def arank(
        self,
        candidates: list[MatchResult],
        anonymized_profiles: dict[str, dict] | None = None,
    ) -> list[tuple[str, float]]:
        """Async rank candidates by Plackett-Luce merit score.

        Same as rank() but uses async LLM calls to avoid nested event loop issues
        when called from within an async context (e.g. orchestrator graph nodes).
        """
        if not candidates or len(candidates) < 2:
            return [(c.profile_id, 1.0) for c in candidates]

        if not self.enabled or self.client is None:
            logger.info("Plackett-Luce disabled or client unavailable, using score-based fallback")
            return self._pointwise_fallback(candidates)

        n_candidates = len(candidates)
        gamma = np.ones(n_candidates)  # merit parameters, initialized to 1.0

        for round_idx in range(self.num_tournament_rounds):
            groups = self._create_groups(candidates, self.group_size)
            all_rankings = []

            for group in groups:
                partial_ranking = await self._judge_group_async(
                    group, anonymized_profiles
                )
                if partial_ranking:
                    all_rankings.append(partial_ranking)

            if all_rankings:
                gamma = self._mm_algorithm(gamma, all_rankings)
                logger.info(
                    f"  Round {round_idx + 1}: gamma range "
                    f"[{gamma.min():.4f}, {gamma.max():.4f}]"
                )

        sorted_indices = np.argsort(-gamma)
        return [
            (candidates[i].profile_id, float(gamma[i]))
            for i in sorted_indices
        ]

    def _create_groups(
        self, candidates: list[MatchResult], group_size: int
    ) -> list[list[MatchResult]]:
        """Shuffle candidates and split into groups."""
        shuffled = list(candidates)
        random.shuffle(shuffled)
        return [
            shuffled[i : i + group_size]
            for i in range(0, len(shuffled), group_size)
        ]

    def _judge_group(
        self,
        group: list[MatchResult],
        anonymized_profiles: dict[str, dict] | None = None,
    ) -> list[int] | None:
        """Ask LLM to rank a group of candidates listwise (sync wrapper)."""
        try:
            import asyncio
            return asyncio.run(self._judge_group_async(group, anonymized_profiles))
        except RuntimeError:
            logger.warning("Cannot run async judge from sync context, falling back")
            return None

    async def _judge_group_async(
        self,
        group: list[MatchResult],
        anonymized_profiles: dict[str, dict] | None = None,
    ) -> list[int] | None:
        """Ask LLM to rank a group of candidates listwise (async).

        Returns list of indices [0, 1, ..., n-1] sorted by merit (best first),
        or None if LLM fails.
        """
        if not group:
            return None

        candidate_descriptions = []
        for i, c in enumerate(group):
            anon = anonymized_profiles.get(c.profile_id) if anonymized_profiles else None
            if anon:
                desc = (
                    f"Candidate {i}: "
                    f"skills={', '.join(anon.get('skills', [])[:10])}, "
                    f"experience={anon.get('experience_years', 'N/A')} years, "
                    f"industry={anon.get('industry', 'N/A')}, "
                    f"total_roles={anon.get('total_roles', 'N/A')}"
                )
            else:
                desc = (
                    f"Candidate {i}: "
                    f"skills={', '.join(c.matched_skills[:10])}, "
                    f"experience={c.experience_years or 0} years, "
                    f"overall_score={c.scores.overall:.2f}"
                )
            candidate_descriptions.append(desc)

        prompt = (
            "You are evaluating candidates for a job. Rank the following "
            f"{len(group)} candidates from best fit to worst fit.\n\n"
            + "\n".join(candidate_descriptions)
            + "\n\nOutput ONLY a comma-separated list of candidate indices "
            "in rank order (best first). Example: 2,0,1,3"
        )

        try:
            from langchain_core.messages import HumanMessage

            response = await self.client.ainvoke([HumanMessage(content=prompt)])
            content = response.content if hasattr(response, "content") else str(response)
            return self._parse_ranking(content, len(group))
        except Exception as e:
            logger.warning(f"Group judge LLM failed: {e}")
            return None

    def _parse_ranking(
        self, content: str, group_size: int
    ) -> list[int] | None:
        """Parse LLM output into a list of candidate indices."""
        try:
            cleaned = content.strip().strip("[]")
            indices = [
                int(x.strip()) for x in cleaned.split(",") if x.strip().isdigit()
            ]
            if len(indices) == group_size and all(
                0 <= i < group_size for i in indices
            ):
                return indices
        except (ValueError, TypeError):
            pass
        return None

    def _mm_algorithm(
        self, gamma: np.ndarray, rankings: list[list[int]]
    ) -> np.ndarray:
        """Minorization-maximization (MM) algorithm for Plackett-Luce.

        Args:
            gamma: Current merit parameters (n,)
            rankings: List of partial rankings, each a list of indices [best, ..., worst]

        Returns:
            Updated gamma parameters
        """
        n = len(gamma)
        gamma = gamma.copy()

        for _iteration in range(self.max_em_iterations):
            old_gamma = gamma.copy()

            for i in range(n):
                numerator = 0.0
                denominator = 1e-10

                for ranking in rankings:
                    if i not in ranking:
                        continue
                    rank_of_i = ranking.index(i)
                    predecessors = ranking[:rank_of_i]
                    denominator += 1.0 / (
                        sum(gamma[j] for j in predecessors) + gamma[i] + 1e-10
                    )
                    numerator += 1.0

                if numerator > 0:
                    gamma[i] = numerator / denominator
                else:
                    gamma[i] = 1e-6

            gamma = gamma / gamma.sum() * n

            change = np.abs(gamma - old_gamma).max()
            if change < 1e-6:
                break

        return gamma

    def _pointwise_fallback(
        self, candidates: list[MatchResult]
    ) -> list[tuple[str, float]]:
        """Fallback: rank by overall score when LLM is unavailable."""
        scored = [(c.profile_id, c.scores.overall) for c in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
