from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional


Message = Dict[str, str]


@dataclass(frozen=True)
class BestOfNRequest:
    """Dataset-independent input consumed by the Best-of-N service."""

    request_id: str
    sample_id: str
    task_text: str
    messages: Optional[List[Message]] = None
    raw_prompt: Optional[str] = None
    gold_answer: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        has_messages = self.messages is not None
        has_raw_prompt = self.raw_prompt is not None
        if has_messages == has_raw_prompt:
            raise ValueError("Exactly one of messages and raw_prompt must be provided")
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if not self.sample_id:
            raise ValueError("sample_id must not be empty")
        if has_messages:
            if not self.messages:
                raise ValueError("messages must not be empty")
            for message in self.messages:
                if message.get("role") not in {"system", "user", "assistant"}:
                    raise ValueError(f"Unsupported message role: {message.get('role')!r}")
                if not isinstance(message.get("content"), str):
                    raise TypeError("Every message content must be a string")


@dataclass
class BestOfNCandidate:
    sequence_idx: int
    candidate_key: str
    text: str
    score: float
    summary_record: Dict[str, Any]
    steps_record: Dict[str, Any]
    bon_rank: Optional[int] = None


@dataclass
class BestOfNResult:
    request: BestOfNRequest
    aggregation: str
    selected_index: int
    selected_candidate_key: str
    selected_text: str
    selected_score: float
    candidates: List[BestOfNCandidate]

    @property
    def n(self) -> int:
        return len(self.candidates)

    @property
    def selected_candidate(self) -> BestOfNCandidate:
        for candidate in self.candidates:
            if candidate.sequence_idx == self.selected_index:
                return candidate
        raise RuntimeError("Selected candidate is missing from candidates")

    @property
    def ranked_candidates(self) -> List[BestOfNCandidate]:
        """Return a deterministic best-to-worst ranking of every candidate.

        Valid finite scores sort before failed candidates. Ties are resolved by the
        original sequence index so ranking is reproducible across runs.
        """

        ranked = sorted(
            self.candidates,
            key=lambda candidate: (
                0 if math.isfinite(candidate.score) else 1,
                -candidate.score if math.isfinite(candidate.score) else 0.0,
                candidate.sequence_idx,
            ),
        )
        for rank, candidate in enumerate(ranked, start=1):
            candidate.bon_rank = rank
        return ranked

    def top_k(self, k: int) -> List[BestOfNCandidate]:
        if not 1 <= k <= self.n:
            raise ValueError(f"k must satisfy 1 <= k <= {self.n}, received {k}")
        return self.ranked_candidates[:k]

    def lightweight_metadata(self) -> Dict[str, Any]:
        return {
            "request_id": self.request.request_id,
            "selected_candidate_key": self.selected_candidate_key,
            "selected_index": self.selected_index,
            "selected_score": self.selected_score,
            "N": self.n,
            "aggregation": self.aggregation,
        }
