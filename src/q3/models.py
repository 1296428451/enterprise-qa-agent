from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Completeness values. ``status`` says whether the run produced an answer at
# all; ``completeness`` says how much of the question that answer covers. The
# two are separate on purpose: a run can be ``completed`` and still leave part
# of the question uncovered, which used to be visible only by inspecting
# ``gaps``.
COMPLETE = "complete"
PARTIAL = "partial"
NONE = "none"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    version: str
    version_date: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    bm25_score: float = 0.0
    vector_score: float = 0.0
    rerank_score: float = 0.0


@dataclass
class Answer:
    """Final result of one question.

    ``completeness`` is derived from ``status``/``answer``/``gaps`` and kept in
    sync by :meth:`refresh_completeness`, so a caller never has to infer it:

    * ``complete`` — an answer passed verification and no gap was reported;
    * ``partial``  — an answer is present but part of the question is uncovered
      (``gaps`` lists what is missing);
    * ``none``     — no answer was produced (``insufficient_evidence`` or one of
      the conflict statuses), so ``gaps`` explains why.
    """

    status: str
    answer: str | None
    citations: list[dict[str, Any]]
    gaps: list[str]
    trace: list[dict[str, Any]]
    retrieval: dict[str, Any]
    verification: dict[str, Any]
    completeness: str = ""

    def __post_init__(self) -> None:
        self.refresh_completeness()

    def refresh_completeness(self) -> str:
        """Recompute ``completeness`` after changing status/answer/gaps."""
        if self.status != "completed" or not (self.answer or "").strip():
            self.completeness = NONE
        elif self.gaps:
            self.completeness = PARTIAL
        else:
            self.completeness = COMPLETE
        return self.completeness

    @property
    def answer_complete(self) -> bool:
        """True only when the returned answer covers the whole question."""
        return self.completeness == COMPLETE

    def decline(self, reason: str) -> "Answer":
        """Withdraw the answer, record why, and update ``completeness``."""
        self.status = "insufficient_evidence"
        self.answer = None
        if reason and reason not in self.gaps:
            self.gaps = [*self.gaps, reason]
        self.refresh_completeness()
        return self
