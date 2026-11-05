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

    ``formatted`` is the same result rendered as the fixed display layout
    (「证据充足，…」+「引用：」/「证据部分充足，…」+「未覆盖：」/「证据不足无法直接回答，
    以下是可能相关的记录：」). It is recomputed together with ``completeness``.
    """

    status: str
    answer: str | None
    citations: list[dict[str, Any]]
    gaps: list[str]
    trace: list[dict[str, Any]]
    retrieval: dict[str, Any]
    verification: dict[str, Any]
    completeness: str = ""
    formatted: str = ""

    def __post_init__(self) -> None:
        self.refresh_completeness()

    def refresh_completeness(self) -> str:
        """Recompute ``completeness`` and re-render ``formatted``.

        Call this after changing ``status``/``answer``/``gaps``.
        """
        if self.status != "completed" or not (self.answer or "").strip():
            self.completeness = NONE
        elif self.gaps:
            self.completeness = PARTIAL
        else:
            self.completeness = COMPLETE
        self.formatted = self.render()
        return self.completeness

    @property
    def answer_complete(self) -> bool:
        """True only when the returned answer covers the whole question."""
        return self.completeness == COMPLETE

    def render(self) -> str:
        """Render the fixed human-readable layout.

        ``answer`` stays the bare model text so verification and the offline
        exact-match metric keep comparing like with like; this is the layer that
        adds the 「证据充足 / 证据部分充足 / 证据不足」 header and the citation list.
        """
        detail = (self.answer or "").strip()
        block = _citation_block(self.citations)
        if self.completeness == COMPLETE:
            body = f"证据充足，{detail}"
            return f"{body}\n\n引用：\n{block}" if block else body
        if self.completeness == PARTIAL:
            parts = [f"证据部分充足，{detail}"]
            if self.gaps:
                parts.append("未覆盖：" + "；".join(self.gaps))
            if block:
                parts.append("引用：\n" + block)
            return "\n\n".join(parts)
        header = "证据不足无法直接回答"
        if self.status in {"conflict_detected", "version_conflict"}:
            header += "（检索到的条款互相冲突）"
        if block:
            return f"{header}，以下是可能相关的记录：\n{block}"
        return f"{header}，知识库中没有检索到相关记录。"

    def decline(self, reason: str) -> "Answer":
        """Withdraw the answer, record why, and update the derived fields."""
        self.status = "insufficient_evidence"
        self.answer = None
        if reason and reason not in self.gaps:
            self.gaps = [*self.gaps, reason]
        self.refresh_completeness()
        return self


def _citation_block(citations: list[dict[str, Any]] | None) -> str:
    """One ``[n] 来源 版本：原文`` line per citation, numbered from 1."""
    lines = []
    for index, item in enumerate(citations or [], 1):
        source = str(item.get("document_id") or item.get("chunk_id") or "未知来源").strip()
        version = str(item.get("version") or "").strip()
        label = f"{source} {version}".strip()
        quote = str(item.get("quote") or "").strip()
        lines.append(f"[{index}] {label}：{quote}" if quote else f"[{index}] {label}")
    return "\n".join(lines)
