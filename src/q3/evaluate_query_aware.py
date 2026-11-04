from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.q3.agent import load_chunks
from src.q3.models import ScoredChunk
from src.q3.query_aware import MockQueryAwareExtractor, estimate_tokens
from src.q3.retrieval import BM25, MockSemanticRetriever, merge_candidates, normalize_scores

NOISY_CASES = [
    {
        "question": "报销期限多久？",
        "documents": [
            {"id": "D01", "text": "本公司成立于2010年，总部位于北京。年会筹备工作由行政部负责。普通费用报销应在费用发生后 30 天内提交。适用于所有正式员工。公司食堂提供免费午餐。"},
            {"id": "D02", "text": "普通费用报销应在费用发生后 15 天内提交。适用于研发部门。研发部员工可申请弹性工作时间。"},
            {"id": "D03", "text": "差旅报销需要提交发票、行程单和审批记录。出差前需在系统中提交出差申请。"},
            {"id": "D04", "text": "员工可以在系统中查看自己的审批进度。系统支持手机端和电脑端访问。"},
        ],
        "expected_evidence_ids": ["D01", "D02"],
    },
    {
        "question": "差旅报销需要什么材料？",
        "documents": [
            {"id": "D01", "text": "普通费用报销应在费用发生后 30 天内提交。适用于所有正式员工。"},
            {"id": "D02", "text": "普通费用报销应在费用发生后 15 天内提交。适用于研发部门。"},
            {"id": "D03", "text": "差旅报销需要提交发票、行程单和审批记录。出差前需在系统中提交出差申请。出差补贴标准按职级划分。"},
            {"id": "D04", "text": "员工可以在系统中查看自己的审批进度。系统支持手机端和电脑端访问。"},
        ],
        "expected_evidence_ids": ["D03"],
    },
    {
        "question": "员工福利积分如何兑换？",
        "documents": [
            {"id": "D01", "text": "普通费用报销应在费用发生后 30 天内提交。"},
            {"id": "D02", "text": "普通费用报销应在费用发生后 15 天内提交。"},
            {"id": "D03", "text": "差旅报销需要提交发票、行程单和审批记录。"},
            {"id": "D04", "text": "员工可以在系统中查看自己的审批进度。"},
        ],
        "expected_evidence_ids": [],
    },
]


def run_evaluation() -> dict:
    extractor = MockQueryAwareExtractor()
    rows = []
    for case in NOISY_CASES:
        question = case["question"]
        items = []
        for i, doc in enumerate(case["documents"], 1):
            from src.q3.models import Chunk
            chunk = Chunk(chunk_id=f"{doc['id']}-v1-c01", document_id=doc["id"],
                          version="v1", version_date="2026-01-01", text=doc["text"])
            items.append(ScoredChunk(chunk, bm25_score=1.0 / i))
        result = extractor.extract(question, items)
        extracted_ids = [e.source_id for e in result.evidence if e.relevance != "none"]
        source_chars = sum(len(d["text"]) for d in case["documents"])
        ext_chars = sum(len(e.quote) for e in result.evidence if e.quote)
        source_tokens = estimate_tokens("".join(d["text"] for d in case["documents"]))
        ext_tokens = estimate_tokens("".join(e.quote for e in result.evidence if e.quote))
        rows.append({
            "question": question,
            "source_chars": source_chars,
            "extracted_chars": ext_chars,
            "char_compression_ratio": round(ext_chars / source_chars, 3) if source_chars else 0,
            "source_tokens": source_tokens,
            "extracted_tokens": ext_tokens,
            "token_compression_ratio": round(ext_tokens / source_tokens, 3) if source_tokens else 0,
            "expected_evidence_ids": case["expected_evidence_ids"],
            "actual_evidence_ids": extracted_ids,
            "evidence_preserved": set(case["expected_evidence_ids"]).issubset(set(extracted_ids)),
            "noise_removed": result.compression_note,
            "evidence_items": [
                {"source_id": e.source_id, "relevance": e.relevance, "quote": e.quote, "key_info": e.key_info}
                for e in result.evidence
            ],
        })
    total_src = sum(r["source_chars"] for r in rows)
    total_ext = sum(r["extracted_chars"] for r in rows)
    total_src_tok = sum(r["source_tokens"] for r in rows)
    total_ext_tok = sum(r["extracted_tokens"] for r in rows)
    preserved = sum(r["evidence_preserved"] for r in rows)
    return {
        "evaluation": "query-aware evidence extraction (mock)",
        "cases": len(rows),
        "evidence_preserved_rate": preserved / len(rows),
        "overall_char_compression_ratio": round(total_ext / total_src, 3) if total_src else 0,
        "overall_token_compression_ratio": round(total_ext_tok / total_src_tok, 3) if total_src_tok else 0,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate query-aware evidence extraction")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_evaluation()
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
