from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.q3.agent import EnterpriseQA, load_chunks


def evaluate(cases_path: str | Path, database_path: str | Path) -> dict:
    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    chunks = load_chunks(database_path)
    agent = EnterpriseQA(chunks)
    rows = []
    for case in cases:
        result = agent.answer(case["question"])
        actual_documents = sorted({citation["document_id"] for citation in result.citations})
        expected_documents = sorted(case["expected_documents"])
        rows.append({
            "case_id": case["case_id"],
            "question": case["question"],
            "expected_status": case["expected_status"],
            "actual_status": result.status,
            "status_match": result.status == case["expected_status"],
            "expected_documents": expected_documents,
            "actual_documents": actual_documents,
            "document_recall": bool(set(expected_documents).issubset(actual_documents)),
            "trace": result.trace,
        })
    status_matches = sum(row["status_match"] for row in rows)
    document_matches = sum(row["document_recall"] for row in rows)
    return {"total": len(rows), "status_accuracy": status_matches / len(rows),
            "document_recall_rate": document_matches / len(rows), "cases": rows}


def main():
    parser = argparse.ArgumentParser(description="Evaluate the question three retrieval workflow")
    parser.add_argument("--cases", default="data/q3_eval_cases.json")
    parser.add_argument("--db", default="data/knowledge.db")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(args.cases, args.db)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
