from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.q3.agent import EnterpriseQA, load_chunks


ANSWERABLE_STATUS = "completed"


def evaluate(cases_path: str | Path, database_path: str | Path) -> dict:
    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    chunks = load_chunks(database_path)
    agent = EnterpriseQA(chunks)
    rows = []
    for case in cases:
        result = agent.answer(case["question"])
        actual_documents = {citation["document_id"] for citation in result.citations}
        expected_documents = set(case["expected_documents"])
        true_positive = len(actual_documents & expected_documents)
        precision = true_positive / len(actual_documents) if actual_documents else float(not expected_documents)
        recall = true_positive / len(expected_documents) if expected_documents else float(not actual_documents)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        documents_exact_match = actual_documents == expected_documents
        expected_status = case["expected_status"]
        status_match = result.status == expected_status
        answerable = expected_status == ANSWERABLE_STATUS
        expected_answer = case.get("expected_answer")
        if answerable and not expected_answer:
            raise ValueError(f"answerable case {case['case_id']} must include expected_answer")
        answer_exact_match = (
            result.status == ANSWERABLE_STATUS and result.answer == expected_answer
            if answerable else None
        )
        correct_abstention = status_match and result.answer is None if not answerable else None
        rows.append({
            "case_id": case["case_id"],
            "question": case["question"],
            "expected_status": expected_status,
            "actual_status": result.status,
            "status_match": status_match,
            "expected_documents": sorted(expected_documents),
            "actual_documents": sorted(actual_documents),
            "document_precision": precision,
            "document_recall": recall,
            "document_f1": f1,
            "documents_exact_match": documents_exact_match,
            "expected_answer": expected_answer,
            "actual_answer": result.answer,
            "answer_exact_match": answer_exact_match,
            "correct_abstention": correct_abstention,
            "trace": result.trace,
        })
    answerable_rows = [row for row in rows if row["answer_exact_match"] is not None]
    abstention_rows = [row for row in rows if row["correct_abstention"] is not None]
    return {
        "total": len(rows),
        "status_accuracy": sum(row["status_match"] for row in rows) / len(rows),
        "query_retrieval_accuracy": sum(row["documents_exact_match"] for row in rows) / len(rows),
        "document_precision_macro": sum(row["document_precision"] for row in rows) / len(rows),
        "document_recall_macro": sum(row["document_recall"] for row in rows) / len(rows),
        "document_f1_macro": sum(row["document_f1"] for row in rows) / len(rows),
        "answer_exact_match_accuracy": sum(row["answer_exact_match"] for row in answerable_rows) / len(answerable_rows),
        "answerable_cases": len(answerable_rows),
        "correct_abstention_accuracy": sum(row["correct_abstention"] for row in abstention_rows) / len(abstention_rows),
        "non_answerable_cases": len(abstention_rows),
        "cases": rows,
    }


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
