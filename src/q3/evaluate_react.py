from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.q3.agent import EnterpriseQA, load_chunks
from src.q3.manage import import_documents


def evaluate(cases_path: str | Path, database_path: str | Path) -> dict:
    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    chunks = load_chunks(database_path)
    agent = EnterpriseQA(chunks)
    rows = []
    for case in cases:
        result = agent.answer(case["question"])
        actual_documents = sorted({citation["document_id"] for citation in result.citations})
        expected_documents = sorted(case["expected_documents"])
        search_steps = [step for step in result.trace if step.get("action") == "search"]
        rows.append({
            "case_id": case["case_id"],
            "question": case["question"],
            "expected_status": case["expected_status"],
            "actual_status": result.status,
            "status_match": result.status == case["expected_status"],
            "expected_documents": expected_documents,
            "actual_documents": actual_documents,
            "documents_exact_match": actual_documents == expected_documents,
            "search_steps": len(search_steps),
            "max_steps_respected": sum(1 for step in result.trace if step.get("step")) <= 5,
            "trace": result.trace,
        })
    return {
        "total": len(rows),
        "status_accuracy": sum(row["status_match"] for row in rows) / len(rows),
        "query_retrieval_accuracy": sum(row["documents_exact_match"] for row in rows) / len(rows),
        "react_step_bound_accuracy": sum(row["max_steps_respected"] for row in rows) / len(rows),
        "cases": rows,
    }


def seed_and_evaluate(seed_path: str | Path, cases_path: str | Path, database_path: str | Path) -> dict:
    import_result = import_documents(seed_path, database_path)
    report = evaluate(cases_path, database_path)
    return {"seed_import": import_result, "evaluation": report}


def main():
    parser = argparse.ArgumentParser(description="Evaluate bounded ReAct search behavior")
    parser.add_argument("--cases", default="data/q3_react_eval_cases.json")
    parser.add_argument("--db", default="data/knowledge.db")
    parser.add_argument("--seed", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = seed_and_evaluate(args.seed, args.cases, args.db) if args.seed else {"evaluation": evaluate(args.cases, args.db)}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
