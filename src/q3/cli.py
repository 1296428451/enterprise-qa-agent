import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.q3.agent import EnterpriseQA, load_chunks


def main():
    parser = argparse.ArgumentParser(description="Question three enterprise QA Agent")
    parser.add_argument("--question")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--candidate-limit", type=int, default=50)
    parser.add_argument("--max-evidence", type=int, default=8)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8")) if args.input else None
    if payload is None and not args.question:
        payload = json.load(sys.stdin)
    payload = payload or {"question": args.question}
    question = payload["question"]
    config = payload.get("config", {})
    chunks = load_chunks(ROOT / "data" / "q3_documents.json")
    result = EnterpriseQA(
        chunks,
        candidate_limit=int(config.get("candidate_limit", args.candidate_limit)),
        max_evidence=int(config.get("max_evidence", args.max_evidence)),
    ).answer(question, payload.get("conversation_history"))
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
