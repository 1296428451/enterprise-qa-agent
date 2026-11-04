import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.q3.agent import EnterpriseQA, load_chunks
from src.q3.llm import MockLLM, OpenAICompatibleLLM
from src.q3.manage import import_documents
from src.q3.provider_reranker import ProviderReranker
from src.q3.retrieval import MockJevReranker


def main():
    parser = argparse.ArgumentParser(description="Question three enterprise QA Agent")
    parser.add_argument("--question")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--candidate-limit", type=int, default=50)
    parser.add_argument("--max-evidence", type=int, default=8)
    parser.add_argument("--llm", choices=("mock", "openai-compatible"), default="mock")
    parser.add_argument("--reranker", choices=("mock", "provider"), default="mock")
    parser.add_argument("--admin-import", action="store_true", help="批量导入知识库文档")
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "knowledge.db")
    parser.add_argument("--rebuild-index", action="store_true")
    args = parser.parse_args()
    if args.admin_import:
        if not args.input:
            parser.error("--admin-import requires --input")
        print(json.dumps(import_documents(args.input, args.db, args.rebuild_index), ensure_ascii=False, indent=2))
        return
    payload = json.loads(args.input.read_text(encoding="utf-8")) if args.input else None
    if payload is None and not args.question:
        payload = json.load(sys.stdin)
    payload = payload or {"question": args.question}
    question = payload["question"]
    config = payload.get("config", {})
    chunks = load_chunks(args.db if args.db.exists() else ROOT / "data" / "q3_documents.json")
    llm = OpenAICompatibleLLM() if args.llm == "openai-compatible" else MockLLM()
    reranker = ProviderReranker.from_environment() if args.reranker == "provider" else MockJevReranker()
    result = EnterpriseQA(
        chunks,
        candidate_limit=int(config.get("candidate_limit", args.candidate_limit)),
        max_evidence=int(config.get("max_evidence", args.max_evidence)),
        reranker=reranker,
        llm=llm,
    ).answer(question, payload.get("conversation_history"))
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
